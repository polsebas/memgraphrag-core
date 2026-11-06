"""Memory-Guided Retriever with layered PPR and hybrid reranking."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Set

from pydantic import BaseModel, Field

from memgraphrag_core.domain.models import (
    EDGE_EVIDENCE_IN,
    LAYER_M_FAC,
    LAYER_M_ONT,
    LAYER_M_PAS,
    Fact,
    GlobalMemoryState,
    Passage,
)
from memgraphrag_core.interfaces.graph_store import BaseGraphStore, GraphEdge, GraphNode
from memgraphrag_core.interfaces.vector_store import BaseVectorStore, VectorSearchResult
from memgraphrag_core.retrieval.direct_passage_access import (
    cited_passage_ids,
    merge_writer_omitted,
    select_direct_passages,
)
from memgraphrag_core.retrieval.ppr_engine import (
    DEFAULT_LAYER_WEIGHTS,
    DEFAULT_RELATION_WEIGHTS,
    LayeredPPRCalculator,
    exclusive_suppressed_passages,
    suppressed_fact_keys,
)
from memgraphrag_core.retrieval.reranker import HybridReranker


class RetrievalResult(BaseModel):
    """Normalized structure returned to the consuming agent."""

    query: str
    passages: List[Passage] = Field(
        default_factory=list, description="Textual evidence for grounding"
    )
    facts: List[Fact] = Field(
        default_factory=list, description="Verified facts from M_fac"
    )
    subgraph: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted sub-network {nodes: [...], edges: [...]}",
    )
    retrieval_mode: str = Field(
        ...,
        description="'memory_guided_ppr', 'hybrid_rerank', or 'vector_fallback'",
    )
    node_scores: Dict[str, float] = Field(default_factory=dict)
    hybrid_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Fused S_final scores for selected fact_/pas_ ids",
    )
    direct_passages: List[Passage] = Field(
        default_factory=list,
        description="Raw M_pas hits independent of the writer (MemoryTax)",
    )
    writer_omitted_passage_ids: List[str] = Field(
        default_factory=list,
        description="Direct M_pas hits that no accepted fact cites",
    )
    process_records: List[Any] = Field(
        default_factory=list,
        description="Process events judged applicable at as_of (MemDev-Bench)",
    )
    applicability_as_of: Optional[datetime] = Field(
        default=None,
        description="Instant used to judge process-record TTL windows",
    )


class MemoryGuidedRetriever:
    """Retrieve facts/passages via layered PPR + hybrid vector/PPR fusion."""

    def __init__(
        self,
        memory_state: GlobalMemoryState,
        graph_store: BaseGraphStore,
        vector_store: BaseVectorStore,
        alpha_ppr: float = 0.85,
        top_k_passages: int = 5,
        top_k_facts: int = 10,
        layer_weights: Optional[Mapping[str, float]] = None,
        relation_weights: Optional[Mapping[str, float]] = None,
        alpha_hybrid: float = 0.6,
        direct_pas_k: int = 5,
        ppr_calculator: Optional[LayeredPPRCalculator] = None,
        reranker: Optional[HybridReranker] = None,
    ) -> None:
        self.memory_state = memory_state
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.alpha_ppr = alpha_ppr
        self.top_k_passages = top_k_passages
        self.top_k_facts = top_k_facts
        self.direct_pas_k = direct_pas_k

        lw = dict(DEFAULT_LAYER_WEIGHTS)
        if layer_weights:
            lw.update(layer_weights)
        rw = dict(DEFAULT_RELATION_WEIGHTS)
        if relation_weights:
            rw.update(relation_weights)

        self.ppr_calculator = ppr_calculator or LayeredPPRCalculator(
            layer_weights=lw,
            relation_weights=rw,
            alpha=alpha_ppr,
            bidirectional=True,
        )
        self.reranker = reranker or HybridReranker(alpha_hybrid=alpha_hybrid)
        self.layer_weights = self.ppr_calculator.layer_weights
        self.alpha_hybrid = self.reranker.alpha_hybrid

    def get_raw_passage(self, passage_id: str) -> Passage:
        """Read one immutable passage from M_pas."""
        return self.memory_state.get_raw_passage(passage_id)

    def expand_fact_evidence(self, fact_id: str) -> str:
        """Return the exact M_pas text a compressed M_fac fact was written from."""
        return self.memory_state.expand_fact_evidence(fact_id)

    async def retrieve(
        self, query: str, *, as_of: Optional[datetime] = None
    ) -> RetrievalResult:
        moment = as_of or datetime.now(timezone.utc)
        applicable = self.memory_state.applicable_process_records(moment)
        pas_k = max(5, self.top_k_passages, self.direct_pas_k)
        ont_hits, fac_hits, pas_hits = await asyncio.gather(
            self.vector_store.search_similarity(query, k=5, namespace=LAYER_M_ONT),
            self.vector_store.search_similarity(query, k=5, namespace=LAYER_M_FAC),
            self.vector_store.search_similarity(query, k=pas_k, namespace=LAYER_M_PAS),
        )
        direct_passages = select_direct_passages(
            pas_hits, self.memory_state.passages, k=self.direct_pas_k
        )

        ppr_scores = await self.ppr_calculator.run(
            self.graph_store,
            ont_hits,
            fac_hits,
            pas_hits,
            alpha=self.alpha_ppr,
            memory=self.memory_state,
            query=query,
            as_of=as_of,
        )
        if not ppr_scores:
            return await self._execute_vector_fallback(
                query, pas_hits, direct_passages, applicable, moment
            )

        suppressed = suppressed_fact_keys(self.memory_state, as_of)
        hidden_passages = exclusive_suppressed_passages(self.memory_state, as_of)

        fact_vec = {h.id: h.score for h in fac_hits if h.score > 0.0}
        pas_vec = {h.id: h.score for h in pas_hits if h.score > 0.0}

        # Promote PPR mass from graph node ids to logical fact/passage ids.
        fact_ppr: Dict[str, float] = {}
        pas_ppr: Dict[str, float] = {}
        for nid, score in ppr_scores.items():
            if nid.startswith("fact_"):
                fact_ppr[nid.replace("fact_", "", 1)] = score
            elif nid.startswith("pas_"):
                pas_ppr[nid.replace("pas_", "", 1)] = score

        # Ensure memory-resident facts/passages touched by PPR are candidated
        # even if they were weak vector hits (hierarchical propagation).
        for fkey in fact_ppr:
            fact_vec.setdefault(fkey, 0.0)
        for pid in pas_ppr:
            pas_vec.setdefault(pid, 0.0)
        for key in suppressed:
            fact_vec.pop(key, None)
            fact_ppr.pop(key, None)
        for pid in hidden_passages:
            pas_vec.pop(pid, None)
            pas_ppr.pop(pid, None)

        ranked_facts = self.reranker.fuse(fact_vec, fact_ppr)
        ranked_passages = self.reranker.fuse(pas_vec, pas_ppr)

        retrieved_facts: List[Fact] = []
        for item in ranked_facts:
            fact = self.memory_state.facts.get(item.item_id)
            if fact is None or fact.fact_key in suppressed:
                continue
            retrieved_facts.append(fact)
            if len(retrieved_facts) >= self.top_k_facts:
                break

        retrieved_passages: List[Passage] = []
        seen_passage_ids: Set[str] = set()
        for item in ranked_passages:
            if item.item_id in self.memory_state.passages and item.item_id not in seen_passage_ids:
                if item.item_id in hidden_passages:
                    continue
                retrieved_passages.append(self.memory_state.passages[item.item_id])
                seen_passage_ids.add(item.item_id)
            if len(retrieved_passages) >= self.top_k_passages:
                break

        # Grounding: attach evidence passages linked to selected facts, then
        # drop facts whose schema instances are not supported by retrieved passages.
        if len(retrieved_passages) < self.top_k_passages:
            for fact in retrieved_facts:
                for pid in fact.passage_ids:
                    if pid in self.memory_state.passages and pid not in seen_passage_ids:
                        if pid in hidden_passages:
                            continue
                        retrieved_passages.append(self.memory_state.passages[pid])
                        seen_passage_ids.add(pid)
                    if len(retrieved_passages) >= self.top_k_passages:
                        break
                if len(retrieved_passages) >= self.top_k_passages:
                    break

        retrieved_facts = self._facts_grounded_in_passages(
            retrieved_facts, retrieved_passages
        )
        ranked_prefix = retrieved_passages[: self.top_k_passages]
        retrieved_passages, omitted_ids = merge_writer_omitted(
            ranked_prefix,
            direct_passages,
            cited_passage_ids(self.memory_state.facts),
        )

        fact_score_by_id = {item.item_id: item.score for item in ranked_facts}
        pas_score_by_id = {item.item_id: item.score for item in ranked_passages}

        selected_node_ids: List[str] = []
        hybrid_scores: Dict[str, float] = {}
        for fact in retrieved_facts:
            nid = f"fact_{fact.fact_key}"
            selected_node_ids.append(nid)
            hybrid_scores[nid] = fact_score_by_id.get(fact.fact_key, 0.0)
            selected_node_ids.append(f"ont_{fact.schema_key}")
        for passage in retrieved_passages:
            nid = f"pas_{passage.id}"
            selected_node_ids.append(nid)
            hybrid_scores[nid] = pas_score_by_id.get(passage.id, 0.0)

        nodes_obj, edges_obj = await self.graph_store.get_subgraph(selected_node_ids)
        nodes_obj, edges_obj = self._filter_grounded_subgraph(
            nodes_obj, edges_obj, retrieved_facts, retrieved_passages
        )

        return RetrievalResult(
            query=query,
            passages=retrieved_passages,
            facts=retrieved_facts[: self.top_k_facts],
            subgraph={
                "nodes": [n.model_dump() for n in nodes_obj],
                "edges": [e.model_dump() for e in edges_obj],
            },
            retrieval_mode="hybrid_rerank",
            node_scores=ppr_scores,
            hybrid_scores=hybrid_scores,
            direct_passages=direct_passages,
            writer_omitted_passage_ids=omitted_ids,
            process_records=applicable,
            applicability_as_of=moment,
        )

    async def _execute_vector_fallback(
        self,
        query: str,
        pas_hits: List[VectorSearchResult],
        direct_passages: Optional[List[Passage]] = None,
        process_records: Optional[List[Any]] = None,
        as_of: Optional[datetime] = None,
    ) -> RetrievalResult:
        hits = pas_hits or await self.vector_store.search_similarity(
            query, k=max(self.top_k_passages, self.direct_pas_k), namespace=LAYER_M_PAS
        )
        passages = [
            self.memory_state.passages[h.id]
            for h in hits
            if h.id in self.memory_state.passages
        ]
        direct = direct_passages if direct_passages is not None else select_direct_passages(
            hits, self.memory_state.passages, k=self.direct_pas_k
        )
        merged, omitted_ids = merge_writer_omitted(
            passages[: self.top_k_passages],
            direct,
            cited_passage_ids(self.memory_state.facts),
        )
        return RetrievalResult(
            query=query,
            passages=merged,
            facts=[],
            subgraph={"nodes": [], "edges": []},
            retrieval_mode="vector_fallback",
            node_scores={},
            hybrid_scores={},
            direct_passages=direct,
            writer_omitted_passage_ids=omitted_ids,
            process_records=list(process_records or []),
            applicability_as_of=as_of,
        )

    @staticmethod
    def _facts_grounded_in_passages(
        facts: List[Fact], passages: List[Passage]
    ) -> List[Fact]:
        """Keep facts that have at least one evidence id among retrieved passages."""
        support = {p.id for p in passages}
        return [f for f in facts if any(pid in support for pid in f.passage_ids)]

    @staticmethod
    def _filter_grounded_subgraph(
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        facts: List[Fact],
        passages: List[Passage],
    ) -> tuple[List[GraphNode], List[GraphEdge]]:
        """Drop nodes/edges that are not supported by retrieved passages."""
        allowed: Set[str] = {f"pas_{p.id}" for p in passages}
        for fact in facts:
            allowed.add(f"fact_{fact.fact_key}")
            allowed.add(f"ont_{fact.schema_key}")
        passage_nodes = {f"pas_{p.id}" for p in passages}

        nodes = [n for n in nodes if n.id in allowed]
        grounded_edges: List[GraphEdge] = []
        for edge in edges:
            if edge.source_id not in allowed or edge.target_id not in allowed:
                continue
            if edge.relation == EDGE_EVIDENCE_IN and edge.target_id not in passage_nodes:
                continue
            grounded_edges.append(edge)
        return nodes, grounded_edges

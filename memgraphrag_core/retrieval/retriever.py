"""Memory-Guided Retriever with layered PPR and hybrid reranking."""

from __future__ import annotations

import asyncio
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
from memgraphrag_core.retrieval.ppr_engine import (
    DEFAULT_LAYER_WEIGHTS,
    DEFAULT_RELATION_WEIGHTS,
    LayeredPPRCalculator,
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
        ppr_calculator: Optional[LayeredPPRCalculator] = None,
        reranker: Optional[HybridReranker] = None,
    ) -> None:
        self.memory_state = memory_state
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.alpha_ppr = alpha_ppr
        self.top_k_passages = top_k_passages
        self.top_k_facts = top_k_facts

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

    async def retrieve(self, query: str) -> RetrievalResult:
        ont_hits, fac_hits, pas_hits = await asyncio.gather(
            self.vector_store.search_similarity(query, k=5, namespace=LAYER_M_ONT),
            self.vector_store.search_similarity(query, k=5, namespace=LAYER_M_FAC),
            self.vector_store.search_similarity(query, k=5, namespace=LAYER_M_PAS),
        )

        ppr_scores = await self.ppr_calculator.run(
            self.graph_store,
            ont_hits,
            fac_hits,
            pas_hits,
            alpha=self.alpha_ppr,
        )
        if not ppr_scores:
            return await self._execute_vector_fallback(query, pas_hits)

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

        ranked_facts = self.reranker.fuse(fact_vec, fact_ppr)
        ranked_passages = self.reranker.fuse(pas_vec, pas_ppr)

        retrieved_facts: List[Fact] = []
        for item in ranked_facts:
            if item.item_id in self.memory_state.facts:
                retrieved_facts.append(self.memory_state.facts[item.item_id])
            if len(retrieved_facts) >= self.top_k_facts:
                break

        retrieved_passages: List[Passage] = []
        seen_passage_ids: Set[str] = set()
        for item in ranked_passages:
            if item.item_id in self.memory_state.passages and item.item_id not in seen_passage_ids:
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
                        retrieved_passages.append(self.memory_state.passages[pid])
                        seen_passage_ids.add(pid)
                    if len(retrieved_passages) >= self.top_k_passages:
                        break
                if len(retrieved_passages) >= self.top_k_passages:
                    break

        retrieved_facts = self._facts_grounded_in_passages(
            retrieved_facts, retrieved_passages
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
            passages=retrieved_passages[: self.top_k_passages],
            facts=retrieved_facts[: self.top_k_facts],
            subgraph={
                "nodes": [n.model_dump() for n in nodes_obj],
                "edges": [e.model_dump() for e in edges_obj],
            },
            retrieval_mode="hybrid_rerank",
            node_scores=ppr_scores,
            hybrid_scores=hybrid_scores,
        )

    async def _execute_vector_fallback(
        self, query: str, pas_hits: List[VectorSearchResult]
    ) -> RetrievalResult:
        hits = pas_hits or await self.vector_store.search_similarity(
            query, k=self.top_k_passages, namespace=LAYER_M_PAS
        )
        passages = [
            self.memory_state.passages[h.id]
            for h in hits
            if h.id in self.memory_state.passages
        ]
        return RetrievalResult(
            query=query,
            passages=passages,
            facts=[],
            subgraph={"nodes": [], "edges": []},
            retrieval_mode="vector_fallback",
            node_scores={},
            hybrid_scores={},
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

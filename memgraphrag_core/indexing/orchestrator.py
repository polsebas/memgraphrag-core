"""Indexing orchestrator: passages → extraction → tau → detection → resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from memgraphrag_core.agents.base import ModelLike
from memgraphrag_core.agents.detector import DetectorAgent
from memgraphrag_core.agents.extractor import ExtractorAgent
from memgraphrag_core.agents.models import ConflictRecord, ResolutionDecision
from memgraphrag_core.agents.resolver import ResolverAgent
from memgraphrag_core.domain.models import (
    EDGE_EVIDENCE_IN,
    EDGE_INSTANTIATES,
    LAYER_M_FAC,
    LAYER_M_ONT,
    LAYER_M_PAS,
    NODE_FACT,
    NODE_PASSAGE,
    NODE_SCHEMA,
    Fact,
    GlobalMemoryState,
    Passage,
)
from memgraphrag_core.interfaces.graph_store import BaseGraphStore, GraphEdge, GraphNode
from memgraphrag_core.interfaces.vector_store import BaseVectorStore


@dataclass
class IndexingReport:
    """Summary of an indexing run."""

    passages_indexed: int = 0
    schemas_registered: int = 0
    facts_accepted: int = 0
    facts_rejected_by_tau: int = 0
    conflicts_detected: int = 0
    decisions_applied: int = 0
    accepted_fact_keys: List[str] = field(default_factory=list)
    rejected_triples: List[Fact] = field(default_factory=list)
    conflicts: List[ConflictRecord] = field(default_factory=list)
    decisions: List[ResolutionDecision] = field(default_factory=list)


class IndexingOrchestrator:
    """Indexing pipeline A_ext → M_ont/M_fac → A_det → A_res (Agno agents)."""

    def __init__(
        self,
        memory: GlobalMemoryState,
        model: ModelLike,
        vector_store: Optional[BaseVectorStore] = None,
        graph_store: Optional[BaseGraphStore] = None,
        run_conflict_pipeline: bool = True,
    ) -> None:
        self.memory = memory
        self.extractor = ExtractorAgent(model)
        self.detector = DetectorAgent(model)
        self.resolver = ResolverAgent(model)
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.run_conflict_pipeline = run_conflict_pipeline

    async def index_passages(self, passages: Sequence[Passage]) -> IndexingReport:
        report = IndexingReport()
        newly_accepted: List[Fact] = []

        for passage in passages:
            self.memory.add_passage(passage)
            report.passages_indexed += 1

            if self.vector_store is not None:
                await self.vector_store.add_texts(
                    texts=[passage.text],
                    ids=[passage.id],
                    metadatas=[passage.metadata],
                    namespace=LAYER_M_PAS,
                )
                if self.graph_store is not None:
                    await self.graph_store.add_node(
                        GraphNode(id=f"pas_{passage.id}", label=NODE_PASSAGE)
                    )

            extraction = await self.extractor.extract(passage)
            candidates = self.extractor.triples_to_facts(extraction.triples, passage.id)

            for fact in candidates:
                schema = self.memory.register_schema_candidate(
                    fact.head_type, fact.relation, fact.tail_type
                )
                report.schemas_registered += 1

                if self.vector_store is not None:
                    await self.vector_store.add_texts(
                        texts=[f"{schema.head_type} {schema.relation} {schema.tail_type}"],
                        ids=[schema.schema_key],
                        namespace=LAYER_M_ONT,
                    )
                if self.graph_store is not None:
                    await self.graph_store.add_node(
                        GraphNode(
                            id=f"ont_{schema.schema_key}",
                            label=NODE_SCHEMA,
                            properties={"frequency": schema.frequency},
                        )
                    )

                accepted = self.memory.add_fact(fact)
                if accepted is None:
                    report.facts_rejected_by_tau += 1
                    report.rejected_triples.append(fact)
                    continue

                report.facts_accepted += 1
                report.accepted_fact_keys.append(accepted.fact_key)
                newly_accepted.append(accepted)
                await self._sync_fact(accepted)

        if self.run_conflict_pipeline and len(self.memory.facts) >= 2:
            focus_keys = set(report.accepted_fact_keys)
            focus_facts = [
                f
                for f in self.memory.facts.values()
                if f.fact_key in focus_keys
                or any(
                    f.head_entity.strip().lower() == n.head_entity.strip().lower()
                    and f.relation.strip().lower() == n.relation.strip().lower()
                    for n in newly_accepted
                )
            ]
            detection = await self.detector.detect(
                focus_facts or list(self.memory.facts.values())
            )
            report.conflicts = detection.conflicts
            report.conflicts_detected = len(detection.conflicts)

            if detection.conflicts:
                resolution = await self.resolver.resolve(
                    detection.conflicts, dict(self.memory.facts)
                )
                report.decisions = resolution.decisions
                self.resolver.apply_decisions(self.memory, resolution.decisions)
                report.decisions_applied = len(resolution.decisions)

        return report

    async def _sync_fact(self, fact: Fact) -> None:
        if self.vector_store is not None:
            await self.vector_store.add_texts(
                texts=[f"{fact.head_entity} {fact.relation} {fact.tail_entity}"],
                ids=[fact.fact_key],
                namespace=LAYER_M_FAC,
            )
        if self.graph_store is not None:
            fid = f"fact_{fact.fact_key}"
            await self.graph_store.add_node(
                GraphNode(
                    id=fid,
                    label=NODE_FACT,
                    properties={
                        "head": fact.head_entity,
                        "relation": fact.relation,
                        "tail": fact.tail_entity,
                    },
                )
            )
            await self.graph_store.add_edge(
                GraphEdge(
                    source_id=f"ont_{fact.schema_key}",
                    target_id=fid,
                    relation=EDGE_INSTANTIATES,
                )
            )
            for pid in fact.passage_ids:
                await self.graph_store.add_node(
                    GraphNode(id=f"pas_{pid}", label=NODE_PASSAGE)
                )
                await self.graph_store.add_edge(
                    GraphEdge(
                        source_id=fid,
                        target_id=f"pas_{pid}",
                        relation=EDGE_EVIDENCE_IN,
                    )
                )

"""Ingestion pipeline: document → passages → extract → resolve → tau-gated facts."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

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
from memgraphrag_core.ingestion.entity_resolver import EntityResolver
from memgraphrag_core.ingestion.extractor import LLMExtractor
from memgraphrag_core.ingestion.schemas import ExtractedFact
from memgraphrag_core.interfaces.graph_store import BaseGraphStore, GraphEdge, GraphNode
from memgraphrag_core.interfaces.llm_provider import BaseLLMProvider
from memgraphrag_core.interfaces.vector_store import BaseVectorStore


def split_into_passages(text: str, document_id: str) -> List[Passage]:
    """Split unstructured text into Passage chunks (paragraph-aware)."""
    chunks = [c.strip() for c in re.split(r"\n\s*\n+", text.strip()) if c.strip()]
    if not chunks:
        chunks = [text.strip()] if text.strip() else []
    # Fallback: if a single huge paragraph, split on sentence boundaries lightly.
    if len(chunks) == 1 and len(chunks[0]) > 500:
        sentences = re.split(r"(?<=[.!?])\s+", chunks[0])
        grouped: List[str] = []
        buf: List[str] = []
        for s in sentences:
            buf.append(s)
            if sum(len(x) for x in buf) >= 280:
                grouped.append(" ".join(buf))
                buf = []
        if buf:
            grouped.append(" ".join(buf))
        chunks = grouped or chunks

    passages: List[Passage] = []
    for i, chunk in enumerate(chunks, start=1):
        passages.append(
            Passage(
                id=f"{document_id}_pas_{i:03d}",
                text=chunk,
                metadata={"document_id": document_id, "ordinal": i},
            )
        )
    return passages


class IngestionPipeline:
    """End-to-end ingestion over abstract LLM / vector / graph ports."""

    def __init__(
        self,
        memory_state: GlobalMemoryState,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore,
        llm: BaseLLMProvider,
        entity_resolver: Optional[EntityResolver] = None,
    ) -> None:
        self.memory_state = memory_state
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.extractor = LLMExtractor(llm)
        self.entity_resolver = entity_resolver or EntityResolver(vector_store)

    async def process_document(self, text: str, document_id: str) -> Dict[str, Any]:
        """Ingest an unstructured document into the three-layer memory."""
        report: Dict[str, Any] = {
            "document_id": document_id,
            "passages": [],
            "schemas_registered": 0,
            "facts_extracted": 0,
            "facts_resolved": 0,
            "facts_accepted": 0,
            "facts_rejected_by_tau": 0,
            "accepted_fact_keys": [],
            "rejected_fact_keys": [],
            "canonical_map": {},
            "errors": [],
            "skipped_empty_document": False,
        }

        if not text or not text.strip():
            report["skipped_empty_document"] = True
            return report

        passages = split_into_passages(text, document_id)
        for passage in passages:
            await self._index_passage(passage)
            report["passages"].append(passage.id)

            try:
                extraction = await self.extractor.extract_from_text(passage.text)
            except Exception as exc:  # noqa: BLE001 — surfaced in report for batch resilience
                report["errors"].append(
                    {"passage_id": passage.id, "error": str(exc)}
                )
                continue

            for schema in extraction.schemas:
                self.memory_state.register_schema_candidate(
                    schema.head_type, schema.relation, schema.tail_type
                )
                report["schemas_registered"] += 1
                await self._index_schema(schema.head_type, schema.relation, schema.tail_type)

            report["facts_extracted"] += len(extraction.facts)
            resolved = await self.entity_resolver.resolve_facts(extraction.facts)
            report["facts_resolved"] += len(resolved)

            for original, canon in zip(extraction.facts, resolved):
                if original.head_entity != canon.head_entity:
                    report["canonical_map"][original.head_entity] = canon.head_entity
                if original.tail_entity != canon.tail_entity:
                    report["canonical_map"][original.tail_entity] = canon.tail_entity

            for fact in resolved:
                accepted = await self._try_add_fact(fact, passage.id)
                if accepted is None:
                    report["facts_rejected_by_tau"] += 1
                    report["rejected_fact_keys"].append(
                        f"{fact.head_entity}--{fact.relation}--{fact.tail_entity}".lower()
                    )
                else:
                    report["facts_accepted"] += 1
                    report["accepted_fact_keys"].append(accepted.fact_key)

        return report

    async def _index_passage(self, passage: Passage) -> None:
        self.memory_state.add_passage(passage)
        await self.vector_store.add_texts(
            texts=[passage.text],
            ids=[passage.id],
            metadatas=[passage.metadata],
            namespace=LAYER_M_PAS,
        )
        await self.graph_store.add_node(
            GraphNode(
                id=f"pas_{passage.id}",
                label=NODE_PASSAGE,
                properties={"document_id": passage.metadata.get("document_id")},
            )
        )

    async def _index_schema(self, head_type: str, relation: str, tail_type: str) -> None:
        temp = self.memory_state.ontologies.get(
            f"({head_type.strip().lower()})-[{relation.strip().lower()}]->({tail_type.strip().lower()})"
        )
        schema_key = (
            temp.schema_key
            if temp is not None
            else f"({head_type.strip().lower()})-[{relation.strip().lower()}]->({tail_type.strip().lower()})"
        )
        await self.vector_store.add_texts(
            texts=[f"{head_type} {relation} {tail_type}"],
            ids=[schema_key],
            metadatas=[{"layer": LAYER_M_ONT}],
            namespace=LAYER_M_ONT,
        )
        await self.graph_store.add_node(
            GraphNode(
                id=f"ont_{schema_key}",
                label=NODE_SCHEMA,
                properties={"schema_key": schema_key},
            )
        )

    async def _try_add_fact(self, extracted: ExtractedFact, passage_id: str) -> Optional[Fact]:
        fact = Fact(
            head_entity=extracted.head_entity,
            head_type=extracted.head_type,
            relation=extracted.relation,
            tail_entity=extracted.tail_entity,
            tail_type=extracted.tail_type,
            passage_ids=[passage_id],
            source_passage_id=passage_id,
            category=extracted.category,
            ttl_seconds=extracted.ttl_seconds,
            is_invalidated=extracted.is_invalidated,
        )
        accepted = self.memory_state.add_fact(fact)
        if accepted is None:
            return None
        await self._sync_accepted_fact(accepted)
        return accepted

    async def _sync_accepted_fact(self, fact: Fact) -> None:
        await self.vector_store.add_texts(
            texts=[f"{fact.head_entity} {fact.relation} {fact.tail_entity}"],
            ids=[fact.fact_key],
            metadatas=[
                {
                    "layer": LAYER_M_FAC,
                    "head_type": fact.head_type,
                    "tail_type": fact.tail_type,
                    "category": fact.category.value,
                    "source_passage_id": fact.source_passage_id,
                }
            ],
            namespace=LAYER_M_FAC,
        )
        # Keep entity registry warm for future canonicalization.
        await self.entity_resolver.register_entity(fact.head_entity, fact.head_type)
        await self.entity_resolver.register_entity(fact.tail_entity, fact.tail_type)

        fact_node = f"fact_{fact.fact_key}"
        await self.graph_store.add_node(
            GraphNode(
                id=fact_node,
                label=NODE_FACT,
                properties=fact.graph_properties(),
            )
        )
        # Canonical hierarchy: Schema -INSTANTIATES-> Fact -EVIDENCE_IN-> Passage
        await self.graph_store.add_edge(
            GraphEdge(
                source_id=f"ont_{fact.schema_key}",
                target_id=fact_node,
                relation=EDGE_INSTANTIATES,
            )
        )
        for pid in fact.passage_ids:
            await self.graph_store.add_edge(
                GraphEdge(
                    source_id=fact_node,
                    target_id=f"pas_{pid}",
                    relation=EDGE_EVIDENCE_IN,
                )
            )

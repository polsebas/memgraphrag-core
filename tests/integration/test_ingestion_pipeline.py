"""E2E tests for LLM ingestion + entity resolution + tau gating."""

from __future__ import annotations

import pytest

from memgraphrag_core.domain.models import GlobalMemoryState
from memgraphrag_core.ingestion.entity_resolver import EntityResolver
from memgraphrag_core.ingestion.extractor import to_pascal_case, to_upper_snake
from memgraphrag_core.ingestion.pipeline import IngestionPipeline
from memgraphrag_core.ingestion.schemas import (
    DocumentExtractionResult,
    ExtractedFact,
    ExtractedSchema,
)
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.mock_llm import MockLLMProvider
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore


def test_normalization_helpers() -> None:
    assert to_upper_snake("located in") == "LOCATED_IN"
    assert to_upper_snake("produces") == "PRODUCES"
    assert to_pascal_case("wine region") == "WineRegion"
    assert to_pascal_case("winery") == "Winery"


@pytest.mark.asyncio
async def test_entity_resolver_unifies_name_variants() -> None:
    vectors = InMemoryVectorStore()
    # Domain-specific noise for this wine fixture only (not core defaults).
    resolver = EntityResolver(
        vectors,
        similarity_threshold=0.5,
        noise_tokens={"winery", "bodega"},
    )

    canonical = await resolver.canonicalize_entity("Catena Zapata", "Winery")
    assert canonical == "Catena Zapata"

    # Surface variants should collapse to the first canonical form.
    v1 = await resolver.canonicalize_entity("Bodega Catena Zapata", "Winery")
    v2 = await resolver.canonicalize_entity("Catena Zapata S.A.", "Winery")
    assert v1 == "Catena Zapata"
    assert v2 == "Catena Zapata"

    facts = await resolver.resolve_facts(
        [
            ExtractedFact(
                head_entity="Bodega Catena Zapata",
                head_type="Winery",
                relation="LOCATED_IN",
                tail_entity="Mendoza",
                tail_type="Region",
            )
        ]
    )
    assert facts[0].head_entity == "Catena Zapata"


@pytest.mark.asyncio
async def test_ingestion_pipeline_tau_and_canonicalization() -> None:
    """Ingest multi-passage text: unify entity variants; tau gates M_fac writes."""

    doc = (
        "Catena Zapata winery is located in Mendoza, Argentina.\n\n"
        "Bodega Catena Zapata produces the Gran Enemigo wine in Gualtallary.\n\n"
        "Catena Zapata S.A. is located in Mendoza and produces Gran Enemigo."
    )

    def extract_responder(prompt: str, _system, _model):
        text = prompt.lower()
        if "bodega catena zapata produces" in text:
            return DocumentExtractionResult(
                schemas=[
                    ExtractedSchema(
                        head_type="Winery", relation="PRODUCES", tail_type="Wine"
                    )
                ],
                facts=[
                    ExtractedFact(
                        head_entity="Bodega Catena Zapata",
                        head_type="Winery",
                        relation="PRODUCES",
                        tail_entity="Gran Enemigo",
                        tail_type="Wine",
                    )
                ],
            )
        if "catena zapata s.a." in text:
            return DocumentExtractionResult(
                schemas=[
                    ExtractedSchema(
                        head_type="Winery", relation="LOCATED_IN", tail_type="Region"
                    ),
                    ExtractedSchema(
                        head_type="Winery", relation="PRODUCES", tail_type="Wine"
                    ),
                ],
                facts=[
                    ExtractedFact(
                        head_entity="Catena Zapata S.A.",
                        head_type="Winery",
                        relation="LOCATED_IN",
                        tail_entity="Mendoza",
                        tail_type="Region",
                    ),
                    ExtractedFact(
                        head_entity="Catena Zapata S.A.",
                        head_type="Winery",
                        relation="PRODUCES",
                        tail_entity="Gran Enemigo",
                        tail_type="Wine",
                    ),
                ],
            )
        # First passage
        return DocumentExtractionResult(
            schemas=[
                ExtractedSchema(
                    head_type="Winery", relation="LOCATED_IN", tail_type="Region"
                )
            ],
            facts=[
                ExtractedFact(
                    head_entity="Catena Zapata",
                    head_type="Winery",
                    relation="LOCATED_IN",
                    tail_entity="Mendoza",
                    tail_type="Region",
                )
            ],
        )

    llm = MockLLMProvider(structured_responses={"DocumentExtractionResult": extract_responder})
    memory = GlobalMemoryState(tau=2)
    vectors = InMemoryVectorStore()
    graph = NetworkXGraphStore()
    pipeline = IngestionPipeline(
        memory_state=memory,
        vector_store=vectors,
        graph_store=graph,
        llm=llm,
        entity_resolver=EntityResolver(
            vectors,
            similarity_threshold=0.5,
            noise_tokens={"winery", "bodega"},
        ),
    )

    report = await pipeline.process_document(doc, document_id="doc_wine_001")

    assert len(report["passages"]) == 3
    assert report["schemas_registered"] >= 2
    # First mentions of each schema are rejected until frequency reaches tau=2.
    assert report["facts_rejected_by_tau"] >= 1
    assert report["facts_accepted"] >= 1

    # Entity variants collapsed to a single canonical winery head.
    winery_heads = {f.head_entity for f in memory.facts.values() if f.head_type == "Winery"}
    assert winery_heads == {"Catena Zapata"}

    # Stable schemas only instantiate accepted facts.
    assert all(
        memory.ontologies[f.schema_key].is_stable(memory.tau) for f in memory.facts.values()
    )

    # Canonical map should record at least one alias rewrite.
    assert report["canonical_map"]
    assert any(
        "catena" in src.lower() and dst == "Catena Zapata"
        for src, dst in report["canonical_map"].items()
    )

    # Graph edges: Schema -INSTANTIATES-> Fact -EVIDENCE_IN-> Passage
    for fkey in report["accepted_fact_keys"]:
        fact = memory.facts[fkey]
        evidence = await graph.get_neighbors(f"fact_{fkey}", relation_type="EVIDENCE_IN")
        assert evidence, f"missing EVIDENCE_IN edges for {fkey}"
        instantiated = await graph.get_neighbors(
            f"ont_{fact.schema_key}", relation_type="INSTANTIATES"
        )
        assert any(n.id == f"fact_{fkey}" for n in instantiated), (
            f"missing INSTANTIATES edge ont→fact for {fkey}"
        )


@pytest.mark.asyncio
async def test_ingestion_skips_empty_document() -> None:
    llm = MockLLMProvider(structured_responses={})
    memory = GlobalMemoryState(tau=2)
    pipeline = IngestionPipeline(
        memory_state=memory,
        vector_store=InMemoryVectorStore(),
        graph_store=NetworkXGraphStore(),
        llm=llm,
    )
    report = await pipeline.process_document("   \n  ", document_id="doc_empty")
    assert report["skipped_empty_document"] is True
    assert report["passages"] == []
    assert report["facts_extracted"] == 0
    assert llm.calls == []


@pytest.mark.asyncio
async def test_ingestion_isolates_passage_extraction_errors() -> None:
    def extract_responder(prompt: str, _system, _model):
        if "boom" in prompt.lower():
            raise RuntimeError("llm timeout")
        return DocumentExtractionResult(
            schemas=[
                ExtractedSchema(
                    head_type="Winery", relation="LOCATED_IN", tail_type="Region"
                )
            ],
            facts=[
                ExtractedFact(
                    head_entity="Catena Zapata",
                    head_type="Winery",
                    relation="LOCATED_IN",
                    tail_entity="Mendoza",
                    tail_type="Region",
                )
            ],
        )

    llm = MockLLMProvider(
        structured_responses={"DocumentExtractionResult": extract_responder}
    )
    memory = GlobalMemoryState(tau=1)
    vectors = InMemoryVectorStore()
    graph = NetworkXGraphStore()
    pipeline = IngestionPipeline(
        memory_state=memory,
        vector_store=vectors,
        graph_store=graph,
        llm=llm,
        entity_resolver=EntityResolver(vectors, similarity_threshold=0.9),
    )

    doc = "This passage will boom on purpose.\n\nCatena Zapata is located in Mendoza."
    report = await pipeline.process_document(doc, document_id="doc_partial")

    assert len(report["passages"]) == 2
    assert len(report["errors"]) == 1
    assert report["errors"][0]["error"] == "llm timeout"
    assert report["facts_accepted"] >= 1
    assert memory.facts

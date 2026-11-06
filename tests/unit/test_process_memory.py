"""MemDev-Bench: process events stay stored, and TTL decides if they apply."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memgraphrag_core.domain.models import Fact, GlobalMemoryState, Passage
from memgraphrag_core.domain.process_memory import (
    ApplicabilityVerdict,
    ApplicabilityWindow,
    ProcessKind,
)
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.infrastructure.mock_llm import MockLLMProvider
from memgraphrag_core.ingestion.pipeline import IngestionPipeline
from memgraphrag_core.ingestion.schemas import DocumentExtractionResult, ExtractedFact
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever

T0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _unstable_fact() -> Fact:
    return Fact(
        head_entity="Catena Zapata",
        head_type="Winery",
        relation="PRODUCES",
        tail_entity="Gran Enemigo",
        tail_type="Wine",
        passage_ids=["pas_001"],
    )


def test_ttl_window_not_yet_applicable_and_expired() -> None:
    window = ApplicabilityWindow.from_ttl(60, valid_from=T0)
    assert window.judge(T0 - timedelta(seconds=1)) is ApplicabilityVerdict.NOT_YET
    assert window.judge(T0) is ApplicabilityVerdict.APPLICABLE
    assert window.judge(T0 + timedelta(seconds=59)) is ApplicabilityVerdict.APPLICABLE
    assert window.judge(T0 + timedelta(seconds=60)) is ApplicabilityVerdict.EXPIRED
    assert ApplicabilityWindow.from_ttl(0, valid_from=T0).judge(T0) is ApplicabilityVerdict.EXPIRED
    with pytest.raises(ValueError):
        ApplicabilityWindow.from_ttl(-1, valid_from=T0)


def test_process_log_keeps_failures_rejections_and_constraints_in_order() -> None:
    memory = GlobalMemoryState(tau=2)
    memory.register_schema_candidate("Winery", "PRODUCES", "Wine")
    failed = memory.record_failed_attempt(
        "pytest exited 1 on test_migration", passage_id="pas_001", valid_from=T0
    )
    rejected = memory.record_rejection(
        _unstable_fact(), "pas_001", reason="schema_below_tau", valid_from=T0
    )
    constraint = memory.record_negative_constraint(
        "do not retry the flaky migration",
        passage_ids=["pas_001"],
        ttl_seconds=60,
        valid_from=T0,
    )

    assert [record.kind for record in memory.process_log.values()] == [
        ProcessKind.FAILED_ATTEMPT,
        ProcessKind.REJECTION,
        ProcessKind.NEGATIVE_CONSTRAINT,
    ]
    assert [failed.ordinal, rejected.ordinal, constraint.ordinal] == [1, 2, 3]
    assert memory.facts == {}
    assert rejected.related_fact_key == "catena zapata--produces--gran enemigo"

    live = memory.applicable_process_records(T0)
    assert [record.record_id for record in live] == [
        failed.record_id,
        rejected.record_id,
        constraint.record_id,
    ]
    later = memory.applicable_process_records(T0 + timedelta(seconds=60))
    assert constraint.record_id not in {record.record_id for record in later}
    assert rejected.record_id in {record.record_id for record in later}
    assert rejected.record_id in memory.process_log


def test_accepting_a_fact_expires_its_rejection_without_deleting_it() -> None:
    memory = GlobalMemoryState(tau=1)
    memory.register_schema_candidate("Winery", "PRODUCES", "Wine")
    rejected = memory.record_rejection(
        _unstable_fact(), "pas_001", reason="schema_below_tau", valid_from=T0
    )
    accepted = memory.add_fact(_unstable_fact())
    assert accepted is not None
    assert rejected.record_id in memory.process_log
    assert memory.applicable_process_records(datetime.now(timezone.utc)) == []
    assert (
        memory.process_log[rejected.record_id].applicability.judge(T0)
        is ApplicabilityVerdict.APPLICABLE
    )


@pytest.mark.asyncio
async def test_ingestion_records_rejection_outside_m_fac() -> None:
    def responder(_prompt, _system, _model):
        return DocumentExtractionResult(
            facts=[
                ExtractedFact(
                    head_entity="Catena Zapata",
                    head_type="Winery",
                    relation="PRODUCES",
                    tail_entity="Gran Enemigo",
                    tail_type="Wine",
                )
            ]
        )

    memory = GlobalMemoryState(tau=2)
    vectors = InMemoryVectorStore()
    graph = NetworkXGraphStore()
    pipeline = IngestionPipeline(
        memory, vectors, graph, MockLLMProvider({"DocumentExtractionResult": responder})
    )
    report = await pipeline.process_document(
        "Catena Zapata produces Gran Enemigo.", document_id="doc_rej"
    )

    assert report["facts_rejected_by_tau"] == 1
    assert report["facts_accepted"] == 0
    assert memory.facts == {}
    rejections = [
        record
        for record in memory.process_log.values()
        if record.kind is ProcessKind.REJECTION
    ]
    assert len(rejections) == 1
    assert rejections[0].metadata["reason"] == "schema_below_tau"
    fact_nodes = [nid for nid in graph._g.nodes if str(nid).startswith("fact_")]  # noqa: SLF001
    assert fact_nodes == []
    fac_hits = await vectors.search_similarity("Gran Enemigo", k=5, namespace="M_fac")
    assert fac_hits == []


@pytest.mark.asyncio
async def test_ingestion_preserves_failed_attempt() -> None:
    def responder(_prompt, _system, _model):
        raise RuntimeError("extractor timed out")

    memory = GlobalMemoryState(tau=1)
    pipeline = IngestionPipeline(
        memory,
        InMemoryVectorStore(),
        NetworkXGraphStore(),
        MockLLMProvider({"DocumentExtractionResult": responder}),
    )
    report = await pipeline.process_document("unreadable blob", document_id="doc_fail")
    assert report["errors"]
    attempts = [
        record
        for record in memory.process_log.values()
        if record.kind is ProcessKind.FAILED_ATTEMPT
    ]
    assert len(attempts) == 1
    assert "timed out" in attempts[0].summary
    assert memory.passages


@pytest.mark.asyncio
async def test_retrieve_applies_ttl_judgment() -> None:
    memory = GlobalMemoryState(tau=1)
    memory.add_passage(Passage(id="pas_001", text="unused"))
    constraint = memory.record_negative_constraint(
        "do not use the deprecated API",
        ttl_seconds=10,
        valid_from=T0,
    )
    retriever = MemoryGuidedRetriever(
        memory, NetworkXGraphStore(), InMemoryVectorStore()
    )
    live = await retriever.retrieve("deprecated API", as_of=T0)
    expired = await retriever.retrieve(
        "deprecated API", as_of=T0 + timedelta(seconds=10)
    )
    assert [record.record_id for record in live.process_records] == [constraint.record_id]
    assert expired.process_records == []
    assert constraint.record_id in memory.process_log
    assert expired.applicability_as_of == T0 + timedelta(seconds=10)

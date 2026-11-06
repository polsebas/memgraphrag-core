"""DiscernMem: prune context that would be misapplied, keep evidence of facts that stay."""

from __future__ import annotations

import pytest

from memgraphrag_core.domain.models import LAYER_M_PAS, Fact, GlobalMemoryState, Passage
from memgraphrag_core.domain.process_memory import (
    ApplicabilityWindow,
    ProcessKind,
    ProcessRecord,
)
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.interfaces.graph_store import GraphEdge, GraphNode
from memgraphrag_core.retrieval.context_pruner import PruneReason, SelectiveContextPruner
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever
from tests.unit.test_process_memory import T0


def _fact(passage_id: str = "pas_ev", **metadata: object) -> Fact:
    return Fact(
        head_entity="Catena Zapata",
        head_type="Winery",
        relation="LOCATED_IN",
        tail_entity="Mendoza",
        tail_type="Region",
        passage_ids=[passage_id],
        metadata=dict(metadata),
    )


def _reasons(result, kind: str) -> dict[str, PruneReason]:
    return {
        decision.item_id: decision.reason
        for decision in result.decisions
        if decision.kind == kind
    }


def test_avoid_boundary_outranks_a_matching_use_condition() -> None:
    fact = _fact(
        avoid_conditions=["Mendoza"],
        use_conditions=["Mendoza"],
    )
    passage = Passage(id="pas_ev", text="Catena Zapata is in Mendoza.")
    pruned = SelectiveContextPruner().prune(
        query="Where is the winery in Mendoza?",
        passages=[passage],
        facts=[fact],
    )
    assert pruned.facts == []
    assert _reasons(pruned, "fact")[fact.fact_key] is PruneReason.AVOID_BOUNDARY
    assert pruned.passages[0].id == "pas_ev"


def test_use_condition_miss_is_out_of_scope() -> None:
    fact = _fact(use_conditions=["cellar code"])
    pruned = SelectiveContextPruner().prune(
        query="Where is the winery located?",
        passages=[Passage(id="pas_ev", text="Catena Zapata is in Mendoza.")],
        facts=[fact],
    )
    assert pruned.facts == []
    assert _reasons(pruned, "fact")[fact.fact_key] is PruneReason.OUT_OF_SCOPE


def test_budget_keeps_evidence_of_a_fact_that_stays() -> None:
    noise = Passage(id="pas_noise", text="Generic tourism brochure about beaches.")
    evidence = Passage(id="pas_ev", text="Catena Zapata is in Mendoza.")
    fact = _fact()
    pruned = SelectiveContextPruner(max_passages=1).prune(
        query="Where is Catena Zapata?",
        passages=[noise, evidence],
        facts=[fact],
    )
    assert [passage.id for passage in pruned.passages] == ["pas_ev"]
    assert pruned.facts[0].fact_key == fact.fact_key
    reasons = _reasons(pruned, "passage")
    assert reasons["pas_noise"] is PruneReason.BUDGET
    assert reasons["pas_ev"] is PruneReason.DEPENDENCY_KEPT


def test_avoidance_overrides_protection_and_drops_the_ungrounded_fact() -> None:
    passage = Passage(
        id="pas_ev",
        text="The cellar access code is 7341.",
        metadata={"avoid_conditions": ["cellar access code"]},
    )
    pruned = SelectiveContextPruner(max_passages=1).prune(
        query="What is the cellar access code?",
        passages=[passage],
        facts=[_fact()],
        protected_passage_ids={"pas_ev"},
    )
    assert pruned.passages == []
    assert pruned.facts == []
    assert _reasons(pruned, "passage")["pas_ev"] is PruneReason.AVOID_BOUNDARY
    assert _reasons(pruned, "fact")[_fact().fact_key] is PruneReason.UNGROUNDED


def test_writer_omitted_passage_survives_a_tight_budget() -> None:
    secret = Passage(id="pas_secret", text="The cellar access code is 7341.")
    noise = Passage(id="pas_noise", text="beaches hotels nightlife")
    pruned = SelectiveContextPruner(max_passages=0).prune(
        query="cellar access code",
        passages=[noise, secret],
        facts=[],
        protected_passage_ids={"pas_secret"},
    )
    assert [passage.id for passage in pruned.passages] == ["pas_secret"]
    assert _reasons(pruned, "passage")["pas_secret"] is PruneReason.DEPENDENCY_KEPT


def test_expired_and_defective_process_records_are_not_injected() -> None:
    window = ApplicabilityWindow.from_ttl(0, valid_from=T0)
    expired = ProcessRecord(
        record_id="neg:0001",
        kind=ProcessKind.NEGATIVE_CONSTRAINT,
        summary="do not retry the flaky migration",
        constraint="do not retry the flaky migration",
        applicability=window,
        ordinal=1,
    )
    defective = ProcessRecord(
        record_id="fail:0002",
        kind=ProcessKind.FAILED_ATTEMPT,
        summary="compile failed",
        applicability=ApplicabilityWindow.from_ttl(None, valid_from=T0),
        ordinal=2,
        metadata={"defective": True},
    )
    original_until = expired.applicability.valid_until
    pruned = SelectiveContextPruner().prune(
        query="migration",
        passages=[],
        facts=[],
        process_records=[expired, defective],
        as_of=T0,
    )
    assert pruned.process_records == []
    reasons = _reasons(pruned, "process")
    assert reasons["neg:0001"] is PruneReason.EXPIRED
    assert reasons["fail:0002"] is PruneReason.DEFECTIVE
    assert expired.applicability.valid_until == original_until


def test_char_budget_drops_the_lowest_ranked_expendable_passage() -> None:
    first = Passage(id="pas_a", text="1234567890")
    second = Passage(id="pas_b", text="abcdefghij")
    pruned = SelectiveContextPruner(max_chars=10).prune(
        query="anything",
        passages=[first, second],
        facts=[],
    )
    assert [passage.id for passage in pruned.passages] == ["pas_a"]
    assert _reasons(pruned, "passage")["pas_b"] is PruneReason.BUDGET


@pytest.mark.asyncio
async def test_retriever_prunes_avoided_fact_but_keeps_raw_passage_access() -> None:
    memory = GlobalMemoryState(tau=1)
    graph = NetworkXGraphStore()
    vectors = InMemoryVectorStore()
    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
    passage = Passage(id="pas_loc", text="Catena Zapata winery is located in Mendoza.")
    memory.add_passage(passage)
    fact = memory.add_fact(
        Fact(
            head_entity="Catena Zapata",
            head_type="Winery",
            relation="LOCATED_IN",
            tail_entity="Mendoza",
            tail_type="Region",
            passage_ids=["pas_loc"],
            metadata={"avoid_conditions": ["Mendoza"]},
        )
    )
    assert fact is not None
    await vectors.add_texts(texts=[passage.text], ids=[passage.id], namespace=LAYER_M_PAS)
    await vectors.add_texts(
        texts=["Catena Zapata LOCATED_IN Mendoza"],
        ids=[fact.fact_key],
        namespace="M_fac",
    )
    await graph.add_node(GraphNode(id=f"ont_{fact.schema_key}", label="Schema"))
    await graph.add_node(GraphNode(id=f"fact_{fact.fact_key}", label="Fact"))
    await graph.add_node(GraphNode(id="pas_pas_loc", label="Passage"))
    await graph.add_edge(
        GraphEdge(
            source_id=f"ont_{fact.schema_key}",
            target_id=f"fact_{fact.fact_key}",
            relation="INSTANTIATES",
        )
    )
    await graph.add_edge(
        GraphEdge(
            source_id=f"fact_{fact.fact_key}",
            target_id="pas_pas_loc",
            relation="EVIDENCE_IN",
        )
    )

    retriever = MemoryGuidedRetriever(
        memory,
        graph,
        vectors,
        context_pruner=SelectiveContextPruner(),
    )
    result = await retriever.retrieve("Where is Catena Zapata in Mendoza?")

    assert fact.fact_key in memory.facts
    assert all(item.fact_key != fact.fact_key for item in result.facts)
    assert any(
        decision.item_id == fact.fact_key and decision.reason is PruneReason.AVOID_BOUNDARY
        for decision in result.prune_decisions
    )
    assert any(item.id == "pas_loc" for item in result.direct_passages)
    assert all(not node["id"].startswith("fact_") for node in result.subgraph["nodes"])

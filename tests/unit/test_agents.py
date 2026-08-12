"""Tests for A_ext / A_det / A_res Agno agents and the indexing orchestrator."""

from __future__ import annotations

import pytest
from agno.models.message import Message

from memgraphrag_core.agents.detector import DetectorAgent, candidate_pairs
from memgraphrag_core.agents.extractor import ExtractorAgent
from memgraphrag_core.agents.models import (
    ConflictRecord,
    DetectionResult,
    ExtractionResult,
    ExtractedTriple,
    ResolutionAction,
    ResolutionDecision,
    ResolutionResult,
)
from memgraphrag_core.agents.resolver import ResolverAgent
from memgraphrag_core.domain.models import ConflictType, Fact, GlobalMemoryState, Passage
from memgraphrag_core.indexing.orchestrator import IndexingOrchestrator
from memgraphrag_core.infrastructure.scripted_model import ScriptedModel


def test_candidate_pairs_same_head_relation() -> None:
    a = Fact(
        head_entity="Catena Zapata",
        head_type="Winery",
        relation="LOCATED_IN",
        tail_entity="Mendoza",
        tail_type="Region",
    )
    b = Fact(
        head_entity="Catena Zapata",
        head_type="Winery",
        relation="LOCATED_IN",
        tail_entity="Buenos Aires",
        tail_type="Region",
    )
    c = Fact(
        head_entity="Norton",
        head_type="Winery",
        relation="PRODUCES",
        tail_entity="Malbec",
        tail_type="Wine",
    )
    pairs = candidate_pairs([a, b, c])
    assert len(pairs) == 1
    assert {pairs[0][0].fact_key, pairs[0][1].fact_key} == {a.fact_key, b.fact_key}


@pytest.mark.asyncio
async def test_extractor_maps_triples_to_facts() -> None:
    model = ScriptedModel(
        structured_responses={
            "ExtractionResult": ExtractionResult(
                triples=[
                    ExtractedTriple(
                        head_entity="Catena Zapata",
                        head_type="Winery",
                        relation="located in",
                        tail_entity="Mendoza",
                        tail_type="Region",
                        confidence=0.9,
                        evidence_span="located in the Mendoza region",
                    )
                ]
            )
        }
    )
    agent = ExtractorAgent(model)
    passage = Passage(id="pas_001", text="Catena Zapata is in Mendoza.")
    result = await agent.extract(passage)
    facts = agent.triples_to_facts(result.triples, passage.id)
    assert len(facts) == 1
    assert facts[0].relation == "LOCATED_IN"
    assert facts[0].passage_ids == ["pas_001"]


@pytest.mark.asyncio
async def test_resolver_apply_keep_b() -> None:
    memory = GlobalMemoryState(tau=1)
    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
    a = Fact(
        head_entity="Catena Zapata",
        head_type="Winery",
        relation="LOCATED_IN",
        tail_entity="Buenos Aires",
        tail_type="Region",
        confidence=0.4,
        passage_ids=["p1"],
    )
    b = Fact(
        head_entity="Catena Zapata",
        head_type="Winery",
        relation="LOCATED_IN",
        tail_entity="Mendoza",
        tail_type="Region",
        confidence=0.95,
        passage_ids=["p2"],
    )
    memory.add_fact(a)
    memory.add_fact(b)

    resolver = ResolverAgent(ScriptedModel())
    decision = ResolutionDecision(
        conflict_type=ConflictType.LOGICAL_CONTRADICTION,
        fact_a_key=a.fact_key,
        fact_b_key=b.fact_key,
        action=ResolutionAction.KEEP_B,
        rationale="Mendoza has stronger evidence",
    )
    resolver.apply_decisions(memory, [decision])
    assert a.fact_key not in memory.facts
    assert b.fact_key in memory.facts


@pytest.mark.asyncio
async def test_indexing_orchestrator_tau_and_conflict() -> None:
    """First schema mention is rejected; once stable, conflict is resolved."""

    extraction_by_passage = {
        "pas_001": ExtractionResult(
            triples=[
                ExtractedTriple(
                    head_entity="Catena Zapata",
                    head_type="Winery",
                    relation="LOCATED_IN",
                    tail_entity="Mendoza",
                    tail_type="Region",
                    confidence=0.9,
                )
            ]
        ),
        "pas_002": ExtractionResult(
            triples=[
                ExtractedTriple(
                    head_entity="Catena Zapata",
                    head_type="Winery",
                    relation="LOCATED_IN",
                    tail_entity="Buenos Aires",
                    tail_type="Region",
                    confidence=0.5,
                )
            ]
        ),
    }

    def extract_responder(messages: list[Message], _rf):
        text = "\n".join(str(m.content) for m in messages)
        for pid, payload in extraction_by_passage.items():
            if pid in text:
                return payload
        return ExtractionResult(triples=[])

    def detect_responder(messages: list[Message], _rf):
        return DetectionResult(
            conflicts=[
                ConflictRecord(
                    conflict_type=ConflictType.LOGICAL_CONTRADICTION,
                    fact_a_key="catena zapata--located_in--mendoza",
                    fact_b_key="catena zapata--located_in--buenos aires",
                    explanation="Cannot be located in two provinces at once",
                    severity=0.9,
                )
            ]
        )

    def resolve_responder(messages: list[Message], _rf):
        return ResolutionResult(
            decisions=[
                ResolutionDecision(
                    conflict_type=ConflictType.LOGICAL_CONTRADICTION,
                    fact_a_key="catena zapata--located_in--mendoza",
                    fact_b_key="catena zapata--located_in--buenos aires",
                    action=ResolutionAction.KEEP_A,
                    rationale="Higher confidence and typical evidence",
                )
            ]
        )

    model = ScriptedModel(
        structured_responses={
            "ExtractionResult": extract_responder,
            "DetectionResult": detect_responder,
            "ResolutionResult": resolve_responder,
        }
    )

    memory = GlobalMemoryState(tau=2)
    orch = IndexingOrchestrator(memory, model)

    r1 = await orch.index_passages(
        [Passage(id="pas_001", text="Catena Zapata is in Mendoza.")]
    )
    assert r1.facts_rejected_by_tau == 1
    assert r1.facts_accepted == 0
    assert len(memory.facts) == 0

    r2 = await orch.index_passages(
        [Passage(id="pas_002", text="Some say Catena Zapata is in Buenos Aires.")]
    )
    assert r2.facts_accepted == 1
    assert "catena zapata--located_in--buenos aires" in memory.facts

    extraction_by_passage["pas_003"] = ExtractionResult(
        triples=[
            ExtractedTriple(
                head_entity="Catena Zapata",
                head_type="Winery",
                relation="LOCATED_IN",
                tail_entity="Mendoza",
                tail_type="Region",
                confidence=0.95,
            )
        ]
    )
    r3 = await orch.index_passages(
        [Passage(id="pas_003", text="Catena Zapata is located in Mendoza.")]
    )
    assert r3.facts_accepted == 1
    assert r3.conflicts_detected == 1
    assert r3.decisions_applied == 1
    assert "catena zapata--located_in--mendoza" in memory.facts
    assert "catena zapata--located_in--buenos aires" not in memory.facts

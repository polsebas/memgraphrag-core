#!/usr/bin/env python3
"""Smoke test for Agno indexing agents (A_ext → tau → A_det → A_res)."""

from __future__ import annotations

import asyncio

from agno.models.message import Message

from memgraphrag_core.agents.models import (
    ConflictRecord,
    DetectionResult,
    ExtractionResult,
    ExtractedTriple,
    ResolutionAction,
    ResolutionDecision,
    ResolutionResult,
)
from memgraphrag_core.domain.models import ConflictType, GlobalMemoryState, Passage
from memgraphrag_core.indexing.orchestrator import IndexingOrchestrator
from memgraphrag_core.infrastructure.scripted_model import ScriptedModel


def _build_model() -> ScriptedModel:
    extractions = {
        "pas_001": ExtractionResult(
            triples=[
                ExtractedTriple(
                    head_entity="Catena Zapata",
                    head_type="Winery",
                    relation="LOCATED_IN",
                    tail_entity="Mendoza",
                    tail_type="Region",
                    confidence=0.9,
                ),
                ExtractedTriple(
                    head_entity="Catena Zapata",
                    head_type="Winery",
                    relation="PRODUCES",
                    tail_entity="Gran Enemigo",
                    tail_type="Wine",
                    confidence=0.85,
                ),
            ]
        ),
        "pas_002": ExtractionResult(
            triples=[
                ExtractedTriple(
                    head_entity="Catena Zapata",
                    head_type="Winery",
                    relation="LOCATED_IN",
                    tail_entity="Mendoza",
                    tail_type="Region",
                    confidence=0.92,
                ),
                ExtractedTriple(
                    head_entity="Catena Zapata",
                    head_type="Winery",
                    relation="PRODUCES",
                    tail_entity="Gran Enemigo",
                    tail_type="Wine",
                    confidence=0.9,
                ),
            ]
        ),
        "pas_003": ExtractionResult(
            triples=[
                ExtractedTriple(
                    head_entity="Catena Zapata",
                    head_type="Winery",
                    relation="LOCATED_IN",
                    tail_entity="Buenos Aires",
                    tail_type="Region",
                    confidence=0.4,
                )
            ]
        ),
    }

    def extract(messages: list[Message], _rf):
        text = "\n".join(str(m.content) for m in messages)
        for pid, payload in extractions.items():
            if pid in text:
                return payload
        return ExtractionResult(triples=[])

    def detect(messages: list[Message], _rf):
        text = "\n".join(str(m.content) for m in messages).lower()
        if "buenos aires" in text:
            return DetectionResult(
                conflicts=[
                    ConflictRecord(
                        conflict_type=ConflictType.LOGICAL_CONTRADICTION,
                        fact_a_key="catena zapata--located_in--mendoza",
                        fact_b_key="catena zapata--located_in--buenos aires",
                        explanation="Mutually exclusive locations",
                        severity=0.95,
                    )
                ]
            )
        return DetectionResult(conflicts=[])

    def resolve(messages: list[Message], _rf):
        return ResolutionResult(
            decisions=[
                ResolutionDecision(
                    conflict_type=ConflictType.LOGICAL_CONTRADICTION,
                    fact_a_key="catena zapata--located_in--mendoza",
                    fact_b_key="catena zapata--located_in--buenos aires",
                    action=ResolutionAction.KEEP_A,
                    rationale="Higher confidence and consistent evidence",
                )
            ]
        )

    return ScriptedModel(
        structured_responses={
            "ExtractionResult": extract,
            "DetectionResult": detect,
            "ResolutionResult": resolve,
        }
    )


async def main() -> None:
    memory = GlobalMemoryState(tau=2)
    orch = IndexingOrchestrator(memory, _build_model())

    print("=" * 65)
    print("INDEXING AGENTS SMOKE TEST (Agno A_ext / A_det / A_res)")
    print("=" * 65)

    passages = [
        Passage(
            id="pas_001",
            text="Catena Zapata winery is located in Mendoza and produces Gran Enemigo.",
        ),
        Passage(
            id="pas_002",
            text="Catena Zapata, located in Mendoza, produces the Gran Enemigo wine.",
        ),
        Passage(
            id="pas_003",
            text="An erroneous source places Catena Zapata in Buenos Aires.",
        ),
    ]

    report = await orch.index_passages(passages)

    print(f"Passages indexed: {report.passages_indexed}")
    print(f"Schema registrations (increments): {report.schemas_registered}")
    print(f"Facts accepted: {report.facts_accepted}")
    print(f"Facts rejected by tau: {report.facts_rejected_by_tau}")
    print(f"Conflicts detected: {report.conflicts_detected}")
    print(f"Decisions applied: {report.decisions_applied}")
    print(f"Stable schemas: {len(memory.get_stable_schemas())}")
    print(f"Final facts in M_fac: {len(memory.facts)}")
    for f in memory.facts.values():
        print(
            f"   • {f.head_entity} -[{f.relation}]-> {f.tail_entity} "
            f"(conf={f.confidence}, passages={f.passage_ids})"
        )
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())

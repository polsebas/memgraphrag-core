"""A_res: Agno conflict resolver agent over M_fac."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Set

from agno.agent import Agent

from memgraphrag_core.agents.base import ModelLike, parse_structured_content
from memgraphrag_core.agents.instructions import RESOLVER_INSTRUCTIONS
from memgraphrag_core.agents.models import (
    ConflictRecord,
    ResolutionAction,
    ResolutionDecision,
    ResolutionResult,
)
from memgraphrag_core.domain.models import Fact, GlobalMemoryState


def _fact_brief(fact: Fact) -> dict:
    return {
        "fact_key": fact.fact_key,
        "head": fact.head_entity,
        "relation": fact.relation,
        "tail": fact.tail_entity,
        "confidence": fact.confidence,
        "passage_ids": fact.passage_ids,
    }


class ResolverAgent:
    """A_res agent (Agno): conflicts → decisions → GlobalMemoryState mutation."""

    def __init__(self, model: ModelLike, temperature: float = 0.0) -> None:
        self.agent = Agent(
            name="A_res",
            model=model,
            description="MemGraphRAG conflict resolver (A_res)",
            instructions=[RESOLVER_INSTRUCTIONS],
            output_schema=ResolutionResult,
            use_json_mode=True,
            markdown=False,
            tool_call_limit=0,
        )
        self.temperature = temperature

    async def resolve(
        self,
        conflicts: Iterable[ConflictRecord],
        facts_by_key: Dict[str, Fact],
    ) -> ResolutionResult:
        conflict_list = list(conflicts)
        if not conflict_list:
            return ResolutionResult(decisions=[])

        payload = []
        for c in conflict_list:
            payload.append(
                {
                    "conflict_type": c.conflict_type.value,
                    "explanation": c.explanation,
                    "severity": c.severity,
                    "fact_a": _fact_brief(facts_by_key[c.fact_a_key])
                    if c.fact_a_key in facts_by_key
                    else {"fact_key": c.fact_a_key},
                    "fact_b": _fact_brief(facts_by_key[c.fact_b_key])
                    if c.fact_b_key in facts_by_key
                    else {"fact_key": c.fact_b_key},
                }
            )

        prompt = (
            "Resolve these conflicts:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        response = await self.agent.arun(prompt)
        return parse_structured_content(response.content, ResolutionResult)

    def apply_decisions(
        self,
        memory: GlobalMemoryState,
        decisions: List[ResolutionDecision],
    ) -> List[str]:
        """Apply decisions to M_fac. Returns touched/removed fact keys."""
        touched: List[str] = []
        remove: Set[str] = set()

        for d in decisions:
            a = memory.facts.get(d.fact_a_key)
            b = memory.facts.get(d.fact_b_key)

            if d.action == ResolutionAction.KEEP_A:
                if d.fact_b_key in memory.facts:
                    remove.add(d.fact_b_key)
                touched.append(d.fact_a_key)

            elif d.action == ResolutionAction.KEEP_B:
                if d.fact_a_key in memory.facts:
                    remove.add(d.fact_a_key)
                touched.append(d.fact_b_key)

            elif d.action == ResolutionAction.DISCARD_BOTH:
                remove.add(d.fact_a_key)
                remove.add(d.fact_b_key)

            elif d.action == ResolutionAction.KEEP_BOTH:
                touched.extend([d.fact_a_key, d.fact_b_key])

            elif d.action == ResolutionAction.MERGE:
                survivor = self._merge_facts(a, b, d.merged_confidence)
                if survivor is None:
                    continue
                loser_key = (
                    d.fact_b_key if survivor.fact_key == d.fact_a_key else d.fact_a_key
                )
                memory.facts[survivor.fact_key] = survivor
                if loser_key != survivor.fact_key:
                    remove.add(loser_key)
                touched.append(survivor.fact_key)

        for key in remove:
            discarded = memory.facts.get(key)
            if discarded is not None:
                passage_id = discarded.passage_ids[0] if discarded.passage_ids else ""
                memory.record_rejection(
                    discarded, passage_id, reason="resolution_discard"
                )
            memory.facts.pop(key, None)
            touched.append(key)

        return touched

    @staticmethod
    def _merge_facts(
        a: Optional[Fact], b: Optional[Fact], merged_confidence: Optional[float]
    ) -> Optional[Fact]:
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a

        # Keep the higher-confidence fact as the merge base
        base, other = (a, b) if a.confidence >= b.confidence else (b, a)
        base.passage_ids = list(set(base.passage_ids + other.passage_ids))
        if merged_confidence is not None:
            base.confidence = merged_confidence
        else:
            base.confidence = max(a.confidence, b.confidence)
        return base

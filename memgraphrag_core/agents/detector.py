"""A_det: Agno conflict detector agent for facts in M_fac."""

from __future__ import annotations

import json
from typing import Iterable, List

from agno.agent import Agent

from memgraphrag_core.agents.base import ModelLike, parse_structured_content
from memgraphrag_core.agents.instructions import DETECTOR_INSTRUCTIONS
from memgraphrag_core.agents.models import DetectionResult
from memgraphrag_core.domain.models import Fact


def _fact_brief(fact: Fact) -> dict:
    return {
        "fact_key": fact.fact_key,
        "head": fact.head_entity,
        "head_type": fact.head_type,
        "relation": fact.relation,
        "tail": fact.tail_entity,
        "tail_type": fact.tail_type,
        "confidence": fact.confidence,
        "passage_ids": fact.passage_ids,
    }


def candidate_pairs(facts: List[Fact]) -> List[tuple[Fact, Fact]]:
    """Candidate pairs: same head+relation, or shared schema_key with overlap."""
    pairs: List[tuple[Fact, Fact]] = []
    for i, a in enumerate(facts):
        for b in facts[i + 1 :]:
            same_head_rel = (
                a.head_entity.strip().lower() == b.head_entity.strip().lower()
                and a.relation.strip().lower() == b.relation.strip().lower()
            )
            same_schema = a.schema_key == b.schema_key and (
                a.head_entity.strip().lower() == b.head_entity.strip().lower()
                or a.tail_entity.strip().lower() == b.tail_entity.strip().lower()
            )
            if same_head_rel or same_schema:
                pairs.append((a, b))
    return pairs


class DetectorAgent:
    """A_det agent (Agno): fact set → DetectionResult."""

    def __init__(self, model: ModelLike, temperature: float = 0.0) -> None:
        self.agent = Agent(
            name="A_det",
            model=model,
            description="MemGraphRAG conflict detector (A_det)",
            instructions=[DETECTOR_INSTRUCTIONS],
            output_schema=DetectionResult,
            use_json_mode=True,
            markdown=False,
            tool_call_limit=0,
        )
        self.temperature = temperature

    async def detect(self, facts: Iterable[Fact]) -> DetectionResult:
        fact_list = list(facts)
        pairs = candidate_pairs(fact_list)
        if not pairs:
            return DetectionResult(conflicts=[])

        payload = [
            {"fact_a": _fact_brief(a), "fact_b": _fact_brief(b)} for a, b in pairs
        ]
        prompt = (
            "Evaluate these fact pairs and report conflicts:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        response = await self.agent.arun(prompt)
        return parse_structured_content(response.content, DetectionResult)

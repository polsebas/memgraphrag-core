"""A_ext: Agno extractor agent for triples from passages (M_pas → candidates)."""

from __future__ import annotations

from agno.agent import Agent

from memgraphrag_core.agents.base import ModelLike, parse_structured_content
from memgraphrag_core.agents.instructions import EXTRACTOR_INSTRUCTIONS
from memgraphrag_core.agents.models import ExtractionResult, ExtractedTriple
from memgraphrag_core.domain.models import Fact, Passage


class ExtractorAgent:
    """A_ext agent (Agno): passage → ExtractionResult → candidate Facts."""

    def __init__(self, model: ModelLike, temperature: float = 0.0) -> None:
        self.agent = Agent(
            name="A_ext",
            model=model,
            description="MemGraphRAG fact extractor (A_ext)",
            instructions=[EXTRACTOR_INSTRUCTIONS],
            output_schema=ExtractionResult,
            use_json_mode=True,
            markdown=False,
            tool_call_limit=0,
        )
        self.temperature = temperature

    async def extract(self, passage: Passage) -> ExtractionResult:
        prompt = (
            f"Passage ID: {passage.id}\n"
            f"Text:\n{passage.text}\n\n"
            "Extract all relevant factual triples."
        )
        response = await self.agent.arun(prompt)
        return parse_structured_content(response.content, ExtractionResult)

    def triples_to_facts(
        self, triples: list[ExtractedTriple], passage_id: str
    ) -> list[Fact]:
        facts: list[Fact] = []
        for t in triples:
            facts.append(
                Fact(
                    head_entity=t.head_entity.strip(),
                    head_type=t.head_type.strip(),
                    relation=t.relation.strip().upper().replace(" ", "_"),
                    tail_entity=t.tail_entity.strip(),
                    tail_type=t.tail_type.strip(),
                    passage_ids=[passage_id],
                    source_passage_id=passage_id,
                    confidence=t.confidence,
                    category=t.category,
                    ttl_seconds=t.ttl_seconds,
                    is_invalidated=t.is_invalidated,
                    metadata={"evidence_span": t.evidence_span} if t.evidence_span else {},
                )
            )
        return facts

"""LLM-backed unstructured text extractor for ingestion."""

from __future__ import annotations

import re

from memgraphrag_core.ingestion.schemas import (
    DocumentExtractionResult,
    ExtractedFact,
    ExtractedSchema,
)
from memgraphrag_core.interfaces.llm_provider import BaseLLMProvider

_SYSTEM_PROMPT = """You are the MemGraphRAG ingestion extractor.
Extract ontological schema candidates and concrete facts from the user text.

Rules:
1. Relations MUST be UPPER_SNAKE_CASE (e.g. LOCATED_IN, MANUFACTURES, WORKS_AT).
2. Entity types MUST be PascalCase (e.g. Organization, City, Product, Person).
3. Do not invent facts unsupported by the text.
4. Prefer canonical entity surface forms without legal suffixes when possible.
5. Return both schemas (type-level) and facts (instance-level).
6. Infer types and relations from the input domain; do not assume a fixed ontology.
7. Set fact category to one of DECLARATIVE (default), HARD_CONSTRAINT (non-negotiable limit, e.g. budget cap), REJECTED_OPTION (explicitly discarded option, e.g. a Miami layover refused for lack of a transit visa), or TOOL_FAILURE (operational error as observation, e.g. HTTP 409 from a booking API). Keep rejections and failures as facts. Do not drop them.

Few-shot example (illustrative only; adapt types/relations to the user's text)
---
Input:
"Acme Corp is located in Berlin and manufactures the Widget Pro device."

Output JSON shape:
{
  "schemas": [
    {"head_type": "Organization", "relation": "LOCATED_IN", "tail_type": "City"},
    {"head_type": "Organization", "relation": "MANUFACTURES", "tail_type": "Product"}
  ],
  "facts": [
    {
      "head_entity": "Acme Corp",
      "head_type": "Organization",
      "relation": "LOCATED_IN",
      "tail_entity": "Berlin",
      "tail_type": "City"
    },
    {
      "head_entity": "Acme Corp",
      "head_type": "Organization",
      "relation": "MANUFACTURES",
      "tail_entity": "Widget Pro",
      "tail_type": "Product"
    }
  ]
}
---
"""


def to_upper_snake(value: str) -> str:
    """Normalize a relation label to UPPER_SNAKE_CASE."""
    s = value.strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def to_pascal_case(value: str) -> str:
    """Normalize an entity type label to PascalCase."""
    parts = re.split(r"[\s_\-]+", value.strip())
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


class LLMExtractor:
    """Extract schemas and facts from unstructured text via BaseLLMProvider."""

    def __init__(self, llm: BaseLLMProvider, temperature: float = 0.0) -> None:
        self.llm = llm
        self.temperature = temperature

    async def extract_from_text(self, text: str) -> DocumentExtractionResult:
        prompt = (
            "Extract schemas and facts from the following text.\n\n"
            f"TEXT:\n{text.strip()}"
        )
        raw = await self.llm.complete_structured(
            prompt=prompt,
            response_model=DocumentExtractionResult,
            system_prompt=_SYSTEM_PROMPT,
            temperature=self.temperature,
        )
        return self._normalize(raw)

    def _normalize(self, result: DocumentExtractionResult) -> DocumentExtractionResult:
        schemas = [
            ExtractedSchema(
                head_type=to_pascal_case(s.head_type),
                relation=to_upper_snake(s.relation),
                tail_type=to_pascal_case(s.tail_type),
            )
            for s in result.schemas
        ]
        facts = [
            ExtractedFact(
                head_entity=f.head_entity.strip(),
                head_type=to_pascal_case(f.head_type),
                relation=to_upper_snake(f.relation),
                tail_entity=f.tail_entity.strip(),
                tail_type=to_pascal_case(f.tail_type),
            )
            for f in result.facts
        ]
        # Ensure every fact contributes a schema candidate even if omitted by the LLM.
        seen = {(s.head_type, s.relation, s.tail_type) for s in schemas}
        for f in facts:
            key = (f.head_type, f.relation, f.tail_type)
            if key not in seen:
                schemas.append(
                    ExtractedSchema(
                        head_type=f.head_type,
                        relation=f.relation,
                        tail_type=f.tail_type,
                    )
                )
                seen.add(key)
        return DocumentExtractionResult(schemas=schemas, facts=facts)

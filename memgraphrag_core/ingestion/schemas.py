"""Structured extraction schemas for the ingestion LLM pipeline."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from memgraphrag_core.domain.models import FactCategory, coerce_fact_category


class ExtractedSchema(BaseModel):
    """Ontological schema candidate extracted from text."""

    head_type: str = Field(..., description="Head entity type in PascalCase (e.g. Organization)")
    relation: str = Field(..., description="Relation in UPPER_SNAKE_CASE (e.g. LOCATED_IN)")
    tail_type: str = Field(..., description="Tail entity type in PascalCase (e.g. City)")


class ExtractedFact(BaseModel):
    """Concrete fact candidate extracted from text."""

    head_entity: str
    head_type: str
    relation: str
    tail_entity: str
    tail_type: str
    category: FactCategory = FactCategory.DECLARATIVE
    ttl_seconds: Optional[int] = Field(default=None, ge=0)
    is_invalidated: bool = False

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value: object) -> object:
        return coerce_fact_category(value)


class DocumentExtractionResult(BaseModel):
    """Structured LLM extraction payload for ingestion (schemas + facts)."""

    schemas: List[ExtractedSchema] = Field(default_factory=list)
    facts: List[ExtractedFact] = Field(default_factory=list)


# Temporary alias for migration; prefer DocumentExtractionResult.
ExtractionResult = DocumentExtractionResult

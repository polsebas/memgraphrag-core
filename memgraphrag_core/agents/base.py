"""Shared helpers for Agno-backed MemGraphRAG agents."""

from __future__ import annotations

from typing import Type, TypeVar, Union

from agno.models.base import Model
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

ModelLike = Union[Model, str]


def parse_structured_content(content: object, schema: Type[T]) -> T:
    """Validate Agno run content into the expected Pydantic schema."""
    if isinstance(content, schema):
        return content
    if isinstance(content, BaseModel):
        return schema.model_validate(content.model_dump())
    if isinstance(content, dict):
        return schema.model_validate(content)
    if isinstance(content, str):
        return schema.model_validate_json(content)
    raise TypeError(
        f"Expected {schema.__name__} from agent run, got {type(content).__name__}: {content!r}"
    )

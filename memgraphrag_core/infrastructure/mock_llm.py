"""Mock LLM provider for ingestion tests (implements BaseLLMProvider)."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from memgraphrag_core.interfaces.llm_provider import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)

Responder = Callable[[str, Optional[str], Type[BaseModel]], Union[BaseModel, dict, str]]


class MockLLMProvider(BaseLLMProvider):
    """Scripted BaseLLMProvider for offline ingestion tests."""

    def __init__(
        self,
        structured_responses: Optional[Dict[str, Union[BaseModel, dict, str, Responder]]] = None,
        default_text: str = "",
    ) -> None:
        self._structured = structured_responses or {}
        self._default_text = default_text
        self.calls: List[dict] = []

    def register(
        self, model_name: str, response: Union[BaseModel, dict, str, Responder]
    ) -> None:
        self._structured[model_name] = response

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {"method": "complete", "prompt": prompt, "system_prompt": system_prompt}
        )
        return self._default_text

    async def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> T:
        self.calls.append(
            {
                "method": "complete_structured",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response_model": response_model.__name__,
            }
        )
        key = response_model.__name__
        if key not in self._structured:
            raise KeyError(
                f"MockLLMProvider has no response for {key}. "
                f"Registered: {list(self._structured)}"
            )
        payload = self._structured[key]
        if callable(payload) and not isinstance(payload, type):
            result = payload(prompt, system_prompt, response_model)
            if isinstance(result, response_model):
                return result
            if isinstance(result, BaseModel):
                return response_model.model_validate(result.model_dump())
            if isinstance(result, dict):
                return response_model.model_validate(result)
            if isinstance(result, str):
                return response_model.model_validate(json.loads(result))
            raise TypeError(f"Unsupported responder return type: {type(result)}")

        if isinstance(payload, response_model):
            return payload
        if isinstance(payload, BaseModel):
            return response_model.model_validate(payload.model_dump())
        if isinstance(payload, dict):
            return response_model.model_validate(payload)
        if isinstance(payload, str):
            return response_model.model_validate(json.loads(payload))
        raise TypeError(f"Unsupported mock payload type: {type(payload)}")

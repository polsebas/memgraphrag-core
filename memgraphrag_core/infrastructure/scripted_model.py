"""Scripted Agno Model for tests and demos without an external LLM API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Type, Union

from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse
from pydantic import BaseModel

Responder = Callable[[List[Message], Optional[Any]], Union[BaseModel, dict, str]]


@dataclass
class ScriptedModel(Model):
    """Returns predefined JSON/Pydantic payloads keyed by response model name.

    Used with ``Agent(..., output_schema=..., use_json_mode=True)`` so Agno
    parses the JSON content into the target schema.
    """

    id: str = "scripted"
    name: str = "Scripted"
    provider: str = "scripted"
    supports_native_structured_outputs: bool = False
    supports_json_schema_outputs: bool = False

    structured_responses: Dict[str, Union[BaseModel, dict, str, Responder]] = field(
        default_factory=dict
    )
    default_text: str = "{}"
    calls: List[dict] = field(default_factory=list)

    def register(
        self, model_name: str, response: Union[BaseModel, dict, str, Responder]
    ) -> None:
        self.structured_responses[model_name] = response

    def _messages_text(self, messages: Optional[List[Message]]) -> str:
        if not messages:
            return ""
        parts: List[str] = []
        for m in messages:
            parts.append(f"{m.role}: {m.content}")
        return "\n".join(parts)

    def _schema_name(self, response_format: Any) -> Optional[str]:
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format.__name__
        # JSON mode often passes {"type": "json_object"} — schema name comes from Agent
        return None

    def _resolve_payload(
        self, messages: Optional[List[Message]], response_format: Any, schema_hint: Optional[str]
    ) -> str:
        key = schema_hint or self._schema_name(response_format)
        if key is None:
            text = self._messages_text(messages).lower()
            if "you are a_ext" in text or "passage id:" in text:
                key = "ExtractionResult"
            elif "you are a_res" in text or "resolve these conflicts" in text:
                key = "ResolutionResult"
            elif "you are a_det" in text or "evaluate these fact pairs" in text:
                key = "DetectionResult"
            elif len(self.structured_responses) == 1:
                key = next(iter(self.structured_responses))

        if key is None or key not in self.structured_responses:
            raise KeyError(
                f"ScriptedModel has no response for key={key!r}. "
                f"Registered: {list(self.structured_responses)}"
            )

        payload = self.structured_responses[key]
        if callable(payload) and not isinstance(payload, type):
            result = payload(messages or [], response_format)
        else:
            result = payload

        if isinstance(result, BaseModel):
            return result.model_dump_json()
        if isinstance(result, dict):
            return json.dumps(result)
        if isinstance(result, str):
            return result
        raise TypeError(f"Unsupported scripted payload type: {type(result)}")

    def _build_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        messages: Optional[List[Message]] = kwargs.get("messages")
        if messages is None and args:
            maybe = args[0]
            if isinstance(maybe, list):
                messages = maybe
        response_format = kwargs.get("response_format")
        schema_hint = kwargs.get("schema_hint")  # optional explicit hint

        self.calls.append(
            {
                "messages": self._messages_text(messages),
                "response_format": str(response_format),
            }
        )
        content = self._resolve_payload(messages, response_format, schema_hint)
        return ModelResponse(content=content, role="assistant")

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._build_response(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._build_response(*args, **kwargs)

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield self._build_response(*args, **kwargs)

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ModelResponse]:
        yield self._build_response(*args, **kwargs)

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        if isinstance(response, ModelResponse):
            return response
        return ModelResponse(content=str(response), role="assistant")

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return self._parse_provider_response(response)

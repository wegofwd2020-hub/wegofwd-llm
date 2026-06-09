"""
wegofwd_llm/anthropic_native.py

Contract-native Anthropic provider. Unlike anthropic_adapter.py (which wraps the
legacy prompt-only AnthropicProvider for parity), this implements the new
`Provider` directly and uses Anthropic's strongest structured-output path —
**tool-use** — when JSON is requested. Forcing a single tool with the caller's
schema as the tool's input_schema is far more reliable than asking for JSON in
prose, which is the whole point of the conformance work (memo §5).

Additive: does not touch anthropic.py. The SDK client may be injected (tests pass
a fake; default constructs `anthropic.Anthropic`). Key discipline: the api_key is
never placed in an exception message, and SDK exceptions (which can stringify the
key) are never chained.
"""

from __future__ import annotations

import json

from wegofwd_llm.contract import Capabilities, LLMRequest, LLMResponse, Provider
from wegofwd_llm.errors import LLMConfigurationError, LLMError, LLMResponseError

# Name of the synthetic tool we force the model to call to emit structured JSON.
_EMIT_TOOL = "emit_result"


class AnthropicNativeProvider(Provider):
    provider_id = "anthropic"
    # JSON is delivered via tool-use, so json_object/json_schema are effectively
    # supported even though Anthropic has no OpenAI-style response_format.
    capabilities = Capabilities(json_object=True, json_schema=True, tools=True, max_context=200_000)

    def __init__(self, *, api_key: str, model: str = "claude-sonnet-4-6", client=None) -> None:
        if not api_key:
            raise LLMConfigurationError("anthropic provider requires a non-empty api_key (BYOK)")
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError:
                raise LLMConfigurationError("anthropic SDK not installed") from None
            self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def generate(self, req: LLMRequest) -> LLMResponse:
        want_json = req.response_format in ("json", "json_schema")
        kwargs: dict = {
            "model": self._model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": [{"role": "user", "content": req.prompt}],
        }
        if req.system:
            kwargs["system"] = req.system
        if want_json:
            schema = (
                req.json_schema
                if req.response_format == "json_schema" and req.json_schema
                else {"type": "object"}
            )
            kwargs["tools"] = [
                {
                    "name": _EMIT_TOOL,
                    "description": "Return the result as a single structured JSON object.",
                    "input_schema": schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": _EMIT_TOOL}

        try:
            message = self._client.messages.create(**kwargs)
        except Exception:
            # Never chain — SDK exceptions may include the api_key in their repr.
            raise LLMError("anthropic call failed") from None

        text = self._extract(message, want_json)
        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text,
            provider_id="anthropic",
            model=self._model,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            tokens_estimated=usage is None,
            raw=None,
        )

    @staticmethod
    def _extract(message, want_json: bool) -> str:
        content = getattr(message, "content", None) or []
        if want_json:
            # Preferred: the forced tool_use block — its `input` IS the JSON object.
            for block in content:
                if getattr(block, "type", None) == "tool_use":
                    return json.dumps(getattr(block, "input", {}) or {})
            # Fall through: model answered in text instead — let conformance validate it.
        for block in content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "") or ""
                if text:
                    return text
        raise LLMResponseError("anthropic returned no usable content")

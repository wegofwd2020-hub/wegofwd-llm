"""
wegofwd_llm/openai_compatible.py

One client for every provider that speaks the OpenAI Chat Completions protocol
(OpenAI, DeepSeek, Qwen, Gemma). Talks raw HTTP via httpx (sync, matching the
existing provider style) — no `openai` SDK dependency.

Key handling (BYOK): the key is sent as a Bearer header per request and is held
only on the instance for the call's lifetime. It is NEVER placed in an exception
message or the LLMResponse.raw payload. HTTP/transport errors are mapped to the
typed LLMError hierarchy with `from None`, so no underlying exception that might
echo the request can chain upward.
"""

from __future__ import annotations

import httpx

from wegofwd_llm.contract import Capabilities, LLMRequest, LLMResponse, Provider
from wegofwd_llm.errors import (
    LLMAuthError,
    LLMConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        provider_id: str = "openai",
        capabilities: Capabilities | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError(
                f"{provider_id} provider requires a non-empty api_key (BYOK)"
            )
        if not base_url:
            raise LLMConfigurationError(f"{provider_id} provider requires a base_url")
        self.provider_id = provider_id
        self.capabilities = capabilities or Capabilities(json_object=True)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        # An injected client (e.g. httpx.MockTransport in tests) is used as-is.
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def model(self) -> str:
        return self._model

    def _messages(self, req: LLMRequest) -> list[dict]:
        msgs: list[dict] = []
        if req.system and self.capabilities.system_prompt:
            msgs.append({"role": "system", "content": req.system})
        msgs.append({"role": "user", "content": req.prompt})
        return msgs

    def _response_format(self, req: LLMRequest) -> dict | None:
        if (
            req.response_format == "json_schema"
            and self.capabilities.json_schema
            and req.json_schema
        ):
            return {"type": "json_schema", "json_schema": req.json_schema}
        if req.response_format in ("json", "json_schema") and self.capabilities.json_object:
            return {"type": "json_object"}
        return None  # capability-gated: fall back to prompt-only JSON

    def generate(self, req: LLMRequest) -> LLMResponse:
        # Clamp to the provider/model's known output ceiling. Our default request
        # budget (16384) exceeds free-tier per-request limits, which reject the
        # call outright (Groq → HTTP 413) rather than truncating. 0 = uncapped.
        cap = self.capabilities.max_output_tokens
        max_tokens = min(req.max_tokens, cap) if cap > 0 else req.max_tokens
        body: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": req.temperature,
            "messages": self._messages(req),
        }
        rf = self._response_format(req)
        if rf is not None:
            body["response_format"] = rf

        try:
            resp = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        except httpx.TimeoutException:
            raise LLMTimeoutError(f"{self.provider_id} request timed out") from None
        except httpx.HTTPError:
            # Never chain — httpx error reprs can include request details.
            raise LLMError(f"{self.provider_id} transport error") from None

        if resp.status_code in (401, 403):
            raise LLMAuthError(f"{self.provider_id} authentication failed")
        if resp.status_code == 429:
            raise LLMRateLimitError(f"{self.provider_id} rate limited")
        if resp.status_code >= 400:
            raise LLMResponseError(f"{self.provider_id} returned HTTP {resp.status_code}")

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMResponseError(f"{self.provider_id} returned a malformed payload") from None
        if not text.strip():
            raise LLMResponseError(f"{self.provider_id} returned an empty response")

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider_id=self.provider_id,
            model=self._model,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            tokens_estimated=not bool(usage),
            raw=data,
        )

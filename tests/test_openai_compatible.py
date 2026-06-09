"""OpenAI-compatible provider — request shaping, response parsing, error mapping,
and the non-negotiable: the api_key never appears in an exception message.

No live calls: an httpx.MockTransport supplies canned responses (CLAUDE.md)."""

from __future__ import annotations

import json

import httpx
import pytest

from wegofwd_llm.contract import Capabilities, LLMRequest
from wegofwd_llm.errors import (
    LLMAuthError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
)
from wegofwd_llm.openai_compatible import OpenAICompatibleProvider

KEY = "sk-secret-do-not-leak-123"


def make_provider(handler, caps=None):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider(
        api_key=KEY,
        base_url="https://api.example.com/v1",
        model="test-model",
        provider_id="openai",
        capabilities=caps or Capabilities(json_object=True),
        client=client,
    )


def ok_response(content="hello", usage=None):
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def test_happy_path_parses_text_and_usage():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return ok_response("the answer", usage={"prompt_tokens": 11, "completion_tokens": 22})

    p = make_provider(handler)
    resp = p.generate(LLMRequest(prompt="q", max_tokens=100, temperature=0.0))

    assert resp.text == "the answer"
    assert resp.provider_id == "openai" and resp.model == "test-model"
    assert resp.input_tokens == 11 and resp.output_tokens == 22
    assert resp.tokens_estimated is False
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["auth"] == f"Bearer {KEY}"
    assert captured["body"]["messages"] == [{"role": "user", "content": "q"}]
    assert captured["body"]["max_tokens"] == 100


def test_max_tokens_clamped_to_capability_ceiling():
    # A free tier rejects an over-budget request (Groq → HTTP 413), so the
    # provider clamps req.max_tokens down to the capability's output ceiling.
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return ok_response("{}")

    p = make_provider(handler, caps=Capabilities(json_object=True, max_output_tokens=8000))
    p.generate(LLMRequest(prompt="q", max_tokens=16384))
    assert captured["body"]["max_tokens"] == 8000


def test_max_tokens_not_raised_when_request_below_ceiling():
    # Clamp is a min(), never a floor — a small request is sent unchanged.
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return ok_response("{}")

    p = make_provider(handler, caps=Capabilities(json_object=True, max_output_tokens=8000))
    p.generate(LLMRequest(prompt="q", max_tokens=2000))
    assert captured["body"]["max_tokens"] == 2000


def test_max_tokens_uncapped_when_capability_zero():
    # 0 = unknown/uncapped (e.g. OpenAI/Anthropic): pass the request budget through.
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return ok_response("{}")

    p = make_provider(
        handler, caps=Capabilities(json_object=True)
    )  # max_output_tokens defaults to 0
    p.generate(LLMRequest(prompt="q", max_tokens=16384))
    assert captured["body"]["max_tokens"] == 16384


def test_system_prompt_and_json_response_format_when_capable():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return ok_response("{}", usage={"prompt_tokens": 1, "completion_tokens": 1})

    p = make_provider(handler, caps=Capabilities(json_object=True, system_prompt=True))
    p.generate(LLMRequest(prompt="q", system="be precise", response_format="json"))

    assert captured["body"]["messages"][0] == {"role": "system", "content": "be precise"}
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_json_request_format_omitted_when_not_capable():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return ok_response("{}")

    p = make_provider(handler, caps=Capabilities(json_object=False))
    p.generate(LLMRequest(prompt="q", response_format="json"))

    assert "response_format" not in captured["body"]  # capability-gated


def test_missing_usage_flags_estimated():
    p = make_provider(lambda r: ok_response("x"))  # no usage block
    resp = p.generate(LLMRequest(prompt="q"))
    assert resp.tokens_estimated is True
    assert resp.input_tokens == 0 and resp.output_tokens == 0


@pytest.mark.parametrize(
    "status,exc",
    [(401, LLMAuthError), (403, LLMAuthError), (429, LLMRateLimitError), (500, LLMResponseError)],
)
def test_http_status_maps_to_typed_error(status, exc):
    p = make_provider(lambda r: httpx.Response(status, json={"error": "x"}))
    with pytest.raises(exc):
        p.generate(LLMRequest(prompt="q"))


def test_malformed_and_empty_payloads_raise_response_error():
    bad = make_provider(lambda r: httpx.Response(200, json={"nope": True}))
    with pytest.raises(LLMResponseError):
        bad.generate(LLMRequest(prompt="q"))

    empty = make_provider(lambda r: ok_response("   "))
    with pytest.raises(LLMResponseError):
        empty.generate(LLMRequest(prompt="q"))


def test_api_key_never_in_exception_messages():
    # Trigger several failure modes and assert the key is absent from each message.
    cases = [
        make_provider(lambda r: httpx.Response(401)),
        make_provider(lambda r: httpx.Response(429)),
        make_provider(lambda r: httpx.Response(500)),
        make_provider(lambda r: httpx.Response(200, json={"bad": 1})),
    ]
    for p in cases:
        try:
            p.generate(LLMRequest(prompt="q"))
        except Exception as exc:
            assert KEY not in str(exc)
        else:
            pytest.fail("expected an error")


def test_empty_key_and_base_url_rejected():
    with pytest.raises(LLMConfigurationError):
        OpenAICompatibleProvider(api_key="", base_url="https://x/v1", model="m")
    with pytest.raises(LLMConfigurationError):
        OpenAICompatibleProvider(api_key=KEY, base_url="", model="m")

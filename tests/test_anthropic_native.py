"""Contract-native Anthropic provider: plain-text path, tool-use JSON path,
text fallback, usage, error mapping, and api-key-never-leaked. A fake SDK client
stands in for `anthropic.Anthropic` — no network, no real SDK behavior."""

from __future__ import annotations

import json

import anthropic
import httpx
import pytest

from wegofwd_llm.anthropic_native import AnthropicNativeProvider, _map_sdk_error
from wegofwd_llm.contract import LLMRequest
from wegofwd_llm.errors import (
    LLMAuthError,
    LLMConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

KEY = "sk-ant-secret-do-not-leak"


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Usage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _Message:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage


class _Messages:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        self._client.calls.append(kwargs)
        if self._client.raise_on_create:
            raise RuntimeError(f"boom with {KEY} in it")  # simulate a key-leaking SDK error
        return self._client.next_message


class FakeAnthropic:
    def __init__(self, message=None, raise_on_create=False):
        self.calls = []
        self.next_message = message
        self.raise_on_create = raise_on_create

    @property
    def messages(self):
        return _Messages(self)


def provider(message=None, raise_on_create=False):
    client = FakeAnthropic(message=message, raise_on_create=raise_on_create)
    p = AnthropicNativeProvider(api_key=KEY, client=client)
    return p, client


def test_plain_text_path_sends_no_tools():
    msg = _Message([_Block(type="text", text="hello world")], _Usage(3, 9))
    p, client = provider(msg)

    resp = p.generate(LLMRequest(prompt="say hi", temperature=0.0))

    assert resp.text == "hello world"
    assert resp.input_tokens == 3 and resp.output_tokens == 9
    call = client.calls[0]
    assert "tools" not in call and "tool_choice" not in call
    assert call["messages"] == [{"role": "user", "content": "say hi"}]


def test_tool_use_json_path_forces_tool_with_schema():
    schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    msg = _Message(
        [_Block(type="tool_use", name="emit_result", input={"title": "Chapter 1"})], _Usage(5, 5)
    )
    p, client = provider(msg)

    resp = p.generate(
        LLMRequest(prompt="make a chapter", response_format="json_schema", json_schema=schema)
    )

    # The tool_use input is serialized to JSON text for the uniform text contract.
    assert json.loads(resp.text) == {"title": "Chapter 1"}
    call = client.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "emit_result"}
    assert call["tools"][0]["name"] == "emit_result"
    assert call["tools"][0]["input_schema"] == schema


def test_json_request_without_schema_uses_open_object_schema():
    msg = _Message([_Block(type="tool_use", input={"k": 1})], _Usage(1, 1))
    p, client = provider(msg)
    p.generate(LLMRequest(prompt="x", response_format="json"))
    assert client.calls[0]["tools"][0]["input_schema"] == {"type": "object"}


def test_json_requested_but_model_returns_text_falls_back():
    msg = _Message([_Block(type="text", text='{"title": "from text"}')], _Usage(1, 1))
    p, _ = provider(msg)
    resp = p.generate(LLMRequest(prompt="x", response_format="json"))
    assert json.loads(resp.text) == {"title": "from text"}


def test_no_usable_content_raises():
    p, _ = provider(_Message([], None))
    with pytest.raises(LLMResponseError):
        p.generate(LLMRequest(prompt="x"))


def test_system_prompt_passed_through():
    msg = _Message([_Block(type="text", text="ok")], _Usage(1, 1))
    p, client = provider(msg)
    p.generate(LLMRequest(prompt="q", system="be terse"))
    assert client.calls[0]["system"] == "be terse"


def test_sdk_error_is_remapped_without_leaking_key():
    p, _ = provider(raise_on_create=True)
    with pytest.raises(LLMError) as ei:
        p.generate(LLMRequest(prompt="x"))
    assert KEY not in str(ei.value)  # the RuntimeError embedded the key; must not surface


def test_empty_key_rejected():
    with pytest.raises(LLMConfigurationError):
        AnthropicNativeProvider(api_key="", client=FakeAnthropic())


def _status_error(cls, status):
    """A real Anthropic SDK status error whose message embeds the key, to prove
    we never surface it."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return cls(f"boom {KEY}", response=response, body=None)


# (SDK exception, expected seam error type) — auth/permission and unknown 4xx are
# non-retryable; rate-limit, timeout, connection and 5xx are transient.
_MAPPING_CASES = [
    (_status_error(anthropic.AuthenticationError, 401), LLMAuthError),
    (_status_error(anthropic.PermissionDeniedError, 403), LLMAuthError),
    (_status_error(anthropic.RateLimitError, 429), LLMRateLimitError),
    (_status_error(anthropic.InternalServerError, 500), LLMResponseError),
    (anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")), LLMTimeoutError),
    (
        anthropic.APIConnectionError(message="c", request=httpx.Request("POST", "https://x")),
        LLMTimeoutError,
    ),
    (ValueError("something unrecognised"), LLMError),
]


@pytest.mark.parametrize("exc, expected", _MAPPING_CASES)
def test_sdk_errors_map_to_typed_seam_errors(exc, expected):
    mapped = _map_sdk_error(exc)
    assert type(mapped) is expected
    assert KEY not in str(mapped)  # static, key-free message only


def test_auth_error_propagates_through_generate_without_leaking_key():
    """A 401 must surface as LLMAuthError (so callers fail fast instead of
    retrying) and must not stringify the key."""

    class _Raises:
        calls = []  # noqa: RUF012 - test stub

        @property
        def messages(self):
            return self

        def create(self, **kwargs):
            raise _status_error(anthropic.AuthenticationError, 401)

    p = AnthropicNativeProvider(api_key=KEY, client=_Raises())
    with pytest.raises(LLMAuthError) as ei:
        p.generate(LLMRequest(prompt="x"))
    assert KEY not in str(ei.value)

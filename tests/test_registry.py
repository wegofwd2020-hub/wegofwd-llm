"""Registry: role resolution, the BYOK factory, and error cases. The factory
constructs providers without network I/O (Anthropic SDK client construction is
offline; the OpenAI-compatible provider takes an injected mock client)."""

from __future__ import annotations

import httpx
import pytest

from wegofwd_llm.anthropic_native import AnthropicNativeProvider
from wegofwd_llm.contract import LLMRequest
from wegofwd_llm.errors import LLMConfigurationError
from wegofwd_llm.openai_compatible import OpenAICompatibleProvider
from wegofwd_llm.registry import (
    PROVIDER_REGISTRY,
    available_providers,
    build_provider,
    resolve_role,
)


def test_registry_has_the_known_providers():
    assert set(available_providers()) == {
        "anthropic",
        "openai",
        "deepseek",
        "qwen",
        "gemma",
        "groq",
        "openrouter",
        "gemini",
    }


def test_anthropic_default_model_is_sonnet_not_opus():
    # ADR-005: Anthropic default stays claude-sonnet-4-6.
    assert PROVIDER_REGISTRY["anthropic"].default_model == "claude-sonnet-4-6"


def test_resolve_role_and_unknown_role():
    assert resolve_role("authoring") == ("anthropic", "claude-sonnet-4-6")
    with pytest.raises(LLMConfigurationError):
        resolve_role("nope")


def test_build_openai_compatible_with_injected_client():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
    )
    p = build_provider("deepseek", api_key="byok-key", http_client=client)
    assert isinstance(p, OpenAICompatibleProvider)
    assert p.provider_id == "deepseek"
    assert p.model == PROVIDER_REGISTRY["deepseek"].default_model
    # the injected transport is actually used
    assert p.generate(LLMRequest(prompt="q")).text == "ok"


def test_build_anthropic_returns_native_provider():
    p = build_provider("anthropic", api_key="sk-ant-fake")
    assert isinstance(p, AnthropicNativeProvider)
    assert p.provider_id == "anthropic"
    assert p.model == "claude-sonnet-4-6"


def test_model_override():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    p = build_provider("openai", api_key="k", model="custom-model", http_client=client)
    assert p.model == "custom-model"


def test_unknown_provider_rejected():
    with pytest.raises(LLMConfigurationError):
        build_provider("not-a-provider", api_key="k")

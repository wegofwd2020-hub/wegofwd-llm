"""Author include/exclude allow-list: filtering the available set, validating a
selection against it, factory enforcement, and the error-type split (unknown
provider -> LLMConfigurationError/422; excluded -> LLMNotAllowedError/403)."""

from __future__ import annotations

import httpx
import pytest

from wegofwd_llm.contract import LLMRequest
from wegofwd_llm.errors import (
    LLMConfigurationError,
    LLMError,
    LLMNotAllowedError,
)
from wegofwd_llm.openai_compatible import OpenAICompatibleProvider
from wegofwd_llm.registry import (
    available_providers,
    build_provider,
    validate_selection,
)

ALL = {"anthropic", "openai", "deepseek", "qwen", "gemma", "groq", "openrouter", "gemini", "zai"}


def _mock_client():
    return httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
    )


# ── available_providers(allowed=) ────────────────────────────────────────────
def test_available_no_restriction_returns_all():
    assert set(available_providers()) == ALL
    assert set(available_providers(None)) == ALL


def test_available_filters_to_allowed_preserving_registry_order():
    assert available_providers({"openai", "anthropic"}) == ["anthropic", "openai"]


def test_available_empty_allowed_yields_nothing():
    assert available_providers(set()) == []


def test_available_ignores_unknown_names_in_allowed():
    assert available_providers({"openai", "totally-made-up"}) == ["openai"]


# ── validate_selection(allowed=) ─────────────────────────────────────────────
def test_validate_allows_member_and_defaults_model():
    assert validate_selection("openai", allowed={"openai", "anthropic"}) == (
        "openai",
        "gpt-4o-mini",
    )


def test_validate_rejects_excluded_provider():
    with pytest.raises(LLMNotAllowedError):
        validate_selection("deepseek", allowed={"openai", "anthropic"})


def test_validate_none_allowed_means_unrestricted():
    assert validate_selection("gemma") == ("gemma", "gemma-2-27b-it")


def test_unknown_provider_beats_allowlist_check():
    # An unknown provider is a 422 regardless of the allow-list (even if listed).
    with pytest.raises(LLMConfigurationError):
        validate_selection("bogus", allowed={"bogus"})


def test_empty_allowlist_excludes_everything():
    with pytest.raises(LLMNotAllowedError):
        validate_selection("anthropic", allowed=set())


# ── build_provider(allowed=) ─────────────────────────────────────────────────
def test_factory_enforces_allowlist_before_constructing():
    with pytest.raises(LLMNotAllowedError):
        build_provider("openai", api_key="k", allowed={"anthropic"}, http_client=_mock_client())


def test_factory_builds_when_allowed():
    p = build_provider(
        "openai", api_key="k", allowed={"openai", "deepseek"}, http_client=_mock_client()
    )
    assert isinstance(p, OpenAICompatibleProvider)
    assert p.generate(LLMRequest(prompt="q")).text == "ok"


# ── error-type contract (so the backend can map 403 vs 422 distinctly) ───────
def test_not_allowed_is_distinct_from_configuration_error():
    assert issubclass(LLMNotAllowedError, LLMError)
    assert not issubclass(LLMNotAllowedError, LLMConfigurationError)
    assert not issubclass(LLMConfigurationError, LLMNotAllowedError)

"""Interface versioning + provenance: a contract version, a per-provider
integration version, and a stampable provenance record (the seam that makes
per-book model pinning + 'made with an outdated integration' detection possible)."""

from __future__ import annotations

import pytest

from wegofwd_llm.contract import LLM_CONTRACT_VERSION
from wegofwd_llm.errors import LLMConfigurationError
from wegofwd_llm.registry import (
    PROVIDER_REGISTRY,
    provenance,
    validate_selection,
)


def test_contract_version_is_a_positive_int():
    assert isinstance(LLM_CONTRACT_VERSION, int) and LLM_CONTRACT_VERSION >= 1


def test_every_provider_has_an_integration_version():
    for spec in PROVIDER_REGISTRY.values():
        assert isinstance(spec.integration_version, int) and spec.integration_version >= 1


def test_validate_selection_defaults_model_and_checks_provider():
    assert validate_selection("anthropic") == ("anthropic", "claude-sonnet-4-6")
    # an arbitrary model string is accepted (we hold no vendor catalogue)
    assert validate_selection("openai", "gpt-4o") == ("openai", "gpt-4o")
    with pytest.raises(LLMConfigurationError):
        validate_selection("nope")


def test_provenance_stamps_all_version_axes():
    prov = provenance("anthropic")
    assert prov == {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "model_verified": True,
        "integration_version": PROVIDER_REGISTRY["anthropic"].integration_version,
        "contract_version": LLM_CONTRACT_VERSION,
    }


def test_provenance_respects_explicit_model_and_flags_unverified():
    prov = provenance("deepseek", "deepseek-chat")
    assert prov["provider"] == "deepseek" and prov["model"] == "deepseek-chat"
    # registry deepseek model is UNVERIFIED — provenance surfaces that honestly
    assert prov["model_verified"] is False
    assert prov["contract_version"] == LLM_CONTRACT_VERSION


def test_provenance_unknown_provider_raises():
    with pytest.raises(LLMConfigurationError):
        provenance("not-a-provider")

"""Content Trust Manifest (ADR-015): the seam emits provenance+validation, the
product attaches the rest, and the serialised manifest carries no secret material
and distinguishes "not assessed" (block absent) from a false value."""

from __future__ import annotations

import dataclasses

import pytest

from wegofwd_llm import (
    TRUST_MANIFEST_VERSION,
    ComplianceBlock,
    IntegrityBlock,
    PolicyBlock,
    ReviewBlock,
    engine_trust,
)
from wegofwd_llm.errors import LLMConfigurationError


def test_engine_trust_stamps_version_and_reuses_provenance():
    m = engine_trust("anthropic", schema_validated=True, schema_id="lesson@1")
    assert m.trust_manifest_version == TRUST_MANIFEST_VERSION
    assert m.provenance.provider == "anthropic"
    assert m.provenance.model == "claude-sonnet-4-6"
    assert m.provenance.model_verified is True
    assert m.validation.schema_validated is True
    assert m.validation.repair_attempts == 0


def test_engine_trust_surfaces_unverified_model_honestly():
    m = engine_trust("deepseek", "deepseek-chat", schema_validated=True)
    assert m.provenance.model_verified is False  # registry says UNVERIFIED


def test_engine_trust_unknown_provider_raises():
    with pytest.raises(LLMConfigurationError):
        engine_trust("not-a-provider", schema_validated=True)


def test_to_public_dict_drops_unset_blocks():
    m = engine_trust("anthropic", schema_validated=True)
    d = m.to_public_dict()
    # seam-owned blocks always present; product blocks absent until attached
    assert "provenance" in d and "validation" in d
    for absent in ("compliance", "integrity", "sourcing", "review", "policy"):
        assert absent not in d


def test_product_attaches_blocks_via_replace():
    m = engine_trust("anthropic", schema_validated=True, repair_attempts=1)
    m = dataclasses.replace(
        m,
        compliance=ComplianceBlock(
            ruleset="mentible-professional@1.0",
            checks_passed=11,
            checks_total=13,
            status="pass_with_notes",
        ),
        integrity=IntegrityBlock(content_hash="sha256:" + "a" * 64, signed=True),
        review=ReviewBlock(human_approved=True, approver_distinct_from_generator=True),
        policy=PolicyBlock(byok=True, prompts_stored=False, key_stored=False),
    )
    d = m.to_public_dict()
    assert d["compliance"]["status"] == "pass_with_notes"
    assert d["validation"]["repair_attempts"] == 1
    assert d["review"]["approver_distinct_from_generator"] is True
    assert d["policy"]["key_stored"] is False


def test_manifest_carries_no_secret_material():
    """A manifest is shipped to a front-end. Its serialisation must contain only
    provenance/validation/compliance facts — never a key, prompt, or raw payload."""
    secret = "sk-ant-LEAKY-FAKE-KEY-FOR-TEST-12345"
    m = engine_trust("anthropic", schema_validated=True, generated_at="2026-06-14T00:00:00Z")
    blob = repr(m.to_public_dict())
    assert secret not in blob
    # there is no field that could even carry one
    assert "api_key" not in blob and "prompt" not in blob and "raw" not in blob

"""
wegofwd_llm/registry.py

Provider metadata + a BYOK factory. Logical *roles* (authoring / toc / fast-draft)
map to a (provider, model) pair so model ids live in one place with one update
policy — application code never hardcodes a model string.

⚠ Model ids marked UNVERIFIED below are placeholders from docs/llm-providers.md
and MUST be validated against each vendor before use (ADR-005 open question).
Capabilities are deliberately conservative; widen them only once confirmed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from wegofwd_llm.contract import LLM_CONTRACT_VERSION, Capabilities, Provider
from wegofwd_llm.errors import LLMConfigurationError, LLMNotAllowedError


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    openai_compatible: bool
    default_model: str
    capabilities: Capabilities
    base_url: str | None = None  # None for providers with their own SDK (Anthropic)
    managed_env_key: str = ""  # env var for the MANAGED key (unused on the BYOK path)
    model_verified: bool = False
    # Expected BYOK key prefix for client/server shape-checks. Not every vendor
    # uses "sk-" (Groq = gsk_, Gemini = AIza, …); "" means no prefix check (length
    # only). The vendor still rejects truly invalid keys.
    key_prefix: str = "sk-"
    # Version of OUR integration for this provider (request shaping, JSON mode,
    # prompt shims). Bump when we change HOW we call the vendor — independent of
    # the model id and the contract version. Recorded in provenance().
    integration_version: int = 1


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        provider_id="anthropic",
        openai_compatible=False,
        default_model="claude-sonnet-4-6",
        # JSON delivered via tool-use (see anthropic_native.py).
        capabilities=Capabilities(
            json_object=True, json_schema=True, tools=True, max_context=200_000
        ),
        managed_env_key="ANTHROPIC_API_KEY",
        model_verified=True,
        key_prefix="sk-ant-",
    ),
    "openai": ProviderSpec(
        provider_id="openai",
        openai_compatible=True,
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",  # UNVERIFIED
        capabilities=Capabilities(
            json_object=True, json_schema=True, tools=True, max_context=128_000
        ),
        managed_env_key="OPENAI_API_KEY",
        key_prefix="sk-",
    ),
    # ── Free OpenAI-compatible providers (BYOK; get a free key) ─────────────────
    # base_url + default_model verified against vendor docs 2026-06-05; re-check
    # periodically as vendors rotate free models.
    "groq": ProviderSpec(
        provider_id="groq",
        openai_compatible=True,
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",  # current Groq production model
        # Free tier enforces a per-request/TPM token limit (~12k); 16384 → HTTP 413.
        # Cap output so input+output stays under it. Verified 2026-06-07.
        capabilities=Capabilities(json_object=True, max_context=128_000, max_output_tokens=8000),
        managed_env_key="GROQ_API_KEY",
        model_verified=True,
        key_prefix="gsk_",  # Groq keys start with gsk_
    ),
    "openrouter": ProviderSpec(
        provider_id="openrouter",
        openai_compatible=True,
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.3-70b-instruct:free",  # a current :free model
        # :free model variants cap completion length; keep output modest.
        capabilities=Capabilities(json_object=True, max_context=128_000, max_output_tokens=8000),
        managed_env_key="OPENROUTER_API_KEY",
        model_verified=True,
        key_prefix="sk-or-",  # OpenRouter keys start with sk-or-
    ),
    "gemini": ProviderSpec(
        provider_id="gemini",
        openai_compatible=True,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        # gemini-2.5-flash — verified live 2026-06-09 (valid JSON, single attempt)
        # via this OpenAI-compat path. 2.0-flash hit free-tier quota (429); 1.5 retired.
        default_model="gemini-2.5-flash",
        # Conservative output cap (2.5-flash supports up to 65536) to stay clear of
        # free-tier per-request/TPM limits that reject over-budget requests.
        capabilities=Capabilities(json_object=True, max_context=1_000_000, max_output_tokens=8192),
        managed_env_key="GEMINI_API_KEY",
        model_verified=True,
        # No prefix check: AI Studio keys may be AIza… OR an AQ.-prefixed OAuth-style
        # token — both work against this endpoint (verified 2026-06-09).
        key_prefix="",
    ),
    "deepseek": ProviderSpec(
        provider_id="deepseek",
        openai_compatible=True,
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",  # UNVERIFIED
        capabilities=Capabilities(json_object=True, tools=True, max_context=64_000),
        managed_env_key="DEEPSEEK_API_KEY",
    ),
    "qwen": ProviderSpec(
        provider_id="qwen",
        openai_compatible=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # UNVERIFIED
        default_model="qwen-max",  # UNVERIFIED
        capabilities=Capabilities(json_object=True, max_context=32_000),
        managed_env_key="QWEN_API_KEY",
    ),
    "gemma": ProviderSpec(
        provider_id="gemma",
        openai_compatible=True,
        base_url="",  # UNVERIFIED — depends on hosting (e.g. OpenRouter / self-host)
        default_model="gemma-2-27b-it",  # UNVERIFIED
        capabilities=Capabilities(json_object=False, max_context=8_000),
        managed_env_key="GEMMA_API_KEY",
    ),
}

# Logical role → (provider_id, model). One place to bump versions / route by cost.
ROLE_DEFAULTS: dict[str, tuple[str, str]] = {
    "authoring": ("anthropic", "claude-sonnet-4-6"),  # long, schema-heavy lessons
    "toc": ("anthropic", "claude-sonnet-4-6"),  # structuring
    "fast-draft": ("openai", "gpt-4o-mini"),  # cheap preview tier (UNVERIFIED model)
}


def available_providers(allowed: Iterable[str] | None = None) -> list[str]:
    """Known providers, in registry order. If `allowed` is given (the author's
    include/exclude set), restrict to it — unknown names in `allowed` are
    ignored, and an empty set yields []. `None` means no restriction. This is
    what the future GET-available-LLMs endpoint hands the mobile picker."""
    ids = list(PROVIDER_REGISTRY)
    if allowed is None:
        return ids
    allowset = set(allowed)
    return [p for p in ids if p in allowset]


def validate_selection(
    provider_id: str, model: str | None = None, *, allowed: Iterable[str] | None = None
) -> tuple[str, str]:
    """Resolve + validate a caller's LLM choice (the seam the future request
    param + mobile selector call). Returns (provider_id, model).

    - Unknown provider -> LLMConfigurationError (maps to 422).
    - Known but outside `allowed` (the author's include/exclude set) ->
      LLMNotAllowedError (maps to 403). `allowed=None` means no restriction.
    The model string is accepted as-is (we hold no vendor catalogue) and defaults
    to the spec default. The unknown check precedes the allow-list check."""
    spec = PROVIDER_REGISTRY.get(provider_id)
    if spec is None:
        raise LLMConfigurationError(f"unknown provider {provider_id!r}")
    if allowed is not None and provider_id not in set(allowed):
        raise LLMNotAllowedError(f"provider {provider_id!r} is excluded by the author's allow-list")
    return provider_id, (model or spec.default_model)


def provenance(provider_id: str, model: str | None = None) -> dict:
    """A stampable record of WHICH LLM + versions produced a generation — meant
    to be stored on each generated unit and on a book's pinned params, so we can
    enforce per-book model pinning and detect content made with an outdated
    integration/model (and offer to regenerate). See multi-provider-directions §6."""
    pid, chosen_model = validate_selection(provider_id, model)
    spec = PROVIDER_REGISTRY[pid]
    return {
        "provider": pid,
        "model": chosen_model,
        "model_verified": spec.model_verified,
        "integration_version": spec.integration_version,
        "contract_version": LLM_CONTRACT_VERSION,
    }


def resolve_role(role: str) -> tuple[str, str]:
    """(provider_id, model) for a logical role."""
    try:
        return ROLE_DEFAULTS[role]
    except KeyError:
        raise LLMConfigurationError(f"unknown role {role!r}") from None


def build_provider(
    provider_id: str,
    *,
    api_key: str,
    model: str | None = None,
    http_client=None,
    allowed: Iterable[str] | None = None,
) -> Provider:
    """Construct a BYOK provider from the registry. `model` overrides the spec
    default. `allowed` (the author's include/exclude set) is enforced here too —
    raises LLMNotAllowedError before any provider is built. `http_client`
    (httpx.Client) is for OpenAI-compatible providers, chiefly to inject a
    MockTransport in tests."""
    provider_id, chosen_model = validate_selection(provider_id, model, allowed=allowed)
    spec = PROVIDER_REGISTRY[provider_id]

    if spec.openai_compatible:
        from wegofwd_llm.openai_compatible import OpenAICompatibleProvider

        kwargs = {
            "api_key": api_key,
            "base_url": spec.base_url or "",
            "model": chosen_model,
            "provider_id": spec.provider_id,
            "capabilities": spec.capabilities,
        }
        if http_client is not None:
            kwargs["client"] = http_client
        return OpenAICompatibleProvider(**kwargs)

    if spec.provider_id == "anthropic":
        # Native provider: uses tool-use for reliable JSON. The legacy-wrapping
        # AnthropicAdapter remains available for the eventual backend rewire.
        from wegofwd_llm.anthropic_native import AnthropicNativeProvider

        return AnthropicNativeProvider(api_key=api_key, model=chosen_model)

    raise LLMConfigurationError(f"no constructor wired for provider {provider_id!r}")

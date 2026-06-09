"""
wegofwd_llm/contract.py

The evolved, provider-agnostic LLM contract (ADR-005 + docs/multi-provider-directions.md).

This is the *new* seam: a typed request/response plus a capability descriptor,
so N providers can be driven through one interface and the managed path can
meter token usage. It is ADDITIVE — the legacy tuple-returning `LLMProvider`
in base.py is untouched until the backend rewire (deferred behind PR #44).

Mentible-owned, NOT vendored from OnDemand. Imports nothing from `backend/`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Version of the LLMRequest/LLMResponse/Provider/Capabilities contract defined in
# THIS module. Bump when the seam's shape changes in a way that stored data or
# callers must notice (e.g. a new required field, changed semantics). Stamped
# into provenance (see registry.provenance) so a generation records which seam
# produced it. Distinct from a provider's *integration_version* and the vendor's
# *model id* — see docs/multi-provider-directions.md §6.
LLM_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class Capabilities:
    """What a provider/model can do — drives how we ask for JSON (see conformance.py)."""

    json_object: bool = False  # supports response_format={"type":"json_object"}
    json_schema: bool = False  # supports strict structured outputs (json_schema)
    tools: bool = False  # supports tool / function calling (Anthropic's strong path)
    system_prompt: bool = True
    max_context: int = 0  # 0 = unknown
    # Per-request output-token ceiling. 0 = unknown/uncapped. Free tiers enforce a
    # tokens-per-request (or per-minute) limit far below our 16384 default and
    # reject an over-budget request outright (Groq's free tier → HTTP 413), so a
    # provider whose ceiling we know clamps req.max_tokens down to this. See
    # openai_compatible.generate / docs/multi-provider-wiring-phase5.md.
    max_output_tokens: int = 0
    vision: bool = False


@dataclass(frozen=True)
class LLMRequest:
    """One generation request, provider-independent.

    `response_format` is a *hint*; a provider only honours it if its
    Capabilities allow, otherwise it falls back to prompt-only JSON.
    """

    prompt: str
    max_tokens: int = 16384
    temperature: float = 0.2
    system: str | None = None
    response_format: str | None = None  # None | "json" | "json_schema"
    json_schema: dict | None = None  # required when response_format == "json_schema"


@dataclass(frozen=True)
class LLMResponse:
    """A generation result with normalized usage (the linchpin of metering).

    `tokens_estimated` flags that input/output counts are a fallback estimate
    rather than provider-reported (some SDKs omit usage).
    """

    text: str
    provider_id: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False
    raw: object | None = field(default=None, repr=False)  # provider payload, debug only


class Provider(ABC):
    """A provider under the new contract. One instance per request (BYOK) or
    pooled (managed). Implementations must NEVER let an API key reach an
    exception message, log line, or `raw` field."""

    provider_id: str = ""
    capabilities: Capabilities = Capabilities()

    @property
    @abstractmethod
    def model(self) -> str:
        """The concrete model id this instance calls."""

    @abstractmethod
    def generate(self, req: LLMRequest) -> LLMResponse:
        """Run one request. Raises a wegofwd_llm.errors.LLMError subclass
        on failure — never a raw SDK/HTTP exception that might stringify a key."""

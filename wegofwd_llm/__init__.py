"""wegofwd_llm — the shared multi-provider LLM seam.

A provider-agnostic seam: a typed request/response, a capability descriptor, a
provider registry with role-pinning + provenance, and a validate→repair JSON
conformance loop. One interface fronts N vendors (Anthropic via tool-use; OpenAI,
Groq, OpenRouter, Gemini, … via one OpenAI-compatible client).

Shared as a LIBRARY, not a service (ADR-012 D8): each consumer imports this and
runs it in-process, making its own vendor call with its own key. The package
NEVER sources keys (the caller passes the api_key string — D3) and NEVER lets a
key reach an exception, log line, `raw` field, or `repr`.

Consumers: Mentible (BYOK), StudyBuddy OnDemand (managed), Pramana (managed).
"""

from __future__ import annotations

from wegofwd_llm.conformance import ConformanceResult, Validator, generate_validated
from wegofwd_llm.contract import (
    LLM_CONTRACT_VERSION,
    Capabilities,
    LLMRequest,
    LLMResponse,
    Provider,
)
from wegofwd_llm.errors import (
    LLMAuthError,
    LLMConfigurationError,
    LLMError,
    LLMNotAllowedError,
    LLMRateLimitError,
    LLMResponseError,
    LLMSchemaError,
    LLMTimeoutError,
)
from wegofwd_llm.registry import (
    PROVIDER_REGISTRY,
    ROLE_DEFAULTS,
    ProviderSpec,
    available_providers,
    build_provider,
    provenance,
    resolve_role,
    validate_selection,
)

__version__ = "0.1.3"

__all__ = [
    "LLM_CONTRACT_VERSION",
    "PROVIDER_REGISTRY",
    "ROLE_DEFAULTS",
    "Capabilities",
    "ConformanceResult",
    "LLMAuthError",
    "LLMConfigurationError",
    "LLMError",
    "LLMNotAllowedError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMResponseError",
    "LLMSchemaError",
    "LLMTimeoutError",
    "Provider",
    "ProviderSpec",
    "Validator",
    "__version__",
    "available_providers",
    "build_provider",
    "generate_validated",
    "provenance",
    "resolve_role",
    "validate_selection",
]

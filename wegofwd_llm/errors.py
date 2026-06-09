"""
wegofwd_llm/errors.py

Typed error hierarchy for the multi-provider seam. Routing/failover and UX
messaging branch on error *type*, and mapping provider SDK/HTTP errors into
these keeps raw exceptions — which may stringify an API key — from leaking
upward. Implementations must raise these with KEY-FREE messages only.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base for all provider errors. Message must never contain key material."""


class LLMConfigurationError(LLMError):
    """Misconfiguration — missing/empty key, unknown provider, bad base_url.
    Maps to a 4xx 'bad request' at the API boundary (the provider isn't real)."""


class LLMNotAllowedError(LLMError):
    """The selected provider is a real, known provider but is excluded by the
    author's include/exclude allow-list. Distinct from LLMConfigurationError so
    the API can map it to 403 (forbidden by policy) vs 422 (unknown provider)."""


class LLMAuthError(LLMError):
    """Provider rejected the credentials (401/403)."""


class LLMRateLimitError(LLMError):
    """Provider rate-limited the request (429). Retryable / failover candidate."""


class LLMTimeoutError(LLMError):
    """The request timed out."""


class LLMResponseError(LLMError):
    """The provider returned an unusable response (empty, malformed, 5xx)."""


class LLMSchemaError(LLMResponseError):
    """Response was returned but failed our JSON/schema validation after the
    repair budget was exhausted (see conformance.py)."""

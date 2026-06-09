"""
wegofwd_llm/conformance.py

The make-or-break of multi-provider support: getting schema-valid JSON out of
models that vary in instruction-following. Instead of blind retries, this runs a
validate -> repair loop — on a validation failure it feeds the validator's error
back to the model and asks for a corrected JSON, up to a repair budget.

`validate` is any callable that parses the response text and RAISES on invalid
output (returning the parsed value on success) — so it stays decoupled from any
specific schema library (the caller can use jsonschema, pydantic, or a hand
check). Only the model's own text is echoed into the repair prompt — never keys.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from wegofwd_llm.contract import LLMRequest, LLMResponse, Provider
from wegofwd_llm.errors import LLMSchemaError

Validator = Callable[[str], Any]


@dataclass(frozen=True)
class ConformanceResult:
    parsed: Any
    response: LLMResponse  # the response that finally validated
    attempts: int  # total provider calls (1 = valid first try)
    repaired: bool  # True if it took >1 attempt
    total_input_tokens: int
    total_output_tokens: int


def _repair_prompt(original: str, error: str, bad_text: str, *, max_echo: int = 4000) -> str:
    return (
        f"{original}\n\n"
        f"--- Your previous response was INVALID ---\n"
        f"Validation error: {error}\n"
        f"Previous response (verbatim):\n{bad_text[:max_echo]}\n\n"
        "Return ONLY corrected, valid JSON that fixes the problem above. "
        "No prose, no code fences."
    )


def generate_validated(
    provider: Provider,
    req: LLMRequest,
    validate: Validator,
    *,
    max_repairs: int = 2,
) -> ConformanceResult:
    """Generate, validate, and repair up to `max_repairs` times.

    Returns a ConformanceResult on success. Raises LLMSchemaError if the
    response still fails validation after the budget is exhausted. Provider
    errors (auth, rate-limit, transport) propagate unchanged.
    """
    in_tokens = 0
    out_tokens = 0
    last_error: Exception | None = None
    current = req

    for attempt in range(1, max_repairs + 2):  # 1 initial + max_repairs
        resp = provider.generate(current)
        in_tokens += resp.input_tokens
        out_tokens += resp.output_tokens
        try:
            parsed = validate(resp.text)
        except Exception as exc:  # validator decides what "invalid" means
            last_error = exc
            current = replace(current, prompt=_repair_prompt(req.prompt, str(exc), resp.text))
            continue
        return ConformanceResult(
            parsed=parsed,
            response=resp,
            attempts=attempt,
            repaired=attempt > 1,
            total_input_tokens=in_tokens,
            total_output_tokens=out_tokens,
        )

    raise LLMSchemaError(
        f"{provider.provider_id} response failed validation after "
        f"{max_repairs + 1} attempts: {last_error}"
    )

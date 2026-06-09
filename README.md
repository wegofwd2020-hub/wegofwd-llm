# wegofwd-llm

The shared **multi-provider LLM seam** for the wegofwd product family. A
provider-agnostic interface so the model underneath is a *parameter* of a
generation, not a hardwired vendor.

> **Library, not a service.** Each consumer `pip install`s this and runs it
> **in-process**, making its own vendor call with its own key — there is no
> `wegofwd-llm` server. See **ADR-012** (StudyBuddy_SelfLearner) for the full
> rationale. Extracted from `StudyBuddy_SelfLearner/pipeline/providers/` at
> `main@2649101`.

## What it is

| Module | Role |
|---|---|
| `contract.py` | `LLMRequest` / `LLMResponse` / `Capabilities` / `Provider` ABC + `LLM_CONTRACT_VERSION` |
| `errors.py` | Typed error hierarchy (`LLMError` + subclasses) — SDK errors are remapped to these, key-free |
| `registry.py` | `ProviderSpec` catalogue, logical-role pinning, `build_provider`, `validate_selection`, `provenance` |
| `conformance.py` | `generate_validated` — a validate→repair loop (schema-agnostic) |
| `anthropic_native.py` | Anthropic via **tool-use** (reliable JSON) |
| `openai_compatible.py` | One client for OpenAI / Groq / OpenRouter / Gemini / … |

## Two rules that make it reusable

1. **The package never sources keys.** The caller always passes the `api_key`
   string (`build_provider(pid, api_key=…)`). BYOK vs managed is entirely a
   caller concern — the package never reads env or a vault.
2. **No key ever leaks.** A key must never reach an exception message, a log
   line, a `raw` field, or a `repr`. Providers remap SDK/HTTP errors to the typed
   hierarchy with key-free messages.

It is also **schema-agnostic**: `generate_validated` takes a caller-supplied
`validate(text) -> parsed` callable that raises on invalid output, so it carries
no product schema (lessons / curriculum units / compliance docs all reuse it).

## Install

```bash
# Local multi-repo dev (editable):
pip install -e ../wegofwd-llm

# Pinned (CI / Docker), with the Anthropic SDK extra:
pip install "wegofwd-llm[anthropic] @ git+https://github.com/wegofwd2020-hub/wegofwd-llm@v0.1.0"
```

`anthropic` is an **optional extra** — OpenAI-compatible-only consumers can skip
it; `anthropic_native.py` imports it lazily and raises `LLMConfigurationError`
if it's missing.

## Use

```python
from wegofwd_llm import build_provider, generate_validated, LLMRequest, provenance

provider = build_provider("anthropic", api_key=key, model="claude-sonnet-4-6")
result = generate_validated(
    provider,
    LLMRequest(prompt=prompt, max_tokens=8000, response_format="json"),
    validate=my_schema_check,   # raises on invalid; returns parsed on success
)
stamp = provenance("anthropic", "claude-sonnet-4-6")  # ride this on the output
```

## Versioning

Three independent axes (ADR-012 D4):

- **package semver** (`__version__`) — the whole;
- **`LLM_CONTRACT_VERSION`** — the request/response shape;
- per-provider **`integration_version`** — how a given vendor is called.

Additive change → minor; a breaking contract change → major + a
`LLM_CONTRACT_VERSION` bump. Consumers pin a version and upgrade deliberately.

## Test

```bash
pip install -e ".[dev]"
pytest          # no live APIs — providers are mocked / replayed
```

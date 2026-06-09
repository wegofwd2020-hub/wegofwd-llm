"""The validate -> repair loop: valid-first-try, repair-then-succeed, budget
exhaustion, and token accumulation across attempts. Uses a scripted fake
provider (no network)."""

from __future__ import annotations

import json

import pytest

from wegofwd_llm.conformance import generate_validated
from wegofwd_llm.contract import LLMRequest, LLMResponse, Provider
from wegofwd_llm.errors import LLMSchemaError


class FakeProvider(Provider):
    provider_id = "fake"

    def __init__(self, texts):
        self._texts = list(texts)
        self.seen_prompts = []

    @property
    def model(self):
        return "fake-1"

    def generate(self, req: LLMRequest) -> LLMResponse:
        self.seen_prompts.append(req.prompt)
        text = self._texts.pop(0)
        return LLMResponse(
            text=text, provider_id="fake", model="fake-1", input_tokens=5, output_tokens=7
        )


def validate_titled_json(text):
    obj = json.loads(text)  # raises on bad JSON
    if "title" not in obj:
        raise ValueError("missing required field 'title'")
    return obj


def test_valid_first_try():
    p = FakeProvider(['{"title": "ok"}'])
    res = generate_validated(p, LLMRequest(prompt="write"), validate_titled_json)
    assert res.parsed == {"title": "ok"}
    assert res.attempts == 1 and res.repaired is False
    assert res.total_input_tokens == 5 and res.total_output_tokens == 7
    assert len(p.seen_prompts) == 1


def test_repairs_then_succeeds_and_feeds_error_back():
    p = FakeProvider(["not json at all", '{"title": "fixed"}'])
    res = generate_validated(p, LLMRequest(prompt="write a lesson"), validate_titled_json)
    assert res.parsed == {"title": "fixed"}
    assert res.attempts == 2 and res.repaired is True
    # token usage accumulates across both calls
    assert res.total_input_tokens == 10 and res.total_output_tokens == 14
    # the repair prompt keeps the original instruction AND surfaces the error + bad text
    repair = p.seen_prompts[1]
    assert "write a lesson" in repair
    assert "INVALID" in repair
    assert "not json at all" in repair


def test_schema_failure_after_budget_raises():
    p = FakeProvider(["bad", "still bad", "nope"])  # 1 initial + 2 repairs all fail
    with pytest.raises(LLMSchemaError) as ei:
        generate_validated(p, LLMRequest(prompt="x"), validate_titled_json, max_repairs=2)
    assert "fake" in str(ei.value)
    assert len(p.seen_prompts) == 3  # exactly the budget


def test_missing_required_field_triggers_repair():
    p = FakeProvider(['{"body": "no title here"}', '{"title": "now present"}'])
    res = generate_validated(p, LLMRequest(prompt="x"), validate_titled_json)
    assert res.repaired is True
    assert "missing required field 'title'" in p.seen_prompts[1]

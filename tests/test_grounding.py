import json
from types import SimpleNamespace

import pytest

from app.agent import graph


class _FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        return SimpleNamespace(content=next(self.responses))


def _state():
    return {
        "question": "What is the warranty period?",
        "classification": "NARRATIVE",
        "messages": [],
        "tool_calls_made": ["retrieve_narrative"],
        "evidence": [
            {
                "citation": "ACME · FY2024 Annual · p.4",
                "content": "The product warranty is two years.",
            }
        ],
        "calculations": [],
        "evidence_sources": ["ACME · FY2024 Annual · p.4"],
        "answer": "The product warranty is five years.",
        "citations": [],
        "grounding_status": None,
        "grounding_issues": [],
        "error": None,
        "start_time": 0.0,
    }


def test_grounding_verdict_requires_clean_pass():
    passed, issues = graph._parse_grounding_verdict('{"passed": true, "issues": []}')
    assert passed
    assert issues == []

    passed, issues = graph._parse_grounding_verdict(
        json.dumps({"passed": True, "issues": ["Warranty claim is contradicted."]})
    )
    assert not passed
    assert issues == ["Warranty claim is contradicted."]


@pytest.mark.parametrize("content", ["not json", '{"passed": true}'])
def test_invalid_grounding_verdict_fails_closed(content):
    passed, issues = graph._parse_grounding_verdict(content)
    assert not passed
    assert issues


@pytest.mark.asyncio
async def test_grounding_failure_gets_one_revision_then_passes(monkeypatch):
    llm = _FakeLLM(
        [
            json.dumps({"passed": False, "issues": ["Five years contradicts the source."]}),
            "The product warranty is two years. [ACME · FY2024 Annual · p.4]",
            json.dumps({"passed": True, "issues": []}),
        ]
    )
    monkeypatch.setattr(graph, "_make_llm", lambda temperature: llm)

    result = await graph.verify_grounding(_state())

    assert result["grounding_status"] == "verified"
    assert result["grounding_issues"] == []
    assert "two years" in result["answer"]
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_unverified_answer_is_not_returned_after_failed_revision(monkeypatch):
    llm = _FakeLLM(
        [
            json.dumps({"passed": False, "issues": ["Claim is unsupported."]}),
            "Still unsupported.",
            json.dumps({"passed": False, "issues": ["Claim remains unsupported."]}),
        ]
    )
    monkeypatch.setattr(graph, "_make_llm", lambda temperature: llm)

    result = await graph.verify_grounding(_state())

    assert result["grounding_status"] == "unverified"
    assert result["answer"].startswith("I couldn't verify")
    assert result["grounding_issues"] == ["Claim remains unsupported."]
    assert result["citations"] == []
    assert llm.calls == 3

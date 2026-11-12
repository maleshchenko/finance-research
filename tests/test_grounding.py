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
        "unverified_draft": None,
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

    passed, issues = graph._parse_grounding_verdict(
        json.dumps(
            {
                "passed": False,
                "issues": [
                    "GOOGL operating margin Q4 FY2023 is unsupported.",
                    "  googl operating margin q4 fy2023 is unsupported.  ",
                ],
            }
        )
    )
    assert not passed
    assert issues == ["GOOGL operating margin Q4 FY2023 is unsupported."]


def test_grounding_verdict_extracts_json_surrounded_by_explanatory_text():
    passed, issues = graph._parse_grounding_verdict(
        'Evaluator response: {"passed": true, "issues": []} End of response.'
    )

    assert passed
    assert issues == []


@pytest.mark.parametrize("content", ["not json", '{"passed": true}'])
def test_invalid_grounding_verdict_fails_closed(content):
    passed, issues = graph._parse_grounding_verdict(content)
    assert not passed
    assert issues


@pytest.mark.asyncio
async def test_grounding_evaluator_uses_json_schema_output(monkeypatch):
    class _StructuredFakeLLM:
        def with_structured_output(self, schema, *, method):
            assert schema == graph._GROUNDING_VERDICT_SCHEMA
            assert method == "json_schema"
            return self

        async def ainvoke(self, _messages):
            return {"passed": True, "issues": []}

    monkeypatch.setattr(graph, "_make_llm", lambda temperature: _StructuredFakeLLM())

    passed, issues = await graph._evaluate_answer(_state())

    assert passed
    assert issues == []


@pytest.mark.asyncio
async def test_classify_query_accepts_markdown_json(monkeypatch):
    llm = _FakeLLM(['```json\n{"classification": "METRICS", "reasoning": "Revenue question"}\n```'])
    monkeypatch.setattr(graph, "_make_llm", lambda temperature: llm)

    result = await graph.classify_query(
        {
            "question": "What was Apple's revenue in Q4 2023?",
            "classification": None,
        }
    )

    assert result["classification"] == "METRICS"


@pytest.mark.asyncio
async def test_classify_query_defaults_to_hybrid_when_response_is_unparseable(monkeypatch):
    llm = _FakeLLM(['"classification"'])
    monkeypatch.setattr(graph, "_make_llm", lambda temperature: llm)

    result = await graph.classify_query(
        {
            "question": "What is the market outlook?",
            "classification": None,
        }
    )

    assert result["classification"] == "HYBRID"


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
    assert result["unverified_draft"] is None
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
    assert "should not be relied on" in result["answer"]
    assert result["grounding_issues"] == ["Claim remains unsupported."]
    assert result["unverified_draft"] == "Still unsupported."
    assert result["citations"] == []
    assert llm.calls == 3

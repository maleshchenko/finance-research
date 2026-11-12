import json

import pytest

from app.agent import graph


def _state(question: str) -> graph.AgentState:
    return {
        "question": question,
        "classification": None,
        "structured_metric_request": None,
        "structured_metric_records": [],
        "deterministic_metrics_answer": False,
        "structured_metrics_table": None,
        "structured_metrics_summary": None,
        "metric_explanation": None,
        "messages": [],
        "tool_calls_made": [],
        "evidence": [],
        "calculations": [],
        "evidence_sources": [],
        "answer": None,
        "citations": [],
        "grounding_status": None,
        "grounding_issues": [],
        "unverified_draft": None,
        "error": None,
        "start_time": 0.0,
    }


def test_parse_full_fiscal_year_operating_margin_comparison():
    assert graph._parse_metric_request(
        "Compare operating margins for AAPL, MSFT, and GOOGL across FY2023"
    ) == {
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "metric_name": "operating_margin",
        "fiscal_year": 2023,
        "quarters": [1, 2, 3, 4],
        "include_explanation": False,
    }


def test_parse_trend_explanation_request():
    request = graph._parse_metric_request(
        "Compare operating margins for AAPL, MSFT, and GOOGL across FY2023 and explain the trend"
    )

    assert request is not None
    assert request["include_explanation"] is True


def test_explanatory_margin_question_stays_on_narrative_path():
    assert (
        graph._parse_metric_request("Explain why operating margins changed for AAPL in FY2023")
        is None
    )


@pytest.mark.asyncio
async def test_classify_routes_explicit_metric_comparison_without_llm(monkeypatch):
    async def fail_if_called(**_kwargs):
        raise AssertionError("explicit numeric comparisons do not need classification")

    monkeypatch.setattr(graph, "_make_llm", fail_if_called)
    state = await graph.classify_query(
        _state("Compare operating margins for AAPL, MSFT, and GOOGL across FY2023")
    )

    assert state["classification"] == "METRICS"
    assert state["structured_metric_request"]["quarters"] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_classify_routes_trend_request_to_structured_metrics_without_llm(monkeypatch):
    question = (
        "Compare operating margins for AAPL, MSFT, and GOOGL across FY2023 and explain the trend"
    )

    async def fail_if_called(**_kwargs):
        raise AssertionError("trend requests must use structured-metric routing")

    monkeypatch.setattr(graph, "_make_llm", fail_if_called)
    state = await graph.classify_query(_state(question))

    assert state["classification"] == "METRICS"
    assert state["structured_metric_request"]["include_explanation"] is True


@pytest.mark.asyncio
async def test_numeric_comparison_uses_sec_rows_without_llm_or_narrative_retrieval(
    monkeypatch,
):
    question = "Compare operating margins for AAPL, MSFT, and GOOGL across FY2023"
    request = graph._parse_metric_request(question)
    records = []
    for ticker, base in (("AAPL", 30.0), ("MSFT", 40.0), ("GOOGL", 25.0)):
        for quarter in range(1, 5):
            records.append(
                {
                    "ticker": ticker,
                    "period": f"FY2023 Q{quarter}",
                    "metric": "operating_margin",
                    "value": base + quarter / 10,
                    "unit": "percent",
                    "source_provider": "sec_companyfacts",
                    "source_note": "SEC sourced; Q4 is derived from annual less Q1–Q3.",
                    "citation": (
                        f"[STRUCTURED DATA: {ticker} operating_margin · "
                        f"FY2023 Q{quarter} · SEC accession](https://data.sec.gov/facts)"
                    ),
                }
            )

    class _MetricTool:
        name = "query_metrics"

        async def ainvoke(self, arguments):
            assert arguments == {
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "metric_names": ["operating_margin"],
                "fiscal_year_min": 2023,
                "fiscal_year_max": 2023,
            }
            return json.dumps(records)

    class _NarrativeTool:
        name = "retrieve_narrative"

        async def ainvoke(self, _arguments):
            raise AssertionError("numeric comparisons must not retrieve narrative chunks")

    monkeypatch.setattr(
        graph,
        "build_tools",
        lambda _db: [_MetricTool(), _NarrativeTool()],
    )
    monkeypatch.setattr(
        graph,
        "_make_llm",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("numeric comparisons must not call an LLM")
        ),
    )

    state = {
        **_state(question),
        "classification": "METRICS",
        "structured_metric_request": request,
    }
    state = await graph.run_agent(state, db=None)
    state = await graph.synthesise(state)
    state = await graph.verify_grounding(state)

    assert state["grounding_status"] == "verified"
    assert len(state["citations"]) == 12
    assert "Consistent structured values available for 12 of 12 cells." in state["answer"]
    assert "SEC sourced; Q4 is derived from annual less Q1–Q3." in state["answer"]
    assert "[[STRUCTURED DATA:" not in state["answer"]
    assert state["tool_calls_made"] == ["query_metrics"]


def _metric_records():
    records = []
    for ticker, base in (("AAPL", 30.0), ("MSFT", 40.0), ("GOOGL", 25.0)):
        for quarter, change in enumerate((0.4, -0.3, -1.0, 0.8), start=1):
            records.append(
                {
                    "ticker": ticker,
                    "period": f"FY2023 Q{quarter}",
                    "metric": "operating_margin",
                    "value": base + change,
                    "unit": "percent",
                    "source_provider": "sec_companyfacts",
                    "source_note": "SEC sourced.",
                    "citation": (
                        f"[STRUCTURED DATA: {ticker} operating_margin · "
                        f"FY2023 Q{quarter} · SEC accession](https://data.sec.gov/facts)"
                    ),
                }
            )
    return records


def _deterministic_metric_state(question: str):
    records = _metric_records()
    return {
        **_state(question),
        "classification": "METRICS",
        "structured_metric_request": graph._parse_metric_request(question),
        "structured_metric_records": records,
        "deterministic_metrics_answer": True,
        "evidence": [{"tool": "query_metrics", "result": records}],
        "evidence_sources": [record["citation"] for record in records],
    }


@pytest.mark.asyncio
async def test_explanation_request_runs_llm_without_changing_metric_table(monkeypatch):
    question = (
        "Compare operating margins for AAPL, MSFT, and GOOGL across FY2023 and explain the trend"
    )
    direction_groups = {
        "overall_direction_groups": {
            "increased": ["AAPL", "MSFT", "GOOGL"],
            "decreased": [],
            "unchanged": [],
        }
    }

    class _FakeLLM:
        async def ainvoke(self, messages):
            if "Group companies" in messages[1].content:
                assert '"ticker": "GOOGL", "overall_direction": "increased"' in messages[1].content
            return direction_groups

    monkeypatch.setattr(graph, "_make_llm", lambda **_kwargs: _FakeLLM())
    state = await graph.synthesise(_deterministic_metric_state(question))
    table = state["structured_metrics_table"]
    state = await graph.verify_grounding(state)

    assert state["grounding_status"] == "verified"
    assert state["answer"].startswith(table)
    assert "### Calculated trend summary" in state["answer"]
    assert "GOOGL: decreased from FY2023 Q1 to FY2023 Q2" in state["answer"]
    assert "overall increased" in state["answer"]
    assert "### LLM synthesis" in state["answer"]
    assert "Overall, operating margins increased for AAPL, MSFT, and GOOGL." in state["answer"]
    assert "[AAPL FY2023 Q1]" in state["answer"]
    assert len(state["citations"]) == 12


@pytest.mark.asyncio
async def test_incorrect_llm_direction_groups_are_withheld_but_trends_are_preserved(monkeypatch):
    question = (
        "Compare operating margins for AAPL, MSFT, and GOOGL across FY2023 and explain the trend"
    )
    incorrect_groups = {
        "overall_direction_groups": {
            "increased": ["AAPL", "MSFT"],
            "decreased": ["GOOGL"],
            "unchanged": [],
        }
    }
    responses = iter([incorrect_groups, incorrect_groups])

    class _FakeLLM:
        async def ainvoke(self, _messages):
            return next(responses)

    monkeypatch.setattr(graph, "_make_llm", lambda **_kwargs: _FakeLLM())
    state = await graph.synthesise(_deterministic_metric_state(question))
    table = state["structured_metrics_table"]
    state = await graph.verify_grounding(state)

    assert state["grounding_status"] == "partial"
    assert state["answer"].startswith(table)
    assert "### Calculated trend summary" in state["answer"]
    assert "LLM synthesis was withheld" in state["answer"]
    assert state["grounding_issues"]
    assert state["unverified_draft"] is None
    assert len(state["citations"]) == 12

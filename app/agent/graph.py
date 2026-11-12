"""
LangGraph agent graph for the financial research assistant.

Graph topology:
  user_question → classify → agent_loop (pre-fetch RAG + ReAct) → synthesise → END
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import date
from typing import Any, Optional, TypedDict

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools import build_tools
from app.config import settings
from app.llm import make_llm
from app.prompts.system import (
    CLASSIFIER_PROMPT,
    GROUNDING_EVALUATOR_PROMPT,
    METRIC_TREND_PROMPT,
    SYNTHESIS_PROMPT,
    SYSTEM_PROMPT,
)

log = structlog.get_logger(__name__)

# Common ticker symbols to look for in questions
_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
_KNOWN_TICKERS = {
    "AAPL",
    "MSFT",
    "GOOGL",
    "GOOG",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "BRK",
    "JPM",
    "V",
    "UNH",
    "JNJ",
    "WMT",
    "XOM",
    "PG",
    "MA",
    "HD",
    "CVX",
    "MRK",
    "LLY",
    "ABBV",
    "PEP",
    "COST",
    "KO",
    "AVGO",
    "MCD",
    "ACN",
    "TMO",
    "CSCO",
    "ABT",
    "DHR",
    "NEE",
    "TXN",
    "VZ",
    "ADBE",
    "CRM",
    "NFLX",
    "INTC",
    "AMD",
    "QCOM",
    "IBM",
    "ORCL",
    "SAP",
}


# ── State definition ──────────────────────────────────────────────────────────


class AgentState(TypedDict):
    question: str
    classification: Optional[str]
    structured_metric_request: Optional[dict[str, Any]]
    structured_metric_records: list[dict[str, Any]]
    deterministic_metrics_answer: bool
    structured_metrics_table: Optional[str]
    structured_metrics_summary: Optional[str]
    metric_explanation: Optional[str]
    messages: list[BaseMessage]
    tool_calls_made: list[str]
    evidence: list[dict[str, Any]]
    calculations: list[dict[str, Any]]
    evidence_sources: list[str]  # citation labels extracted from evidence
    answer: Optional[str]
    citations: list[str]
    grounding_status: Optional[str]
    grounding_issues: list[str]
    unverified_draft: Optional[str]
    error: Optional[str]
    start_time: float


# ── LLM factory ───────────────────────────────────────────────────────────────


def _make_llm(temperature: float = settings.agent_temperature):
    return make_llm(temperature=temperature)


# ── Ticker extraction ─────────────────────────────────────────────────────────


def _extract_tickers(question: str) -> list[str]:
    """Pull recognised ticker symbols out of the question text."""
    found = _TICKER_RE.findall(question)
    return [t for t in dict.fromkeys(found) if t in _KNOWN_TICKERS]


def _parse_metric_request(question: str) -> dict[str, Any] | None:
    """Recognize unambiguous operating-margin comparisons with an explicit FY."""
    normalized = question.lower()
    if not re.search(r"\boperating\s+margins?\b", normalized):
        return None
    if not re.search(r"\b(compare|comparing|across|over)\b", normalized):
        return None
    if re.search(r"\b(why|reason|cause|driver|drove)\b", normalized):
        return None

    year_match = re.search(r"\bfy\s*(20\d{2})\b", normalized)
    tickers = _extract_tickers(question)
    if year_match is None or not tickers:
        return None

    quarters = sorted({int(quarter) for quarter in re.findall(r"\bq([1-4])\b", normalized)}) or [
        1,
        2,
        3,
        4,
    ]
    return {
        "tickers": tickers,
        "metric_name": "operating_margin",
        "fiscal_year": int(year_match.group(1)),
        "quarters": quarters,
        "include_explanation": bool(
            re.search(r"\b(explain|interpret|trend|pattern|insight)\b", normalized)
        ),
    }


# ── Node: classify ────────────────────────────────────────────────────────────

_VALID_CLASSIFICATIONS = {"NARRATIVE", "METRICS", "HYBRID", "DISCOVERY"}

_GROUNDING_VERDICT_SCHEMA = {
    "title": "GroundingVerdict",
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["passed", "issues"],
    "additionalProperties": False,
}

_METRIC_TREND_SCHEMA = {
    "title": "MetricDirectionGroups",
    "type": "object",
    "properties": {
        "overall_direction_groups": {
            "type": "object",
            "properties": {
                "increased": {"type": "array", "items": {"type": "string"}},
                "decreased": {"type": "array", "items": {"type": "string"}},
                "unchanged": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["increased", "decreased", "unchanged"],
            "additionalProperties": False,
        },
    },
    "required": ["overall_direction_groups"],
    "additionalProperties": False,
}


def _coerce_json_like(value: Any) -> Any:
    """Best-effort conversion of LLM output into JSON-compatible Python objects."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value

    text = str(value).strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    if text.startswith("{") and text.endswith("}"):
        candidates.insert(0, text)
    elif text.startswith("[") and text.endswith("]"):
        candidates.insert(0, text)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass

    try:
        import ast

        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def _normalise_classification(value: Any) -> str:
    """Accept a variety of LLM JSON/quote formatting and map to the allowed enum."""
    if isinstance(value, dict):
        for key in ("classification", '"classification"', "'classification'"):
            if key in value:
                return _normalise_classification(value[key])
        return "HYBRID"

    if isinstance(value, str):
        candidate = value.strip().strip("\"'")
        normalized = candidate.upper()
        if normalized in _VALID_CLASSIFICATIONS:
            return normalized

    return "HYBRID"


async def classify_query(state: AgentState) -> AgentState:
    question = state.get("question") or ""
    metric_request = _parse_metric_request(question)
    if metric_request:
        log.info(
            "query_classified",
            classification="METRICS",
            route="structured_metrics",
        )
        return {
            **state,
            "classification": "METRICS",
            "structured_metric_request": metric_request,
        }

    log.info("query_classification_started", question_length=len(question))
    llm = _make_llm(temperature=0.0)
    prompt = CLASSIFIER_PROMPT.format(question=question)
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    parsed = _coerce_json_like(getattr(response, "content", None))
    if isinstance(parsed, dict):
        classification = _normalise_classification(parsed.get("classification"))
    else:
        classification = _normalise_classification(parsed)

    log.info("query_classified", classification=classification)
    return {**state, "classification": classification}


# ── Node: agent ───────────────────────────────────────────────────────────────


async def run_agent(state: AgentState, db: AsyncSession) -> AgentState:
    """
    1. Pre-fetch RAG: one retrieve_narrative call per detected ticker (parallel),
       falling back to a single broad query if no tickers found.
    2. Inject retrieved evidence into the prompt.
    3. Run the ReAct agent for any additional tool calls.
    """
    log.info("agent_started", classification=state.get("classification"))
    tools = build_tools(db)
    metric_request = state.get("structured_metric_request")
    if metric_request:
        metric_tool = next((t for t in tools if t.name == "query_metrics"), None)
        if metric_tool is None:
            raise RuntimeError("The structured metrics tool is not configured.")
        raw_result = await metric_tool.ainvoke(
            {
                "tickers": metric_request["tickers"],
                "metric_names": [metric_request["metric_name"]],
                "fiscal_year_min": metric_request["fiscal_year"],
                "fiscal_year_max": metric_request["fiscal_year"],
            }
        )
        parsed_result = json.loads(raw_result)
        records = (
            parsed_result
            if isinstance(parsed_result, list)
            else parsed_result.get("results")
            if isinstance(parsed_result, dict)
            else None
        )
        if not isinstance(records, list):
            raise ValueError("Structured metrics tool returned an invalid response.")
        evidence = [{"tool": "query_metrics", "result": records}]
        log.info(
            "structured_metrics_query_complete",
            tickers=metric_request["tickers"],
            metric=metric_request["metric_name"],
            records=len(records),
        )
        return {
            **state,
            "tool_calls_made": ["query_metrics"],
            "evidence": evidence,
            "structured_metric_records": records,
            "deterministic_metrics_answer": True,
            "evidence_sources": _extract_sources(evidence),
        }

    search_tool = next((t for t in tools if t.name == "retrieve_narrative"), None)

    # ── Step 1: parallel per-ticker RAG pre-fetch ─────────────────────────────
    pre_evidence: list[dict] = []
    rag_text = ""

    if search_tool:
        tickers = _extract_tickers(state["question"])
        log.info("pre_rag_tickers", tickers=tickers or ["(broad)"])

        async def _fetch(ticker_filter: Optional[str]) -> str:
            kwargs: dict[str, Any] = {"query": state["question"], "top_k": 6}
            if ticker_filter:
                kwargs["tickers"] = [ticker_filter]
            return await search_tool.ainvoke(kwargs)

        try:
            if tickers:
                # One call per ticker in parallel
                raw_results = await asyncio.gather(
                    *[_fetch(t) for t in tickers],
                    return_exceptions=True,
                )
                combined: list[dict] = []
                for ticker, raw in zip(tickers, raw_results):
                    if isinstance(raw, Exception):
                        log.warning("pre_rag_ticker_failed", ticker=ticker, error=str(raw))
                        continue
                    try:
                        chunks = json.loads(raw)
                        if isinstance(chunks, list):
                            combined.extend(chunks)
                    except (json.JSONDecodeError, TypeError):
                        combined.append({"ticker": ticker, "raw": raw})

                if combined:
                    pre_evidence = [{"tool": "retrieve_narrative", "result": combined}]
                    rag_text = (
                        "\n\nRelevant excerpts retrieved from the knowledge base "
                        f"(per-ticker parallel fetch for {', '.join(tickers)}):\n"
                        + json.dumps(combined, ensure_ascii=False)
                    )
                    log.info("pre_rag_ok", tickers=tickers, chunks=len(combined))
            else:
                # Broad single query
                raw = await _fetch(None)
                pre_evidence = [{"tool": "retrieve_narrative", "result": raw}]
                rag_text = f"\n\nRelevant excerpts retrieved from the knowledge base:\n{raw}"
                log.info("pre_rag_ok", tickers=[], chars=len(raw))

        except Exception as exc:
            log.warning("pre_rag_failed", error=str(exc), exc_type=type(exc).__name__)

    # ── Step 2: build question with embedded evidence ─────────────────────────
    classification_hint = (
        f"\n\nQuery classification: {state['classification']}."
        if state.get("classification")
        else ""
    )
    question_with_context = (
        state["question"]
        + rag_text
        + classification_hint
        + "\n\nAnswer using the retrieved excerpts above. "
        "Call query_metrics or calculate if you need structured numbers."
    )

    # ── Step 3: ReAct agent ───────────────────────────────────────────────────
    llm = _make_llm()
    agent = create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)
    input_messages = [HumanMessage(content=question_with_context)]

    try:
        log.info("react_agent_started", available_tools=[tool.name for tool in tools])
        result = await agent.ainvoke(
            {"messages": input_messages},
            config={"recursion_limit": settings.agent_max_iterations * 2},
        )
    except Exception as e:
        log.error("agent_error", error=str(e))
        return {
            **state,
            "evidence": pre_evidence,
            "tool_calls_made": ["retrieve_narrative"],
            "evidence_sources": _extract_sources(pre_evidence),
            "error": str(e),
        }

    messages = result.get("messages", [])
    tool_calls = [
        m.tool_calls[0]["name"] for m in messages if isinstance(m, AIMessage) and m.tool_calls
    ]
    if "retrieve_narrative" not in tool_calls and pre_evidence:
        tool_calls = ["retrieve_narrative"] + tool_calls

    # Collect tool outputs
    evidence = list(pre_evidence)
    calculations = []
    for m in messages:
        if isinstance(m, ToolMessage):
            try:
                parsed = json.loads(m.content)
            except (json.JSONDecodeError, TypeError):
                parsed = {"raw": m.content}
            if isinstance(parsed, list):
                evidence.extend(parsed)
            elif isinstance(parsed, dict):
                if parsed.get("operation"):
                    calculations.append(parsed)
                else:
                    evidence.append(parsed)

    log.info(
        "agent_completed",
        tool_calls=tool_calls,
        evidence_items=len(evidence),
        calculations=len(calculations),
    )
    return {
        **state,
        "messages": messages,
        "tool_calls_made": tool_calls,
        "evidence": evidence,
        "calculations": calculations,
        "evidence_sources": _extract_sources(evidence),
    }


def _extract_sources(evidence: list[dict]) -> list[str]:
    """Pull citation labels out of retrieved chunks."""
    sources: list[str] = []
    for item in evidence:
        result = item.get("result", item)
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(result, list):
            for chunk in result:
                if isinstance(chunk, dict) and "citation" in chunk:
                    sources.append(chunk["citation"])
    return list(dict.fromkeys(sources))  # deduplicate, preserve order


# ── Node: synthesise ──────────────────────────────────────────────────────────


def _calculate_metric_trends(
    request: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    periods = [f"FY{request['fiscal_year']} Q{quarter}" for quarter in request["quarters"]]
    records_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["ticker"], record["period"])
        records_by_cell.setdefault(key, []).append(record)

    trend_facts: list[dict[str, Any]] = []
    source_refs_by_ticker: dict[str, list[str]] = {}
    for ticker in request["tickers"]:
        values: list[float] = []
        source_refs: list[str] = []
        for period in periods:
            candidates = records_by_cell.get((ticker, period), [])
            candidate_values = {float(record["value"]) for record in candidates}
            if not candidates or len(candidate_values) != 1:
                raise ValueError(
                    f"Cannot interpret inconsistent metric data for {ticker} {period}."
                )
            values.append(next(iter(candidate_values)))
            source_refs.append(f"[{ticker} {period}]")
        source_refs_by_ticker[ticker] = source_refs

        changes = [
            {
                "from": periods[index],
                "to": periods[index + 1],
                "direction": (
                    "increased"
                    if values[index + 1] > values[index]
                    else "decreased"
                    if values[index + 1] < values[index]
                    else "was unchanged"
                ),
            }
            for index in range(len(values) - 1)
        ]
        overall_direction = (
            "increased"
            if values[-1] > values[0]
            else "decreased"
            if values[-1] < values[0]
            else "was unchanged"
        )
        trend_facts.append(
            {
                "ticker": ticker,
                "overall_direction": overall_direction,
                "period_changes": changes,
                "source_refs": source_refs,
            }
        )
    return trend_facts, source_refs_by_ticker


def _render_metric_trend_summary(trend_facts: list[dict[str, Any]]) -> str:
    lines = ["### Calculated trend summary", ""]
    for trend in trend_facts:
        changes = [
            f"{change['direction']} from {change['from']} to {change['to']}"
            for change in trend["period_changes"]
        ]
        first_period, last_period = trend["source_refs"][0], trend["source_refs"][-1]
        overall = f"overall {trend['overall_direction']} from {first_period} to {last_period}"
        refs = " ".join(trend["source_refs"])
        lines.append(f"- {trend['ticker']}: {'; '.join(changes)}; {overall}. {refs}")
    return "\n".join(lines)


def _evidence_text(state: AgentState) -> tuple[str, str]:
    evidence_text = json.dumps(state.get("evidence", []), indent=2, ensure_ascii=False)[:6000]
    calc_text = json.dumps(state.get("calculations", []), indent=2, ensure_ascii=False)[:2000]
    return evidence_text, calc_text


async def _generate_answer(
    state: AgentState,
    revision_feedback: Optional[str] = None,
) -> str:
    evidence_text, calc_text = _evidence_text(state)
    prompt = SYNTHESIS_PROMPT.format(
        question=state["question"],
        evidence=evidence_text,
        calculations=calc_text,
        current_date=date.today().isoformat(),
    )
    if revision_feedback:
        prompt += (
            "\n\nGrounding review feedback:\n"
            f"{revision_feedback}\n"
            "Revise the answer using only the supplied evidence and calculations. "
            "Remove or qualify claims the evidence does not support."
        )

    response = await _make_llm(temperature=0.1).ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return response.content


def _answer_citations(state: AgentState, answer: str) -> list[str]:
    cited = re.findall(r"\[([A-Z]+\s*·[^\]]+)\]", answer)
    return list(dict.fromkeys(state.get("evidence_sources", []) + cited))


def _format_metric_value(value: float, unit: str | None) -> str:
    if unit == "percent":
        formatted = f"{value:.4f}".rstrip("0").rstrip(".")
        return f"{formatted}%"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _render_metric_answer(
    request: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[str, list[str], str, list[str]]:
    periods = [f"FY{request['fiscal_year']} Q{q}" for q in request["quarters"]]
    records_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["ticker"], record["period"])
        records_by_cell.setdefault(key, []).append(record)

    lines = [
        f"### Operating Margin — FY{request['fiscal_year']}",
        "",
        "| Company | Period | Value |",
        "|---|---|---:|",
    ]
    citations: list[str] = []
    notes: list[str] = []
    issues: list[str] = []
    available_cells = 0

    for ticker in request["tickers"]:
        for period in periods:
            candidates = records_by_cell.get((ticker, period), [])
            notes.extend(
                record["source_note"] for record in candidates if record.get("source_note")
            )
            references = list(
                dict.fromkeys(record["citation"] for record in candidates if record.get("citation"))
            )
            citations.extend(references)
            refs_text = " ".join(
                ref if ref.startswith("[") and "](" in ref else f"[{ref}]" for ref in references
            )

            if not candidates:
                lines.append(f"| {ticker} | {period} | Not available in structured data |")
                issues.append(f"No structured metric record found for {ticker} {period}.")
                continue

            values = {(float(record["value"]), record.get("unit")) for record in candidates}
            if len(values) > 1:
                lines.append(
                    f"| {ticker} | {period} | Conflicting records; value withheld {refs_text} |"
                )
                issues.append(f"Conflicting structured records for {ticker} {period}.")
                continue

            available_cells += 1
            value, unit = next(iter(values))
            lines.append(
                f"| {ticker} | {period} | {_format_metric_value(value, unit)} {refs_text}|"
            )

    total_cells = len(request["tickers"]) * len(periods)
    status = (
        "verified"
        if available_cells == total_cells
        else "partial"
        if available_cells
        else "insufficient_data"
    )
    lines.extend(
        [
            "",
            f"Consistent structured values available for {available_cells} of {total_cells} cells.",
        ]
    )
    if notes:
        lines.append("Data notes: " + " ".join(dict.fromkeys(notes)))

    return (
        "\n".join(lines),
        list(dict.fromkeys(citations)),
        status,
        issues,
    )


async def _generate_metric_explanation(
    state: AgentState,
    revision_feedback: str = "",
) -> tuple[str | None, list[str]]:
    request = state.get("structured_metric_request")
    if request is None:
        raise RuntimeError("Structured metric results have no associated request.")

    trend_facts, source_refs_by_ticker = _calculate_metric_trends(
        request,
        state.get("structured_metric_records", []),
    )
    direction_groups: dict[str, list[str]] = {
        "increased": [],
        "decreased": [],
        "unchanged": [],
    }
    for trend in trend_facts:
        group = (
            "unchanged"
            if trend["overall_direction"] == "was unchanged"
            else trend["overall_direction"]
        )
        direction_groups[group].append(trend["ticker"])

    prompt = METRIC_TREND_PROMPT.format(
        question=state["question"],
        records=json.dumps(trend_facts, ensure_ascii=False),
        revision_feedback=revision_feedback or "No revision feedback.",
    )
    llm = _make_llm(temperature=0.1)
    with_structured_output = getattr(llm, "with_structured_output", None)
    if callable(with_structured_output):
        llm = with_structured_output(_METRIC_TREND_SCHEMA, method="json_schema")
    response = await llm.ainvoke(
        [
            SystemMessage(content="Interpret only the supplied structured metrics."),
            HumanMessage(content=prompt),
        ]
    )
    content = getattr(response, "content", response)
    if isinstance(content, str):
        parsed_content = _coerce_json_like(content)
        if isinstance(parsed_content, dict):
            content = parsed_content

    if not isinstance(content, dict):
        return None, ["LLM trend synthesis returned a malformed direction report."]
    actual_groups = content.get("overall_direction_groups")
    if not isinstance(actual_groups, dict):
        return None, ["LLM trend synthesis omitted overall direction groups."]

    issues: list[str] = []
    for direction, expected_tickers in direction_groups.items():
        actual_tickers = actual_groups.get(direction)
        if (
            not isinstance(actual_tickers, list)
            or not all(isinstance(ticker, str) for ticker in actual_tickers)
            or len(actual_tickers) != len(set(actual_tickers))
            or set(actual_tickers) != set(expected_tickers)
        ):
            issues.append(
                f"LLM classified {direction} overall trend for "
                f"{actual_tickers!r}; expected {expected_tickers!r}."
            )
    if set(actual_groups) != set(direction_groups):
        issues.append("LLM trend synthesis returned unexpected direction groups.")
    if issues:
        return None, issues

    clauses: list[str] = []
    for direction in ("increased", "decreased", "unchanged"):
        tickers = [ticker for ticker in request["tickers"] if ticker in direction_groups[direction]]
        if not tickers:
            continue
        if len(tickers) == 1:
            names = tickers[0]
        elif len(tickers) == 2:
            names = f"{tickers[0]} and {tickers[1]}"
        else:
            names = f"{', '.join(tickers[:-1])}, and {tickers[-1]}"
        verb = "was unchanged for" if direction == "unchanged" else f"{direction} for"
        clauses.append(f"{verb} {names}")

    explanation = "Overall, operating margins " + "; ".join(clauses) + "."
    source_refs = [
        source_ref
        for ticker in request["tickers"]
        for source_ref in (source_refs_by_ticker[ticker][0], source_refs_by_ticker[ticker][-1])
    ]
    return f"{explanation} {' '.join(source_refs)}", []


async def synthesise(state: AgentState) -> AgentState:
    if state.get("deterministic_metrics_answer"):
        request = state.get("structured_metric_request")
        if request is None:
            raise RuntimeError("Structured metric results have no associated request.")
        table, citations, status, issues = _render_metric_answer(
            request,
            state.get("structured_metric_records", []),
        )
        result = {
            **state,
            "answer": table,
            "citations": citations,
            "grounding_status": status,
            "grounding_issues": issues,
            "unverified_draft": None,
            "structured_metrics_table": table,
            "structured_metrics_summary": None,
            "metric_explanation": None,
        }
        if request.get("include_explanation") and status == "verified":
            trend_facts, _ = _calculate_metric_trends(
                request,
                state.get("structured_metric_records", []),
            )
            summary = _render_metric_trend_summary(trend_facts)
            result.update(
                answer=f"{table}\n\n{summary}",
                structured_metrics_summary=summary,
            )
            log.info(
                "metric_trend_synthesis_started",
                provider=settings.llm_provider,
                records=len(state.get("structured_metric_records", [])),
            )
            explanation, trend_issues = await _generate_metric_explanation(result)
            if trend_issues:
                log.info(
                    "metric_trend_revision_started",
                    issue_count=len(trend_issues),
                )
                explanation, trend_issues = await _generate_metric_explanation(
                    result,
                    revision_feedback="\n".join(trend_issues),
                )
            if explanation is not None:
                result.update(
                    answer=f"{table}\n\n{summary}\n\n### LLM synthesis\n\n{explanation}",
                    metric_explanation=explanation,
                )
                log.info(
                    "metric_trend_synthesis_complete",
                    answer_length=len(explanation),
                )
            else:
                result.update(
                    answer=(
                        f"{table}\n\n{summary}\n\n"
                        "LLM synthesis was withheld because it did not match the "
                        "calculated trend directions."
                    ),
                    grounding_status="partial",
                    grounding_issues=trend_issues,
                )
                log.warning(
                    "metric_trend_synthesis_withheld",
                    issue_count=len(trend_issues),
                )
        elif request.get("include_explanation"):
            log.info(
                "metric_trend_synthesis_skipped",
                reason="structured_data_incomplete",
                status=status,
            )
        return result

    if state.get("error") and not state.get("evidence"):
        return {**state, "answer": f"Error: {state['error']}"}

    log.info("synthesis_started", evidence_items=len(state.get("evidence", [])))
    answer = await _generate_answer(state)
    all_citations = _answer_citations(state, answer)

    log.info(
        "synthesis_complete",
        answer_length=len(answer),
        citations=len(all_citations),
        elapsed=round(time.perf_counter() - state["start_time"], 2),
    )
    return {**state, "answer": answer, "citations": all_citations}


# ── Node: grounding evaluator ─────────────────────────────────────────────────


def _parse_grounding_verdict(content: Any) -> tuple[bool, list[str]]:
    verdict = _coerce_json_like(content)
    if not isinstance(verdict, dict):
        text = str(content)
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                verdict = candidate
                break

    if (
        not isinstance(verdict, dict)
        or not isinstance(verdict.get("passed"), bool)
        or not isinstance(verdict.get("issues"), list)
        or not all(isinstance(issue, str) for issue in verdict["issues"])
    ):
        return False, ["Grounding evaluator returned a malformed verdict."]

    issues: list[str] = []
    seen_issues: set[str] = set()
    for raw_issue in verdict["issues"]:
        issue = " ".join(raw_issue.split())
        normalized_issue = issue.casefold()
        if issue and normalized_issue not in seen_issues:
            issues.append(issue)
            seen_issues.add(normalized_issue)

    passed = verdict["passed"] and not issues
    if not passed and not issues:
        issues = ["The evaluator did not confirm that all material claims are grounded."]
    return passed, issues


async def _evaluate_answer(state: AgentState) -> tuple[bool, list[str]]:
    evidence_text, calc_text = _evidence_text(state)
    review_input = json.dumps(
        {
            "question": state["question"],
            "draft_answer": state.get("answer"),
            "retrieved_evidence": evidence_text,
            "calculations": calc_text,
        },
        ensure_ascii=False,
    )
    llm = _make_llm(temperature=0.0)
    with_structured_output = getattr(llm, "with_structured_output", None)
    if callable(with_structured_output):
        llm = with_structured_output(_GROUNDING_VERDICT_SCHEMA, method="json_schema")
    response = await llm.ainvoke(
        [
            SystemMessage(content=GROUNDING_EVALUATOR_PROMPT),
            HumanMessage(content=review_input),
        ]
    )
    return _parse_grounding_verdict(getattr(response, "content", response))


async def verify_grounding(state: AgentState) -> AgentState:
    if state.get("deterministic_metrics_answer"):
        log.info(
            "grounding_check_skipped",
            reason="structured_metrics_and_validated_trend_groups",
            status=state.get("grounding_status"),
        )
        return state

    if state.get("error") and not state.get("evidence"):
        log.info("grounding_check_skipped", reason="agent_error_without_evidence")
        return {**state, "grounding_status": "not_evaluated", "grounding_issues": []}

    log.info("grounding_check_started", attempt=1)
    passed, issues = await _evaluate_answer(state)
    log.info("grounding_check_completed", attempt=1, passed=passed, issue_count=len(issues))
    if passed:
        return {**state, "grounding_status": "verified", "grounding_issues": []}

    log.info("grounding_revision_started", issue_count=len(issues))
    revised_answer = await _generate_answer(state, revision_feedback="\n".join(issues))
    revised_state = {
        **state,
        "answer": revised_answer,
        "citations": _answer_citations(state, revised_answer),
    }
    log.info("grounding_check_started", attempt=2)
    passed, issues = await _evaluate_answer(revised_state)
    log.info("grounding_check_completed", attempt=2, passed=passed, issue_count=len(issues))
    if passed:
        log.info("grounding_verified_after_revision")
        return {
            **revised_state,
            "grounding_status": "verified",
            "grounding_issues": [],
        }

    log.warning("grounding_failed_after_revision", issue_count=len(issues))
    return {
        **revised_state,
        "citations": [],
        "answer": (
            "Grounding review failed after one revision. The draft is included "
            "separately as unverified and should not be relied on. See "
            "`grounding_issues` for the specific evidence gaps."
        ),
        "grounding_status": "unverified",
        "grounding_issues": issues,
        "unverified_draft": revised_answer,
    }


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_graph(db: AsyncSession) -> Any:
    async def _run_agent(state: AgentState) -> AgentState:
        return await run_agent(state, db)

    builder = StateGraph(AgentState)
    builder.add_node("classify", classify_query)
    builder.add_node("agent", _run_agent)
    builder.add_node("synthesise", synthesise)
    builder.add_node("verify_grounding", verify_grounding)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "agent")
    builder.add_edge("agent", "synthesise")
    builder.add_edge("synthesise", "verify_grounding")
    builder.add_edge("verify_grounding", END)

    return builder.compile()


# ── Entry point ───────────────────────────────────────────────────────────────


async def run_research_query(
    db: AsyncSession,
    question: str,
) -> dict[str, Any]:
    graph = build_graph(db)
    initial_state: AgentState = {
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
        "start_time": time.perf_counter(),
    }

    final_state = await graph.ainvoke(initial_state)
    elapsed = round(time.perf_counter() - initial_state["start_time"], 2)

    return {
        "answer": final_state.get("answer"),
        "classification": final_state.get("classification"),
        "citations": final_state.get("citations", []),
        "grounding_status": final_state.get("grounding_status"),
        "grounding_issues": final_state.get("grounding_issues", []),
        "unverified_draft": final_state.get("unverified_draft"),
        "evidence_sources": final_state.get("evidence_sources", []),
        "tool_calls_made": final_state.get("tool_calls_made", []),
        "elapsed_seconds": elapsed,
        "error": final_state.get("error"),
    }

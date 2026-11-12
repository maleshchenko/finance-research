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
    SYNTHESIS_PROMPT,
    SYSTEM_PROMPT,
)

log = structlog.get_logger(__name__)

# Common ticker symbols to look for in questions
_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
_KNOWN_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "BRK", "JPM", "V", "UNH", "JNJ", "WMT", "XOM", "PG", "MA", "HD",
    "CVX", "MRK", "LLY", "ABBV", "PEP", "COST", "KO", "AVGO", "MCD",
    "ACN", "TMO", "CSCO", "ABT", "DHR", "NEE", "TXN", "VZ", "ADBE",
    "CRM", "NFLX", "INTC", "AMD", "QCOM", "IBM", "ORCL", "SAP",
}


# ── State definition ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    classification: Optional[str]
    messages: list[BaseMessage]
    tool_calls_made: list[str]
    evidence: list[dict[str, Any]]
    calculations: list[dict[str, Any]]
    evidence_sources: list[str]        # citation labels extracted from evidence
    answer: Optional[str]
    citations: list[str]
    grounding_status: Optional[str]
    grounding_issues: list[str]
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


# ── Node: classify ────────────────────────────────────────────────────────────

async def classify_query(state: AgentState) -> AgentState:
    log.info("query_classification_started", question_length=len(state["question"]))
    llm = _make_llm(temperature=0.0)
    prompt = CLASSIFIER_PROMPT.format(question=state["question"])
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        parsed = json.loads(response.content)
        classification = parsed.get("classification", "HYBRID")
    except (json.JSONDecodeError, AttributeError):
        classification = "HYBRID"

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
        if state.get("classification") else ""
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
        m.tool_calls[0]["name"]
        for m in messages
        if isinstance(m, AIMessage) and m.tool_calls
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


async def synthesise(state: AgentState) -> AgentState:
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
    try:
        verdict = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False, ["Grounding evaluator returned invalid JSON."]

    if (
        not isinstance(verdict, dict)
        or not isinstance(verdict.get("passed"), bool)
        or not isinstance(verdict.get("issues"), list)
        or not all(isinstance(issue, str) for issue in verdict["issues"])
    ):
        return False, ["Grounding evaluator returned a malformed verdict."]

    issues = verdict["issues"]
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
    response = await _make_llm(temperature=0.0).ainvoke(
        [
            SystemMessage(content=GROUNDING_EVALUATOR_PROMPT),
            HumanMessage(content=review_input),
        ]
    )
    return _parse_grounding_verdict(response.content)


async def verify_grounding(state: AgentState) -> AgentState:
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

    log.warning("grounding_failed_after_revision", issues=issues)
    return {
        **revised_state,
        "citations": [],
        "answer": (
            "I couldn't verify the generated answer against the available evidence, "
            "even after one revision. The sources available do not support a "
            "sufficiently grounded answer."
        ),
        "grounding_status": "unverified",
        "grounding_issues": issues,
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
        "messages": [],
        "tool_calls_made": [],
        "evidence": [],
        "calculations": [],
        "evidence_sources": [],
        "answer": None,
        "citations": [],
        "grounding_status": None,
        "grounding_issues": [],
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
        "evidence_sources": final_state.get("evidence_sources", []),
        "tool_calls_made": final_state.get("tool_calls_made", []),
        "elapsed_seconds": elapsed,
        "error": final_state.get("error"),
    }

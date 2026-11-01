"""
LangGraph agent graph for the financial research assistant.

Graph topology:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  user_question                                          │
│       │                                                 │
│       ▼                                                 │
│  classify_query                                         │
│       │                                                 │
│       ▼                                                 │
│  agent_loop  (pre-fetches RAG, then ReAct)              │
│       │                                                 │
│       ▼                                                 │
│  synthesise                                             │
│       │                                                 │
│       ▼                                                 │
│  END (answer)                                           │
└─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from sqlalchemy.ext.asyncio import AsyncSession
from typing import TypedDict

from app.agent.tools import build_tools
from app.config import settings
from app.llm import make_llm
from app.prompts.system import CLASSIFIER_PROMPT, SYSTEM_PROMPT, SYNTHESIS_PROMPT

log = structlog.get_logger(__name__)


# ── State definition ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    classification: Optional[str]
    messages: list[BaseMessage]
    tool_calls_made: list[str]
    evidence: list[dict[str, Any]]
    calculations: list[dict[str, Any]]
    answer: Optional[str]
    citations: list[str]
    error: Optional[str]
    start_time: float


# ── LLM factory ───────────────────────────────────────────────────────────────

def _make_llm(temperature: float = settings.agent_temperature):
    return make_llm(temperature=temperature)


# ── Node: classify ────────────────────────────────────────────────────────────

async def classify_query(state: AgentState) -> AgentState:
    llm = _make_llm(temperature=0.0)
    prompt = CLASSIFIER_PROMPT.format(question=state["question"])
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        parsed = json.loads(response.content)
        classification = parsed.get("classification", "HYBRID")
    except (json.JSONDecodeError, AttributeError):
        classification = "HYBRID"

    log.info("query_classified", classification=classification, question=state["question"][:80])
    return {**state, "classification": classification}


# ── Node: agent ───────────────────────────────────────────────────────────────

async def run_agent(state: AgentState, db: AsyncSession) -> AgentState:
    """
    1. Pre-fetch RAG results directly (don't rely on the LLM to call the tool).
    2. Inject them into the prompt so the model always has evidence.
    3. Run the ReAct agent for any additional tool calls.
    """
    tools = build_tools(db)

    # ── Step 1: mandatory RAG pre-fetch ──────────────────────────────────────
    search_tool = next((t for t in tools if t.name == "retrieve_narrative"), None)
    pre_evidence: list[dict] = []
    rag_text = ""
    if search_tool:
        try:
            rag_result = await search_tool.ainvoke(
                {"query": state["question"], "top_k": 8}
            )
            pre_evidence = [{"tool": "retrieve_narrative", "result": rag_result}]
            rag_text = f"\n\nRelevant excerpts already retrieved from the knowledge base:\n{rag_result}"
            log.info("pre_rag_ok", chars=len(rag_result))
        except Exception as exc:
            import traceback
            log.warning("pre_rag_failed", error=str(exc), exc_type=type(exc).__name__, tb=traceback.format_exc()[-500:])

    # ── Step 2: build the question with embedded evidence ────────────────────
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
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    input_messages = [HumanMessage(content=question_with_context)]

    try:
        result = await agent.ainvoke(
            {"messages": input_messages},
            config={"recursion_limit": settings.agent_max_iterations * 2},
        )
    except Exception as e:
        log.error("agent_error", error=str(e))
        # Still try to synthesise from the pre-fetched RAG evidence
        return {
            **state,
            "evidence": pre_evidence,
            "tool_calls_made": ["search_documents"],
            "error": str(e),
        }

    messages = result.get("messages", [])
    tool_calls = [
        m.tool_calls[0]["name"]
        for m in messages
        if isinstance(m, AIMessage) and m.tool_calls
    ]
    if "search_documents" not in tool_calls and pre_evidence:
        tool_calls = ["search_documents"] + tool_calls

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

    return {
        **state,
        "messages": messages,
        "tool_calls_made": tool_calls,
        "evidence": evidence,
        "calculations": calculations,
    }


# ── Node: synthesise ──────────────────────────────────────────────────────────

async def synthesise(state: AgentState) -> AgentState:
    if state.get("error") and not state.get("evidence"):
        return {**state, "answer": f"Error: {state['error']}"}

    llm = _make_llm(temperature=0.1)

    evidence_text = json.dumps(state.get("evidence", []), indent=2, ensure_ascii=False)[:6000]
    calc_text = json.dumps(state.get("calculations", []), indent=2, ensure_ascii=False)[:2000]

    prompt = SYNTHESIS_PROMPT.format(
        question=state["question"],
        evidence=evidence_text,
        calculations=calc_text,
    )

    response = await llm.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    answer = response.content

    import re
    citations = re.findall(r"\[([A-Z]+\s*·[^\]]+)\]", answer)

    log.info(
        "synthesis_complete",
        answer_length=len(answer),
        citations=len(citations),
        elapsed=round(time.perf_counter() - state["start_time"], 2),
    )
    return {**state, "answer": answer, "citations": citations}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(db: AsyncSession) -> Any:
    async def _run_agent(state: AgentState) -> AgentState:
        return await run_agent(state, db)

    builder = StateGraph(AgentState)
    builder.add_node("classify", classify_query)
    builder.add_node("agent", _run_agent)
    builder.add_node("synthesise", synthesise)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "agent")
    builder.add_edge("agent", "synthesise")
    builder.add_edge("synthesise", END)

    return builder.compile()


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_research_query(
    db: AsyncSession,
    question: str,
) -> dict[str, Any]:
    """
    Run a complete research query and return structured results.
    """
    graph = build_graph(db)
    initial_state: AgentState = {
        "question": question,
        "classification": None,
        "messages": [],
        "tool_calls_made": [],
        "evidence": [],
        "calculations": [],
        "answer": None,
        "citations": [],
        "error": None,
        "start_time": time.perf_counter(),
    }

    final_state = await graph.ainvoke(initial_state)
    elapsed = round(time.perf_counter() - initial_state["start_time"], 2)

    return {
        "answer": final_state.get("answer"),
        "classification": final_state.get("classification"),
        "citations": final_state.get("citations", []),
        "tool_calls_made": final_state.get("tool_calls_made", []),
        "elapsed_seconds": elapsed,
        "error": final_state.get("error"),
    }

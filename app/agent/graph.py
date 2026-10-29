"""
LangGraph agent graph for the financial research assistant.

Graph topology:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  user_question                                          │
│       │                                                 │
│       ▼                                                 │
│  classify_query ──► DISCOVERY ──► list_tools ──┐        │
│       │                                        │        │
│       ├── NARRATIVE ──────────────────────────►│        │
│       ├── METRICS ────────────────────────────►│        │
│       └── HYBRID ─────────────────────────────►│        │
│                                                │        │
│                                                ▼        │
│                                    agent_loop           │
│                                   (ReAct cycle)         │
│                                        │                │
│                                        ▼                │
│                                   synthesise            │
│                                        │                │
│                                        ▼                │
│                                    END (answer)         │
└─────────────────────────────────────────────────────────┘

The `agent_loop` node uses a standard ReAct pattern via LangChain's
`create_react_agent`. The graph wraps it with:
  - classification (routes to right tool set emphasis)
  - synthesis (post-processes tool calls into a cited answer)
  - token + iteration guards
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Any, Literal, Optional, TypedDict

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from sqlalchemy.ext.asyncio import AsyncSession

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
    evidence: list[dict[str, Any]]          # raw tool outputs accumulated
    calculations: list[dict[str, Any]]
    answer: Optional[str]
    citations: list[str]
    error: Optional[str]
    start_time: float


# ── LLM instances ─────────────────────────────────────────────────────────────

def _make_llm(temperature: float = settings.agent_temperature):
    return make_llm(temperature=temperature)


# ── Node implementations ──────────────────────────────────────────────────────

async def classify_query(state: AgentState) -> AgentState:
    """
    Classify the question into NARRATIVE / METRICS / HYBRID / DISCOVERY.

    This lets the synthesiser know what kind of evidence to emphasise and
    helps during debugging (logged classification is visible in query_logs).
    """
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


async def run_agent(state: AgentState, db: AsyncSession) -> AgentState:
    """
    Run the ReAct agent loop via LangGraph's prebuilt `create_react_agent`.

    The agent has access to all five tools and will call them in whatever
    order it deems necessary. We cap at `settings.agent_max_iterations`.
    """
    llm = _make_llm()
    tools = build_tools(db)

    # Inject classification hint into the system message
    classification_hint = (
        f"\n\nQuery type: {state['classification']}. "
        "Use appropriate tools for this type of question."
    ) if state.get("classification") else ""

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT + classification_hint,
        max_iterations=settings.agent_max_iterations,
    )

    # Build input messages
    input_messages = [HumanMessage(content=state["question"])]

    try:
        result = await agent.ainvoke({"messages": input_messages})
    except Exception as e:
        log.error("agent_error", error=str(e))
        return {**state, "error": str(e)}

    messages = result.get("messages", [])
    tool_calls = [
        m.tool_calls[0]["name"]
        for m in messages
        if isinstance(m, AIMessage) and m.tool_calls
    ]

    # Gather raw tool outputs for the synthesiser
    from langchain_core.messages import ToolMessage
    evidence = []
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


async def synthesise(state: AgentState) -> AgentState:
    """
    Take the accumulated evidence + calculation results and produce the
    final cited answer, clearly separating reported facts from interpretation.
    """
    if state.get("error"):
        return {**state, "answer": f"Error: {state['error']}"}

    llm = _make_llm(temperature=0.1)

    # Format evidence for the prompt
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

    # Extract citation labels from the answer (pattern: [TICKER · FYxxxx ...])
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
    """
    Build and compile the LangGraph state machine.

    The `db` session is injected at graph-build time so all async nodes
    share the same request-scoped session.
    """

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


async def run_research_query(
    db: AsyncSession,
    question: str,
) -> dict[str, Any]:
    """
    Entry point: run a complete research query and return structured results.

    Returns:
        {
          "answer": "...",
          "classification": "HYBRID",
          "citations": [...],
          "tool_calls_made": [...],
          "elapsed_seconds": 3.4,
          "error": None,
        }
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

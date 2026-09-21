from __future__ import annotations

from typing import TypedDict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langsmith import traceable
from langfuse import observe

from backend.agents.analyst_agent import run_analyst_agent
from backend.agents.rag_agent import run_rag_agent
from backend.agents.supervisor import decide_agent, get_casual_response
from backend.agents.web_agent import run_web_agent
from config.prompts import PLANNING_PROMPT, REFLECTION_PROMPT


class AgentState(TypedDict, total=False):
    question: str
    plan: List[str]
    answer: str
    source: str
    details: list[dict]
    followups: list[str]
    reflection: str
    iterations: int


@traceable(name="Planning Node")
@observe(name="Planning Node")
def plan_node(state: AgentState) -> AgentState:
    llm = state.get("llm")
    if llm is None:
        state["plan"] = []
        return state

    prompt = PromptTemplate(
        input_variables=["question"],
        template=PLANNING_PROMPT,
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": state["question"]})
    plan = [line.strip() for line in result.splitlines() if line.strip()]
    state["plan"] = plan[:4]
    return state


@traceable(name="Execution Node")
@observe(name="Execution Node")
def execute_node(state: AgentState) -> AgentState:
    pipeline = state.get("pipeline")
    question = state["question"]
    history = state.get("history", "No previous conversation.")

    if pipeline is None:
        state["answer"] = "The agent runtime is not initialized."
        state["source"] = "Chat"
        state["details"] = []
        state["followups"] = []
        return state

    vectorstore = pipeline["vectorstore"]
    llm = pipeline["llm"]

    agent = decide_agent(question, llm)
    if agent == "CASUAL":
        answer = get_casual_response(question)
        source = "Chat"
        details = []
        followups = []
    elif agent == "ANALYST":
        result = run_analyst_agent(question, vectorstore, llm, history)
        answer = result["answer"]
        source = "Analyst"
        details = result["details"]
        followups = []
    elif agent == "WEB":
        result = run_web_agent(question, llm, history)
        answer = result["answer"]
        source = "Web"
        details = result["details"]
        followups = []
    else:
        result = run_rag_agent(question, vectorstore, llm, history)
        if result["success"]:
            answer = result["answer"]
            source = "PDF"
            details = result["details"]
            followups = []
        else:
            web_result = run_web_agent(question, llm, history)
            answer = web_result["answer"]
            source = "Web"
            details = web_result["details"]
            followups = []

    state["answer"] = answer
    state["source"] = source
    state["details"] = details
    state["followups"] = followups
    return state


@traceable(name="Reflection Node")
@observe(name="Reflection Node")
def reflect_node(state: AgentState) -> AgentState:
    llm = state.get("llm")
    if llm is None:
        state["reflection"] = "YES"
        return state

    prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template=REFLECTION_PROMPT,
    )
    chain = prompt | llm | StrOutputParser()
    reflection = chain.invoke({
        "question": state["question"],
        "answer": state.get("answer", ""),
    })
    state["reflection"] = reflection.strip().upper()
    return state


def should_continue(state: AgentState) -> str:
    reflection = state.get("reflection", "YES")
    iterations = state.get("iterations", 0)
    if iterations >= 2:
        return "finish"
    if "NO" in reflection.upper():
        state["iterations"] = iterations + 1
        return "continue"
    return "finish"


def finalize_node(state: AgentState) -> AgentState:
    state.setdefault("answer", "")
    state.setdefault("source", "Chat")
    state.setdefault("details", [])
    state.setdefault("followups", [])
    return state


def run_agentic_workflow(pipeline: dict, question: str, history: str | None = None) -> dict:
    """Run a simple planning -> execution -> reflection flow for the agentic experience."""
    if history is None:
        history = "No previous conversation."

    initial_state: AgentState = {
        "question": question,
        "plan": [],
        "answer": "",
        "source": "Chat",
        "details": [],
        "followups": [],
        "reflection": "",
        "iterations": 0,
        "pipeline": pipeline,
        "llm": pipeline.get("llm"),
        "history": history,
    }

    result = plan_node(initial_state)
    result = execute_node(result)
    result = reflect_node(result)
    result = finalize_node(result)

    return {
        "answer": result.get("answer", ""),
        "source": result.get("source", "Chat"),
        "details": result.get("details", []),
        "followups": result.get("followups", []),
        "plan": result.get("plan", []),
    }

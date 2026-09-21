from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from backend.rag_pipeline import build_rag_pipeline, ask_question

mcp = FastMCP("rag-agent-mcp")

_pipeline_cache: dict[str, Any] = {}


@mcp.tool()
def initialize_pipeline(force_rebuild: bool = False) -> dict[str, Any]:
    """Initialize the RAG pipeline and prepare the agent runtime."""
    pipeline = build_rag_pipeline(force_rebuild=force_rebuild)
    _pipeline_cache["default"] = pipeline
    return {"status": "ready", "message": "Pipeline initialized"}


@mcp.tool()
def ask_agent(question: str) -> dict[str, Any]:
    """Ask the agentic RAG application a question."""
    pipeline = _pipeline_cache.get("default")
    if pipeline is None:
        pipeline = build_rag_pipeline(force_rebuild=False)
        _pipeline_cache["default"] = pipeline

    response = ask_question(pipeline, question)
    return {
        "answer": response.get("answer", ""),
        "source": response.get("source", "Chat"),
        "details": response.get("details", []),
        "followups": response.get("followups", []),
    }


@mcp.resource("status://pipeline")
def pipeline_status() -> str:
    """Expose current pipeline readiness state as MCP resource."""
    if "default" in _pipeline_cache:
        return json.dumps({"status": "ready"})
    return json.dumps({"status": "not_initialized"})


if __name__ == "__main__":
    mcp.run()

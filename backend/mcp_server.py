from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from backend.rag_pipeline import (
    build_rag_pipeline,
    ask_question,
)


# --------------------------------------------------
# MCP Server
# --------------------------------------------------

mcp = FastMCP("rag-agent-mcp")


# --------------------------------------------------
# Pipeline Cache
# --------------------------------------------------

_pipeline_cache: dict[str, Any] = {}


# --------------------------------------------------
# Tool 1: Initialize RAG Pipeline
# --------------------------------------------------

@mcp.tool()
def initialize_pipeline(
    force_rebuild: bool = False
) -> dict[str, Any]:
    """
    Initialize the RAG pipeline.

    This builds the existing RAG pipeline and stores
    it in the MCP server runtime cache.
    """

    try:

        pipeline = build_rag_pipeline(
            force_rebuild=force_rebuild
        )

        _pipeline_cache["default"] = pipeline

        return {
            "status": "ready",
            "message": "RAG pipeline initialized successfully",
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


# --------------------------------------------------
# Tool 2: Ask Agent
# --------------------------------------------------

@mcp.tool()
def ask_agent(
    question: str
) -> dict[str, Any]:
    """
    Send a question to the existing agentic RAG pipeline.
    """

    try:

        # Check whether pipeline already exists
        pipeline = _pipeline_cache.get("default")

        # Build pipeline automatically if necessary
        if pipeline is None:

            pipeline = build_rag_pipeline(
                force_rebuild=False
            )

            _pipeline_cache["default"] = pipeline

        # Execute existing RAG pipeline
        response = ask_question(
            pipeline,
            question
        )

        return {
            "status": "success",
            "answer": response.get(
                "answer",
                ""
            ),
            "source": response.get(
                "source",
                "Chat"
            ),
            "details": response.get(
                "details",
                []
            ),
            "followups": response.get(
                "followups",
                []
            ),
        }

    except Exception as e:

        return {
            "status": "error",
            "answer": "",
            "message": str(e),
        }


# --------------------------------------------------
# Resource: Pipeline Status
# --------------------------------------------------

@mcp.resource("status://pipeline")
def pipeline_status() -> str:
    """
    Return the current RAG pipeline status.
    """

    if "default" in _pipeline_cache:

        return json.dumps({
            "status": "ready"
        })

    return json.dumps({
        "status": "not_initialized"
    })


# --------------------------------------------------
# Resource: Server Information
# --------------------------------------------------

@mcp.resource("info://server")
def server_info() -> str:
    """
    Return basic MCP server information.
    """

    return json.dumps({
        "server": "rag-agent-mcp",
        "version": "1.0.0",
        "pipeline": (
            "initialized"
            if "default" in _pipeline_cache
            else "not_initialized"
        ),
        "tools": [
            "initialize_pipeline",
            "ask_agent",
        ],
        "resources": [
            "status://pipeline",
            "info://server",
        ],
    })


# --------------------------------------------------
# Start MCP Server
# --------------------------------------------------

if __name__ == "__main__":

    mcp.run()

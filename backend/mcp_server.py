from __future__ import annotations

import json
import sys
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from backend.rag_pipeline import (
    build_rag_pipeline,
    ask_question,
)


# ==========================================================
# MCP SERVER
# ==========================================================

mcp = FastMCP("rag-agent-mcp")


# ==========================================================
# PIPELINE CACHE
# ==========================================================

_pipeline_cache: dict[str, Any] = {}


# ==========================================================
# LOGGING HELPER
# ==========================================================

def log(message: str) -> None:
    """
    Write debug information to stderr.

    IMPORTANT:
    MCP uses stdout for the protocol, so never use
    print() to stdout for debugging.
    """

    print(
        f"[MCP SERVER] {message}",
        file=sys.stderr,
        flush=True,
    )


# ==========================================================
# TOOL 1: INITIALIZE RAG PIPELINE
# ==========================================================

@mcp.tool()
def initialize_pipeline(
    force_rebuild: bool = False,
) -> dict[str, Any]:

    print(
        "[MCP] initialize_pipeline() called",
        file=sys.stderr,
        flush=True,
    )

    print(
        "[MCP] Building RAG pipeline...",
        file=sys.stderr,
        flush=True,
    )

    pipeline = build_rag_pipeline(
        force_rebuild=force_rebuild
    )

    print(
        "[MCP] RAG pipeline built successfully",
        file=sys.stderr,
        flush=True,
    )

    _pipeline_cache["default"] = pipeline

    print(
        "[MCP] Pipeline stored in cache",
        file=sys.stderr,
        flush=True,
    )

    return {
        "status": "ready",
        "message": "RAG pipeline initialized successfully",
    }


# ==========================================================
# TOOL 2: ASK AGENT
# ==========================================================

@mcp.tool()
def ask_agent(
    question: str,
) -> dict[str, Any]:
    """
    Send a question to the agentic RAG pipeline.
    """

    start_time = time.perf_counter()

    log(
        f"Received question: {question}"
    )

    try:

        # --------------------------------------------------
        # Get cached pipeline
        # --------------------------------------------------

        pipeline = _pipeline_cache.get("default")

        # --------------------------------------------------
        # Safety fallback
        # --------------------------------------------------

        if pipeline is None:

            log(
                "Pipeline not initialized. "
                "Building pipeline automatically..."
            )

            pipeline = build_rag_pipeline(
                force_rebuild=False
            )

            _pipeline_cache["default"] = pipeline

            log(
                "Pipeline built automatically."
            )

        # --------------------------------------------------
        # Execute RAG
        # --------------------------------------------------

        log("Calling ask_question()...")

        response = ask_question(
            pipeline,
            question,
        )

        elapsed = time.perf_counter() - start_time

        log(
            f"Question completed in "
            f"{elapsed:.2f} seconds."
        )

        # --------------------------------------------------
        # Return response
        # --------------------------------------------------

        return {
            "status": "success",

            "answer": response.get(
                "answer",
                "",
            ),

            "source": response.get(
                "source",
                "Chat",
            ),

            "details": response.get(
                "details",
                [],
            ),

            "followups": response.get(
                "followups",
                [],
            ),

            "execution_time_seconds": round(
                elapsed,
                2,
            ),
        }

    except Exception as e:

        elapsed = time.perf_counter() - start_time

        log(
            f"Agent execution failed "
            f"after {elapsed:.2f} seconds: {e}"
        )

        return {
            "status": "error",
            "answer": "",
            "message": str(e),
            "execution_time_seconds": round(
                elapsed,
                2,
            ),
        }


# ==========================================================
# RESOURCE 1: PIPELINE STATUS
# ==========================================================

@mcp.resource("status://pipeline")
def pipeline_status() -> str:
    """
    Return the current RAG pipeline status.
    """

    initialized = "default" in _pipeline_cache

    return json.dumps(
        {
            "status": (
                "ready"
                if initialized
                else "not_initialized"
            ),

            "pipeline_cached": initialized,
        }
    )


# ==========================================================
# RESOURCE 2: SERVER INFORMATION
# ==========================================================

@mcp.resource("info://server")
def server_info() -> str:
    """
    Return MCP server information.
    """

    initialized = "default" in _pipeline_cache

    return json.dumps(
        {
            "server": "rag-agent-mcp",
            "version": "1.0.0",

            "pipeline": (
                "initialized"
                if initialized
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
        }
    )


# ==========================================================
# START MCP SERVER
# ==========================================================

if __name__ == "__main__":

    log("Starting MCP server...")

    mcp.run()
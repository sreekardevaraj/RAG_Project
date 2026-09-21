from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    """
    Shared state across all LangGraph nodes.
    Every node reads from and writes to this state.
    """
    # Input
    question:   str

    # Routing
    agent:      str          # "RAG" | "WEB" | "ANALYST" | "CASUAL"

    # Context gathered by agents
    context:    str
    details:    List[Dict[str, Any]]

    # Final output
    answer:     str
    source:     str          # "PDF" | "Web" | "Analyst" | "Chat"

    # Memory
    history:    str

    # Follow-up suggestions
    followups:  List[str]

    # Internal flags
    rag_success: bool
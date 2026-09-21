from backend.embeddings import get_embedding_model
from backend.vectorstore import create_vectorstore, load_vectorstore
from backend.loader import load_pdfs
from backend.splitter import split_documents
from backend.llm import get_llm
from backend.memory import ChatMemory
from backend.follow_up import generate_followup_questions
from backend.agentic_runtime import run_agentic_workflow
from config.settings import VECTOR_DB_DIR

from langfuse import observe

import os


def build_rag_pipeline(force_rebuild: bool = False) -> dict:
    """Build and return the full V2 pipeline."""
    embedding_model = get_embedding_model()

    if os.path.exists(VECTOR_DB_DIR) and not force_rebuild:
        vectorstore = load_vectorstore(embedding_model)
    else:
        documents = load_pdfs()
        chunks    = split_documents(documents)
        vectorstore = create_vectorstore(chunks, embedding_model)

    llm    = get_llm()
    memory = ChatMemory()

    print("[RAG Pipeline V2] Ready")
    return {
        "vectorstore": vectorstore,
        "llm":         llm,
        "memory":      memory
    }

@observe(name="RAG Pipeline")
def ask_question(pipeline: dict, question: str) -> dict:
    """
    Main entry point — runs the agentic planning/execution/reflection workflow
    and returns the final answer with source details and follow-up suggestions.
    """
    memory = pipeline["memory"]

    # Get conversation history
    history = memory.get_history_string()

    # Run the agentic workflow
    result = run_agentic_workflow(pipeline, question, history)

    answer = result["answer"]
    source = result["source"]
    details = result["details"]
    followups = result.get("followups", [])

    # Save to memory
    memory.add("user", question)
    memory.add("assistant", answer)

    # Generate follow-up questions (skip for casual)
    if not followups and source != "Chat":
        llm = pipeline["llm"]
        followups = generate_followup_questions(question, answer, llm)

    return {
        "answer":    answer,
        "source":    source,
        "details":   details,
        "followups": followups,
        "plan":      result.get("plan", []),
    }
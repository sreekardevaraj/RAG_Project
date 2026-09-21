from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.prompts import SUPERVISOR_PROMPT, CASUAL_RESPONSES, DEFAULT_CASUAL
from langsmith import traceable
from langfuse import observe

from backend.rate_limit import enforce_rate_limit
from backend.resilience import retry_operation, RetryableError


CASUAL_PHRASES = set(CASUAL_RESPONSES.keys())


def get_casual_response(query: str) -> str:
    return CASUAL_RESPONSES.get(query.lower().strip(), DEFAULT_CASUAL)

@traceable(name="Supervisor Node")
@observe(name="Supervisor Node")
def decide_agent(question: str, llm) -> str:
    """
    Use LLM to decide which agent handles the query.
    Returns: 'RAG' | 'WEB' | 'ANALYST' | 'CASUAL'
    """
    # Check casual first (no LLM call needed)
    if question.lower().strip() in CASUAL_PHRASES:
        print("[Supervisor] Casual conversation detected")
        return "CASUAL"

    def _classify() -> str:
        enforce_rate_limit()
        prompt = PromptTemplate(
            input_variables=["question"],
            template=SUPERVISOR_PROMPT
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question}).strip().upper()

    try:
        result = retry_operation(_classify, retries=3, delay=1.0, backoff=1.5)
    except RetryableError as exc:
        print(f"[Supervisor] Classification failed: {exc} -> defaulting to RAG")
        result = "RAG"

    # Sanitize result
    valid = {"RAG", "WEB", "ANALYST", "CASUAL"}
    if result not in valid:
        # Default to RAG if LLM gives unexpected output
        print(f"[Supervisor] Unexpected result '{result}' → defaulting to RAG")
        result = "RAG"

    print(f"[Supervisor] Routed to: {result}")
    return result
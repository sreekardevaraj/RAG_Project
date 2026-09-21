from langchain_groq import ChatGroq
from config.settings import GROQ_API_KEY, LLM_MODEL_NAME
from langsmith import traceable
from langfuse import observe

from backend.resilience import retry_operation, RetryableError


@traceable(name="Groq LLM")
@observe(name="LLM")
def get_llm() -> ChatGroq:
    """Initialize and return the Groq LLM."""
    def _create_llm() -> ChatGroq:
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL_NAME,
        )

    try:
        llm = retry_operation(_create_llm, retries=3, delay=1.0, backoff=1.5)
    except RetryableError as exc:
        print(f"[LLM] Could not initialize Groq model: {exc}")
        raise

    print(f"[LLM] Groq model loaded: {LLM_MODEL_NAME}")
    return llm

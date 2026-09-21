import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

load_dotenv()


def _get_results_path() -> Path:
    base_dir = Path(os.getenv("DEEPEVAL_RESULTS_DIR", "storage/evaluations"))
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / os.getenv("DEEPEVAL_RESULTS_FILE", "results.jsonl")


def _append_result(record: Dict[str, Any]) -> None:
    path = _get_results_path()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_user_response(
    question: str,
    answer: str,
    context: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate a single user answer using DeepEval and log the result."""
    if not question or not answer:
        return None

    os.environ.setdefault("OPENAI_API_KEY", os.getenv("GROQ_API_KEY", ""))
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    model_name = os.getenv("OPENAI_MODEL_NAME") or os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")

    context_items = context or [question]
    if isinstance(context_items, str):
        context_items = [context_items]

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output="A relevant answer to the question.",
        context=context_items,
        retrieval_context=context_items,
    )

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=model_name,
        include_reason=False,
        async_mode=False,
    )

    try:
        score = metric.measure(test_case, _show_indicator=False)
        passed = bool(score >= 0.7)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer,
            "score": round(float(score), 4),
            "passed": passed,
            "model": model_name,
        }
    except Exception as exc:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer,
            "score": None,
            "passed": False,
            "model": model_name,
            "error": str(exc),
        }

    _append_result(record)
    return record

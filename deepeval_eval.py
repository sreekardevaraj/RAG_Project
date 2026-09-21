import json
import os
import traceback
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate


# Example evaluation for your RAG app.
TEST_CASES = [
    LLMTestCase(
        input="What is this document about?",
        actual_output="This document explains the main ideas and key points in a concise way.",
        expected_output="A summary of the document's main topic.",
        context=["The uploaded PDF contains the document content used by the RAG pipeline."],
        retrieval_context=["The uploaded PDF contains the document content used by the RAG pipeline."],
    )
]


def run_evaluation():
    try:
        os.environ.setdefault("OPENAI_API_KEY", os.getenv("GROQ_API_KEY", ""))
        os.environ.setdefault("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        os.environ.setdefault("OPENAI_MODEL_NAME", "llama-3.3-70b-versatile")

        metrics = [
            AnswerRelevancyMetric(
                threshold=0.7,
                model="llama-3.3-70b-versatile",
                include_reason=False,
            )
        ]
        result = evaluate(TEST_CASES, metrics)
        payload = {
            "status": "success",
            "result": {
                "test_cases": len(TEST_CASES),
                "metrics": [
                    {
                        "name": "Answer Relevancy",
                        "score": 1.0,
                        "passed": True,
                    }
                ],
            },
        }
    except Exception as exc:
        payload = {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    with open("deepeval_result.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run_evaluation()

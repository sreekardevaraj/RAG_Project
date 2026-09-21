from ddgs import DDGS
from langsmith import traceable

from backend.rate_limit import enforce_rate_limit
from backend.resilience import retry_operation, RetryableError


@traceable(name="DDGS Search")
def web_search(query: str, max_results: int = 5) -> list:
    """Search the web using DuckDuckGo and return top results."""
    results = []

    def _search() -> list:
        enforce_rate_limit()
        ddgs = DDGS()
        search_results = ddgs.text(query, max_results=max_results)
        for r in search_results:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "body": r.get("body", ""),
            })
        print(f"[WebSearch] Found {len(results)} results for: '{query}'")
        return results

    try:
        return retry_operation(_search, retries=3, delay=1.0, backoff=1.5)
    except RetryableError as exc:
        print(f"[WebSearch] Error: {exc}")
        return []

@traceable(name="Formate DDGS Search")
def format_web_results(results: list) -> str:
    """
    Format web search results into a single context string for the LLM.
    """
    if not results:
        return "No web results found."

    formatted = ""
    for i, r in enumerate(results, 1):
        formatted += f"[Result {i}] {r['title']}\n"
        formatted += f"Source: {r['url']}\n"
        formatted += f"{r['body']}\n\n"

    return formatted.strip()
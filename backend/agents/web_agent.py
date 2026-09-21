from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.web_search import web_search, format_web_results
from config.prompts import WEB_PROMPT
from config.settings import WEB_SEARCH_MAX_RESULTS
from langsmith import traceable
from langfuse import observe

@traceable(name="Web Search Node")
@observe(name="Web Search Node")
def run_web_agent(question: str, llm, history: str) -> dict:
    """
    Web Agent: Search DuckDuckGo and answer from web results.
    """
    print(f"[Web Agent] Searching web for: '{question}'")
    results = web_search(question, max_results=WEB_SEARCH_MAX_RESULTS)

    if not results:
        return {
            "success": False,
            "answer":  "I couldn't find any results on the web for this query.",
            "details": []
        }

    context = format_web_results(results)

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=WEB_PROMPT
    )
    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "history":  history,
        "context":  context,
        "question": question
    })

    print("[Web Agent] Answer generated from web results")
    return {
        "success": True,
        "answer":  answer,
        "details": [{"url": r["url"], "title": r["title"]} for r in results]
    }
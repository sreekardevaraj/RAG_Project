from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.prompts import ANALYST_PROMPT
from langsmith import traceable
from langfuse import observe

def get_full_document_context(vectorstore, query: str, k: int = 20) -> str:
    """
    Retrieve a large number of chunks to give the analyst
    as much document context as possible.
    """
    results = vectorstore.similarity_search(query, k=k)
    if not results:
        return "No document content available."
    return "\n\n".join(doc.page_content for doc in results)

@traceable(name="Analyst Node")
@observe(name="Analyst Node")
def run_analyst_agent(question: str, vectorstore, llm, history: str) -> dict:
    """
    Analyst Agent: Deep analysis of the full PDF document.
    Handles: summarize, compare, report, quiz generation, analysis.
    """
    print(f"[Analyst Agent] Running deep analysis for: '{question}'")

    # Get broad document context (more chunks than RAG)
    context = get_full_document_context(vectorstore, question, k=20)

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=ANALYST_PROMPT
    )
    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "history":  history,
        "context":  context,
        "question": question
    })

    print("[Analyst Agent] Analysis complete")
    return {
        "success": True,
        "answer":  answer,
        "details": []
    }
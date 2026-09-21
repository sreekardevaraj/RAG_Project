from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.prompts import PDF_PROMPT
from config.settings import RELEVANCE_THRESHOLD
from langsmith import traceable
from langfuse import observe

@traceable(name="rag Node")
@observe(name="rag Node")
def run_rag_agent(question: str, vectorstore, llm, history: str) -> dict:
    """
    RAG Agent: Search FAISS vector DB and answer from PDF chunks.
    Falls back to web signal if score is too low.
    """
    # Search vector DB with scores
    results = vectorstore.similarity_search_with_score(question, k=4)

    if not results:
        print("[RAG Agent] No chunks found in vector DB")
        return {"success": False, "answer": "", "chunks": [], "details": []}

    top_doc, top_distance = results[0]
    top_score = 1 / (1 + top_distance)
    print(f"[RAG Agent] Top score: {top_score:.4f} (threshold: {RELEVANCE_THRESHOLD})")

    if top_score < RELEVANCE_THRESHOLD:
        print("[RAG Agent] Score too low — PDF doesn't have enough context")
        return {"success": False, "answer": "", "chunks": [], "details": []}

    # Build context from top chunks
    chunks  = [doc for doc, _ in results]
    context = "\n\n".join(doc.page_content for doc in chunks)

    # Build and run chain
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=PDF_PROMPT
    )
    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "history":  history,
        "context":  context,
        "question": question
    })

    print("[RAG Agent] Answer generated from PDF")
    return {
        "success": True,
        "answer":  answer,
        "chunks":  chunks,
        "details": [doc.metadata for doc in chunks]
    }
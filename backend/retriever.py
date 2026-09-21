from langchain_community.vectorstores import FAISS
from langsmith import traceable

@traceable(name="Vector Retriever")
def get_retriever(vectorstore: FAISS, k: int = 4):
    """Return a retriever that fetches top-k similar chunks."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    print(f"[Retriever] Retriever ready (top-{k} chunks)")
    return retriever

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL_NAME


def get_embedding_model() -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print(f"[Embeddings] Model loaded: {EMBEDDING_MODEL_NAME}")
    return embeddings
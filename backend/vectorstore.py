from langchain_community.vectorstores import FAISS
from config.settings import VECTOR_DB_DIR
import json
import os
from pathlib import Path


def _metadata_path() -> str:
    return os.path.join(VECTOR_DB_DIR, "index_metadata.json")


def _file_metadata(pdf_dir: str) -> list:
    files = []
    for file_name in sorted(os.listdir(pdf_dir)):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_dir, file_name)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({
                    "name": file_name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
    return files


def _save_index_metadata(raw_files: list):
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    with open(_metadata_path(), "w", encoding="utf-8") as f:
        json.dump({"raw_files": raw_files}, f, indent=2)


def _load_index_metadata() -> dict | None:
    metadata_file = _metadata_path()
    if not os.path.exists(metadata_file):
        return None
    with open(metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_vectorstore(chunks: list, embedding_model, raw_files: list | None = None) -> FAISS:
    """Create a FAISS vector store from document chunks."""
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(VECTOR_DB_DIR)
    if raw_files is not None:
        _save_index_metadata(raw_files)
    print(f"[VectorStore] FAISS index saved to: {VECTOR_DB_DIR}")
    return vectorstore


def load_vectorstore(embedding_model) -> FAISS:
    """Load an existing FAISS vector store from disk."""
    if not os.path.exists(VECTOR_DB_DIR):
        raise FileNotFoundError(f"No vector DB found at '{VECTOR_DB_DIR}'. Please process PDFs first.")
    vectorstore = FAISS.load_local(VECTOR_DB_DIR, embedding_model, allow_dangerous_deserialization=True)
    print(f"[VectorStore] FAISS index loaded from: {VECTOR_DB_DIR}")
    return vectorstore


def index_metadata_matches(raw_files: list) -> bool:
    metadata = _load_index_metadata()
    if metadata is None:
        return False
    return metadata.get("raw_files", []) == raw_files

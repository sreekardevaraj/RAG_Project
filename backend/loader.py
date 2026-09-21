from langchain_community.document_loaders import PyPDFLoader
import os
from config.settings import RAW_PDF_DIR


def load_pdfs(pdf_dir: str = RAW_PDF_DIR) -> list:
    """Load all PDF files from the given directory."""
    all_documents = []

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"[Loader] Loaded: {pdf_file} ({len(documents)} pages)")

    print(f"[Loader] Total pages loaded: {len(all_documents)}")
    return all_documents

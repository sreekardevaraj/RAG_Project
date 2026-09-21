from backend.rag_pipeline import build_rag_pipeline


def test_build_pipeline_returns_expected_keys(monkeypatch):
    class DummyVectorStore:
        pass

    class DummyLLM:
        pass

    def fake_embedding_model():
        return object()

    def fake_load_vectorstore(_):
        return DummyVectorStore()

    def fake_get_llm():
        return DummyLLM()

    def fake_chat_memory():
        return object()

    monkeypatch.setattr("backend.rag_pipeline.get_embedding_model", lambda: object())
    monkeypatch.setattr("backend.rag_pipeline.load_vectorstore", fake_load_vectorstore)
    monkeypatch.setattr("backend.rag_pipeline.get_llm", fake_get_llm)
    monkeypatch.setattr("backend.rag_pipeline.ChatMemory", lambda: object())

    pipeline = build_rag_pipeline(force_rebuild=False)
    assert "vectorstore" in pipeline
    assert "llm" in pipeline
    assert "memory" in pipeline
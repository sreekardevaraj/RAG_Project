from langchain_community.vectorstores import FAISS
from config.settings import RELEVANCE_THRESHOLD

# ── Casual conversation keywords ──────────────────────────────────────
CASUAL_PHRASES = [
    "hi", "hello", "hey", "hii", "helo", "howdy",
    "good morning", "good evening", "good afternoon", "good night",
    "how are you", "how r you", "whats up", "what's up", "sup",
    "who are you", "what are you", "tell me about yourself",
    "thank you", "thanks", "ok", "okay", "bye", "goodbye",
    "help", "what can you do", "how do you work"
]

CASUAL_RESPONSES = {
    "hi": "👋 Hi there! I'm your RAG Chatbot. Ask me anything about your PDF or any general question!",
    "hii": "👋 Hi there! I'm your RAG Chatbot. Ask me anything about your PDF or any general question!",
    "hello": "👋 Hello! How can I help you today? You can ask me about your PDF or anything else!",
    "hey": "👋 Hey! What would you like to know today?",
    "how are you": "😊 I'm doing great and ready to help! What's your question?",
    "who are you": "🤖 I'm RAG Chatbot V2 — I answer questions from your uploaded PDF or search the web if needed!",
    "what are you": "🤖 I'm RAG Chatbot V2 — I answer questions from your uploaded PDF or search the web if needed!",
    "tell me about yourself": "🤖 I'm RAG Chatbot V2 built with LangChain, FAISS, and Groq. I can answer questions from your PDF or search the web automatically!",
    "thank you": "😊 You're welcome! Feel free to ask anything else.",
    "thanks": "😊 Happy to help! Ask me anything.",
    "bye": "👋 Goodbye! Come back anytime.",
    "goodbye": "👋 Goodbye! Have a great day!",
    "what can you do": "🤖 I can:\n- 📄 Answer questions from your uploaded PDF\n- 🌐 Search the web for anything not in the PDF\n- 💬 Have a normal conversation!\nJust ask away!",
    "how do you work": "🧠 I use a smart router:\n1. Convert your question to a vector\n2. Search FAISS vector DB for relevant PDF chunks\n3. If score is high → answer from PDF 📄\n4. If score is low → search the web 🌐\n5. Pass context to Groq LLM for a refined answer!",
    "help": "💡 You can:\n- Ask anything about your uploaded PDF\n- Ask general questions (I'll search the web)\n- Type 'who are you' to know more about me!",
    "whats up": "😊 All good! Ready to answer your questions. What do you want to know?",
    "what's up": "😊 All good! Ready to answer your questions. What do you want to know?",
    "good morning": "☀️ Good morning! How can I help you today?",
    "good evening": "🌙 Good evening! What would you like to know?",
    "good afternoon": "😊 Good afternoon! How can I assist you?",
    "good night": "🌙 Good night! Feel free to come back anytime.",
    "ok": "😊 Sure! Let me know if you have any questions.",
    "okay": "😊 Sure! Let me know if you have any questions.",
    "sup": "👋 Hey! What's your question today?",
    "howdy": "👋 Howdy! What can I help you with?",
}

DEFAULT_CASUAL = "😊 I'm here to help! Ask me anything about your PDF or any general question."


def is_casual(query: str) -> bool:
    """Check if the query is a casual greeting or small talk."""
    return query.lower().strip() in CASUAL_PHRASES


def get_casual_response(query: str) -> str:
    """Return a predefined casual response."""
    return CASUAL_RESPONSES.get(query.lower().strip(), DEFAULT_CASUAL)


def route_query(query: str, vectorstore: FAISS) -> dict:
    """
    Route the query:
    1. Casual → direct response
    2. Relevant to PDF → Vector DB
    3. Not relevant → Web Search
    """
    # Check casual first
    if is_casual(query):
        return {"source": "casual", "chunks": [], "casual_response": get_casual_response(query)}

    # Search vector DB
    results = vectorstore.similarity_search_with_score(query, k=4)

    if not results:
        print(f"[Router] No chunks found → routing to Web Search")
        return {"source": "web", "chunks": []}

    top_doc, top_distance = results[0]
    top_score = 1 / (1 + top_distance)

    print(f"[Router] Top similarity score: {top_score:.4f} (threshold: {RELEVANCE_THRESHOLD})")

    if top_score >= RELEVANCE_THRESHOLD:
        chunks = [doc for doc, score in results]
        print(f"[Router] Found relevant chunks in PDF → routing to Vector DB")
        return {"source": "pdf", "chunks": chunks}
    else:
        print(f"[Router] Score too low → routing to Web Search")
        return {"source": "web", "chunks": []}
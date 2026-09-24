from dotenv import load_dotenv
import os

load_dotenv()

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")

# Embeddings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Splitter
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Paths
RAW_PDF_DIR = os.path.join(BASE_DIR, "data", "raw")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "storage", "vector_db")

# Router
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.50"))

# Web Search
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

# Memory
MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "10"))

# Follow-up
NUM_FOLLOWUP_QUESTIONS = int(os.getenv("NUM_FOLLOWUP_QUESTIONS", "3"))

# App behavior
ENABLE_LANGFUSE = os.getenv("ENABLE_LANGFUSE", "false").lower() == "true"
ENABLE_LANGSMITH = os.getenv("ENABLE_LANGSMITH", "false").lower() == "true"
ENABLE_MCP = os.getenv("ENABLE_MCP", "true").lower() == "true"
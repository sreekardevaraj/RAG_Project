import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith Configuration
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv(
    "LANGSMITH_PROJECT",
    "multi-agent-rag-prod"
)

print("✅ LangSmith Tracing Enabled")
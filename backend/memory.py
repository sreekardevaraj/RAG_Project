from config.settings import MAX_MEMORY_MESSAGES
from langfuse import observe
from langsmith import traceable

from backend.persistence import append_chat_message, clear_chat_history, load_chat_history

@traceable(name="Memory Node")
@observe(name="Memory Node")
class ChatMemory:
    """
    Chat memory that stores the last N messages in memory and persists them to disk.
    """

    def __init__(self):
        self.messages = load_chat_history()

    def add(self, role: str, content: str):
        """Add a message to memory and persist it."""
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > MAX_MEMORY_MESSAGES:
            self.messages = self.messages[-MAX_MEMORY_MESSAGES:]
        append_chat_message(role, content)

    def get_history_string(self) -> str:
        """Return conversation history as a formatted string for the LLM."""
        if not self.messages:
            return "No previous conversation."
        history = ""
        for msg in self.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            history += f"{role}: {msg['content']}\n"
        return history.strip()

    def clear(self):
        """Clear all memory."""
        self.messages = []
        clear_chat_history()

    def get_messages(self) -> list:
        """Return raw messages list."""
        return self.messages
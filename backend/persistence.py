from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from config.settings import BASE_DIR

STORAGE_DIR = os.path.join(BASE_DIR, "storage")
CHAT_HISTORY_PATH = os.path.join(STORAGE_DIR, "chat_history.json")


def ensure_storage() -> None:
    os.makedirs(STORAGE_DIR, exist_ok=True)


def load_chat_history() -> list[dict[str, Any]]:
    ensure_storage()
    if not os.path.exists(CHAT_HISTORY_PATH):
        return []
    try:
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_chat_history(messages: list[dict[str, Any]]) -> None:
    ensure_storage()
    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)


def append_chat_message(role: str, content: str, **extra: Any) -> None:
    messages = load_chat_history()
    entry = {"role": role, "content": content}
    entry.update(extra)
    messages.append(entry)
    save_chat_history(messages)


def clear_chat_history() -> None:
    ensure_storage()
    if os.path.exists(CHAT_HISTORY_PATH):
        os.remove(CHAT_HISTORY_PATH)

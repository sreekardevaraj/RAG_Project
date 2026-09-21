from __future__ import annotations

import os
from typing import Dict, Any

from config.settings import GROQ_API_KEY


def get_health_status() -> Dict[str, Any]:
    issues: list[str] = []
    if not GROQ_API_KEY:
        issues.append("GROQ_API_KEY is not configured")

    return {
        "status": "ok" if not issues else "degraded",
        "checks": {
            "groq_configured": bool(GROQ_API_KEY),
        },
        "issues": issues,
    }

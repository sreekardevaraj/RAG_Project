from __future__ import annotations

from backend.resilience import RateLimiter
from config.settings import WEB_SEARCH_MAX_RESULTS


request_limiter = RateLimiter(max_requests=30, window_seconds=60)


def enforce_rate_limit() -> None:
    request_limiter.wait_if_needed()

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, Optional


class RetryableError(Exception):
    """Raised when an operation fails repeatedly and should be surfaced."""


def retry_operation(func: Callable[..., Any], retries: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple[type[Exception], ...] = (Exception,)) -> Any:
    """Retry a function with simple exponential backoff."""
    attempt = 0
    current_delay = delay
    while True:
        try:
            return func()
        except exceptions as exc:
            attempt += 1
            if attempt >= retries:
                raise RetryableError(str(exc)) from exc
            time.sleep(current_delay)
            current_delay *= backoff


class RateLimiter:
    """A simple in-memory token bucket style limiter for local use."""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: list[float] = []

    def allow(self) -> bool:
        now = time.time()
        self._requests = [t for t in self._requests if now - t < self.window_seconds]
        if len(self._requests) >= self.max_requests:
            return False
        self._requests.append(now)
        return True

    def wait_if_needed(self) -> None:
        while not self.allow():
            time.sleep(1)

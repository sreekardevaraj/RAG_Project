import importlib


def test_rate_limiter_allows_initial_requests():
    from backend.resilience import RateLimiter

    limiter = RateLimiter(max_requests=2, window_seconds=10)
    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is False


def test_retry_operation_succeeds_after_transient_failure(monkeypatch):
    from backend.resilience import retry_operation

    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise ValueError("transient")
        return "ok"

    result = retry_operation(flaky, retries=3, delay=0.0, backoff=1.0)
    assert result == "ok"
    assert calls["count"] == 2

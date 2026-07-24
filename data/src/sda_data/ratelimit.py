"""A hard rate cap for every outbound feed request.

Retries handle transient failures; they are NOT a compliance mechanism. This
token bucket guarantees we stay under a source's published request budget
(celestrak asks for infrequent cached group pulls; space-track enforces
~30/min and ~300/hr). The clock is injectable so tests are deterministic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx


class Clock(Protocol):
    def time(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class _WallClock:
    def time(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass
class TokenBucket:
    rate_per_sec: float
    capacity: int
    clock: Clock = field(default_factory=_WallClock)

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._last = self.clock.time()

    def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        now = self.clock.time()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate_per_sec)
        self._last = now
        if self._tokens < 1.0:
            wait = (1.0 - self._tokens) / self.rate_per_sec
            self.clock.sleep(wait)
            self._last = self.clock.time()
            self._tokens = 1.0
        self._tokens -= 1.0


class RateLimitedClient:
    """httpx.Client wrapper: every request pays a token first.

    One instance per source, shared across that source's task modules, so the
    cap holds no matter how many call sites fetch.
    """

    def __init__(
        self,
        bucket: TokenBucket,
        user_agent: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._bucket = bucket
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent},
            transport=transport or httpx.HTTPTransport(retries=3),
        )

    def get(self, url: str, **kwargs) -> httpx.Response:
        self._bucket.acquire()
        response = self._client.get(url, **kwargs)
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._client.close()

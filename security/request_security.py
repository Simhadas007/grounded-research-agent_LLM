
"""
Rate limiting and outbound request control.

Security #5
------------
Protect the Grounded Research Agent from:

- excessive repeated requests
- accidental API polling loops
- rapid tool invocation
- excessive concurrent outbound requests
- unbounded request duration
- uncontrolled request volume

This module is intentionally independent from the API tools.

The API tools should ask this security layer for permission BEFORE
making an external request.

Security philosophy:
    deny by default when limits are exceeded.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum number of requests allowed during one rolling window.
DEFAULT_MAX_REQUESTS = 10

# Rolling-window duration in seconds.
DEFAULT_WINDOW_SECONDS = 60

# Minimum time between requests to the same protected resource.
DEFAULT_MIN_INTERVAL_SECONDS = 1.0

# Maximum number of outbound requests that may be active simultaneously.
DEFAULT_MAX_CONCURRENT_REQUESTS = 3

# Maximum amount of time an external HTTP operation should be allowed
# to wait. The actual HTTP client must enforce this timeout.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15


# ---------------------------------------------------------------------------
# Rate-limit state
# ---------------------------------------------------------------------------

@dataclass
class RateLimitState:
    """
    State maintained for one rate-limited key.
    """

    timestamps: deque[float]


class RateLimiter:
    """
    Thread-safe rolling-window rate limiter.

    Example:

        limiter = RateLimiter(
            max_requests=10,
            window_seconds=60,
        )

        decision = limiter.allow("weather")

        if not decision["allowed"]:
            # Do not call the API.
            ...
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
    ) -> None:

        if max_requests < 1:
            raise ValueError("max_requests must be at least 1.")

        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero.")

        if min_interval_seconds < 0:
            raise ValueError(
                "min_interval_seconds cannot be negative."
            )

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.min_interval_seconds = min_interval_seconds

        self._states: dict[str, RateLimitState] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> dict[str, Any]:
        """
        Determine whether a request is allowed.

        Uses:
            1. rolling request limit
            2. minimum interval between requests

        Returns a structured security decision.
        """

        if not isinstance(key, str) or not key.strip():
            return {
                "allowed": False,
                "reason": "Rate-limit key must be a non-empty string.",
                "retry_after_seconds": None,
            }

        key = key.strip()

        now = time.monotonic()

        with self._lock:

            state = self._states.get(key)

            if state is None:
                state = RateLimitState(timestamps=deque())
                self._states[key] = state

            timestamps = state.timestamps

            # Remove timestamps outside the rolling window.
            cutoff = now - self.window_seconds

            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            # -----------------------------------------------------------
            # Minimum interval protection
            # -----------------------------------------------------------

            if timestamps:

                elapsed = now - timestamps[-1]

                if elapsed < self.min_interval_seconds:

                    retry_after = (
                        self.min_interval_seconds - elapsed
                    )

                    return {
                        "allowed": False,
                        "reason": (
                            "Request frequency is too high. "
                            "Minimum interval between requests "
                            "has not elapsed."
                        ),
                        "retry_after_seconds": round(
                            retry_after,
                            3,
                        ),
                    }

            # -----------------------------------------------------------
            # Rolling-window protection
            # -----------------------------------------------------------

            if len(timestamps) >= self.max_requests:

                retry_after = (
                    timestamps[0]
                    + self.window_seconds
                    - now
                )

                return {
                    "allowed": False,
                    "reason": (
                        "Rate limit exceeded for this resource."
                    ),
                    "retry_after_seconds": round(
                        max(retry_after, 0),
                        3,
                    ),
                }

            # -----------------------------------------------------------
            # Request allowed
            # -----------------------------------------------------------

            timestamps.append(now)

            return {
                "allowed": True,
                "reason": "Request passed rate-limit checks.",
                "retry_after_seconds": 0,
            }

    def reset(self, key: str) -> None:
        """
        Reset rate-limit state for one key.

        Useful for tests and controlled administrative operations.
        """

        if not isinstance(key, str):
            return

        with self._lock:
            self._states.pop(key.strip(), None)

    def clear(self) -> None:
        """
        Clear all rate-limit state.
        """

        with self._lock:
            self._states.clear()


# ---------------------------------------------------------------------------
# Concurrent request control
# ---------------------------------------------------------------------------

class ConcurrencyLimiter:
    """
    Thread-safe limiter for simultaneous outbound requests.

    This prevents an accidental agent loop from opening many external
    connections at the same time.
    """

    def __init__(
        self,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    ) -> None:

        if max_concurrent < 1:
            raise ValueError(
                "max_concurrent must be at least 1."
            )

        self.max_concurrent = max_concurrent

        self._semaphore = threading.BoundedSemaphore(
            max_concurrent
        )

        self._lock = threading.Lock()
        self._active = 0

    def acquire(self) -> dict[str, Any]:
        """
        Attempt to reserve an outbound request slot.

        Non-blocking behavior is intentional.

        If all slots are occupied, the request is rejected instead of
        waiting indefinitely.
        """

        acquired = self._semaphore.acquire(
            blocking=False
        )

        if not acquired:
            return {
                "allowed": False,
                "reason": (
                    "Maximum concurrent outbound requests "
                    "has been reached."
                ),
                "active_requests": self.active_requests,
            }

        with self._lock:
            self._active += 1

        return {
            "allowed": True,
            "reason": "Outbound request slot acquired.",
            "active_requests": self.active_requests,
        }

    def release(self) -> None:
        """
        Release a previously acquired request slot.
        """

        with self._lock:

            if self._active > 0:
                self._active -= 1

        try:
            self._semaphore.release()
        except ValueError:
            # Defensive protection against accidental double-release.
            pass

    @property
    def active_requests(self) -> int:
        """
        Return the number of currently active outbound requests.
        """

        with self._lock:
            return self._active


# ---------------------------------------------------------------------------
# Request-control policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RequestSecurityPolicy:
    """
    Central security policy for outbound requests.
    """

    max_requests: int = DEFAULT_MAX_REQUESTS
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Protected request controller
# ---------------------------------------------------------------------------

class RequestSecurityController:
    """
    Combines rate limiting and concurrency protection.

    Each external tool can have its own logical key.

    Example keys:

        "weather"
        "stackexchange"
        "weather:chennai"

    The caller MUST acquire permission before making the HTTP request.
    """

    def __init__(
        self,
        policy: RequestSecurityPolicy | None = None,
    ) -> None:

        self.policy = (
            policy
            if policy is not None
            else RequestSecurityPolicy()
        )

        self.rate_limiter = RateLimiter(
            max_requests=self.policy.max_requests,
            window_seconds=self.policy.window_seconds,
            min_interval_seconds=self.policy.min_interval_seconds,
        )

        self.concurrency_limiter = ConcurrencyLimiter(
            max_concurrent=self.policy.max_concurrent_requests,
        )

    def authorize(
        self,
        key: str,
    ) -> dict[str, Any]:
        """
        Authorize an outbound request.

        Both controls must pass:

            rate limit
                 AND
            concurrency limit

        If either fails, the request is denied.
        """

        rate_result = self.rate_limiter.allow(key)

        if not rate_result["allowed"]:
            return {
                "allowed": False,
                "stage": "rate_limit",
                "reason": rate_result["reason"],
                "retry_after_seconds": (
                    rate_result["retry_after_seconds"]
                ),
            }

        concurrency_result = self.concurrency_limiter.acquire()

        if not concurrency_result["allowed"]:

            return {
                "allowed": False,
                "stage": "concurrency_limit",
                "reason": concurrency_result["reason"],
                "retry_after_seconds": 0,
            }

        return {
            "allowed": True,
            "stage": "authorized",
            "reason": (
                "Request passed rate-limit and concurrency checks."
            ),
            "timeout_seconds": self.policy.timeout_seconds,
        }

    def release(self) -> None:
        """
        Release the concurrency slot after the HTTP operation finishes.

        IMPORTANT:
        This MUST be called in a finally block by the caller.
        """

        self.concurrency_limiter.release()


# ---------------------------------------------------------------------------
# Default application controller
# ---------------------------------------------------------------------------

request_security = RequestSecurityController()

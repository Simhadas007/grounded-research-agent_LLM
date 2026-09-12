"""
Thread-safe in-memory TTL cache.

Used to reduce repeated calls to external APIs while the
Grounded Research Agent is running.

Security/reliability goals:
- Avoid unnecessary external API requests.
- Reduce pressure on public APIs.
- Expire stale data automatically.
- Bound cache size.
- Never cache failed results.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


DEFAULT_MAX_ENTRIES = 256


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class TTLCache:
    """
    Small thread-safe in-memory cache with TTL expiration.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:

        if (
            not isinstance(max_entries, int)
            or isinstance(max_entries, bool)
            or max_entries < 1
        ):
            raise ValueError(
                "max_entries must be a positive integer."
            )

        self.max_entries = max_entries
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """
        Return a cached value if it exists and has not expired.

        Returns None for cache miss or expired entries.
        """

        if not isinstance(key, str) or not key:
            return None

        now = time.monotonic()

        with self._lock:

            entry = self._entries.get(key)

            if entry is None:
                return None

            if entry.expires_at <= now:
                del self._entries[key]
                return None

            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float,
    ) -> None:
        """
        Store a value with a bounded TTL.
        """

        if not isinstance(key, str) or not key:
            return

        if (
            not isinstance(ttl_seconds, (int, float))
            or isinstance(ttl_seconds, bool)
            or ttl_seconds <= 0
        ):
            return

        now = time.monotonic()

        with self._lock:

            # Remove expired entries first.
            expired_keys = [
                cache_key
                for cache_key, entry in self._entries.items()
                if entry.expires_at <= now
            ]

            for cache_key in expired_keys:
                del self._entries[cache_key]

            # If full, remove the entry that expires soonest.
            if (
                key not in self._entries
                and len(self._entries) >= self.max_entries
            ):
                oldest_key = min(
                    self._entries,
                    key=lambda cache_key:
                    self._entries[cache_key].expires_at,
                )

                del self._entries[oldest_key]

            self._entries[key] = _CacheEntry(
                value=value,
                expires_at=now + float(ttl_seconds),
            )

    def delete(self, key: str) -> None:
        """Remove one cached entry."""

        if not isinstance(key, str) or not key:
            return

        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        """Remove all cached entries."""

        with self._lock:
            self._entries.clear()

    def size(self) -> int:
        """Return the number of currently stored entries."""

        now = time.monotonic()

        with self._lock:

            expired_keys = [
                cache_key
                for cache_key, entry in self._entries.items()
                if entry.expires_at <= now
            ]

            for cache_key in expired_keys:
                del self._entries[cache_key]

            return len(self._entries)


# Shared process-level cache.
#
# The cache lives only while the application process is running.
external_api_cache = TTLCache(max_entries=DEFAULT_MAX_ENTRIES)

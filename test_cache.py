import time

from utils.cache import TTLCache


def test_cache_hit():
    cache = TTLCache()

    cache.set(
        "test-key",
        {"value": 123},
        ttl_seconds=10,
    )

    assert cache.get("test-key") == {"value": 123}


def test_cache_miss():
    cache = TTLCache()

    assert cache.get("missing-key") is None


def test_cache_expiration():
    cache = TTLCache()

    cache.set(
        "short-lived",
        "value",
        ttl_seconds=0.05,
    )

    assert cache.get("short-lived") == "value"

    time.sleep(0.08)

    assert cache.get("short-lived") is None


def test_cache_delete():
    cache = TTLCache()

    cache.set(
        "delete-me",
        "value",
        ttl_seconds=10,
    )

    cache.delete("delete-me")

    assert cache.get("delete-me") is None


def test_cache_clear():
    cache = TTLCache()

    cache.set(
        "one",
        1,
        ttl_seconds=10,
    )

    cache.set(
        "two",
        2,
        ttl_seconds=10,
    )

    cache.clear()

    assert cache.size() == 0


def test_cache_max_entries():
    cache = TTLCache(max_entries=2)

    cache.set("one", 1, ttl_seconds=10)
    cache.set("two", 2, ttl_seconds=10)
    cache.set("three", 3, ttl_seconds=10)

    assert cache.size() == 2


def test_invalid_ttl_is_ignored():
    cache = TTLCache()

    cache.set(
        "invalid",
        "value",
        ttl_seconds=0,
    )

    assert cache.get("invalid") is None

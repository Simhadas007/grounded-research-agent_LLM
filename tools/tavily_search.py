from __future__ import annotations

from typing import Any

from tavily import TavilyClient

from config import TAVILY_API_KEY
from security.request_security import RequestSecurityController, RequestSecurityPolicy
from security.url_security import validate_external_url
from utils.cache import external_api_cache


CACHE_TTL_SECONDS = 600
MAX_RESULTS = 5
MAX_QUERY_LENGTH = 500
MAX_RESPONSE_TEXT = 4000


TAVILY_REQUEST_SECURITY = RequestSecurityController(
    RequestSecurityPolicy(
        max_requests=10,
        window_seconds=60,
        min_interval_seconds=1,
        max_concurrent_requests=2,
        timeout_seconds=15,
    )
)


def _clean_text(value: Any, max_length: int = MAX_RESPONSE_TEXT) -> str:
    if not isinstance(value, str):
        return ""

    value = " ".join(value.split())
    return value[:max_length]


def _valid_result_url(url: Any) -> bool:
    if not isinstance(url, str) or not url:
        return False

    try:
        validate_external_url(
            url,
            allowed_hosts=None,
        )
        return True
    except Exception:
        return False


def search_tavily(query: str) -> dict:
    """
    Search the public web using Tavily.

    Returns:
        {
            "success": bool,
            "error": str | None,
            "results": list[dict]
        }
    """

    if not isinstance(query, str):
        return {
            "success": False,
            "error": "Search query must be a string.",
            "results": [],
        }

    query = query.strip()

    if not query:
        return {
            "success": False,
            "error": "Search query cannot be empty.",
            "results": [],
        }

    if len(query) > MAX_QUERY_LENGTH:
        return {
            "success": False,
            "error": "Search query is too long.",
            "results": [],
        }

    cache_key = f"tavily:{query.lower()}"

    cached = external_api_cache.get(cache_key)

    if cached is not None:
        return cached

    authorization = TAVILY_REQUEST_SECURITY.authorize("tavily")

    if not authorization.get("allowed", False):
        return {
            "success": False,
            "error": authorization.get("reason", "Tavily request denied."),
            "results": [],

        }

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=query,
            max_results=MAX_RESULTS,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )

        if not isinstance(response, dict):
            return {
                "success": False,
                "error": "Invalid response received from Tavily.",
                "results": [],
            }

        raw_results = response.get("results", [])

        if not isinstance(raw_results, list):
            return {
                "success": False,
                "error": "Invalid search results received from Tavily.",
                "results": [],
            }

        results: list[dict] = []

        for item in raw_results[:MAX_RESULTS]:
            if not isinstance(item, dict):
                continue

            url = item.get("url")

            if not _valid_result_url(url):
                continue

            title = _clean_text(item.get("title"), 500)
            content = _clean_text(item.get("content"), MAX_RESPONSE_TEXT)

            if not title and not content:
                continue

            results.append(
                {
                    "source": "Tavily Web Search",
                    "source_type": "tavily",
                    "title": title,
                    "content": content,
                    "url": url,
                }
            )

        result = {
            "success": True,
            "error": None,
            "results": results,
        }

        external_api_cache.set(
            cache_key,
            result,
            CACHE_TTL_SECONDS,
        )

        return result

    except Exception as exc:
        return {
            "success": False,
            "error": f"Tavily search failed: {type(exc).__name__}.",
            "results": [],
        }

    finally:
        TAVILY_REQUEST_SECURITY.release()
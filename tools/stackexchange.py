"""
Stack Exchange Community Retrieval
==================================

Stack Exchange is used as the Quora substitute for the
Grounded Research Agent.

Design goals:
- Grounded community retrieval
- Stack Overflow / Information Security / Software Engineering
- Sequential requests to reduce API throttling
- TTL caching
- Query fallback
- Top-answer retrieval
- URL validation
- HTML sanitization
- Bounded response sizes
- Rate limiting
- Concurrency limiting
- Detailed errors
- No execution of retrieved content
"""

import html
import re
import time
from typing import Any

import requests

from utils.cache import external_api_cache
from utils.url_security import validate_external_url
from security.request_security import (
    RequestSecurityController,
    RequestSecurityPolicy,
)


# ============================================================================
# SECURITY / REQUEST POLICY
# ============================================================================

STACKEXCHANGE_REQUEST_SECURITY = RequestSecurityController(
    RequestSecurityPolicy(
        max_requests=12,
        window_seconds=60,
        min_interval_seconds=1.0,
        max_concurrent_requests=1,
        timeout_seconds=10,
    )
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

SEARCH_URL = "https://api.stackexchange.com/2.3/search/advanced"
ANSWERS_URL = "https://api.stackexchange.com/2.3/questions"


# ============================================================================
# LIMITS
# ============================================================================

MAX_QUERY_LENGTH = 500

MAX_TEXT_LENGTH = 12000
MAX_TITLE_LENGTH = 500
MAX_BODY_LENGTH = 5000
MAX_ANSWER_LENGTH = 5000

MAX_RESULTS_PER_SITE = 3
MAX_TOTAL_RESULTS = 8

MAX_RESPONSE_SIZE = 2_000_000

CACHE_TTL_SECONDS = 600

# Only use the most relevant communities.
# This dramatically reduces unnecessary API requests.
PRIORITY_SITES = {
    "stackoverflow": "Stack Overflow",
    "security": "Information Security",
    "softwareengineering": "Software Engineering",
}


# ============================================================================
# USER AGENT
# ============================================================================

USER_AGENT = (
    "GroundedResearchAgent/1.0 "
    "(educational grounded-research project)"
)


# ============================================================================
# QUERY STOPWORDS
# ============================================================================

QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "should",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}


# ============================================================================
# QUERY NORMALIZATION
# ============================================================================

def _normalize_query(query: str) -> str:
    """Normalize whitespace."""

    return " ".join(
        query.strip().split()
    )


def _compact_query(query: str) -> str:
    """
    Create a compact generic search query.

    This is only a retrieval fallback.
    It does not determine the answer.
    """

    normalized = _normalize_query(query)

    cleaned = re.sub(
        r"[^\w\s./:+#-]",
        " ",
        normalized,
        flags=re.UNICODE,
    )

    words = cleaned.split()

    useful_words = []

    for word in words:

        lowered = word.lower()

        if lowered in QUERY_STOPWORDS:
            continue

        if len(lowered) <= 2:
            continue

        useful_words.append(word)

    compact = " ".join(
        useful_words
    )

    if not compact:
        compact = normalized

    return compact[:MAX_QUERY_LENGTH].strip()


def _query_variants(query: str) -> list[str]:
    """
    Produce at most two search variants.

    Original question is always first.
    """

    normalized = _normalize_query(query)

    variants = [
        normalized
    ]

    compact = _compact_query(
        normalized
    )

    if (
        compact
        and compact.lower()
        != normalized.lower()
    ):
        variants.append(
            compact
        )

    return variants[:2]


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_html(
    text: Any,
) -> str:
    """
    Convert Stack Exchange HTML to bounded plain text.

    Retrieved content is evidence only.
    It is never executed as instructions.
    """

    if not isinstance(
        text,
        str,
    ):
        return ""

    text = html.unescape(
        text
    )

    # Remove script/style content.
    text = re.sub(
        r"<(?:script|style)[^>]*>.*?</(?:script|style)>",
        " ",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    # Structural tags.
    text = re.sub(
        r"<(?:br|/p|/div|/li|/pre|/blockquote)>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remaining HTML.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()[
        :MAX_TEXT_LENGTH
    ]


# ============================================================================
# BASIC HELPERS
# ============================================================================

def _safe_int(
    value: Any,
    default: int = 0,
) -> int:

    if isinstance(
        value,
        bool,
    ):
        return default

    if isinstance(
        value,
        int,
    ):
        return value

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_bool(
    value: Any,
    default: bool = False,
) -> bool:

    if isinstance(
        value,
        bool,
    ):
        return value

    return default


def _safe_tags(
    value: Any,
) -> list[str]:

    if not isinstance(
        value,
        list,
    ):
        return []

    tags = []

    for tag in value:

        if not isinstance(
            tag,
            str,
        ):
            continue

        tag = tag.strip()

        if not tag:
            continue

        tags.append(
            tag[:100]
        )

    return tags[:30]


# ============================================================================
# SAFE HTTP GET
# ============================================================================

def safe_get(
    url: str,
    params: dict[str, Any],
):
    """
    Security-controlled Stack Exchange GET.

    Important:
    - validates destination URL
    - uses request-security controller
    - respects API 429 responses
    - reads Stack Exchange JSON error details
    - captures API backoff
    """

    validation = validate_external_url(
        url
    )

    if not validation["allowed"]:
        raise ValueError(
            "External Stack Exchange URL failed "
            "security validation."
        )

    authorization = (
        STACKEXCHANGE_REQUEST_SECURITY.authorize(
            "stackexchange"
        )
    )

    if not authorization["allowed"]:
        raise RuntimeError(
            "Stack Exchange request blocked by "
            "security policy: "
            + authorization["reason"]
        )

    try:

        response = requests.get(
            url,
            params=params,
            timeout=authorization[
                "timeout_seconds"
            ],
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )

        if len(response.content) > MAX_RESPONSE_SIZE:
            raise ValueError(
                "External Stack Exchange response "
                "exceeded the allowed size."
            )

        # ------------------------------------------------------------
        # 429 THROTTLING
        # ------------------------------------------------------------

        if response.status_code == 429:

            retry_after = (
                response.headers.get(
                    "Retry-After"
                )
            )

            message = (
                "Stack Exchange API throttled the request."
            )

            if retry_after:
                message += (
                    f" Retry-After={retry_after}"
                )

            raise RuntimeError(
                message
            )

        # ------------------------------------------------------------
        # NON-2XX RESPONSE
        # ------------------------------------------------------------

        if not response.ok:

            try:
                error_data = response.json()

            except ValueError:
                error_data = {}

            if isinstance(
                error_data,
                dict,
            ):

                error_id = (
                    error_data.get(
                        "error_id"
                    )
                )

                error_name = (
                    error_data.get(
                        "error_name"
                    )
                )

                error_message = (
                    error_data.get(
                        "error_message"
                    )
                )

                if (
                    error_id is not None
                    or error_name
                    or error_message
                ):

                    raise ValueError(
                        "Stack Exchange API error"
                        + (
                            f" [{error_name}]"
                            if error_name
                            else ""
                        )
                        + (
                            f" (id={error_id})"
                            if error_id is not None
                            else ""
                        )
                        + (
                            f": {str(error_message)[:500]}"
                            if error_message
                            else ""
                        )
                    )

            raise requests.HTTPError(
                "Stack Exchange HTTP "
                f"{response.status_code}: "
                f"{response.reason}"
            )

        # ------------------------------------------------------------
        # SUCCESS RESPONSE
        # ------------------------------------------------------------

        try:

            data = response.json()

        except ValueError as exc:

            raise ValueError(
                "Stack Exchange returned invalid JSON."
            ) from exc

        # ------------------------------------------------------------
        # API BACKOFF
        # ------------------------------------------------------------

        if isinstance(
            data,
            dict,
        ):

            backoff = _safe_int(
                data.get(
                    "backoff",
                    0,
                )
            )

            if backoff > 0:
                # Do not sleep here for a large value.
                # Store the backoff in the exception so callers
                # know that the API asked us to wait.
                raise RuntimeError(
                    "Stack Exchange API requested "
                    f"a backoff of {backoff} seconds."
                )

        return response

    finally:

        STACKEXCHANGE_REQUEST_SECURITY.release()


# ============================================================================
# API RESPONSE VALIDATION
# ============================================================================

def _validate_api_response(
    data: Any,
    site_name: str,
) -> list[dict]:

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            f"Unexpected response from {site_name}."
        )

    error_id = data.get(
        "error_id"
    )

    if error_id is not None:

        error_name = data.get(
            "error_name",
            "unknown_error",
        )

        error_message = str(
            data.get(
                "error_message",
                "Stack Exchange API error.",
            )
        )[:500]

        raise ValueError(
            f"{site_name}: "
            f"{error_name}: "
            f"{error_message}"
        )

    items = data.get(
        "items",
        [],
    )

    if not isinstance(
        items,
        list,
    ):
        raise ValueError(
            f"Invalid result format from "
            f"{site_name}."
        )

    return items


# ============================================================================
# SEARCH ONE COMMUNITY
# ============================================================================

def _search_site(
    query: str,
    site_parameter: str,
    site_name: str,
) -> list[dict]:
    """
    Search one Stack Exchange community.

    Deliberately does NOT request question bodies here.
    """

    response = safe_get(
        SEARCH_URL,
        {
            "order": "desc",
            "sort": "relevance",
            "q": query[:MAX_QUERY_LENGTH],
            "site": site_parameter,
            "pagesize": MAX_RESULTS_PER_SITE,
        },
    )

    data = response.json()

    questions = (
        _validate_api_response(
            data,
            site_name,
        )
    )

    results = []

    for question in questions[
        :MAX_RESULTS_PER_SITE
    ]:

        if not isinstance(
            question,
            dict,
        ):
            continue

        question_id = (
            question.get(
                "question_id"
            )
        )

        if not isinstance(
            question_id,
            int,
        ):
            continue

        question_url = (
            question.get(
                "link",
                "",
            )
        )

        if not isinstance(
            question_url,
            str,
        ):
            continue

        question_url = (
            question_url.strip()
        )

        url_check = (
            validate_external_url(
                question_url
            )
        )

        if not url_check[
            "allowed"
        ]:
            continue

        title = clean_html(
            question.get(
                "title",
                "",
            )
        )[:MAX_TITLE_LENGTH]

        question_body = clean_html(
            question.get(
                "body",
                "",
            )
        )[:MAX_BODY_LENGTH]

        if not title and not question_body:
            continue

        results.append(
            {
                "question_id": question_id,
                "site": site_parameter,
                "site_name": site_name,
                "title": title,
                "question_body": question_body,
                "question_score": _safe_int(
                    question.get(
                        "score",
                        0,
                    )
                ),
                "tags": _safe_tags(
                    question.get(
                        "tags",
                        [],
                    )
                ),
                "url": question_url,
                "is_answered": _safe_bool(
                    question.get(
                        "is_answered",
                        False,
                    )
                ),
            }
        )

    return results


# ============================================================================
# TOP ANSWER
# ============================================================================

def _get_top_answer(
    question_id: int,
    site_parameter: str,
) -> dict:

    if (
        not isinstance(
            question_id,
            int,
        )
        or question_id <= 0
    ):
        return {
            "answer_body": "",
            "answer_score": 0,
        }

    response = safe_get(
        f"{ANSWERS_URL}/{question_id}/answers",
        {
            "order": "desc",
            "sort": "votes",
            "site": site_parameter,
            "pagesize": 1,
            "filter": "withbody",
        },
    )

    data = response.json()

    answers = (
        _validate_api_response(
            data,
            site_parameter,
        )
    )

    if not answers:
        return {
            "answer_body": "",
            "answer_score": 0,
        }

    answer = answers[0]

    if not isinstance(
        answer,
        dict,
    ):
        return {
            "answer_body": "",
            "answer_score": 0,
        }

    answer_body = clean_html(
        answer.get(
            "body",
            "",
        )
    )[:MAX_ANSWER_LENGTH]

    answer_score = _safe_int(
        answer.get(
            "score",
            0,
        )
    )

    return {
        "answer_body": answer_body,
        "answer_score": answer_score,
    }


# ============================================================================
# DEDUPLICATION
# ============================================================================

def _deduplicate_results(
    results: list[dict],
) -> list[dict]:

    seen = set()
    unique = []

    for result in results:

        key = (
            str(
                result.get(
                    "site",
                    "",
                )
            ),
            _safe_int(
                result.get(
                    "question_id"
                )
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            result
        )

    return unique


# ============================================================================
# RESULT RANKING
# ============================================================================

def _result_rank(
    item: dict,
) -> tuple[int, int, int]:

    return (
        _safe_int(
            item.get(
                "answer_score",
                0,
            )
        ),
        _safe_int(
            item.get(
                "question_score",
                0,
            )
        ),
        (
            1
            if item.get(
                "is_answered",
                False,
            )
            else 0
        ),
    )


# ============================================================================
# ERROR FORMATTER
# ============================================================================

def _format_errors(
    errors: list[str],
) -> str:

    if not errors:
        return "Unknown Stack Exchange error."

    unique = []

    for error in errors:

        if error not in unique:
            unique.append(
                error
            )

    return " | ".join(
        unique[:6]
    )[:2000]


# ============================================================================
# PUBLIC SEARCH
# ============================================================================

def search_stackexchange(
    query: str,
) -> dict:
    """
    Search Stack Exchange for grounded evidence.

    Requests are deliberately sequential to avoid triggering
    Stack Exchange's IP/method throttles.
    """

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------

    if not isinstance(
        query,
        str,
    ):
        return {
            "success": False,
            "error": (
                "Search query must be valid text."
            ),
            "results": [],
        }

    query = _normalize_query(
        query
    )

    if not query:

        return {
            "success": False,
            "error": (
                "Search query cannot be empty."
            ),
            "results": [],
        }

    query = query[
        :MAX_QUERY_LENGTH
    ].strip()

    # ------------------------------------------------------------------
    # CACHE
    # ------------------------------------------------------------------

    cache_key = (
        "stackexchange:v5:"
        + query.lower()
    )

    cached = (
        external_api_cache.get(
            cache_key
        )
    )

    if cached is not None:
        return cached

    # ------------------------------------------------------------------
    # Query variants
    # ------------------------------------------------------------------

    variants = _query_variants(
        query
    )

    all_results = []
    errors = []

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    for variant in variants:

        for site_parameter, site_name in (
            PRIORITY_SITES.items()
        ):

            # Explicit delay between API calls.
            #
            # Stack Exchange recommends avoiding semantically
            # identical requests more often than once per minute,
            # and the API has IP/method throttles.
            time.sleep(1.2)

            try:

                results = _search_site(
                    variant,
                    site_parameter,
                    site_name,
                )

                if results:

                    all_results.extend(
                        results
                    )

                    # Once we have useful evidence from one
                    # community, do not hammer additional sites.
                    if len(
                        all_results
                    ) >= MAX_TOTAL_RESULTS:

                        break

            except requests.Timeout:

                errors.append(
                    f"{site_parameter}: timeout"
                )

            except requests.RequestException as exc:

                errors.append(
                    f"{site_parameter}: "
                    f"request failed: "
                    f"{str(exc)[:300]}"
                )

            except RuntimeError as exc:

                errors.append(
                    f"{site_parameter}: "
                    f"{str(exc)[:500]}"
                )

            except ValueError as exc:

                errors.append(
                    f"{site_parameter}: "
                    f"{str(exc)[:500]}"
                )

            except Exception as exc:

                errors.append(
                    f"{site_parameter}: "
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:300]}"
                )

        if all_results:
            break

    # ------------------------------------------------------------------
    # DEDUPLICATE
    # ------------------------------------------------------------------

    all_results = (
        _deduplicate_results(
            all_results
        )
    )

    # ------------------------------------------------------------------
    # NO RESULTS
    # ------------------------------------------------------------------

    if not all_results:

        result = {
            "success": False,
            "error": (
                "Stack Exchange retrieval failed: "
                + _format_errors(
                    errors
                )
            ),
            "results": [],
        }

        # Do not cache throttling failures.
        return result

    # ------------------------------------------------------------------
    # ANSWERS
    # ------------------------------------------------------------------

    enriched = []

    answer_errors = []

    for item in all_results[
        :MAX_TOTAL_RESULTS
    ]:

        try:

            answer = (
                _get_top_answer(
                    item[
                        "question_id"
                    ],
                    item[
                        "site"
                    ],
                )
            )

        except Exception as exc:

            answer_errors.append(
                f"{item['site']}: "
                f"answer retrieval failed: "
                f"{type(exc).__name__}: "
                f"{str(exc)[:300]}"
            )

            answer = {
                "answer_body": "",
                "answer_score": 0,
            }

        item.update(
            answer
        )

        enriched.append(
            item
        )

    # ------------------------------------------------------------------
    # RANK
    # ------------------------------------------------------------------

    enriched.sort(
        key=_result_rank,
        reverse=True,
    )

    # ------------------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------------------

    final_results = enriched[
        :MAX_TOTAL_RESULTS
    ]

    result = {
        "success": True,
        "error": (
            None
            if not (
                errors
                or answer_errors
            )
            else _format_errors(
                errors
                + answer_errors
            )
        ),
        "results": final_results,
    }

    # Cache only successful retrieval.
    external_api_cache.set(
        cache_key,
        result,
        CACHE_TTL_SECONDS,
    )

    return result
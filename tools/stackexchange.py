"""
Stack Exchange Community Retrieval
===================================

Searches multiple Stack Exchange communities for grounded
community-generated questions and answers.

This is used as the Quora substitute in the project.
"""

import html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from utils.url_security import validate_external_url


SEARCH_URL = "https://api.stackexchange.com/2.3/search/advanced"
QUESTIONS_URL = "https://api.stackexchange.com/2.3/questions"

TIMEOUT_SECONDS = 10
MAX_RESULTS_PER_SITE = 2
MAX_TOTAL_RESULTS = 8
MAX_TEXT_LENGTH = 12000
MAX_QUERY_LENGTH = 500
MAX_RESPONSE_SIZE = 2_000_000


# Multiple Stack Exchange communities relevant to
# programming, development, systems, and technical questions.
STACK_EXCHANGE_SITES = {
    "stackoverflow": "Stack Overflow",
    "softwareengineering": "Software Engineering",
    "superuser": "Super User",
    "serverfault": "Server Fault",
    "askubuntu": "Ask Ubuntu",
    "unix": "Unix & Linux",
}


def clean_html(text: str) -> str:
    """Convert HTML content into safe plain text."""

    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()[:MAX_TEXT_LENGTH]


def safe_get(url: str, params: dict):
    """
    Perform a validated external GET request.

    Only HTTPS URLs that pass our URL security validation
    are allowed.
    """

    validation = validate_external_url(url)

    if not validation["allowed"]:
        raise ValueError(
            "External source URL failed security validation."
        )

    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    if len(response.content) > MAX_RESPONSE_SIZE:
        raise ValueError(
            "External response exceeded the allowed size."
        )

    return response


def _search_site(
    query: str,
    site_parameter: str,
    site_name: str,
) -> list[dict]:
    """
    Search one Stack Exchange community.
    """

    response = safe_get(
        SEARCH_URL,
        {
            "order": "desc",
            "sort": "relevance",
            "q": query[:MAX_QUERY_LENGTH],
            "site": site_parameter,
            "pagesize": MAX_RESULTS_PER_SITE,
            "filter": "withbody",
        },
    )

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError(
            f"Unexpected response from {site_name}."
        )

    questions = data.get("items", [])

    if not isinstance(questions, list):
        raise ValueError(
            f"Invalid result format from {site_name}."
        )

    results = []

    for question in questions[:MAX_RESULTS_PER_SITE]:

        if not isinstance(question, dict):
            continue

        question_id = question.get("question_id")

        if not isinstance(question_id, int):
            continue

        question_url = question.get("link", "")

        url_check = validate_external_url(question_url)

        if not url_check["allowed"]:
            continue

        results.append(
            {
                "question_id": question_id,
                "site": site_parameter,
                "site_name": site_name,
                "title": clean_html(
                    question.get("title", "")
                ),
                "question_body": clean_html(
                    question.get("body", "")
                ),
                "question_score": question.get(
                    "score",
                    0,
                ),
                "tags": question.get(
                    "tags",
                    [],
                ),
                "url": question_url,
                "is_answered": question.get(
                    "is_answered",
                    False,
                ),
            }
        )

    return results


def _get_top_answer(
    question_id: int,
    site_parameter: str,
) -> dict:
    """
    Retrieve the highest-voted answer for a question.
    """

    response = safe_get(
        f"{QUESTIONS_URL}/{question_id}/answers",
        {
            "order": "desc",
            "sort": "votes",
            "site": site_parameter,
            "pagesize": 1,
            "filter": "withbody",
        },
    )

    data = response.json()

    if not isinstance(data, dict):
        return {
            "answer_body": "",
            "answer_score": 0,
        }

    answers = data.get("items", [])

    if not isinstance(answers, list):
        return {
            "answer_body": "",
            "answer_score": 0,
        }

    if not answers:
        return {
            "answer_body": "",
            "answer_score": 0,
        }

    best_answer = answers[0]

    if not isinstance(best_answer, dict):
        return {
            "answer_body": "",
            "answer_score": 0,
        }

    return {
        "answer_body": clean_html(
            best_answer.get("body", "")
        ),
        "answer_score": best_answer.get(
            "score",
            0,
        ),
    }


def search_stackexchange(query: str) -> dict:
    """
    Search multiple Stack Exchange communities.

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
            "error": "Search query must be valid text.",
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
        query = query[:MAX_QUERY_LENGTH]

    all_results = []

    site_errors = []

    # Search communities concurrently.
    #
    # This keeps the overall latency reasonable even though
    # multiple communities are being queried.
    with ThreadPoolExecutor(
        max_workers=len(STACK_EXCHANGE_SITES)
    ) as executor:

        futures = {
            executor.submit(
                _search_site,
                query,
                site_parameter,
                site_name,
            ): site_parameter
            for site_parameter, site_name
            in STACK_EXCHANGE_SITES.items()
        }

        for future in as_completed(futures):

            site_parameter = futures[future]

            try:
                results = future.result()
                all_results.extend(results)

            except requests.Timeout:
                site_errors.append(
                    f"{site_parameter}: timeout"
                )

            except requests.RequestException:
                site_errors.append(
                    f"{site_parameter}: request failed"
                )

            except ValueError:
                site_errors.append(
                    f"{site_parameter}: invalid response"
                )

            except Exception:
                site_errors.append(
                    f"{site_parameter}: unexpected error"
                )

    if not all_results:

        if site_errors:
            return {
                "success": False,
                "error": (
                    "Stack Exchange communities could not "
                    "return usable results."
                ),
                "results": [],
            }

        return {
            "success": True,
            "error": None,
            "results": [],
        }

    # Sort first by question score and then answer score.
    #
    # Answer score is added after retrieving answers below.
    enriched_results = []

    for result in all_results:

        try:
            answer = _get_top_answer(
                result["question_id"],
                result["site"],
            )

        except (
            requests.Timeout,
            requests.RequestException,
            ValueError,
        ):
            answer = {
                "answer_body": "",
                "answer_score": 0,
            }

        result.update(answer)

        enriched_results.append(result)

    # Higher-scoring community answers/questions first.
    enriched_results.sort(
        key=lambda item: (
            item.get("answer_score", 0),
            item.get("question_score", 0),
        ),
        reverse=True,
    )

    # Keep the evidence set bounded.
    final_results = enriched_results[
        :MAX_TOTAL_RESULTS
    ]

    return {
        "success": True,
        "error": (
            None
            if not site_errors
            else "Some Stack Exchange communities were unavailable."
        ),
        "results": final_results,
    }
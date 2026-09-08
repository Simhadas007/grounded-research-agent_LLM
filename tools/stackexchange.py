import html
import re
import requests

from utils.url_security import validate_external_url


SEARCH_URL = "https://api.stackexchange.com/2.3/search/advanced"
QUESTIONS_URL = "https://api.stackexchange.com/2.3/questions"

TIMEOUT_SECONDS = 10
MAX_RESULTS = 5
MAX_TEXT_LENGTH = 12000


def clean_html(text: str) -> str:
    """Convert HTML into safe plain text and limit its size."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text[:MAX_TEXT_LENGTH]


def safe_get(url: str, params: dict):
    """Perform a validated external HTTPS request."""
    validation = validate_external_url(url)

    if not validation["allowed"]:
        raise ValueError("External source URL failed security validation.")

    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    # Prevent unexpectedly large responses.
    if len(response.content) > 2_000_000:
        raise ValueError("External response exceeded the allowed size.")

    return response


def search_stackexchange(query: str) -> dict:
    query = query.strip()

    if not query:
        return {
            "success": False,
            "error": "Search query cannot be empty.",
            "results": [],
        }

    try:
        response = safe_get(
            SEARCH_URL,
            {
                "order": "desc",
                "sort": "relevance",
                "q": query[:500],
                "site": "stackoverflow",
                "pagesize": MAX_RESULTS,
                "filter": "withbody",
            },
        )

        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("Unexpected Stack Exchange response.")

        questions = data.get("items", [])

        if not isinstance(questions, list):
            raise ValueError("Invalid Stack Exchange result format.")

        results = []

        for question in questions[:MAX_RESULTS]:

            if not isinstance(question, dict):
                continue

            question_id = question.get("question_id")

            if not isinstance(question_id, int):
                continue

            answer_body = ""
            answer_score = 0

            try:
                answer_response = safe_get(
                    f"{QUESTIONS_URL}/{question_id}/answers",
                    {
                        "order": "desc",
                        "sort": "votes",
                        "site": "stackoverflow",
                        "pagesize": 1,
                        "filter": "withbody",
                    },
                )

                answer_data = answer_response.json()

                if not isinstance(answer_data, dict):
                    continue

                answers = answer_data.get("items", [])

                if isinstance(answers, list) and answers:
                    best_answer = answers[0]

                    if isinstance(best_answer, dict):
                        answer_body = clean_html(
                            best_answer.get("body", "")
                        )

                        answer_score = best_answer.get(
                            "score",
                            0,
                        )

            except (
                requests.RequestException,
                ValueError,
            ):
                # A failed answer request should not expose
                # internal details or crash the entire search.
                pass

            url = question.get("link", "")

            url_check = validate_external_url(url)

            if not url_check["allowed"]:
                continue

            results.append(
                {
                    "title": clean_html(
                        question.get("title", "")
                    ),
                    "question_body": clean_html(
                        question.get("body", "")
                    ),
                    "answer_body": answer_body,
                    "question_score": question.get(
                        "score",
                        0,
                    ),
                    "answer_score": answer_score,
                    "tags": question.get(
                        "tags",
                        [],
                    ),
                    "url": url,
                    "is_answered": question.get(
                        "is_answered",
                        False,
                    ),
                }
            )

        return {
            "success": True,
            "error": None,
            "results": results,
        }

    except requests.Timeout:
        return {
            "success": False,
            "error": "Stack Exchange service timed out.",
            "results": [],
        }

    except requests.RequestException:
        return {
            "success": False,
            "error": "Stack Exchange service request failed.",
            "results": [],
        }

    except ValueError:
        return {
            "success": False,
            "error": "Stack Exchange returned invalid data.",
            "results": [],
        }
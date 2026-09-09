"""
LLM security boundaries for the Grounded Research Agent.

This module validates LLM outputs before they are trusted by the application.

Security principles:
- Fail closed on malformed or unexpected model output.
- Never trust arbitrary fields returned by the model.
- Enforce strict route allowlisting.
- Enforce bounded string sizes.
- Enforce citation requirements.
- Never allow citations that were not present in retrieved evidence.
- Reject answers that claim grounding without valid evidence.
- Reject empty or excessively large answers.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_LLM_OUTPUT_CHARS = 20_000
MAX_REASON_CHARS = 2_000
MAX_ANSWER_CHARS = 8_000
MAX_CITATIONS = 10
MAX_CITATION_URL_CHARS = 2_000

ALLOWED_ROUTES = {
    "stackexchange",
    "weather",
    "both",
    "unsupported",
}

ALLOWED_QUALITY = {
    "high",
    "medium",
    "low",
}


# ---------------------------------------------------------------------------
# Secure LLM result models
# ---------------------------------------------------------------------------

class SecureSearchPlan(BaseModel):
    """
    Strict representation of the router LLM's decision.

    extra='forbid' is important:
    unexpected model-generated fields are rejected instead of being trusted.
    """

    model_config = ConfigDict(extra="forbid")

    route: Literal[
        "stackexchange",
        "weather",
        "both",
        "unsupported",
    ]

    search_query: str = Field(default="", max_length=2_000)

    location: str = Field(default="", max_length=500)

    reason: str = Field(default="", max_length=2_000)


class SecureResearchAnswer(BaseModel):
    """
    Strict representation of the answer LLM's result.
    """

    model_config = ConfigDict(extra="forbid")

    relevant: bool
    sufficient: bool

    quality: Literal["high", "medium", "low"]

    reason: str = Field(max_length=2_000)

    answer: str = Field(max_length=8_000)

    citations: list[str] = Field(default_factory=list, max_length=10)


# ---------------------------------------------------------------------------
# Generic LLM output checks
# ---------------------------------------------------------------------------

def validate_llm_output_size(content: Any) -> str:
    """
    Validate and normalize raw LLM output.

    Fail closed:
    - non-string output is rejected
    - empty output is rejected
    - oversized output is rejected
    """

    if not isinstance(content, str):
        raise ValueError("LLM output must be a string.")

    content = content.strip()

    if not content:
        raise ValueError("LLM returned empty output.")

    if len(content) > MAX_LLM_OUTPUT_CHARS:
        raise ValueError("LLM output exceeds the allowed size.")

    return content


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def parse_llm_json(content: Any) -> dict[str, Any]:
    """
    Parse a JSON object returned by the LLM.

    The model is expected to return JSON only.

    We intentionally do not execute, evaluate, or interpret model output
    as Python code.
    """

    content = validate_llm_output_size(content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("LLM JSON output must be an object.")

    return data


# ---------------------------------------------------------------------------
# Router output validation
# ---------------------------------------------------------------------------

def validate_search_plan(data: Any) -> SecureSearchPlan:
    """
    Validate an LLM-generated routing/search plan.

    This is an authorization boundary:
    the LLM may suggest a route, but only explicitly allowed routes
    can reach the tools.
    """

    if not isinstance(data, dict):
        raise ValueError("Search plan must be a JSON object.")

    try:
        plan = SecureSearchPlan.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            "LLM produced an invalid search plan."
        ) from exc

    if plan.route not in ALLOWED_ROUTES:
        raise ValueError("LLM selected an unauthorized route.")

    # Normalize user/tool-bound strings.
    plan.search_query = plan.search_query.strip()
    plan.location = plan.location.strip()
    plan.reason = plan.reason.strip()

    # Route-specific authorization checks.
    if plan.route == "stackexchange":
        if not plan.search_query:
            raise ValueError(
                "Stack Exchange route requires a search query."
            )

    elif plan.route == "weather":
        if not plan.location:
            raise ValueError(
                "Weather route requires a location."
            )

    elif plan.route == "both":
        if not plan.search_query:
            raise ValueError(
                "Combined route requires a search query."
            )

        if not plan.location:
            raise ValueError(
                "Combined route requires a location."
            )

    elif plan.route == "unsupported":
        # Explicitly allowed, but it must not reach retrieval.
        pass

    return plan


def validate_search_plan_json(content: Any) -> SecureSearchPlan:
    """
    Parse and validate raw router LLM output in one operation.
    """

    data = parse_llm_json(content)
    return validate_search_plan(data)


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    return url.strip()


def _is_https_url(url: str) -> bool:
    """
    Basic URL validation.

    This does NOT replace security.url_security.validate_external_url.
    It only verifies that citations have the expected URL shape.
    """

    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme.lower() != "https":
        return False

    if not parsed.netloc:
        return False

    if parsed.username or parsed.password:
        return False

    return True


def extract_evidence_urls(evidence: Iterable[dict[str, Any]]) -> set[str]:
    """
    Extract exact citation URLs from retrieved evidence.

    Only URLs supplied by trusted application retrieval tools are eligible
    to become citations.
    """

    allowed_urls: set[str] = set()

    for item in evidence:
        if not isinstance(item, dict):
            continue

        url = item.get("url")

        if not isinstance(url, str):
            continue

        url = _normalize_url(url)

        if not url:
            continue

        if len(url) > MAX_CITATION_URL_CHARS:
            continue

        if not _is_https_url(url):
            continue

        allowed_urls.add(url)

    return allowed_urls


def validate_citations(
    citations: Any,
    evidence: Iterable[dict[str, Any]],
) -> list[str]:
    """
    Validate citations against exact URLs present in retrieved evidence.

    This is the primary anti-fabrication boundary.

    The model cannot create a new citation.
    It can only cite an exact URL supplied by the retrieval layer.
    """

    if citations is None:
        return []

    if not isinstance(citations, list):
        raise ValueError("Citations must be a list.")

    if len(citations) > MAX_CITATIONS:
        raise ValueError("Too many citations returned by the LLM.")

    allowed_urls = extract_evidence_urls(evidence)

    validated: list[str] = []

    for citation in citations:
        if not isinstance(citation, str):
            raise ValueError("Citation must be a string.")

        citation = _normalize_url(citation)

        if not citation:
            raise ValueError("Empty citation is not allowed.")

        if len(citation) > MAX_CITATION_URL_CHARS:
            raise ValueError("Citation URL is too long.")

        if not _is_https_url(citation):
            raise ValueError(
                "Citation must be a valid HTTPS URL."
            )

        if citation not in allowed_urls:
            raise ValueError(
                "LLM returned a citation that was not present "
                "in the retrieved evidence."
            )

        if citation not in validated:
            validated.append(citation)

    return validated


# ---------------------------------------------------------------------------
# Research answer validation
# ---------------------------------------------------------------------------

def validate_research_answer(
    data: Any,
    evidence: Iterable[dict[str, Any]],
) -> SecureResearchAnswer:
    """
    Validate an answer generated from retrieved evidence.

    Important rule:

        sufficient=True
        =>
        at least one valid evidence citation is required.

    If the answer is insufficient, citations are removed rather than trusted.
    """

    if not isinstance(data, dict):
        raise ValueError("Research answer must be a JSON object.")

    try:
        result = SecureResearchAnswer.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            "LLM produced an invalid research answer."
        ) from exc

    result.reason = result.reason.strip()
    result.answer = result.answer.strip()

    if not result.reason:
        raise ValueError("Research answer reason cannot be empty.")

    if result.sufficient and not result.answer:
        raise ValueError(
            "A sufficient answer cannot have empty answer text."
        )

    citations = validate_citations(
        result.citations,
        evidence,
    )

    if not result.sufficient:
        # Insufficient answers must never carry apparently authoritative
        # citations.
        result.citations = []
        return result

    if not result.relevant:
        raise ValueError(
            "An answer cannot be sufficient when marked irrelevant."
        )

    if not citations:
        raise ValueError(
            "A sufficient grounded answer requires at least one "
            "valid evidence citation."
        )

    result.citations = citations

    return result


def validate_research_answer_json(
    content: Any,
    evidence: Iterable[dict[str, Any]],
) -> SecureResearchAnswer:
    """
    Parse and securely validate raw answer-model output.
    """

    data = parse_llm_json(content)

    return validate_research_answer(
        data,
        evidence,
    )


# ---------------------------------------------------------------------------
# Safe failure helper
# ---------------------------------------------------------------------------

def llm_security_error_message(stage: str) -> str:
    """
    Return a safe user-facing error.

    Internal validation details should not be exposed to users.
    """

    stage = stage.strip().lower()

    if stage == "router":
        return (
            "I could not safely determine which research source "
            "should handle this question."
        )

    if stage == "answer":
        return (
            "I could not safely validate the research answer "
            "against the retrieved evidence."
        )

    return "The request could not be safely completed."


__all__ = [
    "SecureSearchPlan",
    "SecureResearchAnswer",
    "validate_llm_output_size",
    "parse_llm_json",
    "validate_search_plan",
    "validate_search_plan_json",
    "extract_evidence_urls",
    "validate_citations",
    "validate_research_answer",
    "validate_research_answer_json",
    "llm_security_error_message",
]

"""
Security layer for untrusted research evidence.

Retrieved content from Stack Exchange, Reddit, Quora, APIs, etc.
must NEVER be treated as trusted instructions.

This module:
1. Detects likely prompt-injection attempts.
2. Detects instruction-like content attempting to control the agent.
3. Sanitizes suspicious text.
4. Allows the application to fail closed before evidence reaches the LLM.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_EVIDENCE_TEXT_LENGTH = 12000

# Strong prompt-injection indicators.
INJECTION_PATTERNS = [
    # Instruction override
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bignore\s+(all\s+)?prior\s+instructions\b",
    r"\bdisregard\s+(all\s+)?previous\s+instructions\b",
    r"\bdisregard\s+(all\s+)?prior\s+instructions\b",

    # System/developer instruction manipulation
    r"\bignore\s+(the\s+)?system\s+(message|prompt|instructions)\b",
    r"\bignore\s+(the\s+)?developer\s+(message|prompt|instructions)\b",
    r"\boverride\s+(the\s+)?system\s+(message|prompt|instructions)\b",
    r"\boverride\s+(the\s+)?developer\s+(message|prompt|instructions)\b",

    # Prompt extraction
    r"\breveal\s+(your|the)\s+(system\s+prompt|hidden\s+prompt)\b",
    r"\bshow\s+(me\s+)?(your|the)\s+(system\s+prompt|hidden\s+prompt)\b",
    r"\bprint\s+(your|the)\s+(system\s+prompt|hidden\s+instructions)\b",
    r"\boutput\s+(your|the)\s+(system\s+prompt|hidden\s+instructions)\b",

    # Role manipulation
    r"\byou\s+are\s+now\s+(a|an)\b",
    r"\bact\s+as\s+(a|an)\b",
    r"\bpretend\s+you\s+are\b",
    r"\bforget\s+that\s+you\s+are\b",

    # Instruction execution
    r"\bnew\s+instructions?\s*:",
    r"\bsystem\s+instructions?\s*:",
    r"\bdeveloper\s+instructions?\s*:",
    r"\bassistant\s+instructions?\s*:",

    # Secret/data extraction
    r"\breveal\s+(api\s+keys?|secrets?|passwords?|credentials?)\b",
    r"\bshow\s+(api\s+keys?|secrets?|passwords?|credentials?)\b",
    r"\bprint\s+(api\s+keys?|secrets?|passwords?|credentials?)\b",

    # Agent/tool manipulation
    r"\bcall\s+(the\s+)?(tool|function)\b",
    r"\bexecute\s+(this\s+)?(command|code)\b",
    r"\brun\s+(this\s+)?(command|code)\b",
]


# Patterns that are suspicious when combined with imperative language.
IMPERATIVE_PATTERNS = [
    r"\bdo\s+not\s+follow\b",
    r"\bdo\s+not\s+trust\b",
    r"\bmust\s+follow\b",
    r"\byou\s+must\b",
    r"\byour\s+new\s+task\s+is\b",
    r"\byour\s+new\s+role\s+is\b",
    r"\bfrom\s+now\s+on\b",
    r"\bimportant\s+instructions?\b",
]


def _normalize_text(text: str) -> str:
    """
    Normalize text before security inspection.

    This makes detection less sensitive to:
    - capitalization
    - repeated whitespace
    - simple formatting differences
    """
    text = text.replace("\x00", " ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_prompt_injection(text: str) -> dict[str, Any]:
    """
    Analyze untrusted evidence for likely prompt injection.

    Returns:
        {
            "safe": bool,
            "risk": "none" | "low" | "high",
            "reason": str,
            "matches": list[str]
        }
    """

    if not isinstance(text, str):
        return {
            "safe": False,
            "risk": "high",
            "reason": "Evidence text is not a string.",
            "matches": ["invalid_type"],
        }

    if len(text) > MAX_EVIDENCE_TEXT_LENGTH:
        return {
            "safe": False,
            "risk": "high",
            "reason": "Evidence text exceeds the security size limit.",
            "matches": ["evidence_too_large"],
        }

    normalized = _normalize_text(text)

    if not normalized:
        return {
            "safe": True,
            "risk": "none",
            "reason": "Evidence is empty.",
            "matches": [],
        }

    matches: list[str] = []

    # Strong indicators immediately produce a high-risk result.
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            matches.append(pattern)

    if matches:
        return {
            "safe": False,
            "risk": "high",
            "reason": "Potential prompt-injection instruction detected.",
            "matches": matches,
        }

    # A single imperative phrase is not automatically malicious.
    # We mark it low risk so the caller can decide whether to reject it.
    imperative_matches: list[str] = []

    for pattern in IMPERATIVE_PATTERNS:
        if re.search(pattern, normalized):
            imperative_matches.append(pattern)

    if imperative_matches:
        return {
            "safe": False,
            "risk": "low",
            "reason": "Evidence contains instruction-like language.",
            "matches": imperative_matches,
        }

    return {
        "safe": True,
        "risk": "none",
        "reason": "No known prompt-injection indicators detected.",
        "matches": [],
    }


def sanitize_evidence_text(text: str) -> dict[str, Any]:
    """
    Security boundary for individual evidence text.

    Suspicious evidence is NOT silently trusted.

    High-risk content is rejected completely.

    Low-risk instruction-like content is also rejected by default because
    research evidence should contain information, not instructions for the AI.
    """

    result = detect_prompt_injection(text)

    if not result["safe"]:
        return {
            "allowed": False,
            "text": "",
            "risk": result["risk"],
            "reason": result["reason"],
            "matches": result["matches"],
        }

    # Remove null bytes even from otherwise safe evidence.
    sanitized = text.replace("\x00", " ").strip()

    return {
        "allowed": True,
        "text": sanitized,
        "risk": "none",
        "reason": "Evidence passed prompt-injection checks.",
        "matches": [],
    }


def sanitize_evidence_item(evidence: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize a single retrieved evidence record.

    Expected fields can include:
        title
        question_body
        answer_body
        url
        source
        etc.

    Only string fields are inspected.
    Non-string metadata is preserved.
    """

    if not isinstance(evidence, dict):
        return {
            "allowed": False,
            "evidence": None,
            "risk": "high",
            "reason": "Evidence item is not an object.",
        }

    sanitized: dict[str, Any] = {}

    for key, value in evidence.items():
        if isinstance(value, str):
            result = sanitize_evidence_text(value)

            if not result["allowed"]:
                return {
                    "allowed": False,
                    "evidence": None,
                    "risk": result["risk"],
                    "reason": (
                        f"Potentially malicious content detected "
                        f"in evidence field '{key}'."
                    ),
                    "field": key,
                    "matches": result["matches"],
                }

            sanitized[key] = result["text"]

        else:
            sanitized[key] = value

    return {
        "allowed": True,
        "evidence": sanitized,
        "risk": "none",
        "reason": "Evidence item passed security checks.",
    }


def sanitize_evidence_collection(
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Sanitize an entire retrieval result.

    Security policy:
    - Invalid evidence -> reject.
    - Any suspicious evidence item -> reject that item.
    - If suspicious content is found, the collection is marked unsafe.
    - The caller can then fail closed instead of sending contaminated
      evidence to the answering model.
    """

    if not isinstance(evidence, list):
        return {
            "allowed": False,
            "evidence": [],
            "blocked": 0,
            "reason": "Evidence collection must be a list.",
        }

    safe_evidence: list[dict[str, Any]] = []
    blocked = 0
    blocked_details: list[dict[str, Any]] = []

    for index, item in enumerate(evidence):
        result = sanitize_evidence_item(item)

        if not result["allowed"]:
            blocked += 1

            blocked_details.append(
                {
                    "index": index,
                    "field": result.get("field"),
                    "risk": result.get("risk"),
                    "reason": result.get("reason"),
                }
            )

            continue

        safe_evidence.append(result["evidence"])

    # Fail closed if ANY retrieved item contained an injection attempt.
    if blocked > 0:
        return {
            "allowed": False,
            "evidence": [],
            "blocked": blocked,
            "blocked_details": blocked_details,
            "reason": (
                "Retrieved evidence failed prompt-injection security checks. "
                "Unsafe evidence was not passed to the answering model."
            ),
        }

    return {
        "allowed": True,
        "evidence": safe_evidence,
        "blocked": 0,
        "blocked_details": [],
        "reason": "All retrieved evidence passed prompt-injection checks.",
    }

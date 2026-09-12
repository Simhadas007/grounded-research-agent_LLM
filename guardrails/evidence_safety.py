"""
Evidence safety guardrails for the Grounded Research Agent.

Purpose
-------
Retrieved web/social/API content is untrusted data. This module creates a
security boundary between external evidence and the reasoning model.

Design goals
------------
1. Detect prompt-injection attempts in content-bearing evidence fields.
2. Avoid false positives from ordinary URLs and metadata.
3. Hard-block strong injection patterns.
4. Use correlated signals for weaker/ambiguous injection language.
5. Validate URLs separately from natural-language content.
6. Bound evidence size to reduce prompt/token abuse.
7. Fail closed when an evidence item is genuinely unsafe.
8. Never execute, follow, or obey instructions contained in evidence.
9. Preserve structured Open-Meteo weather evidence.
10. Preserve the existing public API used by app.py and tests.

Public functions
----------------
detect_prompt_injection(text)
sanitize_evidence_text(text)
sanitize_evidence_item(item)
sanitize_evidence_collection(evidence)
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from typing import Any
from urllib.parse import urlparse


# ============================================================================
# Limits
# ============================================================================

MAX_EVIDENCE_TEXT_LENGTH = 12000
MAX_COLLECTION_ITEMS = 50
MAX_BLOCKED_DETAILS = 20
MAX_STRING_FIELDS_PER_ITEM = 40
MAX_METADATA_VALUE_LENGTH = 2000

# Weather-specific limits
MAX_WEATHER_LOCATION_LENGTH = 200
MAX_WEATHER_COUNTRY_LENGTH = 100
MAX_WEATHER_TIMEZONE_LENGTH = 100
MAX_WEATHER_TIME_LENGTH = 100


# ============================================================================
# Field classification
# ============================================================================

# Natural-language fields retrieved from external sources.
# These MUST be treated as untrusted content.
CONTENT_FIELDS = {
    "title",
    "name",
    "description",
    "content",
    "body",
    "text",
    "snippet",
    "summary",
    "question",
    "question_body",
    "answer",
    "answer_body",
    "comment",
    "comments",
    "top_comment",
    "top_comments",
    "reply",
    "replies",
    "post",
    "post_body",
    "article",
    "article_body",
    "message",
    "messages",
    "details",
    "reason",
}


# URL values are validated structurally rather than prompt-injection scanned.
URL_FIELDS = {
    "url",
    "link",
    "source_url",
    "canonical_url",
    "permalink",
    "href",
}


# Metadata fields.
METADATA_FIELDS = {
    "source",
    "source_type",
    "source_name",
    "domain",
    "author",
    "username",
    "user",
    "id",
    "post_id",
    "comment_id",
    "answer_id",
    "score",
    "upvotes",
    "downvotes",
    "rank",
    "location",
    "country",
    "timezone",
    "current_time",
    "time",
    "timestamp",
    "created_at",
    "updated_at",
    "tags",
    "category",
    "type",
    "weather_code",
}


# Structured fields that are intentionally handled separately.
#
# "current" is NOT treated as arbitrary metadata because the weather tool
# returns a structured dictionary containing trusted schema fields such as:
#
# {
#     "temperature_c": 30.2,
#     "relative_humidity_percent": 74,
#     "apparent_temperature_c": 35.6,
#     "precipitation_mm": 0.0,
#     "weather_code": 3,
#     "weather_description": "Overcast",
#     "wind_speed_kmh": 10.8,
#     "time": "2026-09-11T10:15"
# }
#
# The values still originate from an external API, so they must be bounded
# and validated. They are never treated as instructions.
STRUCTURED_FIELDS = {
    "current",
}


# ============================================================================
# Prompt-injection detection
# ============================================================================

# Strong indicators.
#
# A single strong indicator is enough to block the content because these
# phrases are characteristic of attempts to control the assistant rather
# than ordinary factual content.
STRONG_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bignore\s+(all\s+)?prior\s+instructions\b",
    r"\bdisregard\s+(all\s+)?previous\s+instructions\b",
    r"\bdisregard\s+(all\s+)?prior\s+instructions\b",
    r"\bforget\s+(all\s+)?previous\s+instructions\b",
    r"\bforget\s+(all\s+)?prior\s+instructions\b",

    r"\bignore\s+(the\s+)?system\s+(message|prompt|instructions)\b",
    r"\bdisregard\s+(the\s+)?system\s+(message|prompt|instructions)\b",
    r"\boverride\s+(the\s+)?system\s+(message|prompt|instructions)\b",

    r"\bignore\s+(the\s+)?developer\s+(message|prompt|instructions)\b",
    r"\bdisregard\s+(the\s+)?developer\s+(message|prompt|instructions)\b",
    r"\boverride\s+(the\s+)?developer\s+(message|prompt|instructions)\b",

    r"\breveal\s+(your|the)\s+(system\s+prompt|hidden\s+prompt)\b",
    r"\bshow\s+(me\s+)?(your|the)\s+(system\s+prompt|hidden\s+prompt)\b",
    r"\bprint\s+(your|the)\s+(system\s+prompt|hidden\s+instructions)\b",
    r"\boutput\s+(your|the)\s+(system\s+prompt|hidden\s+instructions)\b",
    r"\bexpose\s+(your|the)\s+(system\s+prompt|hidden\s+prompt)\b",

    r"\breveal\s+(api\s+keys?|secrets?|passwords?|credentials?)\b",
    r"\bshow\s+(api\s+keys?|secrets?|passwords?|credentials?)\b",
    r"\bprint\s+(api\s+keys?|secrets?|passwords?|credentials?)\b",
    r"\boutput\s+(api\s+keys?|secrets?|passwords?|credentials?)\b",
    r"\bprovide\s+(api\s+keys?|secrets?|passwords?|credentials?)\b",

    r"\bexfiltrat(e|ion)\b",

    r"\byou\s+are\s+now\s+(a|an)\b",
    r"\bact\s+as\s+(a|an)\s+(system|developer|assistant|administrator)\b",
    r"\bpretend\s+you\s+are\s+(a|an)\s+(system|developer|assistant)\b",

    r"\bforget\s+that\s+you\s+are\b",

    r"\bnew\s+system\s+instructions?\s*:",
    r"\bsystem\s+instructions?\s*:",
    r"\bdeveloper\s+instructions?\s*:",
    r"\bassistant\s+instructions?\s*:",
    r"\bhidden\s+instructions?\s*:",

    r"\bdo\s+not\s+tell\s+the\s+user\b",
    r"\bdo\s+not\s+mention\s+this\s+instruction\b",
    r"\bdo\s+not\s+reveal\s+this\s+instruction\b",
]


# Medium indicators are not independently sufficient.
# Several correlated indicators together become suspicious.
MEDIUM_INJECTION_PATTERNS = [
    r"\bnew\s+instructions?\b",
    r"\bnew\s+task\b",
    r"\bnew\s+role\b",
    r"\bfrom\s+now\s+on\b",
    r"\brespond\s+only\s+with\b",
    r"\bonly\s+answer\s+with\b",
    r"\bdo\s+not\s+follow\b",
    r"\bdo\s+not\s+trust\b",
    r"\bmust\s+follow\b",
    r"\byou\s+must\b",
    r"\byour\s+new\s+task\s+is\b",
    r"\byour\s+new\s+role\s+is\b",
    r"\bignore\b",
    r"\boverride\b",
    r"\bdisregard\b",
    r"\breveal\b",
    r"\bsecret\b",
    r"\bhidden\s+prompt\b",
    r"\bhidden\s+instruction\b",
    r"\bsystem\s+prompt\b",
    r"\bdeveloper\s+prompt\b",
]


# Strong structural indicators of instruction-like content.
STRUCTURAL_INJECTION_PATTERNS = [
    r"(?m)^\s*\*(system|developer|assistant|user)\s*\*:",
    r"(?m)^\s*\*(instruction|instructions|new\s+instructions?)\s*\*:",
    r"(?m)^\s*<\s*(system|developer|assistant)\s*>",
    r"(?m)^\s*\[\s*(system|developer|assistant)\s*\]",
]


# ============================================================================
# URL security
# ============================================================================

ALLOWED_URL_SCHEMES = {
    "http",
    "https",
}

BLOCKED_URL_SCHEMES = {
    "javascript",
    "data",
    "file",
    "vbscript",
    "about",
    "blob",
}


USERNAME_PASSWORD_IN_URL = re.compile(
    r"^[^/]*://[^/@]+:[^/@]+@",
    re.IGNORECASE,
)


# ============================================================================
# Utility functions
# ============================================================================

def _normalize_text(value: str) -> str:
    """
    Normalize text for security inspection without destroying its meaning.

    Unicode normalization reduces trivial obfuscation using compatibility
    characters. Whitespace normalization helps detect injections split across
    newlines/tabs.
    """

    value = unicodedata.normalize("NFKC", value)

    # Remove null/control characters while preserving normal whitespace.
    value = "".join(
        char
        for char in value
        if (
            char in "\n\r\t"
            or not unicodedata.category(char).startswith("C")
        )
    )

    value = html.unescape(value)

    # Collapse repeated whitespace for pattern matching.
    value = re.sub(r"\s+", " ", value)

    return value.strip().lower()


def _clean_text(value: str) -> str:
    """
    Produce bounded, harmless text for downstream use.

    This is not a trust decision. Security classification happens separately.
    """

    value = unicodedata.normalize("NFKC", value)

    # Null bytes are never useful in retrieved textual evidence.
    value = value.replace("\x00", "")

    if len(value) > MAX_EVIDENCE_TEXT_LENGTH:
        value = value[:MAX_EVIDENCE_TEXT_LENGTH]

    return value


def _match_patterns(
    text: str,
    patterns: list[str],
) -> list[str]:
    """Return all static patterns that match the supplied text."""

    matches: list[str] = []

    for pattern in patterns:
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                matches.append(pattern)
        except re.error:
            # Internal static patterns should never be malformed.
            # If one somehow is, fail safely by recording it.
            matches.append(pattern)

    return matches


def _validate_url(value: str) -> tuple[bool, str]:
    """
    Validate an evidence URL structurally.

    URLs are metadata, not natural-language instructions, so they are not
    passed through prompt-injection phrase detection.
    """

    if not isinstance(value, str):
        return False, "URL is not a string."

    value = value.strip()

    if not value:
        return False, "URL is empty."

    if len(value) > MAX_METADATA_VALUE_LENGTH:
        return False, "URL is too long."

    try:
        parsed = urlparse(value)
    except Exception:
        return False, "URL could not be parsed."

    scheme = parsed.scheme.lower()

    if scheme in BLOCKED_URL_SCHEMES:
        return False, f"Blocked URL scheme: {scheme}."

    if scheme not in ALLOWED_URL_SCHEMES:
        return False, "URL must use http or https."

    if not parsed.netloc:
        return False, "URL has no network location."

    # Do not allow userinfo/credentials in evidence URLs.
    if parsed.username is not None or parsed.password is not None:
        return False, "URLs containing credentials are not allowed."

    if USERNAME_PASSWORD_IN_URL.search(value):
        return False, "URLs containing embedded credentials are not allowed."

    hostname = parsed.hostname

    if not hostname:
        return False, "URL hostname is missing."

    if any(char.isspace() for char in hostname):
        return False, "URL hostname contains whitespace."

    return True, ""


def _is_finite_number(value: Any) -> bool:
    """Return True only for finite int/float values."""

    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return math.isfinite(float(value))


def _bounded_float(
    value: Any,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """
    Validate and bound a numeric weather value.

    The value must be numeric, finite, and within the physically reasonable
    range expected for the specific field.
    """

    if not _is_finite_number(value):
        raise ValueError("Weather numeric value is invalid.")

    number = float(value)

    if number < minimum or number > maximum:
        raise ValueError("Weather numeric value is outside the allowed range.")

    return number


def _bounded_string(
    value: Any,
    *,
    field_name: str,
    maximum_length: int,
) -> str:
    """Validate and bound a simple structured string."""

    if not isinstance(value, str):
        raise ValueError(
            f"Weather field '{field_name}' must be a string."
        )

    value = _clean_text(value).strip()

    if len(value) > maximum_length:
        raise ValueError(
            f"Weather field '{field_name}' exceeds the allowed length."
        )

    return value


# ============================================================================
# Weather structured evidence
# ============================================================================

def _sanitize_weather_current(current: Any) -> dict[str, Any]:
    """
    Sanitize the structured Open-Meteo 'current' object.

    Security principle
    ------------------
    Weather values are external data, not instructions.

    Only known numeric/string fields are copied. Arbitrary nested objects,
    lists, or unknown fields are discarded.

    Required fields
    ---------------
    temperature_c

    Optional fields
    ---------------
    time
    relative_humidity_percent
    apparent_temperature_c
    precipitation_mm
    weather_code
    weather_description
    wind_speed_kmh
    """

    if not isinstance(current, dict):
        raise ValueError("Weather 'current' field must be a dictionary.")

    sanitized: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Required temperature
    # ------------------------------------------------------------------

    if "temperature_c" not in current:
        raise ValueError(
            "Weather evidence is missing current temperature."
        )

    sanitized["temperature_c"] = _bounded_float(
        current["temperature_c"],
        minimum=-100.0,
        maximum=100.0,
    )

    # ------------------------------------------------------------------
    # Observation time
    # ------------------------------------------------------------------

    if current.get("time") is not None:
        sanitized["time"] = _bounded_string(
            current["time"],
            field_name="time",
            maximum_length=MAX_WEATHER_TIME_LENGTH,
        )

    # ------------------------------------------------------------------
    # Relative humidity
    # ------------------------------------------------------------------

    if current.get("relative_humidity_percent") is not None:
        sanitized["relative_humidity_percent"] = _bounded_float(
            current["relative_humidity_percent"],
            minimum=0.0,
            maximum=100.0,
        )

    # ------------------------------------------------------------------
    # Apparent temperature
    # ------------------------------------------------------------------

    if current.get("apparent_temperature_c") is not None:
        sanitized["apparent_temperature_c"] = _bounded_float(
            current["apparent_temperature_c"],
            minimum=-100.0,
            maximum=100.0,
        )

    # ------------------------------------------------------------------
    # Precipitation
    # ------------------------------------------------------------------

    if current.get("precipitation_mm") is not None:
        sanitized["precipitation_mm"] = _bounded_float(
            current["precipitation_mm"],
            minimum=0.0,
            maximum=10000.0,
        )

    # ------------------------------------------------------------------
    # Weather code
    # ------------------------------------------------------------------

    if current.get("weather_code") is not None:
        weather_code = current["weather_code"]

        if not _is_finite_number(weather_code):
            raise ValueError("Weather code is invalid.")

        weather_code_number = int(weather_code)

        if weather_code_number < 0 or weather_code_number > 99:
            raise ValueError("Weather code is outside the allowed range.")

        sanitized["weather_code"] = weather_code_number

    # ------------------------------------------------------------------
    # Weather description
    #
    # This is externally supplied natural-language data, so it MUST still
    # pass prompt-injection detection.
    # ------------------------------------------------------------------

    if current.get("weather_description") is not None:
        description = current["weather_description"]

        if not isinstance(description, str):
            raise ValueError(
                "Weather description must be a string."
            )

        sanitized_description = sanitize_evidence_text(description)

        if len(sanitized_description) > 500:
            raise ValueError(
                "Weather description exceeds the allowed length."
            )

        sanitized["weather_description"] = sanitized_description

    # ------------------------------------------------------------------
    # Wind speed
    # ------------------------------------------------------------------

    if current.get("wind_speed_kmh") is not None:
        sanitized["wind_speed_kmh"] = _bounded_float(
            current["wind_speed_kmh"],
            minimum=0.0,
            maximum=1000.0,
        )

    return sanitized


def _sanitize_weather_evidence(item: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize one structured Open-Meteo evidence item.

    The output intentionally contains only the schema required by the
    research-answer layer.
    """

    source_type = item.get("source_type")

    source = item.get("source")

    # Recognize Open-Meteo weather evidence.
    is_weather = (
        source_type == "weather"
        or source == "Open-Meteo"
    )

    if not is_weather:
        raise ValueError("Not a weather evidence item.")

    sanitized: dict[str, Any] = {
        "source_type": "weather",
    }

    # ------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------

    if source is not None:
        sanitized["source"] = _bounded_string(
            source,
            field_name="source",
            maximum_length=MAX_METADATA_VALUE_LENGTH,
        )

    # ------------------------------------------------------------------
    # URL
    # ------------------------------------------------------------------

    if item.get("url") is not None:
        valid, reason = _validate_url(item["url"])

        if not valid:
            raise ValueError(
                f"Unsafe evidence URL: {reason}"
            )

        sanitized["url"] = item["url"].strip()

    # ------------------------------------------------------------------
    # Location
    # ------------------------------------------------------------------

    if item.get("location") is not None:
        sanitized["location"] = _bounded_string(
            item["location"],
            field_name="location",
            maximum_length=MAX_WEATHER_LOCATION_LENGTH,
        )

    # ------------------------------------------------------------------
    # Country
    # ------------------------------------------------------------------

    if item.get("country") is not None:
        sanitized["country"] = _bounded_string(
            item["country"],
            field_name="country",
            maximum_length=MAX_WEATHER_COUNTRY_LENGTH,
        )

    # ------------------------------------------------------------------
    # Latitude / longitude
    # ------------------------------------------------------------------

    if item.get("latitude") is not None:
        sanitized["latitude"] = _bounded_float(
            item["latitude"],
            minimum=-90.0,
            maximum=90.0,
        )

    if item.get("longitude") is not None:
        sanitized["longitude"] = _bounded_float(
            item["longitude"],
            minimum=-180.0,
            maximum=180.0,
        )

    # ------------------------------------------------------------------
    # Timezone
    # ------------------------------------------------------------------

    if item.get("timezone") is not None:
        sanitized["timezone"] = _bounded_string(
            item["timezone"],
            field_name="timezone",
            maximum_length=MAX_WEATHER_TIMEZONE_LENGTH,
        )

    # ------------------------------------------------------------------
    # Current weather data
    #
    # THIS IS THE CRITICAL FIX.
    #
    # Keep current as a dictionary rather than passing it through
    # _safe_metadata_value(), which would stringify the dictionary.
    # ------------------------------------------------------------------

    if "current" not in item:
        raise ValueError(
            "Weather evidence is missing the current weather object."
        )

    sanitized["current"] = _sanitize_weather_current(
        item["current"]
    )

    return sanitized


# ============================================================================
# Public API: prompt injection detection
# ============================================================================

def detect_prompt_injection(text: Any) -> dict[str, Any]:
    """
    Detect high-confidence prompt injection in untrusted retrieved content.

    Policy:
    - Strong injection patterns are always blocked.
    - Structural role/instruction markers are always blocked.
    - Medium phrases are NOT dangerous by themselves.
    - Medium signals are blocked only when they form a clearly
      instruction-manipulation combination.
    """

    if not isinstance(text, str):
        return {
            "safe": False,
            "risk": "high",
            "reason": "Evidence text is not a string.",
        }

    if len(text) > MAX_EVIDENCE_TEXT_LENGTH:
        return {
            "safe": False,
            "risk": "high",
            "reason": "Evidence text exceeds the security size limit.",
        }

    normalized = _normalize_text(text)

    if not normalized:
        return {
            "safe": True,
            "risk": "none",
            "reason": "Empty evidence text.",
        }

    # ---------------------------------------------------------------
    # 1. Strong indicators — ALWAYS BLOCK
    # ---------------------------------------------------------------

    strong_matches = _match_patterns(
        normalized,
        STRONG_INJECTION_PATTERNS,
    )

    if strong_matches:
        return {
            "safe": False,
            "risk": "high",
            "reason": "Strong prompt-injection pattern detected.",
        }

    # ---------------------------------------------------------------
    # 2. Structural role markers — ALWAYS BLOCK
    # ---------------------------------------------------------------

    structural_matches = _match_patterns(
        normalized,
        STRUCTURAL_INJECTION_PATTERNS,
    )

    if structural_matches:
        return {
            "safe": False,
            "risk": "high",
            "reason": (
                "Instruction-role marker detected in retrieved content."
            ),
        }

    # ---------------------------------------------------------------
    # 3. Medium signals
    #
    # These phrases are common in legitimate articles.
    # Examples:
    #   "you must..."
    #   "new task"
    #   "from now on"
    #
    # Therefore we only block when there is a strong combination
    # indicating an attempt to control the assistant.
    # ---------------------------------------------------------------

    medium_matches = _match_patterns(
        normalized,
        MEDIUM_INJECTION_PATTERNS,
    )

    medium_text = set(medium_matches)

    # ---------------------------------------------------------------
    # High-confidence combinations
    # ---------------------------------------------------------------

    manipulation_words = {
        r"\bignore\b",
        r"\boverride\b",
        r"\bdisregard\b",
        r"\bdo\s+not\s+follow\b",
        r"\bdo\s+not\s+trust\b",
    }

    role_control_words = {
        r"\bnew\s+instructions?\b",
        r"\bnew\s+task\b",
        r"\bnew\s+role\b",
        r"\byour\s+new\s+task\s+is\b",
        r"\byour\s+new\s+role\s+is\b",
    }

    secret_control_words = {
        r"\breveal\b",
        r"\bsecret\b",
        r"\bhidden\s+prompt\b",
        r"\bhidden\s+instruction\b",
        r"\bsystem\s+prompt\b",
        r"\bdeveloper\s+prompt\b",
    }

    response_control_words = {
        r"\brespond\s+only\s+with\b",
        r"\bonly\s+answer\s+with\b",
    }

    has_manipulation = bool(medium_text & manipulation_words)
    has_role_control = bool(medium_text & role_control_words)
    has_secret_control = bool(medium_text & secret_control_words)
    has_response_control = bool(medium_text & response_control_words)

    # Explicit instruction manipulation.
    if has_manipulation and (
        has_role_control
        or has_secret_control
        or has_response_control
    ):
        return {
            "safe": False,
            "risk": "high",
            "reason": (
                "Correlated instruction-manipulation indicators "
                "detected in retrieved content."
            ),
        }

    # Explicit role/task takeover.
    if has_role_control and has_response_control:
        return {
            "safe": False,
            "risk": "high",
            "reason": (
                "Retrieved content contains coordinated "
                "role or response manipulation."
            ),
        }

    # ---------------------------------------------------------------
    # 4. Ordinary medium phrases are retained.
    # ---------------------------------------------------------------

    if medium_matches:
        return {
            "safe": True,
            "risk": "low",
            "reason": (
                "Low-confidence instruction-like language detected, "
                "but no high-confidence prompt injection combination."
            ),
        }

    return {
        "safe": True,
        "risk": "none",
        "reason": "No prompt-injection indicators detected.",
    }


# ============================================================================
# Public API: text sanitization
# ============================================================================

def sanitize_evidence_text(text: Any) -> str:
    """
    Validate and sanitize one piece of retrieved evidence text.

    Raises
    ------
    ValueError
        If the text is invalid or contains a high-confidence injection.
    """

    if not isinstance(text, str):
        raise ValueError("Evidence text must be a string.")

    if len(text) > MAX_EVIDENCE_TEXT_LENGTH:
        raise ValueError(
            "Evidence text exceeds the security size limit."
        )

    decision = detect_prompt_injection(text)

    if not decision["safe"]:
        raise ValueError(
            "Retrieved evidence contains unsafe instruction-like content."
        )

    return _clean_text(text)


# ============================================================================
# Evidence item helpers
# ============================================================================

def _safe_metadata_value(value: Any) -> Any:
    """
    Bound metadata while preserving simple JSON-compatible values.

    Important:
    ----------
    This function is intentionally NOT used for structured weather.current.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    if isinstance(value, str):
        return value[:MAX_METADATA_VALUE_LENGTH]

    if isinstance(value, list):
        bounded: list[Any] = []

        for item in value[:50]:
            if (
                isinstance(item, (str, int, float, bool))
                or item is None
            ):
                if isinstance(item, str):
                    bounded.append(item[:500])
                else:
                    bounded.append(item)

        return bounded

    # Avoid passing arbitrary nested objects into downstream prompts.
    return str(value)[:MAX_METADATA_VALUE_LENGTH]


def _sanitize_content_field(
    field_name: str,
    value: Any,
) -> tuple[bool, Any, str]:
    """
    Sanitize one content-bearing field.
    """

    if value is None:
        return True, value, ""

    if not isinstance(value, str):
        # Lists such as comments/replies can legitimately contain strings.
        if isinstance(value, list):
            sanitized_items: list[Any] = []

            for item in value[:50]:
                if isinstance(item, str):
                    try:
                        sanitized_items.append(
                            sanitize_evidence_text(item)
                        )
                    except ValueError:
                        return (
                            False,
                            None,
                            (
                                f"Unsafe content detected in field "
                                f"'{field_name}'."
                            ),
                        )

                elif (
                    isinstance(item, (int, float, bool))
                    or item is None
                ):
                    sanitized_items.append(item)

            return True, sanitized_items, ""

        return (
            False,
            None,
            f"Content field '{field_name}' must contain text.",
        )

    try:
        return True, sanitize_evidence_text(value), ""

    except ValueError as exc:
        return False, None, str(exc)


# ============================================================================
# Public API: evidence item
# ============================================================================

def sanitize_evidence_item(item: Any) -> dict[str, Any]:
    """
    Sanitize a single retrieved evidence item.

    Security boundary
    -----------------
    - Content fields are prompt-injection scanned.
    - URL fields are structurally validated.
    - Metadata is bounded.
    - Structured weather data is schema-validated.
    - Unknown string fields are treated conservatively as content.

    Raises
    ------
    ValueError
        If the item contains unsafe content or an unsafe URL.
    """

    if not isinstance(item, dict):
        raise ValueError("Evidence item must be a dictionary.")

    if len(item) > MAX_STRING_FIELDS_PER_ITEM:
        raise ValueError(
            "Evidence item contains too many fields."
        )

    # ------------------------------------------------------------------
    # Weather branch
    #
    # This MUST happen before the generic metadata handling because
    # "current" is a nested structured object.
    # ------------------------------------------------------------------

    source_type = item.get("source_type")
    source = item.get("source")

    if (
        source_type == "weather"
        or source == "Open-Meteo"
    ):
        return _sanitize_weather_evidence(item)

    # ------------------------------------------------------------------
    # General evidence
    # ------------------------------------------------------------------

    sanitized: dict[str, Any] = {}

    for raw_key, raw_value in item.items():

        if not isinstance(raw_key, str):
            raise ValueError(
                "Evidence field names must be strings."
            )

        field_name = raw_key.strip().lower()

        if not field_name:
            continue

        # --------------------------------------------------------------
        # URL fields
        # --------------------------------------------------------------

        if field_name in URL_FIELDS:

            if raw_value is None:
                continue

            valid, reason = _validate_url(
                str(raw_value)
            )

            if not valid:
                raise ValueError(
                    (
                        f"Unsafe evidence URL in field "
                        f"'{field_name}': {reason}"
                    )
                )

            sanitized[field_name] = str(
                raw_value
            ).strip()

            continue

        # --------------------------------------------------------------
        # Explicit structured fields
        # --------------------------------------------------------------

        if field_name in STRUCTURED_FIELDS:

            # A structured field that is not recognized by a dedicated
            # sanitizer is rejected rather than stringified.
            raise ValueError(
                (
                    f"Structured evidence field '{field_name}' "
                    "has an unsupported schema."
                )
            )

        # --------------------------------------------------------------
        # Explicit content fields
        # --------------------------------------------------------------

        if field_name in CONTENT_FIELDS:

            allowed, value, reason = _sanitize_content_field(
                field_name,
                raw_value,
            )

            if not allowed:
                raise ValueError(reason)

            sanitized[field_name] = value

            continue

        # --------------------------------------------------------------
        # Known metadata
        # --------------------------------------------------------------

        if field_name in METADATA_FIELDS:

            sanitized[field_name] = _safe_metadata_value(
                raw_value
            )

            continue

        # --------------------------------------------------------------
        # Unknown fields
        #
        # We cannot safely assume an unknown string field is metadata.
        # Therefore strings are treated as untrusted content.
        # --------------------------------------------------------------

        if isinstance(raw_value, str):

            try:
                sanitized[field_name] = sanitize_evidence_text(
                    raw_value
                )

            except ValueError as exc:

                raise ValueError(
                    (
                        f"Unsafe content in evidence field "
                        f"'{field_name}': {exc}"
                    )
                ) from exc

        else:

            sanitized[field_name] = _safe_metadata_value(
                raw_value
            )

    return sanitized


# ============================================================================
# Public API: evidence collection
# ============================================================================

def sanitize_evidence_collection(
    evidence: Any,
) -> dict[str, Any]:
    """
    Apply the evidence security boundary to a collection.

    Security policy
    ---------------
    The collection fails closed if ANY evidence item is unsafe.

    This is intentional. A downstream model must never receive a mixture of
    verified and unverified evidence when the application claims that the
    evidence set passed its security boundary.

    Returns
    -------
    dict
        {
            "allowed": bool,
            "evidence": list[dict],
            "blocked": int,
            "blocked_details": list[dict],
        }
    """

    if evidence is None:
        return {
            "allowed": True,
            "evidence": [],
            "blocked": 0,
            "blocked_details": [],
        }

    if not isinstance(evidence, list):
        return {
            "allowed": False,
            "evidence": [],
            "blocked": 1,
            "blocked_details": [
                {
                    "index": None,
                    "reason": (
                        "Evidence collection is not a list."
                    ),
                }
            ],
        }

    if len(evidence) > MAX_COLLECTION_ITEMS:
        return {
            "allowed": False,
            "evidence": [],
            "blocked": 1,
            "blocked_details": [
                {
                    "index": None,
                    "reason": (
                        "Evidence collection exceeds the item limit."
                    ),
                }
            ],
        }

    sanitized_evidence: list[dict[str, Any]] = []
    blocked_details: list[dict[str, Any]] = []

    for index, item in enumerate(evidence):

        try:
            sanitized_item = sanitize_evidence_item(item)

            sanitized_evidence.append(
                sanitized_item
            )

        except Exception as exc:

            if len(blocked_details) < MAX_BLOCKED_DETAILS:
                blocked_details.append(
                    {
                        "index": index,
                        "reason": str(exc)[:500],
                    }
                )

    blocked = len(evidence) - len(sanitized_evidence)

    # Fail closed if ANY item failed the trust boundary.
    if blocked > 0:
        return {
            "allowed": False,
            "evidence": [],
            "blocked": blocked,
            "blocked_details": blocked_details,
        }

    return {
        "allowed": True,
        "evidence": sanitized_evidence,
        "blocked": 0,
        "blocked_details": [],
    }
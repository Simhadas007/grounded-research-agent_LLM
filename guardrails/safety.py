import re


MAX_INPUT_LENGTH = 2000


BLOCKED_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?system\s+instructions",
    r"reveal\s+(your|the)\s+(system\s+prompt|hidden\s+instructions)",
    r"show\s+(me\s+)?your\s+(system\s+prompt|hidden\s+instructions)",
    r"disregard\s+(all\s+)?previous\s+instructions",
]


def check_user_input(question: str) -> dict:
    """
    Validate and perform basic security checks on user input.

    Returns:
        {
            "allowed": bool,
            "reason": str
        }
    """

    # Type validation
    if not isinstance(question, str):
        return {
            "allowed": False,
            "reason": "Invalid input type."
        }

    # Remove surrounding whitespace
    question = question.strip()

    # Empty input
    if not question:
        return {
            "allowed": False,
            "reason": "Question cannot be empty."
        }

    # Length protection
    if len(question) > MAX_INPUT_LENGTH:
        return {
            "allowed": False,
            "reason": "Question is too long."
        }

    # Prompt-injection detection
    normalized = question.lower()

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, normalized):
            return {
                "allowed": False,
                "reason": "Potential prompt injection detected."
            }

    return {
        "allowed": True,
        "reason": "Input passed security checks."
    }
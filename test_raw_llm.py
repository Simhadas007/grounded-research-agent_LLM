
"""
Offline smoke tests for the configured LLM.

IMPORTANT:
This file intentionally does NOT make a live Groq API request.

Live provider testing should be performed manually when needed,
rather than automatically during pytest collection.
"""

from agent import llm


def test_llm_is_configured():
    """
    Verify that the application has created an LLM instance.
    """

    assert llm is not None


def test_llm_has_model_name():
    """
    Verify that the configured model is present.
    """

    assert getattr(llm, "model", None)


def test_llm_temperature_is_deterministic():
    """
    The research agent should use deterministic generation.
    """

def test_llm_temperature_is_deterministic():
    temperature = getattr(llm, "temperature", None)
    assert temperature is not None
    assert temperature < 1e-6

def test_llm_has_token_limit():
    """
    Verify that generated responses have a configured token bound.
    """

    max_tokens = getattr(
        llm,
        "max_tokens",
        None,
    )

    assert max_tokens is not None
    assert max_tokens > 0

import json
from unittest.mock import patch

import pytest

from router import (
    SearchPlan,
    _extract_json,
    _validate_plan,
    create_search_plan,
    route_question,
)


# ============================================================================
# route_question() compatibility tests
# ============================================================================


def test_route_question_stackexchange():
    """Programming questions should route to Stack Exchange."""

    fake_plan = SearchPlan(
        route="stackexchange",
        search_query="React useEffect running twice in development mode",
        location="",
        reason="Programming question about React behavior",
    )

    with patch("router.create_search_plan", return_value=fake_plan) as mock_router:
        result = route_question(
            "Why is my React useEffect running twice?"
        )

    assert result.route == "stackexchange"
    assert result.search_query == (
        "React useEffect running twice in development mode"
    )
    assert result.location == ""
    assert result.reason == "Programming question about React behavior"

    mock_router.assert_called_once_with(
        "Why is my React useEffect running twice?"
    )


def test_route_question_weather():
    """Weather questions should route to the weather source."""

    fake_plan = SearchPlan(
        route="weather",
        search_query="",
        location="Chennai",
        reason="Weather question for Chennai",
    )

    with patch("router.create_search_plan", return_value=fake_plan) as mock_router:
        result = route_question(
            "What is the weather in Chennai today?"
        )

    assert result.route == "weather"
    assert result.search_query == ""
    assert result.location == "Chennai"
    assert result.reason == "Weather question for Chennai"

    mock_router.assert_called_once_with(
        "What is the weather in Chennai today?"
    )


def test_route_question_both():
    """Questions needing both discussion and weather should support both."""

    fake_plan = SearchPlan(
        route="both",
        search_query="Chennai umbrella recommendations",
        location="Chennai",
        reason="Question combines weather and practical discussion.",
    )

    with patch("router.create_search_plan", return_value=fake_plan) as mock_router:
        result = route_question(
            "What is the weather in Chennai and should I take an umbrella?"
        )

    assert result.route == "both"
    assert result.search_query == "Chennai umbrella recommendations"
    assert result.location == "Chennai"

    mock_router.assert_called_once_with(
        "What is the weather in Chennai and should I take an umbrella?"
    )


def test_route_question_unsupported():
    """Unsupported questions should be represented as unsupported."""

    fake_plan = SearchPlan(
        route="unsupported",
        search_query="",
        location="",
        reason="The available sources cannot ground this request.",
    )

    with patch("router.create_search_plan", return_value=fake_plan) as mock_router:
        result = route_question("Tell me a funny joke.")

    assert result.route == "unsupported"
    assert result.search_query == ""
    assert result.location == ""
    assert result.reason == (
        "The available sources cannot ground this request."
    )

    mock_router.assert_called_once_with(
        "Tell me a funny joke."
    )


def test_route_question_tavily():
    """General/current web research should support the Tavily route."""

    fake_plan = SearchPlan(
        route="tavily",
        search_query="latest React security vulnerabilities",
        location="",
        reason="Current general web research question.",
    )

    with patch("router.create_search_plan", return_value=fake_plan) as mock_router:
        result = route_question(
            "What are the latest React security vulnerabilities?"
        )

    assert result.route == "tavily"
    assert result.search_query == "latest React security vulnerabilities"
    assert result.location == ""
    assert result.reason == "Current general web research question."

    mock_router.assert_called_once_with(
        "What are the latest React security vulnerabilities?"
    )


def test_route_question_preserves_search_plan():
    """route_question should return the planner's complete SearchPlan."""

    fake_plan = SearchPlan(
        route="stackexchange",
        search_query="Python FastAPI dependency injection",
        location="",
        reason="Technical programming question.",
    )

    with patch("router.create_search_plan", return_value=fake_plan):
        result = route_question(
            "How does dependency injection work in FastAPI?"
        )

    assert isinstance(result, SearchPlan)
    assert result == fake_plan


# ============================================================================
# Test helpers
# ============================================================================


def _fake_llm(response_content: str):
    """Create a minimal fake LLM object for router tests."""

    class FakeResponse:
        content = response_content

    class FakeLLM:
        def invoke(self, prompt):
            return FakeResponse()

    return FakeLLM()


# ============================================================================
# create_search_plan() tests
# ============================================================================


def test_create_search_plan_tavily(monkeypatch):
    """
    The router should correctly parse an LLM-generated Tavily plan.

    The complete router.llm object is replaced because ChatGroq is a
    Pydantic model and its invoke attribute cannot be directly replaced.
    """

    response = json.dumps(
        {
            "route": "tavily",
            "search_query": "latest React security vulnerabilities",
            "location": "",
            "reason": "Current general web research question.",
        }
    )

    fake_llm = _fake_llm(response)

    monkeypatch.setattr("router.llm", fake_llm)

    plan = create_search_plan(
        "What are the latest React security vulnerabilities?"
    )

    assert isinstance(plan, SearchPlan)
    assert plan.route == "tavily"
    assert plan.search_query == "latest React security vulnerabilities"
    assert plan.location == ""
    assert plan.reason == "Current general web research question."


def test_create_search_plan_tavily_clears_location(monkeypatch):
    """
    Tavily does not use weather location.

    Even if the model incorrectly supplies a location, validation should
    normalize it to an empty string.
    """

    response = json.dumps(
        {
            "route": "tavily",
            "search_query": "latest cybersecurity news",
            "location": "Chennai",
            "reason": "General current web research.",
        }
    )

    monkeypatch.setattr("router.llm", _fake_llm(response))

    plan = create_search_plan(
        "What are the latest cybersecurity news stories?"
    )

    assert plan.route == "tavily"
    assert plan.search_query == "latest cybersecurity news"
    assert plan.location == ""


def test_create_search_plan_stackexchange_clears_location(monkeypatch):
    """Stack Exchange plans should never retain a weather location."""

    response = json.dumps(
        {
            "route": "stackexchange",
            "search_query": "FastAPI dependency injection",
            "location": "Chennai",
            "reason": "Programming question.",
        }
    )

    monkeypatch.setattr("router.llm", _fake_llm(response))

    plan = create_search_plan(
        "How does dependency injection work in FastAPI?"
    )

    assert plan.route == "stackexchange"
    assert plan.location == ""


def test_create_search_plan_rejects_empty_question():
    """Empty user input should be rejected before the LLM is called."""

    with pytest.raises(ValueError, match="Question cannot be empty"):
        create_search_plan("")


def test_create_search_plan_rejects_whitespace_question():
    """Whitespace-only input should be rejected."""

    with pytest.raises(ValueError, match="Question cannot be empty"):
        create_search_plan("     ")


def test_create_search_plan_rejects_oversized_question():
    """Questions above the router input limit should be rejected."""

    question = "A" * 2001

    with pytest.raises(ValueError, match="Question is too long"):
        create_search_plan(question)


def test_create_search_plan_calls_llm(monkeypatch):
    """The planner must actually invoke the configured LLM."""

    response = json.dumps(
        {
            "route": "tavily",
            "search_query": "latest AI news",
            "location": "",
            "reason": "Current web research.",
        }
    )

    class FakeResponse:
        content = response

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def invoke(self, prompt):
            self.calls.append(prompt)
            return FakeResponse()

    fake_llm = FakeLLM()

    monkeypatch.setattr("router.llm", fake_llm)

    question = "What are the latest AI developments?"

    plan = create_search_plan(question)

    assert plan.route == "tavily"
    assert len(fake_llm.calls) == 1

    prompt = fake_llm.calls[0]

    assert "Router and Planner" in prompt
    assert question in prompt
    assert "untrusted user data" in prompt


# ============================================================================
# JSON extraction tests
# ============================================================================


def test_extract_json_plain_object():
    """Plain JSON should be parsed successfully."""

    content = """
    {
        "route": "tavily",
        "search_query": "latest AI news",
        "location": "",
        "reason": "Current web research"
    }
    """

    result = _extract_json(content)

    assert result["route"] == "tavily"
    assert result["search_query"] == "latest AI news"


def test_extract_json_markdown_fence():
    """Router should tolerate JSON wrapped in markdown."""

    content = """
    ```json
    {
        "route": "tavily",
        "search_query": "latest cloud security news",
        "location": "",
        "reason": "Current web research"
    }
    ```
    """

    result = _extract_json(content)

    assert result["route"] == "tavily"
    assert result["search_query"] == "latest cloud security news"


def test_extract_json_with_surrounding_text():
    """Router should recover JSON surrounded by harmless text."""

    content = """
    Here is the routing plan:

    {
        "route": "tavily",
        "search_query": "latest Kubernetes news",
        "location": "",
        "reason": "Current web information"
    }

    End of response.
    """

    result = _extract_json(content)

    assert result["route"] == "tavily"


def test_extract_json_empty_response():
    """Empty model output should fail safely."""

    with pytest.raises(
        ValueError,
        match="empty response",
    ):
        _extract_json("")


def test_extract_json_no_json_object():
    """Response without a JSON object should fail safely."""

    with pytest.raises(
        ValueError,
        match="valid JSON object",
    ):
        _extract_json(
            "The router could not determine a valid plan."
        )


def test_extract_json_invalid_json():
    """Malformed/incomplete JSON should fail safely."""

    with pytest.raises(
        ValueError,
        match="valid JSON object",
    ):
        _extract_json(
            '{"route": "tavily", "search_query": "broken"'
        )


# ============================================================================
# SearchPlan validation tests
# ============================================================================


def test_validate_plan_accepts_tavily():
    """Tavily must be a valid planner route."""

    data = {
        "route": "tavily",
        "search_query": "latest technology news",
        "location": "",
        "reason": "General web research.",
    }

    plan = _validate_plan(data)

    assert plan.route == "tavily"
    assert plan.search_query == "latest technology news"
    assert plan.location == ""


def test_validate_plan_rejects_unknown_route():
    """Unknown routes must never reach tool execution."""

    data = {
        "route": "google",
        "search_query": "latest news",
        "location": "",
        "reason": "Invalid route.",
    }

    with pytest.raises(
        ValueError,
        match="unsupported route",
    ):
        _validate_plan(data)


def test_validate_plan_unsupported_clears_search_and_location():
    """Unsupported plans must not carry executable retrieval parameters."""

    data = {
        "route": "unsupported",
        "search_query": "some query",
        "location": "Chennai",
        "reason": "No supported source.",
    }

    plan = _validate_plan(data)

    assert plan.route == "unsupported"
    assert plan.search_query == ""
    assert plan.location == ""


def test_validate_plan_weather_without_location_becomes_unsupported():
    """Weather without a usable location must fail closed."""

    data = {
        "route": "weather",
        "search_query": "",
        "location": "",
        "reason": "Weather question.",
    }

    plan = _validate_plan(data)

    assert plan.route == "unsupported"
    assert plan.search_query == ""
    assert plan.location == ""


def test_validate_plan_both_without_location_becomes_unsupported():
    """The combined route requires a weather location."""

    data = {
        "route": "both",
        "search_query": "umbrella recommendations",
        "location": "",
        "reason": "Combined research.",
    }

    plan = _validate_plan(data)

    assert plan.route == "unsupported"
    assert plan.search_query == ""
    assert plan.location == ""


def test_validate_plan_tavily_clears_location():
    """Tavily must never retain a weather location."""

    data = {
        "route": "tavily",
        "search_query": "latest cybersecurity news",
        "location": "Chennai",
        "reason": "General web research.",
    }

    plan = _validate_plan(data)

    assert plan.route == "tavily"
    assert plan.location == ""


def test_validate_plan_bounds_search_query():
    """Search queries should be bounded to the configured maximum."""

    long_query = "A" * 1000

    data = {
        "route": "tavily",
        "search_query": long_query,
        "location": "",
        "reason": "General web research.",
    }

    plan = _validate_plan(data)

    assert len(plan.search_query) == 500
    assert plan.search_query == "A" * 500


def test_validate_plan_bounds_location():
    """Weather locations should be bounded."""

    long_location = "C" * 500

    data = {
        "route": "weather",
        "search_query": "",
        "location": long_location,
        "reason": "Weather research.",
    }

    plan = _validate_plan(data)

    assert plan.route == "weather"
    assert len(plan.location) == 200


def test_validate_plan_bounds_reason():
    """Planner explanations should be bounded."""

    long_reason = "R" * 2000

    data = {
        "route": "tavily",
        "search_query": "latest news",
        "location": "",
        "reason": long_reason,
    }

    plan = _validate_plan(data)

    assert len(plan.reason) == 1000


# ============================================================================
# Type/normalization tests
# ============================================================================


def test_validate_plan_normalizes_route_case():
    """Route values should be normalized to lowercase."""

    data = {
        "route": "TAVILY",
        "search_query": "latest AI news",
        "location": "",
        "reason": "Current research.",
    }

    plan = _validate_plan(data)

    assert plan.route == "tavily"


def test_validate_plan_normalizes_whitespace():
    """String fields should be stripped before validation."""

    data = {
        "route": "  tavily  ",
        "search_query": "  latest AI news  ",
        "location": "   ",
        "reason": "  Current research.  ",
    }

    plan = _validate_plan(data)

    assert plan.route == "tavily"
    assert plan.search_query == "latest AI news"
    assert plan.location == ""
    assert plan.reason == "Current research."


def test_validate_plan_missing_optional_fields():
    """Missing planner fields should normalize safely."""

    data = {
        "route": "tavily",
    }

    plan = _validate_plan(data)

    assert plan.route == "tavily"
    assert plan.search_query == ""
    assert plan.location == ""
    assert plan.reason == ""


def test_validate_plan_non_dict_rejected():
    """Planner validation should reject non-dictionary data."""

    with pytest.raises(
        ValueError,
        match="JSON object",
    ):
        _validate_plan(["tavily", "latest news"])


# ============================================================================
# End-to-end mocked router behavior
# ============================================================================


def test_general_current_question_produces_tavily_plan(monkeypatch):
    """
    A realistic current/general question should produce a valid Tavily
    SearchPlan when the LLM returns the Tavily route.
    """

    response = json.dumps(
        {
            "route": "tavily",
            "search_query": "latest React security vulnerabilities",
            "location": "",
            "reason": "The question requires current web information.",
        }
    )

    monkeypatch.setattr(
        "router.llm",
        _fake_llm(response),
    )

    plan = create_search_plan(
        "What are the latest React security vulnerabilities?"
    )

    assert plan == SearchPlan(
        route="tavily",
        search_query="latest React security vulnerabilities",
        location="",
        reason="The question requires current web information.",
    )

"""
Offline tests for the evidence evaluator.

IMPORTANT:
These tests do NOT call the real Groq API.
A FakeLLM is injected so pytest can run without consuming
Groq API tokens.
"""

import json

import utils.evidence as evidence_module
from utils.evidence import evaluate_evidence


QUESTION = "Why is my React useEffect running twice?"

EVIDENCE = [
    {
        "title": "useEffect runs twice in React 18",
        "url": "https://stackoverflow.com/example",
        "score": 120,
        "tags": ["reactjs", "javascript"],
        "is_answered": True,
    },
    {
        "title": "React StrictMode causes effects to run twice",
        "url": "https://stackoverflow.com/example2",
        "score": 85,
        "tags": ["reactjs", "react-hooks"],
        "is_answered": True,
    },
]


class FakeResponse:
    """Fake LangChain response object."""

    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """
    Fake LLM used by pytest.

    It returns a deterministic JSON response instead of
    making a real Groq API request.
    """

    def invoke(self, prompt):
        response = {
            "relevant": True,
            "sufficient": True,
            "quality": "high",
            "reason": (
                "The retrieved Stack Overflow evidence directly addresses "
                "why React useEffect can run twice."
            ),
        }

        return FakeResponse(json.dumps(response))


def test_evaluate_evidence_with_mocked_llm(monkeypatch):
    """
    Verify that the evidence evaluator works without a real LLM call.
    """

    fake_llm = FakeLLM()

    monkeypatch.setattr(
        evidence_module,
        "llm",
        fake_llm,
        raising=False,
    )

    decision = evaluate_evidence(
        question=QUESTION,
        route="stackexchange",
        evidence=EVIDENCE,
    )

    assert decision is not None
    assert decision.relevant is True
    assert decision.sufficient is True
    assert decision.quality == "high"
    assert decision.reason


def test_evaluate_evidence_returns_relevance_decision(monkeypatch):
    """
    Verify that the evaluator returns the expected decision fields.
    """

    fake_llm = FakeLLM()

    monkeypatch.setattr(
        evidence_module,
        "llm",
        fake_llm,
        raising=False,
    )

    decision = evaluate_evidence(
        question=QUESTION,
        route="stackexchange",
        evidence=EVIDENCE,
    )

    assert hasattr(decision, "relevant")
    assert hasattr(decision, "sufficient")
    assert hasattr(decision, "quality")
    assert hasattr(decision, "reason")


def test_evaluate_evidence_uses_stackexchange_route(monkeypatch):
    """
    Verify that the Stack Exchange route can be evaluated.
    """

    fake_llm = FakeLLM()

    monkeypatch.setattr(
        evidence_module,
        "llm",
        fake_llm,
        raising=False,
    )

    decision = evaluate_evidence(
        question=QUESTION,
        route="stackexchange",
        evidence=EVIDENCE,
    )

    assert decision.relevant is True
    assert decision.sufficient is True


def test_evaluate_evidence_handles_multiple_sources(monkeypatch):
    """
    Verify that multiple evidence items can be evaluated together.
    """

    fake_llm = FakeLLM()

    monkeypatch.setattr(
        evidence_module,
        "llm",
        fake_llm,
        raising=False,
    )

    decision = evaluate_evidence(
        question=QUESTION,
        route="stackexchange",
        evidence=EVIDENCE,
    )

    assert len(EVIDENCE) == 2
    assert decision.quality in {"high", "medium", "low"}

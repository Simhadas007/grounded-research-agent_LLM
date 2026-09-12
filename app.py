from __future__ import annotations

from typing import Any

from guardrails.evidence_safety import sanitize_evidence_collection
from router import create_search_plan
from tools.stackexchange import search_stackexchange
from tools.tavily_search import search_tavily
from tools.weather import get_weather
from utils.research_answer import generate_research_answer


ALLOWED_ROUTES = {
    "stackexchange",
    "weather",
    "tavily",
    "both",
    "unsupported",
}


def _empty_retrieval_status() -> dict[str, str]:
    return {
        "stackexchange": "not_requested",
        "weather": "not_requested",
        "tavily": "not_requested",
    }


def _blocked_result(
    route: str,
    search_query: str,
    location: str,
    retrieval_status: dict[str, str],
    blocked_evidence: int = 0,
) -> dict[str, Any]:
    return {
        "success": True,
        "stage": "completed",
        "route": route,
        "search_query": search_query,
        "location": location,
        "answer": "I don't have enough grounded evidence to answer that safely.",
        "citations": [],
        "reason": "No safe, source-backed evidence was available.",
        "relevant": False,
        "sufficient": False,
        "quality": "low",
        "evidence": [],
        "evidence_count": 0,
        "retrieval_status": retrieval_status,
        "evidence_security": "failed" if blocked_evidence else "passed",
        "blocked_evidence": blocked_evidence,
    }


def _normalize_stackexchange_result(result: Any) -> list[dict[str, Any]]:
    """
    Convert the Stack Exchange tool response into a list of evidence items.
    """

    if not isinstance(result, dict):
        return []

    if not result.get("success", False):
        return []

    data = result.get("data")

    if isinstance(data, list):
        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    results = result.get("results")

    if isinstance(results, list):
        return [
            item
            for item in results
            if isinstance(item, dict)
        ]

    return []


def _normalize_tavily_result(result: Any) -> list[dict[str, Any]]:
    """
    Convert the Tavily tool response into a list of evidence items.
    """

    if not isinstance(result, dict):
        return []

    if not result.get("success", False):
        return []

    results = result.get("results")

    if isinstance(results, list):
        return [
            item
            for item in results
            if isinstance(item, dict)
        ]

    data = result.get("data")

    if isinstance(data, list):
        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    return []


def _normalize_weather_result(result: Any) -> list[dict[str, Any]]:
    """
    Convert the Open-Meteo weather response into a single
    structured evidence item.
    """

    if not isinstance(result, dict):
        return []

    if not result.get("success", False):
        return []

    data = result.get("data")

    if not isinstance(data, dict):
        return []

    evidence = dict(data)

    evidence["source_type"] = "weather"

    return [evidence]


def run_research(question: str) -> dict[str, Any]:
    """
    Main end-to-end research pipeline.

    Flow:

        Question
            ↓
        AI Router
            ↓
        Selected live source(s)
            ↓
        Evidence security
            ↓
        Grounded answer generation
            ↓
        Citations + metadata
    """

    if not isinstance(question, str):
        return {
            "success": False,
            "stage": "validation",
            "error": "Question must be a string.",
        }

    question = question.strip()

    if not question:
        return {
            "success": False,
            "stage": "validation",
            "error": "Question cannot be empty.",
        }

    if len(question) > 2000:
        return {
            "success": False,
            "stage": "validation",
            "error": "Question is too long.",
        }

    # ---------------------------------------------------------
    # 1. AI ROUTER
    # ---------------------------------------------------------

    try:
        plan = create_search_plan(question)
    except Exception:
        return {
            "success": False,
            "stage": "routing",
            "error": "Unable to create a safe research plan.",
        }

    route = getattr(plan, "route", None)
    search_query = getattr(plan, "search_query", "") or ""
    location = getattr(plan, "location", "") or ""
    router_reason = getattr(plan, "reason", "") or ""

    if route not in ALLOWED_ROUTES:
        return {
            "success": False,
            "stage": "routing",
            "error": "Router returned an unsupported route.",
        }

    retrieval_status = _empty_retrieval_status()

    # ---------------------------------------------------------
    # 2. UNSUPPORTED
    # ---------------------------------------------------------

    if route == "unsupported":
        return {
            "success": True,
            "stage": "completed",
            "route": "unsupported",
            "search_query": search_query,
            "location": location,
            "answer": (
                "I can't answer that safely because the requested "
                "information is outside the grounded research scope."
            ),
            "citations": [],
            "reason": router_reason,
            "relevant": False,
            "sufficient": False,
            "quality": "low",
            "evidence": [],
            "evidence_count": 0,
            "retrieval_status": retrieval_status,
            "evidence_security": "passed",
            "blocked_evidence": 0,
        }

    evidence: list[dict[str, Any]] = []

    # ---------------------------------------------------------
    # 3. STACK EXCHANGE
    # ---------------------------------------------------------

    if route in {"stackexchange", "both"}:
        retrieval_status["stackexchange"] = "requested"

        try:
            stack_result = search_stackexchange(search_query)

            stack_evidence = _normalize_stackexchange_result(
                stack_result
            )

            if stack_evidence:
                evidence.extend(stack_evidence)
                retrieval_status["stackexchange"] = "success"
            else:
                retrieval_status["stackexchange"] = "no_results"

        except Exception:
            retrieval_status["stackexchange"] = "error"

    # ---------------------------------------------------------
    # 4. WEATHER / OPEN-METEO
    # ---------------------------------------------------------

    if route in {"weather", "both"}:
        retrieval_status["weather"] = "requested"

        try:
            weather_result = get_weather(location)

            weather_evidence = _normalize_weather_result(
                weather_result
            )

            if weather_evidence:
                evidence.extend(weather_evidence)
                retrieval_status["weather"] = "success"
            else:
                retrieval_status["weather"] = "no_results"

        except Exception:
            retrieval_status["weather"] = "error"

    # ---------------------------------------------------------
    # 5. TAVILY
    # ---------------------------------------------------------

    if route in {"tavily", "both"}:
        retrieval_status["tavily"] = "requested"

        try:
            tavily_result = search_tavily(search_query)

            tavily_evidence = _normalize_tavily_result(
                tavily_result
            )

            if tavily_evidence:
                evidence.extend(tavily_evidence)
                retrieval_status["tavily"] = "success"
            else:
                retrieval_status["tavily"] = "no_results"

        except Exception:
            retrieval_status["tavily"] = "error"

    # ---------------------------------------------------------
    # 6. NO RETRIEVED EVIDENCE
    # ---------------------------------------------------------

    if not evidence:
        return _blocked_result(
            route=route,
            search_query=search_query,
            location=location,
            retrieval_status=retrieval_status,
        )

    # ---------------------------------------------------------
    # 7. EVIDENCE SECURITY
    # ---------------------------------------------------------

    try:
        security_result = sanitize_evidence_collection(
            evidence
        )

        safe_evidence = security_result.get(
            "evidence",
            [],
        )

        blocked_count = security_result.get(
            "blocked",
            0,
        )

        evidence_allowed = security_result.get(
            "allowed",
            False,
        )

    except Exception:
        return _blocked_result(
            route=route,
            search_query=search_query,
            location=location,
            retrieval_status=retrieval_status,
            blocked_evidence=len(evidence),
        )

    if not isinstance(safe_evidence, list):
        safe_evidence = []

    if not evidence_allowed or not safe_evidence:
        return _blocked_result(
            route=route,
            search_query=search_query,
            location=location,
            retrieval_status=retrieval_status,
            blocked_evidence=blocked_count or len(evidence),
        )

    # ---------------------------------------------------------
    # 8. GROUNDED ANSWER
    # ---------------------------------------------------------

    try:
        answer_result = generate_research_answer(
            question=question,
            source=route,
            evidence=safe_evidence,
        )

    except Exception:
        return {
            "success": False,
            "stage": "answer_generation",
            "route": route,
            "search_query": search_query,
            "location": location,
            "answer": (
                "The sources were retrieved, but I could not safely "
                "generate a grounded answer."
            ),
            "citations": [],
            "reason": "Answer generation failed safely.",
            "relevant": False,
            "sufficient": False,
            "quality": "low",
            "evidence": safe_evidence,
            "evidence_count": len(safe_evidence),
            "retrieval_status": retrieval_status,
            "evidence_security": "passed",
            "blocked_evidence": blocked_count,
        }

    # ---------------------------------------------------------
    # 9. NORMALIZE ANSWER RESULT
    # ---------------------------------------------------------

    if hasattr(answer_result, "model_dump"):
        answer_data = answer_result.model_dump()

    elif isinstance(answer_result, dict):
        answer_data = answer_result

    else:
        answer_data = {}

    answer = answer_data.get(
        "answer",
        "",
    )

    citations = answer_data.get(
        "citations",
        [],
    )

    relevant = bool(
        answer_data.get(
            "relevant",
            False,
        )
    )

    sufficient = bool(
        answer_data.get(
            "sufficient",
            False,
        )
    )

    quality = answer_data.get(
        "quality",
        "low",
    )

    reason = answer_data.get(
        "reason",
        "",
    )

    if not isinstance(answer, str):
        answer = ""

    if not isinstance(citations, list):
        citations = []

    citations = [
        citation
        for citation in citations
        if isinstance(citation, str)
    ]

    if quality not in {
        "high",
        "medium",
        "low",
    }:
        quality = "low"

    # ---------------------------------------------------------
    # 10. FINAL RESPONSE
    # ---------------------------------------------------------

    return {
        "success": True,
        "stage": "completed",
        "route": route,
        "search_query": search_query,
        "location": location,
        "answer": answer,
        "citations": citations,
        "reason": reason or router_reason,
        "relevant": relevant,
        "sufficient": sufficient,
        "quality": quality,
        "evidence": safe_evidence,
        "evidence_count": len(safe_evidence),
        "retrieval_status": retrieval_status,
        "evidence_security": "passed",
        "blocked_evidence": blocked_count,
    }


def main() -> None:
    print("=" * 70)
    print("GROUNDED RESEARCH AGENT")
    print("=" * 70)

    question = input(
        "\nEnter your research question: "
    ).strip()

    if not question:
        print("\nQuestion cannot be empty.")
        return

    result = run_research(question)

    print("\nSuccess:", result.get("success"))
    print("Stage:", result.get("stage"))
    print("Route:", result.get("route"))

    print("\nRetrieval status:")

    retrieval_status = result.get(
        "retrieval_status",
        {},
    )

    for source, status in retrieval_status.items():
        print(f"  {source}: {status}")

    print(
        "\nEvidence count:",
        result.get(
            "evidence_count",
            0,
        ),
    )

    print(
        "Evidence security:",
        result.get(
            "evidence_security",
            "unknown",
        ),
    )

    print(
        "Blocked evidence:",
        result.get(
            "blocked_evidence",
            0,
        ),
    )

    print(
        "Evidence quality:",
        result.get(
            "quality",
            "low",
        ),
    )

    print("\nAnswer:")
    print(
        result.get(
            "answer",
            "",
        )
    )

    citations = result.get(
        "citations",
        [],
    )

    if citations:
        print("\nSources:")

        for citation in citations:
            print(f"- {citation}")


if __name__ == "__main__":
    main()
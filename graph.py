from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from router import route_question
from planner import create_search_plan

from tools.stackexchange import search_stackexchange
from tools.weather import get_weather

from utils.evidence import evaluate_evidence
from utils.answer import generate_answer
from utils.output_safety import evaluate_output
from utils.citations import validate_citations


# ============================================================
# Agent State
# ============================================================

class AgentState(TypedDict, total=False):

    # --------------------------------------------------------
    # User input
    # --------------------------------------------------------

    question: str

    # --------------------------------------------------------
    # AI routing
    # --------------------------------------------------------

    route: str
    route_reason: str

    # --------------------------------------------------------
    # AI planning
    # --------------------------------------------------------

    search_query: str
    weather_location: str

    # --------------------------------------------------------
    # Retrieved evidence
    # --------------------------------------------------------

    evidence: list[dict]

    # --------------------------------------------------------
    # AI evidence evaluation
    # --------------------------------------------------------

    evidence_relevant: bool
    evidence_sufficient: bool
    evidence_quality: str
    evidence_reason: str

    # --------------------------------------------------------
    # Generated answer
    # --------------------------------------------------------

    answer: str

    # --------------------------------------------------------
    # Output safety
    # --------------------------------------------------------

    output_safe: bool
    output_grounded: bool
    output_safety_reason: str

    # --------------------------------------------------------
    # Citations
    # --------------------------------------------------------

    citations: list[str]

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    error: str


# ============================================================
# Router Node
# ============================================================

def router_node(state: AgentState) -> AgentState:
    """
    Use the AI router to decide which research source is required.
    """

    question = state.get("question", "").strip()

    if not question:
        return {
            **state,
            "error": "No question provided.",
        }

    try:
        decision = route_question(question)

        return {
            **state,
            "route": decision.route,
            "route_reason": decision.reason,
            "error": "",
        }

    except Exception as exc:
        return {
            **state,
            "route": "",
            "route_reason": "",
            "error": f"Research routing failed: {exc}",
        }


# ============================================================
# Planner Node
# ============================================================

def planner_node(state: AgentState) -> AgentState:
    """
    Use the AI planner to create the search query/location
    required by the selected research route.
    """

    question = state.get("question", "").strip()
    route = state.get("route", "").strip()

    if not question:
        return {
            **state,
            "error": "No question provided to planner.",
        }

    if not route:
        return {
            **state,
            "error": "No research route selected.",
        }

    # --------------------------------------------------------
    # Unsupported questions do not require planning.
    # --------------------------------------------------------

    if route == "unsupported":
        return {
            **state,
            "search_query": "",
            "weather_location": "",
            "error": "",
        }

    try:
        plan = create_search_plan(
            question=question,
            route=route,
        )

        return {
            **state,
            "search_query": plan.search_query,
            "weather_location": plan.location,
            "error": "",
        }

    except Exception as exc:
        return {
            **state,
            "search_query": "",
            "weather_location": "",
            "error": f"Research planning failed: {exc}",
        }


# ============================================================
# Retrieval Node
# ============================================================

def retrieval_node(state: AgentState) -> AgentState:
    """
    Retrieve live external evidence based on the AI-selected route.

    Security principle:
    Only the source selected by the router is called.
    """

    route = state.get("route", "").strip()

    search_query = state.get(
        "search_query",
        "",
    ).strip()

    weather_location = state.get(
        "weather_location",
        "",
    ).strip()

    # --------------------------------------------------------
    # Unsupported question
    # --------------------------------------------------------

    if route == "unsupported":
        return {
            **state,
            "evidence": [],
            "error": "",
        }

    if route not in (
        "stackexchange",
        "weather",
        "both",
    ):
        return {
            **state,
            "evidence": [],
            "error": "Invalid research route.",
        }

    evidence: list[dict] = []

    # ========================================================
    # Stack Exchange
    # ========================================================

    if route in ("stackexchange", "both"):

        if not search_query:
            return {
                **state,
                "evidence": evidence,
                "error": (
                    "No Stack Exchange search query "
                    "was generated."
                ),
            }

        try:
            result = search_stackexchange(
                search_query
            )

        except Exception as exc:
            return {
                **state,
                "evidence": evidence,
                "error": (
                    f"Stack Exchange retrieval failed: {exc}"
                ),
            }

        if not result.get("success"):

            return {
                **state,
                "evidence": evidence,
                "error": result.get(
                    "error",
                    "Stack Exchange retrieval failed.",
                ),
            }

        results = result.get("results", [])

        if not isinstance(results, list):
            return {
                **state,
                "evidence": evidence,
                "error": (
                    "Stack Exchange returned invalid evidence."
                ),
            }

        evidence.extend(results)

    # ========================================================
    # Weather
    # ========================================================

    if route in ("weather", "both"):

        if not weather_location:
            return {
                **state,
                "evidence": evidence,
                "error": (
                    "No weather location was extracted."
                ),
            }

        try:
            result = get_weather(
                weather_location
            )

        except Exception as exc:
            return {
                **state,
                "evidence": evidence,
                "error": (
                    f"Weather retrieval failed: {exc}"
                ),
            }

        if not result.get("success"):

            return {
                **state,
                "evidence": evidence,
                "error": result.get(
                    "error",
                    "Weather retrieval failed.",
                ),
            }

        weather_data = result.get("data")

        if not isinstance(weather_data, dict):
            return {
                **state,
                "evidence": evidence,
                "error": (
                    "Weather service returned invalid evidence."
                ),
            }

        evidence.append(weather_data)

    # ========================================================
    # Retrieval completed
    # ========================================================

    return {
        **state,
        "evidence": evidence,
        "error": "",
    }


# ============================================================
# Evidence Evaluation Node
# ============================================================

def evidence_node(state: AgentState) -> AgentState:
    """
    Use the AI evaluator to determine whether retrieved evidence
    is relevant and sufficient.

    Important:
    An evaluator failure is NOT treated as insufficient evidence.
    """

    question = state.get(
        "question",
        "",
    ).strip()

    route = state.get(
        "route",
        "",
    ).strip()

    evidence = state.get(
        "evidence",
        [],
    )

    if not question:
        return {
            **state,
            "evidence_relevant": False,
            "evidence_sufficient": False,
            "evidence_quality": "low",
            "evidence_reason": (
                "No question provided to evidence evaluator."
            ),
            "error": (
                "No question provided "
                "to evidence evaluator."
            ),
        }

    try:

        decision = evaluate_evidence(
            question=question,
            route=route,
            evidence=evidence,
        )

        return {
            **state,
            "evidence_relevant": decision.relevant,
            "evidence_sufficient": decision.sufficient,
            "evidence_quality": decision.quality,
            "evidence_reason": decision.reason,
            "error": "",
        }

    except Exception as exc:

        return {
            **state,
            "evidence_relevant": False,
            "evidence_sufficient": False,
            "evidence_quality": "low",
            "evidence_reason": (
                "Evidence evaluation failed."
            ),
            "error": (
                f"Evidence evaluation failed: {exc}"
            ),
        }


# ============================================================
# Answer Node
# ============================================================

def answer_node(state: AgentState) -> AgentState:
    """
    Generate an answer only when the previous stages completed
    successfully.

    Important:
    A failed evaluator/retrieval/planner must never be disguised
    as insufficient evidence.
    """

    question = state.get(
        "question",
        "",
    ).strip()

    route = state.get(
        "route",
        "",
    )

    evidence = state.get(
        "evidence",
        [],
    )

    evidence_sufficient = state.get(
        "evidence_sufficient",
        False,
    )

    existing_error = state.get(
        "error",
        "",
    )

    # --------------------------------------------------------
    # Stop on upstream errors
    # --------------------------------------------------------

    if existing_error:

        return {
            **state,
            "answer": (
                "I couldn't safely complete the research "
                "because an internal research step failed."
            ),
            "citations": [],
        }

    # --------------------------------------------------------
    # Genuine insufficient evidence
    # --------------------------------------------------------

    if not evidence_sufficient:

        return {
            **state,
            "answer": (
                "I don't have enough grounded evidence from "
                "the available sources to answer this reliably."
            ),
            "citations": [],
            "error": "",
        }

    # --------------------------------------------------------
    # Generate grounded answer
    # --------------------------------------------------------

    try:

        generated = generate_answer(
            question=question,
            route=route,
            evidence=evidence,
            evidence_sufficient=evidence_sufficient,
        )

        return {
            **state,
            "answer": generated.answer,
            "citations": generated.citations,
            "error": "",
        }

    except Exception as exc:

        return {
            **state,
            "answer": (
                "I couldn't safely complete the answer "
                "generation step."
            ),
            "citations": [],
            "error": (
                f"Answer generation failed: {exc}"
            ),
        }


# ============================================================
# Output Safety Node
# ============================================================

def output_safety_node(state: AgentState) -> AgentState:
    """
    Evaluate the generated answer before returning it.
    """

    question = state.get(
        "question",
        "",
    ).strip()

    route = state.get(
        "route",
        "",
    )

    evidence = state.get(
        "evidence",
        [],
    )

    answer = state.get(
        "answer",
        "",
    )

    citations = state.get(
        "citations",
        [],
    )

    existing_error = state.get(
        "error",
        "",
    )

    # --------------------------------------------------------
    # Do not run another LLM safety evaluation when an
    # upstream stage has already failed.
    # --------------------------------------------------------

    if existing_error:

        return {
            **state,
            "output_safe": False,
            "output_grounded": False,
            "output_safety_reason": (
                "Pipeline stopped because an upstream "
                "research step failed."
            ),
            "answer": (
                "I couldn't safely complete the research "
                "request."
            ),
            "citations": [],
        }

    try:

        decision = evaluate_output(
            question=question,
            route=route,
            evidence=evidence,
            answer=answer,
            citations=citations,
        )

        if not decision.safe or not decision.grounded:

            return {
                **state,
                "output_safe": False,
                "output_grounded": False,
                "output_safety_reason": decision.reason,
                "answer": (
                    "I couldn't safely generate a "
                    "grounded answer."
                ),
                "citations": [],
                "error": (
                    "Output safety check failed."
                ),
            }

        return {
            **state,
            "output_safe": True,
            "output_grounded": True,
            "output_safety_reason": decision.reason,
            "error": "",
        }

    except Exception as exc:

        return {
            **state,
            "output_safe": False,
            "output_grounded": False,
            "output_safety_reason": str(exc),
            "answer": (
                "I couldn't safely complete the "
                "output safety check."
            ),
            "citations": [],
            "error": (
                f"Output safety evaluation failed: {exc}"
            ),
        }


# ============================================================
# Citation Node
# ============================================================

def citation_node(state: AgentState) -> AgentState:
    """
    Validate explicit citations returned by the answer generator.
    """

    answer = state.get(
        "answer",
        "",
    )

    evidence = state.get(
        "evidence",
        [],
    )

    citations = state.get(
        "citations",
        [],
    )

    output_safe = state.get(
        "output_safe",
        False,
    )

    evidence_sufficient = state.get(
        "evidence_sufficient",
        False,
    )

    existing_error = state.get(
        "error",
        "",
    )

    # --------------------------------------------------------
    # If output safety failed, do not validate citations.
    # --------------------------------------------------------

    if not output_safe:

        return {
            **state,
            "citations": [],
        }

    # --------------------------------------------------------
    # Preserve upstream errors.
    # --------------------------------------------------------

    if existing_error:

        return {
            **state,
            "citations": [],
        }

    try:

        result = validate_citations(
            answer=answer,
            evidence=evidence,
            citations=citations,
            evidence_sufficient=evidence_sufficient,
        )

        if not result["valid"]:

            return {
                **state,
                "citations": [],
                "error": result["reason"],
            }

        return {
            **state,
            "citations": result["citations"],
            "error": "",
        }

    except Exception as exc:

        return {
            **state,
            "citations": [],
            "error": (
                f"Citation validation failed: {exc}"
            ),
        }


# ============================================================
# Build LangGraph
# ============================================================

def build_graph():

    workflow = StateGraph(
        AgentState
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    workflow.add_node(
        "router",
        router_node,
    )

    workflow.add_node(
        "planner",
        planner_node,
    )

    workflow.add_node(
        "retrieval",
        retrieval_node,
    )

    workflow.add_node(
        "evidence",
        evidence_node,
    )

    workflow.add_node(
        "answer",
        answer_node,
    )

    workflow.add_node(
        "output_safety",
        output_safety_node,
    )

    workflow.add_node(
        "citation",
        citation_node,
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    workflow.add_edge(
        START,
        "router",
    )

    workflow.add_edge(
        "router",
        "planner",
    )

    workflow.add_edge(
        "planner",
        "retrieval",
    )

    workflow.add_edge(
        "retrieval",
        "evidence",
    )

    workflow.add_edge(
        "evidence",
        "answer",
    )

    workflow.add_edge(
        "answer",
        "output_safety",
    )

    workflow.add_edge(
        "output_safety",
        "citation",
    )

    workflow.add_edge(
        "citation",
        END,
    )

    return workflow.compile()


# ============================================================
# Compiled Application
# ============================================================

app = build_graph()

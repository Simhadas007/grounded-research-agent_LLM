"""
Scope enforcement for the Grounded Research Agent.

Security objective
------------------
The agent must only operate within its explicitly supported research scope.

Supported research capabilities:
    1. Stack Exchange technical/programming research
    2. Live weather research through the weather API
    3. Combined technical + weather research when both are genuinely needed

Anything else must be rejected before retrieval and answer generation.

This module validates the AI router's decision at a security boundary.

IMPORTANT:
The router is responsible for understanding the user's question.
This module is responsible for enforcing what the application is ALLOWED
to do.

That separation prevents an LLM routing mistake from becoming an
unauthorized tool invocation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Authorized application scope
# ---------------------------------------------------------------------------

ALLOWED_ROUTES = {
    "stackexchange",
    "weather",
    "both",
    "unsupported",
}


# ---------------------------------------------------------------------------
# Security result
# ---------------------------------------------------------------------------

class ScopeDecision(BaseModel):
    """
    Result of the scope security check.
    """

    allowed: bool
    route: Literal[
        "stackexchange",
        "weather",
        "both",
        "unsupported",
    ]

    reason: str = Field(min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Route validation
# ---------------------------------------------------------------------------

def validate_route(route: str) -> ScopeDecision:
    """
    Validate the route selected by the AI router.

    Security principle:
        Unknown routes are NEVER allowed.

    This protects against:
        - malformed LLM output
        - prompt manipulation
        - accidental future tool names
        - unauthorized tool selection
    """

    if not isinstance(route, str):
        return ScopeDecision(
            allowed=False,
            route="unsupported",
            reason="Router returned an invalid route type.",
        )

    normalized_route = route.strip().lower()

    if normalized_route not in ALLOWED_ROUTES:
        return ScopeDecision(
            allowed=False,
            route="unsupported",
            reason=(
                "Router selected a route outside the application's "
                "authorized research scope."
            ),
        )

    if normalized_route == "unsupported":
        return ScopeDecision(
            allowed=False,
            route="unsupported",
            reason=(
                "The requested topic is outside the supported research "
                "scope."
            ),
        )

    return ScopeDecision(
        allowed=True,
        route=normalized_route,  # type: ignore[arg-type]
        reason="Router decision is within the authorized application scope.",
    )


# ---------------------------------------------------------------------------
# Search-plan validation
# ---------------------------------------------------------------------------

def validate_search_plan(plan) -> ScopeDecision:
    """
    Validate the complete AI-generated search plan.

    The existing router produces:
        route
        search_query
        location
        reason

    This function verifies the route before any external retrieval occurs.
    """

    if plan is None:
        return ScopeDecision(
            allowed=False,
            route="unsupported",
            reason="No search plan was provided.",
        )

    route = getattr(plan, "route", None)

    decision = validate_route(route)

    if not decision.allowed:
        return decision

    # ---------------------------------------------------------------
    # Route-specific security checks
    # ---------------------------------------------------------------

    if decision.route == "stackexchange":

        search_query = getattr(plan, "search_query", "")

        if not isinstance(search_query, str) or not search_query.strip():
            return ScopeDecision(
                allowed=False,
                route="unsupported",
                reason=(
                    "Stack Exchange route requires a valid research "
                    "query."
                ),
            )

    elif decision.route == "weather":

        location = getattr(plan, "location", "")

        if not isinstance(location, str) or not location.strip():
            return ScopeDecision(
                allowed=False,
                route="unsupported",
                reason=(
                    "Weather route requires a valid location."
                ),
            )

    elif decision.route == "both":

        search_query = getattr(plan, "search_query", "")
        location = getattr(plan, "location", "")

        if not isinstance(search_query, str) or not search_query.strip():
            return ScopeDecision(
                allowed=False,
                route="unsupported",
                reason=(
                    "Combined route requires a valid technical "
                    "research query."
                ),
            )

        if not isinstance(location, str) or not location.strip():
            return ScopeDecision(
                allowed=False,
                route="unsupported",
                reason=(
                    "Combined route requires a valid weather location."
                ),
            )

    return decision


# ---------------------------------------------------------------------------
# Retrieval authorization
# ---------------------------------------------------------------------------

def authorize_retrieval(plan) -> dict:
    """
    Final authorization boundary before external tools are called.

    Returns:

        {
            "allowed": True/False,
            "route": "...",
            "reason": "..."
        }

    SECURITY RULE:
    No external retrieval should happen unless this function returns
    allowed=True.
    """

    decision = validate_search_plan(plan)

    return {
        "allowed": decision.allowed,
        "route": decision.route,
        "reason": decision.reason,
    }


# ---------------------------------------------------------------------------
# User-facing scope decision
# ---------------------------------------------------------------------------

def scope_rejection_message() -> str:
    """
    Safe response for unsupported questions.

    Do not expose internal routing/security implementation details.
    """

    return (
        "I can only answer questions grounded in the supported research "
        "sources: technical discussions from Stack Exchange and live "
        "weather data. I don't have sufficient supported sources to "
        "answer that question."
    )

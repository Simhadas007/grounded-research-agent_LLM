"""
Tool permission and agent execution security.

This module controls which tools the agent is allowed to execute,
how many tool calls may happen during one request, and whether a
specific tool is authorized for the selected research route.

Security principles:
- Fail closed.
- Explicit tool allowlisting.
- Explicit route -> tool permissions.
- Maximum tool calls per request.
- No duplicate/unbounded tool execution.
- No unauthorized tool switching.
- Per-request execution state is isolated.
- Tool names are treated as untrusted input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Literal


# ---------------------------------------------------------------------------
# Allowed routes
# ---------------------------------------------------------------------------

ALLOWED_ROUTES = {
    "stackexchange",
    "weather",
    "both",
    "unsupported",
}


# ---------------------------------------------------------------------------
# Allowed tools
# ---------------------------------------------------------------------------

ALLOWED_TOOLS = {
    "stackexchange",
    "weather",
}


# ---------------------------------------------------------------------------
# Route -> tool authorization
# ---------------------------------------------------------------------------

ROUTE_TOOL_PERMISSIONS: dict[str, set[str]] = {
    "stackexchange": {
        "stackexchange",
    },
    "weather": {
        "weather",
    },
    "both": {
        "stackexchange",
        "weather",
    },
    "unsupported": set(),
}


# ---------------------------------------------------------------------------
# Security limits
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOOL_CALLS = 4

# Prevent repeatedly calling exactly the same tool during one request.
DEFAULT_MAX_CALLS_PER_TOOL = 2


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolAuthorizationResult:
    """
    Result returned by the tool authorization boundary.
    """

    allowed: bool
    tool: str
    route: str
    reason: str
    remaining_calls: int


# ---------------------------------------------------------------------------
# Per-request execution state
# ---------------------------------------------------------------------------

@dataclass
class ToolExecutionState:
    """
    Isolated state for one agent request.

    This state must NOT be shared between unrelated users/requests.
    """

    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_calls_per_tool: int = DEFAULT_MAX_CALLS_PER_TOOL

    total_calls: int = 0

    calls_by_tool: dict[str, int] = field(default_factory=dict)

    call_history: list[str] = field(default_factory=list)

    _lock: Lock = field(default_factory=Lock, repr=False)

    def authorize(
        self,
        route: str,
        tool: str,
    ) -> ToolAuthorizationResult:
        """
        Authorize one tool execution.

        Every tool execution must pass through this method before
        the actual external API call occurs.
        """

        route = _normalize_name(route)
        tool = _normalize_name(tool)

        # ---------------------------------------------------------------
        # Validate route
        # ---------------------------------------------------------------

        if route not in ALLOWED_ROUTES:
            return ToolAuthorizationResult(
                allowed=False,
                tool=tool,
                route=route,
                reason="Unauthorized route.",
                remaining_calls=self._remaining_calls(),
            )

        # ---------------------------------------------------------------
        # Validate tool
        # ---------------------------------------------------------------

        if tool not in ALLOWED_TOOLS:
            return ToolAuthorizationResult(
                allowed=False,
                tool=tool,
                route=route,
                reason="Unauthorized tool.",
                remaining_calls=self._remaining_calls(),
            )

        # ---------------------------------------------------------------
        # Route -> tool permission check
        # ---------------------------------------------------------------

        allowed_tools = ROUTE_TOOL_PERMISSIONS.get(route, set())

        if tool not in allowed_tools:
            return ToolAuthorizationResult(
                allowed=False,
                tool=tool,
                route=route,
                reason=(
                    f"Tool '{tool}' is not authorized for "
                    f"route '{route}'."
                ),
                remaining_calls=self._remaining_calls(),
            )

        # ---------------------------------------------------------------
        # Enforce execution limits atomically
        # ---------------------------------------------------------------

        with self._lock:

            if self.total_calls >= self.max_tool_calls:
                return ToolAuthorizationResult(
                    allowed=False,
                    tool=tool,
                    route=route,
                    reason="Maximum tool calls per request exceeded.",
                    remaining_calls=0,
                )

            current_tool_calls = self.calls_by_tool.get(tool, 0)

            if current_tool_calls >= self.max_calls_per_tool:
                return ToolAuthorizationResult(
                    allowed=False,
                    tool=tool,
                    route=route,
                    reason=(
                        f"Maximum calls for tool '{tool}' "
                        "per request exceeded."
                    ),
                    remaining_calls=(
                        self.max_tool_calls - self.total_calls
                    ),
                )

            # -----------------------------------------------------------
            # Reserve the tool call BEFORE execution.
            # -----------------------------------------------------------

            self.total_calls += 1
            self.calls_by_tool[tool] = current_tool_calls + 1
            self.call_history.append(tool)

            return ToolAuthorizationResult(
                allowed=True,
                tool=tool,
                route=route,
                reason="Tool execution authorized.",
                remaining_calls=(
                    self.max_tool_calls - self.total_calls
                ),
            )

    def _remaining_calls(self) -> int:
        with self._lock:
            return max(
                0,
                self.max_tool_calls - self.total_calls,
            )

    def reset(self) -> None:
        """
        Reset state.

        Normally create a new ToolExecutionState for each request.
        This method exists for controlled testing/reuse.
        """

        with self._lock:
            self.total_calls = 0
            self.calls_by_tool.clear()
            self.call_history.clear()


# ---------------------------------------------------------------------------
# Tool security controller
# ---------------------------------------------------------------------------

class ToolSecurityController:
    """
    High-level controller for agent tool authorization.

    Create one controller/state per user request.
    """

    def __init__(
        self,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_calls_per_tool: int = DEFAULT_MAX_CALLS_PER_TOOL,
    ) -> None:

        if max_tool_calls < 1:
            raise ValueError(
                "max_tool_calls must be at least 1."
            )

        if max_calls_per_tool < 1:
            raise ValueError(
                "max_calls_per_tool must be at least 1."
            )

        if max_calls_per_tool > max_tool_calls:
            raise ValueError(
                "max_calls_per_tool cannot exceed max_tool_calls."
            )

        self.state = ToolExecutionState(
            max_tool_calls=max_tool_calls,
            max_calls_per_tool=max_calls_per_tool,
        )

    def authorize_tool(
        self,
        route: str,
        tool: str,
    ) -> ToolAuthorizationResult:
        """
        Authorize a tool before calling it.
        """

        return self.state.authorize(
            route=route,
            tool=tool,
        )

    @property
    def total_calls(self) -> int:
        return self.state.total_calls

    @property
    def call_history(self) -> list[str]:
        return list(self.state.call_history)


# ---------------------------------------------------------------------------
# Convenience authorization function
# ---------------------------------------------------------------------------

def authorize_tool(
    route: str,
    tool: str,
) -> ToolAuthorizationResult:
    """
    Stateless single-call authorization helper.

    Useful when the application only needs to check whether a
    tool is permitted for a route.

    It does NOT maintain execution-count state.
    """

    route = _normalize_name(route)
    tool = _normalize_name(tool)

    if route not in ALLOWED_ROUTES:
        return ToolAuthorizationResult(
            allowed=False,
            tool=tool,
            route=route,
            reason="Unauthorized route.",
            remaining_calls=0,
        )

    if tool not in ALLOWED_TOOLS:
        return ToolAuthorizationResult(
            allowed=False,
            tool=tool,
            route=route,
            reason="Unauthorized tool.",
            remaining_calls=0,
        )

    if tool not in ROUTE_TOOL_PERMISSIONS.get(route, set()):
        return ToolAuthorizationResult(
            allowed=False,
            tool=tool,
            route=route,
            reason=(
                f"Tool '{tool}' is not authorized for "
                f"route '{route}'."
            ),
            remaining_calls=0,
        )

    return ToolAuthorizationResult(
        allowed=True,
        tool=tool,
        route=route,
        reason="Tool execution authorized.",
        remaining_calls=0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_name(value: str) -> str:
    """
    Normalize a route/tool identifier.

    Reject non-string values instead of silently converting them.
    """

    if not isinstance(value, str):
        return ""

    return value.strip().lower()


__all__ = [
    "ALLOWED_ROUTES",
    "ALLOWED_TOOLS",
    "ROUTE_TOOL_PERMISSIONS",
    "ToolAuthorizationResult",
    "ToolExecutionState",
    "ToolSecurityController",
    "authorize_tool",
]


import pytest

from security.tool_security import (
    ToolExecutionState,
    ToolSecurityController,
    authorize_tool,
)


# ---------------------------------------------------------------------------
# Test 1: Stack Exchange route allows Stack Exchange
# ---------------------------------------------------------------------------

def test_stackexchange_route_allows_stackexchange():
    result = authorize_tool(
        "stackexchange",
        "stackexchange",
    )

    assert result.allowed is True


# ---------------------------------------------------------------------------
# Test 2: Weather route allows Weather
# ---------------------------------------------------------------------------

def test_weather_route_allows_weather():
    result = authorize_tool(
        "weather",
        "weather",
    )

    assert result.allowed is True


# ---------------------------------------------------------------------------
# Test 3: Stack Exchange cannot call Weather
# ---------------------------------------------------------------------------

def test_stackexchange_route_blocks_weather():
    result = authorize_tool(
        "stackexchange",
        "weather",
    )

    assert result.allowed is False


# ---------------------------------------------------------------------------
# Test 4: Weather cannot call Stack Exchange
# ---------------------------------------------------------------------------

def test_weather_route_blocks_stackexchange():
    result = authorize_tool(
        "weather",
        "stackexchange",
    )

    assert result.allowed is False


# ---------------------------------------------------------------------------
# Test 5: Both route allows Stack Exchange
# ---------------------------------------------------------------------------

def test_both_route_allows_stackexchange():
    result = authorize_tool(
        "both",
        "stackexchange",
    )

    assert result.allowed is True


# ---------------------------------------------------------------------------
# Test 6: Both route allows Weather
# ---------------------------------------------------------------------------

def test_both_route_allows_weather():
    result = authorize_tool(
        "both",
        "weather",
    )

    assert result.allowed is True


# ---------------------------------------------------------------------------
# Test 7: Unsupported route blocks everything
# ---------------------------------------------------------------------------

def test_unsupported_route_blocks_tools():
    stack_result = authorize_tool(
        "unsupported",
        "stackexchange",
    )

    weather_result = authorize_tool(
        "unsupported",
        "weather",
    )

    assert stack_result.allowed is False
    assert weather_result.allowed is False


# ---------------------------------------------------------------------------
# Test 8: Unknown tool rejected
# ---------------------------------------------------------------------------

def test_unknown_tool_rejected():
    result = authorize_tool(
        "both",
        "database",
    )

    assert result.allowed is False


# ---------------------------------------------------------------------------
# Test 9: Unknown route rejected
# ---------------------------------------------------------------------------

def test_unknown_route_rejected():
    result = authorize_tool(
        "admin",
        "weather",
    )

    assert result.allowed is False


# ---------------------------------------------------------------------------
# Test 10: Tool name normalization
# ---------------------------------------------------------------------------

def test_tool_name_normalization():
    result = authorize_tool(
        " WEATHER ",
        " WEATHER ",
    )

    assert result.allowed is True
    assert result.route == "weather"
    assert result.tool == "weather"


# ---------------------------------------------------------------------------
# Test 11: Non-string route rejected
# ---------------------------------------------------------------------------

def test_non_string_route_rejected():
    result = authorize_tool(
        None,
        "weather",
    )

    assert result.allowed is False


# ---------------------------------------------------------------------------
# Test 12: Non-string tool rejected
# ---------------------------------------------------------------------------

def test_non_string_tool_rejected():
    result = authorize_tool(
        "weather",
        None,
    )

    assert result.allowed is False


# ---------------------------------------------------------------------------
# Test 13: First tool call allowed
# ---------------------------------------------------------------------------

def test_first_tool_call_allowed():
    controller = ToolSecurityController(
        max_tool_calls=4,
        max_calls_per_tool=2,
    )

    result = controller.authorize_tool(
        "weather",
        "weather",
    )

    assert result.allowed is True
    assert controller.total_calls == 1


# ---------------------------------------------------------------------------
# Test 14: Maximum total tool calls enforced
# ---------------------------------------------------------------------------

def test_maximum_total_tool_calls():
    controller = ToolSecurityController(
        max_tool_calls=2,
        max_calls_per_tool=2,
    )

    first = controller.authorize_tool(
        "both",
        "weather",
    )

    second = controller.authorize_tool(
        "both",
        "stackexchange",
    )

    third = controller.authorize_tool(
        "both",
        "weather",
    )

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False

    assert controller.total_calls == 2


# ---------------------------------------------------------------------------
# Test 15: Maximum calls per individual tool enforced
# ---------------------------------------------------------------------------

def test_maximum_calls_per_tool():
    controller = ToolSecurityController(
        max_tool_calls=4,
        max_calls_per_tool=2,
    )

    first = controller.authorize_tool(
        "weather",
        "weather",
    )

    second = controller.authorize_tool(
        "weather",
        "weather",
    )

    third = controller.authorize_tool(
        "weather",
        "weather",
    )

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False

    assert controller.total_calls == 2


# ---------------------------------------------------------------------------
# Test 16: Different tools can execute under both route
# ---------------------------------------------------------------------------

def test_both_route_allows_two_different_tools():
    controller = ToolSecurityController(
        max_tool_calls=4,
        max_calls_per_tool=2,
    )

    weather = controller.authorize_tool(
        "both",
        "weather",
    )

    stackexchange = controller.authorize_tool(
        "both",
        "stackexchange",
    )

    assert weather.allowed is True
    assert stackexchange.allowed is True

    assert controller.total_calls == 2


# ---------------------------------------------------------------------------
# Test 17: Unauthorized call does not consume a tool-call slot
# ---------------------------------------------------------------------------

def test_denied_call_does_not_consume_slot():
    controller = ToolSecurityController(
        max_tool_calls=2,
        max_calls_per_tool=2,
    )

    denied = controller.authorize_tool(
        "weather",
        "stackexchange",
    )

    allowed = controller.authorize_tool(
        "weather",
        "weather",
    )

    assert denied.allowed is False
    assert allowed.allowed is True

    assert controller.total_calls == 1


# ---------------------------------------------------------------------------
# Test 18: Call history is recorded
# ---------------------------------------------------------------------------

def test_call_history():
    controller = ToolSecurityController(
        max_tool_calls=4,
        max_calls_per_tool=2,
    )

    controller.authorize_tool(
        "both",
        "weather",
    )

    controller.authorize_tool(
        "both",
        "stackexchange",
    )

    assert controller.call_history == [
        "weather",
        "stackexchange",
    ]


# ---------------------------------------------------------------------------
# Test 19: New controllers have isolated state
# ---------------------------------------------------------------------------

def test_request_state_isolation():
    first_request = ToolSecurityController(
        max_tool_calls=1,
        max_calls_per_tool=1,
    )

    second_request = ToolSecurityController(
        max_tool_calls=1,
        max_calls_per_tool=1,
    )

    first = first_request.authorize_tool(
        "weather",
        "weather",
    )

    second = second_request.authorize_tool(
        "weather",
        "weather",
    )

    assert first.allowed is True
    assert second.allowed is True


# ---------------------------------------------------------------------------
# Test 20: State reset works
# ---------------------------------------------------------------------------

def test_state_reset():
    state = ToolExecutionState(
        max_tool_calls=2,
        max_calls_per_tool=2,
    )

    first = state.authorize(
        "weather",
        "weather",
    )

    assert first.allowed is True
    assert state.total_calls == 1

    state.reset()

    assert state.total_calls == 0
    assert state.call_history == []

    second = state.authorize(
        "weather",
        "weather",
    )

    assert second.allowed is True


# ---------------------------------------------------------------------------
# Test 21: Invalid max_tool_calls rejected
# ---------------------------------------------------------------------------

def test_invalid_max_tool_calls_rejected():
    with pytest.raises(ValueError):
        ToolSecurityController(
            max_tool_calls=0,
            max_calls_per_tool=1,
        )


# ---------------------------------------------------------------------------
# Test 22: Invalid max_calls_per_tool rejected
# ---------------------------------------------------------------------------

def test_invalid_max_calls_per_tool_rejected():
    with pytest.raises(ValueError):
        ToolSecurityController(
            max_tool_calls=4,
            max_calls_per_tool=0,
        )


# ---------------------------------------------------------------------------
# Test 23: Per-tool limit cannot exceed total limit
# ---------------------------------------------------------------------------

def test_per_tool_limit_cannot_exceed_total():
    with pytest.raises(ValueError):
        ToolSecurityController(
            max_tool_calls=2,
            max_calls_per_tool=3,
        )


# ---------------------------------------------------------------------------
# Test 24: Call count never exceeds configured maximum
# ---------------------------------------------------------------------------

def test_call_count_never_exceeds_maximum():
    controller = ToolSecurityController(
        max_tool_calls=3,
        max_calls_per_tool=3,
    )

    for _ in range(10):
        controller.authorize_tool(
            "weather",
            "weather",
        )

    assert controller.total_calls == 3


# ---------------------------------------------------------------------------
# Test 25: Failed authorization returns remaining count
# ---------------------------------------------------------------------------

def test_failed_authorization_reports_remaining_count():
    controller = ToolSecurityController(
        max_tool_calls=1,
        max_calls_per_tool=1,
    )

    first = controller.authorize_tool(
        "weather",
        "weather",
    )

    second = controller.authorize_tool(
        "weather",
        "weather",
    )

    assert first.allowed is True
    assert first.remaining_calls == 0

    assert second.allowed is False
    assert second.remaining_calls == 0

"""
Tool permission and agent execution security.

This module is the authorization boundary between the agent/router and
external tools.

Security principles
-------------------
- Fail closed.
- Explicit tool allowlisting.
- Explicit route -> tool permissions.
- Maximum total tool calls per request.
- Maximum calls per individual tool per request.
- No unauthorized tool switching.
- No duplicate/unbounded tool execution.
- Per-request execution state is isolated.
- Tool names and routes are treated as untrusted input.
- Authorization and execution-slot reservation happen atomically.
- Denied authorization attempts never consume execution slots.
- Execution state is safe to inspect for observability.
- No secrets, query contents, URLs, or external API responses are stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Final


# ============================================================================
# Allowed routes
# ============================================================================

ALLOWED_ROUTES: Final[frozenset[str]] = frozenset(
    {
        "stackexchange",
        "weather",
        "tavily",
        "both",
        "unsupported",
    }
)


# ============================================================================
# Allowed tools
# ============================================================================

ALLOWED_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "stackexchange",
        "weather",
        "tavily",
    }
)


# ============================================================================
# Explicit route -> tool authorization matrix
# ============================================================================

ROUTE_TOOL_PERMISSIONS: Final[dict[str, frozenset[str]]] = {
    "stackexchange": frozenset(
        {
            "stackexchange",
        }
    ),
    "weather": frozenset(
        {
            "weather",
        }
    ),
    "tavily": frozenset(
        {
            "tavily",
        }
    ),
    "both": frozenset(
        {
            "stackexchange",
            "weather",
            "tavily",
        }
    ),
    "unsupported": frozenset(),
}


# ============================================================================
# Security limits
# ============================================================================

DEFAULT_MAX_TOOL_CALLS: Final[int] = 4

# Prevent repeated calls to the same external tool during one request.
DEFAULT_MAX_CALLS_PER_TOOL: Final[int] = 2


# ============================================================================
# Result objects
# ============================================================================

@dataclass(frozen=True)
class ToolAuthorizationResult:
    """
    Result returned by the tool authorization boundary.

    Attributes
    ----------
    allowed:
        Whether the tool execution is authorized.

    tool:
        Normalized tool name.

    route:
        Normalized research route.

    reason:
        Human-readable security decision.

    remaining_calls:
        Number of total execution slots remaining after this decision.

    tool_calls:
        Number of calls already reserved for this tool.

    remaining_tool_calls:
        Number of additional calls this tool may make before hitting
        its per-tool limit.
    """

    allowed: bool
    tool: str
    route: str
    reason: str
    remaining_calls: int
    tool_calls: int = 0
    remaining_tool_calls: int = 0


@dataclass(frozen=True)
class ToolExecutionRecord:
    """
    Safe execution authorization record.

    This intentionally stores only security metadata.

    It does NOT store:
    - API keys
    - user questions
    - search queries
    - URLs
    - API responses
    - model output
    """

    sequence: int
    tool: str
    route: str


# ============================================================================
# Per-request execution state
# ============================================================================

@dataclass
class ToolExecutionState:
    """
    Isolated security state for one agent request.

    IMPORTANT:
        Create a new state/controller for every user request.

    The state tracks reservations, not successful API executions.

    A slot is reserved when authorization succeeds. This prevents the
    agent from repeatedly requesting authorization before executing a
    tool and bypassing execution limits.
    """

    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_calls_per_tool: int = DEFAULT_MAX_CALLS_PER_TOOL

    total_calls: int = 0

    calls_by_tool: dict[str, int] = field(default_factory=dict)

    call_history: list[str] = field(default_factory=list)

    denied_attempts: int = 0

    _lock: Lock = field(
        default_factory=Lock,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate state configuration immediately.
        """

        _validate_limits(
            max_tool_calls=self.max_tool_calls,
            max_calls_per_tool=self.max_calls_per_tool,
        )

    def authorize(
        self,
        route: str,
        tool: str,
    ) -> ToolAuthorizationResult:
        """
        Authorize and reserve one tool execution slot.

        Authorization and reservation are intentionally performed
        atomically under the same lock.

        This is important for concurrent agent execution because:

            request A -> authorize
            request B -> authorize

        cannot both observe the same remaining slot and exceed the limit.

        Returns
        -------
        ToolAuthorizationResult
            Structured authorization decision.
        """

        normalized_route = _normalize_name(route)
        normalized_tool = _normalize_name(tool)

        # ------------------------------------------------------------------
        # Validate route
        # ------------------------------------------------------------------

        if normalized_route not in ALLOWED_ROUTES:
            return self._deny(
                tool=normalized_tool,
                route=normalized_route,
                reason="Unauthorized route.",
            )

        # ------------------------------------------------------------------
        # Validate tool
        # ------------------------------------------------------------------

        if normalized_tool not in ALLOWED_TOOLS:
            return self._deny(
                tool=normalized_tool,
                route=normalized_route,
                reason="Unauthorized tool.",
            )

        # ------------------------------------------------------------------
        # Explicit route -> tool permission
        # ------------------------------------------------------------------

        allowed_tools = ROUTE_TOOL_PERMISSIONS.get(
            normalized_route,
            frozenset(),
        )

        if normalized_tool not in allowed_tools:
            return self._deny(
                tool=normalized_tool,
                route=normalized_route,
                reason=(
                    f"Tool '{normalized_tool}' is not authorized for "
                    f"route '{normalized_route}'."
                ),
            )

        # ------------------------------------------------------------------
        # Atomic execution reservation
        # ------------------------------------------------------------------

        with self._lock:
            current_total = self.total_calls
            current_tool_calls = self.calls_by_tool.get(
                normalized_tool,
                0,
            )

            remaining_total = max(
                0,
                self.max_tool_calls - current_total,
            )

            remaining_for_tool = max(
                0,
                self.max_calls_per_tool - current_tool_calls,
            )

            # --------------------------------------------------------------
            # Global execution limit
            # --------------------------------------------------------------

            if current_total >= self.max_tool_calls:
                self.denied_attempts += 1

                return ToolAuthorizationResult(
                    allowed=False,
                    tool=normalized_tool,
                    route=normalized_route,
                    reason=(
                        "Maximum tool calls per request exceeded."
                    ),
                    remaining_calls=0,
                    tool_calls=current_tool_calls,
                    remaining_tool_calls=remaining_for_tool,
                )

            # --------------------------------------------------------------
            # Per-tool execution limit
            # --------------------------------------------------------------

            if current_tool_calls >= self.max_calls_per_tool:
                self.denied_attempts += 1

                return ToolAuthorizationResult(
                    allowed=False,
                    tool=normalized_tool,
                    route=normalized_route,
                    reason=(
                        f"Maximum calls for tool '{normalized_tool}' "
                        "per request exceeded."
                    ),
                    remaining_calls=remaining_total,
                    tool_calls=current_tool_calls,
                    remaining_tool_calls=0,
                )

            # --------------------------------------------------------------
            # Reserve BEFORE external execution
            # --------------------------------------------------------------

            self.total_calls += 1

            self.calls_by_tool[normalized_tool] = (
                current_tool_calls + 1
            )

            self.call_history.append(normalized_tool)

            return ToolAuthorizationResult(
                allowed=True,
                tool=normalized_tool,
                route=normalized_route,
                reason="Tool execution authorized.",
                remaining_calls=(
                    self.max_tool_calls - self.total_calls
                ),
                tool_calls=current_tool_calls + 1,
                remaining_tool_calls=(
                    self.max_calls_per_tool
                    - current_tool_calls
                    - 1
                ),
            )

    def _deny(
        self,
        tool: str,
        route: str,
        reason: str,
    ) -> ToolAuthorizationResult:
        """
        Record a denied authorization attempt.

        No execution slot is consumed.
        """

        with self._lock:
            self.denied_attempts += 1

            current_tool_calls = self.calls_by_tool.get(
                tool,
                0,
            )

            return ToolAuthorizationResult(
                allowed=False,
                tool=tool,
                route=route,
                reason=reason,
                remaining_calls=max(
                    0,
                    self.max_tool_calls - self.total_calls,
                ),
                tool_calls=current_tool_calls,
                remaining_tool_calls=max(
                    0,
                    self.max_calls_per_tool - current_tool_calls,
                ),
            )

    def _remaining_calls(self) -> int:
        """
        Return remaining global execution slots safely.
        """

        with self._lock:
            return max(
                0,
                self.max_tool_calls - self.total_calls,
            )

    def remaining_calls_for_tool(self, tool: str) -> int:
        """
        Return remaining execution slots for one normalized tool.
        """

        normalized_tool = _normalize_name(tool)

        with self._lock:
            current = self.calls_by_tool.get(
                normalized_tool,
                0,
            )

            return max(
                0,
                self.max_calls_per_tool - current,
            )

    def snapshot(self) -> dict:
        """
        Return a safe immutable-style snapshot of security state.

        This is intended for logs/tracing/observability.

        Sensitive request content is intentionally excluded.
        """

        with self._lock:
            return {
                "max_tool_calls": self.max_tool_calls,
                "max_calls_per_tool": self.max_calls_per_tool,
                "total_calls": self.total_calls,
                "calls_by_tool": dict(self.calls_by_tool),
                "call_history": list(self.call_history),
                "denied_attempts": self.denied_attempts,
                "remaining_calls": max(
                    0,
                    self.max_tool_calls - self.total_calls,
                ),
            }

    def reset(self) -> None:
        """
        Reset execution state.

        Normally, creating a new controller per request is preferable.

        This method exists for controlled testing and explicit reuse.
        """

        with self._lock:
            self.total_calls = 0
            self.calls_by_tool.clear()
            self.call_history.clear()
            self.denied_attempts = 0


# ============================================================================
# High-level security controller
# ============================================================================

class ToolSecurityController:
    """
    High-level controller for agent tool authorization.

    Create one controller per user request.

    Example
    -------
    controller = ToolSecurityController()

    authorization = controller.authorize_tool(
        route="tavily",
        tool="tavily",
    )

    if authorization.allowed:
        # Execute Tavily
        ...
    """

    def __init__(
        self,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_calls_per_tool: int = DEFAULT_MAX_CALLS_PER_TOOL,
    ) -> None:

        _validate_limits(
            max_tool_calls=max_tool_calls,
            max_calls_per_tool=max_calls_per_tool,
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
        Authorize a tool before executing it.
        """

        return self.state.authorize(
            route=route,
            tool=tool,
        )

    @property
    def total_calls(self) -> int:
        """
        Number of reserved tool execution slots.
        """

        return self.state.total_calls

    @property
    def call_history(self) -> list[str]:
        """
        Return a defensive copy of the call history.
        """

        with self.state._lock:
            return list(self.state.call_history)

    @property
    def denied_attempts(self) -> int:
        """
        Number of denied authorization attempts.
        """

        with self.state._lock:
            return self.state.denied_attempts

    @property
    def remaining_calls(self) -> int:
        """
        Remaining global execution slots.
        """

        return self.state._remaining_calls()

    def remaining_calls_for_tool(self, tool: str) -> int:
        """
        Return remaining calls available for a specific tool.
        """

        return self.state.remaining_calls_for_tool(tool)

    def snapshot(self) -> dict:
        """
        Return a safe security snapshot for observability.
        """

        return self.state.snapshot()


# ============================================================================
# Stateless convenience authorization
# ============================================================================

def authorize_tool(
    route: str,
    tool: str,
) -> ToolAuthorizationResult:
    """
    Stateless authorization helper.

    This checks route/tool permission but does not maintain execution
    counters.

    Use ToolSecurityController when execution limits must be enforced.
    """

    normalized_route = _normalize_name(route)
    normalized_tool = _normalize_name(tool)

    # ----------------------------------------------------------------------
    # Route validation
    # ----------------------------------------------------------------------

    if normalized_route not in ALLOWED_ROUTES:
        return ToolAuthorizationResult(
            allowed=False,
            tool=normalized_tool,
            route=normalized_route,
            reason="Unauthorized route.",
            remaining_calls=0,
        )

    # ----------------------------------------------------------------------
    # Tool validation
    # ----------------------------------------------------------------------

    if normalized_tool not in ALLOWED_TOOLS:
        return ToolAuthorizationResult(
            allowed=False,
            tool=normalized_tool,
            route=normalized_route,
            reason="Unauthorized tool.",
            remaining_calls=0,
        )

    # ----------------------------------------------------------------------
    # Permission matrix
    # ----------------------------------------------------------------------

    if normalized_tool not in ROUTE_TOOL_PERMISSIONS.get(
        normalized_route,
        frozenset(),
    ):
        return ToolAuthorizationResult(
            allowed=False,
            tool=normalized_tool,
            route=normalized_route,
            reason=(
                f"Tool '{normalized_tool}' is not authorized for "
                f"route '{normalized_route}'."
            ),
            remaining_calls=0,
        )

    return ToolAuthorizationResult(
        allowed=True,
        tool=normalized_tool,
        route=normalized_route,
        reason="Tool execution authorized.",
        remaining_calls=0,
    )


# ============================================================================
# Validation helpers
# ============================================================================

def _validate_limits(
    max_tool_calls: int,
    max_calls_per_tool: int,
) -> None:
    """
    Validate execution limits.

    Fail closed against invalid security configuration.
    """

    if (
        isinstance(max_tool_calls, bool)
        or not isinstance(max_tool_calls, int)
    ):
        raise ValueError(
            "max_tool_calls must be an integer."
        )

    if (
        isinstance(max_calls_per_tool, bool)
        or not isinstance(max_calls_per_tool, int)
    ):
        raise ValueError(
            "max_calls_per_tool must be an integer."
        )

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


def _normalize_name(value: str) -> str:
    """
    Normalize a route/tool identifier.

    Non-string values are rejected rather than implicitly converted.
    """

    if not isinstance(value, str):
        return ""

    normalized = value.strip().lower()

    # Prevent pathological identifiers from entering authorization
    # or observability state.
    if len(normalized) > 100:
        return ""

    return normalized


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "ALLOWED_ROUTES",
    "ALLOWED_TOOLS",
    "ROUTE_TOOL_PERMISSIONS",
    "DEFAULT_MAX_TOOL_CALLS",
    "DEFAULT_MAX_CALLS_PER_TOOL",
    "ToolAuthorizationResult",
    "ToolExecutionRecord",
    "ToolExecutionState",
    "ToolSecurityController",
    "authorize_tool",
]
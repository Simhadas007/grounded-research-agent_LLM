import pytest

from security.tool_security import (
    ALLOWED_ROUTES,
    ALLOWED_TOOLS,
    ROUTE_TOOL_PERMISSIONS,
    ToolExecutionState,
    ToolSecurityController,
    authorize_tool,
)


# ============================================================================
# Basic route/tool authorization
# ============================================================================

def test_stackexchange_route_allows_stackexchange():
    result = authorize_tool(
        "stackexchange",
        "stackexchange",
    )

    assert result.allowed is True


def test_weather_route_allows_weather():
    result = authorize_tool(
        "weather",
        "weather",
    )

    assert result.allowed is True


def test_tavily_route_allows_tavily():
    result = authorize_tool(
        "tavily",
        "tavily",
    )

    assert result.allowed is True


def test_stackexchange_route_blocks_weather():
    result = authorize_tool(
        "stackexchange",
        "weather",
    )

    assert result.allowed is False


def test_stackexchange_route_blocks_tavily():
    result = authorize_tool(
        "stackexchange",
        "tavily",
    )

    assert result.allowed is False


def test_weather_route_blocks_stackexchange():
    result = authorize_tool(
        "weather",
        "stackexchange",
    )

    assert result.allowed is False


def test_weather_route_blocks_tavily():
    result = authorize_tool(
        "weather",
        "tavily",
    )

    assert result.allowed is False


def test_tavily_route_blocks_weather():
    result = authorize_tool(
        "tavily",
        "weather",
    )

    assert result.allowed is False


def test_tavily_route_blocks_stackexchange():
    result = authorize_tool(
        "tavily",
        "stackexchange",
    )

    assert result.allowed is False


# ============================================================================
# Both route
# ============================================================================

def test_both_route_allows_stackexchange():
    result = authorize_tool(
        "both",
        "stackexchange",
    )

    assert result.allowed is True


def test_both_route_allows_weather():
    result = authorize_tool(
        "both",
        "weather",
    )

    assert result.allowed is True


def test_both_route_allows_tavily():
    result = authorize_tool(
        "both",
        "tavily",
    )

    assert result.allowed is True


# ============================================================================
# Unsupported / invalid input
# ============================================================================

def test_unsupported_route_blocks_tools():
    stack_result = authorize_tool(
        "unsupported",
        "stackexchange",
    )

    weather_result = authorize_tool(
        "unsupported",
        "weather",
    )

    tavily_result = authorize_tool(
        "unsupported",
        "tavily",
    )

    assert stack_result.allowed is False
    assert weather_result.allowed is False
    assert tavily_result.allowed is False


def test_unknown_tool_rejected():
    result = authorize_tool(
        "both",
        "database",
    )

    assert result.allowed is False


def test_unknown_route_rejected():
    result = authorize_tool(
        "admin",
        "weather",
    )

    assert result.allowed is False


def test_tool_name_normalization():
    result = authorize_tool(
        " WEATHER ",
        " WEATHER ",
    )

    assert result.allowed is True
    assert result.route == "weather"
    assert result.tool == "weather"


def test_tavily_name_normalization():
    result = authorize_tool(
        " TAVILY ",
        " TAVILY ",
    )

    assert result.allowed is True
    assert result.route == "tavily"
    assert result.tool == "tavily"


def test_non_string_route_rejected():
    result = authorize_tool(
        None,
        "weather",
    )

    assert result.allowed is False


def test_non_string_tool_rejected():
    result = authorize_tool(
        "weather",
        None,
    )

    assert result.allowed is False


def test_empty_route_rejected():
    result = authorize_tool(
        "",
        "weather",
    )

    assert result.allowed is False


def test_empty_tool_rejected():
    result = authorize_tool(
        "weather",
        "",
    )

    assert result.allowed is False


# ============================================================================
# Configuration / allowlists
# ============================================================================

def test_tavily_is_in_allowed_routes():
    assert "tavily" in ALLOWED_ROUTES


def test_tavily_is_in_allowed_tools():
    assert "tavily" in ALLOWED_TOOLS


def test_tavily_permission_matrix_is_explicit():
    assert ROUTE_TOOL_PERMISSIONS["tavily"] == {
        "tavily",
    }


def test_both_permission_matrix_contains_all_tools():
    assert ROUTE_TOOL_PERMISSIONS["both"] == {
        "stackexchange",
        "weather",
        "tavily",
    }


def test_unsupported_permission_matrix_is_empty():
    assert ROUTE_TOOL_PERMISSIONS["unsupported"] == set()


# ============================================================================
# Stateful execution authorization
# ============================================================================

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
    assert result.remaining_calls == 3
    assert result.tool_calls == 1
    assert result.remaining_tool_calls == 1


def test_first_tavily_call_allowed():
    controller = ToolSecurityController()

    result = controller.authorize_tool(
        "tavily",
        "tavily",
    )

    assert result.allowed is True
    assert controller.total_calls == 1
    assert controller.call_history == ["tavily"]


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
        "tavily",
    )

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False

    assert controller.total_calls == 2


def test_maximum_calls_per_tool_enforced():
    controller = ToolSecurityController(
        max_tool_calls=4,
        max_calls_per_tool=2,
    )

    first = controller.authorize_tool(
        "tavily",
        "tavily",
    )

    second = controller.authorize_tool(
        "tavily",
        "tavily",
    )

    third = controller.authorize_tool(
        "tavily",
        "tavily",
    )

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False

    assert controller.total_calls == 2
    assert controller.remaining_calls_for_tool("tavily") == 0


def test_both_route_allows_three_different_tools():
    controller = ToolSecurityController(
        max_tool_calls=4,
        max_calls_per_tool=2,
    )

    stackexchange = controller.authorize_tool(
        "both",
        "stackexchange",
    )

    weather = controller.authorize_tool(
        "both",
        "weather",
    )

    tavily = controller.authorize_tool(
        "both",
        "tavily",
    )

    assert stackexchange.allowed is True
    assert weather.allowed is True
    assert tavily.allowed is True

    assert controller.total_calls == 3


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
    assert controller.denied_attempts == 1


def test_call_history_is_recorded():
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

    controller.authorize_tool(
        "both",
        "tavily",
    )

    assert controller.call_history == [
        "weather",
        "stackexchange",
        "tavily",
    ]


def test_call_history_is_defensive_copy():
    controller = ToolSecurityController()

    controller.authorize_tool(
        "tavily",
        "tavily",
    )

    history = controller.call_history

    history.clear()

    assert controller.call_history == ["tavily"]


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
        "tavily",
        "tavily",
    )

    assert first.allowed is True
    assert second.allowed is True


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
    assert state.denied_attempts == 0

    second = state.authorize(
        "tavily",
        "tavily",
    )

    assert second.allowed is True


# ============================================================================
# Limits
# ============================================================================

def test_invalid_max_tool_calls_rejected():
    with pytest.raises(ValueError):
        ToolSecurityController(
            max_tool_calls=0,
            max_calls_per_tool=1,
        )


def test_invalid_max_calls_per_tool_rejected():
    with pytest.raises(ValueError):
        ToolSecurityController(
            max_tool_calls=4,
            max_calls_per_tool=0,
        )


def test_per_tool_limit_cannot_exceed_total_limit():
    with pytest.raises(ValueError):
        ToolSecurityController(
            max_tool_calls=2,
            max_calls_per_tool=3,
        )


def test_boolean_max_tool_calls_rejected():
    with pytest.raises(ValueError):
        ToolSecurityController(
            max_tool_calls=True,
            max_calls_per_tool=1,
        )


def test_boolean_max_calls_per_tool_rejected():
    with pytest.raises(ValueError):
        ToolSecurityController(
            max_tool_calls=2,
            max_calls_per_tool=False,
        )


def test_non_integer_max_tool_calls_rejected():
    with pytest.raises(ValueError):
        ToolSecurityController(
            max_tool_calls="4",
            max_calls_per_tool=1,
        )


def test_non_integer_max_calls_per_tool_rejected():
    with pytest.raises(ValueError):
        ToolSecurityController(
            max_tool_calls=4,
            max_calls_per_tool="2",
        )


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


def test_remaining_calls_for_tavily():
    controller = ToolSecurityController(
        max_tool_calls=4,
        max_calls_per_tool=2,
    )

    assert controller.remaining_calls_for_tool("tavily") == 2

    controller.authorize_tool(
        "tavily",
        "tavily",
    )

    assert controller.remaining_calls_for_tool("tavily") == 1

    controller.authorize_tool(
        "tavily",
        "tavily",
    )

    assert controller.remaining_calls_for_tool("tavily") == 0


# ============================================================================
# Snapshot / observability
# ============================================================================

def test_security_snapshot_contains_safe_metadata():
    controller = ToolSecurityController()

    controller.authorize_tool(
        "tavily",
        "tavily",
    )

    snapshot = controller.snapshot()

    assert snapshot["max_tool_calls"] == 4
    assert snapshot["max_calls_per_tool"] == 2
    assert snapshot["total_calls"] == 1
    assert snapshot["calls_by_tool"] == {
        "tavily": 1,
    }
    assert snapshot["call_history"] == [
        "tavily",
    ]
    assert snapshot["remaining_calls"] == 3


def test_snapshot_is_defensive():
    controller = ToolSecurityController()

    controller.authorize_tool(
        "tavily",
        "tavily",
    )

    snapshot = controller.snapshot()

    snapshot["call_history"].clear()
    snapshot["calls_by_tool"].clear()

    fresh_snapshot = controller.snapshot()

    assert fresh_snapshot["call_history"] == ["tavily"]
    assert fresh_snapshot["calls_by_tool"] == {
        "tavily": 1,
    }


# ============================================================================
# Additional security boundary tests
# ============================================================================

def test_route_cannot_switch_from_tavily_to_weather():
    controller = ToolSecurityController()

    tavily = controller.authorize_tool(
        "tavily",
        "tavily",
    )

    weather = controller.authorize_tool(
        "tavily",
        "weather",
    )

    assert tavily.allowed is True
    assert weather.allowed is False

    assert controller.total_calls == 1


def test_route_cannot_switch_from_weather_to_tavily():
    controller = ToolSecurityController()

    weather = controller.authorize_tool(
        "weather",
        "weather",
    )

    tavily = controller.authorize_tool(
        "weather",
        "tavily",
    )

    assert weather.allowed is True
    assert tavily.allowed is False

    assert controller.total_calls == 1


def test_unauthorized_tavily_attempt_does_not_consume_slot():
    controller = ToolSecurityController(
        max_tool_calls=1,
        max_calls_per_tool=1,
    )

    denied = controller.authorize_tool(
        "weather",
        "tavily",
    )

    allowed = controller.authorize_tool(
        "weather",
        "weather",
    )

    assert denied.allowed is False
    assert allowed.allowed is True

    assert controller.total_calls == 1
    assert controller.denied_attempts == 1


def test_tavily_can_be_used_twice_but_not_three_times():
    controller = ToolSecurityController(
        max_tool_calls=4,
        max_calls_per_tool=2,
    )

    first = controller.authorize_tool(
        "tavily",
        "tavily",
    )

    second = controller.authorize_tool(
        "tavily",
        "tavily",
    )

    third = controller.authorize_tool(
        "tavily",
        "tavily",
    )

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False

    assert controller.total_calls == 2


def test_total_limit_takes_precedence_after_limit_reached():
    controller = ToolSecurityController(
        max_tool_calls=2,
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

    denied = controller.authorize_tool(
        "both",
        "tavily",
    )

    assert denied.allowed is False
    assert denied.remaining_calls == 0
    assert controller.total_calls == 2
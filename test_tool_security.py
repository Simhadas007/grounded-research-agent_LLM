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
# Test 10: Tool and route names are normalized
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
# Test 16: Both route allows two different tools
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
# Test 25: Failed authorization reports remaining count
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

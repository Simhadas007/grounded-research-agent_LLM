from unittest.mock import patch

from security.tool_security import ToolSecurityController


def test_stackexchange_tool_requires_authorization():
    controller = ToolSecurityController()

    authorization = controller.authorize_tool(
        route="stackexchange",
        tool="stackexchange",
    )

    assert authorization.allowed is True


def test_weather_tool_requires_authorization():
    controller = ToolSecurityController()

    authorization = controller.authorize_tool(
        route="weather",
        tool="weather",
    )

    assert authorization.allowed is True


def test_stackexchange_route_cannot_authorize_weather():
    controller = ToolSecurityController()

    authorization = controller.authorize_tool(
        route="stackexchange",
        tool="weather",
    )

    assert authorization.allowed is False


def test_weather_route_cannot_authorize_stackexchange():
    controller = ToolSecurityController()

    authorization = controller.authorize_tool(
        route="weather",
        tool="stackexchange",
    )

    assert authorization.allowed is False


def test_both_route_can_authorize_both_tools():
    controller = ToolSecurityController()

    stackexchange = controller.authorize_tool(
        route="both",
        tool="stackexchange",
    )

    weather = controller.authorize_tool(
        route="both",
        tool="weather",
    )

    assert stackexchange.allowed is True
    assert weather.allowed is True
    assert controller.total_calls == 2


def test_denied_tool_does_not_consume_call_slot():
    controller = ToolSecurityController()

    authorization = controller.authorize_tool(
        route="stackexchange",
        tool="weather",
    )

    assert authorization.allowed is False
    assert controller.total_calls == 0


def test_authorization_happens_before_tool_execution():
    controller = ToolSecurityController()

    execution_order = []

    authorization = controller.authorize_tool(
        route="weather",
        tool="weather",
    )

    execution_order.append("authorized")

    if authorization.allowed:
        execution_order.append("tool_called")

    assert execution_order == [
        "authorized",
        "tool_called",
    ]


def test_denied_authorization_prevents_execution():
    controller = ToolSecurityController()

    tool_called = False

    authorization = controller.authorize_tool(
        route="stackexchange",
        tool="weather",
    )

    if authorization.allowed:
        tool_called = True

    assert authorization.allowed is False
    assert tool_called is False


def test_fresh_controller_isolated_per_request():
    request_one = ToolSecurityController()
    request_two = ToolSecurityController()

    first = request_one.authorize_tool(
        route="weather",
        tool="weather",
    )

    assert first.allowed is True
    assert request_one.total_calls == 1
    assert request_two.total_calls == 0


def test_stackexchange_not_called_when_authorization_fails():
    controller = ToolSecurityController()

    with patch(
        "tools.stackexchange.search_stackexchange"
    ) as mock_search:

        authorization = controller.authorize_tool(
            route="weather",
            tool="stackexchange",
        )

        if authorization.allowed:
            mock_search("test")

        assert authorization.allowed is False
        mock_search.assert_not_called()


def test_weather_not_called_when_authorization_fails():
    controller = ToolSecurityController()

    with patch(
        "tools.weather.get_weather"
    ) as mock_weather:

        authorization = controller.authorize_tool(
            route="stackexchange",
            tool="weather",
        )

        if authorization.allowed:
            mock_weather("Chennai")

        assert authorization.allowed is False
        mock_weather.assert_not_called()

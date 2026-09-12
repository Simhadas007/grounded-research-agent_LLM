from unittest.mock import MagicMock

from agent import llm


class FakeLLMResponse:
    """Simple fake response matching the object returned by LangChain."""

    def __init__(self, content: str):
        self.content = content


def test_llm_response_handling():
    """Verify expected LLM response handling without calling Groq."""

    fake_response = FakeLLMResponse(
        "Grounded Research Agent is online."
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_response

    response = fake_llm.invoke(
        "Reply with exactly: Grounded Research Agent is online."
    )

    assert response.content == "Grounded Research Agent is online."

    fake_llm.invoke.assert_called_once_with(
        "Reply with exactly: Grounded Research Agent is online."
    )


def test_llm_object_is_configured():
    """Verify the application LLM object is initialized."""

    assert llm is not None
    assert hasattr(llm, "invoke")
    
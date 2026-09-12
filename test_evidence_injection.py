import utils.research_answer as research_answer


EVIDENCE = [
    {
        "site": "stackoverflow",
        "site_name": "Stack Overflow",
        "title": "Why useEffect running twice and how to handle it well in React?",
        "question_body": (
            "I have a counter and a console.log() in an useEffect "
            "to log every change in my state, but the useEffect is "
            "getting called two times on mount. I am using React 18."
        ),
        "answer_body": (
            "useEffect being called twice on mount is normal since "
            "React version 18 when you are in development with StrictMode."
        ),
        "question_score": 174,
        "answer_score": 284,
        "tags": ["javascript", "reactjs", "next.js", "react-hooks"],
        "url": (
            "https://stackoverflow.com/questions/"
            "72238175/why-useeffect-running-twice-and-how-to-handle-it-well-in-react"
        ),
        "is_answered": True,
    }
]


def test_generate_research_answer_with_mocked_llm(monkeypatch):
    """
    Test answer generation without making a real Groq API request.
    """

    class FakeLLM:
        def invoke(self, prompt):
            class FakeResponse:
                content = (
                    '{"relevant":true,'
                    '"sufficient":true,'
                    '"quality":"high",'
                    '"reason":"The retrieved Stack Overflow evidence directly '
                    'explains the React 18 StrictMode behavior.",'
                    '"answer":"React 18 StrictMode can intentionally run effects '
                    'twice during development to help detect side effects. '
                    'This behavior does not mean the effect runs twice in '
                    'production.",'
                    '"citations":["https://stackoverflow.com/questions/'
                    '72238175/why-useeffect-running-twice-and-how-to-handle-it-well-in-react"]}'
                )

            return FakeResponse()

    monkeypatch.setattr(
        research_answer,
        "llm",
        FakeLLM(),
        raising=False,
    )

    result = research_answer.generate_research_answer(
        question="Why does useEffect run twice in React 18?",
        source="stackexchange",
        evidence=EVIDENCE,
    )

    assert result.relevant is True
    assert result.sufficient is True
    assert result.quality == "high"
    assert result.answer
    assert len(result.citations) == 1
    assert result.citations[0] == EVIDENCE[0]["url"]
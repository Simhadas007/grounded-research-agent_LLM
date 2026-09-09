from utils.research_answer import generate_research_answer

evidence = [
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

result = generate_research_answer(
    question="Why does useEffect run twice in React 18?",
    source="stackexchange",
    evidence=evidence,
)

print("\n================ RESULT ================\n")
print(result)
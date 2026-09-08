from utils.evidence import evaluate_evidence


question = "Why is my React useEffect running twice?"

evidence = [
    {
        "title": "useEffect runs twice in React 18",
        "url": "https://stackoverflow.com/example",
        "score": 120,
        "tags": ["reactjs", "javascript"],
        "is_answered": True,
    },
    {
        "title": "React StrictMode causes effects to run twice",
        "url": "https://stackoverflow.com/example2",
        "score": 85,
        "tags": ["reactjs", "react-hooks"],
        "is_answered": True,
    },
]


decision = evaluate_evidence(
    question=question,
    route="stackexchange",
    evidence=evidence,
)


print("=" * 60)
print("Relevant:", decision.relevant)
print("Sufficient:", decision.sufficient)
print("Quality:", decision.quality)
print("Reason:", decision.reason)
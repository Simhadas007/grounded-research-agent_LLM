from graph import retrieval_node


state = {
    "question": "Why is my React useEffect running twice?",
    "route": "stackexchange",
    "search_query": "React useEffect running twice",
}

result = retrieval_node(state)

print("\nRetrieval result:")
print("Error:", result.get("error"))
print("Evidence count:", len(result.get("evidence", [])))

for item in result.get("evidence", []):
    print("\nTitle:", item.get("title"))
    print("URL:", item.get("url"))
    print("Question:", item.get("question_body", "")[:300])
    print("Answer:", item.get("answer_body", "")[:500])
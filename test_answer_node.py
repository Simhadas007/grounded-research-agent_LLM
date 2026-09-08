from graph import retrieval_node, evidence_node, answer_node


state = {
    "question": "Why is my React useEffect running twice?",
    "route": "stackexchange",
    "search_query": "React useEffect running twice",
}

state = retrieval_node(state)

state = evidence_node(state)

state = answer_node(state)

print("\n==============================")
print("GROUNDED ANSWER")
print("==============================")
print(state["answer"])
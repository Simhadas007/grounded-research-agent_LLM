from graph import retrieval_node, evidence_node


state = {
    "question": "Why is my React useEffect running twice?",
    "route": "stackexchange",
    "search_query": "React useEffect running twice",
}

# First retrieve live evidence
state = retrieval_node(state)

# Then evaluate the evidence with AI
state = evidence_node(state)

print("\nEvidence node result:")
print("Relevant:", state.get("evidence_relevant"))
print("Sufficient:", state.get("evidence_sufficient"))
print("Quality:", state.get("evidence_quality"))
print("Reason:", state.get("evidence_reason"))
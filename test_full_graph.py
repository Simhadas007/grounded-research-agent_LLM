from graph import app


question = "Why is my React useEffect running twice?"


initial_state = {
    "question": question,
}


result = app.invoke(initial_state)


print("\n================================")
print("GROUNDED RESEARCH AGENT")
print("================================")

print("\nQuestion:")
print(result.get("question"))

print("\nRoute:")
print(result.get("route"))

print("\nRoute reason:")
print(result.get("route_reason"))

print("\nSearch query:")
print(result.get("search_query"))

print("\nEvidence count:")
print(len(result.get("evidence", [])))

print("\nEvidence quality:")
print(result.get("evidence_quality"))

print("\nEvidence sufficient:")
print(result.get("evidence_sufficient"))

print("\nAnswer:")
print(result.get("answer"))

print("\nCitations:")
for citation in result.get("citations", []):
    print("-", citation)

print("\nError:")
print(result.get("error"))
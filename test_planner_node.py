from graph import planner_node


state = {
    "question": "Why is my React useEffect running twice?",
    "route": "stackexchange",
    "route_reason": "Technical React question.",
}

result = planner_node(state)

print("\nPlanner node result:")
print(result)
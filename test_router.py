from router import route_question


questions = [
    "Why is my React useEffect running twice?",
    "What is the weather in Chennai today?",
    "What is the weather in Chennai and should I take an umbrella?",
    "Tell me a funny joke.",
]


for question in questions:
    result = route_question(question)

    print("\nQuestion:", question)
    print("Route:", result.route)
    print("Reason:", result.reason)
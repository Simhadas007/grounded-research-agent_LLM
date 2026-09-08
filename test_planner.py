from planner import create_search_plan


tests = [
    (
        "Technical",
        "Why does useEffect run twice in React development mode?",
        "stackexchange",
    ),
    (
        "Weather",
        "What is the current weather in Chennai?",
        "weather",
    ),
    (
        "Unsupported",
        "Who is the greatest actor in the world?",
        "unsupported",
    ),
]


for name, question, route in tests:

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    try:

        plan = create_search_plan(
            question=question,
            route=route,
        )

        print("Question:", question)
        print("Route:", route)
        print("Search query:", plan.search_query)
        print("Location:", plan.location)

    except Exception as exc:

        print("ERROR:", exc)
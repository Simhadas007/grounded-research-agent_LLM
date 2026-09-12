from tools.stackexchange import search_stackexchange


queries = [
    "React useEffect StrictMode twice",
    "React 18 StrictMode useEffect",
    "useEffect duplicate API request",
    "useEffect AbortController fetch",
]


for query in queries:

    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    result = search_stackexchange(query)

    print("Success:", result["success"])
    print("Error:", result["error"])
    print("Results:", len(result["results"]))

    for i, item in enumerate(result["results"], 1):

        print("\n--- RESULT", i, "---")
        print("Site:", item["site_name"])
        print("Title:", item["title"])
        print("Score:", item["question_score"])
        print("Answer score:", item["answer_score"])
        print("URL:", item["url"])

        print("\nQuestion:")
        print(item["question_body"][:500])

        print("\nAnswer:")
        print(item["answer_body"][:800])
from tools.stackexchange import search_stackexchange


query = "React useEffect running twice"

result = search_stackexchange(query)

print("\nSuccess:", result["success"])
print("Error:", result["error"])
print("Number of results:", len(result["results"]))

for index, item in enumerate(result["results"], start=1):

    print("\n==============================")
    print(f"RESULT {index}")
    print("==============================")

    print("Title:", item["title"])
    print("URL:", item["url"])
    print("Tags:", item["tags"])
    print("Question score:", item["question_score"])
    print("Answer score:", item["answer_score"])
    print("Answered:", item["is_answered"])

    print("\nQuestion:")
    print(item["question_body"][:500])

    print("\nTop Answer:")
    print(item["answer_body"][:1000])
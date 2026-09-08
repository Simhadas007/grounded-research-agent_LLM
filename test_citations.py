from utils.citations import validate_citations


evidence = [
    {
        "title": "React useEffect runs twice",
        "url": "https://stackoverflow.com/questions/123456",
    }
]


good_answer = """
React 18 StrictMode can cause useEffect to run twice during
development.

Source:
https://stackoverflow.com/questions/123456
"""


bad_answer = """
React 18 StrictMode can cause useEffect to run twice.

Source:
https://stackoverflow.com/questions/fake-made-up-source
"""


print("GOOD CITATION:")
print(
    validate_citations(
        good_answer,
        evidence,
    )
)


print("\nFAKE CITATION:")
print(
    validate_citations(
        bad_answer,
        evidence,
    )
)
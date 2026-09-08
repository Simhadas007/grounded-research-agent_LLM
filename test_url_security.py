from utils.url_security import validate_external_url


tests = [
    "https://stackoverflow.com/questions/123/test",
    "https://api.stackexchange.com/2.3/questions",
    "https://api.open-meteo.com/v1/forecast",
    "http://stackoverflow.com/questions/123",
    "https://example.com/test",
    "https://localhost:8000/test",
    "https://127.0.0.1:8000/test",
]


for url in tests:
    result = validate_external_url(url)

    print("\nURL:", url)
    print("Allowed:", result["allowed"])
    print("Reason:", result["reason"])
from tools.weather import get_weather


locations = [
    "Chennai",
    "London",
    "Tokyo",
]


for location in locations:
    print("=" * 60)
    print("Location:", location)

    result = get_weather(location)

    print("Success:", result["success"])

    if result["success"]:
        print("Data:", result["data"])
    else:
        print("Error:", result["error"])
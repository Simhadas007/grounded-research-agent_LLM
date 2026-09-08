from utils.rate_limit import check_rate_limit


client_id = "test-user"

print("==============================")
print("RATE LIMIT TEST")
print("==============================")

for i in range(12):
    result = check_rate_limit(client_id)

    print(
        f"Request {i + 1}: "
        f"Allowed={result['allowed']} | "
        f"{result['reason']}"
    )
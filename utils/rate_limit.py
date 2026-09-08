import time
from collections import defaultdict, deque


MAX_REQUESTS = 10
WINDOW_SECONDS = 60

_requests = defaultdict(deque)


def check_rate_limit(client_id: str) -> dict:
    now = time.time()

    timestamps = _requests[client_id]

    # Remove expired requests
    while timestamps and now - timestamps[0] > WINDOW_SECONDS:
        timestamps.popleft()

    if len(timestamps) >= MAX_REQUESTS:
        return {
            "allowed": False,
            "reason": "Rate limit exceeded. Please try again later.",
        }

    timestamps.append(now)

    return {
        "allowed": True,
        "reason": "Request allowed.",
    }   
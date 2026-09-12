import requests

from utils.url_security import validate_external_url
from utils.cache import external_api_cache

from security.request_security import (
    RequestSecurityController,
    RequestSecurityPolicy,
)


# ============================================================
# Approved external services
# ============================================================

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# Security / reliability limits
# ============================================================

TIMEOUT_SECONDS = 10

MAX_LOCATION_LENGTH = 200
MAX_RESPONSE_SIZE = 2_000_000

CACHE_TTL_SECONDS = 300


# ============================================================
# Request security controller
# ============================================================

WEATHER_REQUEST_SECURITY = RequestSecurityController(
    RequestSecurityPolicy(
        max_requests=20,
        window_seconds=60,
        min_interval_seconds=1,
        max_concurrent_requests=2,
        timeout_seconds=TIMEOUT_SECONDS,
    )
)


# ============================================================
# Secure HTTP helper
# ============================================================

def safe_get(url: str, params: dict):
    """
    Perform a secure HTTPS request to an approved external API.

    Security controls:
    - HTTPS / URL validation
    - external-request rate limiting
    - minimum request interval
    - concurrency limiting
    - timeout
    - response-size limit
    - controlled User-Agent
    """

    validation = validate_external_url(url)

    if not validation["allowed"]:
        raise ValueError(
            "External weather source failed security validation."
        )

    authorization = WEATHER_REQUEST_SECURITY.authorize(
        f"open-meteo:{url}"
    )

    if not authorization["allowed"]:
        raise RuntimeError(
            "Weather request blocked by security policy: "
            + authorization["reason"]
        )

    try:
        response = requests.get(
            url,
            params=params,
            timeout=authorization["timeout_seconds"],
            headers={
                "User-Agent": "GroundedResearchAgent/1.0"
            },
        )

        response.raise_for_status()

        if len(response.content) > MAX_RESPONSE_SIZE:
            raise ValueError(
                "External weather response exceeded the allowed size."
            )

        return response

    finally:
        WEATHER_REQUEST_SECURITY.release()


# ============================================================
# Validation helpers
# ============================================================

def _valid_coordinate(
    value,
    minimum: float,
    maximum: float,
) -> bool:
    """
    Validate latitude / longitude values.
    """

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and minimum <= value <= maximum
    )


def _valid_number(value) -> bool:
    """
    Validate numeric weather values.
    """

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def _safe_string(
    value,
    maximum_length: int = 200,
) -> str:
    """
    Convert external text into a bounded string.

    External API text is treated as untrusted data.
    """

    if value is None:
        return ""

    value = str(value).strip()

    return value[:maximum_length]


# ============================================================
# Weather-code description
# ============================================================

def _weather_description(code) -> str:
    """
    Convert Open-Meteo WMO weather code into a safe
    human-readable description.

    This is deterministic presentation logic, not
    AI-based reasoning.
    """

    descriptions = {
        0: "Clear sky",

        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",

        45: "Fog",
        48: "Depositing rime fog",

        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",

        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",

        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",

        66: "Light freezing rain",
        67: "Heavy freezing rain",

        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",

        77: "Snow grains",

        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",

        85: "Slight snow showers",
        86: "Heavy snow showers",

        95: "Thunderstorm",

        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    try:
        numeric_code = int(code)
    except (TypeError, ValueError):
        return "Unknown conditions"

    return descriptions.get(
        numeric_code,
        "Unknown conditions",
    )


# ============================================================
# Weather retrieval
# ============================================================

def get_weather(location: str) -> dict:
    """
    Retrieve current weather for a location.

    The complete successful lookup is cached as ONE entry.

    This means:
        one location lookup -> one cache entry

    Security/request failures from safe_get() are intentionally
    allowed to propagate. The application layer can handle those
    failures while tests and callers can distinguish request
    security failures from normal API responses.
    """

    # ========================================================
    # 1. Validate location
    # ========================================================

    if not isinstance(location, str):
        return {
            "success": False,
            "error": "Location must be valid text.",
            "data": None,
        }

    location = location.strip()

    if not location:
        return {
            "success": False,
            "error": "Location cannot be empty.",
            "data": None,
        }

    if len(location) > MAX_LOCATION_LENGTH:
        return {
            "success": False,
            "error": "Location is too long.",
            "data": None,
        }

    # ========================================================
    # 2. Check ONE combined weather cache entry
    # ========================================================

    cache_key = (
        "open-meteo-weather-location:"
        + location.lower()
    )

    cached_result = external_api_cache.get(
        cache_key
    )

    if cached_result is not None:
        return cached_result

    # ========================================================
    # 3. Geocode location
    # ========================================================

    geo_response = safe_get(
        GEOCODING_URL,
        {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        },
    )

    geo_data = geo_response.json()

    if not isinstance(geo_data, dict):
        raise ValueError(
            "Invalid geocoding response."
        )

    # ========================================================
    # 4. Validate geocoding response
    # ========================================================

    results = geo_data.get(
        "results",
        [],
    )

    if not isinstance(results, list):
        raise ValueError(
            "Invalid geocoding result format."
        )

    if not results:
        return {
            "success": False,
            "error": (
                f"Location '{location}' "
                "was not found."
            ),
            "data": None,
        }

    place = results[0]

    if not isinstance(place, dict):
        raise ValueError(
            "Invalid location data."
        )

    # ========================================================
    # 5. Validate coordinates
    # ========================================================

    latitude = place.get(
        "latitude"
    )

    longitude = place.get(
        "longitude"
    )

    if not _valid_coordinate(
        latitude,
        -90,
        90,
    ):
        return {
            "success": False,
            "error": (
                "Invalid latitude returned "
                "by weather service."
            ),
            "data": None,
        }

    if not _valid_coordinate(
        longitude,
        -180,
        180,
    ):
        return {
            "success": False,
            "error": (
                "Invalid longitude returned "
                "by weather service."
            ),
            "data": None,
        }

    # ========================================================
    # 6. Retrieve current weather
    # ========================================================

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "weather_code,"
            "wind_speed_10m"
        ),
        "timezone": "auto",
    }

    weather_response = safe_get(
        WEATHER_URL,
        weather_params,
    )

    weather_data = weather_response.json()

    if not isinstance(weather_data, dict):
        raise ValueError(
            "Invalid weather response."
        )

    # ========================================================
    # 7. Validate weather response
    # ========================================================

    current = weather_data.get(
        "current"
    )

    if not isinstance(current, dict):
        return {
            "success": False,
            "error": (
                "Weather data was unavailable."
            ),
            "data": None,
        }

    # ========================================================
    # 8. Extract weather values
    # ========================================================

    temperature = current.get(
        "temperature_2m"
    )

    humidity = current.get(
        "relative_humidity_2m"
    )

    apparent_temperature = current.get(
        "apparent_temperature"
    )

    precipitation = current.get(
        "precipitation"
    )

    weather_code = current.get(
        "weather_code"
    )

    wind_speed = current.get(
        "wind_speed_10m"
    )

    # ========================================================
    # 9. Validate weather values
    # ========================================================

    numeric_fields = {
        "temperature_2m": temperature,
        "relative_humidity_2m": humidity,
        "apparent_temperature": apparent_temperature,
        "precipitation": precipitation,
        "weather_code": weather_code,
        "wind_speed_10m": wind_speed,
    }

    for field_name, value in numeric_fields.items():

        if not _valid_number(value):
            return {
                "success": False,
                "error": (
                    f"Weather field "
                    f"'{field_name}' "
                    "was invalid."
                ),
                "data": None,
            }

    # ========================================================
    # 10. Validate weather time / metadata
    # ========================================================

    observation_time = _safe_string(
        current.get("time"),
        100,
    )

    timezone = _safe_string(
        weather_data.get("timezone"),
        100,
    )

    # ========================================================
    # 11. Build structured grounded evidence
    # ========================================================

    data = {
        "source_type": "weather",

        "source": "Open-Meteo",

        "url": WEATHER_URL,

        "location": _safe_string(
            place.get(
                "name",
                location,
            )
        ),

        "country": _safe_string(
            place.get(
                "country",
                "",
            ),
            100,
        ),

        "latitude": latitude,

        "longitude": longitude,

        "timezone": timezone,

        "current": {
            "time": observation_time,

            "temperature_c": temperature,

            "relative_humidity_percent": humidity,

            "apparent_temperature_c": (
                apparent_temperature
            ),

            "precipitation_mm": precipitation,

            "weather_code": weather_code,

            "weather_description": (
                _weather_description(
                    weather_code
                )
            ),

            "wind_speed_kmh": wind_speed,
        },
    }

    # ========================================================
    # 12. Build final result
    # ========================================================

    result = {
        "success": True,
        "error": None,
        "data": data,
    }

    # ========================================================
    # 13. Cache ONE complete lookup
    # ========================================================

    external_api_cache.set(
        cache_key,
        result,
        CACHE_TTL_SECONDS,
    )

    return result
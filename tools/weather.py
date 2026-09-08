import requests

from utils.url_security import validate_external_url


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


# ============================================================
# HTTP helper
# ============================================================

def safe_get(url: str, params: dict):
    """
    Perform a secure HTTPS request to an approved external API.

    Security controls:
    - HTTPS only
    - Domain allowlist
    - Private/local network protection
    - Request timeout
    - Response-size limit
    """

    validation = validate_external_url(url)

    if not validation["allowed"]:
        raise ValueError(
            "External weather source failed security validation."
        )

    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT_SECONDS,
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


# ============================================================
# Validation helpers
# ============================================================

def _valid_coordinate(value, minimum: float, maximum: float) -> bool:
    """Validate latitude/longitude values."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and minimum <= value <= maximum
    )


def _safe_string(value, maximum_length: int = 200) -> str:
    """Convert external text safely to a bounded string."""

    if value is None:
        return ""

    return str(value).strip()[:maximum_length]


# ============================================================
# Weather retrieval
# ============================================================

def get_weather(location: str) -> dict:
    """
    Retrieve current weather for a location.

    The input should be a clean location such as:
        Chennai
        London
        New York

    It should NOT be the entire natural-language question.
    """

    # --------------------------------------------------------
    # 1. Validate location input
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 2. Geocode location
    # --------------------------------------------------------

    try:

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
            raise ValueError("Invalid geocoding response.")

        results = geo_data.get("results", [])

        if not isinstance(results, list):
            raise ValueError(
                "Invalid geocoding result format."
            )

        if not results:
            return {
                "success": False,
                "error": f"Location '{location}' was not found.",
                "data": None,
            }

        place = results[0]

        if not isinstance(place, dict):
            raise ValueError(
                "Invalid location data."
            )

        # ----------------------------------------------------
        # 3. Validate coordinates
        # ----------------------------------------------------

        latitude = place.get("latitude")
        longitude = place.get("longitude")

        if not _valid_coordinate(
            latitude,
            -90,
            90,
        ):
            return {
                "success": False,
                "error": "Invalid latitude returned by weather service.",
                "data": None,
            }

        if not _valid_coordinate(
            longitude,
            -180,
            180,
        ):
            return {
                "success": False,
                "error": "Invalid longitude returned by weather service.",
                "data": None,
            }

        # ----------------------------------------------------
        # 4. Retrieve current weather
        # ----------------------------------------------------

        weather_response = safe_get(
            WEATHER_URL,
            {
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
            },
        )

        weather_data = weather_response.json()

        if not isinstance(weather_data, dict):
            raise ValueError(
                "Invalid weather response."
            )

        current = weather_data.get("current")

        if not isinstance(current, dict):
            return {
                "success": False,
                "error": "Weather data was unavailable.",
                "data": None,
            }

        # ----------------------------------------------------
        # 5. Validate expected weather fields
        # ----------------------------------------------------

        temperature = current.get("temperature_2m")
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

        numeric_fields = {
            "temperature_2m": temperature,
            "relative_humidity_2m": humidity,
            "apparent_temperature": apparent_temperature,
            "precipitation": precipitation,
            "weather_code": weather_code,
            "wind_speed_10m": wind_speed,
        }

        for field_name, value in numeric_fields.items():

            if not isinstance(value, (int, float)):
                return {
                    "success": False,
                    "error": (
                        f"Weather field '{field_name}' "
                        "was invalid."
                    ),
                    "data": None,
                }

        # ----------------------------------------------------
        # 6. Build bounded, structured evidence
        # ----------------------------------------------------

        data = {
            "source": "Open-Meteo",
            "location": _safe_string(
                place.get("name", location)
            ),
            "country": _safe_string(
                place.get("country", ""),
                100,
            ),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": _safe_string(
                weather_data.get("timezone"),
                100,
            ),
            "current": {
                "time": _safe_string(
                    current.get("time"),
                    100,
                ),
                "temperature_c": temperature,
                "relative_humidity_percent": humidity,
                "apparent_temperature_c": apparent_temperature,
                "precipitation_mm": precipitation,
                "weather_code": weather_code,
                "wind_speed_kmh": wind_speed,
            },
        }

        return {
            "success": True,
            "error": None,
            "data": data,
        }

    # --------------------------------------------------------
    # 7. Safe external-service error handling
    # --------------------------------------------------------

    except requests.Timeout:

        return {
            "success": False,
            "error": "Weather service timed out.",
            "data": None,
        }

    except requests.HTTPError:

        return {
            "success": False,
            "error": "Weather service returned an HTTP error.",
            "data": None,
        }

    except requests.RequestException:

        return {
            "success": False,
            "error": "Weather service request failed.",
            "data": None,
        }

    except ValueError:

        return {
            "success": False,
            "error": "Weather service returned invalid data.",
            "data": None,
        }
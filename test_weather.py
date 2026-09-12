import pytest
from unittest.mock import patch

from tools.weather import get_weather
from utils.cache import external_api_cache


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = str(data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data

    @property
    def content(self):
        return self.text.encode("utf-8")


@pytest.fixture(autouse=True)
def clear_weather_cache():
    """
    Keep every test independent by clearing the shared API cache
    before and after each test.
    """
    external_api_cache.clear()

    yield

    external_api_cache.clear()


def test_get_weather_success():
    """Weather lookup should succeed with valid mocked API responses."""

    fake_geocoding_response = {
        "results": [
            {
                "name": "Chennai",
                "country": "India",
                "latitude": 13.0827,
                "longitude": 80.2707,
                "timezone": "Asia/Kolkata",
            }
        ]
    }

    fake_forecast_response = {
        "current": {
            "time": "2026-09-08T10:00",
            "temperature_2m": 30.5,
            "relative_humidity_2m": 70,
            "apparent_temperature": 35.2,
            "precipitation": 0.0,
            "weather_code": 1,
            "wind_speed_10m": 12.0,
        }
    }

    with patch("tools.weather.safe_get") as mock_get:
        mock_get.side_effect = [
            FakeResponse(fake_geocoding_response),
            FakeResponse(fake_forecast_response),
        ]

        result = get_weather("Chennai")

    assert result["success"] is True

    assert result["data"]["location"] == "Chennai"
    assert result["data"]["country"] == "India"

    assert result["data"]["latitude"] == 13.0827
    assert result["data"]["longitude"] == 80.2707

    assert result["data"]["current"]["temperature_c"] == 30.5
    assert result["data"]["current"]["relative_humidity_percent"] == 70
    assert result["data"]["current"]["apparent_temperature_c"] == 35.2

    assert mock_get.call_count == 2


def test_get_weather_invalid_location():
    """Empty locations should be rejected without making an API call."""

    result = get_weather("")

    assert result["success"] is False
    assert result["error"]


def test_get_weather_geocoding_failure():
    """A geocoding request failure should propagate as RuntimeError."""

    with patch("tools.weather.safe_get") as mock_get:
        mock_get.side_effect = RuntimeError("Request blocked")

        with pytest.raises(RuntimeError, match="Request blocked"):
            get_weather("Chennai")

        assert mock_get.call_count == 1


def test_get_weather_cache():
    """A successful weather response should be cached."""

    fake_geocoding_response = {
        "results": [
            {
                "name": "Chennai",
                "country": "India",
                "latitude": 13.0827,
                "longitude": 80.2707,
                "timezone": "Asia/Kolkata",
            }
        ]
    }

    fake_forecast_response = {
        "current": {
            "time": "2026-09-08T10:00",
            "temperature_2m": 30.5,
            "relative_humidity_2m": 70,
            "apparent_temperature": 35.2,
            "precipitation": 0.0,
            "weather_code": 1,
            "wind_speed_10m": 12.0,
        }
    }

    with patch("tools.weather.safe_get") as mock_get:
        mock_get.side_effect = [
            FakeResponse(fake_geocoding_response),
            FakeResponse(fake_forecast_response),
        ]

        first_result = get_weather("Chennai")
        second_result = get_weather("Chennai")

    assert first_result["success"] is True
    assert second_result["success"] is True

    assert first_result["data"]["location"] == "Chennai"
    assert second_result["data"]["location"] == "Chennai"

    # Only the first request should hit the external API.
    # The second request should come from cache.
    assert mock_get.call_count == 2

    assert external_api_cache.size() == 1


def test_get_weather_different_locations_are_not_same_cache_entry():
    """Different locations should use separate cache entries."""

    chennai_geocoding = {
        "results": [
            {
                "name": "Chennai",
                "country": "India",
                "latitude": 13.0827,
                "longitude": 80.2707,
                "timezone": "Asia/Kolkata",
            }
        ]
    }

    chennai_forecast = {
        "current": {
            "time": "2026-09-08T10:00",
            "temperature_2m": 30.5,
            "relative_humidity_2m": 70,
            "apparent_temperature": 35.2,
            "precipitation": 0.0,
            "weather_code": 1,
            "wind_speed_10m": 12.0,
        }
    }

    london_geocoding = {
        "results": [
            {
                "name": "London",
                "country": "United Kingdom",
                "latitude": 51.5074,
                "longitude": -0.1278,
                "timezone": "Europe/London",
            }
        ]
    }

    london_forecast = {
        "current": {
            "time": "2026-09-08T10:00",
            "temperature_2m": 18.0,
            "relative_humidity_2m": 65,
            "apparent_temperature": 18.5,
            "precipitation": 0.0,
            "weather_code": 2,
            "wind_speed_10m": 10.0,
        }
    }

    with patch("tools.weather.safe_get") as mock_get:
        mock_get.side_effect = [
            FakeResponse(chennai_geocoding),
            FakeResponse(chennai_forecast),
            FakeResponse(london_geocoding),
            FakeResponse(london_forecast),
        ]

        chennai_result = get_weather("Chennai")
        london_result = get_weather("London")

    assert chennai_result["success"] is True
    assert london_result["success"] is True

    assert chennai_result["data"]["location"] == "Chennai"
    assert london_result["data"]["location"] == "London"

    assert mock_get.call_count == 4
    assert external_api_cache.size() == 2
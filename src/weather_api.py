import requests

def fetch_live_weather(lat, lon):
    """Fetches raw weather metrics from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain",
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"Open-Meteo API connection failed: {response.status_code}")

    current_metrics = response.json()["current"]
    return {
        "temp": current_metrics["temperature_2m"],
        "rh": current_metrics["relative_humidity_2m"],
        "ws": current_metrics["wind_speed_10m"],
        "rain": current_metrics["rain"]
    }
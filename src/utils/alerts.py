"""AQI classification, colors/emojis, hazardous alerts, health recommendations."""
from __future__ import annotations

from src import config

_HEALTH = {
    "Good": "Air quality is satisfactory; air pollution poses little or no risk.",
    "Moderate": "Acceptable; unusually sensitive people should consider limiting prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Sensitive groups (heart/lung disease, children, elderly) should reduce prolonged outdoor exertion.",
    "Unhealthy": "Everyone may begin to experience health effects; sensitive groups may feel more serious effects.",
    "Very Unhealthy": "Health alert: everyone may experience more serious health effects. Avoid outdoor exertion.",
    "Hazardous": "Emergency conditions. Everyone should avoid all outdoor exertion. Stay indoors.",
}


def classify_aqi(aqi: float) -> dict:
    """Return level, color, emoji, health note for an AQI value."""
    aqi = float(aqi)
    for lo, hi, label, color, emoji in config.AQI_LEVELS:
        if lo <= aqi <= hi:
            return {"aqi": aqi, "level": label, "color": color,
                    "emoji": emoji, "health": _HEALTH[label]}
    # above 500 -> hazardous
    _, _, label, color, emoji = config.AQI_LEVELS[-1]
    return {"aqi": aqi, "level": label, "color": color,
            "emoji": emoji, "health": _HEALTH[label]}


def get_color(aqi: float) -> str:
    return classify_aqi(aqi)["color"]


def get_emoji(aqi: float) -> str:
    return classify_aqi(aqi)["emoji"]


def is_hazardous(aqi: float) -> bool:
    return float(aqi) >= config.ALERT_THRESHOLD


def check_alerts(forecast) -> list[dict]:
    """Scan a 72h forecast for hazardous hours; return alert records."""
    alerts = []
    for _, row in forecast.iterrows():
        aqi = float(row["predicted_aqi"])
        if is_hazardous(aqi):
            info = classify_aqi(aqi)
            alerts.append({
                "timestamp": str(row["timestamp"]),
                "aqi": aqi,
                "level": info["level"],
                "emoji": info["emoji"],
                "message": f"{info['emoji']} {info['level']} AQI {aqi:.0f} — {info['health']}",
            })
    return alerts

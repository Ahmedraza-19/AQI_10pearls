"""Flask REST API — 7 GET endpoints over the AQI forecasting service."""
from __future__ import annotations

import logging

from flask import Flask, jsonify, request

from src import config
from src.feature_pipeline.feature_store import fetch_features
from src.inference_pipeline.predict import (
    daily_summary,
    predict_current,
    predict_next_72h,
)
from src.training_pipeline.model_registry import best_model_name
from src.utils.alerts import check_alerts, classify_aqi

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)


def _err(msg, code=400):
    return jsonify({"error": msg}), code


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "aqi-predictor", "cities": config.CITY_NAMES})


@app.get("/cities")
def cities():
    return jsonify({"cities": config.CITY_NAMES})


@app.get("/current")
def current():
    city = request.args.get("city", "london")
    if city not in config.CITIES:
        return _err(f"unknown city: {city}")
    cur = predict_current(city)
    cur.update(classify_aqi(cur["aqi"]))
    return jsonify(cur)


@app.get("/predict")
def predict():
    city = request.args.get("city", "london")
    model = request.args.get("model")
    if city not in config.CITIES:
        return _err(f"unknown city: {city}")
    try:
        fc = predict_next_72h(city, model)
    except Exception as e:  # noqa: BLE001
        return _err(str(e), 503)
    return jsonify({
        "city": city,
        "model_used": fc["model_used"].iloc[0],
        "forecast": fc.assign(timestamp=fc["timestamp"].astype(str)).to_dict("records"),
        "daily_summary": daily_summary(fc).assign(date=lambda d: d["date"].astype(str)).to_dict("records"),
    })


@app.get("/history")
def history():
    city = request.args.get("city", "london")
    limit = int(request.args.get("limit", 72))
    if city not in config.CITIES:
        return _err(f"unknown city: {city}")
    df = fetch_features(city, limit=limit)
    if df is None or df.empty:
        return jsonify({"city": city, "history": []})
    cols = [c for c in ["timestamp", "aqi", "pm25", "pm10"] if c in df.columns]
    df = df[cols].assign(timestamp=df["timestamp"].astype(str))
    return jsonify({"city": city, "history": df.to_dict("records")})


@app.get("/alerts")
def alerts():
    city = request.args.get("city", "london")
    if city not in config.CITIES:
        return _err(f"unknown city: {city}")
    try:
        fc = predict_next_72h(city)
    except Exception as e:  # noqa: BLE001
        return _err(str(e), 503)
    al = check_alerts(fc)
    return jsonify({"city": city, "alert_count": len(al), "alerts": al})


@app.get("/models")
def models():
    out = {c: best_model_name(c) for c in config.CITY_NAMES}
    return jsonify({"best_models": out})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

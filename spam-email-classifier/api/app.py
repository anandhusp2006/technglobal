"""
app.py
------
Flask REST API for the Spam Email Classifier.

Endpoints:
    GET  /api/health          -> service status
    GET  /api/metrics         -> saved training metrics for all candidate models
    POST /api/predict         -> {"text": "..."}  ->  {"prediction": "spam"|"ham", "confidence": 0.0-1.0}

Run:
    python api/app.py
Then visit http://localhost:5000 for the built-in web UI.
"""

import json
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict  # noqa: E402

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health")
def health():
    model_ready = os.path.exists(os.path.join(MODEL_DIR, "spam_classifier.joblib"))
    return jsonify({"status": "ok", "model_loaded": model_ready})


@app.route("/api/metrics")
def metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(path):
        return jsonify({"error": "No metrics found. Run src/train.py first."}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400
    try:
        result = predict(text)
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run src/train.py first."}), 503
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

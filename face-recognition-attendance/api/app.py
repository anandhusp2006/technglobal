"""
api/app.py
----------
Optional REST API wrapping the attendance pipeline, so it can be integrated
into a larger system (e.g. a school's or office's own portal) rather than
only used via the Streamlit dashboard.

Endpoints:
    GET  /api/health              -> service status
    GET  /api/records             -> full attendance log as JSON (optional ?date=YYYY-MM-DD, ?person=name)
    GET  /api/roster              -> registered people
    POST /api/mark-attendance     -> multipart form file "image" -> recognizes + logs attendance

Run:
    python api/app.py
"""

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attendance import LOG_PATH, mark_attendance_from_image  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
REGISTRY_PATH = os.path.join(BASE_DIR, "data", "registry.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_classifier.joblib")

app = Flask(__name__)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)})


@app.route("/api/roster")
def roster():
    if not os.path.exists(REGISTRY_PATH):
        return jsonify({"error": "No roster found."}), 404
    with open(REGISTRY_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/records")
def records():
    if not os.path.exists(LOG_PATH):
        return jsonify([])
    df = pd.read_csv(LOG_PATH)
    if "date" in request.args:
        df = df[df["date"] == request.args["date"]]
    if "person" in request.args:
        df = df[df["name"] == request.args["person"]]
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/mark-attendance", methods=["POST"])
def mark_attendance():
    if "image" not in request.files:
        return jsonify({"error": "Send an image file under the 'image' field."}), 400

    file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image."}), 400

    try:
        results = mark_attendance_from_image(img_bgr)
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run src/train.py first."}), 503

    return jsonify({"faces_detected": len(results), "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)

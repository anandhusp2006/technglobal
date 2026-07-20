"""
attendance.py
-------------
Step 7 of the project brief: attendance marking system integrating
recognition and timestamp.

Given an input image (a webcam frame or uploaded photo), this:
  1. Detects all faces in the frame
  2. Classifies each detected face against the trained roster
  3. Logs a CSV attendance record (person, date, time) -- at most once per
     person per day, so walking past the camera repeatedly doesn't spam the log
  4. Returns annotated results for display in a GUI/dashboard
"""

import csv
import json
import os
from datetime import datetime

import cv2
import joblib
import numpy as np

from detect_and_preprocess import crop_and_normalize, detect_faces
from train import extract_embedding

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "attendance_logs")
REGISTRY_PATH = os.path.join(BASE_DIR, "data", "registry.json")
LOG_PATH = os.path.join(LOG_DIR, "attendance.csv")

CONFIDENCE_THRESHOLD = 0.4  # below this, classify as "unknown" rather than guessing

os.makedirs(LOG_DIR, exist_ok=True)

_classifier = None
_registry = None


def _load():
    global _classifier, _registry
    if _classifier is None:
        _classifier = joblib.load(os.path.join(MODEL_DIR, "face_classifier.joblib"))
    if _registry is None:
        with open(REGISTRY_PATH) as f:
            _registry = json.load(f)
    return _classifier, _registry


def _ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["person_id", "name", "date", "time", "confidence"])


def _already_marked_today(person_id, date_str):
    if not os.path.exists(LOG_PATH):
        return False
    with open(LOG_PATH) as f:
        for row in csv.DictReader(f):
            if row["person_id"] == person_id and row["date"] == date_str:
                return True
    return False


def _log_attendance(person_id, name, confidence):
    _ensure_log_header()
    now = datetime.now()
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    if _already_marked_today(person_id, date_str):
        return False, "already marked today"
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([person_id, name, date_str, time_str, round(confidence, 4)])
    return True, "marked"


def mark_attendance_from_image(image_bgr: np.ndarray) -> list:
    """
    Runs the full pipeline on a single frame/image and marks attendance for
    every recognized face. Returns a list of per-face result dicts.
    """
    classifier, registry = _load()
    boxes = detect_faces(image_bgr)
    results = []

    for box in boxes:
        face_gray = crop_and_normalize(image_bgr, box)
        embedding = extract_embedding(face_gray).reshape(1, -1)

        person_id = classifier.predict(embedding)[0]
        if hasattr(classifier, "predict_proba"):
            confidence = float(max(classifier.predict_proba(embedding)[0]))
        else:
            confidence = 1.0

        if confidence < CONFIDENCE_THRESHOLD:
            results.append({"box": list(map(int, box)), "identity": "unknown", "confidence": confidence, "logged": False})
            continue

        name = registry.get(person_id, person_id)
        logged, reason = _log_attendance(person_id, name, confidence)
        results.append({
            "box": list(map(int, box)),
            "identity": name,
            "person_id": person_id,
            "confidence": round(confidence, 4),
            "logged": logged,
            "note": reason,
        })

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python attendance.py <path_to_image>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    for r in mark_attendance_from_image(img):
        print(r)

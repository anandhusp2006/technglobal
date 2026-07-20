"""
test_robustness.py
-------------------
Step 8 of the project brief: test the system under different lighting and angles.

Takes each source raw image, synthesizes brightness variants (simulating dim/bright
lighting) and rotation variants (simulating head-angle changes) that were NOT part
of the augmented training set, then checks whether the trained classifier still
recognizes the correct identity.
"""

import os

import cv2
import joblib
import numpy as np

from detect_and_preprocess import RAW_DIR, crop_and_normalize, detect_faces
from train import extract_embedding

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")


LIGHTING_VARIANTS = {
    "very_dim": lambda img: cv2.convertScaleAbs(img, alpha=0.5, beta=-30),
    "very_bright": lambda img: cv2.convertScaleAbs(img, alpha=1.3, beta=50),
    "low_contrast": lambda img: cv2.convertScaleAbs(img, alpha=0.7, beta=20),
}

ANGLE_VARIANTS = {
    "rotate_20": lambda img: _rotate(img, 20),
    "rotate_-20": lambda img: _rotate(img, -20),
    "rotate_35": lambda img: _rotate(img, 35),
}


def _rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def classify(classifier, img_bgr):
    boxes = detect_faces(img_bgr)
    if len(boxes) == 0:
        return None, 0.0
    box = max(boxes, key=lambda b: b[2] * b[3])
    face_gray = crop_and_normalize(img_bgr, box)
    embedding = extract_embedding(face_gray).reshape(1, -1)
    pred = classifier.predict(embedding)[0]
    conf = float(max(classifier.predict_proba(embedding)[0])) if hasattr(classifier, "predict_proba") else 1.0
    return pred, conf


def main():
    classifier = joblib.load(os.path.join(MODEL_DIR, "face_classifier.joblib"))

    all_variants = {**{f"lighting:{k}": v for k, v in LIGHTING_VARIANTS.items()},
                     **{f"angle:{k}": v for k, v in ANGLE_VARIANTS.items()}}

    total, correct = 0, 0
    print(f"{'person':<12}{'variant':<20}{'predicted':<12}{'conf':<8}{'result'}")
    print("-" * 65)

    for person_id in sorted(os.listdir(RAW_DIR)):
        person_dir = os.path.join(RAW_DIR, person_id)
        if not os.path.isdir(person_dir):
            continue
        images = [f for f in sorted(os.listdir(person_dir)) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not images:
            continue
        img_path = os.path.join(person_dir, images[0])
        base_img = cv2.imread(img_path)
        if base_img is None:
            continue

        for variant_name, fn in all_variants.items():
            variant_img = fn(base_img)
            pred, conf = classify(classifier, variant_img)
            total += 1
            is_correct = pred == person_id
            correct += int(is_correct)
            print(f"{person_id:<12}{variant_name:<20}{str(pred):<12}{conf:<8.3f}{'OK' if is_correct else 'MISS'}")

    print("-" * 65)
    print(f"Robustness accuracy under lighting/angle stress: {correct}/{total} = {correct/total:.1%}" if total else "No tests run")


if __name__ == "__main__":
    main()

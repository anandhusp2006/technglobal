"""
train.py
--------
Steps 5-6 of the project brief:
  - Extract facial "embeddings" from each preprocessed face
  - Train a classifier (KNN and SVM, best one is kept) on those embeddings

EMBEDDING METHOD NOTE
The brief suggests FaceNet / Dlib for embeddings. Those require downloading large
pretrained weight files, which isn't possible in this offline environment. Instead,
this uses Local Binary Pattern (LBP) histograms -- a classic, dependency-light
texture descriptor that captures facial micro-patterns well enough for a small
closed-set attendance roster, computed with scikit-image (already installed).

To upgrade to real deep embeddings later:
  pip install face_recognition dlib
  # then replace extract_embedding() with face_recognition.face_encodings(img)[0]
  # (128-d embedding) and retrain -- the rest of the pipeline (KNN/SVM training,
  # attendance marking) is unchanged.
"""

import json
import os

import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_faces")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
GRID = 4  # split the face into a GRID x GRID blocks and histogram each block


def extract_embedding(face_gray: np.ndarray) -> np.ndarray:
    """
    Computes a block-wise Local Binary Pattern histogram "embedding" for a
    preprocessed (128x128 grayscale) face. Splitting into a grid preserves
    coarse spatial layout (eyes/nose/mouth regions), which a single global
    histogram would lose.
    """
    lbp = local_binary_pattern(face_gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    h, w = lbp.shape
    cell_h, cell_w = h // GRID, w // GRID
    n_bins = LBP_POINTS + 2

    features = []
    for i in range(GRID):
        for j in range(GRID):
            cell = lbp[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            hist, _ = np.histogram(cell, bins=n_bins, range=(0, n_bins), density=True)
            features.append(hist)
    return np.concatenate(features)


def load_dataset():
    X, y = [], []
    for person_id in sorted(os.listdir(PROCESSED_DIR)):
        person_dir = os.path.join(PROCESSED_DIR, person_id)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            img = cv2.imread(os.path.join(person_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(extract_embedding(img))
            y.append(person_id)
    return np.array(X), np.array(y)


def main():
    X, y = load_dataset()
    print(f"Loaded {len(X)} face embeddings across {len(set(y))} identities")
    if len(X) == 0:
        print("No processed faces found. Run detect_and_preprocess.py first.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    candidates = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "SVM": SVC(kernel="linear", probability=True),
    }

    results = []
    trained = {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        trained[name] = clf
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        print(f"\n{name}")
        print(classification_report(y_test, preds, zero_division=0))
        results.append({"model": name, "accuracy": round(acc, 4), "f1_macro": round(f1, 4)})

    best = max(results, key=lambda r: r["f1_macro"])
    best_model = trained[best["model"]]
    print(f"\nBest model: {best['model']}  {best}")

    # Refit best model on ALL data (train+test) so attendance marking has max data
    best_model.fit(X, y)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "face_classifier.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best["model"], "all_results": results}, f, indent=2)

    print(f"\nSaved classifier to {MODEL_DIR}/face_classifier.joblib")


if __name__ == "__main__":
    main()

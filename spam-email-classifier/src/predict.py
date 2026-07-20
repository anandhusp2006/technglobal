"""
predict.py
----------
Load the trained model + vectorizer and classify new, unseen messages.

Run:
    python src/predict.py "Congratulations! You've won a free prize, click here!"
"""

import os
import sys

import joblib

from preprocess import clean_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_model = None
_vectorizer = None


def _load():
    global _model, _vectorizer
    if _model is None:
        _model = joblib.load(os.path.join(MODEL_DIR, "spam_classifier.joblib"))
        _vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
    return _model, _vectorizer


def predict(text: str) -> dict:
    model, vectorizer = _load()
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    label = model.predict(X)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = round(float(max(proba)), 4)
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        confidence = round(float(1 / (1 + pow(2.718281828, -abs(score)))), 4)  # sigmoid squash

    return {"text": text, "prediction": label, "confidence": confidence}


SAMPLE_TEST_MESSAGES = [
    "Hey, are you free for a call later today?",
    "You have won a FREE cruise! Call now to claim your prize!!!",
    "Reminder: your dentist appointment is tomorrow at 10am.",
    "URGENT: verify your bank account now or it will be suspended.",
    "Can you send me the meeting notes from earlier?",
    "Cheap watches, 90% off, buy now click this link!!!",
]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(predict(text))
    else:
        print("No message passed as an argument -- running sample test messages:\n")
        for msg in SAMPLE_TEST_MESSAGES:
            result = predict(msg)
            print(f"[{result['prediction'].upper():5s}] (conf={result['confidence']})  {msg}")

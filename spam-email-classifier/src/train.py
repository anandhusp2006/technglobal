"""
train.py
--------
End-to-end training pipeline for the Spam Email Classifier.

Steps (per the project brief):
  1. Load labeled spam/ham dataset
  2. Preprocess text
  3. Vectorize with TF-IDF
  4. Train/test split
  5. Train multiple candidate models: Naive Bayes, Logistic Regression, SVM
  6. Evaluate with accuracy, precision, recall, F1-score
  7. Save the best model + vectorizer to disk for the prediction API

Run:
    python src/train.py
"""

import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocess import clean_series

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """
    Loads data/spam_dataset.csv if present (i.e. the real Kaggle/UCI dataset
    the user dropped in), otherwise falls back to the generated synthetic
    dataset so the pipeline still runs end-to-end.
    """
    real_path = os.path.join(DATA_DIR, "spam_dataset.csv")
    synthetic_path = os.path.join(DATA_DIR, "spam_dataset_synthetic.csv")

    if os.path.exists(real_path):
        print(f"Loading real dataset: {real_path}")
        df = pd.read_csv(real_path, encoding="latin-1")
    else:
        print(f"Real dataset not found, using synthetic dataset: {synthetic_path}")
        df = pd.read_csv(synthetic_path)

    # Normalize expected column names (UCI dataset sometimes uses v1/v2)
    cols = {c.lower(): c for c in df.columns}
    if "v1" in cols and "v2" in cols:
        df = df.rename(columns={cols["v1"]: "label", cols["v2"]: "text"})
    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    return df.reset_index(drop=True)


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, pos_label="spam"), 4),
        "recall": round(recall_score(y_test, preds, pos_label="spam"), 4),
        "f1_score": round(f1_score(y_test, preds, pos_label="spam"), 4),
    }
    print(f"\n{name}")
    print("-" * len(name))
    print(classification_report(y_test, preds))
    return metrics


def main():
    df = load_dataset()
    print(f"Dataset size: {len(df)} rows  |  ham: {(df.label == 'ham').sum()}  spam: {(df.label == 'spam').sum()}")

    df["clean_text"] = clean_series(df["text"])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    candidates = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (Linear)": LinearSVC(),
    }

    results = []
    trained = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        trained[name] = model
        results.append(evaluate(name, model, X_test, y_test))

    results_sorted = sorted(results, key=lambda r: r["f1_score"], reverse=True)
    best_name = results_sorted[0]["model"]
    best_model = trained[best_name]

    print("\n=== Model comparison (sorted by F1) ===")
    for r in results_sorted:
        print(r)
    print(f"\nBest model: {best_name}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "spam_classifier.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "all_results": results_sorted}, f, indent=2)

    print(f"\nSaved model + vectorizer to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

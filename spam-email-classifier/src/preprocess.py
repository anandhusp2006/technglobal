"""
preprocess.py
-------------
Text preprocessing utilities for the Spam Email Classifier.

Pipeline (per the project brief):
  1. Lowercase
  2. Remove punctuation / non-alphabetic characters
  3. Tokenize
  4. Remove stopwords
  5. Return cleaned, joined text ready for vectorization
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def _ensure_nltk_data():
    """Download required NLTK corpora if not already present (no-op if cached)."""
    for pkg, path in [("stopwords", "corpora/stopwords"), ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()
STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits, tokenize, and remove stopwords."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # strip URLs
    text = re.sub(r"\S+@\S+", " ", text)                  # strip emails
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                      # strip standalone numbers
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    return " ".join(tokens)


def clean_series(series):
    """Apply clean_text to a pandas Series of raw messages."""
    return series.astype(str).apply(clean_text)


if __name__ == "__main__":
    sample = "WINNER!! You have been selected to receive a $1000 Walmart gift card. Click here: http://bit.ly/abc"
    print("Raw:    ", sample)
    print("Cleaned:", clean_text(sample))

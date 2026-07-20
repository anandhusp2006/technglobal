# Spam Email Classifier

A minor ML project that classifies email/SMS messages as **spam** or **ham** (not spam),
built end to end: preprocessing → TF-IDF vectorization → model training/comparison →
evaluation → a Flask API → a small mobile-responsive web UI.

## Objective
Classify incoming messages as spam or not spam, so they can be automatically routed
to the right tray (inbox vs. spam folder).

## Project structure
```
spam-email-classifier/
├── data/
│   ├── generate_sample_data.py   # generates a synthetic demo dataset (see note below)
│   └── spam_dataset_synthetic.csv
├── src/
│   ├── preprocess.py             # text cleaning: lowercase, strip punctuation, tokenize, remove stopwords
│   ├── train.py                  # TF-IDF + trains/compares Naive Bayes, Logistic Regression, SVM
│   └── predict.py                # loads the saved model and classifies new messages
├── models/                       # created after training: spam_classifier.joblib, vectorizer.joblib, metrics.json
├── api/
│   └── app.py                    # Flask REST API serving the trained model
├── static/
│   └── index.html                # mobile-responsive front end that calls the API
└── requirements.txt
```

## A note on the dataset
This environment has no internet access to kaggle.com, so `data/generate_sample_data.py`
generates a small **synthetic** labeled dataset (300 rows, balanced spam/ham) in the exact
same `label,text` schema as the real dataset, purely so the whole pipeline is runnable
out of the box. Because the synthetic messages are templated, the trained model reports
near-perfect accuracy on synthetic test data — that number does **not** reflect real-world
performance and shouldn't be reported as such.

**For a real submission, swap in the actual dataset:**
1. Download one of:
   - https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
   - https://www.kaggle.com/datasets/jayashree02/spam-ham-email-dataset
2. Save it as `data/spam_dataset.csv` with columns `label,text` (rename `v1`/`v2` if needed —
   `train.py` handles that automatically).
3. Re-run `python src/train.py`. It automatically prefers `spam_dataset.csv` over the
   synthetic fallback if both exist.

## Setup
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

## Usage

**1. Generate the demo dataset (skip if you added the real one):**
```bash
cd data && python generate_sample_data.py && cd ..
```

**2. Train and evaluate:**
```bash
python src/train.py
```
This preprocesses the text, vectorizes it with TF-IDF (unigrams + bigrams, 3000 features),
trains **Naive Bayes**, **Logistic Regression**, and **Linear SVM**, evaluates each on a held-out
20% test split (accuracy, precision, recall, F1), and saves the best-performing model
to `models/`.

**3. Classify new messages from the command line:**
```bash
python src/predict.py "Congratulations! You've won a free prize, click here!"
# or, with no argument, runs a batch of sample test messages
python src/predict.py
```

**4. Run the API + web UI:**
```bash
python api/app.py
```
Then open **http://localhost:5000** in a browser — paste a message and click **Sort this
message** to see it stamped "Inbox" or "Spam Tray" with a confidence score. The page is
mobile-responsive.

### API endpoints
| Method | Path            | Description                                   |
|--------|-----------------|------------------------------------------------|
| GET    | `/api/health`   | Service + model status                         |
| GET    | `/api/metrics`  | Accuracy/precision/recall/F1 for all 3 models   |
| POST   | `/api/predict`  | `{"text": "..."}` → `{"prediction", "confidence"}` |

## Methodology summary
- **Preprocessing:** lowercase, URL/email stripping, punctuation and digit removal,
  tokenization (NLTK), stopword removal.
- **Feature extraction:** TF-IDF with unigrams + bigrams, capped at 3000 features.
- **Models compared:** Multinomial Naive Bayes, Logistic Regression, Linear SVM.
- **Evaluation:** accuracy, precision, recall, F1-score on a stratified 80/20 split;
  the model with the best F1 is auto-selected and persisted.
- **Testing:** `predict.py` includes a batch of hand-written messages that were never
  part of training, to sanity-check generalization beyond the test split.

## Tools used
Python, Pandas, NumPy, scikit-learn, NLTK, Flask.

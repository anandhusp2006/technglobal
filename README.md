# Machine Learning with Python — Projects

Two end-to-end ML projects, each with its own preprocessing, model training,
evaluation, REST API, and user interface.

| Project | Type | Folder | Summary |
|---|---|---|---|
| **Spam Email Classifier** | Minor Project | [`spam-email-classifier/`](./spam-email-classifier) | Classifies email/SMS messages as spam or ham using TF-IDF + Naive Bayes / Logistic Regression / SVM. Flask API + mobile-responsive web UI. |
| **Face Recognition Attendance System** | Major Project | [`face-recognition-attendance/`](./face-recognition-attendance) | Detects and recognizes faces to automatically mark timestamped attendance. Haar Cascade + LBP embeddings + KNN/SVM. Streamlit dashboard + Flask API. |

Each project folder has its own detailed `README.md` with full setup, usage,
methodology, and dataset notes — start there for anything project-specific.
This file is just the map between the two.

## Quick start

```bash
# Spam Email Classifier
cd spam-email-classifier
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
python src/train.py
python api/app.py            # open http://localhost:5000

# Face Recognition Attendance System
cd ../face-recognition-attendance
pip install -r requirements.txt
python src/detect_and_preprocess.py
python src/train.py
streamlit run dashboard/app.py
```

## Common guidelines followed in both projects
- **Documented code** — every module has a docstring explaining what it does and why, plus inline comments on non-obvious steps.
- **README.md** — each project folder has its own, covering setup, usage, architecture, and methodology.
- **Mobile responsiveness** — the spam classifier's web UI is responsive by design; the attendance system's Streamlit dashboard adapts to smaller screens by default.
- **API endpoints** — both projects expose a Flask REST API (`/api/...`) so the ML pipeline can be integrated into other systems, not just used through the bundled UI.

## A note on data/environment constraints
Neither project could reach Kaggle or download pretrained deep-learning weights
(FaceNet/Dlib) from this sandboxed environment, so both include a documented,
runnable substitute (a synthetic dataset for spam, and Haar Cascade + LBP
features for face recognition) plus exact instructions for swapping in the
real datasets/models referenced in the original project brief. See each
project's own README for details — this is called out clearly so results
aren't mistaken for production-grade numbers.

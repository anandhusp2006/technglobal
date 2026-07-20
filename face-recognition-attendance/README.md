# Face Recognition Attendance System

A major ML project that automates attendance marking using facial recognition:
detect a face → recognize who it is → log a timestamped attendance record —
exposed through both a Streamlit dashboard and a REST API.

## Objective
Automate attendance using facial recognition, replacing manual roll calls.

## Project structure
```
face-recognition-attendance/
├── data/
│   ├── raw_faces/<person_id>/       # source photos per registered person
│   ├── processed_faces/<person_id>/ # detected, cropped, normalized, augmented faces
│   └── registry.json                # person_id -> display name
├── src/
│   ├── detect_and_preprocess.py     # Haar Cascade detection, crop/resize/normalize, augmentation
│   ├── train.py                     # LBP "embedding" extraction + KNN/SVM classifier training
│   ├── attendance.py                # recognition + timestamped CSV logging (dedup per day)
│   └── test_robustness.py           # accuracy under simulated lighting & angle changes
├── models/                          # created after training: face_classifier.joblib, metrics.json
├── attendance_logs/
│   └── attendance.csv               # created after first successful recognition
├── api/
│   └── app.py                       # Flask REST API
├── dashboard/
│   └── app.py                       # Streamlit GUI: scan photos, view records, view roster
└── requirements.txt
```

## Important notes on substitutions made for this environment

This sandbox has no internet access to Kaggle or to pretrained deep-learning model
weight servers, so two substitutions were made to keep the **full pipeline runnable
end-to-end** while following the same architecture the brief describes:

| Brief asked for | Used instead | Why | How to upgrade |
|---|---|---|---|
| Face detection: Haar Cascade / **MTCNN** | Haar Cascade only | Ships with `opencv-contrib-python`, no download needed | `pip install mtcnn`, swap into `detect_faces()` in `detect_and_preprocess.py` |
| Embeddings: **FaceNet / Dlib** | Local Binary Pattern (LBP) histograms via scikit-image | FaceNet/Dlib need large pretrained weight downloads unavailable here | `pip install face_recognition dlib`, swap `extract_embedding()` in `train.py` to `face_recognition.face_encodings(img)[0]` |
| Dataset: Kaggle face datasets (CelebA / Faces-in-the-Wild) | 4 identities (6 photos each after augmentation) sourced from the **public MIT-licensed test images** in the `face_recognition` and OpenCV GitHub repos, with generic placeholder names | Kaggle isn't reachable from this environment | Swap in `data/raw_faces/<person_id>/` with real registered users' photos and re-run the pipeline |

**This means the accuracy numbers reported here (see `test_robustness.py` output
below) reflect a tiny 4-person demo roster, not production performance.** With a
real dataset (dozens of photos per person, varied lighting/angles/expressions) and
the FaceNet/Dlib upgrade above, accuracy would be substantially higher and more
reliable — LBP + Haar Cascade is a reasonable **lightweight, dependency-free**
baseline, not a substitute for those tools at scale.

## Setup
```bash
pip install -r requirements.txt
```

## Usage — full pipeline

**1. Add face photos** (or use the bundled demo roster in `data/raw_faces/`):
```
data/raw_faces/<person_id>/photo1.jpg, photo2.jpg, ...
```
Update `data/registry.json` to map each `person_id` to a display name.

**2. Detect, crop, normalize, and augment:**
```bash
python src/detect_and_preprocess.py
```
For each source photo: detects the largest face (Haar Cascade), crops, converts
to grayscale, resizes to 128×128, histogram-equalizes for lighting normalization,
then generates 6 augmented variants (brightness up/down, ±12° rotation, horizontal
flip) so a handful of source photos becomes a usable training set.

**3. Extract embeddings and train the classifier:**
```bash
python src/train.py
```
Extracts a block-wise LBP histogram "embedding" per face, trains both **KNN** and
**SVM** classifiers on a stratified train/test split, evaluates accuracy/F1, and
saves the better-performing model.

**4. Test under different lighting and angles:**
```bash
python src/test_robustness.py
```
Synthesizes dim/bright/low-contrast lighting and ±20°/35° rotation variants
(none seen during training) and reports whether the classifier still identifies
the right person — this is the honest measure of how the system holds up outside
ideal conditions.

**5. Mark attendance from a photo (CLI):**
```bash
python src/attendance.py path/to/photo.jpg
```

**6. Run the dashboard:**
```bash
streamlit run dashboard/app.py
```
Three tabs: **Scan & Mark** (upload a photo, run recognition, log attendance),
**Attendance Records** (filterable table + "present today" count), **Roster**
(registered people + how many training images each has).

**7. Run the REST API** (for integrating into another system):
```bash
python api/app.py
```

| Method | Path                    | Description |
|--------|-------------------------|--------------|
| GET    | `/api/health`           | Service + model status |
| GET    | `/api/roster`           | Registered people |
| GET    | `/api/records`          | Attendance log as JSON (`?date=`, `?person=` filters) |
| POST   | `/api/mark-attendance`  | multipart `image` file → recognizes faces + logs attendance |

## Architecture summary
```
Photo/frame
   │
   ▼
Haar Cascade face detection ──► crop, grayscale, resize 128x128, histogram-equalize
   │
   ▼
Block-wise LBP histogram (16 cells x 10 bins = 160-d "embedding")
   │
   ▼
KNN / SVM classifier (best of the two, by F1) ──► identity + confidence
   │
   ▼
If confidence ≥ 0.4 and not already logged today ──► append to attendance_logs/attendance.csv
```

## Privacy & ethics note
Facial recognition is biometric data. A production deployment of a system like
this should, at minimum: get informed consent from everyone enrolled, let people
opt out with a non-biometric fallback (e.g. manual sign-in), store face data and
logs securely with access controls, define a retention/deletion policy, and check
local regulations on biometric data before deploying (rules vary significantly by
country/state). This demo project is for learning the ML pipeline, not a
ready-to-deploy compliance-checked product.

## Tools used
Python, OpenCV (Haar Cascade), scikit-image (LBP), scikit-learn (KNN/SVM), NumPy,
Pandas, Streamlit, Flask.

"""
dashboard/app.py
-----------------
Step 8 of the project brief: GUI/dashboard to view attendance records.

Run:
    streamlit run dashboard/app.py

Features:
  - Upload a photo to run recognition + mark attendance live
  - View/filter the attendance log by date and person
  - See the registered roster and how many training photos each person has
"""

import json
import os
import sys
from datetime import date

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attendance import LOG_PATH, mark_attendance_from_image  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
REGISTRY_PATH = os.path.join(BASE_DIR, "data", "registry.json")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_faces")

st.set_page_config(page_title="Attendance Dashboard", page_icon="🗓️", layout="wide")

st.title("🗓️ Face Recognition Attendance Dashboard")
st.caption("Upload a photo to mark attendance, then review logged records below.")

tab_scan, tab_records, tab_roster = st.tabs(["Scan & Mark", "Attendance Records", "Roster"])

with tab_scan:
    st.subheader("Mark attendance from a photo")
    uploaded = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded image", width=400)

        if st.button("Run recognition & mark attendance", type="primary"):
            try:
                results = mark_attendance_from_image(img_bgr)
            except FileNotFoundError:
                st.error("No trained model found. Run `python src/train.py` first.")
                results = []

            if not results:
                st.warning("No faces detected in this image.")
            for r in results:
                if r["identity"] == "unknown":
                    st.warning(f"Unrecognized face (confidence {r['confidence']:.2f}) — not logged.")
                elif r["logged"]:
                    st.success(f"✅ Marked present: **{r['identity']}** (confidence {r['confidence']:.2f})")
                else:
                    st.info(f"ℹ️ {r['identity']} already marked today — skipped duplicate entry.")

with tab_records:
    st.subheader("Attendance log")
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        col1, col2 = st.columns(2)
        with col1:
            people = ["All"] + sorted(df["name"].unique().tolist())
            person_filter = st.selectbox("Filter by person", people)
        with col2:
            dates = ["All"] + sorted(df["date"].unique().tolist(), reverse=True)
            date_filter = st.selectbox("Filter by date", dates)

        filtered = df.copy()
        if person_filter != "All":
            filtered = filtered[filtered["name"] == person_filter]
        if date_filter != "All":
            filtered = filtered[filtered["date"] == date_filter]

        st.dataframe(filtered.sort_values(["date", "time"], ascending=False), use_container_width=True)
        st.caption(f"{len(filtered)} record(s) shown of {len(df)} total.")

        today_str = date.today().isoformat()
        present_today = df[df["date"] == today_str]["name"].nunique()
        st.metric("Present today", present_today)
    else:
        st.info("No attendance records yet. Mark someone present in the **Scan & Mark** tab.")

with tab_roster:
    st.subheader("Registered roster")
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        rows = []
        for person_id, name in registry.items():
            person_dir = os.path.join(PROCESSED_DIR, person_id)
            n_images = len(os.listdir(person_dir)) if os.path.isdir(person_dir) else 0
            rows.append({"person_id": person_id, "name": name, "training_images": n_images})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No roster found at data/registry.json")

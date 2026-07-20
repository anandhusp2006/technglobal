"""
detect_and_preprocess.py
-------------------------
Step 2-4 of the project brief:
  - Collect face images (data/raw_faces/<person_id>/*.jpg)
  - Detect faces with Haar Cascade
  - Preprocess: crop to the face, resize, normalize, grayscale
  - Augment the dataset (brightness jitter, rotation, horizontal flip)
    to synthesize more training variety from a small number of source photos

Writes processed 128x128 grayscale face crops to data/processed_faces/<person_id>/.

NOTE ON FACE DETECTOR CHOICE
The brief lists Haar Cascade / MTCNN for detection. This implementation uses
OpenCV's built-in Haar Cascade (haarcascade_frontalface_default.xml) since it
ships with opencv-contrib-python and needs no extra downloads. Swapping in
MTCNN (via the `mtcnn` pip package) is a drop-in replacement for `detect_faces()`
below if you want better recall on off-angle faces.
"""

import os

import cv2
import numpy as np

FACE_SIZE = (128, 128)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_detector = cv2.CascadeClassifier(CASCADE_PATH)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_faces")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_faces")


def detect_faces(image_bgr, min_size=(60, 60)):
    """Returns a list of (x, y, w, h) bounding boxes for detected faces."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # normalize lighting before detection
    faces = _detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size
    )
    return faces


def crop_and_normalize(image_bgr, box):
    """Crop to the given box, convert to grayscale, resize, and normalize to [0,255] uint8."""
    x, y, w, h = box
    face = image_bgr[y:y + h, x:x + w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def augment(face_gray):
    """Generate augmented variants: original, brighter, darker, rotated +/-12deg, flipped."""
    variants = {"orig": face_gray}

    variants["bright"] = np.clip(face_gray.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    variants["dark"] = np.clip(face_gray.astype(np.int16) - 40, 0, 255).astype(np.uint8)

    h, w = face_gray.shape
    for angle, name in [(12, "rot_pos12"), (-12, "rot_neg12")]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        variants[name] = cv2.warpAffine(face_gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    variants["flip"] = cv2.flip(face_gray, 1)

    return variants


def process_person(person_id):
    src_dir = os.path.join(RAW_DIR, person_id)
    dst_dir = os.path.join(PROCESSED_DIR, person_id)
    os.makedirs(dst_dir, exist_ok=True)

    count = 0
    for fname in sorted(os.listdir(src_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(src_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [skip] could not read {img_path}")
            continue

        faces = detect_faces(img)
        if len(faces) == 0:
            print(f"  [skip] no face detected in {fname}")
            continue

        # take the largest detected face
        box = max(faces, key=lambda b: b[2] * b[3])
        face_gray = crop_and_normalize(img, box)

        for variant_name, variant_img in augment(face_gray).items():
            out_name = f"{os.path.splitext(fname)[0]}_{variant_name}.png"
            cv2.imwrite(os.path.join(dst_dir, out_name), variant_img)
            count += 1

    print(f"{person_id}: wrote {count} processed face images")
    return count


def main():
    if not os.path.isdir(RAW_DIR):
        print(f"No raw_faces directory found at {RAW_DIR}")
        return

    total = 0
    for person_id in sorted(os.listdir(RAW_DIR)):
        person_dir = os.path.join(RAW_DIR, person_id)
        if os.path.isdir(person_dir):
            total += process_person(person_id)

    print(f"\nDone. {total} total processed face images in {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Face Data Collector - Enterprise CLI Module
Captures standardized facial training vectors from webcam stream and
registers them into the local biometric repository.
"""

import os
import sys
import time
import cv2
import numpy as np

# Ensure local core module is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.face_detector import FaceDetector
from core.dataset_manager import DatasetManager


def main():
    print("=" * 65)
    print("   ENTERPRISE BIOMETRIC DATA ACQUISITION & ENROLLMENT CLI")
    print("=" * 65)

    dataset_manager = DatasetManager()
    detector = FaceDetector()

    person_name = input("[PROMPT] Enter full name of the subject: ").strip()
    if not person_name:
        print("[ERROR] Subject name cannot be empty.")
        return

    emp_id = input("[PROMPT] Enter Employee / Subject ID (optional): ").strip() or None
    dept = input("[PROMPT] Enter Department (default: Engineering): ").strip() or "Engineering"
    target_samples = 50

    print(f"\n[INFO] Initializing video capture device...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam device. Please verify camera connections.")
        return

    print("[INFO] Camera initialized successfully.")
    print(f"[INFO] Position subject in front of camera. Capturing {target_samples} face vectors...")
    print("[INFO] Press 'q' at any time to abort capture.\n")

    face_vectors = []
    sample_crops = []
    frame_skip = 0

    try:
        while len(face_vectors) < target_samples:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            faces = detector.detect_faces(frame)

            if len(faces) > 0:
                # Pick primary (largest) face
                primary_face = faces[0]
                x, y, w, h = primary_face

                # Render preview HUD
                preview_frame = frame.copy()
                detector.draw_biometric_overlay(
                    preview_frame,
                    primary_face,
                    name=f"Capturing: {len(face_vectors)}/{target_samples}",
                    confidence=100.0 * (len(face_vectors) / target_samples),
                )

                if frame_skip % 3 == 0:
                    crop = detector.extract_face_crop(frame, primary_face)
                    vec = detector.extract_face_vector(crop)
                    face_vectors.append(vec)
                    sample_crops.append(crop)
                    
                    pct = int((len(face_vectors) / target_samples) * 100)
                    sys.stdout.write(f"\r[ACQUISITION] Progress: [{('=' * (pct // 4)).ljust(25)}] {len(face_vectors)}/{target_samples} frames ({pct}%)")
                    sys.stdout.flush()

                frame_skip += 1
                cv2.imshow("Biometric Data Collection - Press 'q' to Cancel", preview_frame)
            else:
                cv2.imshow("Biometric Data Collection - Press 'q' to Cancel", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Acquisition aborted by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(face_vectors) > 0:
        print(f"\n\n[PROCESSING] Serializing {len(face_vectors)} vectors to dataset...")
        vectors_arr = np.array(face_vectors, dtype=np.float32)

        # Generate avatar thumbnail from first crop
        avatar_b64 = None
        if len(sample_crops) > 0:
            avatar_b64 = detector.encode_image_to_base64(sample_crops[0])

        record = dataset_manager.enroll_subject(
            full_name=person_name,
            face_vectors=vectors_arr,
            employee_id=emp_id,
            department=dept,
            avatar_base64=avatar_b64,
        )

        print("[SUCCESS] Enrollment completed successfully!")
        print(f"          Subject Name : {record['full_name']}")
        print(f"          Employee ID  : {record['employee_id']}")
        print(f"          Total Samples: {record['sample_count']}")
        print(f"          Vector Dim   : {record['vector_dim']}")
        print("=" * 65)
    else:
        print("\n[WARN] No facial samples were captured.")


if __name__ == "__main__":
    main()

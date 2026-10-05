#!/usr/bin/env python3
"""
Real-Time Face Recognition & Biometric Attendance CLI
Executes live webcam video feed inference using pure KNN classification,
distance-weighted voting, and automated attendance logging.
"""

import os
import sys
import time
import cv2
import numpy as np

# Ensure local core module is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.knn_engine import KNNEngine
from core.face_detector import FaceDetector
from core.dataset_manager import DatasetManager
from core.attendance_logger import AttendanceLogger


def main():
    print("=" * 65)
    print("      ENTERPRISE REAL-TIME BIOMETRIC RECOGNITION CLI")
    print("=" * 65)

    dataset_manager = DatasetManager()
    attendance_logger = AttendanceLogger()
    detector = FaceDetector()

    # Pre-seed demo subjects if dataset is totally empty
    dataset_manager.seed_initial_demo_profiles()

    print("[INFO] Loading facial training vectors from dataset...")
    X_train, y_train, label_names = dataset_manager.assemble_training_matrix()

    if len(X_train) == 0:
        print("[ERROR] No enrolled biometric subjects found in dataset.")
        print("[HINT] Run `python face_data_collector.py` or use the web dashboard to enroll subjects.")
        return

    print(f"[INFO] Initializing KNN Classification Engine (K=5, Metric=Euclidean)...")
    knn = KNNEngine(k=5, metric="euclidean", weights="distance", unknown_threshold=3200.0)
    knn.fit(X_train, y_train, label_names)

    print(f"[INFO] Model fitted with {len(X_train)} samples across {len(label_names)} subjects.")
    for cid, name in label_names.items():
        count = int(np.sum(y_train == cid))
        print(f"       • Class {cid}: {name.ljust(22)} ({count} samples)")

    print("\n[INFO] Starting webcam video stream...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam device.")
        return

    print("[INFO] Stream active. Press 'q' to exit.\n")

    fps_tracker = []
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            current_time = time.time()
            fps = 1.0 / max(1e-5, current_time - prev_time)
            prev_time = current_time
            fps_tracker.append(fps)
            if len(fps_tracker) > 30:
                fps_tracker.pop(0)
            avg_fps = sum(fps_tracker) / len(fps_tracker)

            faces = detector.detect_faces(frame)

            for bbox in faces:
                crop = detector.extract_face_crop(frame, bbox)
                vec = detector.extract_face_vector(crop)
                
                # Predict with KNN Engine
                pred = knn.predict_one(vec)
                label = pred["label"]
                conf = pred["confidence"]
                is_unknown = pred["is_unknown"]
                dist = pred["distance"]

                # Log attendance entry
                if not is_unknown:
                    crop_b64 = detector.encode_image_to_base64(crop, quality=70)
                    log_entry = attendance_logger.log_recognition_event(
                        person_name=label,
                        confidence=conf,
                        distance=dist,
                        is_unknown=False,
                        camera_id="CLI_Camera_01",
                        snapshot_b64=crop_b64,
                    )
                    if log_entry:
                        print(f"[CHECK-IN] Verified: {label} (Conf: {conf}%, Dist: {dist}) at {log_entry['timestamp']}")
                else:
                    attendance_logger.log_recognition_event(
                        person_name="Unknown Subject",
                        confidence=conf,
                        distance=dist,
                        is_unknown=True,
                        camera_id="CLI_Camera_01",
                    )

                # Draw high-tech HUD overlay on frame
                detector.draw_biometric_overlay(
                    frame,
                    bbox,
                    name=label,
                    confidence=conf,
                    is_unknown=is_unknown,
                )

            # Global HUD bar on top
            hud_text = f"FPS: {avg_fps:.1f} | Active Algorithm: K-NN (k=5) | Enrolled Subjects: {len(label_names)} | Detected: {len(faces)}"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (15, 20, 28), -1)
            cv2.putText(frame, hud_text, (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1, cv2.LINE_AA)

            cv2.imshow("Biometric Face Recognition System - Press 'q' to Exit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Session terminated gracefully.")


if __name__ == "__main__":
    main()

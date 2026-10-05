"""
Enterprise Biometric Face Recognition Platform - REST API Server
High-throughput FastAPI application orchestrating pure K-Nearest Neighbors inference,
biometric enrollment pipelines, attendance telemetry, and dataset lifecycle management.
"""

import io
import os
import sys
import time
from typing import Dict, List, Optional, Union
import cv2
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, Field
import uvicorn

# Ensure local packages are on import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.knn_engine import KNNEngine
from core.face_detector import FaceDetector
from core.dataset_manager import DatasetManager
from core.attendance_logger import AttendanceLogger

# Initialize FastAPI App
app = FastAPI(
    title="Enterprise Biometric Face Recognition & Attendance System",
    description="Vectorized KNN facial classification, real-time telemetry, and audit reporting API.",
    version="2.0.0",
)

# CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATASET_DIR = os.path.join(BASE_DIR, "face_dataset")
DB_PATH = os.path.join(BASE_DIR, "face_registry.db")

dataset_manager = DatasetManager(dataset_dir=DATASET_DIR, db_path=DB_PATH)
attendance_logger = AttendanceLogger(db_path=DB_PATH, cooldown_seconds=45)
detector = FaceDetector()

# KNN Engine Singleton
knn_engine = KNNEngine(k=5, metric="euclidean", weights="distance", unknown_threshold=3400.0)


def reload_knn_model():
    """Reloads training vectors from dataset and re-fits the active KNN engine."""
    dataset_manager.seed_initial_demo_profiles()
    X_train, y_train, label_names = dataset_manager.assemble_training_matrix()
    knn_engine.fit(X_train, y_train, label_names)
    print(f"[KNN ENGINE] Reloaded {len(X_train)} training vectors across {len(label_names)} subjects.")


# Initial model fitting on boot
reload_knn_model()


# ---------------------------------------------------------------------------
# Pydantic Request/Response Schemas
# ---------------------------------------------------------------------------

class RecognizeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image or camera frame")
    camera_id: str = Field("Kiosk_Cam_01", description="Camera identifier")
    draw_hud: bool = Field(True, description="Whether to return annotated frame")
    auto_log: bool = Field(True, description="Whether to record attendance event")


class EnrollRequest(BaseModel):
    full_name: str = Field(..., min_length=2)
    employee_id: Optional[str] = None
    department: str = Field("Engineering")
    role: str = Field("Staff Member")
    email: Optional[str] = None
    images_base64: List[str] = Field(..., min_length=1, description="List of base64 face crops or frames")


class KNNConfigRequest(BaseModel):
    k: int = Field(5, ge=1, le=25)
    metric: str = Field("euclidean", description="euclidean, manhattan, cosine")
    weights: str = Field("distance", description="uniform, distance")
    unknown_threshold: float = Field(3400.0, ge=100.0, le=20000.0)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """System status and readiness check."""
    return {
        "status": "healthy",
        "service": "Enterprise Face Recognition Biometric API",
        "knn_fitted": knn_engine.is_fitted,
        "total_enrolled": len(dataset_manager.get_all_subjects()),
        "total_samples": len(knn_engine.X_train) if knn_engine.X_train is not None else 0,
        "timestamp": time.time(),
    }


@app.post("/api/recognize")
def recognize_frame(payload: RecognizeRequest):
    """
    Detects faces in frame, performs pure KNN biometric classification,
    and records verified check-in events.
    """
    try:
        frame_bgr = detector.decode_base64_image(payload.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    start_time = time.time()
    faces = detector.detect_faces(frame_bgr)
    results = []
    annotated_frame = frame_bgr.copy() if payload.draw_hud else None

    # Retrieve subject details lookup
    all_subjects = {s["full_name"]: s for s in dataset_manager.get_all_subjects()}

    for bbox in faces:
        x, y, w, h = bbox
        crop = detector.extract_face_crop(frame_bgr, bbox)
        vec = detector.extract_face_vector(crop)

        # Run KNN prediction
        prediction = knn_engine.predict_one(vec)
        label = prediction["label"]
        confidence = prediction["confidence"]
        is_unknown = prediction["is_unknown"]
        distance = prediction["distance"]

        subj_meta = all_subjects.get(label, {})
        emp_id = subj_meta.get("employee_id", "N/A")
        dept = subj_meta.get("department", "General")

        # Auto-log attendance
        log_entry = None
        if payload.auto_log:
            crop_b64 = detector.encode_image_to_base64(crop, quality=75)
            log_entry = attendance_logger.log_recognition_event(
                person_name=label if not is_unknown else "Unknown Subject",
                confidence=confidence,
                distance=distance,
                employee_id=emp_id if not is_unknown else None,
                department=dept if not is_unknown else None,
                is_unknown=is_unknown,
                camera_id=payload.camera_id,
                snapshot_b64=crop_b64,
            )

        if payload.draw_hud and annotated_frame is not None:
            detector.draw_biometric_overlay(
                annotated_frame,
                bbox,
                name=label,
                confidence=confidence,
                is_unknown=is_unknown,
            )

        results.append({
            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "label": label,
            "confidence": confidence,
            "distance": distance,
            "is_unknown": is_unknown,
            "employee_id": emp_id,
            "department": dept,
            "logged": log_entry is not None,
            "neighbors": prediction["neighbors"],
            "vote_distribution": prediction["vote_distribution"],
        })

    inference_ms = round((time.time() - start_time) * 1000.0, 2)
    annotated_b64 = None
    if payload.draw_hud and annotated_frame is not None:
        annotated_b64 = detector.encode_image_to_base64(annotated_frame, quality=80)

    return {
        "success": True,
        "faces_detected": len(results),
        "inference_time_ms": inference_ms,
        "results": results,
        "annotated_image": annotated_b64,
    }


@app.post("/api/enroll")
def enroll_subject(payload: EnrollRequest):
    """
    Enrolls a new subject by extracting face feature vectors from submitted image samples.
    """
    if not payload.images_base64:
        raise HTTPException(status_code=400, detail="At least one face sample image is required.")

    extracted_vectors = []
    primary_avatar_b64 = None

    for idx, b64_img in enumerate(payload.images_base64):
        try:
            img_bgr = detector.decode_base64_image(b64_img)
            # Check if image is already a cropped face or a full frame
            if img_bgr.shape[0] == 100 and img_bgr.shape[1] == 100:
                crop = img_bgr
            else:
                faces = detector.detect_faces(img_bgr)
                if faces:
                    crop = detector.extract_face_crop(img_bgr, faces[0])
                else:
                    crop = cv2.resize(img_bgr, (100, 100), interpolation=cv2.INTER_AREA)

            vec = detector.extract_face_vector(crop)
            extracted_vectors.append(vec)

            if idx == 0:
                primary_avatar_b64 = detector.encode_image_to_base64(crop, quality=85)
        except Exception as e:
            print(f"[WARN] Failed to process enrollment image index {idx}: {e}")

    if not extracted_vectors:
        raise HTTPException(status_code=400, detail="Could not extract valid facial feature vectors from provided images.")

    vectors_arr = np.array(extracted_vectors, dtype=np.float32)

    subject_record = dataset_manager.enroll_subject(
        full_name=payload.full_name,
        face_vectors=vectors_arr,
        employee_id=payload.employee_id,
        department=payload.department,
        role=payload.role,
        email=payload.email,
        avatar_base64=primary_avatar_b64,
    )

    # Re-train active KNN engine
    reload_knn_model()

    return {
        "success": True,
        "message": f"Successfully enrolled {payload.full_name}",
        "subject": subject_record,
        "samples_enrolled": len(extracted_vectors),
        "total_dataset_samples": len(knn_engine.X_train) if knn_engine.X_train is not None else 0,
    }


@app.get("/api/registry")
def get_registry():
    """Returns all enrolled biometric subjects in the database."""
    subjects = dataset_manager.get_all_subjects()
    return {
        "total": len(subjects),
        "subjects": subjects,
    }


@app.delete("/api/registry/{person_key}")
def delete_subject(person_key: str):
    """Deletes an enrolled subject and its .npy dataset file."""
    deleted = dataset_manager.delete_subject(person_key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Subject not found in registry.")

    reload_knn_model()
    return {
        "success": True,
        "message": f"Subject '{person_key}' removed successfully.",
    }


@app.get("/api/attendance")
def get_attendance_logs(
    limit: int = 100,
    status: Optional[str] = "ALL",
    department: Optional[str] = "ALL",
    search: Optional[str] = None,
):
    """Retrieves attendance events with optional filters and KPI summary."""
    logs = attendance_logger.get_logs(limit=limit, status=status, department=department, search=search)
    metrics = attendance_logger.get_summary_metrics()
    return {
        "metrics": metrics,
        "total_records": len(logs),
        "logs": logs,
    }


@app.get("/api/attendance/export")
def export_attendance_csv():
    """Generates and downloads attendance report as CSV."""
    csv_content = attendance_logger.export_csv()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="biometric_attendance_report.csv"'},
    )


@app.post("/api/attendance/clear")
def clear_attendance_logs():
    """Resets historical attendance logs."""
    attendance_logger.clear_logs()
    return {"success": True, "message": "Attendance log history cleared."}


@app.get("/api/knn/diagnostics")
def get_knn_diagnostics():
    """Returns real-time KNN classification telemetry, parameters, and class distribution."""
    diagnostics = knn_engine.get_diagnostics()

    # Calculate inter-class centroid distance matrix for visual diagnostic matrix
    distance_matrix = {}
    if knn_engine.is_fitted and knn_engine.X_train is not None and knn_engine.y_train is not None:
        unique_classes = np.unique(knn_engine.y_train)
        centroids = {}
        for c in unique_classes:
            idx = np.where(knn_engine.y_train == c)[0]
            centroids[int(c)] = np.mean(knn_engine.X_train[idx], axis=0)

        for c1 in unique_classes:
            name1 = knn_engine.label_map.get(int(c1), f"Class_{c1}")
            distance_matrix[name1] = {}
            for c2 in unique_classes:
                name2 = knn_engine.label_map.get(int(c2), f"Class_{c2}")
                d = np.sqrt(np.sum((centroids[int(c1)] - centroids[int(c2)]) ** 2))
                distance_matrix[name1][name2] = round(float(d), 1)

    diagnostics["inter_class_distance_matrix"] = distance_matrix
    return diagnostics


@app.post("/api/knn/configure")
def configure_knn(config: KNNConfigRequest):
    """Updates hyperparameter configuration for the active KNN Engine."""
    knn_engine.k = config.k
    knn_engine.metric = config.metric.lower()
    knn_engine.weights = config.weights.lower()
    knn_engine.unknown_threshold = config.unknown_threshold

    return {
        "success": True,
        "message": "KNN configuration updated successfully.",
        "config": knn_engine.get_diagnostics(),
    }


# Mount static web UI assets
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the Single Page Application UI."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Enterprise Face Recognition API Online</h1>")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

"""
Core Biometric Recognition & Classification Subsystems
"""

from .knn_engine import KNNEngine, legacy_knn, legacy_distance
from .face_detector import FaceDetector
from .dataset_manager import DatasetManager
from .attendance_logger import AttendanceLogger

__all__ = [
    "KNNEngine",
    "legacy_knn",
    "legacy_distance",
    "FaceDetector",
    "DatasetManager",
    "AttendanceLogger",
]

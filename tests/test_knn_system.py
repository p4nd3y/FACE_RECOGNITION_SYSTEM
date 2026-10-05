"""
Unit Tests for K-Nearest Neighbors Biometric Engine & Core Subsystems
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.knn_engine import KNNEngine, legacy_knn, legacy_distance
from core.dataset_manager import DatasetManager
from core.attendance_logger import AttendanceLogger


class TestKNNSystem(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Create 3 synthetic clusters in 100-dimensional space
        self.class_0 = np.random.normal(loc=10.0, scale=1.0, size=(10, 100)).astype(np.float32)
        self.class_1 = np.random.normal(loc=50.0, scale=1.0, size=(10, 100)).astype(np.float32)
        self.class_2 = np.random.normal(loc=90.0, scale=1.0, size=(10, 100)).astype(np.float32)

        self.X_train = np.concatenate([self.class_0, self.class_1, self.class_2], axis=0)
        self.y_train = np.concatenate([
            np.zeros(10, dtype=np.int32),
            np.ones(10, dtype=np.int32),
            2 * np.ones(10, dtype=np.int32)
        ])
        self.label_names = {0: "Subject Alpha", 1: "Subject Beta", 2: "Subject Gamma"}

    def test_knn_fit_and_prediction(self):
        engine = KNNEngine(k=3, metric="euclidean", weights="distance", unknown_threshold=500.0)
        engine.fit(self.X_train, self.y_train, self.label_names)

        self.assertTrue(engine.is_fitted)

        # Test sample close to Class 0
        test_sample_0 = np.random.normal(loc=10.2, scale=0.5, size=(100,)).astype(np.float32)
        pred_0 = engine.predict_one(test_sample_0)

        self.assertEqual(pred_0["label"], "Subject Alpha")
        self.assertFalse(pred_0["is_unknown"])
        self.assertGreater(pred_0["confidence"], 70.0)

        # Test sample close to Class 2
        test_sample_2 = np.random.normal(loc=89.8, scale=0.5, size=(100,)).astype(np.float32)
        pred_2 = engine.predict_one(test_sample_2)

        self.assertEqual(pred_2["label"], "Subject Gamma")
        self.assertFalse(pred_2["is_unknown"])

    def test_unknown_threshold_rejection(self):
        engine = KNNEngine(k=3, metric="euclidean", unknown_threshold=20.0)
        engine.fit(self.X_train, self.y_train, self.label_names)

        # Anomaly vector far away from all clusters
        impostor_sample = np.full((100,), 250.0, dtype=np.float32)
        pred = engine.predict_one(impostor_sample)

        self.assertTrue(pred["is_unknown"])
        self.assertEqual(pred["label"], "Unknown")

    def test_legacy_knn_compatibility(self):
        # Verify backward compatibility with original knn implementation
        train_matrix = np.concatenate([self.X_train, self.y_train.reshape(-1, 1)], axis=1)
        test_sample = np.random.normal(loc=50.1, scale=0.5, size=(100,)).astype(np.float32)

        result = legacy_knn(train_matrix, test_sample, k=3)
        self.assertEqual(int(result), 1)

    def test_distance_metrics(self):
        for metric in ["euclidean", "manhattan", "cosine"]:
            engine = KNNEngine(k=3, metric=metric, unknown_threshold=1000.0)
            engine.fit(self.X_train, self.y_train, self.label_names)
            pred = engine.predict_one(self.class_1[0])
            self.assertEqual(pred["label"], "Subject Beta")


if __name__ == "__main__":
    unittest.main()

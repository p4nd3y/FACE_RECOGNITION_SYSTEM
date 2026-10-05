"""
Vectorized K-Nearest Neighbors (KNN) Biometric Classification Engine
Enterprise implementation using pure NumPy with configurable distance metrics,
distance-weighted voting, unknown threshold calibration, and top-k diagnostics.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class KNNEngine:
    """
    High-performance K-Nearest Neighbors classifier tailored for face vector matching.
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        weights: str = "distance",
        unknown_threshold: float = 3500.0,
    ):
        """
        Initialize the KNN Engine.

        :param k: Number of nearest neighbors to consider (default: 5)
        :param metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        :param weights: Neighbor voting weight ('uniform', 'distance')
        :param unknown_threshold: Distance cutoff beyond which a face is flagged as Unknown
        """
        self.k = max(1, int(k))
        self.metric = metric.lower()
        self.weights = weights.lower()
        self.unknown_threshold = float(unknown_threshold)

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.label_map: Dict[int, str] = {}
        self.reverse_label_map: Dict[str, int] = {}
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray, label_names: Optional[Dict[int, str]] = None) -> "KNNEngine":
        """
        Fit the KNN model with training feature vectors and class labels.

        :param X: Feature matrix of shape (N, D)
        :param y: Label vector of shape (N,) containing integer class IDs
        :param label_names: Mapping from class ID to human-readable string name
        """
        if X is None or len(X) == 0:
            self.X_train = None
            self.y_train = None
            self.is_fitted = False
            return self

        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.int32)

        if label_names:
            self.label_map = {int(k): str(v) for k, v in label_names.items()}
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        else:
            unique_classes = np.unique(self.y_train)
            self.label_map = {int(c): f"Class_{c}" for c in unique_classes}
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        self.is_fitted = True
        return self

    def _compute_distances(self, x_test: np.ndarray) -> np.ndarray:
        """
        Compute vectorized distance between a test vector and all training samples.

        :param x_test: 1D feature vector of shape (D,)
        :return: 1D array of distances of shape (N,)
        """
        if self.X_train is None:
            raise ValueError("KNN model is not fitted with training data.")

        x_vec = np.asarray(x_test, dtype=np.float32).reshape(1, -1)

        if self.metric == "euclidean":
            # Vectorized Euclidean Distance: sqrt(sum((X - x)^2))
            diff = self.X_train - x_vec
            distances = np.sqrt(np.sum(diff ** 2, axis=1))
        elif self.metric == "manhattan":
            # Vectorized Manhattan Distance: sum(|X - x|)
            distances = np.sum(np.abs(self.X_train - x_vec), axis=1)
        elif self.metric == "cosine":
            # Cosine Distance: 1 - (A . B) / (||A|| * ||B||)
            dot = np.dot(self.X_train, x_vec.T).flatten()
            norm_train = np.linalg.norm(self.X_train, axis=1)
            norm_test = np.linalg.norm(x_vec)
            denominator = np.maximum(norm_train * norm_test, 1e-8)
            cosine_similarity = dot / denominator
            distances = 1.0 - cosine_similarity
        else:
            diff = self.X_train - x_vec
            distances = np.sqrt(np.sum(diff ** 2, axis=1))

        return distances

    def predict_one(self, x_test: np.ndarray) -> Dict[str, Union[str, int, float, bool, List[Dict]]]:
        """
        Classify a single face vector and return full biometric prediction diagnostics.

        :param x_test: 1D feature vector of shape (D,)
        :return: Dictionary with prediction outcome, confidence, distance, and top-k neighbors
        """
        if not self.is_fitted or self.X_train is None or len(self.X_train) == 0:
            return {
                "class_id": -1,
                "label": "Unknown",
                "confidence": 0.0,
                "distance": float("inf"),
                "is_unknown": True,
                "neighbors": [],
                "vote_distribution": {},
            }

        effective_k = min(self.k, len(self.X_train))
        distances = self._compute_distances(x_test)

        # Get indices of the k smallest distances
        nearest_indices = np.argpartition(distances, effective_k - 1)[:effective_k]
        # Sort the k nearest partition
        sorted_k_indices = nearest_indices[np.argsort(distances[nearest_indices])]

        top_distances = distances[sorted_k_indices]
        top_labels = self.y_train[sorted_k_indices]

        # Calculate neighbor weights
        if self.weights == "distance":
            # Inverse distance weighting: 1 / (d + eps)
            neighbor_weights = 1.0 / (top_distances + 1e-5)
        else:
            neighbor_weights = np.ones_like(top_distances)

        # Aggregate weighted votes per class
        class_votes: Dict[int, float] = {}
        class_counts: Dict[int, int] = {}
        for lbl, w in zip(top_labels, neighbor_weights):
            lbl_int = int(lbl)
            class_votes[lbl_int] = class_votes.get(lbl_int, 0.0) + float(w)
            class_counts[lbl_int] = class_counts.get(lbl_int, 0) + 1

        total_weight = sum(class_votes.values())
        best_class_id = max(class_votes.keys(), key=lambda c: class_votes[c])
        best_class_weight = class_votes[best_class_id]
        min_distance = float(top_distances[0])

        # Confidence Estimation Formula
        vote_ratio = best_class_weight / max(total_weight, 1e-8)
        distance_factor = max(0.0, 1.0 - (min_distance / max(self.unknown_threshold, 1.0)))
        raw_confidence = (vote_ratio * 0.6 + distance_factor * 0.4) * 100.0
        confidence_pct = round(min(99.5, max(5.0, raw_confidence)), 1)

        # Impostor / Unknown Detection check
        is_unknown = min_distance > self.unknown_threshold or (vote_ratio < 0.4 and effective_k > 2)

        predicted_label = "Unknown" if is_unknown else self.label_map.get(best_class_id, f"Subject_{best_class_id}")

        # Build diagnostic neighbor list
        neighbors_info = []
        for rank, (idx, dist, lbl) in enumerate(zip(sorted_k_indices, top_distances, top_labels), start=1):
            neighbors_info.append({
                "rank": rank,
                "sample_index": int(idx),
                "class_id": int(lbl),
                "label": self.label_map.get(int(lbl), f"Class_{lbl}"),
                "distance": round(float(dist), 2),
                "weight": round(float(1.0 / (dist + 1e-5)), 4),
            })

        vote_dist = {
            self.label_map.get(k, f"Class_{k}"): {
                "votes": class_counts[k],
                "weighted_score": round(class_votes[k], 4),
                "share_pct": round((class_votes[k] / total_weight) * 100.0, 1),
            }
            for k in class_votes
        }

        return {
            "class_id": int(best_class_id) if not is_unknown else -1,
            "label": predicted_label,
            "confidence": confidence_pct if not is_unknown else round(max(0.0, 100.0 - (min_distance / self.unknown_threshold * 100.0)), 1),
            "distance": round(min_distance, 2),
            "is_unknown": is_unknown,
            "neighbors": neighbors_info,
            "vote_distribution": vote_dist,
        }

    def predict_batch(self, X_test: np.ndarray) -> List[Dict[str, Union[str, int, float, bool, List[Dict]]]]:
        """
        Classify multiple face vectors.
        """
        return [self.predict_one(x) for x in X_test]

    def get_diagnostics(self) -> Dict:
        """
        Return metadata and configuration diagnostics for the KNN classifier.
        """
        total_samples = len(self.X_train) if self.X_train is not None else 0
        vector_dim = self.X_train.shape[1] if self.X_train is not None and len(self.X_train) > 0 else 0
        classes_info = []

        if self.is_fitted and self.y_train is not None:
            unique_classes, counts = np.unique(self.y_train, return_counts=True)
            for c, cnt in zip(unique_classes, counts):
                c_int = int(c)
                classes_info.append({
                    "class_id": c_int,
                    "name": self.label_map.get(c_int, f"Subject_{c_int}"),
                    "sample_count": int(cnt),
                })

        return {
            "k": self.k,
            "metric": self.metric,
            "weights": self.weights,
            "unknown_threshold": self.unknown_threshold,
            "is_fitted": self.is_fitted,
            "total_samples": total_samples,
            "feature_dimensions": vector_dim,
            "classes": classes_info,
            "total_classes": len(classes_info),
        }


def legacy_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Original Euclidean distance function preserved for 100% backward compatibility.
    """
    return float(np.sqrt(((v1 - v2) ** 2).sum()))


def legacy_knn(train: np.ndarray, test: np.ndarray, k: int = 5) -> Union[int, float]:
    """
    Original KNN function preserved for 100% backward compatibility with legacy scripts.
    """
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = legacy_distance(test, ix)
        dist.append((d, iy))

    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array([label for _, label in dk])
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

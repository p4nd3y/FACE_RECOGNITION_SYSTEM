"""
Enterprise Face Dataset & Biometric Registry Manager
Manages .npy dataset tensors, SQLite metadata catalog, subject enrollments,
and training set assembly for the KNN classifier.
"""

import datetime
import json
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class DatasetManager:
    """
    Manages facial vector storage, metadata catalog, and KNN training sets.
    """

    def __init__(
        self,
        dataset_dir: str = "./face_dataset",
        db_path: str = "./face_registry.db",
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.db_path = os.path.abspath(db_path)
        os.makedirs(self.dataset_dir, exist_ok=True)
        self._init_db()
        self._sync_existing_npy_files()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize the SQLite schema for biometric subject records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subjects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_key TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    employee_id TEXT,
                    department TEXT DEFAULT 'Engineering',
                    role TEXT DEFAULT 'Staff Member',
                    email TEXT,
                    sample_count INTEGER DEFAULT 0,
                    vector_dim INTEGER DEFAULT 30000,
                    avatar_base64 TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    @staticmethod
    def sanitize_key(name: str) -> str:
        """Create a filesystem-safe identifier from person name."""
        clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip().lower())
        clean = re.sub(r"_+", "_", clean).strip("_")
        return clean or "subject"

    def _sync_existing_npy_files(self):
        """
        Scan face_dataset directory and register any pre-existing .npy files into SQLite.
        """
        if not os.path.exists(self.dataset_dir):
            return

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for fname in os.listdir(self.dataset_dir):
                if fname.endswith(".npy"):
                    person_key = fname[:-4]
                    fpath = os.path.join(self.dataset_dir, fname)
                    try:
                        data = np.load(fpath)
                        sample_count = data.shape[0]
                        vector_dim = data.shape[1] if len(data.shape) > 1 else data.shape[0]

                        cursor.execute("SELECT id FROM subjects WHERE person_key = ?", (person_key,))
                        row = cursor.fetchone()
                        if not row:
                            # Generate a friendly display name
                            display_name = person_key.replace("_", " ").title()
                            cursor.execute("""
                                INSERT INTO subjects (
                                    person_key, full_name, employee_id, department, role, email,
                                    sample_count, vector_dim, avatar_base64, created_at, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                person_key,
                                display_name,
                                f"EMP-{1000 + abs(hash(person_key)) % 9000}",
                                "Biometrics Unit",
                                "Authorized Subject",
                                f"{person_key}@enterprise.local",
                                sample_count,
                                vector_dim,
                                None,
                                now,
                                now,
                            ))
                        else:
                            cursor.execute("""
                                UPDATE subjects SET sample_count = ?, vector_dim = ?, updated_at = ?
                                WHERE person_key = ?
                            """, (sample_count, vector_dim, now, person_key))
                    except Exception as e:
                        print(f"[WARN] Error parsing {fpath}: {e}")
            conn.commit()

    def get_all_subjects(self) -> List[Dict]:
        """Retrieve all enrolled subjects with metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM subjects ORDER BY full_name ASC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_subject_by_key(self, person_key: str) -> Optional[Dict]:
        """Retrieve subject details by key."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM subjects WHERE person_key = ?", (person_key,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def enroll_subject(
        self,
        full_name: str,
        face_vectors: np.ndarray,
        employee_id: Optional[str] = None,
        department: str = "Engineering",
        role: str = "Staff Member",
        email: Optional[str] = None,
        avatar_base64: Optional[str] = None,
    ) -> Dict:
        """
        Enroll a new person or append face samples to an existing person.

        :param full_name: Full display name
        :param face_vectors: Numpy array of shape (N, 30000)
        :param employee_id: Unique corporate ID
        :param department: Department / division
        :param role: Job title
        :param email: Contact email
        :param avatar_base64: Profile thumbnail
        :return: Created/updated subject record
        """
        person_key = self.sanitize_key(full_name)
        npy_path = os.path.join(self.dataset_dir, f"{person_key}.npy")

        # Convert to float32
        new_data = np.asarray(face_vectors, dtype=np.float32)
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(1, -1)

        # Merge with existing data if present
        if os.path.exists(npy_path):
            try:
                existing_data = np.load(npy_path)
                combined_data = np.concatenate([existing_data, new_data], axis=0)
            except Exception:
                combined_data = new_data
        else:
            combined_data = new_data

        np.save(npy_path, combined_data)

        sample_count = int(combined_data.shape[0])
        vector_dim = int(combined_data.shape[1])
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if not employee_id:
            employee_id = f"EMP-{1000 + abs(hash(person_key)) % 9000}"
        if not email:
            email = f"{person_key}@enterprise.local"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM subjects WHERE person_key = ?", (person_key,))
            existing = cursor.fetchone()

            if existing:
                cursor.execute("""
                    UPDATE subjects SET
                        full_name = ?, employee_id = ?, department = ?, role = ?,
                        email = ?, sample_count = ?, vector_dim = ?,
                        avatar_base64 = COALESCE(?, avatar_base64), updated_at = ?
                    WHERE person_key = ?
                """, (
                    full_name, employee_id, department, role,
                    email, sample_count, vector_dim,
                    avatar_base64, now, person_key
                ))
            else:
                cursor.execute("""
                    INSERT INTO subjects (
                        person_key, full_name, employee_id, department, role, email,
                        sample_count, vector_dim, avatar_base64, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    person_key, full_name, employee_id, department, role, email,
                    sample_count, vector_dim, avatar_base64, now, now
                ))
            conn.commit()

        return self.get_subject_by_key(person_key)

    def delete_subject(self, person_key: str) -> bool:
        """
        Delete a subject record and its corresponding .npy file.
        """
        npy_path = os.path.join(self.dataset_dir, f"{person_key}.npy")
        if os.path.exists(npy_path):
            try:
                os.remove(npy_path)
            except Exception as e:
                print(f"[WARN] Failed to delete {npy_path}: {e}")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM subjects WHERE person_key = ?", (person_key,))
            conn.commit()
            return cursor.rowcount > 0

    def assemble_training_matrix(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """
        Assemble the full training matrix X and label vector y for KNN training.

        :return: (X_matrix, y_vector, label_names_dict)
        """
        self._sync_existing_npy_files()
        face_data = []
        labels = []
        names = {}
        class_id = 0

        subjects = self.get_all_subjects()
        for subj in subjects:
            pkey = subj["person_key"]
            npy_path = os.path.join(self.dataset_dir, f"{pkey}.npy")
            if os.path.exists(npy_path):
                try:
                    data_item = np.load(npy_path)
                    if data_item.size == 0:
                        continue
                    if len(data_item.shape) == 1:
                        data_item = data_item.reshape(1, -1)
                    
                    face_data.append(data_item)
                    names[class_id] = subj["full_name"]
                    target = class_id * np.ones((data_item.shape[0],), dtype=np.int32)
                    labels.append(target)
                    class_id += 1
                except Exception as e:
                    print(f"[WARN] Error loading {npy_path}: {e}")

        if not face_data:
            return np.empty((0, 30000), dtype=np.float32), np.empty((0,), dtype=np.int32), {}

        X_train = np.concatenate(face_data, axis=0).astype(np.float32)
        y_train = np.concatenate(labels, axis=0).astype(np.int32)
        return X_train, y_train, names

    def seed_initial_demo_profiles(self):
        """
        Pre-seed realistic enterprise demo subjects if registry is completely empty.
        Generates realistic facial vector clusters with slight Gaussian variance.
        """
        subjects = self.get_all_subjects()
        if len(subjects) > 0:
            return

        demo_profiles = [
            {
                "full_name": "Dr. Alan Turing",
                "department": "Cryptographic Intelligence",
                "role": "Chief Research Scientist",
                "employee_id": "EMP-1912",
                "base_val": 110.0,
            },
            {
                "full_name": "Ada Lovelace",
                "department": "Algorithmic Architecture",
                "role": "Principal Software Engineer",
                "employee_id": "EMP-1843",
                "base_val": 140.0,
            },
            {
                "full_name": "Grace Hopper",
                "department": "Compiler Systems",
                "role": "Systems Architect",
                "employee_id": "EMP-1906",
                "base_val": 165.0,
            },
            {
                "full_name": "Nikola Tesla",
                "department": "Power & Signal Engineering",
                "role": "Senior Hardware Lead",
                "employee_id": "EMP-1856",
                "base_val": 90.0,
            },
        ]

        np.random.seed(42)
        for p in demo_profiles:
            # Generate 35 realistic face sample vectors centered around unique base signature
            mean_vector = np.full((30000,), p["base_val"], dtype=np.float32)
            # Add facial gradient structure
            gradient = np.tile(np.linspace(0, 50, 100, dtype=np.float32), 300)
            mean_vector = (mean_vector + gradient) % 255.0

            # 35 noisy sample vectors
            samples = []
            for _ in range(35):
                noise = np.random.normal(0, 12.0, size=(30000,)).astype(np.float32)
                vec = np.clip(mean_vector + noise, 0, 255)
                samples.append(vec)

            samples_arr = np.array(samples, dtype=np.float32)

            # Create synthetic avatar thumbnail
            avatar_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.circle(avatar_img, (50, 42), 22, (200, 220, 240), -1)
            cv2.ellipse(avatar_img, (50, 85), (32, 22), 0, 0, 180, (180, 200, 220), -1)
            # Initials
            initials = "".join([part[0] for part in p["full_name"].replace("Dr. ", "").split()[:2]])
            cv2.putText(avatar_img, initials, (32, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 40, 60), 2)

            _, buf = cv2.imencode(".jpg", avatar_img)
            import base64
            avatar_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

            self.enroll_subject(
                full_name=p["full_name"],
                face_vectors=samples_arr,
                employee_id=p["employee_id"],
                department=p["department"],
                role=p["role"],
                email=f"{self.sanitize_key(p['full_name'])}@enterprise.io",
                avatar_base64=avatar_b64,
            )
        print("[INFO] Pre-seeded 4 initial enterprise biometric profiles.")

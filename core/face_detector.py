"""
Enterprise Face Detection & Image Preprocessing Subsystem
Provides OpenCV Haar Cascade detection, illumination normalization,
face bounding box alignment, feature vector extraction, and HUD rendering.
"""

import base64
import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class FaceDetector:
    """
    Robust Face Detection and Extraction module.
    """

    def __init__(
        self,
        cascade_path: Optional[str] = None,
        target_size: Tuple[int, int] = (100, 100),
        scale_factor: float = 1.2,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (60, 60),
    ):
        """
        Initialize the Face Detector.

        :param cascade_path: Path to Haar Cascade XML file
        :param target_size: Standardized (width, height) for face crops (default: 100x100)
        :param scale_factor: Cascade scale factor
        :param min_neighbors: Cascade min neighbors
        :param min_size: Minimum face bounding box size
        """
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        # Resolve cascade path
        if cascade_path and os.path.exists(cascade_path):
            self.cascade_path = cascade_path
        else:
            default_locations = [
                "haarcascade_frontalface_default.xml",
                os.path.join(os.path.dirname(__file__), "..", "haarcascade_frontalface_default.xml"),
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            ]
            self.cascade_path = None
            for loc in default_locations:
                if os.path.exists(loc):
                    self.cascade_path = loc
                    break

        if not self.cascade_path or not os.path.exists(self.cascade_path):
            raise FileNotFoundError("haarcascade_frontalface_default.xml could not be located.")

        self.classifier = cv2.CascadeClassifier(self.cascade_path)

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect face bounding boxes in an image frame.

        :param frame_bgr: BGR image from OpenCV
        :return: List of bounding boxes as (x, y, w, h)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply Contrast Limited Adaptive Histogram Equalization for illumination invariance
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        faces = self.classifier.detectMultiScale(
            gray_eq,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        if len(faces) == 0:
            return []

        # Return sorted by area descending (largest face first)
        faces_list = [tuple(f) for f in faces]
        faces_list.sort(key=lambda b: b[2] * b[3], reverse=True)
        return faces_list

    def extract_face_crop(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        offset_ratio: float = 0.1,
    ) -> np.ndarray:
        """
        Crop, normalize, and resize a face region.

        :param frame_bgr: Input BGR image
        :param bbox: Bounding box (x, y, w, h)
        :param offset_ratio: Margin percentage around the face
        :return: Standardized BGR face crop array of shape (target_h, target_w, 3)
        """
        x, y, w, h = bbox
        h_img, w_img = frame_bgr.shape[:2]

        offset_x = int(w * offset_ratio)
        offset_y = int(h * offset_ratio)

        x1 = max(0, x - offset_x)
        y1 = max(0, y - offset_y)
        x2 = min(w_img, x + w + offset_x)
        y2 = min(h_img, y + h + offset_y)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame_bgr[y:y+h, x:x+w]

        resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_AREA)
        return resized

    def extract_face_vector(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        """
        Flatten a face crop into a 1D feature vector of shape (30000,).
        """
        if face_crop_bgr.shape[:2] != self.target_size:
            face_crop_bgr = cv2.resize(face_crop_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        return face_crop_bgr.flatten().astype(np.float32)

    @staticmethod
    def decode_base64_image(base64_str: str) -> np.ndarray:
        """
        Decode a base64 string or data URI into an OpenCV BGR numpy array.
        """
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from base64 data.")
        return img

    @staticmethod
    def encode_image_to_base64(img_bgr: np.ndarray, format: str = "jpg", quality: int = 85) -> str:
        """
        Encode an OpenCV BGR image into a base64 data URI string.
        """
        ext = f".{format.lower()}"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if format.lower() in ("jpg", "jpeg") else []
        success, encoded_img = cv2.imencode(ext, img_bgr, params)
        if not success:
            raise ValueError("Failed to encode image to base64.")
        b64_str = base64.b64encode(encoded_img).decode("utf-8")
        mime = "image/jpeg" if format.lower() in ("jpg", "jpeg") else "image/png"
        return f"data:{mime};base64,{b64_str}"

    def draw_biometric_overlay(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        name: str,
        confidence: float,
        is_unknown: bool = False,
    ) -> np.ndarray:
        """
        Render futuristic enterprise HUD overlay with corner brackets and telemetry tag.
        """
        x, y, w, h = bbox
        color = (50, 60, 240) if is_unknown else (200, 240, 0)  # Crimson or Cyan (BGR)
        
        # Corner brackets length
        line_len = int(min(w, h) * 0.22)
        thickness = 2

        # Draw 4 corner reticles
        # Top-Left
        cv2.line(frame_bgr, (x, y), (x + line_len, y), color, thickness)
        cv2.line(frame_bgr, (x, y), (x, y + line_len), color, thickness)
        # Top-Right
        cv2.line(frame_bgr, (x + w, y), (x + w - line_len, y), color, thickness)
        cv2.line(frame_bgr, (x + w, y), (x + w, y + line_len), color, thickness)
        # Bottom-Left
        cv2.line(frame_bgr, (x, y + h), (x + line_len, y + h), color, thickness)
        cv2.line(frame_bgr, (x, y + h), (x, y + h - line_len), color, thickness)
        # Bottom-Right
        cv2.line(frame_bgr, (x + w, y + h), (x + w - line_len, y + h), color, thickness)
        cv2.line(frame_bgr, (x + w, y + h), (x + w, y + h - line_len), color, thickness)

        # Subtle bounding box
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 1)

        # Header badge with name & confidence
        label_text = f"{name.upper()} [{confidence:.1f}%]" if not is_unknown else "UNKNOWN SUBJECT"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        text_size, _ = cv2.getTextSize(label_text, font, font_scale, 1)

        # Background pill
        badge_y1 = max(0, y - text_size[1] - 12)
        badge_y2 = y
        cv2.rectangle(frame_bgr, (x, badge_y1), (x + text_size[0] + 16, badge_y2), (20, 25, 35), -1)
        cv2.rectangle(frame_bgr, (x, badge_y1), (x + text_size[0] + 16, badge_y2), color, 1)

        cv2.putText(
            frame_bgr,
            label_text,
            (x + 8, y - 6),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame_bgr

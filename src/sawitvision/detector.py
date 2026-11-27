"""Palm crown detection models: adaptive morphological detector and deep learning interface."""

import os
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np


class BaseDetector:
    """Abstract interface for crown detectors."""

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class AdaptiveCrownDetector(BaseDetector):
    """High-resolution aerial palm crown detector based on spectral contrast and local maxima.
    
    Provides accurate baseline localization and confidence scores on aerial orthomosaics.
    """

    def __init__(
        self,
        min_distance_px: int = 24,
        crown_radius_px: int = 18,
        veg_hue_range: tuple = (25, 95),
    ):
        self.min_distance_px = min_distance_px
        self.crown_radius_px = crown_radius_px
        self.veg_hue_range = veg_hue_range

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Failed to load image at: {image}")
        else:
            img = image.copy()

        h, w = img.shape[:2]

        # 1. Vegetation mask using HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min, h_max = self.veg_hue_range
        veg_mask = cv2.inRange(
            hsv, np.array([h_min, 30, 30]), np.array([h_max, 255, 255])
        )

        # 2. Excess Green Index (ExG = 2G - R - B)
        b, g, r = cv2.split(img.astype(np.float32))
        exg = 2.0 * g - r - b
        smoothed = cv2.GaussianBlur(exg, (15, 15), 3.0)

        # 3. Spatial dilation filter to find local spectral maxima
        kernel_size = 2 * self.min_distance_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        local_max = cv2.dilate(smoothed, kernel)

        # Vegetation background filtering
        valid_pixels = smoothed[veg_mask > 0]
        if len(valid_pixels) == 0:
            return []

        base_threshold = np.percentile(valid_pixels, 30)
        max_val = np.max(smoothed)
        val_range = max(1e-5, max_val - base_threshold)

        # Candidate peaks
        peak_mask = (
            (smoothed == local_max)
            & (veg_mask > 0)
            & (smoothed > base_threshold)
        )
        peak_coords = np.argwhere(peak_mask)

        detections: List[Dict[str, Any]] = []
        r_box = self.crown_radius_px

        for y_center, x_center in peak_coords:
            score = float((smoothed[y_center, x_center] - base_threshold) / val_range)
            score = float(np.clip(score, 0.0, 1.0))

            if score < conf_threshold:
                continue

            x1 = max(0, x_center - r_box)
            y1 = max(0, y_center - r_box)
            x2 = min(w, x_center + r_box)
            y2 = min(h, y_center + r_box)
            bw = x2 - x1
            bh = y2 - y1

            detections.append(
                {
                    "center_x": float(x_center),
                    "center_y": float(y_center),
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(bw),
                    "height": float(bh),
                    "confidence": score,
                    "class_name": "oil_palm",
                }
            )

        return detections


class YOLOCrownDetector(BaseDetector):
    """Deep learning detector wrapper for trained PyTorch YOLO models."""

    def __init__(self, model_path: str = "runs/sawit/weights/best.pt"):
        self.model_path = model_path
        self.model = None

    def load(self):
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        if self.model is None:
            self.load()

        results = self.model(image, conf=conf_threshold, verbose=False)
        detections: List[Dict[str, Any]] = []

        for r in results:
            boxes = r.boxes
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy
                bw = x2 - x1
                bh = y2 - y1
                cx = x1 + bw / 2.0
                cy = y1 + bh / 2.0

                detections.append(
                    {
                        "center_x": float(cx),
                        "center_y": float(cy),
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(bw),
                        "height": float(bh),
                        "confidence": conf,
                        "class_name": "oil_palm",
                    }
                )

        return detections

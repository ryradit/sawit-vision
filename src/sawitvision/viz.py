"""Visualization tools for numbered crown detection preview and density maps."""

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def sort_detections_deterministically(
    detections: List[Dict[str, Any]], band_height_px: float = 60.0
) -> List[Dict[str, Any]]:
    """Sort detections systematically top-to-bottom across horizontal bands, then left-to-right.
    
    Assigns sequential tree_id from 1 to N.
    """
    if not detections:
        return []

    # Assign each detection to a discrete row band based on vertical centroid
    def get_sort_key(d: Dict[str, Any]) -> Tuple[int, float]:
        cy = d["center_y"]
        cx = d["center_x"]
        band_idx = int(cy // band_height_px)
        return (band_idx, cx)

    sorted_dets = sorted(detections, key=get_sort_key)

    # Assign sequential 1..N IDs
    for idx, d in enumerate(sorted_dets, start=1):
        d["tree_id"] = idx

    return sorted_dets


def draw_numbered_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (0, 255, 128),  # Vibrant Green
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.45,
    thickness: int = 1,
) -> np.ndarray:
    """Draw numbered bounding boxes and badges on an image."""
    canvas = image.copy()

    for det in detections:
        x1 = int(det["x"])
        y1 = int(det["y"])
        x2 = int(det["x"] + det["width"])
        y2 = int(det["y"] + det["height"])
        tree_id = det.get("tree_id", 0)

        # Draw bounding rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)

        # Label formatting: sequential number
        label = str(tree_id)
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Badge coordinates above or inside top edge
        bx1 = max(0, x1)
        by1 = max(0, y1 - th - 4)
        bx2 = bx1 + tw + 4
        by2 = y1

        if by1 <= 0:
            by1 = y1
            by2 = y1 + th + 4

        # Background badge for contrast against varying foliage & shadow
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), box_color, 1)

        # Draw label text
        cv2.putText(
            canvas,
            label,
            (bx1 + 2, by2 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    # Draw total inventory summary overlay at top-left
    total_count = len(detections)
    summary_text = f"SawitVision Stand Inventory: {total_count} trees"
    (stw, sth), _ = cv2.getTextSize(
        summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    )
    cv2.rectangle(canvas, (10, 10), (20 + stw, 20 + sth + 10), (10, 10, 10), -1)
    cv2.rectangle(canvas, (10, 10), (20 + stw, 20 + sth + 10), box_color, 1)
    cv2.putText(
        canvas,
        summary_text,
        (15, 15 + sth + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def generate_density_heatmap(
    image_shape: Tuple[int, int],
    detections: List[Dict[str, Any]],
    kernel_size: int = 51,
) -> np.ndarray:
    """Generate a spatial density heatmap from crown centroids."""
    h, w = image_shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)

    for det in detections:
        cx = int(np.clip(det["center_x"], 0, w - 1))
        cy = int(np.clip(det["center_y"], 0, h - 1))
        density_map[cy, cx] += 1.0

    # Apply Gaussian smoothing to produce continuous density field
    smoothed = cv2.GaussianBlur(density_map, (kernel_size, kernel_size), 0)
    norm = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return heatmap

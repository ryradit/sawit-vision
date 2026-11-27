"""Sliced windowing inference and cross-tile deduplication for high-resolution aerial orthomosaics."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


def compute_iou_and_ios(
    box1: Dict[str, float], box2: Dict[str, float]
) -> Tuple[float, float]:
    """Calculate Intersection-over-Union (IoU) and Intersection-over-Smaller (IoS) between two boxes."""
    x1_a, y1_a = box1["x"], box1["y"]
    x2_a, y2_a = x1_a + box1["width"], y1_a + box1["height"]

    x1_b, y1_b = box2["x"], box2["y"]
    x2_b, y2_b = x1_b + box2["width"], y1_b + box2["height"]

    # Coordinates of intersection
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)

    inter_w = max(0.0, x2_i - x1_i)
    inter_h = max(0.0, y2_i - y1_i)
    inter_area = inter_w * inter_h

    area_a = box1["width"] * box1["height"]
    area_b = box2["width"] * box2["height"]
    union_area = area_a + area_b - inter_area

    iou = inter_area / max(1e-5, union_area)
    ios = inter_area / max(1e-5, min(area_a, area_b))

    return iou, ios


def merge_cross_tile_detections(
    detections: List[Dict[str, Any]],
    match_metric: str = "IOS",
    match_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """Deduplicate overlapping detections across tile borders using Greedy-NMM.
    
    Prevents trees lying along slice seams from being counted multiple times.
    """
    if not detections:
        return []

    # Sort descending by confidence
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 1.0), reverse=True)
    kept_detections: List[Dict[str, Any]] = []

    for cand in sorted_dets:
        discard = False
        for kept in kept_detections:
            iou, ios = compute_iou_and_ios(cand, kept)
            metric_val = ios if match_metric == "IOS" else iou

            if metric_val >= match_threshold:
                # Duplicate detection on border seam
                discard = True
                break

        if not discard:
            kept_detections.append(cand)

    return kept_detections


def sliced_window_inference(
    image: np.ndarray,
    predict_fn: Callable[[np.ndarray, float], List[Dict[str, Any]]],
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_ratio: float = 0.20,
    conf_threshold: float = 0.25,
    match_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """Run windowed sliced inference over large orthomosaics with cross-tile deduplication.
    
    Args:
        image: Full image array (H, W, C).
        predict_fn: Function taking a sub-image and conf_threshold, returning detections.
        slice_height: Height of sliding window.
        slice_width: Width of sliding window.
        overlap_ratio: Overlap percentage between adjacent windows (default 20%).
        conf_threshold: Confidence threshold for predictions.
        match_threshold: Deduplication threshold for merging border detections.
        
    Returns:
        List of merged, global-coordinate detections.
    """
    img_h, img_w = image.shape[:2]

    # If image fits cleanly inside slice dimensions, run directly
    if img_h <= slice_height and img_w <= slice_width:
        return predict_fn(image, conf_threshold)

    stride_y = int(slice_height * (1.0 - overlap_ratio))
    stride_x = int(slice_width * (1.0 - overlap_ratio))

    y_starts = list(range(0, max(1, img_h - slice_height + 1), stride_y))
    if y_starts[-1] + slice_height < img_h:
        y_starts.append(img_h - slice_height)

    x_starts = list(range(0, max(1, img_w - slice_width + 1), stride_x))
    if x_starts[-1] + slice_width < img_w:
        x_starts.append(img_w - slice_width)

    all_detections: List[Dict[str, Any]] = []

    for y_offset in y_starts:
        for x_offset in x_starts:
            tile = image[y_offset : y_offset + slice_height, x_offset : x_offset + slice_width]
            tile_dets = predict_fn(tile, conf_threshold)

            # Translate slice-local coordinates to global orthomosaic coordinates
            for d in tile_dets:
                global_det = dict(d)
                global_det["x"] = d["x"] + x_offset
                global_det["y"] = d["y"] + y_offset
                global_det["center_x"] = d["center_x"] + x_offset
                global_det["center_y"] = d["center_y"] + y_offset
                all_detections.append(global_det)

    # Perform cross-tile deduplication
    merged = merge_cross_tile_detections(
        all_detections, match_metric="IOS", match_threshold=match_threshold
    )

    return merged

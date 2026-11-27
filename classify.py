#!/usr/bin/env python3
"""Program classify: Extract detected palm tree crowns and classify canopy condition and vigor attributes."""

import argparse
import csv
import os
import sys
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from sawitvision.detector import AdaptiveCrownDetector


def classify_canopy_foliage(crop: np.ndarray) -> str:
    """Classify palm crown foliage condition based on HSV color profile."""
    if crop.size == 0:
        return "empty"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_val = float(np.mean(hsv[:, :, 2]))
    mean_sat = float(np.mean(hsv[:, :, 1]))
    mean_hue = float(np.mean(hsv[:, :, 0]))

    if mean_val < 50:
        return "shadowed"
    elif 35 <= mean_hue <= 85 and mean_sat > 40:
        return "healthy_green"
    elif mean_hue < 35 and mean_sat > 30:
        return "yellowing_stressed"
    else:
        return "mixed_canopy"


def extract_and_classify(
    image_path: str = "count.jpg",
    csv_path: str = "output_count/count_detections.csv",
    output_dir: str = "output_count/crops",
    margin_px: int = 4,
    max_crops: int = 100,
):
    if not os.path.exists(image_path):
        fallback = "data/samples/sample_plantation.jpg"
        if os.path.exists(fallback):
            image_path = fallback
        else:
            raise FileNotFoundError(f"Input image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img_h, img_w = image.shape[:2]

    detections = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                detections.append({
                    "tree_id": int(row.get("tree_id", len(detections) + 1)),
                    "box_x": float(row.get("box_x", 0)),
                    "box_y": float(row.get("box_y", 0)),
                    "box_w": float(row.get("box_w", 0)),
                    "box_h": float(row.get("box_h", 0)),
                    "confidence": float(row.get("confidence", 0.5)),
                })
    else:
        # Fallback to detector if CSV not generated yet
        print("Detections CSV not found; running automated crown detector...")
        detector = AdaptiveCrownDetector()
        raw_dets = detector.predict(image)
        for i, d in enumerate(raw_dets, start=1):
            detections.append({
                "tree_id": i,
                "box_x": d["center_x"] - d["width"] / 2,
                "box_y": d["center_y"] - d["height"] / 2,
                "box_w": d["width"],
                "box_h": d["height"],
                "confidence": d["confidence"],
            })

    os.makedirs(output_dir, exist_ok=True)
    results = []
    class_counts = {}

    for i, det in enumerate(detections[:max_crops]):
        x1 = max(0, int(det["box_x"] - margin_px))
        y1 = max(0, int(det["box_y"] - margin_px))
        x2 = min(img_w, int(det["box_x"] + det["box_w"] + margin_px))
        y2 = min(img_h, int(det["box_y"] + det["box_h"] + margin_px))

        crop = image[y1:y2, x1:x2]
        category = classify_canopy_foliage(crop)
        class_counts[category] = class_counts.get(category, 0) + 1

        crop_fname = f"tree_{det['tree_id']:04d}_{category}.jpg"
        crop_path = os.path.join(output_dir, crop_fname)
        if crop.size > 0:
            cv2.imwrite(crop_path, crop)

        results.append({
            "tree_id": det["tree_id"],
            "crop_filename": crop_fname,
            "category": category,
            "confidence": det["confidence"],
            "box_x": det["box_x"],
            "box_y": det["box_y"],
            "box_w": det["box_w"],
            "box_h": det["box_h"],
        })

    summary_csv = os.path.join(output_dir, "classification_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tree_id", "crop_filename", "category", "confidence", "box_x", "box_y", "box_w", "box_h"
        ])
        writer.writeheader()
        writer.writerows(results)

    return results, class_counts, summary_csv


def main():
    parser = argparse.ArgumentParser(
        description="Extract tree crown crops and classify canopy condition and attributes."
    )
    parser.add_argument("--input", "-i", default="count.jpg", help="Path to aerial survey image.")
    parser.add_argument("--csv", default="output_count/count_detections.csv", help="Path to detections CSV.")
    parser.add_argument("--output-dir", "-o", default="output_count/crops", help="Directory for extracted chips.")
    parser.add_argument("--margin", type=int, default=4, help="Pixel margin padding around box.")
    parser.add_argument("--max-crops", type=int, default=100, help="Max crops to process.")

    args = parser.parse_args()

    print(f"Extracting and classifying crown crops from: {args.input}")
    results, class_counts, summary_csv = extract_and_classify(
        image_path=args.input,
        csv_path=args.csv,
        output_dir=args.output_dir,
        margin_px=args.margin,
        max_crops=args.max_crops,
    )

    print("\n" + "=" * 50)
    print("CANOPY CROWN CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total Crowns Evaluated : {len(results)}")
    for cat, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = (count / max(1, len(results))) * 100.0
        print(f"  - {cat.replace('_', ' ').capitalize():<22}: {count:4d} ({pct:5.1f}%)")
    print("=" * 50)
    print(f"Saved crop metadata and chips to: {summary_csv}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract audit crops and analyze canopy attributes (FR-12, FR-15)."""

import argparse
import csv
import os
import sys
import cv2
import numpy as np


def analyze_canopy_color(crop: np.ndarray) -> str:
    """Classify canopy foliage condition based on HSV color profile."""
    if crop.size == 0:
        return "empty"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_val = np.mean(hsv[:, :, 2])
    mean_sat = np.mean(hsv[:, :, 1])
    mean_hue = np.mean(hsv[:, :, 0])

    if mean_val < 50:
        return "shadowed"
    elif mean_hue >= 35 and mean_hue <= 85 and mean_sat > 40:
        return "healthy_green"
    elif mean_hue < 35 and mean_sat > 30:
        return "yellowing_or_soil"
    else:
        return "mixed_canopy"


def main():
    parser = argparse.ArgumentParser(
        description="Extract tree crown crops and classify canopy attributes."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/samples/sample_plantation.jpg",
        help="Path to source aerial image.",
    )
    parser.add_argument(
        "--csv",
        default="runs/detections.csv",
        help="Path to detections CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="runs/crops",
        help="Directory to save extracted crops.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=4,
        help="Pixel margin padding around bounding box.",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=50,
        help="Maximum number of sample crops to extract.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.csv):
        print(f"Error: Detections CSV not found: {args.csv}")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Failed to read image: {args.input}")
        sys.exit(1)

    img_h, img_w = image.shape[:2]
    os.makedirs(args.output_dir, exist_ok=True)

    crops_metadata = []

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if count >= args.max_crops:
                break

            tree_id = int(row["tree_id"])
            x = float(row["box_x"])
            y = float(row["box_y"])
            w = float(row["box_w"])
            h = float(row["box_h"])
            conf = float(row["confidence"])

            # Add padding
            x1 = max(0, int(x - args.margin))
            y1 = max(0, int(y - args.margin))
            x2 = min(img_w, int(x + w + args.margin))
            y2 = min(img_h, int(y + h + args.margin))

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            canopy_class = analyze_canopy_color(crop)
            review_flag = "review_needed" if conf < 0.40 else "accepted"

            # Create class directory
            class_dir = os.path.join(args.output_dir, canopy_class)
            os.makedirs(class_dir, exist_ok=True)

            crop_filename = f"tree_{tree_id:04d}_conf_{int(conf * 100)}.jpg"
            crop_path = os.path.join(class_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            crops_metadata.append(
                {
                    "tree_id": tree_id,
                    "confidence": conf,
                    "canopy_class": canopy_class,
                    "review_flag": review_flag,
                    "crop_file": os.path.relpath(crop_path, args.output_dir),
                }
            )
            count += 1

    # Save summary metadata
    meta_path = os.path.join(args.output_dir, "crops_inventory.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tree_id",
                "confidence",
                "canopy_class",
                "review_flag",
                "crop_file",
            ],
        )
        writer.writeheader()
        writer.writerows(crops_metadata)

    print(f"Extracted {len(crops_metadata)} tree crown chips to: {args.output_dir}")
    print(f"Saved crop metadata inventory to: {meta_path}")


if __name__ == "__main__":
    main()

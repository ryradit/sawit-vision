#!/usr/bin/env python3
"""Run palm tree detection, deterministic numbering (1..N), and export audit artifacts."""

import argparse
import csv
import os
import sys
import cv2

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sawitvision.detector import AdaptiveCrownDetector, YOLOCrownDetector
from sawitvision.geo import export_geojson
from sawitvision.viz import (
    draw_numbered_detections,
    generate_density_heatmap,
    sort_detections_deterministically,
)


def main():
    parser = argparse.ArgumentParser(
        description="Detect and number oil palm trees from aerial imagery."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/samples/sample_plantation.jpg",
        help="Path to input aerial image (GeoTIFF/JPG/PNG).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="runs/preview_numbered.jpg",
        help="Path to save numbered preview image.",
    )
    parser.add_argument(
        "--csv",
        default="runs/detections.csv",
        help="Path to save detections roster CSV.",
    )
    parser.add_argument(
        "--geojson",
        default="runs/detections.geojson",
        help="Path to save detections GeoJSON.",
    )
    parser.add_argument(
        "--density-map",
        default="runs/density_heatmap.jpg",
        help="Path to save continuous density heatmap.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0 to 1.0).",
    )
    parser.add_argument(
        "--detector",
        choices=["adaptive", "yolo"],
        default="adaptive",
        help="Detector backend.",
    )
    parser.add_argument(
        "--model-weights",
        default="runs/sawit/weights/best.pt",
        help="Path to weights file if using YOLO detector.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Ensure output directories exist
    for out_file in [args.output, args.csv, args.geojson, args.density_map]:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    print(f"Loading image from: {args.input}")
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Unable to decode image at {args.input}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"Image dimensions: {w}x{h} px")

    # Select detector
    if args.detector == "yolo" and os.path.exists(args.model_weights):
        print(f"Using YOLO detector with weights: {args.model_weights}")
        detector = YOLOCrownDetector(model_path=args.model_weights)
    else:
        print("Using Adaptive Aerial Crown Detector baseline...")
        detector = AdaptiveCrownDetector()

    raw_detections = detector.predict(image, conf_threshold=args.conf)
    print(f"Raw detected tree crowns: {len(raw_detections)}")

    # 1. Deterministic sorting: top-to-bottom across bands, then left-to-right (FR-4)
    sorted_detections = sort_detections_deterministically(raw_detections)
    total_trees = len(sorted_detections)
    print(f"Assigned deterministic IDs 1..{total_trees}")

    # 2. Render numbered preview overlay
    annotated = draw_numbered_detections(image, sorted_detections)
    cv2.imwrite(args.output, annotated)
    print(f"Saved numbered audit preview to: {args.output}")

    # 3. Export CSV roster
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tree_id",
                "center_x",
                "center_y",
                "box_x",
                "box_y",
                "box_w",
                "box_h",
                "confidence",
                "class_name",
            ]
        )
        for d in sorted_detections:
            writer.writerow(
                [
                    d["tree_id"],
                    round(d["center_x"], 2),
                    round(d["center_y"], 2),
                    round(d["x"], 2),
                    round(d["y"], 2),
                    round(d["width"], 2),
                    round(d["height"], 2),
                    round(d["confidence"], 4),
                    d["class_name"],
                ]
            )
    print(f"Saved detection roster CSV to: {args.csv}")

    # 4. Export GeoJSON
    export_geojson(sorted_detections, args.geojson)
    print(f"Saved spatial vector GeoJSON to: {args.geojson}")

    # 5. Generate density heatmap
    heatmap = generate_density_heatmap((h, w), sorted_detections)
    cv2.imwrite(args.density_map, heatmap)
    print(f"Saved density heatmap to: {args.density_map}")

    print("\n" + "=" * 50)
    print("SawitVision Detection Summary:")
    print(f"  Total Tree Count  : {total_trees}")
    print(f"  Confidence Cutoff : {args.conf}")
    print(f"  Preview Image     : {args.output}")
    print(f"  GeoJSON Vectors   : {args.geojson}")
    print(f"  Roster CSV        : {args.csv}")
    print("=" * 50)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Program count: Detect, count, and label each palm tree in count.jpg with sequential numbered bounding boxes starting from 1."""

import argparse
import csv
import os
import sys
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from sawitvision.blocks import (
    ManagementBlock,
    assign_trees_to_blocks,
    compute_block_summary,
    detect_stand_vacancies,
)
from sawitvision.infer import sliced_window_inference


def detect_palm_trees(
    image: np.ndarray,
    min_distance_px: int = 24,
    crown_radius_px: int = 18,
    conf_threshold: float = 0.25,
):
    """Detect palm crowns using vegetation spectral contrast and local maxima."""
    h, w = image.shape[:2]

    # 1. Vegetation mask in HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    veg_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))

    # 2. Excess Green Index (ExG = 2G - R - B)
    b, g, r = cv2.split(image.astype(np.float32))
    exg = 2.0 * g - r - b
    smoothed = cv2.GaussianBlur(exg, (15, 15), 3.0)

    # 3. Spatial dilation filter to locate local maxima
    kernel_size = 2 * min_distance_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    local_max = cv2.dilate(smoothed, kernel)

    valid_pixels = smoothed[veg_mask > 0]
    if len(valid_pixels) == 0:
        return []

    base_threshold = np.percentile(valid_pixels, 30)
    max_val = np.max(smoothed)
    val_range = max(1e-5, max_val - base_threshold)

    peak_mask = (smoothed == local_max) & (veg_mask > 0) & (smoothed > base_threshold)
    peak_coords = np.argwhere(peak_mask)

    detections = []
    r_box = crown_radius_px

    for y_center, x_center in peak_coords:
        score = float((smoothed[y_center, x_center] - base_threshold) / val_range)
        score = float(np.clip(score, 0.0, 1.0))
        if score < conf_threshold:
            continue

        x1 = max(0, x_center - r_box)
        y1 = max(0, y_center - r_box)
        x2 = min(w, x_center + r_box)
        y2 = min(h, y_center + r_box)

        detections.append(
            {
                "center_x": float(x_center),
                "center_y": float(y_center),
                "x": float(x1),
                "y": float(y1),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "confidence": score,
            }
        )

    # 4. Deterministic spatial sort: top-to-bottom across row bands, then left-to-right
    band_height = 60.0
    sorted_dets = sorted(
        detections,
        key=lambda d: (int(d["center_y"] // band_height), d["center_x"]),
    )

    # Assign sequential numbers starting from 1
    for idx, d in enumerate(sorted_dets, start=1):
        d["tree_id"] = idx

    return sorted_dets


def draw_numbered_boxes(
    image: np.ndarray,
    detections: list,
    box_color=(0, 255, 128),
    text_color=(255, 255, 255),
):
    """Draw bounding boxes and sequential 1..N badges."""
    canvas = image.copy()

    for det in detections:
        x1 = int(det["x"])
        y1 = int(det["y"])
        x2 = int(det["x"] + det["width"])
        y2 = int(det["y"] + det["height"])
        num_str = str(det["tree_id"])

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 1)

        # Number badge with dark background for contrast
        (tw, th), _ = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bx1 = max(0, x1)
        by1 = max(0, y1 - th - 4)
        bx2 = bx1 + tw + 4
        by2 = y1
        if by1 <= 0:
            by1 = y1
            by2 = y1 + th + 4

        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), box_color, 1)
        cv2.putText(
            canvas,
            num_str,
            (bx1 + 2, by2 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

    # Stand inventory header
    header = f"Total Plantation Stand: {len(detections)} trees (Numbered 1 to {len(detections)})"
    (hw, hh), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(canvas, (10, 10), (20 + hw, 25 + hh), (15, 15, 15), -1)
    cv2.rectangle(canvas, (10, 10), (20 + hw, 25 + hh), box_color, 1)
    cv2.putText(
        canvas,
        header,
        (15, 15 + hh),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        text_color,
        2,
        cv2.LINE_AA,
    )

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Detect, count, and label each palm tree in count.jpg starting from 1."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="count.jpg",
        help="Path to input aerial image (default: count.jpg).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output_count/count_output.jpg",
        help="Path for numbered annotated output image (default: output_count/count_output.jpg).",
    )
    parser.add_argument(
        "--csv",
        default="output_count/count_detections.csv",
        help="Path to save detection roster CSV (default: output_count/count_detections.csv).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence cutoff (default: 0.25).",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=None,
        help="Tile window size for sliced orthomosaic inference (e.g. 640 or 1024).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.20,
        help="Overlap ratio for tiled window inference (default: 0.20).",
    )
    parser.add_argument(
        "--blocks",
        action="store_true",
        default=True,
        help="Perform estate management block aggregation and vacancy detection.",
    )
    parser.add_argument(
        "--graph",
        default="output_count/count_metrics_evaluation.png",
        help="Path to save graphical metrics evaluation dashboard (default: output_count/count_metrics_evaluation.png).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input image not found at '{args.input}'")
        sys.exit(1)

    print(f"Ingesting aerial image: {args.input}")
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not decode image '{args.input}'")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"Image resolution: {w} x {h} px")

    # Ensure output folder exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)

    # Detect and number trees
    print("Running palm tree crown detection and sequential numbering...")
    if args.tile is not None and (h > args.tile or w > args.tile):
        print(f"Applying Sliced Sliding Window Inference (Tile: {args.tile}x{args.tile}, Overlap: {int(args.overlap*100)}%)...")
        detections = sliced_window_inference(
            image,
            predict_fn=lambda img, conf: detect_palm_trees(img, conf_threshold=conf),
            slice_height=args.tile,
            slice_width=args.tile,
            overlap_ratio=args.overlap,
            conf_threshold=args.conf,
        )
        # Re-sort deterministically across full image
        detections = sorted(detections, key=lambda d: (int(d["center_y"] // 60.0), d["center_x"]))
        for idx, d in enumerate(detections, start=1):
            d["tree_id"] = idx
    else:
        detections = detect_palm_trees(image, conf_threshold=args.conf)
    total_count = len(detections)

    # Render numbered image
    annotated = draw_numbered_boxes(image, detections)
    cv2.imwrite(args.output, annotated)

    # Export CSV roster
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["tree_id", "center_x", "center_y", "box_x", "box_y", "box_w", "box_h", "confidence"]
        )
        for d in detections:
            writer.writerow(
                [
                    d["tree_id"],
                    round(d["center_x"], 1),
                    round(d["center_y"], 1),
                    round(d["x"], 1),
                    round(d["y"], 1),
                    round(d["width"], 1),
                    round(d["height"], 1),
                    round(d["confidence"], 4),
                ]
            )

    # Optional Estate Management Block Aggregation & Vacancy Detection
    block_summary_path = os.path.join(os.path.dirname(args.csv), "block_summary.csv")
    if args.blocks and total_count > 0:
        margin = 15
        y_split1, y_split2 = int(h * 0.35), int(h * 0.70)
        blocks = [
            ManagementBlock("BLK-A", "North Division", [(margin, margin), (w - margin, margin), (w - margin, y_split1 - 5), (margin, y_split1 - 5)], 140.0),
            ManagementBlock("BLK-B", "Central Division", [(margin, y_split1 + 5), (w - margin, y_split1 + 5), (w - margin, y_split2 - 5), (margin, y_split2 - 5)], 140.0),
            ManagementBlock("BLK-C", "South Division", [(margin, y_split2 + 5), (w - margin, y_split2 + 5), (w - margin, h - margin), (margin, h - margin)], 140.0),
        ]
        b_map, _ = assign_trees_to_blocks(detections, blocks)
        b_summaries = []
        for b in blocks:
            b_trees = b_map.get(b.block_id, [])
            pts = np.array([[t["center_x"], t["center_y"]] for t in b_trees], dtype=np.float32)
            vacancies = detect_stand_vacancies(pts, b.polygon_coords, gsd_m=0.08)
            b_summaries.append(compute_block_summary(b, b_trees, vacancies, gsd_m=0.08))

        with open(block_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(b_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(b_summaries)

    # Generate graphical metrics evaluation dashboard
    try:
        from scripts.evaluate_count import generate_count_evaluation_figure
        generate_count_evaluation_figure(args.csv, args.graph)
    except Exception as e:
        print(f"Notice: Could not generate evaluation graph ({e})")

    print("\n" + "=" * 55)
    print("PROGRAM COUNT - SUMMARY RESULTS")
    print(f"  Input Image         : {args.input}")
    print(f"  Total Trees Counted : {total_count}")
    print(f"  Numbering Range     : 1 to {total_count}")
    print(f"  Annotated Output    : {args.output}")
    print(f"  Roster Export CSV   : {args.csv}")
    if args.blocks and os.path.exists(block_summary_path):
        print(f"  Block Ledger CSV    : {block_summary_path}")
    if os.path.exists(args.graph):
        print(f"  Metrics Graph       : {args.graph}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run plantation management block containment, planting regularity, and stand vacancy detection."""

import argparse
import csv
import json
import os
import sys
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sawitvision.blocks import (
    ManagementBlock,
    assign_trees_to_blocks,
    calculate_planting_regularity,
    compute_block_summary,
    detect_stand_vacancies,
)
from sawitvision.detector import AdaptiveCrownDetector


def create_default_estate_blocks(img_w: int, img_h: int) -> list:
    """Create representative commercial management blocks for the surveyed parcel."""
    # Split into 3 management sectors: Block North (A), Block Central (B), Block South (C)
    margin = 15
    y_split1 = int(img_h * 0.35)
    y_split2 = int(img_h * 0.70)

    # Block A: North Parcel
    poly_a = [
        (margin, margin),
        (img_w - margin, margin),
        (img_w - margin, y_split1 - 5),
        (margin, y_split1 - 5),
    ]

    # Block B: Central Parcel
    poly_b = [
        (margin, y_split1 + 5),
        (img_w - margin, y_split1 + 5),
        (img_w - margin, y_split2 - 5),
        (margin, y_split2 - 5),
    ]

    # Block C: South Parcel
    poly_c = [
        (margin, y_split2 + 5),
        (img_w - margin, y_split2 + 5),
        (img_w - margin, img_h - margin),
        (margin, img_h - margin),
    ]

    return [
        ManagementBlock("BLK-A", "North Division", poly_a, target_density_per_ha=140.0),
        ManagementBlock("BLK-B", "Central Division", poly_b, target_density_per_ha=140.0),
        ManagementBlock("BLK-C", "South Division", poly_c, target_density_per_ha=140.0),
    ]


def render_agronomic_map(
    image: np.ndarray,
    blocks: list,
    block_tree_map: dict,
    block_vacancies_map: dict,
    block_summaries: list,
) -> np.ndarray:
    """Render high-resolution agronomic map with block boundaries, trees, and vacancy alerts."""
    canvas = image.copy()
    overlay = canvas.copy()

    # Block colors
    block_colors = {
        "BLK-A": (255, 200, 0),  # Cyan-Gold
        "BLK-B": (0, 220, 255),  # Amber
        "BLK-C": (200, 100, 255),  # Violet
    }

    # 1. Draw block boundary polygons and shaded areas
    for b in blocks:
        col = block_colors.get(b.block_id, (255, 255, 255))
        pts = b.polygon_coords.astype(np.int32)
        cv2.polylines(canvas, [pts], True, col, 3, cv2.LINE_AA)
        cv2.fillPoly(overlay, [pts], col)

    # Blend block tint
    cv2.addWeighted(overlay, 0.08, canvas, 0.92, 0, canvas)

    # 2. Draw detected trees (vibrant green dots)
    for b in blocks:
        trees = block_tree_map.get(b.block_id, [])
        for t in trees:
            cx, cy = int(t["center_x"]), int(t["center_y"])
            cv2.circle(canvas, (cx, cy), 4, (0, 255, 100), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 6, (0, 180, 50), 1, cv2.LINE_AA)

    # 3. Draw missing stand vacancies (High-visibility Red Alert Markers)
    for b in blocks:
        vacancies = block_vacancies_map.get(b.block_id, [])
        for v in vacancies:
            vx, vy = int(v["center_x"]), int(v["center_y"])
            # Red pulse ring
            cv2.circle(canvas, (vx, vy), 9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (vx, vy), 3, (0, 0, 255), -1, cv2.LINE_AA)
            # Red 'X'
            s = 6
            cv2.line(canvas, (vx - s, vy - s), (vx + s, vy + s), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (vx - s, vy + s), (vx + s, vy - s), (0, 0, 255), 2, cv2.LINE_AA)

    # 4. Render block summary cards
    for idx, s in enumerate(block_summaries):
        bid = s["block_id"]
        col = block_colors.get(bid, (255, 255, 255))
        # Place cards on left margin within each sector
        card_y = 40 + idx * 260
        card_x = 25
        card_w = 320
        card_h = 135

        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), col, 2)

        title = f"{s['name']} ({bid})"
        cv2.putText(canvas, title, (card_x + 10, card_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

        lines = [
            f"Stand Count: {s['actual_trees']} / {s['target_trees']} trees ({s['completion_rate_pct']}%)",
            f"Stand Density: {s['observed_density_ha']} t/ha (Target: {s['target_density_ha']})",
            f"Vacancies (Missing): {s['detected_vacancies']} spots",
            f"Planting Regularity: {s['regularity_score']}% (Spacing: {s['mean_spacing_m']}m)",
        ]
        for l_idx, line in enumerate(lines):
            t_col = (0, 100, 255) if "Vacancies" in line and s['detected_vacancies'] > 0 else (240, 240, 240)
            cv2.putText(
                canvas,
                line,
                (card_x + 10, card_y + 50 + l_idx * 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                t_col,
                1,
                cv2.LINE_AA,
            )

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Run block stand density & missing tree vacancy detection.")
    parser.add_argument("--input", "-i", default="count.jpg", help="Path to aerial image.")
    parser.add_argument("--csv", default="count_detections.csv", help="Path to detections CSV.")
    parser.add_argument("--output-img", default="runs/agronomic_inspection.jpg", help="Output annotated agronomic map.")
    parser.add_argument("--output-csv", default="runs/block_summary.csv", help="Output summary CSV.")
    parser.add_argument("--output-vacancies", default="runs/vacancies.geojson", help="Output vacancies GeoJSON.")
    parser.add_argument("--gsd", type=float, default=0.08, help="Ground Sampling Distance in meters/pixel (default: 0.08m).")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not decode image: {args.input}")
        sys.exit(1)

    h, w = image.shape[:2]

    # Load or run tree detections
    detections = []
    if os.path.exists(args.csv):
        print(f"Loading tree detections from: {args.csv}")
        with open(args.csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                detections.append(
                    {
                        "tree_id": int(r.get("tree_id", 0)),
                        "center_x": float(r["center_x"]),
                        "center_y": float(r["center_y"]),
                        "x": float(r["box_x"]),
                        "y": float(r["box_y"]),
                        "width": float(r["box_w"]),
                        "height": float(r["box_h"]),
                        "confidence": float(r.get("confidence", 1.0)),
                    }
                )
    else:
        print("Running detector on aerial image...")
        detector = AdaptiveCrownDetector()
        detections = detector.predict(image)

    print(f"Total detected trees in survey: {len(detections)}")

    # 1. Define management blocks
    blocks = create_default_estate_blocks(w, h)
    print(f"Configured {len(blocks)} management blocks.")

    # 2. Assign trees to blocks
    block_tree_map, unassigned = assign_trees_to_blocks(detections, blocks)

    # 3. Detect stand vacancies and compute summaries
    block_summaries = []
    block_vacancies_map = {}
    all_vacancies = []

    for b in blocks:
        trees = block_tree_map.get(b.block_id, [])
        tree_pts = np.array([[t["center_x"], t["center_y"]] for t in trees], dtype=np.float32)

        # Detect vacancies
        vacancies = detect_stand_vacancies(tree_pts, b.polygon_coords, gsd_m=args.gsd)
        for v in vacancies:
            v["block_id"] = b.block_id
        block_vacancies_map[b.block_id] = vacancies
        all_vacancies.extend(vacancies)

        # Compute summary
        summary = compute_block_summary(b, trees, vacancies, gsd_m=args.gsd)
        block_summaries.append(summary)

    # 4. Render visual agronomic inspection map
    print("Rendering high-resolution agronomic inspection map...")
    agronomic_map = render_agronomic_map(image, blocks, block_tree_map, block_vacancies_map, block_summaries)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_img)), exist_ok=True)
    cv2.imwrite(args.output_img, agronomic_map)
    print(f"Saved agronomic map to: {args.output_img}")

    # 5. Export Block Summary CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(block_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(block_summaries)
    print(f"Saved block summary ledger to: {args.output_csv}")

    # 6. Export Vacancies GeoJSON & CSV
    geojson_features = []
    for v in all_vacancies:
        geojson_features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [v["center_x"], v["center_y"]]},
                "properties": {
                    "vacancy_id": v["vacancy_id"],
                    "block_id": v["block_id"],
                    "dist_nearest_m": v["dist_nearest_m"],
                    "confidence": v["confidence"],
                },
            }
        )

    geojson_doc = {
        "type": "FeatureCollection",
        "name": "plantation_stand_vacancies",
        "features": geojson_features,
    }
    with open(args.output_vacancies, "w", encoding="utf-8") as f:
        json.dump(geojson_doc, f, indent=2)
    print(f"Saved stand vacancies GeoJSON to: {args.output_vacancies}")

    # 7. Print summary report
    print("\n" + "=" * 70)
    print("AGRONOMIC INTELLIGENCE & STAND INVENTORY REPORT")
    print(f"{'Block ID':<8} {'Division':<18} {'Area (ha)':<10} {'Stand':<8} {'Density (t/ha)':<15} {'Vacancies':<10} {'Reg. Score'}")
    print("-" * 70)
    for s in block_summaries:
        print(
            f"{s['block_id']:<8} {s['name']:<18} {s['area_ha']:<10.2f} {s['actual_trees']:<8} "
            f"{s['observed_density_ha']:<15.1f} {s['detected_vacancies']:<10} {s['regularity_score']}%"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

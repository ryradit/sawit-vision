#!/usr/bin/env python3
"""Cross-dataset and multi-condition generalization benchmark for Program Count."""

import argparse
import glob
import os
import sys
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sawitvision.blocks import calculate_planting_regularity
from sawitvision.detector import AdaptiveCrownDetector
from sawitvision.infer import sliced_window_inference
from sawitvision.viz import draw_numbered_detections, sort_detections_deterministically


def evaluate_tile(image_path: str, output_dir: str, tile_size: int = 640):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping unreadable file: {image_path}")
        return None

    h, w = img.shape[:2]
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]

    detector = AdaptiveCrownDetector()
    if h > tile_size or w > tile_size:
        raw_dets = sliced_window_inference(
            img,
            predict_fn=lambda tile, conf: detector.predict(tile, conf_threshold=conf),
            slice_height=tile_size,
            slice_width=tile_size,
            overlap_ratio=0.20,
            conf_threshold=0.25,
        )
    else:
        raw_dets = detector.predict(img, conf_threshold=0.25)

    sorted_dets = sort_detections_deterministically(raw_dets)
    total_count = len(sorted_dets)

    # Calculate spatial metrics
    if total_count >= 4:
        pts = np.array([[d["center_x"], d["center_y"]] for d in sorted_dets])
        reg = calculate_planting_regularity(pts, gsd_m=0.08)
    else:
        reg = {"mean_spacing_m": 0.0, "cv_spacing": 0.0, "clark_evans_r": 0.0, "regularity_score": 0.0}

    # Render annotated preview
    annotated = draw_numbered_detections(img, sorted_dets)
    preview_path = os.path.join(output_dir, f"{base_name}_numbered.jpg")
    cv2.imwrite(preview_path, annotated)

    # Estimate area in hectares (GSD = 0.08m/px)
    area_ha = (w * 0.08 * h * 0.08) / 10000.0
    density_ha = total_count / max(1e-4, area_ha)

    return {
        "dataset_tile": filename,
        "resolution": f"{w}x{h}",
        "trees_counted": total_count,
        "area_ha": round(area_ha, 3),
        "density_t_ha": round(density_ha, 1),
        "mean_spacing_m": reg["mean_spacing_m"],
        "spacing_cv": reg["cv_spacing"],
        "regularity_r": reg["clark_evans_r"],
        "preview_path": preview_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run cross-dataset and multi-condition benchmark.")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing benchmark tiles.")
    parser.add_argument("--output-dir", default="output_count/benchmark_comparison", help="Output directory.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_files = sorted(glob.glob(os.path.join(args.data_dir, "*.jpg")) + glob.glob(os.path.join(args.data_dir, "*.tif")))

    if not image_files:
        print(f"No benchmark images found in: {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} benchmark image(s) in {args.data_dir}/\n")
    results = []

    for img_p in image_files:
        print(f"Evaluating: {os.path.basename(img_p)}...")
        res = evaluate_tile(img_p, args.output_dir)
        if res:
            results.append(res)

    print("\n" + "=" * 88)
    print("CROSS-DATASET & MULTI-CONDITION GENERALIZATION BENCHMARK")
    print(f"{'Condition Tile':<34} {'Dim':<10} {'Stand':<8} {'Area (ha)':<10} {'Density (t/ha)':<15} {'Spacing':<8} {'Reg. R'}")
    print("-" * 88)
    for r in results:
        print(
            f"{r['dataset_tile']:<34} {r['resolution']:<10} {r['trees_counted']:<8} {r['area_ha']:<10.3f} "
            f"{r['density_t_ha']:<15.1f} {r['mean_spacing_m']:<8.2f}m {r['regularity_r']}"
        )
    print("=" * 88)
    print(f"All annotated benchmark previews saved to: {args.output_dir}/\n")


if __name__ == "__main__":
    main()

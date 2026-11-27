#!/usr/bin/env python3
"""Compute comprehensive detection, counting, and spatial metrics evaluation.

Evaluates predictions across test splits and multi-condition benchmark datasets,
generating machine-readable JSON/CSV reports and updating summary statistics.
"""

import argparse
import glob
import json
import os
import sys
import cv2
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sawitvision.blocks import calculate_planting_regularity
from sawitvision.detector import AdaptiveCrownDetector
from sawitvision.infer import sliced_window_inference
from sawitvision.metrics import compute_count_metrics
from sawitvision.viz import sort_detections_deterministically


def compute_box_iou(box1, box2):
    """Compute Intersection-over-Union between two normalized bounding boxes [cx, cy, w, h]."""
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
    return inter_area / max(1e-6, union_area)


def evaluate_split(split_file: str, conf_thresh: float = 0.25, iou_thresh: float = 0.30):
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    detector = AdaptiveCrownDetector()
    gt_counts = []
    pred_counts = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    per_image_results = []

    for img_p in image_paths:
        if not os.path.exists(img_p):
            # Resolve relative path or normalize cross-machine absolute paths
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if "data/" in img_p:
                candidate = os.path.join(repo_root, "data", img_p.split("data/")[-1])
            else:
                candidate = os.path.join(repo_root, img_p)
            if os.path.exists(candidate):
                img_p = candidate

        lbl_p = img_p.replace("/images/", "/labels/").replace(".jpg", ".txt")
        gt_boxes = []
        if os.path.exists(lbl_p):
            with open(lbl_p, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_boxes.append([float(x) for x in parts[1:5]])

        img = cv2.imread(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = detector.predict(img, conf_threshold=conf_thresh)

        pred_boxes = []
        for d in dets:
            pred_boxes.append([d["center_x"] / w, d["center_y"] / h, d["width"] / w, d["height"] / h])

        gt_counts.append(len(gt_boxes))
        pred_counts.append(len(pred_boxes))

        # Greedy bipartite matching
        matched_gt = set()
        tp = 0
        fp = 0
        for pb in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, gb in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = compute_box_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn = len(gt_boxes) - len(matched_gt)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        per_image_results.append(
            {
                "file": os.path.basename(img_p),
                "ground_truth": len(gt_boxes),
                "predicted": len(pred_boxes),
                "error": len(pred_boxes) - len(gt_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    precision = tp_total / max(1, tp_total + fp_total)
    recall = tp_total / max(1, tp_total + fn_total)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    errors = np.array(pred_counts) - np.array(gt_counts)
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors) / np.maximum(gt_counts, 1.0)) * 100.0)
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))

    if len(pred_counts) > 1 and np.std(pred_counts) > 1e-6 and np.std(gt_counts) > 1e-6:
        r2 = float(np.corrcoef(pred_counts, gt_counts)[0, 1] ** 2)
    else:
        r2 = 1.0 if np.allclose(pred_counts, gt_counts) else 0.0

    return {
        "num_images": len(image_paths),
        "total_ground_truth": int(sum(gt_counts)),
        "total_predicted": int(sum(pred_counts)),
        "detection_metrics": {
            "iou_threshold": iou_thresh,
            "true_positives": tp_total,
            "false_positives": fp_total,
            "false_negatives": fn_total,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "count_metrics": {
            "mae": round(mae, 3),
            "mape_pct": round(mape, 3),
            "rmse": round(rmse, 3),
            "bias": round(bias, 3),
            "r2_score": round(r2, 4),
            "accuracy_pct": round(100.0 - mape, 3),
        },
        "per_image": per_image_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection and counting metrics.")
    parser.add_argument("--test-split", default="data/splits/test.txt", help="Path to test split.")
    parser.add_argument("--output-json", default="output_count/metrics_evaluation.json", help="Path for JSON output.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.30, help="IoU threshold for box match.")
    args = parser.parse_args()

    print(f"Evaluating split: {args.test_split} (conf={args.conf}, iou={args.iou})...")
    report = evaluate_split(args.test_split, conf_thresh=args.conf, iou_thresh=args.iou)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("SAWITVISION QUANTITATIVE METRICS EVALUATION")
    print("=" * 65)
    print(f"Evaluated Images:       {report['num_images']}")
    print(f"Ground Truth Stand:     {report['total_ground_truth']} palms")
    print(f"Detected Stand:         {report['total_predicted']} palms")
    print("-" * 65)
    print("OBJECT DETECTION METRICS:")
    print(f"  Precision:            {report['detection_metrics']['precision'] * 100:.2f}%")
    print(f"  Recall:               {report['detection_metrics']['recall'] * 100:.2f}%")
    print(f"  F1-Score:             {report['detection_metrics']['f1_score'] * 100:.2f}%")
    print(f"  True Positives (TP):  {report['detection_metrics']['true_positives']}")
    print(f"  False Positives (FP): {report['detection_metrics']['false_positives']}")
    print(f"  False Negatives (FN): {report['detection_metrics']['false_negatives']}")
    print("-" * 65)
    print("COUNT ESTIMATION METRICS:")
    print(f"  Counting Accuracy:    {report['count_metrics']['accuracy_pct']:.2f}%")
    print(f"  MAE (Mean Abs Error): {report['count_metrics']['mae']:.2f} palms/tile")
    print(f"  MAPE:                 {report['count_metrics']['mape_pct']:.2f}%")
    print(f"  RMSE:                 {report['count_metrics']['rmse']:.2f} palms/tile")
    print(f"  Systematic Bias:      {report['count_metrics']['bias']:.2f} palms/tile")
    print(f"  R^2 (Correlation):    {report['count_metrics']['r2_score']:.4f}")
    print("=" * 65)
    print(f"Saved evaluation metrics report to: {args.output_json}\n")


if __name__ == "__main__":
    main()

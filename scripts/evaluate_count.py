#!/usr/bin/env python3
"""Generate graphical metrics evaluation dashboard for Program Count."""

import argparse
import csv
import os
import sys
import numpy as np

# Use Agg backend for headless environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def generate_count_evaluation_figure(
    csv_path: str,
    output_path: str,
    gsd_m: float = 0.08,
):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(
                {
                    "tree_id": int(r["tree_id"]),
                    "cx": float(r["center_x"]),
                    "cy": float(r["center_y"]),
                    "conf": float(r["confidence"]),
                }
            )

    total_trees = len(records)
    confs = np.array([r["conf"] for r in records])
    points = np.array([[r["cx"], r["cy"]] for r in records])

    # 1. Nearest Neighbor Distance (NND)
    kdtree = KDTree(points)
    distances, _ = kdtree.query(points, k=2)
    nnd_px = distances[:, 1]
    nnd_m = nnd_px * gsd_m

    mean_spacing_m = float(np.mean(nnd_m))
    std_spacing_m = float(np.std(nnd_m))
    cv_spacing = float(std_spacing_m / max(1e-5, mean_spacing_m))

    # Clark Evans Index
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    area_m2 = (max_x - min_x) * (max_y - min_y) * (gsd_m**2)
    area_ha = area_m2 / 10000.0
    density_per_m2 = total_trees / max(1.0, area_m2)
    r_exp = 1.0 / (2.0 * np.sqrt(max(1e-6, density_per_m2)))
    clark_evans_r = mean_spacing_m / max(1e-5, r_exp)
    stand_density_ha = total_trees / max(1e-3, area_ha)

    # 2. Threshold sensitivity curve
    thresholds = np.linspace(0.1, 0.85, 30)
    counts_at_thresh = [np.sum(confs >= t) for t in thresholds]

    # Create 4-panel dashboard
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), dpi=200)
    fig.patch.set_facecolor("#0F141C")

    # Styling helper
    for ax in axs.flat:
        ax.set_facecolor("#161D27")
        ax.tick_params(colors="#CCCCCC", labelsize=9)
        ax.xaxis.label.set_color("#FFFFFF")
        ax.yaxis.label.set_color("#FFFFFF")
        ax.title.set_color("#FFFFFF")
        for spine in ax.spines.values():
            spine.set_color("#2A3644")

    # Panel 1: Confidence Score Distribution
    ax1 = axs[0, 0]
    n, bins, patches = ax1.hist(confs, bins=25, color="#00E676", edgecolor="#0A1017", alpha=0.85)
    ax1.axvline(np.mean(confs), color="#FFEB3B", linestyle="--", linewidth=2, label=f"Mean: {np.mean(confs):.2f}")
    ax1.axvline(0.25, color="#FF5252", linestyle=":", linewidth=2, label="Cutoff: 0.25")
    ax1.set_title("Crown Detection Confidence Distribution", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Confidence Score", fontsize=10)
    ax1.set_ylabel("Tree Count (Frequency)", fontsize=10)
    ax1.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#FFFFFF", fontsize=9)

    # Panel 2: Count Sensitivity vs Confidence Threshold
    ax2 = axs[0, 1]
    ax2.plot(thresholds, counts_at_thresh, color="#00B0FF", linewidth=2.5, marker="o", markersize=4)
    ax2.axvline(0.25, color="#FF5252", linestyle="--", alpha=0.7, label=f"Operational Threshold (Count = {total_trees})")
    ax2.set_title("Count Stability vs. Detection Threshold", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Confidence Threshold", fontsize=10)
    ax2.set_ylabel("Total Detected Stand Count", fontsize=10)
    ax2.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#FFFFFF", fontsize=9)

    # Panel 3: Nearest-Neighbor Distance (Inter-Tree Spacing)
    ax3 = axs[1, 0]
    ax3.hist(nnd_m, bins=22, color="#7C4DFF", edgecolor="#0A1017", alpha=0.85)
    ax3.axvline(mean_spacing_m, color="#FFEB3B", linestyle="--", linewidth=2, label=f"Observed: {mean_spacing_m:.2f} m")
    ax3.axvline(9.0, color="#00E676", linestyle=":", linewidth=2, label="Commercial Standard: 9.0 m")
    ax3.set_title("Inter-Tree Spacing Distribution (NND)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Distance to Nearest Palm (Meters)", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#FFFFFF", fontsize=9)

    # Panel 4: Spatial Density Quadrant Map & KPI Card
    ax4 = axs[1, 1]
    # Hexbin spatial density plot
    hb = ax4.hexbin(points[:, 0], points[:, 1], gridsize=18, cmap="viridis", mincnt=1)
    cb = fig.colorbar(hb, ax=ax4, orientation="vertical", pad=0.03, shrink=0.85)
    cb.set_label("Local Tree Count Density", color="#FFFFFF", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#CCCCCC")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#CCCCCC")
    ax4.invert_yaxis()  # Image coordinate convention
    ax4.set_title("Spatial Stand Density Distribution", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xlabel("X Coordinate (Pixels)", fontsize=10)
    ax4.set_ylabel("Y Coordinate (Pixels)", fontsize=10)

    # Supertitle and Agronomic KPI Header
    fig.suptitle(
        f"SawitVision Inventory Evaluation | Total Stand: {total_trees} Palms | Density: {stand_density_ha:.1f} t/ha | Regularity (R): {clark_evans_r:.2f}",
        fontsize=14,
        fontweight="bold",
        color="#FFFFFF",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved Count Metrics Evaluation Graph to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate count evaluation dashboard.")
    parser.add_argument("--csv", default="output_count/count_detections.csv", help="Detections CSV path.")
    parser.add_argument("--output", "-o", default="output_count/count_metrics_evaluation.png", help="Output PNG graph path.")
    args = parser.parse_args()
    generate_count_evaluation_figure(args.csv, args.output)

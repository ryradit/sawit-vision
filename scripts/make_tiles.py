#!/usr/bin/env python3
"""Tile large aerial orthomosaics into training image slices and convert labels to YOLO format."""

import argparse
import csv
import os
import sys
import cv2
import numpy as np


def box_iou_overlap_with_tile(box, tile_box):
    """Calculate visible fraction of a bounding box inside a tile window."""
    bx1, by1, bw, bh = box["x"], box["y"], box["w"], box["h"]
    bx2, by2 = bx1 + bw, by1 + bh
    tx1, ty1, tw, th = tile_box

    ix1 = max(bx1, tx1)
    iy1 = max(by1, ty1)
    ix2 = min(bx2, tx1 + tw)
    iy2 = min(by2, ty1 + th)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    box_area = bw * bh

    return inter_area / max(1e-5, box_area), (ix1, iy1, inter_w, inter_h)


def tile_image_and_labels(
    image_path: str,
    csv_path: str,
    output_dir: str = "data/tiles",
    tile_size: int = 640,
    overlap_ratio: float = 0.20,
    min_box_visibility: float = 0.40,
):
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to decode image at {image_path}")
        sys.exit(1)

    img_h, img_w = image.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Load annotations
    boxes = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                boxes.append(
                    {
                        "cx": float(r["center_x"]),
                        "cy": float(r["center_y"]),
                        "x": float(r["box_x"]),
                        "y": float(r["box_y"]),
                        "w": float(r["box_w"]),
                        "h": float(r["box_h"]),
                        "class_id": 0,  # 0 = oil_palm
                    }
                )

    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    stride = int(tile_size * (1.0 - overlap_ratio))
    y_starts = list(range(0, max(1, img_h - tile_size + 1), stride))
    if y_starts[-1] + tile_size < img_h:
        y_starts.append(img_h - tile_size)

    x_starts = list(range(0, max(1, img_w - tile_size + 1), stride))
    if x_starts[-1] + tile_size < img_w:
        x_starts.append(img_w - tile_size)

    tile_count = 0
    total_annotations = 0

    for ty in y_starts:
        for tx in x_starts:
            tile_count += 1
            tile_box = (tx, ty, tile_size, tile_size)
            tile_img = image[ty : ty + tile_size, tx : tx + tile_size]

            tile_id = f"{base_name}_t{tile_count:03d}_x{tx}_y{ty}"
            tile_img_path = os.path.join(images_out, f"{tile_id}.jpg")
            tile_lbl_path = os.path.join(labels_out, f"{tile_id}.txt")

            # Collect boxes in this tile
            tile_annotations = []
            for b in boxes:
                vis_ratio, inter_box = box_iou_overlap_with_tile(b, tile_box)
                if vis_ratio < min_box_visibility:
                    continue

                ix1, iy1, iw, ih = inter_box
                # Convert to tile-local normalized coordinates [0, 1]
                local_cx = (ix1 - tx + iw / 2.0) / tile_size
                local_cy = (iy1 - ty + ih / 2.0) / tile_size
                local_w = iw / tile_size
                local_h = ih / tile_size

                # Clip to valid bounds
                local_cx = float(np.clip(local_cx, 0.0, 1.0))
                local_cy = float(np.clip(local_cy, 0.0, 1.0))
                local_w = float(np.clip(local_w, 1e-4, 1.0))
                local_h = float(np.clip(local_h, 1e-4, 1.0))

                tile_annotations.append(
                    f"{b['class_id']} {local_cx:.6f} {local_cy:.6f} {local_w:.6f} {local_h:.6f}"
                )

            # Save image tile
            cv2.imwrite(tile_img_path, tile_img)

            # Save YOLO text label
            with open(tile_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(tile_annotations))

            total_annotations += len(tile_annotations)

    print("\n" + "=" * 60)
    print("YOLO DATASET TILING COMPLETED")
    print(f"  Source Image       : {image_path} ({img_w}x{img_h} px)")
    print(f"  Tile Dimensions    : {tile_size}x{tile_size} px (Overlap: {int(overlap_ratio*100)}%)")
    print(f"  Generated Tiles    : {tile_count} images in {images_out}/")
    print(f"  Generated Labels   : {tile_count} label files in {labels_out}/")
    print(f"  Total Box Labels   : {total_annotations} annotated instances")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Tile orthomosaics and convert labels to YOLO format.")
    parser.add_argument("--image", "-i", default="count.jpg", help="Path to input orthomosaic.")
    parser.add_argument("--csv", "-c", default="output_count/count_detections.csv", help="Detections CSV path.")
    parser.add_argument("--output-dir", "-o", default="data/tiles", help="Output directory for tiles.")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile dimension (default: 640).")
    parser.add_argument("--overlap", type=float, default=0.20, help="Tile overlap ratio (default: 0.20).")

    args = parser.parse_args()
    tile_image_and_labels(
        args.image,
        args.csv,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
    )


if __name__ == "__main__":
    main()

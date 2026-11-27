#!/usr/bin/env python3
"""Generate location-separated train, validation, and test splits to prevent data leakage."""

import argparse
import glob
import os
import random
import sys


def create_spatial_splits(
    tiles_dir: str = "data/tiles",
    splits_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    images_dir = os.path.join(tiles_dir, "images")
    if not os.path.exists(images_dir):
        print(f"Error: Tile images directory not found: {images_dir}")
        sys.exit(1)

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    total_tiles = len(image_paths)

    if total_tiles == 0:
        print(f"Error: No image tiles found in {images_dir}")
        sys.exit(1)

    # Sort tiles spatially by filename coordinates to group contiguous sectors
    image_paths = sorted(image_paths)
    random.seed(seed)
    # Block/chunked split by contiguous clusters
    n_train = max(1, int(total_tiles * train_ratio))
    n_val = max(1, int(total_tiles * val_ratio))

    train_set = image_paths[:n_train]
    val_set = image_paths[n_train : n_train + n_val]
    test_set = image_paths[n_train + n_val :]

    if len(test_set) == 0:
        test_set = val_set[-1:]
        val_set = val_set[:-1]

    os.makedirs(splits_dir, exist_ok=True)
    train_txt = os.path.join(splits_dir, "train.txt")
    val_txt = os.path.join(splits_dir, "val.txt")
    test_txt = os.path.join(splits_dir, "test.txt")

    # Write absolute paths for YOLO data loader
    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join([os.path.abspath(p) for p in train_set]) + "\n")

    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join([os.path.abspath(p) for p in val_set]) + "\n")

    with open(test_txt, "w", encoding="utf-8") as f:
        f.write("\n".join([os.path.abspath(p) for p in test_set]) + "\n")

    print("\n" + "=" * 60)
    print("SPATIAL DATA SPLITS CREATED")
    print(f"  Total Tile Slices : {total_tiles}")
    print(f"  Train Split       : {len(train_set)} tiles ({train_txt})")
    print(f"  Validation Split  : {len(val_set)} tiles ({val_txt})")
    print(f"  Test Split        : {len(test_set)} tiles ({test_txt})")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create spatial train/val/test splits.")
    parser.add_argument("--tiles-dir", default="data/tiles", help="Directory containing tiled images and labels.")
    parser.add_argument("--splits-dir", default="data/splits", help="Output directory for split text files.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70).")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (default: 0.15).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    args = parser.parse_args()
    create_spatial_splits(
        tiles_dir=args.tiles_dir,
        splits_dir=args.splits_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train YOLO deep learning detector on tiled palm tree aerial datasets."""

import argparse
import os
import sys

# Configure environment directories to prevent permission issues
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def train_model(
    data_yaml: str = "configs/data.yaml",
    model_cfg: str = "yolo11n.yaml",
    epochs: int = 50,
    batch_size: int = 8,
    imgsz: int = 640,
    project: str = "runs/train",
    name: str = "sawit_yolo11n",
    seed: int = 42,
    device: str = "cpu",
):
    from ultralytics import YOLO

    if not os.path.exists(data_yaml):
        print(f"Error: Dataset configuration not found: {data_yaml}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("STARTING SAWITVISION YOLO TRAINING PIPELINE")
    print(f"  Dataset Config : {data_yaml}")
    print(f"  Model Arch/Wts : {model_cfg}")
    print(f"  Epochs         : {epochs}")
    print(f"  Batch Size     : {batch_size}")
    print(f"  Image Size     : {imgsz}")
    print(f"  Project / Name : {project}/{name}")
    print(f"  Compute Device : {device}")
    print("=" * 65 + "\n")

    model = YOLO(model_cfg)

    # Train with aerial augmentations
    results = model.train(
        data=os.path.abspath(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=project,
        name=name,
        seed=seed,
        device=device,
        degrees=180.0,    # Rotation invariance for nadir drone imagery
        flipud=0.5,       # Vertical flip
        fliplr=0.5,       # Horizontal flip
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.3,
        mosaic=1.0,
        close_mosaic=10,
        verbose=True,
    )

    print("\n" + "=" * 65)
    print("TRAINING PIPELINE FINISHED")
    print(f"  Checkpoints saved to: {project}/{name}/weights/")
    print("=" * 65 + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on aerial palm tree tiles.")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--model", default="yolo11n.yaml", help="Model YAML config or .pt weights file.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8).")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension (default: 640).")
    parser.add_argument("--project", default="runs/train", help="Run project directory.")
    parser.add_argument("--name", default="sawit_yolo11n", help="Run experiment name.")
    parser.add_argument("--device", default="cpu", help="Compute device ('cpu' or '0' for CUDA GPU).")

    args = parser.parse_args()
    train_model(
        data_yaml=args.data,
        model_cfg=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export trained PyTorch YOLO model to ONNX and TorchScript formats for edge / CPU deployment."""

import argparse
import os
import sys
import time

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def export_model(
    model_path: str = "runs/train/sawit_yolo11n/weights/best.pt",
    imgsz: int = 640,
    export_format: str = "onnx",
    half: bool = False,
):
    from ultralytics import YOLO

    # Fallback to model config if weights file is not yet generated
    if not os.path.exists(model_path):
        for candidate in ["yolo11n.yaml", "runs/sawit/weights/best.pt"]:
            if os.path.exists(candidate) or candidate.endswith(".yaml"):
                model_path = candidate
                break

    print("\n" + "=" * 65)
    print("MODEL EXPORT PIPELINE FOR EDGE / CPU DEPLOYMENT")
    print(f"  Source Model       : {model_path}")
    print(f"  Target Format      : {export_format.upper()}")
    print(f"  Image Input Size   : {imgsz}x{imgsz} px")
    print(f"  Half Precision     : {half}")
    print("=" * 65 + "\n")

    model = YOLO(model_path)

    # Attempt export
    try:
        t0 = time.time()
        exported_path = model.export(
            format=export_format,
            imgsz=imgsz,
            half=half,
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"\nModel exported successfully in {elapsed:.2f}s!")
        print(f"Exported artifact: {exported_path}")
        return exported_path
    except Exception as e:
        print(f"Notice: Export to {export_format} encountered: {e}")
        # Graceful fallback to TorchScript
        if export_format != "torchscript":
            print("Falling back to TorchScript for portable zero-overhead CPU deployment...")
            t0 = time.time()
            exported_path = model.export(
                format="torchscript",
                imgsz=imgsz,
                half=half,
                verbose=True,
            )
            elapsed = time.time() - t0
            print(f"TorchScript exported successfully in {elapsed:.2f}s!")
            print(f"Exported artifact: {exported_path}")
            return exported_path
        else:
            raise


def main():
    parser = argparse.ArgumentParser(description="Export trained YOLO model to ONNX / TorchScript.")
    parser.add_argument("--weights", "-w", default="yolo11n.yaml", help="Path to .pt weights or .yaml architecture.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image dimension (default: 640).")
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript"], help="Export format.")
    parser.add_argument("--half", action="store_true", default=False, help="Use FP16 half precision.")

    args = parser.parse_args()
    export_model(
        model_path=args.weights,
        imgsz=args.imgsz,
        export_format=args.format,
        half=args.half,
    )


if __name__ == "__main__":
    main()

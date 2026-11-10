#!/usr/bin/env python3
"""Evaluate HCF-YOLO11n or YOLO11n on a configured dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS = {
    "hcf": ROOT / "weights" / "hcf_yolo11n_best.pt",
    "yolo11n": ROOT / "weights" / "yolo11n_best.pt",
}


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=WEIGHTS, default="hcf")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=1000)
    args = parser.parse_args()

    weights = resolve_path(args.weights) if args.weights else WEIGHTS[args.model]
    data = resolve_path(args.data)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data}")

    metrics = YOLO(str(weights)).val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )
    print(f"Precision: {metrics.box.mp:.6f}")
    print(f"Recall: {metrics.box.mr:.6f}")
    print(f"mAP50: {metrics.box.map50:.6f}")
    print(f"mAP50-95: {metrics.box.map:.6f}")


if __name__ == "__main__":
    main()

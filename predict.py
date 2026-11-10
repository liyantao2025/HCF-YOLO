#!/usr/bin/env python3
"""Run image or folder prediction with HCF-YOLO11n or YOLO11n."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=WEIGHTS, default="hcf")
    parser.add_argument("--weights", default=None, help="Optional checkpoint override")
    parser.add_argument("--source", required=True, help="Image, video, directory, URL, or camera index")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default=None, help="GPU index such as 0, cpu, or omit for auto")
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = resolve_path(args.weights) if args.weights else WEIGHTS[args.model]
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    source = args.source
    candidate = resolve_path(source)
    if candidate.exists():
        source = str(candidate)

    model = YOLO(str(weights))
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        save=True,
        project=str(resolve_path(args.project)),
        name=args.name or args.model,
        exist_ok=True,
    )
    print(f"Prediction complete: {len(results)} input(s)")
    print(f"Results directory: {resolve_path(args.project) / (args.name or args.model)}")


if __name__ == "__main__":
    main()

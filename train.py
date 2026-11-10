#!/usr/bin/env python3
"""Train HCF-YOLO11n or the YOLO11n baseline from a YAML definition."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "hcf": ROOT / "models" / "hcf_yolo11n.yaml",
    "yolo11n": ROOT / "models" / "yolo11n.yaml",
}


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_FILES, default="hcf")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="GPU index such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the training set to use")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--check", action="store_true", help="Build the model and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_file = MODEL_FILES[args.model]
    if not model_file.is_file():
        raise FileNotFoundError(f"Model definition not found: {model_file}")

    model = YOLO(str(model_file))
    parameters = sum(p.numel() for p in model.model.parameters())
    print(f"Model check passed: {args.model} ({parameters:,} parameters)")
    if args.check:
        return

    data_file = resolve_path(args.data)
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_file}")

    train_args = {
        "data": str(data_file),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "fraction": args.fraction,
        "deterministic": True,
        "pretrained": False,
        "optimizer": "Adam",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "patience": 300,
        "amp": not args.no_amp,
        "val": True,
        "plots": True,
        "verbose": True,
        "max_det": 1000,
        "close_mosaic": 20,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "translate": 0.08,
        "scale": 0.2,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "auto_augment": None,
        "erasing": 0.0,
        "project": str(resolve_path(args.project)),
        "name": args.name or args.model,
        "exist_ok": True,
    }
    if args.model == "hcf":
        train_args.update(
            box_loss_type="sa_wiou_v3",
            use_center_guidance=True,
            center_gain=0.1,
        )
    else:
        train_args.update(
            box_loss_type="ciou",
            use_center_guidance=False,
        )

    model.train(**train_args)


if __name__ == "__main__":
    main()

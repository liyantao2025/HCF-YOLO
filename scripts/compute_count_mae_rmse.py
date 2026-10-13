"""Compute per-image counting MAE/RMSE for two YOLO detection models."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

# Prefer the project-local Ultralytics source so custom HCF-YOLO modules can
# be reconstructed from the saved checkpoint.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--hcf", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/count_metrics"))
    p.add_argument("--expected-images", type=int, default=104)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def ground_truth_count(label_dir: Path, image: Path) -> int:
    label = label_dir / f"{image.stem}.txt"
    if not label.is_file():
        raise FileNotFoundError(f"Missing label for {image.name}: {label}")
    return sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())


def predict_counts(model_path: Path, images: list[Path], args: argparse.Namespace) -> list[int]:
    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(x) for x in images], imgsz=args.imgsz, conf=args.conf,
        iou=args.iou, max_det=args.max_det, device=args.device,
        save=False, verbose=False, stream=False,
    )
    return [len(r.boxes) for r in results]


def summarize(errors: list[int]) -> tuple[float, float, float]:
    n = len(errors)
    mae = sum(abs(x) for x in errors) / n
    rmse = math.sqrt(sum(x * x for x in errors) / n)
    bias = sum(errors) / n
    return mae, rmse, bias


def main() -> None:
    args = parse_args()
    images = sorted(x for x in args.images.iterdir() if x.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) != args.expected_images:
        raise RuntimeError(
            f"Expected {args.expected_images} test images, found {len(images)} in {args.images}. "
            "Refusing to produce manuscript metrics from an incomplete split."
        )

    gt = [ground_truth_count(args.labels, x) for x in images]
    baseline = predict_counts(args.baseline, images, args)
    hcf = predict_counts(args.hcf, images, args)
    args.output.mkdir(parents=True, exist_ok=True)

    detail = args.output / "count_metrics_per_image.csv"
    with detail.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "ground_truth", "YOLO11n_pred", "YOLO11n_error", "HCF-YOLO11n_pred", "HCF-YOLO11n_error"])
        for image, true, b, h in zip(images, gt, baseline, hcf):
            writer.writerow([image.name, true, b, b - true, h, h - true])

    summary = args.output / "count_metrics_summary.csv"
    rows = []
    for name, pred in (("YOLO11n", baseline), ("HCF-YOLO11n", hcf)):
        errors = [p - t for p, t in zip(pred, gt)]
        mae, rmse, bias = summarize(errors)
        rows.append([name, len(images), args.conf, args.imgsz, mae, rmse, bias])
    with summary.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "images", "confidence", "imgsz", "MAE", "RMSE", "mean_error_bias"])
        writer.writerows(rows)

    print(summary.resolve())
    for row in rows:
        print(f"{row[0]}: MAE={row[4]:.4f}, RMSE={row[5]:.4f}, bias={row[6]:.4f}")


if __name__ == "__main__":
    main()

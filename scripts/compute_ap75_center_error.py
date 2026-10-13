"""Compute AP75 and center-localization error for two YOLO detectors.

AP75 is obtained with the same validation confidence/NMS settings used in the
project (conf=0.001, NMS IoU=0.7, max_det=1000). Center error is calculated on
one-to-one Hungarian matches with IoU >= 0.50 at the fixed operating point
conf=0.15, NMS IoU=0.5, max_det=1000. The error is reported in pixels and as a
percentage of the image diagonal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def xywhn_to_xyxy(label_path: Path, width: int, height: int) -> np.ndarray:
    rows = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, xc, yc, bw, bh = map(float, parts[:5])
            rows.append([
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height,
            ])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def matched_center_errors(gt: np.ndarray, pred: np.ndarray, iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    ious = pairwise_iou(gt, pred)
    if ious.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    rows, cols = linear_sum_assignment(1.0 - ious)
    keep = ious[rows, cols] >= iou_thr
    rows, cols = rows[keep], cols[keep]
    gt_c = (gt[rows, :2] + gt[rows, 2:]) / 2
    pr_c = (pred[cols, :2] + pred[cols, 2:]) / 2
    return np.linalg.norm(pr_c - gt_c, axis=1), ious[rows, cols]


def local_data_yaml(out_dir: Path) -> Path:
    path = (ROOT / "dataset").resolve().as_posix()
    p = out_dir / "data_local.yaml"
    p.write_text(
        f"path: {path}\ntrain: images/train_improved\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: cotton\n",
        encoding="utf-8",
    )
    return p


def evaluate(name: str, weights: Path, data_yaml: Path, out_dir: Path) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=640, batch=1, device="cpu",
        conf=0.001, iou=0.7, max_det=1000, plots=False, save=False,
        project=str(out_dir / "validation"), name=name, exist_ok=True, verbose=False,
    )
    ap75 = float(metrics.box.map75)

    image_dir = ROOT / "dataset" / "images" / "test"
    label_dir = ROOT / "dataset" / "labels" / "test"
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    predictions = model.predict(
        source=[str(p) for p in image_paths], imgsz=640, device="cpu",
        conf=0.15, iou=0.5, max_det=1000, save=False, verbose=False, stream=False,
    )

    all_px: list[float] = []
    all_norm: list[float] = []
    all_iou: list[float] = []
    rows = []
    for image_path, result in zip(image_paths, predictions):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read {image_path}")
        h, w = image.shape[:2]
        gt = xywhn_to_xyxy(label_dir / f"{image_path.stem}.txt", w, h)
        pred = result.boxes.xyxy.cpu().numpy().astype(np.float32) if result.boxes is not None else np.empty((0, 4), np.float32)
        errors, match_ious = matched_center_errors(gt, pred, 0.50)
        norm = errors / math.hypot(w, h) * 100.0
        all_px.extend(errors.tolist())
        all_norm.extend(norm.tolist())
        all_iou.extend(match_ious.tolist())
        rows.append({
            "model": name,
            "image": image_path.name,
            "ground_truth_count": len(gt),
            "prediction_count_conf_0.15": len(pred),
            "matched_count_iou_0.50": len(errors),
            "mean_center_error_px": float(errors.mean()) if len(errors) else None,
            "mean_normalized_center_error_percent": float(norm.mean()) if len(norm) else None,
            "mean_matched_iou": float(match_ious.mean()) if len(match_ious) else None,
        })

    px = np.asarray(all_px, dtype=float)
    norm = np.asarray(all_norm, dtype=float)
    ious = np.asarray(all_iou, dtype=float)
    summary = {
        "model": name,
        "weights": str(weights.resolve()),
        "test_images": len(image_paths),
        "ap75": ap75,
        "center_error_conf": 0.15,
        "center_match_iou": 0.50,
        "matched_targets": int(len(px)),
        "mean_center_error_px": float(px.mean()),
        "std_center_error_px": float(px.std(ddof=1)),
        "median_center_error_px": float(np.median(px)),
        "mean_normalized_center_error_percent": float(norm.mean()),
        "std_normalized_center_error_percent": float(norm.std(ddof=1)),
        "mean_matched_iou": float(ious.mean()),
    }
    return {"summary": summary, "per_image": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "third_priority_metrics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output / "ultralytics_config"))
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    data_yaml = local_data_yaml(args.output)

    models = [
        ("YOLO11n", ROOT / "runs" / "ablation" / "yolo11n_baseline" / "weights" / "best.pt"),
        ("HCF-YOLO11n", ROOT / "runs" / "ablation" / "diagnose_E_plus_GCG_only" / "weights" / "best.pt"),
    ]
    results = [evaluate(name, weights, data_yaml, args.output) for name, weights in models]

    summaries = [x["summary"] for x in results]
    details = [row for x in results for row in x["per_image"]]
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader(); writer.writerows(details)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

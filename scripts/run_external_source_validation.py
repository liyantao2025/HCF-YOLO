"""Train from scratch on COT-AD and evaluate once on untouched Cotton-BNYQS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


DATA = ROOT / "external_validation" / "external_validation.yaml"
PROJECT = ROOT / "runs" / "external_source_validation"
MODELS = {
    "yolo11n": {
        "display": "YOLO11n",
        "yaml": ROOT / "models" / "yolo11n.yaml",
        "use_center_guidance": False,
    },
    "hcf_yolo11n": {
        "display": "HCF-YOLO11n",
        "yaml": ROOT / "models" / "yolo11n-drf-safpn-midlite-p2safe-r12-v1-gcg.yaml",
        "use_center_guidance": True,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_metrics(model: YOLO, conf: float = 0.25) -> dict:
    image_dir = ROOT / "external_validation" / "dataset" / "images" / "test"
    label_dir = ROOT / "external_validation" / "dataset" / "labels" / "test"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    results = model.predict(
        source=[str(p) for p in images], imgsz=640, conf=conf, iou=0.70,
        max_det=1000, device=0, batch=16, save=False, verbose=False,
    )
    errors = []
    for image, result in zip(images, results):
        label = label_dir / f"{image.stem}.txt"
        true_count = sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())
        errors.append(len(result.boxes) - true_count)
    return {
        "count_MAE": float(np.mean(np.abs(errors))),
        "count_RMSE": float(math.sqrt(np.mean(np.square(errors)))),
        "count_bias": float(np.mean(errors)),
        "count_conf": conf,
    }


def train_and_evaluate(model_key: str, seed: int, epochs: int, batch: int) -> dict:
    spec = MODELS[model_key]
    run_name = f"{model_key}_seed{seed}"
    result_json = PROJECT / run_name / "external_test_metrics.json"
    if result_json.exists():
        print(f"[skip] {run_name}")
        return json.loads(result_json.read_text(encoding="utf-8"))

    set_seed(seed)
    model = YOLO(str(spec["yaml"]))
    model.train(
        data=str(DATA), epochs=epochs, patience=300, batch=batch, imgsz=640,
        device=0, workers=4, project=str(PROJECT), name=run_name,
        pretrained=False, resume=False, optimizer="Adam", lr0=0.001, lrf=0.01,
        momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5,
        box_loss_type="ciou", use_center_guidance=spec["use_center_guidance"],
        center_gain=0.1, seed=seed, deterministic=True, amp=True,
        mosaic=0.5, close_mosaic=20, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, degrees=0.0, translate=0.08,
        scale=0.2, shear=0.0, perspective=0.0, fliplr=0.5, flipud=0.0,
        auto_augment=None, erasing=0.0, conf=0.001, iou=0.70, max_det=1000,
        plots=True, val=True, exist_ok=False,
    )
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    evaluated = YOLO(str(best))
    metrics = evaluated.val(
        data=str(DATA), split="test", imgsz=640, batch=batch, device=0,
        conf=0.001, iou=0.70, max_det=1000, plots=True,
        project=str(PROJECT), name=f"{run_name}_external_test",
    )
    rd = metrics.results_dict
    ap75 = float(getattr(metrics.box, "ap75", 0.0))
    result = {
        "Model": spec["display"], "seed": seed, "train_images": 842,
        "val_images": 98, "external_test_images": 100,
        "Precision": float(rd["metrics/precision(B)"]),
        "Recall": float(rd["metrics/recall(B)"]),
        "mAP50": float(rd["metrics/mAP50(B)"]),
        "mAP50-95": float(rd["metrics/mAP50-95(B)"]),
        "AP75": ap75, "weights": str(best), "pretrained": False,
    }
    result.update(count_metrics(evaluated))
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Protocol: COT-AD train/val only; Cotton-BNYQS external test only; random initialization")
    if args.check:
        for key, spec in MODELS.items():
            m = YOLO(str(spec["yaml"]))
            params = sum(p.numel() for p in m.model.parameters())
            print(f"{key}: {params / 1e6:.3f} M parameters")
        return

    keys = list(MODELS) if args.model == "all" else [args.model]
    results = [train_and_evaluate(key, args.seed, args.epochs, args.batch) for key in keys]
    out = PROJECT / f"external_test_summary_seed{args.seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline vs full 检测效果可视化对比（带框图、热力图，单图输出不拼接）。"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BASELINE_BGR = (0, 165, 255)  # orange
FULL_BGR = (0, 255, 0)  # green
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare baseline YOLOv11n vs full model detections for paper visualization."
    )
    p.add_argument(
        "--baseline",
        type=str,
        default="ablation/yolo11n_baseline/weights/best.pt",
        help="Baseline model weights",
    )
    p.add_argument(
        "--full",
        type=str,
        default="ablation/yolo11n_full/weights/best.pt",
        help="Full model weights",
    )
    p.add_argument("--data", type=str, default="data.yaml", help="Dataset yaml")
    p.add_argument("--source", type=str, default=None, help="Image directory (optional)")
    p.add_argument("--num", type=int, default=10, help="Number of images to visualize")
    p.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=1000, help="Max detections per image")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    p.add_argument("--save-dir", type=str, default="outputs/vis_compare", help="Output root directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed for --random")
    p.add_argument("--random", action="store_true", help="Randomly sample images")
    p.add_argument("--crop-size", type=int, default=512, help="Crop window size for summary.csv region stats")
    p.add_argument("--heatmap-alpha", type=float, default=0.45, help="Heatmap overlay alpha")
    p.add_argument(
        "--line-width",
        type=int,
        default=0,
        help="Box line width in pixels; 0 = auto thin (1px for most images)",
    )
    p.add_argument(
        "--hide-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide class labels on boxes (default: hide)",
    )
    p.add_argument("--show-conf", action="store_true", help="Show confidence on boxes")
    p.add_argument("--grid", type=int, default=5, choices=[4, 5], help="Grid size for crop region selection")
    return p.parse_args()


def resolve_path(path_str: str | Path, base: Path | None = None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    root = base or ROOT
    return (root / p).resolve()


def load_data_yaml(data_path: Path) -> dict[str, Any]:
    with open(data_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def images_dir_from_data(data_path: Path) -> Path:
    """Resolve test images dir from data.yaml; fallback to val."""
    cfg = load_data_yaml(data_path)
    dataset_root = cfg.get("path", ".")
    dataset_root = resolve_path(dataset_root, ROOT)

    for key in ("test", "val"):
        sub = cfg.get(key)
        if not sub:
            continue
        candidate = resolve_path(sub, dataset_root)
        if candidate.is_dir():
            return candidate
        # allow file path style ending with images/...
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find test/val image directory in {data_path}. "
        "Set 'test' or 'val' in data.yaml or pass --source."
    )


def get_image_list(
    source: Path | None,
    data_yaml: Path,
    num: int,
    random_pick: bool,
    seed: int,
) -> list[Path]:
    if source is not None:
        img_dir = resolve_path(source)
    else:
        img_dir = images_dir_from_data(data_yaml)

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    images = sorted(
        p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    if random_pick:
        rng = random.Random(seed)
        k = min(num, len(images))
        return rng.sample(images, k)
    return images[: min(num, len(images))]


def load_yolo(weights: Path, label: str) -> YOLO:
    if not weights.exists():
        raise FileNotFoundError(f"{label} weights not found: {weights}")
    try:
        model = YOLO(str(weights))
        model.model.eval()
        return model
    except Exception as exc:
        if label.lower() == "full":
            hint = (
                "\n[Full model load failed] Please check:\n"
                "  - ultralytics/nn/modules/drf_safpn.py exists\n"
                "  - ultralytics/nn/modules/__init__.py registers custom modules\n"
                "  - Run from project root: python scripts/compare_baseline_full_visual.py\n"
                f"  - Weights: {weights}\n"
            )
            raise RuntimeError(hint) from exc
        raise


def predict_boxes(
    model: YOLO,
    image_path: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xyxy (N,4), conf (N,), cls (N,) in original image coordinates."""
    with torch.no_grad():
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    xyxy = r0.boxes.xyxy.cpu().numpy()
    conf = r0.boxes.conf.cpu().numpy()
    cls = r0.boxes.cls.cpu().numpy()
    return xyxy, conf, cls


def box_line_width(h: int, w: int, override: int = 0) -> int:
    if override > 0:
        return override
    # 高密度小目标：默认 1px；大图最多 2px
    return max(1, min(2, int(round(min(h, w) / 1200))))


def draw_detections(
    image_bgr: np.ndarray,
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    names: dict[int, str],
    color_bgr: tuple[int, int, int],
    hide_label: bool,
    show_conf: bool,
    line_width: int = 0,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    lw = box_line_width(h, w, line_width)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, lw, lineType=cv2.LINE_AA)
        if hide_label and not show_conf:
            continue
        parts: list[str] = []
        if not hide_label:
            cidx = int(cls[i]) if i < len(cls) else 0
            parts.append(names.get(cidx, str(cidx)))
        if show_conf and i < len(conf):
            parts.append(f"{conf[i]:.2f}")
        if parts:
            label = " ".join(parts)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(y1, th + 4)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 2, ty + baseline), color_bgr, -1)
            cv2.putText(
                out,
                label,
                (x1 + 1, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def generate_detection_heatmap(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    mode: str = "max",
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_TURBO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build detection-response heatmap from box centers and confidences."""
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    if len(boxes):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            bw = max(float(x2 - x1), 1.0)
            bh = max(float(y2 - y1), 1.0)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            radius = max(3.0, 0.5 * min(bw, bh))
            sigma = max(radius / 2.0, 1.0)
            gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
            gauss *= float(score)
            if mode == "max":
                heatmap = np.maximum(heatmap, gauss)
            else:
                heatmap += gauss

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_gray = (heatmap * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, colormap)
    heatmap_overlay = cv2.addWeighted(image, 1.0 - alpha, heatmap_color, alpha, 0)
    return heatmap_color, heatmap_overlay, heatmap_gray


def select_crop_region(
    image_shape: tuple[int, int],
    boxes_xyxy: np.ndarray,
    crop_size: int,
    grid_n: int = 5,
) -> tuple[int, int, int, int]:
    h, w = image_shape
    half = crop_size // 2

    if len(boxes_xyxy) == 0:
        cx, cy = w // 2, h // 2
    else:
        centers_x = 0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2])
        centers_y = 0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3])
        cell_w = w / grid_n
        cell_h = h / grid_n
        counts = np.zeros((grid_n, grid_n), dtype=int)
        for cx_pt, cy_pt in zip(centers_x, centers_y):
            gi = min(int(cy_pt / cell_h), grid_n - 1)
            gj = min(int(cx_pt / cell_w), grid_n - 1)
            counts[gi, gj] += 1
        gi, gj = np.unravel_index(np.argmax(counts), counts.shape)
        cx = int((gj + 0.5) * cell_w)
        cy = int((gi + 0.5) * cell_h)

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    if x2 - x1 < crop_size:
        x1 = max(0, x2 - crop_size)
    if y2 - y1 < crop_size:
        y1 = max(0, y2 - crop_size)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    return int(x1), int(y1), int(x2), int(y2)


def conf_stats(conf: np.ndarray) -> dict[str, float]:
    if len(conf) == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    return {"mean": float(conf.mean()), "max": float(conf.max()), "min": float(conf.min())}


def ensure_dirs(save_root: Path) -> dict[str, Path]:
    sub = {
        "boxed_baseline": save_root / "boxed" / "baseline",
        "boxed_full": save_root / "boxed" / "full",
        "heatmap_baseline": save_root / "heatmap" / "baseline",
        "heatmap_full": save_root / "heatmap" / "full",
        "heatmap_raw_baseline": save_root / "heatmap_raw" / "baseline",
        "heatmap_raw_full": save_root / "heatmap_raw" / "full",
    }
    for d in sub.values():
        d.mkdir(parents=True, exist_ok=True)
    return sub


def main() -> None:
    args = parse_args()
    save_root = resolve_path(args.save_dir)
    data_yaml = resolve_path(args.data)
    baseline_w = resolve_path(args.baseline)
    full_w = resolve_path(args.full)
    source = resolve_path(args.source) if args.source else None

    dirs = ensure_dirs(save_root)
    images = get_image_list(source, data_yaml, args.num, args.random, args.seed)
    print(f"Images: {len(images)} from {images[0].parent}")
    print(f"Save dir: {save_root}")

    print(f"Loading baseline: {baseline_w}")
    model_baseline = load_yolo(baseline_w, "Baseline")
    print(f"Loading full: {full_w}")
    model_full = load_yolo(full_w, "Full")

    names_b = model_baseline.names if hasattr(model_baseline, "names") else {0: "cotton"}
    names_f = model_full.names if hasattr(model_full, "names") else names_b

    summary_rows: list[dict[str, Any]] = []
    lw = args.line_width

    for img_path in images:
        out_name = img_path.name
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] Skip unreadable image: {img_path}")
            continue

        bb_xyxy, bb_conf, bb_cls = predict_boxes(model_baseline, img_path, args)
        ff_xyxy, ff_conf, ff_cls = predict_boxes(model_full, img_path, args)

        boxed_b = draw_detections(
            bgr, bb_xyxy, bb_conf, bb_cls, names_b, BASELINE_BGR, args.hide_label, args.show_conf, lw
        )
        boxed_f = draw_detections(
            bgr, ff_xyxy, ff_conf, ff_cls, names_f, FULL_BGR, args.hide_label, args.show_conf, lw
        )

        hm_b_color, hm_b_overlay, hm_b_gray = generate_detection_heatmap(
            bgr, bb_xyxy, bb_conf, mode="max", alpha=args.heatmap_alpha
        )
        hm_f_color, hm_f_overlay, hm_f_gray = generate_detection_heatmap(
            bgr, ff_xyxy, ff_conf, mode="max", alpha=args.heatmap_alpha
        )

        cv2.imwrite(str(dirs["boxed_baseline"] / out_name), boxed_b)
        cv2.imwrite(str(dirs["boxed_full"] / out_name), boxed_f)
        cv2.imwrite(str(dirs["heatmap_baseline"] / out_name), hm_b_overlay)
        cv2.imwrite(str(dirs["heatmap_full"] / out_name), hm_f_overlay)
        cv2.imwrite(str(dirs["heatmap_raw_baseline"] / out_name), hm_b_color)
        cv2.imwrite(str(dirs["heatmap_raw_full"] / out_name), hm_f_color)

        crop_box = select_crop_region(bgr.shape[:2], ff_xyxy, args.crop_size, args.grid)

        bb_st = conf_stats(bb_conf)
        ff_st = conf_stats(ff_conf)
        summary_rows.append(
            {
                "image_name": out_name,
                "image_path": str(img_path.resolve()),
                "baseline_num_boxes": len(bb_xyxy),
                "full_num_boxes": len(ff_xyxy),
                "baseline_mean_conf": bb_st["mean"],
                "full_mean_conf": ff_st["mean"],
                "baseline_max_conf": bb_st["max"],
                "full_max_conf": ff_st["max"],
                "baseline_min_conf": bb_st["min"],
                "full_min_conf": ff_st["min"],
                "baseline_heatmap_mean": float(hm_b_gray.mean()),
                "full_heatmap_mean": float(hm_f_gray.mean()),
                "baseline_heatmap_max": int(hm_b_gray.max()),
                "full_heatmap_max": int(hm_f_gray.max()),
                "crop_x1": crop_box[0],
                "crop_y1": crop_box[1],
                "crop_x2": crop_box[2],
                "crop_y2": crop_box[3],
            }
        )
        print(
            f"  {out_name}: baseline={len(bb_xyxy)} full={len(ff_xyxy)} "
            f"crop={crop_box}"
        )

    summary_path = save_root / "summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else [
        "image_name",
        "image_path",
        "baseline_num_boxes",
        "full_num_boxes",
        "baseline_mean_conf",
        "full_mean_conf",
        "baseline_max_conf",
        "full_max_conf",
        "baseline_min_conf",
        "full_min_conf",
        "baseline_heatmap_mean",
        "full_heatmap_mean",
        "baseline_heatmap_max",
        "full_heatmap_max",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()

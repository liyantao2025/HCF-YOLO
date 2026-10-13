#!/usr/bin/env python3
"""Reproducible NCNN latency benchmark for Raspberry Pi 5.

The source images are decoded once before timing. Reported end-to-end latency
therefore includes Ultralytics preprocessing, NCNN inference, and NMS/postprocess,
but excludes disk I/O and result visualization/saving.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import ncnn
import numpy as np
import psutil
import torch
import ultralytics
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def temperature_c() -> float:
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return float(path.read_text(encoding="utf-8").strip()) / 1000.0
    except (OSError, ValueError):
        return float("nan")


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return "unavailable"


def directory_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024**2)


def load_images(source: Path) -> tuple[list[Path], list[np.ndarray]]:
    paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"No supported images found under: {source}")
    images: list[np.ndarray] = []
    valid_paths: list[Path] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            valid_paths.append(path)
            images.append(image)
    if not images:
        raise SystemExit(f"OpenCV could not decode any images under: {source}")
    return valid_paths, images


def configure_ncnn_threads(model: YOLO, threads: int) -> int:
    if hasattr(ncnn, "set_omp_dynamic"):
        ncnn.set_omp_dynamic(0)
    if hasattr(ncnn, "set_omp_num_threads"):
        ncnn.set_omp_num_threads(threads)
    backend = getattr(getattr(model, "predictor", None), "model", None)
    net = getattr(backend, "net", None)
    if net is None:
        raise RuntimeError("Ultralytics NCNN backend was not initialized.")
    net.opt.num_threads = threads
    return int(net.opt.num_threads)


def predict_once(model: YOLO, image: np.ndarray, args: argparse.Namespace):
    return model.predict(
        source=image,
        imgsz=args.imgsz,
        device="cpu",
        batch=1,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=False,
        verbose=False,
    )[0]


def summary(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_ms": statistics.fmean(values),
        f"{prefix}_median_ms": statistics.median(values),
        f"{prefix}_std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        f"{prefix}_p95_ms": percentile(values, 0.95),
        f"{prefix}_min_ms": min(values),
        f"{prefix}_max_ms": max(values),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one Ultralytics NCNN detector on Raspberry Pi 5.")
    parser.add_argument("--model", required=True, help="Path to the *_ncnn_model directory.")
    parser.add_argument("--name", required=True, help="Short model name used in output files.")
    parser.add_argument("--source", default="dataset/images/test", help="Directory containing replay images.")
    parser.add_argument("--output", default="results/formal_benchmark", help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--cooldown", type=float, default=20.0, help="Seconds between repeats.")
    parser.add_argument("--sample-every", type=int, default=10, help="Resource sampling interval.")
    parser.add_argument("--quick", action="store_true", help="Allow a short pipeline check; never use its values in a paper.")
    args = parser.parse_args()

    if not args.quick and (args.warmup < 100 or args.iterations < 1000 or args.repeats < 3):
        raise SystemExit("Formal protocol requires warmup>=100, iterations>=1000, and repeats>=3.")

    model_path = Path(args.model).resolve()
    source_path = Path(args.source).resolve()
    output_path = Path(args.output).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"NCNN model directory not found: {model_path}")

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    image_paths, images = load_images(source_path)
    process = psutil.Process(os.getpid())

    print(f"Loaded {len(images)} replay images into memory; disk I/O is excluded from timing.")
    print(f"Loading NCNN model: {model_path}")
    model = YOLO(str(model_path), task="detect")

    # Initialize the Ultralytics predictor/backend once, then apply an explicit
    # thread count to both global OpenMP and this NCNN Net instance.
    _ = predict_once(model, images[0], args)
    actual_threads = configure_ncnn_threads(model, args.threads)
    print(f"NCNN threads: {actual_threads}")

    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_model": read_text("/proc/device-tree/model"),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "ncnn": getattr(ncnn, "__version__", "unknown"),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_memory_gib": psutil.virtual_memory().total / (1024**3),
        "model_name": args.name,
        "model_path": str(model_path),
        "model_size_mb": directory_size_mb(model_path),
        "source": str(source_path),
        "decoded_replay_images": len(images),
        "input_size": args.imgsz,
        "batch_size": 1,
        "precision": "FP32",
        "runtime": "NCNN via Ultralytics",
        "ncnn_threads": actual_threads,
        "warmup_iterations_per_repeat": args.warmup,
        "timed_iterations_per_repeat": args.iterations,
        "repeats": args.repeats,
        "confidence_threshold": args.conf,
        "nms_iou_threshold": args.iou,
        "max_detections": args.max_det,
        "end_to_end_definition": "decoded BGR frame -> preprocess -> NCNN inference -> NMS/postprocess; excludes disk I/O and saving",
        "power_measurement": "not measured by software; use an external USB-C power meter",
    }
    output_path.mkdir(parents=True, exist_ok=True)
    env_path = output_path / f"{args.name}_environment.json"
    env_path.write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_rows: list[dict] = []
    repeat_rows: list[dict] = []

    for repeat in range(1, args.repeats + 1):
        if repeat > 1 and args.cooldown > 0:
            print(f"Cooling down for {args.cooldown:.0f} s before repeat {repeat}...")
            time.sleep(args.cooldown)

        print(f"Repeat {repeat}/{args.repeats}: warming up for {args.warmup} iterations...")
        for index in range(args.warmup):
            _ = predict_once(model, images[index % len(images)], args)

        gc.collect()
        psutil.cpu_percent(interval=None)
        start_temp = temperature_c()
        start_rss = process.memory_info().rss / (1024**2)
        e2e_values: list[float] = []
        pre_values: list[float] = []
        infer_values: list[float] = []
        post_values: list[float] = []
        detections: list[int] = []
        temps: list[float] = [start_temp]
        rss_values: list[float] = [start_rss]
        cpu_values: list[float] = []

        print(f"Repeat {repeat}/{args.repeats}: running {args.iterations} timed iterations...")
        for iteration in range(1, args.iterations + 1):
            image_index = (iteration - 1) % len(images)
            t0 = time.perf_counter_ns()
            result = predict_once(model, images[image_index], args)
            e2e_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            speed = result.speed or {}
            pre_ms = float(speed.get("preprocess", float("nan")))
            infer_ms = float(speed.get("inference", float("nan")))
            post_ms = float(speed.get("postprocess", float("nan")))
            count = len(result.boxes) if result.boxes is not None else 0

            sampled = iteration % args.sample_every == 0 or iteration == 1
            temp = temperature_c() if sampled else float("nan")
            rss = process.memory_info().rss / (1024**2) if sampled else float("nan")
            cpu = psutil.cpu_percent(interval=None) if sampled else float("nan")
            if sampled:
                temps.append(temp)
                rss_values.append(rss)
                cpu_values.append(cpu)

            e2e_values.append(e2e_ms)
            pre_values.append(pre_ms)
            infer_values.append(infer_ms)
            post_values.append(post_ms)
            detections.append(count)
            detail_rows.append(
                {
                    "model": args.name,
                    "repeat": repeat,
                    "iteration": iteration,
                    "image": image_paths[image_index].name,
                    "preprocess_ms": pre_ms,
                    "inference_ms": infer_ms,
                    "postprocess_ms": post_ms,
                    "end_to_end_ms": e2e_ms,
                    "detections": count,
                    "temperature_c": temp,
                    "rss_mb": rss,
                    "system_cpu_percent": cpu,
                }
            )

        valid_temps = [x for x in temps if np.isfinite(x)]
        valid_rss = [x for x in rss_values if np.isfinite(x)]
        valid_cpu = [x for x in cpu_values if np.isfinite(x)]
        row = {
            "model": args.name,
            "repeat": repeat,
            "images_in_replay_set": len(images),
            "iterations": args.iterations,
            "threads": actual_threads,
            "imgsz": args.imgsz,
            "batch": 1,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            **summary(pre_values, "preprocess"),
            **summary(infer_values, "inference"),
            **summary(post_values, "postprocess"),
            **summary(e2e_values, "end_to_end"),
            "sustained_fps": 1000.0 / statistics.fmean(e2e_values),
            "mean_detections": statistics.fmean(detections),
            "peak_rss_mb": max(valid_rss),
            "rss_increase_mb": max(valid_rss) - start_rss,
            "mean_system_cpu_percent": statistics.fmean(valid_cpu) if valid_cpu else float("nan"),
            "start_temperature_c": start_temp,
            "max_temperature_c": max(valid_temps),
            "end_temperature_c": valid_temps[-1],
        }
        repeat_rows.append(row)
        print(
            f"Repeat {repeat}: inference={row['inference_mean_ms']:.2f} ms, "
            f"end-to-end={row['end_to_end_mean_ms']:.2f} ms, "
            f"FPS={row['sustained_fps']:.2f}, max temp={row['max_temperature_c']:.1f} C"
        )

    overall = dict(repeat_rows[0])
    overall["repeat"] = "overall"
    numeric_keys = [
        key
        for key, value in repeat_rows[0].items()
        if key not in {"repeat"} and isinstance(value, (int, float))
    ]
    for key in numeric_keys:
        overall[key] = statistics.fmean(float(row[key]) for row in repeat_rows)
    repeat_rows.append(overall)

    detail_path = output_path / f"{args.name}_detail.csv"
    summary_path = output_path / f"{args.name}_summary.csv"
    write_csv(detail_path, detail_rows)
    write_csv(summary_path, repeat_rows)
    print(f"Saved detail CSV: {detail_path}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved environment JSON: {env_path}")


if __name__ == "__main__":
    main()

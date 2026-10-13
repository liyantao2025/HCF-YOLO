#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate speed and accuracy for multiple YOLO models in one CSV table."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_inference_stable import (  # noqa: E402
    bench_case,
    model_static_stats,
    parse_model_specs,
    run_val,
)


def fmt(value) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified speed and accuracy evaluation for paper tables.")
    parser.add_argument("--model", action="append", default=[], help="Model spec as name=path. Can be repeated.")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--speed-batch", type=int, default=1)
    parser.add_argument("--val-batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--metrics-project", default="runs/benchmark_metrics")
    parser.add_argument("--metrics-name-prefix", default="eval_speed_accuracy")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--out", default="outputs/eval_speed_accuracy.csv")
    args = parser.parse_args()

    if not args.model:
        raise SystemExit("At least one --model name=path is required.")
    if args.warmup < 100 or args.iters < 500 or args.repeats < 10:
        raise SystemExit("Use warmup>=100, iters>=500, repeats>=10 for stable reporting.")

    bench_args = argparse.Namespace(
        imgsz=args.imgsz,
        batch=args.speed_batch,
        device=args.device,
        half=args.half,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        data=args.data,
        split=args.split,
        val_batch=args.val_batch,
        save_json=args.save_json,
        metrics_project=args.metrics_project,
        metrics_name_prefix=args.metrics_name_prefix,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )

    rows = []
    for model_name, spec in parse_model_specs(args.model):
        print(f"\n== {model_name}: {spec} ==")
        try:
            static = model_static_stats(spec, bench_args)
            raw_row, _ = bench_case(model_name, spec, "raw_forward", False, False, False, None, bench_args)
            fused_row, _ = bench_case(model_name, spec, "fused_forward", True, False, False, None, bench_args)
            decode_row, _ = bench_case(model_name, spec, "fused_forward_decode", True, True, False, None, bench_args)
        except Exception as exc:
            print(f"WARNING: speed failed for {model_name}: {exc}")
            static = {"params_m": math.nan, "flops_g": math.nan, "output_points": "", "output_shape": ""}
            raw_row = fused_row = decode_row = None
        try:
            metrics = run_val(model_name, spec, bench_args)
        except Exception as exc:
            print(f"WARNING: val failed for {model_name}: {exc}")
            metrics = {"precision": math.nan, "recall": math.nan, "map50": math.nan, "map50_95": math.nan}

        row = {
            "model": model_name,
            "weight": spec,
            "params_m": static["params_m"],
            "flops_g": static["flops_g"],
            "raw_forward_ms": math.nan if raw_row is None else raw_row["median_ms"],
            "fused_forward_ms": math.nan if fused_row is None else fused_row["median_ms"],
            "fused_forward_decode_ms": math.nan if decode_row is None else decode_row["median_ms"],
            "fps": math.nan if decode_row is None else decode_row["fps_by_median"],
            "output_points": static["output_points"],
            "output_shape": static["output_shape"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
            "split": args.split,
            "imgsz": args.imgsz,
            "speed_batch": args.speed_batch,
            "val_batch": args.val_batch,
        }
        rows.append(row)
        print(
            f"{model_name:24s} P={fmt(row['precision'])} R={fmt(row['recall'])} "
            f"mAP50={fmt(row['map50'])} mAP50-95={fmt(row['map50_95'])} "
            f"fused+decode={fmt(row['fused_forward_decode_ms'])}ms"
        )

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

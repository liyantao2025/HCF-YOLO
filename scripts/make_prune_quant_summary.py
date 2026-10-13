#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build one summary table for pruning and quantization comparisons.

The table covers YOLOv11n baseline and Final E+GCG under:
- original PyTorch checkpoint,
- BN-gamma 5% pruning,
- BN-gamma 10% pruning,
- optional TensorRT FP16/INT8 quantized artifacts from quantize_export_eval.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalSpec:
    key: str
    family: str
    variant: str
    pruning_rate: str
    path: str


PYTORCH_SPECS = [
    EvalSpec(
        "YOLOv11n_original",
        "YOLOv11n baseline",
        "Original",
        "0%",
        "runs/ablation/yolo11n_baseline/weights/best.pt",
    ),
    EvalSpec(
        "YOLOv11n_BN5",
        "YOLOv11n baseline",
        "BN-gamma 5%",
        "5%",
        "runs/prune/classic_bn_gamma_yolo11n_baseline_ratio005/weights/best.pt",
    ),
    EvalSpec(
        "YOLOv11n_BN10",
        "YOLOv11n baseline",
        "BN-gamma 10%",
        "10%",
        "runs/prune/classic_bn_gamma_yolo11n_baseline_ratio010/weights/best.pt",
    ),
    EvalSpec(
        "Final_EGCG_original",
        "Final E+GCG",
        "Original",
        "0%",
        "runs/ablation/diagnose_E_plus_GCG_only/weights/best.pt",
    ),
    EvalSpec(
        "Final_EGCG_BN5",
        "Final E+GCG",
        "BN-gamma 5%",
        "5%",
        "runs/prune/classic_bn_gamma_final_EGCG_ratio005/weights/best.pt",
    ),
    EvalSpec(
        "Final_EGCG_BN10",
        "Final E+GCG",
        "BN-gamma 10%",
        "10%",
        "runs/prune/classic_bn_gamma_final_EGCG_ratio010/weights/best.pt",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-pytorch-eval", action="store_true", help="Run eval_speed_accuracy.py for PyTorch rows.")
    parser.add_argument("--pytorch-csv", default="outputs/prune_quant_pytorch_eval.csv")
    parser.add_argument(
        "--quant-csv",
        action="append",
        default=[],
        help="CSV from scripts/quantize_export_eval.py. Can be repeated.",
    )
    parser.add_argument("--out", default="outputs/prune_quant_summary.csv")
    parser.add_argument("--markdown-out", default="outputs/prune_quant_summary.md")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--speed-batch", type=int, default=1)
    parser.add_argument("--val-batch", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--include-quant-pt", action="store_true", help="Keep PyTorch rows from quantization CSV too.")
    parser.add_argument("--strict", action="store_true", help="Raise when an expected checkpoint/CSV is missing.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def as_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: Any, ndigits: int = 4) -> str:
    number = as_float(value)
    if math.isnan(number):
        return ""
    return f"{number:.{ndigits}f}"


def available_pytorch_specs(strict: bool) -> list[EvalSpec]:
    specs: list[EvalSpec] = []
    for spec in PYTORCH_SPECS:
        path = resolve_path(spec.path)
        if path.exists():
            specs.append(spec)
        else:
            msg = f"[skip] missing checkpoint: {spec.key} -> {path}"
            if strict:
                raise FileNotFoundError(msg)
            print(msg)
    return specs


def run_pytorch_eval(args: argparse.Namespace) -> None:
    specs = available_pytorch_specs(args.strict)
    if not specs:
        raise SystemExit("No PyTorch checkpoints available for evaluation.")
    out = resolve_path(args.pytorch_csv)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_speed_accuracy.py"),
        "--data",
        args.data,
        "--split",
        args.split,
        "--imgsz",
        str(args.imgsz),
        "--speed-batch",
        str(args.speed_batch),
        "--val-batch",
        str(args.val_batch),
        "--device",
        str(args.device),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--repeats",
        str(args.repeats),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--max-det",
        str(args.max_det),
        "--out",
        str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),
    ]
    for spec in specs:
        cmd.extend(["--model", f"{spec.key}={spec.path}"])
    print("[eval] running PyTorch pruning comparison:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def spec_by_key() -> dict[str, EvalSpec]:
    return {spec.key: spec for spec in PYTORCH_SPECS}


def infer_spec(model_name: str) -> EvalSpec:
    key = model_name.strip()
    mapping = spec_by_key()
    if key in mapping:
        return mapping[key]

    lowered = key.lower()
    family = "Final E+GCG" if "final" in lowered or "egcg" in lowered else "YOLOv11n baseline"
    if "10" in lowered or "010" in lowered or "bn10" in lowered:
        return EvalSpec(key, family, "BN-gamma 10%", "10%", "")
    if "5" in lowered or "005" in lowered or "bn5" in lowered:
        return EvalSpec(key, family, "BN-gamma 5%", "5%", "")
    return EvalSpec(key, family, "Original", "0%", "")


def pytorch_rows(csv_path: Path, strict: bool) -> list[dict[str, Any]]:
    if not csv_path.exists():
        if strict:
            raise FileNotFoundError(csv_path)
        print(f"[skip] missing PyTorch eval CSV: {csv_path}")
        return []
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            spec = infer_spec(row.get("model", ""))
            latency = row.get("fused_forward_decode_ms", "")
            rows.append(
                {
                    "model_family": spec.family,
                    "variant": spec.variant,
                    "pruning_rate": spec.pruning_rate,
                    "backend": "PyTorch",
                    "precision_type": "FP32",
                    "params_M": fmt(row.get("params_m")),
                    "flops_G": fmt(row.get("flops_g")),
                    "latency_ms": fmt(latency),
                    "fps": fmt(row.get("fps"), 2),
                    "precision": fmt(row.get("precision")),
                    "recall": fmt(row.get("recall")),
                    "mAP50": fmt(row.get("map50")),
                    "mAP50_95": fmt(row.get("map50_95")),
                    "output_points": row.get("output_points", ""),
                    "artifact": row.get("weight", ""),
                }
            )
    return rows


def quant_rows(csv_paths: list[str], include_pt: bool, strict: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in csv_paths:
        csv_path = resolve_path(raw_path)
        if not csv_path.exists():
            if strict:
                raise FileNotFoundError(csv_path)
            print(f"[skip] missing quant CSV: {csv_path}")
            continue
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                backend = row.get("backend", "")
                precision_type = row.get("precision", "")
                if backend == "PyTorch" and not include_pt:
                    continue
                spec = infer_spec(row.get("model", ""))
                total_ms = row.get("total_ms", "")
                if not total_ms:
                    total_ms = row.get("latency_ms", "")
                rows.append(
                    {
                        "model_family": spec.family,
                        "variant": spec.variant,
                        "pruning_rate": spec.pruning_rate,
                        "backend": backend,
                        "precision_type": precision_type,
                        "params_M": "",
                        "flops_G": "",
                        "latency_ms": fmt(total_ms),
                        "fps": fmt(row.get("fps"), 2),
                        "precision": fmt(row.get("precision_metric", row.get("precision", ""))),
                        "recall": fmt(row.get("recall")),
                        "mAP50": fmt(row.get("mAP50", row.get("map50", ""))),
                        "mAP50_95": fmt(row.get("mAP50_95", row.get("map50_95", ""))),
                        "output_points": "",
                        "artifact": row.get("artifact", ""),
                    }
                )
    return rows


def sort_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    family_order = {"YOLOv11n baseline": 0, "Final E+GCG": 1}
    prune_order = {"0%": 0, "5%": 1, "10%": 2}
    backend_order = {"PyTorch": 0, "TensorRT": 1, "ONNXRuntime": 2}
    precision_order = {"FP32": 0, "FP16": 1, "INT8": 2}
    return (
        family_order.get(row["model_family"], 9),
        prune_order.get(row["pruning_rate"], 9),
        backend_order.get(row["backend"], 9),
        precision_order.get(row["precision_type"], 9),
    )


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        raise RuntimeError("No rows to write.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_family",
        "variant",
        "pruning_rate",
        "backend",
        "precision_type",
        "params_M",
        "flops_G",
        "latency_ms",
        "fps",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "output_points",
        "artifact",
    ]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] saved CSV: {out_path}")


def write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    fields = [
        "model_family",
        "variant",
        "pruning_rate",
        "backend",
        "precision_type",
        "params_M",
        "flops_G",
        "latency_ms",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
    ]
    labels = {
        "model_family": "Model",
        "variant": "Variant",
        "pruning_rate": "Prune",
        "backend": "Backend",
        "precision_type": "Precision",
        "params_M": "Params/M",
        "flops_G": "FLOPs/G",
        "latency_ms": "Latency/ms",
        "precision": "P",
        "recall": "R",
        "mAP50": "mAP50",
        "mAP50_95": "mAP50-95",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(labels[f] for f in fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] saved Markdown: {out_path}")


def main() -> None:
    args = parse_args()
    if args.run_pytorch_eval:
        run_pytorch_eval(args)
    rows = pytorch_rows(resolve_path(args.pytorch_csv), args.strict)
    rows.extend(quant_rows(args.quant_csv, args.include_quant_pt, args.strict))
    rows.sort(key=sort_key)
    write_csv(rows, resolve_path(args.out))
    write_markdown(rows, resolve_path(args.markdown_out))


if __name__ == "__main__":
    main()

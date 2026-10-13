#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable inference latency benchmark without changing model outputs.

Measures the same model under three paths:
- raw_forward: unfused model, Detect head returns raw logits, no decode.
- fused_forward: fused model, Detect head returns raw logits, no decode.
- fused_forward_decode: fused model, normal eval output with DFL decode.

Optional channels_last and torch.compile variants are output-equivalence checks only.
They do not change weights, validation thresholds, max_det, or the number of decoded points.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import math
import statistics
import sys
import time
import tempfile
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.gettempdir())

if importlib.util.find_spec("cv2") is None:
    cv2 = types.ModuleType("cv2")
    cv2.setNumThreads = lambda *a, **k: None
    cv2.imshow = lambda *a, **k: None
    cv2.imwrite = lambda *a, **k: True
    cv2.imread = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: -1
    cv2.destroyAllWindows = lambda *a, **k: None
    for _name, _value in {
        "IMREAD_COLOR": 1,
        "IMREAD_GRAYSCALE": 0,
        "IMREAD_UNCHANGED": -1,
        "INTER_LINEAR": 1,
        "INTER_NEAREST": 0,
        "INTER_AREA": 3,
        "INTER_CUBIC": 2,
        "BORDER_CONSTANT": 0,
        "BORDER_REPLICATE": 1,
        "FONT_HERSHEY_SIMPLEX": 0,
        "CAP_PROP_FRAME_COUNT": 7,
        "CAP_PROP_FPS": 5,
        "CAP_PROP_FRAME_WIDTH": 3,
        "CAP_PROP_FRAME_HEIGHT": 4,
    }.items():
        setattr(cv2, _name, _value)
    cv2.VideoCapture = lambda *a, **k: None
    sys.modules["cv2"] = cv2

from ultralytics import YOLO  # noqa: E402


def parse_model_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).stem
        specs.append((name.strip(), path.strip()))
    if not specs:
        raise ValueError("At least one --model name=path must be provided.")
    return specs


def torch_device(device: str) -> torch.device:
    text = str(device).strip().lower()
    if text == "cpu":
        return torch.device("cpu")
    if text.isdigit():
        return torch.device(f"cuda:{text}" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def find_head(model: nn.Module) -> nn.Module:
    head = model.model[-1]
    if not hasattr(head, "nl") or not hasattr(head, "reg_max"):
        raise RuntimeError(f"Last module does not look like a Detect head: {type(head).__name__}")
    return head


def set_head_decode(model: nn.Module, decode: bool) -> None:
    """Toggle only the Detect head decode branch while keeping all Conv/BN modules in eval mode."""
    model.eval()
    head = find_head(model)
    head.training = not decode
    head.export = False
    if hasattr(head, "center_preds"):
        head.center_preds = None


def output_signature(output) -> str:
    if isinstance(output, torch.Tensor):
        return "Tensor" + str(tuple(output.shape))
    if isinstance(output, (list, tuple)):
        return "[" + ",".join(output_signature(x) for x in output) + "]"
    if isinstance(output, dict):
        return "{" + ",".join(f"{k}:{output_signature(v)}" for k, v in output.items()) + "}"
    return type(output).__name__


def flatten_tensors(output) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (list, tuple)):
        tensors: list[torch.Tensor] = []
        for item in output:
            tensors.extend(flatten_tensors(item))
        return tensors
    if isinstance(output, dict):
        tensors = []
        for item in output.values():
            tensors.extend(flatten_tensors(item))
        return tensors
    return []


@torch.inference_mode()
def run_once(model: nn.Module, x: torch.Tensor):
    return model(x)


def compare_outputs(ref, out) -> tuple[bool, float, float]:
    ref_tensors = flatten_tensors(ref)
    out_tensors = flatten_tensors(out)
    if len(ref_tensors) != len(out_tensors):
        return False, math.inf, math.inf
    max_abs = 0.0
    mean_abs = 0.0
    elems = 0
    for a, b in zip(ref_tensors, out_tensors):
        if tuple(a.shape) != tuple(b.shape):
            return False, math.inf, math.inf
        diff = (a.float() - b.float()).abs()
        max_abs = max(max_abs, float(diff.max().item()) if diff.numel() else 0.0)
        mean_abs += float(diff.sum().item())
        elems += diff.numel()
    return True, max_abs, mean_abs / max(elems, 1)


def count_bn(model: nn.Module) -> int:
    return sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())


def load_model(
    spec: str,
    device: torch.device,
    fused: bool,
    decode: bool,
    half: bool,
    channels_last: bool,
    compile_mode: str | None,
) -> tuple[YOLO, nn.Module]:
    yolo = YOLO(spec)
    if fused:
        yolo.fuse()
    model = yolo.model.to(device).eval()
    if half:
        model.half()
    if channels_last:
        model.to(memory_format=torch.channels_last)
    set_head_decode(model, decode=decode)
    if compile_mode:
        model = torch.compile(model, mode=compile_mode, backend="inductor")
    return yolo, model


def make_input(device: torch.device, imgsz: int, batch: int, half: bool, channels_last: bool) -> torch.Tensor:
    dtype = torch.float16 if half else torch.float32
    x = torch.zeros(batch, 3, imgsz, imgsz, device=device, dtype=dtype)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


@torch.inference_mode()
def time_model(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    repeats: int,
) -> tuple[list[float], list[float]]:
    repeat_medians: list[float] = []
    all_times: list[float] = []
    use_cuda_events = x.device.type == "cuda"

    for _ in range(repeats):
        for _ in range(warmup):
            _ = model(x)
        if use_cuda_events:
            torch.cuda.synchronize()

        times: list[float] = []
        for _ in range(iters):
            if use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(x)
                end.record()
                torch.cuda.synchronize()
                dt = float(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(x)
                dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt / x.shape[0])

        repeat_medians.append(statistics.median(times))
        all_times.extend(times)

    return repeat_medians, all_times


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values), 6),
        "median_ms": round(statistics.median(values), 6),
        "min_ms": round(min(values), 6),
        "max_ms": round(max(values), 6),
        "std_ms": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
    }


def output_points(head: nn.Module, imgsz: int) -> int:
    total = 0
    for stride in getattr(head, "stride", torch.empty(0)).tolist():
        if stride:
            total += int(imgsz / float(stride)) ** 2
    return total


def conv_flops(model: nn.Module, imgsz: int, batch: int, device: torch.device, half: bool) -> float:
    total = 0
    hooks = []

    def hook(m, _inp, out):
        nonlocal total
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            b, oc, oh, ow = out.shape
            kh, kw = m.kernel_size
            total += b * oc * oh * ow * (m.in_channels // m.groups) * kh * kw * 2

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook))
    x = make_input(device, imgsz, batch, half, False)
    with torch.inference_mode():
        _ = model(x)
    for handle in hooks:
        handle.remove()
    return float(total) / 1e9


def model_static_stats(spec: str, args: argparse.Namespace) -> dict:
    device = torch_device(args.device)
    yolo, model = load_model(spec, device, True, True, args.half, False, None)
    head = find_head(model)
    x = make_input(device, args.imgsz, args.batch, args.half, False)
    with torch.inference_mode():
        out = model(x)
    return {
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 6),
        "flops_g": round(conv_flops(model, args.imgsz, args.batch, device, args.half), 6),
        "output_points": output_points(head, args.imgsz),
        "output_shape": output_signature(out),
        "reg_max": int(getattr(head, "reg_max", 0)),
    }


def bench_case(
    model_name: str,
    spec: str,
    case_name: str,
    fused: bool,
    decode: bool,
    channels_last: bool,
    compile_mode: str | None,
    args: argparse.Namespace,
    ref_output=None,
) -> tuple[dict, object]:
    device = torch_device(args.device)
    _, model = load_model(spec, device, fused, decode, args.half, channels_last, compile_mode)
    x = make_input(device, args.imgsz, args.batch, args.half, channels_last)

    output = run_once(model, x)
    sig = output_signature(output)
    shape_ok, max_abs, mean_abs = (True, 0.0, 0.0) if ref_output is None else compare_outputs(ref_output, output)
    head = find_head(model._orig_mod if hasattr(model, "_orig_mod") else model)
    bn_remaining = count_bn(model._orig_mod if hasattr(model, "_orig_mod") else model)

    repeat_medians, all_times = time_model(model, x, args.warmup, args.iters, args.repeats)
    stats = summarize(repeat_medians)
    sample_stats = summarize(all_times)
    row = {
        "model": model_name,
        "spec": spec,
        "case": case_name,
        "mode": "decode" if decode else "raw_logits",
        "fused": fused,
        "channels_last": channels_last,
        "compile_mode": compile_mode or "",
        "half": args.half,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "head": type(head).__name__,
        "reg_max": int(getattr(head, "reg_max", 0)),
        "output_points": output_points(head, args.imgsz) if decode else "",
        "output_signature": sig,
        "shape_equal_to_ref": shape_ok,
        "max_abs_diff_to_ref": max_abs,
        "mean_abs_diff_to_ref": mean_abs,
        "bn_remaining_after_fuse": bn_remaining if fused else "",
        "sample_mean_ms": sample_stats["mean_ms"],
        "sample_median_ms": sample_stats["median_ms"],
        "sample_min_ms": sample_stats["min_ms"],
        "sample_max_ms": sample_stats["max_ms"],
        "sample_std_ms": sample_stats["std_ms"],
        **stats,
        "fps_by_median": round(1000.0 / stats["median_ms"], 3) if stats["median_ms"] else 0.0,
    }
    return row, output


def metric_value(metrics, names: tuple[str, ...]) -> float:
    box = getattr(metrics, "box", None)
    for name in names:
        if box is not None and hasattr(box, name):
            value = getattr(box, name)
            return float(value() if callable(value) else value)
    rd = getattr(metrics, "results_dict", {}) or {}
    for name in names:
        if name in rd:
            return float(rd.get(name) or 0.0)
    return float("nan")


def run_val(model_name: str, spec: str, args: argparse.Namespace) -> dict[str, float]:
    yolo = YOLO(spec)
    metrics = yolo.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.val_batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        split=args.split,
        plots=False,
        verbose=False,
        save_json=args.save_json,
        project=args.metrics_project,
        name=f"{args.metrics_name_prefix}_{model_name}",
        exist_ok=True,
    )
    return {
        "precision": metric_value(metrics, ("mp", "metrics/precision(B)")),
        "recall": metric_value(metrics, ("mr", "metrics/recall(B)")),
        "map50": metric_value(metrics, ("map50", "metrics/mAP50(B)")),
        "map50_95": metric_value(metrics, ("map", "metrics/mAP50-95(B)")),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Stable forward/decode benchmark for fixed-output YOLO models.")
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec as name=path. Pass once for baseline and once for sharedhead.",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true")
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--channels-last", action="store_true", help="Add channels_last fused decode comparison.")
    p.add_argument("--compile", dest="compile_mode", default="", help="Add torch.compile mode, e.g. default or reduce-overhead.")
    p.add_argument("--out", default="outputs/stable_inference_latency.csv")
    p.add_argument("--run-val", action="store_true", help="Deprecated alias for --eval-metrics.")
    p.add_argument("--eval-metrics", action="store_true", help="Run model.val() after speed benchmarking and write a combined CSV.")
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--val-batch", type=int, default=8)
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--metrics-project", default="runs/benchmark_metrics")
    p.add_argument("--metrics-name-prefix", default="benchmark")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=1000)
    args = p.parse_args()

    if args.warmup < 100 or args.iters < 500 or args.repeats < 10:
        raise SystemExit("Use warmup>=100, iters>=500, repeats>=10 for stable reporting.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    rows: list[dict] = []
    summary_rows: list[dict] = []
    eval_metrics = bool(args.eval_metrics or args.run_val)

    for model_name, spec in parse_model_specs(args.model):
        print(f"\n== {model_name}: {spec} ==")
        try:
            raw_row, _ = bench_case(model_name, spec, "raw_forward", False, False, False, None, args)
            rows.append(raw_row)
            print(f"raw_forward median={raw_row['median_ms']:.4f} ms")

            fused_raw_row, _ = bench_case(model_name, spec, "fused_forward", True, False, False, None, args)
            rows.append(fused_raw_row)
            print(f"fused_forward median={fused_raw_row['median_ms']:.4f} ms")

            fused_decode_row, ref_output = bench_case(
                model_name, spec, "fused_forward_decode", True, True, False, None, args
            )
            rows.append(fused_decode_row)
            print(
                f"fused_forward_decode median={fused_decode_row['median_ms']:.4f} ms "
                f"shape={fused_decode_row['output_signature']}"
            )
            speed_failed = False
        except Exception as exc:
            print(f"WARNING: speed benchmark failed for {model_name}: {exc}")
            raw_row = fused_raw_row = fused_decode_row = None
            ref_output = None
            speed_failed = True

        if args.channels_last and not speed_failed:
            row, _ = bench_case(
                model_name,
                spec,
                "fused_forward_decode_channels_last",
                True,
                True,
                True,
                None,
                args,
                ref_output=ref_output,
            )
            rows.append(row)
            print(
                f"channels_last median={row['median_ms']:.4f} ms "
                f"shape_ok={row['shape_equal_to_ref']} max_abs={row['max_abs_diff_to_ref']:.3g}"
            )

        if args.compile_mode and not speed_failed:
            row, _ = bench_case(
                model_name,
                spec,
                f"fused_forward_decode_compile_{args.compile_mode}",
                True,
                True,
                False,
                args.compile_mode,
                args,
                ref_output=ref_output,
            )
            rows.append(row)
            print(
                f"compile({args.compile_mode}) median={row['median_ms']:.4f} ms "
                f"shape_ok={row['shape_equal_to_ref']} max_abs={row['max_abs_diff_to_ref']:.3g}"
            )

        if eval_metrics:
            try:
                static = model_static_stats(spec, args)
            except Exception as exc:
                print(f"WARNING: static stats failed for {model_name}: {exc}")
                static = {
                    "params_m": float("nan"),
                    "flops_g": float("nan"),
                    "output_points": "",
                    "output_shape": "",
                    "reg_max": "",
                }
            try:
                val = run_val(model_name, spec, args)
                print(
                    f"val P={val['precision']:.4f} R={val['recall']:.4f} "
                    f"mAP50={val['map50']:.4f} mAP50-95={val['map50_95']:.4f}"
                )
            except Exception as exc:
                print(f"WARNING: val failed for {model_name}: {exc}")
                val = {"precision": float("nan"), "recall": float("nan"), "map50": float("nan"), "map50_95": float("nan")}
            summary_rows.append(
                {
                    "model": model_name,
                    "weight": spec,
                    "params_m": static["params_m"],
                    "flops_g": static["flops_g"],
                    "raw_forward_ms": float("nan") if raw_row is None else raw_row["median_ms"],
                    "fused_forward_ms": float("nan") if fused_raw_row is None else fused_raw_row["median_ms"],
                    "fused_forward_decode_ms": float("nan") if fused_decode_row is None else fused_decode_row["median_ms"],
                    "fps": float("nan") if fused_decode_row is None else fused_decode_row["fps_by_median"],
                    "output_points": static["output_points"],
                    "output_shape": static["output_shape"],
                    "precision": val["precision"],
                    "recall": val["recall"],
                    "map50": val["map50"],
                    "map50_95": val["map50_95"],
                    "split": args.split,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "val_batch": args.val_batch,
                }
            )

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        write_rows = summary_rows if eval_metrics else rows
        writer = csv.DictWriter(f, fieldnames=list(write_rows[0].keys()))
        writer.writeheader()
        writer.writerows(write_rows)
    print(f"\nSaved {'speed+metrics' if eval_metrics else 'latency'} CSV: {out}")


if __name__ == "__main__":
    main()

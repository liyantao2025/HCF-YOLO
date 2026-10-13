#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export and evaluate deployment quantization variants for paper models.

The script focuses on deployment quantization:
- TensorRT FP16 engine export.
- TensorRT INT8 PTQ engine export with calibration data.
- Optional ONNX export for smoke testing.

It does not change validation thresholds, max_det, model structure, or weights.
TensorRT engines are hardware/TensorRT-version specific and should be built on
the same AutoDL/GPU environment used for deployment benchmarking.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs" / "ultralytics_config"))

YOLO = None


DEFAULT_MODELS = {
    "A_yolo11n": "runs/ablation/yolo11n_baseline/weights/best.pt",
    "A_classicBN005": "runs/prune/classic_bn_gamma_yolo11n_baseline_ratio005/weights/best.pt",
    "A_classicBN010": "runs/prune/classic_bn_gamma_yolo11n_baseline_ratio010/weights/best.pt",
    "Final_EGCG": "runs/ablation/diagnose_E_plus_GCG_only/weights/best.pt",
    "Final_EGCG_classicBN005": "runs/prune/classic_bn_gamma_final_EGCG_ratio005/weights/best.pt",
    "Final_EGCG_classicBN010": "runs/prune/classic_bn_gamma_final_EGCG_ratio010/weights/best.pt",
}

FORMAT_ALIASES = {
    "fp16": "engine_fp16",
    "fp16_engine": "engine_fp16",
    "engine_fp16": "engine_fp16",
    "trt_fp16": "engine_fp16",
    "int8": "engine_int8",
    "int8_engine": "engine_int8",
    "engine_int8": "engine_int8",
    "trt_int8": "engine_int8",
    "onnx": "onnx",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class ArtifactSpec:
    model_name: str
    source_path: Path
    backend: str
    precision: str
    artifact_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("export", "eval", "all"), default="all")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec name=path. If omitted, uses baseline/final/pruned defaults when files exist.",
    )
    parser.add_argument(
        "--formats",
        default="fp16,int8",
        help="Comma-separated export formats: fp16,int8,onnx. Defaults to TensorRT FP16 and INT8.",
    )
    parser.add_argument("--out-dir", default="outputs/quantized")
    parser.add_argument("--csv", default="outputs/quantization_compare.csv")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--export-batch", type=int, default=1, help="Static export batch. Keep 1 for latency tables.")
    parser.add_argument("--val-batch", type=int, default=1, help="Use 1 for static TensorRT engines.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace in GiB.")
    parser.add_argument("--fraction", type=float, default=1.0, help="INT8 calibration data fraction.")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--no-simplify", dest="simplify", action="store_false")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--no-pt", action="store_true", help="Do not include original PyTorch checkpoints in eval CSV.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing exported artifacts.")
    parser.add_argument("--strict", action="store_true", help="Raise on first export/eval failure.")
    return parser.parse_args()


def load_runtime_dependencies() -> None:
    global YOLO
    if YOLO is not None:
        return
    if importlib.util.find_spec("cv2") is None:
        raise SystemExit("OpenCV is required by Ultralytics. Install it with: pip install opencv-python-headless")
    try:
        from ultralytics import YOLO as yolo_cls
    except ModuleNotFoundError as exc:
        raise SystemExit("Ultralytics/PyTorch environment is not active. Activate yolov11-cotton first.") from exc
    YOLO = yolo_cls


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())


def parse_models(values: list[str]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if values:
        for value in values:
            if "=" in value:
                name, path = value.split("=", 1)
            else:
                path = value
                name = Path(path).stem
            specs.append(ModelSpec(safe_name(name), resolve_path(path)))
    else:
        for name, path in DEFAULT_MODELS.items():
            p = resolve_path(path)
            if p.exists():
                specs.append(ModelSpec(name, p))
            else:
                print(f"[model] skip missing default: {name} -> {p}")
    if not specs:
        raise SystemExit("No models to process. Pass --model name=path or create the default checkpoints first.")
    return specs


def parse_formats(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in FORMAT_ALIASES:
            allowed = ", ".join(sorted(FORMAT_ALIASES))
            raise SystemExit(f"Unsupported format {raw!r}. Allowed: {allowed}")
        fmt = FORMAT_ALIASES[key]
        if fmt not in out:
            out.append(fmt)
    if not out:
        raise SystemExit("No export formats selected.")
    return out


def artifact_path(out_dir: Path, model_name: str, fmt: str) -> Path:
    suffix = ".onnx" if fmt == "onnx" else ".engine"
    return out_dir / f"{safe_name(model_name)}_{fmt}{suffix}"


def backend_precision(fmt: str) -> tuple[str, str]:
    if fmt == "engine_fp16":
        return "TensorRT", "FP16"
    if fmt == "engine_int8":
        return "TensorRT", "INT8"
    if fmt == "onnx":
        return "ONNXRuntime", "FP32"
    raise ValueError(fmt)


def export_one(model: ModelSpec, fmt: str, args: argparse.Namespace) -> ArtifactSpec:
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = artifact_path(out_dir, model.name, fmt)
    backend, precision = backend_precision(fmt)
    if dst.exists() and args.skip_existing:
        print(f"[export] reuse existing {backend} {precision}: {dst}")
        return ArtifactSpec(model.name, model.path, backend, precision, dst)

    print(f"[export] {model.name}: {model.path}")
    print(f"[export] -> {backend} {precision}")
    yolo = YOLO(str(model.path))
    common = {
        "imgsz": args.imgsz,
        "batch": args.export_batch,
        "device": str(args.device),
        "simplify": bool(args.simplify),
        "nms": False,
    }
    if fmt == "onnx":
        exported = yolo.export(format="onnx", opset=args.opset, dynamic=False, **common)
    elif fmt == "engine_fp16":
        exported = yolo.export(format="engine", half=True, int8=False, workspace=args.workspace, **common)
    elif fmt == "engine_int8":
        exported = yolo.export(
            format="engine",
            half=False,
            int8=True,
            data=str(resolve_path(args.data)),
            fraction=float(args.fraction),
            workspace=args.workspace,
            **common,
        )
    else:
        raise ValueError(fmt)

    exported_path = resolve_path(exported)
    if not exported_path.exists():
        raise FileNotFoundError(f"Export finished but artifact is missing: {exported_path}")
    if exported_path.resolve() != dst.resolve():
        shutil.copy2(exported_path, dst)
    print(f"[export] saved copy: {dst}")
    return ArtifactSpec(model.name, model.path, backend, precision, dst)


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


def speed_value(metrics, key: str) -> float:
    speed = getattr(metrics, "speed", {}) or {}
    try:
        return float(speed.get(key, float("nan")))
    except AttributeError:
        return float("nan")


def eval_one(
    model_name: str,
    artifact_path_: Path,
    backend: str,
    precision: str,
    source_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"[eval] {model_name} {backend} {precision}: {artifact_path_}")
    yolo = YOLO(str(artifact_path_))
    metrics = yolo.val(
        data=str(resolve_path(args.data)),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.val_batch,
        device=str(args.device),
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        plots=False,
        verbose=False,
        project=str(ROOT / "runs" / "quant_eval"),
        name=f"{safe_name(model_name)}_{precision.lower()}",
        exist_ok=True,
    )
    preprocess = speed_value(metrics, "preprocess")
    inference = speed_value(metrics, "inference")
    postprocess = speed_value(metrics, "postprocess")
    total = preprocess + inference + postprocess
    fps = 1000.0 / total if total > 0 else float("nan")
    return {
        "model": model_name,
        "backend": backend,
        "precision": precision,
        "artifact": str(artifact_path_),
        "source_pt": str(source_path),
        "precision_metric": metric_value(metrics, ("mp", "metrics/precision(B)")),
        "recall": metric_value(metrics, ("mr", "metrics/recall(B)")),
        "mAP50": metric_value(metrics, ("map50", "metrics/mAP50(B)")),
        "mAP50_95": metric_value(metrics, ("map", "metrics/mAP50-95(B)")),
        "preprocess_ms": preprocess,
        "inference_ms": inference,
        "postprocess_ms": postprocess,
        "total_ms": total,
        "fps": fps,
        "imgsz": args.imgsz,
        "batch": args.val_batch,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
    }


def collect_artifacts(models: list[ModelSpec], formats: list[str], args: argparse.Namespace) -> list[ArtifactSpec]:
    artifacts: list[ArtifactSpec] = []
    if not args.no_pt:
        artifacts.extend(ArtifactSpec(m.name, m.path, "PyTorch", "FP32", m.path) for m in models)
    for model in models:
        for fmt in formats:
            backend, precision = backend_precision(fmt)
            path = artifact_path(resolve_path(args.out_dir), model.name, fmt)
            if path.exists():
                artifacts.append(ArtifactSpec(model.name, model.path, backend, precision, path))
            else:
                print(f"[eval] skip missing artifact: {path}")
    return artifacts


def write_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        raise RuntimeError("No evaluation rows were produced.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] saved: {out_path}")


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    models = parse_models(args.model)
    formats = parse_formats(args.formats)

    exported: list[ArtifactSpec] = []
    if args.stage in {"export", "all"}:
        for model in models:
            for fmt in formats:
                try:
                    exported.append(export_one(model, fmt, args))
                except Exception as exc:
                    print(f"[export] FAILED {model.name} {fmt}: {exc}")
                    if args.strict:
                        raise

    if args.stage in {"eval", "all"}:
        artifacts = exported if args.stage == "all" else collect_artifacts(models, formats, args)
        if not args.no_pt:
            artifacts = [ArtifactSpec(m.name, m.path, "PyTorch", "FP32", m.path) for m in models] + artifacts
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for artifact in artifacts:
            key = (artifact.model_name, artifact.backend, artifact.precision)
            if key in seen:
                continue
            seen.add(key)
            try:
                rows.append(
                    eval_one(
                        artifact.model_name,
                        artifact.artifact_path,
                        artifact.backend,
                        artifact.precision,
                        artifact.source_path,
                        args,
                    )
                )
            except Exception as exc:
                print(f"[eval] FAILED {artifact.model_name} {artifact.backend} {artifact.precision}: {exc}")
                if args.strict:
                    raise
        write_rows(rows, resolve_path(args.csv))


if __name__ == "__main__":
    main()

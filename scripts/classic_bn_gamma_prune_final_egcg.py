#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classic BN-gamma pruning for Final E+GCG and YOLOv11n baseline.

This script intentionally avoids torch-pruning. It supports the classic
BN-gamma sparse-training workflow and then uses the learned BN statistics to
generate a non-uniform channel-pruned YAML. Final E+GCG uses a P2-protected
policy, while the YOLOv11n baseline uses a shallow-backbone protected policy.
The pruning stage keeps detection scales, reg_max, nc, stride, and output point
count unchanged for each target.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs" / "ultralytics_config"))

SOURCE_YAML = "models/yolo11n-drf-safpn-midlite-p2safe-r12-v1-gcg.yaml"
SOURCE_WEIGHTS = "runs/ablation/diagnose_E_plus_GCG_only/weights/best.pt"
SPARSE_EXP_NAME = "classic_bn_gamma_sparse_final_EGCG"
SPARSE_BEST = f"runs/prune/{SPARSE_EXP_NAME}/weights/best.pt"

BASELINE_SOURCE_YAML = "models/yolo11n.yaml"
BASELINE_SOURCE_WEIGHTS = "runs/ablation/yolo11n_baseline/weights/best.pt"
BASELINE_SPARSE_EXP_NAME = "classic_bn_gamma_sparse_yolo11n_baseline"

EXPECTED_POINTS = 34000
EXPECTED_STRIDE = (4, 8, 16, 32)
EXPECTED_SHAPE = (1, 5, EXPECTED_POINTS)

RATIO_TAGS = {
    0.05: "005",
    0.10: "010",
}

CHANNEL_MODULES = {
    "Conv",
    "C3k2",
    "LiteC3k2",
    "LightDRF_LSK",
    "FastSAF",
    "SPPF",
    "C2PSA",
}

# Protect the shallow P2 path and the direct P2 detection input.
P2_PROTECT_LAYERS = {0, 1, 2, 20}

# Keep nearby P2/P3 semantic layers less sparse and less aggressively pruned.
P2_WEAK_LAYERS = {3, 4, 16, 17, 21, 22, 23}

# Candidate high/mid-level layers where pruning is usually less destructive.
HIGH_VALUE_PRUNE_LAYERS = {5, 6, 7, 8, 9, 10, 13, 24, 25, 26, 27, 29}

# For the P3/P4/P5 baseline, protect shallow feature extraction and prune mostly
# in deeper backbone/neck layers.
BASELINE_PROTECT_LAYERS = {0, 1, 2}
BASELINE_WEAK_LAYERS = {3, 4, 16}
BASELINE_HIGH_VALUE_PRUNE_LAYERS = {5, 6, 7, 8, 9, 10, 13, 17, 19, 20, 22}

torch = None
nn = None
YOLO = None
yaml = None


@dataclass(frozen=True)
class ModelStats:
    params_m: float
    flops_g: float


@dataclass(frozen=True)
class BNLayerStat:
    name: str
    layer_index: int | None
    category: str
    num_channels: int
    mean_abs: float
    min_abs: float
    max_abs: float
    sparse_ratio: float
    recommendation: str


@dataclass(frozen=True)
class LayerSummary:
    layer_index: int
    mean_abs: float | None
    sparse_ratio: float | None
    num_bn_channels: int


@dataclass(frozen=True)
class TargetSpec:
    key: str
    display_name: str
    source_yaml: str
    source_weights: str
    sparse_exp_name: str
    yaml_prefix: str
    prune_exp_prefix: str
    importance_csv: str
    plan_prefix: str
    compare_csv: str
    compare_raw_csv: str
    expected_points: int
    expected_stride: tuple[int, ...]
    expected_reg_max: int
    protect_layers: frozenset[int]
    weak_layers: frozenset[int]
    high_value_layers: frozenset[int]
    use_center_guidance: bool
    protection_label: str

    @property
    def sparse_best(self) -> str:
        return f"runs/prune/{self.sparse_exp_name}/weights/best.pt"


TARGETS = {
    "final_egcg": TargetSpec(
        key="final_egcg",
        display_name="Final E+GCG",
        source_yaml=SOURCE_YAML,
        source_weights=SOURCE_WEIGHTS,
        sparse_exp_name=SPARSE_EXP_NAME,
        yaml_prefix="final-egcg-classic-bngamma",
        prune_exp_prefix="classic_bn_gamma_final_EGCG",
        importance_csv="classic_bn_gamma_importance_final_EGCG.csv",
        plan_prefix="classic_bn_gamma_prune_plan_final_EGCG",
        compare_csv="classic_prune_final_EGCG_compare.csv",
        compare_raw_csv="classic_prune_final_EGCG_compare_raw.csv",
        expected_points=EXPECTED_POINTS,
        expected_stride=EXPECTED_STRIDE,
        expected_reg_max=12,
        protect_layers=frozenset(P2_PROTECT_LAYERS),
        weak_layers=frozenset(P2_WEAK_LAYERS),
        high_value_layers=frozenset(HIGH_VALUE_PRUNE_LAYERS),
        use_center_guidance=True,
        protection_label="P2-protected",
    ),
    "baseline": TargetSpec(
        key="baseline",
        display_name="YOLOv11n baseline",
        source_yaml=BASELINE_SOURCE_YAML,
        source_weights=BASELINE_SOURCE_WEIGHTS,
        sparse_exp_name=BASELINE_SPARSE_EXP_NAME,
        yaml_prefix="yolo11n-classic-bngamma",
        prune_exp_prefix="classic_bn_gamma_yolo11n_baseline",
        importance_csv="classic_bn_gamma_importance_yolo11n_baseline.csv",
        plan_prefix="classic_bn_gamma_prune_plan_yolo11n_baseline",
        compare_csv="classic_prune_yolo11n_baseline_compare.csv",
        compare_raw_csv="classic_prune_yolo11n_baseline_compare_raw.csv",
        expected_points=8400,
        expected_stride=(8, 16, 32),
        expected_reg_max=16,
        protect_layers=frozenset(BASELINE_PROTECT_LAYERS),
        weak_layers=frozenset(BASELINE_WEAK_LAYERS),
        high_value_layers=frozenset(BASELINE_HIGH_VALUE_PRUNE_LAYERS),
        use_center_guidance=False,
        protection_label="shallow-backbone-protected",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("sparse_train", "analyze", "prune", "finetune", "eval"),
        help="Workflow stage to run.",
    )
    parser.add_argument(
        "--target",
        default="final_egcg",
        choices=tuple(TARGETS),
        help="Pruning target. Default keeps backward compatibility with the existing Final E+GCG workflow.",
    )
    parser.add_argument("--ratio", type=float, default=0.05, help="Pruning ratio for prune/finetune: 0.05 or 0.10.")
    parser.add_argument("--source-yaml", default=None)
    parser.add_argument("--source-weights", default=None)
    parser.add_argument("--sparse-weights", default=None)
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/prune")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sparsity", type=float, default=1e-4)
    parser.add_argument("--p2-sparse-scale", type=float, default=0.25)
    parser.add_argument("--gamma-thr", type=float, default=1e-3, help="Gamma threshold for sparse-ratio analysis.")
    parser.add_argument("--skip-val-smoke", action="store_true", help="Skip the post-prune validation smoke test.")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def load_runtime_dependencies() -> None:
    global YOLO, nn, torch, yaml
    if torch is not None and nn is not None and YOLO is not None and yaml is not None:
        return
    if importlib.util.find_spec("yaml") is None:
        raise SystemExit("PyYAML is required. Install it with: pip install pyyaml")
    try:
        import yaml as yaml_module
        import torch as torch_module
        from torch import nn as nn_module
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch and PyYAML are required. Activate the YOLO training environment first.") from exc
    try:
        from ultralytics import YOLO as yolo_cls
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            raise SystemExit("OpenCV is required by Ultralytics. Install it with: pip install opencv-python-headless") from exc
        raise

    yaml = yaml_module
    torch = torch_module
    nn = nn_module
    YOLO = yolo_cls


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def normalize_ratio(raw: float) -> tuple[float, str]:
    for ratio, tag in RATIO_TAGS.items():
        if abs(raw - ratio) < 1e-6:
            return ratio, tag
    allowed = ", ".join(str(x) for x in RATIO_TAGS)
    raise SystemExit(f"Unsupported --ratio {raw}. Use {allowed}.")


def get_target(args: argparse.Namespace) -> TargetSpec:
    return TARGETS[str(args.target)]


def target_source_yaml(args: argparse.Namespace, target: TargetSpec) -> str:
    return args.source_yaml or target.source_yaml


def target_source_weights(args: argparse.Namespace, target: TargetSpec) -> str:
    return args.source_weights or target.source_weights


def target_sparse_weights(args: argparse.Namespace, target: TargetSpec) -> str:
    return args.sparse_weights or target.sparse_best


def as_device_arg(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device.isdigit() and torch.cuda.is_available():
        return f"cuda:{device}"
    return "cpu"


def load_yaml_dict(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_dict(data: dict, path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    path.write_text(header.rstrip() + "\n\n" + body, encoding="utf-8")


def make_divisible_ceil(x: float, divisor: int = 8) -> int:
    return int(math.ceil(x / divisor) * divisor)


def align_down(x: float, divisor: int = 8) -> int:
    return int(math.floor(x / divisor) * divisor)


def parse_layer_index(module_name: str) -> int | None:
    match = re.search(r"(?:^|\.)model\.(\d+)(?:\.|$)", module_name)
    return int(match.group(1)) if match else None


def bn_category(name: str, layer_index: int | None, target: TargetSpec) -> str:
    lowered = name.lower()
    if "dfl" in lowered:
        return "protect_dfl"
    if layer_index in target.protect_layers:
        return "protect"
    if layer_index in target.weak_layers:
        return "weak"
    return "candidate"


def bn_sparse_coeff(
    name: str,
    layer_index: int | None,
    sparsity: float,
    p2_sparse_scale: float,
    target: TargetSpec,
) -> float:
    category = bn_category(name, layer_index, target)
    if category.startswith("protect"):
        return 0.0
    if category == "weak":
        return float(sparsity) * float(p2_sparse_scale)
    return float(sparsity)


def collect_bn_stats(model, gamma_thr: float, target: TargetSpec) -> list[BNLayerStat]:
    rows: list[BNLayerStat] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        gamma = module.weight.detach().abs().float().cpu()
        layer_index = parse_layer_index(name)
        category = bn_category(name, layer_index, target)
        sparse_ratio = float((gamma < gamma_thr).float().mean().item()) if gamma.numel() else 0.0
        if category.startswith("protect"):
            recommendation = "protect"
        elif category == "weak":
            recommendation = "weak_sparse_or_mild_prune"
        elif sparse_ratio >= 0.20:
            recommendation = "high_prune_candidate"
        else:
            recommendation = "candidate"
        rows.append(
            BNLayerStat(
                name=name,
                layer_index=layer_index,
                category=category,
                num_channels=int(gamma.numel()),
                mean_abs=float(gamma.mean().item()) if gamma.numel() else 0.0,
                min_abs=float(gamma.min().item()) if gamma.numel() else 0.0,
                max_abs=float(gamma.max().item()) if gamma.numel() else 0.0,
                sparse_ratio=sparse_ratio,
                recommendation=recommendation,
            )
        )
    return rows


def summarize_bn_by_layer(rows: list[BNLayerStat]) -> dict[int, LayerSummary]:
    grouped: dict[int, list[BNLayerStat]] = {}
    for row in rows:
        if row.layer_index is not None:
            grouped.setdefault(row.layer_index, []).append(row)
    summaries: dict[int, LayerSummary] = {}
    for layer_index, stats in grouped.items():
        total_ch = sum(s.num_channels for s in stats)
        if total_ch <= 0:
            summaries[layer_index] = LayerSummary(layer_index, None, None, 0)
            continue
        mean_abs = sum(s.mean_abs * s.num_channels for s in stats) / total_ch
        sparse_ratio = sum(s.sparse_ratio * s.num_channels for s in stats) / total_ch
        summaries[layer_index] = LayerSummary(layer_index, mean_abs, sparse_ratio, total_ch)
    return summaries


def candidate_global_mean(rows: list[BNLayerStat]) -> float:
    vals = [r.mean_abs for r in rows if r.category == "candidate" and r.num_channels > 0]
    return float(sum(vals) / len(vals)) if vals else 1.0


def print_bn_recommendations(rows: list[BNLayerStat], target: TargetSpec) -> None:
    protect = [r.name for r in rows if r.recommendation == "protect"]
    high = [r.name for r in rows if r.recommendation == "high_prune_candidate"]
    mild = [r.name for r in rows if r.recommendation == "weak_sparse_or_mild_prune"]
    print(f"[analyze] BN layers: total={len(rows)}, protect={len(protect)}, weak={len(mild)}, high_candidate={len(high)}")
    print("[analyze] high prune candidates (first 20):")
    for name in high[:20]:
        print(f"  - {name}")
    print(f"[analyze] protected {target.protection_label}/DFL layers (first 20):")
    for name in protect[:20]:
        print(f"  - {name}")


def write_bn_stats_csv(rows: list[BNLayerStat], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "layer_index",
        "category",
        "num_channels",
        "mean_abs_gamma",
        "min_abs_gamma",
        "max_abs_gamma",
        "sparse_ratio",
        "recommendation",
    ]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row.name,
                    "layer_index": "" if row.layer_index is None else row.layer_index,
                    "category": row.category,
                    "num_channels": row.num_channels,
                    "mean_abs_gamma": f"{row.mean_abs:.8f}",
                    "min_abs_gamma": f"{row.min_abs:.8f}",
                    "max_abs_gamma": f"{row.max_abs:.8f}",
                    "sparse_ratio": f"{row.sparse_ratio:.8f}",
                    "recommendation": row.recommendation,
                }
            )
    print(f"[analyze] saved: {out_path}")


def extract_prediction(output: Any):
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, list):
        if output and torch.is_tensor(output[0]) and output[0].ndim == 3:
            return output[0]
        raise RuntimeError(f"Model returned feature list instead of decoded prediction: {[tuple(x.shape) for x in output]}")
    if not torch.is_tensor(output):
        raise RuntimeError(f"Unsupported model output type: {type(output).__name__}")
    return output


def model_stride(model) -> tuple[int, ...]:
    stride = getattr(model, "stride", None)
    if stride is None and hasattr(model, "model"):
        stride = getattr(model.model[-1], "stride", None)
    if stride is None:
        return ()
    if torch.is_tensor(stride):
        return tuple(int(x) for x in stride.detach().cpu().tolist())
    return tuple(int(x) for x in stride)


def output_points_from_stride(stride: tuple[int, ...], imgsz: int) -> int:
    return sum(int(imgsz / s) ** 2 for s in stride if s)


def conv_flops_g(model, imgsz: int, device) -> float:
    total = 0
    handles = []

    def hook(module, _inp, out) -> None:
        nonlocal total
        if torch.is_tensor(out) and out.dim() == 4:
            b, oc, oh, ow = out.shape
            kh, kw = module.kernel_size
            total += b * oc * oh * ow * (module.in_channels // module.groups) * kh * kw * 2

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.inference_mode():
        _ = model(x)
    for handle in handles:
        handle.remove()
    model.train(was_training)
    return float(total) / 1e9


def model_stats(model, imgsz: int, device) -> ModelStats:
    return ModelStats(
        params_m=sum(p.numel() for p in model.parameters()) / 1e6,
        flops_g=conv_flops_g(model, imgsz, device),
    )


def print_stats(label: str, stats: ModelStats) -> None:
    print(f"[stats] {label}: Params={stats.params_m:.6f}M, FLOPs={stats.flops_g:.6f}G")


def assert_reduced(original: ModelStats, pruned: ModelStats, label: str) -> None:
    if pruned.params_m >= original.params_m or pruned.flops_g >= original.flops_g:
        raise RuntimeError(
            f"{label} is not smaller than original: "
            f"Params {original.params_m:.6f}M -> {pruned.params_m:.6f}M, "
            f"FLOPs {original.flops_g:.6f}G -> {pruned.flops_g:.6f}G."
        )


def smoke_check(model, imgsz: int, device, label: str, target: TargetSpec) -> None:
    model.eval()
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.inference_mode():
        output = model(x)
        pred = extract_prediction(output)
    shape = tuple(pred.shape)
    stride = model_stride(model)
    points = output_points_from_stride(stride, imgsz)
    head = model.model[-1] if hasattr(model, "model") else None
    center_preds = getattr(head, "center_preds", None)
    reg_max = getattr(head, "reg_max", None)
    print(f"[smoke] {label}: output_shape={shape}, stride={stride}, output_points={points}, reg_max={reg_max}")
    expected_shape = (1, 5, target.expected_points)
    if shape != expected_shape:
        raise RuntimeError(f"{label} output shape mismatch: got {shape}, expected {expected_shape}")
    if stride != target.expected_stride:
        raise RuntimeError(f"{label} stride mismatch: got {stride}, expected {target.expected_stride}")
    if points != target.expected_points:
        raise RuntimeError(f"{label} point count mismatch: got {points}, expected {target.expected_points}")
    if center_preds is not None:
        raise RuntimeError(f"{label} eval forward kept center_preds; expected None.")
    if reg_max != target.expected_reg_max:
        raise RuntimeError(f"{label} reg_max mismatch: got {reg_max}, expected {target.expected_reg_max}.")


def run_val_smoke(weight_path: Path, args: argparse.Namespace, label: str) -> None:
    print(f"[val] smoke validation for {label}")
    YOLO(str(weight_path)).val(
        data=str(resolve_path(args.data)),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=str(args.device),
        workers=args.workers,
        conf=0.001,
        iou=0.7,
        max_det=1000,
        plots=False,
    )


def train_kwargs(args: argparse.Namespace, name: str, epochs: int, target: TargetSpec) -> dict[str, Any]:
    return {
        "data": str(resolve_path(args.data)),
        "project": str(resolve_path(args.project)),
        "name": name,
        "imgsz": args.imgsz,
        "epochs": epochs,
        "batch": args.batch,
        "device": str(args.device),
        "workers": args.workers,
        "optimizer": "Adam",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "seed": 42,
        "deterministic": True,
        "mosaic": 0.5,
        "close_mosaic": 20,
        "scale": 0.2,
        "translate": 0.08,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "fliplr": 0.5,
        "flipud": 0.0,
        "erasing": 0.0,
        "auto_augment": None,
        "max_det": 1000,
        "patience": 300,
        "amp": True,
        "val": True,
        "plots": True,
        "exist_ok": True,
        "box_loss_type": "ciou",
        "use_center_guidance": target.use_center_guidance,
    }


def make_sparse_trainer_class(sparsity: float, p2_sparse_scale: float, target: TargetSpec):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class BNGammaSparseTrainer(DetectionTrainer):
        def optimizer_step(self):
            self.scaler.unscale_(self.optimizer)
            model = unwrap_model(self.model)
            applied = 0
            for name, module in model.named_modules():
                if not isinstance(module, nn.BatchNorm2d):
                    continue
                layer_index = parse_layer_index(name)
                coeff = bn_sparse_coeff(name, layer_index, sparsity, p2_sparse_scale, target)
                if coeff <= 0 or module.weight.grad is None:
                    continue
                module.weight.grad.data.add_(coeff * torch.sign(module.weight.data))
                applied += 1
            if not getattr(self, "_bn_gamma_sparse_logged", False):
                print(
                    "[sparse] BN-gamma L1 gradient enabled: "
                    f"sparsity={sparsity:g}, p2_sparse_scale={p2_sparse_scale:g}, applied_bn_layers={applied}"
                )
                self._bn_gamma_sparse_logged = True
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

    return BNGammaSparseTrainer


def stage_sparse_train(args: argparse.Namespace) -> None:
    target = get_target(args)
    source_weights = resolve_path(target_source_weights(args, target))
    data_path = resolve_path(args.data)
    if not source_weights.exists():
        raise FileNotFoundError(source_weights)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    epochs = int(args.epochs or 50)
    trainer_cls = make_sparse_trainer_class(args.sparsity, args.p2_sparse_scale, target)
    print(f"[sparse] training {target.display_name} with BN-gamma L1 gradient")
    print(f"[sparse] source_weights={source_weights}")
    print(f"[sparse] output=runs/prune/{target.sparse_exp_name}/weights/best.pt")
    print(f"[sparse] use_center_guidance={target.use_center_guidance}, box_loss_type=ciou, sa_wiou_v3 disabled")
    yolo = YOLO(str(source_weights))
    yolo.train(trainer=trainer_cls, **train_kwargs(args, target.sparse_exp_name, epochs, target))


def load_stats_model(weights: Path, device) -> Any:
    yolo = YOLO(str(weights))
    yolo.model.to(device).eval()
    return yolo


def stage_analyze(args: argparse.Namespace) -> None:
    target = get_target(args)
    weights = resolve_path(target_sparse_weights(args, target))
    if not weights.exists():
        raise FileNotFoundError(f"Sparse checkpoint not found: {weights}. Run --stage sparse_train first.")
    device = torch.device(as_device_arg(str(args.device)))
    yolo = load_stats_model(weights, device)
    rows = collect_bn_stats(yolo.model, args.gamma_thr, target)
    out_path = ROOT / "outputs" / target.importance_csv
    write_bn_stats_csv(rows, out_path)
    print_bn_recommendations(rows, target)


def layer_scale(
    layer_index: int,
    summary: LayerSummary | None,
    global_mean: float,
    ratio: float,
    target: TargetSpec,
) -> tuple[float, str]:
    if layer_index in target.protect_layers:
        return 1.0, "protect"
    mean_abs = summary.mean_abs if summary and summary.mean_abs is not None else global_mean
    sparse_ratio = summary.sparse_ratio if summary and summary.sparse_ratio is not None else 0.0
    redundancy = max(0.0, min(1.0, (1.0 - min(mean_abs / max(global_mean, 1e-8), 1.0)) * 0.6 + sparse_ratio * 0.4))

    if layer_index in target.weak_layers:
        scale = 1.0 - ratio * (0.35 + 0.25 * redundancy)
        return max(scale, 0.93 if ratio <= 0.05 else 0.90), "weak_gamma_guided"
    if layer_index in target.high_value_layers:
        scale = 1.0 - ratio * (0.85 + 0.65 * redundancy)
        return max(scale, 0.90 if ratio <= 0.05 else 0.84), "candidate_gamma_guided"
    scale = 1.0 - ratio * (0.65 + 0.45 * redundancy)
    return max(scale, 0.92 if ratio <= 0.05 else 0.86), "default_gamma_guided"


def effective_channels(raw_channels: int, width: float, max_channels: int) -> int:
    return make_divisible_ceil(min(int(raw_channels), int(max_channels)) * float(width), 8)


def raw_channels_for_effective(target_eff: int, width: float) -> int:
    return max(8, int(math.floor(float(target_eff) / max(float(width), 1e-8))))


def target_effective_channels(old_eff: int, scale: float, ratio: float, category: str) -> int:
    if category == "protect" or old_eff <= 32:
        return old_eff
    if old_eff <= 64 and ratio <= 0.05:
        return old_eff
    target = align_down(old_eff * scale, 8)
    target = max(8, target)
    if target >= old_eff and scale < 0.999 and old_eff >= 128:
        target = old_eff - 8
    return max(8, min(old_eff, target))


def generate_bngamma_pruned_yaml(
    source_yaml: Path,
    ratio: float,
    tag: str,
    rows: list[BNLayerStat],
    target: TargetSpec,
) -> tuple[Path, Path]:
    data = load_yaml_dict(source_yaml)
    if "scales" not in data or "n" not in data["scales"]:
        raise RuntimeError(f"{source_yaml} does not contain scales.n.")
    depth, width, max_channels = list(data["scales"]["n"])
    width = float(width)
    max_channels = int(max_channels)
    summaries = summarize_bn_by_layer(rows)
    global_mean = candidate_global_mean(rows)
    plan_rows: list[dict[str, Any]] = []

    absolute_layers: list[tuple[str, int, list]] = []
    idx = 0
    for layer in data.get("backbone", []):
        absolute_layers.append(("backbone", idx, layer))
        idx += 1
    for layer in data.get("head", []):
        absolute_layers.append(("head", idx, layer))
        idx += 1

    changed = 0
    for section_name, layer_index, layer in absolute_layers:
        module_name = str(layer[2])
        args = layer[3] if len(layer) > 3 else []
        if module_name not in CHANNEL_MODULES or not args or not isinstance(args[0], int):
            continue
        old_raw = int(args[0])
        old_eff = effective_channels(old_raw, width, max_channels)
        summary = summaries.get(layer_index)
        scale, reason = layer_scale(layer_index, summary, global_mean, ratio, target)
        new_eff = target_effective_channels(old_eff, scale, ratio, reason)
        new_raw = old_raw
        if new_eff < old_eff:
            new_raw = raw_channels_for_effective(new_eff, width)
            args[0] = int(new_raw)
            changed += 1
        plan_rows.append(
            {
                "layer_index": layer_index,
                "section": section_name,
                "module": module_name,
                "old_raw_channels": old_raw,
                "new_raw_channels": new_raw,
                "old_effective_channels": old_eff,
                "new_effective_channels": new_eff,
                "scale": f"{scale:.6f}",
                "reason": reason,
                "bn_mean_abs_gamma": "" if summary is None or summary.mean_abs is None else f"{summary.mean_abs:.8f}",
                "bn_sparse_ratio": "" if summary is None or summary.sparse_ratio is None else f"{summary.sparse_ratio:.8f}",
                "bn_channels": 0 if summary is None else summary.num_bn_channels,
            }
        )

    data["scale"] = "n"
    try:
        source_yaml_display = str(source_yaml.relative_to(ROOT))
    except ValueError:
        source_yaml_display = str(source_yaml)

    data["classic_bn_gamma_prune"] = {
        "source_yaml": source_yaml_display,
        "source_sparse_weights": target.sparse_best,
        "target": target.key,
        "ratio": ratio,
        "method": f"{target.protection_label} BN-gamma guided layer-wise YAML channel pruning",
        "protected_layers": sorted(target.protect_layers),
        "weak_layers": sorted(target.weak_layers),
        "changed_channel_layers": changed,
        "note": "Detect/GCGDetect, nc, reg_max, stride, and output points are kept unchanged.",
    }
    out_yaml = ROOT / "models" / "pruned" / f"{target.yaml_prefix}-ratio{tag}.yaml"
    header = (
        f"# Auto-generated {target.protection_label} BN-gamma pruning YAML for {target.display_name}\n"
        f"# ratio={ratio:.2f}; non-uniform YAML channel pruning guided by sparse BN gamma statistics\n"
        "# Detection scales and output dimensions are unchanged."
    )
    save_yaml_dict(data, out_yaml, header)

    plan_path = ROOT / "outputs" / f"{target.plan_prefix}_ratio{tag}.csv"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(plan_rows[0].keys()) if plan_rows else ["layer_index"])
        writer.writeheader()
        writer.writerows(plan_rows)
    print(f"[prune] generated YAML: {out_yaml}")
    print(f"[prune] generated plan: {plan_path}")
    print(f"[prune] changed channel layers: {changed}")
    if changed == 0:
        raise RuntimeError("BN-gamma YAML pruning changed zero channel layers; refusing to create a fake prune result.")
    return out_yaml, plan_path


def transfer_overlapping_weights(source_weights: Path, pruned_yolo) -> tuple[int, int, int, int, int]:
    source_yolo = YOLO(str(source_weights))
    source_state = source_yolo.model.float().state_dict()
    target_state = pruned_yolo.model.float().state_dict()
    updated_state = copy.deepcopy(target_state)

    exact_tensors = 0
    sliced_tensors = 0
    copied_elements = 0
    total_tensors = len(target_state)
    total_elements = sum(v.numel() for v in target_state.values())

    for key, dst in target_state.items():
        src = source_state.get(key)
        if src is None or src.ndim != dst.ndim:
            continue
        if tuple(src.shape) == tuple(dst.shape):
            updated_state[key] = src.detach().to(dtype=dst.dtype)
            exact_tensors += 1
            copied_elements += src.numel()
            continue
        slices = tuple(slice(0, min(int(s), int(d))) for s, d in zip(src.shape, dst.shape))
        if not slices:
            continue
        new_tensor = dst.detach().clone()
        new_tensor[slices] = src.detach()[slices].to(dtype=dst.dtype)
        updated_state[key] = new_tensor
        sliced_tensors += 1
        copied_elements += int(new_tensor[slices].numel())

    pruned_yolo.model.load_state_dict(updated_state, strict=False)
    print(
        "[transfer] exact tensors: "
        f"{exact_tensors}/{total_tensors}, sliced tensors: {sliced_tensors}/{total_tensors}, "
        f"copied_elements={copied_elements}/{total_elements} ({copied_elements / max(total_elements, 1):.2%})"
    )
    return exact_tensors, sliced_tensors, total_tensors, copied_elements, total_elements


def save_pruned_checkpoint(
    pruned_yolo,
    yaml_path: Path,
    source_weights: Path,
    ratio: float,
    tag: str,
    out_path: Path,
    target: TargetSpec,
) -> None:
    model = copy.deepcopy(pruned_yolo.model).half().cpu()
    ckpt = {
        "model": model,
        "ema": None,
        "optimizer": None,
        "updates": None,
        "train_args": {
            "model": str(yaml_path),
            "task": "detect",
            "imgsz": 640,
            "data": "data.yaml",
            "use_center_guidance": target.use_center_guidance,
            "box_loss_type": "ciou",
        },
        "date": datetime.now().isoformat(timespec="seconds"),
        "classic_bn_gamma_prune_meta": {
            "target": target.key,
            "source_yaml": target.source_yaml,
            "source_weights": str(source_weights),
            "ratio": ratio,
            "tag": tag,
            "generated_yaml": str(yaml_path),
            "method": f"{target.protection_label} BN-gamma guided YAML channel pruning",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"[save] pruned checkpoint: {out_path}")


def stage_prune(args: argparse.Namespace) -> None:
    target = get_target(args)
    ratio, tag = normalize_ratio(args.ratio)
    source_yaml = resolve_path(target_source_yaml(args, target))
    source_weights = resolve_path(target_source_weights(args, target))
    sparse_weights = resolve_path(target_sparse_weights(args, target))
    analysis_weights = sparse_weights if sparse_weights.exists() else source_weights
    if not source_yaml.exists():
        raise FileNotFoundError(source_yaml)
    if not source_weights.exists():
        raise FileNotFoundError(source_weights)
    if not sparse_weights.exists():
        print(f"[prune] WARNING: sparse checkpoint not found, using source weights for BN stats: {source_weights}")

    device = torch.device(as_device_arg(str(args.device)))
    original = YOLO(str(source_weights))
    original.model.to(device).eval()
    original_stats = model_stats(original.model, args.imgsz, device)
    print_stats(f"original {target.display_name}", original_stats)
    smoke_check(original.model, args.imgsz, device, f"original {target.display_name}", target)

    stat_model = YOLO(str(analysis_weights))
    stat_model.model.to(device).eval()
    rows = collect_bn_stats(stat_model.model, args.gamma_thr, target)
    write_bn_stats_csv(rows, ROOT / "outputs" / target.importance_csv)
    print_bn_recommendations(rows, target)

    yaml_path, _ = generate_bngamma_pruned_yaml(source_yaml, ratio, tag, rows, target)
    pruned = YOLO(str(yaml_path))
    pruned.model.to(device).eval()
    pruned_stats = model_stats(pruned.model, args.imgsz, device)
    print_stats("classic BN-gamma YAML pruned", pruned_stats)
    assert_reduced(original_stats, pruned_stats, "Classic BN-gamma YAML pruned model")
    smoke_check(pruned.model, args.imgsz, device, "classic BN-gamma YAML pruned", target)

    transfer_overlapping_weights(analysis_weights, pruned)
    pruned.model.to(device).eval()
    transferred_stats = model_stats(pruned.model, args.imgsz, device)
    print_stats("transferred pruned model", transferred_stats)
    assert_reduced(original_stats, transferred_stats, "Transferred pruned model")
    smoke_check(pruned.model, args.imgsz, device, "transferred pruned model", target)

    out_dir = resolve_path(args.project) / f"{target.prune_exp_prefix}_ratio{tag}"
    pruned_pt = out_dir / "pruned.pt"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.exist_ok and not pruned_pt.exists():
        raise SystemExit(f"Output directory already exists: {out_dir}. Use --exist-ok to reuse it.")
    save_pruned_checkpoint(pruned, yaml_path, analysis_weights, ratio, tag, pruned_pt, target)

    reloaded = YOLO(str(pruned_pt))
    reloaded.model.to(device).eval()
    reloaded_stats = model_stats(reloaded.model, args.imgsz, device)
    print_stats("reloaded pruned.pt", reloaded_stats)
    assert_reduced(original_stats, reloaded_stats, "Reloaded pruned.pt")
    smoke_check(reloaded.model, args.imgsz, device, "reloaded pruned.pt", target)

    if not args.skip_val_smoke:
        run_val_smoke(pruned_pt, args, f"{target.key}_classic_bn_gamma_ratio{tag}")


def stage_finetune(args: argparse.Namespace) -> None:
    target = get_target(args)
    ratio, tag = normalize_ratio(args.ratio)
    pruned_pt = resolve_path(args.project) / f"{target.prune_exp_prefix}_ratio{tag}" / "pruned.pt"
    if not pruned_pt.exists():
        raise FileNotFoundError(f"Missing pruned checkpoint: {pruned_pt}. Run --stage prune --ratio {ratio} first.")
    epochs = int(args.epochs or 100)
    print(f"[finetune] training from {pruned_pt}")
    print(f"[finetune] use_center_guidance={target.use_center_guidance}, box_loss_type=ciou, sa_wiou_v3 disabled")
    YOLO(str(pruned_pt)).train(**train_kwargs(args, f"{target.prune_exp_prefix}_ratio{tag}", epochs, target))


def stage_eval(args: argparse.Namespace) -> None:
    target = get_target(args)
    raw_out = ROOT / "outputs" / target.compare_raw_csv
    final_out = ROOT / "outputs" / target.compare_csv
    if target.key == "baseline":
        model_specs = [
            ("YOLOv11n", target.source_weights),
            ("ClassicBN005", "runs/prune/classic_bn_gamma_yolo11n_baseline_ratio005/weights/best.pt"),
            ("ClassicBN010", "runs/prune/classic_bn_gamma_yolo11n_baseline_ratio010/weights/best.pt"),
        ]
        method_map = {
            "YOLOv11n": ("original", "0%"),
            "ClassicBN005": ("classic_bn_gamma", "5%"),
            "ClassicBN010": ("classic_bn_gamma", "10%"),
        }
    else:
        model_specs = [
            ("Final_EGCG", target.source_weights),
            ("Width005", "runs/prune/prune_final_EGCG_ratio005/weights/best.pt"),
            ("ClassicBN005", "runs/prune/classic_bn_gamma_final_EGCG_ratio005/weights/best.pt"),
            ("ClassicBN010", "runs/prune/classic_bn_gamma_final_EGCG_ratio010/weights/best.pt"),
        ]
        method_map = {
            "Final_EGCG": ("original", "0%"),
            "Width005": ("yaml_width", "5%"),
            "ClassicBN005": ("classic_bn_gamma", "5%"),
            "ClassicBN010": ("classic_bn_gamma", "10%"),
        }
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_speed_accuracy.py"),
        "--data",
        str(args.data),
        "--split",
        "val",
        "--imgsz",
        str(args.imgsz),
        "--speed-batch",
        "1",
        "--val-batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--repeats",
        str(args.repeats),
        "--out",
        str(raw_out.relative_to(ROOT)),
    ]
    for name, path in model_specs:
        cmd.extend(["--model", f"{name}={path}"])
    print("[eval] running unified eval:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    rows = []
    with raw_out.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method, ratio = method_map.get(row["model"], ("unknown", ""))
            rows.append(
                {
                    "model": row["model"],
                    "pruning_method": method,
                    "ratio": ratio,
                    "params_M": row.get("params_m", ""),
                    "flops_G": row.get("flops_g", ""),
                    "latency_ms": row.get("fused_forward_decode_ms", ""),
                    "precision": row.get("precision", ""),
                    "recall": row.get("recall", ""),
                    "mAP50": row.get("map50", ""),
                    "mAP50_95": row.get("map50_95", ""),
                }
            )
    final_out.parent.mkdir(parents=True, exist_ok=True)
    with final_out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "pruning_method",
                "ratio",
                "params_M",
                "flops_G",
                "latency_ms",
                "precision",
                "recall",
                "mAP50",
                "mAP50_95",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] saved: {final_out}")


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    if args.stage == "sparse_train":
        stage_sparse_train(args)
    elif args.stage == "analyze":
        stage_analyze(args)
    elif args.stage == "prune":
        stage_prune(args)
    elif args.stage == "finetune":
        stage_finetune(args)
    elif args.stage == "eval":
        stage_eval(args)
    else:
        raise SystemExit(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-check the 2^3 ablation YAMLs for P2, improved fusion and GCG."""

from __future__ import annotations

import argparse
import csv
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs" / "ultralytics_config"))
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")


def install_cv2_stub_if_needed() -> None:
    """Allow model-only smoke tests in minimal environments without OpenCV."""
    if importlib.util.find_spec("cv2") is not None:
        return
    cv2 = types.ModuleType("cv2")
    cv2.__file__ = "<codex-cv2-stub>"
    cv2.__loader__ = None
    cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)

    def _noop(*_args, **_kwargs):
        return None

    cv2.setNumThreads = _noop
    cv2.imshow = _noop
    cv2.imwrite = lambda *_args, **_kwargs: False
    cv2.imread = lambda *_args, **_kwargs: None
    cv2.imdecode = lambda *_args, **_kwargs: None
    cv2.imdecodemulti = lambda *_args, **_kwargs: (False, [])
    cv2.imencode = lambda *_args, **_kwargs: (False, None)
    cv2.waitKey = lambda *_args, **_kwargs: -1
    cv2.destroyAllWindows = _noop
    for name, value in {
        "IMREAD_COLOR": 1,
        "IMREAD_GRAYSCALE": 0,
        "IMREAD_UNCHANGED": -1,
        "INTER_LINEAR": 1,
        "INTER_NEAREST": 0,
        "INTER_AREA": 3,
        "BORDER_CONSTANT": 0,
        "FONT_HERSHEY_SIMPLEX": 0,
    }.items():
        setattr(cv2, name, value)

    def _getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return 0

    cv2.__getattr__ = _getattr
    sys.modules["cv2"] = cv2


install_cv2_stub_if_needed()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402


GROUPS = [
    {
        "group": "A",
        "name": "YOLOv11n",
        "yaml": "models/yolo11n.yaml",
        "p2": False,
        "improved_fusion": False,
        "gcg": False,
        "box_loss": "ciou",
    },
    {
        "group": "B",
        "name": "YOLOv11n + P2",
        "yaml": "models/yolo11n-p2.yaml",
        "p2": True,
        "improved_fusion": False,
        "gcg": False,
        "box_loss": "ciou",
    },
    {
        "group": "C",
        "name": "YOLOv11n + improved multi-scale fusion structure",
        "yaml": "models/yolo11n-drf-safpn-midlite-nop2-r12-v1.yaml",
        "p2": False,
        "improved_fusion": True,
        "gcg": False,
        "box_loss": "ciou",
    },
    {
        "group": "D",
        "name": "YOLOv11n + GCG + SA-WIoU v3",
        "yaml": "models/yolo11n-gcg.yaml",
        "p2": False,
        "improved_fusion": False,
        "gcg": True,
        "box_loss": "sa_wiou_v3",
    },
    {
        "group": "E",
        "name": "YOLOv11n + P2 + improved multi-scale fusion structure",
        "yaml": "models/yolo11n-drf-safpn-midlite-p2safe-r12-v1.yaml",
        "p2": True,
        "improved_fusion": True,
        "gcg": False,
        "box_loss": "ciou",
    },
    {
        "group": "F",
        "name": "YOLOv11n + P2 + GCG + SA-WIoU v3",
        "yaml": "models/yolo11n-p2-gcg.yaml",
        "p2": True,
        "improved_fusion": False,
        "gcg": True,
        "box_loss": "sa_wiou_v3",
    },
    {
        "group": "G",
        "name": "YOLOv11n + improved multi-scale fusion structure + GCG + SA-WIoU v3",
        "yaml": "models/yolo11n-drf-safpn-midlite-nop2-r12-v1-gcg.yaml",
        "p2": False,
        "improved_fusion": True,
        "gcg": True,
        "box_loss": "sa_wiou_v3",
    },
    {
        "group": "H",
        "name": "YOLOv11n + P2 + improved multi-scale fusion structure + GCG + SA-WIoU v3",
        "yaml": "models/yolo11n-drf-safpn-midlite-p2safe-r12-v1-gcg.yaml",
        "p2": True,
        "improved_fusion": True,
        "gcg": True,
        "box_loss": "sa_wiou_v3",
    },
]

FORBIDDEN = ("P2Direct", "P2SemanticGate", "P2ToP3Adapter", "P2Aux", "AuxKD", "RepHead", "TopK")
IMPROVED_TOKENS = ("LightDRF_LSK", "FastSAF", "LiteC3k2", "SharedLiteR12")


def conv_flops(model: nn.Module, x: torch.Tensor) -> float:
    total = 0
    hooks = []

    def hook(m: nn.Conv2d, _inp, out):
        nonlocal total
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            b, oc, oh, ow = out.shape
            kh, kw = m.kernel_size
            total += b * oc * oh * ow * (m.in_channels // m.groups) * kh * kw * 2

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        _ = model(x)
    for handle in hooks:
        handle.remove()
    return total / 1e9


def output_signature(output) -> str:
    if isinstance(output, torch.Tensor):
        return "Tensor" + str(tuple(output.shape))
    if isinstance(output, (list, tuple)):
        return "[" + ",".join(output_signature(x) for x in output) + "]"
    return type(output).__name__


def check_yaml_text(meta: dict) -> None:
    path = ROOT / meta["yaml"]
    text = path.read_text(encoding="utf-8")
    missing = [token for token in FORBIDDEN if token in text]
    if missing:
        raise AssertionError(f"{meta['yaml']} contains forbidden modules: {missing}")
    has_improved = any(token in text for token in IMPROVED_TOKENS)
    if has_improved != meta["improved_fusion"]:
        raise AssertionError(
            f"{meta['yaml']} improved_fusion expected={meta['improved_fusion']} got={has_improved}"
        )
    if meta["improved_fusion"]:
        absent = [token for token in IMPROVED_TOKENS if token not in text]
        if absent:
            raise AssertionError(f"{meta['yaml']} missing improved-fusion tokens: {absent}")
    has_gcg_capable = "GCGDetect" in text
    if meta["gcg"] and not has_gcg_capable:
        raise AssertionError(f"{meta['yaml']} should use a GCG-capable head")
    if (not meta["gcg"]) and (not meta["improved_fusion"]) and has_gcg_capable:
        raise AssertionError(f"{meta['yaml']} unexpectedly uses a GCG-capable head")


def check_one(meta: dict, args: argparse.Namespace) -> dict:
    path = ROOT / meta["yaml"]
    if not path.exists():
        raise FileNotFoundError(path)
    check_yaml_text(meta)

    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() and torch.cuda.is_available() else "cpu")
    yolo = YOLO(str(path))
    model = yolo.model.to(device)
    head = model.model[-1]
    x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)

    model.eval()
    with torch.inference_mode():
        out = model(x)
    pred = out[0] if isinstance(out, tuple) else out
    strides = [int(s) for s in head.stride.tolist()]
    expected_strides = [4, 8, 16, 32] if meta["p2"] else [8, 16, 32]
    expected_points = 34000 if meta["p2"] else 8400
    expected_shape = (1, 5, expected_points)

    if strides != expected_strides:
        raise AssertionError(f"{meta['group']} stride expected={expected_strides} got={strides}")
    if tuple(pred.shape) != expected_shape:
        raise AssertionError(f"{meta['group']} output expected={expected_shape} got={tuple(pred.shape)}")
    if meta["gcg"] and getattr(head, "center_preds", None) is not None:
        raise AssertionError(f"{meta['group']} eval center_preds should be None")

    flops_g = conv_flops(model, x)

    model.train()
    model.args = SimpleNamespace(
        box=7.5,
        cls=0.5,
        dfl=1.5,
        box_loss_type=meta["box_loss"],
        use_center_guidance=meta["gcg"],
        center_gain=0.1,
        aux_gain=0.0,
    )
    criterion = v8DetectionLoss(model)
    preds = model(x.requires_grad_(True))
    if meta["gcg"] and getattr(head, "center_preds", None) is None:
        raise AssertionError(f"{meta['group']} train center_preds missing")
    batch = {
        "batch_idx": torch.zeros(0, device=device),
        "cls": torch.zeros(0, device=device),
        "bboxes": torch.zeros(0, 4, device=device),
    }
    loss, _items = criterion(preds, batch)
    loss.sum().backward()

    return {
        "group": meta["group"],
        "name": meta["name"],
        "yaml": meta["yaml"],
        "head": type(head).__name__,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 6),
        "flops_g": round(flops_g, 6),
        "output_points": expected_points,
        "output_shape": tuple(pred.shape),
        "output_signature": output_signature(out),
        "stride": strides,
        "p2": meta["p2"],
        "improved_fusion": meta["improved_fusion"],
        "gcg": meta["gcg"],
        "box_loss": meta["box_loss"],
        "status": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="outputs/ablation_yaml_check.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for meta in GROUPS:
        row = check_one(meta, args)
        rows.append(row)
        print(
            f"{row['group']} PASS | {row['yaml']} | head={row['head']} | "
            f"params={row['params_m']:.3f}M flops={row['flops_g']:.3f}G | "
            f"shape={row['output_shape']} stride={row['stride']}"
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAll ablation YAML smoke tests passed. CSV: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a P2-AuxKD training checkpoint to an inference-only YOLO model."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import torch

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


def load_state(weights: str) -> dict:
    model = YOLO(weights).model
    return model.state_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip P2 auxiliary branch from a P2-AuxKD checkpoint.")
    parser.add_argument("--weights", required=True, help="Training checkpoint with P2 auxiliary head.")
    parser.add_argument("--infer-yaml", required=True, help="Inference-only YAML with P3/P4/P5 head.")
    parser.add_argument("--out", required=True, help="Output .pt path.")
    args = parser.parse_args()

    infer = YOLO(args.infer_yaml).model
    src = load_state(args.weights)
    dst = infer.state_dict()
    matched = {}
    skipped = []
    for key, value in src.items():
        if key in dst and tuple(dst[key].shape) == tuple(value.shape):
            matched[key] = value
        else:
            skipped.append(key)
    missing, unexpected = infer.load_state_dict(matched, strict=False)
    ckpt = {"model": infer, "train_args": {"stripped_from": args.weights, "infer_yaml": args.infer_yaml}}
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"Saved inference-only checkpoint: {out}")
    print(f"matched={len(matched)} skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}")


if __name__ == "__main__":
    main()

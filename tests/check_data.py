#!/usr/bin/env python3
"""检查 data.yaml 与预训练权重。"""
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "data.yaml") as f:
    d = yaml.safe_load(f)
assert d["nc"] == 1, f"nc={d['nc']}"
for name, key in [("train", "train"), ("val", "val"), ("test", "test")]:
    p = Path(d["path"]) / d[key]
    n = len(list(p.glob("*"))) if p.exists() else 0
    assert p.exists() and n > 0, f"{name} path missing or empty: {p}"
    print(f"{name}: {n} files, nc={d['nc']}")
YOLO(str(ROOT / "yolo11n.pt"))
print("yolo11n.pt OK")

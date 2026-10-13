#!/usr/bin/env python3
"""仅导出 ONNX（部署常用）。"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Export YOLO to ONNX")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()
    w = Path(args.weights)
    model = YOLO(str(w))
    model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, dynamic=False, simplify=True)
    print(f"ONNX: {w.with_suffix('.onnx')}")


if __name__ == "__main__":
    main()

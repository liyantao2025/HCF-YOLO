#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DRF-SAFPN-YOLO 推理脚本（读取 ablation_config 中的 predict_* 参数）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_utils import build_predict_kwargs, load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, default="dataset/images/test_improved")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "ablation_config.yaml"))
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--max-det", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    pred_kw = build_predict_kwargs(cfg, overrides={
        k: v for k, v in {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
        }.items() if v is not None
    })

    model = YOLO(args.weights)
    print(f"predict: conf={pred_kw['conf']} iou={pred_kw['iou']} max_det={pred_kw['max_det']}")
    model.predict(source=args.source, save=True, **pred_kw)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""模型结构与前向传播检查（8 组消融对应 4 个基础 YAML）。"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from ablation_utils import resolve_model_yaml
from ultralytics.nn.tasks import DetectionModel

CONFIG = ROOT / "configs" / "ablation_config.yaml"


def inspect_model(name: str, exp_cfg: dict):
    model_path = resolve_model_yaml(exp_cfg["model"], use_gcg=bool(exp_cfg.get("sa_wiou_gcg")))
    print(f"\n{'='*60}\n实验: {name} ({exp_cfg['model']})")
    m = DetectionModel(str(model_path), nc=1, verbose=False)
    m.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        m(x)

    types = {type(mod).__name__ for mod in m.modules()}
    detect = m.model[-1]
    nl = getattr(detect, "nl", 0)
    stride = detect.stride.tolist() if hasattr(detect, "stride") else []

    print(f"Detect nl={nl}, stride={stride}")
    print(f"SAF={'SAF' in types}, DRF_LSK={'DRF_LSK' in types}, GCGDetect={'GCGDetect' in types}")
    params = sum(p.numel() for p in m.parameters())
    print(f"Params: {params:,}")

    if nl == 4:
        assert stride == [4.0, 8.0, 16.0, 32.0] or stride == [4, 8, 16, 32], f"四尺度 stride 异常: {stride}"
        print("四尺度检测: OK")

    return {"name": name, "nl": nl, "stride": stride, "params": params}


def main():
    with open(CONFIG, encoding="utf-8") as f:
        experiments = yaml.safe_load(f)["experiments"]
    print("=== 8 组消融前向结构检查（4 个基础 YAML）===")
    rows = []
    for name, cfg in experiments.items():
        try:
            rows.append(inspect_model(name, cfg))
            print("forward: OK")
        except Exception as e:
            print(f"FAILED: {e}")
            rows.append({"name": name, "error": str(e)})
    print("\n汇总:", rows)


if __name__ == "__main__":
    main()

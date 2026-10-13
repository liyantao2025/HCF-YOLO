#!/usr/bin/env python3
"""消融配置结构检查（不训练）。"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_utils import resolve_model_yaml  # noqa: E402
from ultralytics.nn.tasks import DetectionModel

CONFIG = ROOT / "configs" / "ablation_config.yaml"

try:
    from ultralytics.utils.torch_utils import get_flops
except Exception:
    get_flops = None


def load_experiments() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["experiments"]


def inspect(exp: str, cfg: dict):
    model_path = resolve_model_yaml(cfg["model"], use_gcg=bool(cfg.get("sa_wiou_gcg")))
    m = DetectionModel(str(model_path), nc=1, verbose=False)
    types = {type(mod).__name__ for mod in m.modules()}
    det = m.model[-1]
    row = {
        "exp": exp,
        "model": cfg["model"],
        "box_loss": cfg["box_loss_type"],
        "gcg_flag": cfg["use_center_guidance"],
        "nl": det.nl,
        "stride": det.stride.tolist(),
        "SAF": "SAF" in types,
        "DRF_LSK": "DRF_LSK" in types,
        "GCGDetect": "GCGDetect" in types,
        "params": sum(p.numel() for p in m.parameters()),
    }
    if get_flops:
        try:
            row["GFLOPs"] = round(get_flops(m, imgsz=640), 1)
        except Exception:
            row["GFLOPs"] = "N/A"
    return row


def main():
    experiments = load_experiments()
    print(f"{'exp':<14} {'nl':>2} {'SAF':>4} {'DRF':>4} {'GCG':>4} {'box_loss':<12} {'params':>10}")
    for name, cfg in experiments.items():
        r = inspect(name, cfg)
        print(
            f"{r['exp']:<14} {r['nl']:>2} {str(r['SAF']):>4} {str(r['DRF_LSK']):>4} {str(r['GCGDetect']):>4} "
            f"{r['box_loss']:<12} {r['params']:>10,}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""消融实验入口：高密度棉苗小目标检测，严格从零训练。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from ultralytics import YOLO

from ablation_utils import (  # noqa: E402
    DEFAULT_CONFIG,
    apply_mode,
    build_train_kwargs,
    build_val_kwargs,
    exp_display_name,
    exp_table_flags,
    load_config,
    read_best_epoch,
    resolve_model_yaml,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_no_old_weights(exp_dir: Path) -> None:
    for w in ("best.pt", "last.pt"):
        wp = exp_dir / "weights" / w
        if wp.exists():
            raise RuntimeError(f"检测到已有 {w}（{wp}），请使用 --fresh --yes 删除旧目录")


def check_train_from_scratch(model_path: Path, exp_dir: Path, cfg: dict) -> None:
    if str(model_path).lower().endswith(".pt"):
        raise RuntimeError(f"禁止从 .pt 加载: {model_path}")
    if not str(model_path).lower().endswith(".yaml"):
        raise AssertionError("必须从 YAML 构建模型")
    if cfg["init_mode"] == "random":
        assert not cfg["pretrained"], "pretrained 必须为 False"
        assert not cfg["resume"], "resume 必须为 False"
    if not model_path.is_file():
        raise FileNotFoundError(f"YAML 不存在: {model_path}")
    check_no_old_weights(exp_dir)


def maybe_fresh_exp_dir(exp_dir: Path, yes: bool) -> None:
    if not exp_dir.exists():
        return
    print(f"即将删除旧实验目录：{exp_dir}")
    if not yes:
        ans = input("确认删除并重新从 0 训练？[y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            raise SystemExit("已取消")
    shutil.rmtree(exp_dir)
    print(f"已删除: {exp_dir}")


def load_density_summary() -> dict:
    p = ROOT / "outputs" / "dataset_density.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def log_gcg_head(model) -> None:
    head = model.model.model[-1]
    if not type(head).__name__.endswith("GCGDetect"):
        print("GCG enabled: False")
        return
    if len(head.cv_center) == 0 and len(head.stride):
        head.init_center_branches()
    strides = [float(s) for s in head.stride.tolist()]
    sel = [strides[i] for i in head.center_layers] if head.center_layers else []
    print("GCG enabled: True")
    print(f"GCG selected layers: {list(head.center_layers)}")
    print(f"GCG selected strides: {sel}")


def log_init_settings(exp_key: str, exp_cfg: dict, model_path: Path, cfg: dict) -> str:
    flags = exp_table_flags(exp_cfg)
    lines = [
        f"实验键: {exp_key}",
        f"实验名: {exp_cfg['name']}",
        f"Model: {exp_display_name(exp_key, exp_cfg)}",
        f"init_mode={cfg['init_mode']}",
        f"pretrained={cfg['pretrained']}",
        f"resume={cfg['resume']}",
        f"seed={cfg['seed']}",
        f"model_source={model_path}",
        f"max_det={cfg.get('val_max_det', cfg.get('max_det'))}",
        f"val_conf={cfg.get('val_conf')} val_iou={cfg.get('val_iou')}",
        f"mosaic={cfg.get('mosaic')} scale={cfg.get('scale')} patience={cfg.get('patience')}",
        f"P2={flags['P2']} DRF-SAFPN={flags['DRF-SAFPN']} "
        f"SA-WIoU+GCG={flags['SA-WIoU v3 + Gaussian']}",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    return text


def run_one_exp(exp_key: str, cfg: dict, args: argparse.Namespace) -> dict:
    exp_cfg = cfg["experiments"][exp_key]
    model_path = resolve_model_yaml(exp_cfg["model"], use_gcg=bool(exp_cfg.get("sa_wiou_gcg")))
    project = ROOT / cfg["project"] if not Path(cfg["project"]).is_absolute() else Path(cfg["project"])
    exp_dir = project / exp_cfg["name"]
    density = load_density_summary()

    if args.fresh and exp_dir.exists():
        maybe_fresh_exp_dir(exp_dir, yes=args.yes)
    else:
        check_train_from_scratch(model_path, exp_dir, cfg)

    set_seed(int(cfg["seed"]))
    init_text = log_init_settings(exp_key, exp_cfg, model_path, cfg)

    print(f"\n{'=' * 60}\n实验: {exp_key} | {exp_cfg['desc']}\n{'=' * 60}\n")

    model = YOLO(str(model_path))
    log_gcg_head(model)

    train_kw = build_train_kwargs(cfg, exp_cfg, args)
    train_kw["project"] = str(project)
    train_kw["name"] = exp_cfg["name"]
    model.train(**train_kw)

    exp_dir = Path(getattr(model.trainer, "save_dir", exp_dir))
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "init_settings.txt").write_text(init_text, encoding="utf-8")

    metrics = model.val(**build_val_kwargs(cfg, args))
    rd = metrics.results_dict
    speed = metrics.speed or {}
    infer_ms = float(speed.get("inference", 0) or 0)

    target_epochs = int(args.epochs or cfg["epochs"])
    best_ep, actual_ep, notes = read_best_epoch(exp_dir, target_epochs)
    flags = exp_table_flags(exp_cfg)
    params = sum(p.numel() for p in model.model.parameters())

    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = round(get_flops(model.model, imgsz=int(cfg["imgsz"])), 2)
    except Exception:
        gflops = ""

    return {
        "Model": exp_display_name(exp_key, exp_cfg),
        "exp": exp_key,
        "name": exp_cfg["name"],
        "P2": flags["P2"],
        "DRF-SAFPN": flags["DRF-SAFPN"],
        "SA-WIoU v3 + Gaussian": flags["SA-WIoU v3 + Gaussian"],
        "Precision": float(rd.get("metrics/precision(B)", 0)),
        "Recall": float(rd.get("metrics/recall(B)", 0)),
        "mAP50": float(rd.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(rd.get("metrics/mAP50-95(B)", 0)),
        "Params/M": round(params / 1e6, 3),
        "FLOPs/G": gflops,
        "Latency/ms": round(infer_ms, 2) if infer_ms else "",
        "FPS": round(1000.0 / infer_ms, 1) if infer_ms else "",
        "max_det": cfg.get("val_max_det", cfg.get("max_det")),
        "val_conf": cfg.get("val_conf"),
        "val_iou": cfg.get("val_iou"),
        "train_images": density.get("train", {}).get("images", ""),
        "train_instances": density.get("train", {}).get("instances", ""),
        "avg_instances_per_image": density.get("train", {}).get("avg_per_image", ""),
        "Init Mode": cfg["init_mode"],
        "Pretrained": cfg["pretrained"],
        "Resume": cfg["resume"],
        "Train From Scratch": cfg["init_mode"] == "random" and not cfg["pretrained"],
        "best_epoch": best_ep,
        "actual_epochs": actual_ep,
        "Notes": notes,
        "model_source": str(model_path),
        "seed": cfg["seed"],
        "exp_dir": str(exp_dir),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DRF-SAFPN-YOLO 消融实验（高密度小目标）")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--exp", type=str, default=None)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--mode", choices=["random", "pretrained"], default="random")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_mode(load_config(Path(args.config)), args.mode)
    exp_keys = list(cfg["experiments"].keys())
    target = args.only or args.exp

    if not target and not args.all:
        print("请指定 --exp / --only 或 --all。可选:")
        for k, v in cfg["experiments"].items():
            print(f"  {k:14} {v['name']:28} {v['model']}")
        sys.exit(0)

    if target and target not in exp_keys:
        raise SystemExit(f"未知实验: {target}")

    exps = [target] if target else [k for k in exp_keys if k not in set(args.skip)]
    rows = []
    for k in exps:
        row = run_one_exp(k, cfg, args)
        rows.append(row)
        print(
            f"[{k}] P={row['Precision']:.4f} R={row['Recall']:.4f} "
            f"mAP50={row['mAP50']:.4f} max_det={row['max_det']}"
        )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"ablation_results_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy(csv_path, out_dir / "ablation_results.csv")
    print(f"\n结果: {csv_path}")

    if args.all and len(exps) == len(exp_keys):
        print("全部 8 组训练完成，正在生成汇总表…")
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_ablation_results.py")], check=False)
    else:
        print(f"汇总: python scripts/collect_ablation_results.py")


if __name__ == "__main__":
    main()

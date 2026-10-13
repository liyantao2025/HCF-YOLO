#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 runs/ablation 消融结果，生成 CSV / XLSX / MD 与公平性报告。"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_utils import (  # noqa: E402
    DEFAULT_CONFIG,
    build_val_kwargs,
    exp_display_name,
    exp_table_flags,
    load_config,
    read_best_epoch,
)

OUT_DIR = ROOT / "outputs"

MD_COLS = [
    "Model",
    "P2",
    "DRF-SAFPN",
    "SA-WIoU v3 + Gaussian",
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95",
    "Params/M",
    "FLOPs/G",
    "Latency/ms",
    "FPS",
]

CSV_COLS = MD_COLS + [
    "max_det",
    "val_conf",
    "val_iou",
    "train_images",
    "train_instances",
    "avg_instances_per_image",
    "Init Mode",
    "Pretrained",
    "Resume",
    "Train From Scratch",
    "best_epoch",
    "actual_epochs",
    "Notes",
    "exp",
    "name",
    "model_source",
    "seed",
    "exp_dir",
]


def load_density() -> dict:
    p = OUT_DIR / "dataset_density.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def reval_exp(exp_key: str, exp_cfg: dict, cfg: dict) -> dict | None:
    project = ROOT / cfg["project"]
    exp_dir = project / exp_cfg["name"]
    weights = exp_dir / "weights" / "best.pt"
    if not weights.exists():
        return None

    model = YOLO(str(weights))
    metrics = model.val(**build_val_kwargs(cfg))
    rd = metrics.results_dict
    speed = metrics.speed or {}
    infer_ms = float(speed.get("inference", 0) or 0)
    params = sum(p.numel() for p in model.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = round(get_flops(model.model, imgsz=int(cfg["imgsz"])), 2)
    except Exception:
        gflops = ""

    target_epochs = int(cfg["epochs"])
    best_ep, actual_ep, notes = read_best_epoch(exp_dir, target_epochs)
    flags = exp_table_flags(exp_cfg)
    density = load_density()
    tr = density.get("train", {})

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
        "train_images": tr.get("images", ""),
        "train_instances": tr.get("instances", ""),
        "avg_instances_per_image": tr.get("avg_per_image", ""),
        "Init Mode": cfg.get("init_mode"),
        "Pretrained": cfg.get("pretrained"),
        "Resume": cfg.get("resume"),
        "Train From Scratch": cfg.get("init_mode") == "random" and not cfg.get("pretrained"),
        "best_epoch": best_ep,
        "actual_epochs": actual_ep,
        "Notes": notes,
        "model_source": str(ROOT / exp_cfg["model"]),
        "seed": cfg.get("seed"),
        "exp_dir": str(exp_dir),
    }


def scan_args_warnings(cfg: dict) -> list[str]:
    warnings = []
    project = ROOT / cfg["project"]
    expected_max_det = int(cfg.get("val_max_det", cfg.get("max_det", 1000)))
    for key, exp in cfg["experiments"].items():
        args_file = project / exp["name"] / "args.yaml"
        if not args_file.exists():
            continue
        with open(args_file, encoding="utf-8") as f:
            args = yaml.safe_load(f) or {}
        md = args.get("max_det")
        if md is not None and int(md) != expected_max_det:
            warnings.append(f"`{exp['name']}` args.yaml max_det={md} != 期望 {expected_max_det}")
        if str(args.get("model", "")).endswith(".pt"):
            warnings.append(f"`{exp['name']}` 使用了 .pt 作为训练入口: {args.get('model')}")
    return warnings


def fairness_report(cfg: dict, rows: list[dict], warnings: list[str]) -> str:
    density_md = OUT_DIR / "dataset_density_report.md"
    density_excerpt = ""
    if density_md.exists():
        density_excerpt = density_md.read_text(encoding="utf-8")

    checks = [
        ("所有模型从 YAML 构建", all(not str(r.get("model_source", "")).endswith(".pt") for r in rows)),
        ("pretrained=False（配置）", not cfg.get("pretrained")),
        ("resume=False（配置）", not cfg.get("resume")),
        ("init_mode=random（配置）", cfg.get("init_mode") == "random"),
        ("pretrained_weights 为空", cfg.get("pretrained_weights") in (None, "", "null")),
        (f"max_det={cfg.get('val_max_det', 1000)}（配置）", int(cfg.get("val_max_det", 1000)) == 1000),
        (f"patience={cfg.get('patience')} 一致", cfg.get("patience") == 300),
        ("mosaic=0.5", float(cfg.get("mosaic", -1)) == 0.5),
        ("scale=0.2", float(cfg.get("scale", -1)) == 0.2),
        ("auto_augment 已关闭", cfg.get("auto_augment") in (None, "null", "")),
    ]

    lines = [
        "# 消融实验公平性报告",
        "",
        f"生成时间: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 高密度数据集说明",
        "",
        "训练图像数量有限，但单图棉苗实例密集（详见 `outputs/dataset_density_report.md`）。",
        "验证/推理统一 `max_det=1000`，避免默认 300 限制召回。",
        "",
        "## 统一训练/验证参数",
        "",
        f"- max_det / val_max_det: **{cfg.get('val_max_det', 1000)}**",
        f"- val_conf: **{cfg.get('val_conf')}**, val_iou: **{cfg.get('val_iou')}**",
        f"- predict_conf: **{cfg.get('predict_conf')}**, predict_iou: **{cfg.get('predict_iou')}**（仅可视化）",
        f"- mosaic: **{cfg.get('mosaic')}**, scale: **{cfg.get('scale')}**, patience: **{cfg.get('patience')}**",
        f"- epochs: **{cfg.get('epochs')}**, seed: **{cfg.get('seed')}**",
        "",
        "## 初始化公平性检查",
        "",
    ]
    for title, ok in checks:
        lines.append(f"- [{'x' if ok else ' '}] {title}: **{'通过' if ok else '未通过'}**")

    if warnings:
        lines.extend(["", "## 警告", ""])
        for w in warnings:
            lines.append(f"- ⚠ {w}")

    lines.extend(["", "## 各实验 epoch 记录", ""])
    for r in rows:
        lines.append(
            f"- `{r.get('name')}`: best_epoch={r.get('best_epoch')}, "
            f"actual_epochs={r.get('actual_epochs')}, Notes={r.get('Notes') or '—'}"
        )

    passed = all(ok for _, ok in checks) and not warnings
    lines.extend(["", "## 结论", ""])
    if passed:
        lines.append(
            "**初始化公平性检查：通过。** 所有消融模型均从随机初始化开始训练，"
            "未加载历史/预训练权重；训练、验证、测速参数一致。"
        )
    else:
        lines.append("**初始化公平性检查：存在未通过项或警告，请核对上述列表。**")

    if density_excerpt:
        lines.extend(["", "---", "", "## 附录：数据集密度统计", "", density_excerpt])
    return "\n".join(lines) + "\n"


def to_md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_无数据_\n"
    h = "| " + " | ".join(cols) + " |"
    s = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([h, s, *body]) + "\n"


def main() -> None:
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, exp in cfg["experiments"].items():
        row = reval_exp(key, exp, cfg)
        if row:
            rows.append(row)

    warnings = scan_args_warnings(cfg)
    csv_path = OUT_DIR / "ablation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md_path = OUT_DIR / "ablation_results.md"
    md_path.write_text("# 消融实验结果（论文核心字段）\n\n" + to_md_table(rows, MD_COLS), encoding="utf-8")

    xlsx_path = OUT_DIR / "ablation_results.xlsx"
    try:
        import pandas as pd

        pd.DataFrame(rows).to_excel(xlsx_path, index=False)
    except ImportError:
        xlsx_path = None

    fair_path = OUT_DIR / "ablation_fairness_report.md"
    fair_path.write_text(fairness_report(cfg, rows, warnings), encoding="utf-8")

    meta = {
        "csv": str(csv_path),
        "md": str(md_path),
        "xlsx": str(xlsx_path) if xlsx_path else None,
        "fairness": str(fair_path),
        "n_experiments": len(rows),
        "warnings": warnings,
    }
    (OUT_DIR / "collect_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

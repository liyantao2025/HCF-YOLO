#!/usr/bin/env python3
"""Summarize repeated YOLO11n/HCF-YOLO11n test results for reviewer response."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "reviewer_repeat_main.yaml"
METRICS = ("Precision", "Recall", "mAP50", "mAP50-95")


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load_rows(config_path: Path) -> tuple[dict, list[dict]]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = ROOT / cfg["project"]
    rows: list[dict] = []
    for model_key, model_cfg in cfg["models"].items():
        for seed in cfg["seeds"]:
            result_file = project / f"{model_key}_seed{seed}" / "test_metrics.json"
            if not result_file.exists():
                continue
            row = json.loads(result_file.read_text(encoding="utf-8"))
            rows.append(row)
    return cfg, rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(config_path: Path = DEFAULT_CONFIG) -> dict:
    cfg, rows = load_rows(config_path)
    out_dir = ROOT / "outputs" / "reviewer_repeat_main"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_fields = [
        "Model", "model_key", "seed", *METRICS, "best_epoch", "actual_epochs",
        "Params/M", "FLOPs/G", "model_source", "weights", "eval_split",
        "pretrained", "resume", "init_mode",
    ]
    write_csv(out_dir / "repeated_runs_raw.csv", rows, raw_fields)

    summary_rows: list[dict] = []
    for model_key, model_cfg in cfg["models"].items():
        model_rows = [r for r in rows if r["model_key"] == model_key]
        if not model_rows:
            continue
        item: dict = {
            "Model": model_cfg["display_name"],
            "n": len(model_rows),
            "seeds": ",".join(str(r["seed"]) for r in sorted(model_rows, key=lambda x: x["seed"])),
        }
        for metric in METRICS:
            avg, sd = mean_sd([float(r[metric]) for r in model_rows])
            item[f"{metric}_mean"] = avg
            item[f"{metric}_sd"] = sd
            item[f"{metric}_mean±sd"] = f"{avg:.4f} ± {sd:.4f}"
        summary_rows.append(item)

    summary_fields = ["Model", "n", "seeds"]
    for metric in METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_sd", f"{metric}_mean±sd"])
    write_csv(out_dir / "repeated_runs_summary.csv", summary_rows, summary_fields)

    paired_rows: list[dict] = []
    base_by_seed = {int(r["seed"]): r for r in rows if r["model_key"] == "yolo11n"}
    hcf_by_seed = {int(r["seed"]): r for r in rows if r["model_key"] == "hcf_yolo11n"}
    common_seeds = sorted(set(base_by_seed) & set(hcf_by_seed))
    for metric in METRICS:
        deltas = [float(hcf_by_seed[s][metric]) - float(base_by_seed[s][metric]) for s in common_seeds]
        if not deltas:
            continue
        avg, sd = mean_sd(deltas)
        # For the planned n=3 paired runs, t(0.975, df=2)=4.303.
        t_critical = 4.303 if len(deltas) == 3 else 1.96
        half_width = t_critical * sd / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
        paired_rows.append({
            "Metric": metric,
            "n_pairs": len(deltas),
            "seeds": ",".join(map(str, common_seeds)),
            "mean_delta_HCF_minus_YOLO11n": avg,
            "delta_sd": sd,
            "95%CI_low": avg - half_width,
            "95%CI_high": avg + half_width,
        })
    write_csv(
        out_dir / "paired_differences.csv",
        paired_rows,
        ["Metric", "n_pairs", "seeds", "mean_delta_HCF_minus_YOLO11n", "delta_sd", "95%CI_low", "95%CI_high"],
    )

    md = [
        "# YOLO11n与HCF-YOLO11n多随机种子重复实验",
        "",
        f"计划种子：{', '.join(map(str, cfg['seeds']))}；最终评价集合：{cfg.get('eval_split', 'test')}。",
        "",
        "## 均值与标准差",
        "",
        "| Model | n | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r['Model']} | {r['n']} | {r['Precision_mean±sd']} | "
            f"{r['Recall_mean±sd']} | {r['mAP50_mean±sd']} | {r['mAP50-95_mean±sd']} |"
        )
    md.extend([
        "",
        "## 同种子配对差值（HCF-YOLO11n − YOLO11n）",
        "",
        "| Metric | n pairs | Mean delta | SD | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ])
    for r in paired_rows:
        md.append(
            f"| {r['Metric']} | {r['n_pairs']} | {r['mean_delta_HCF_minus_YOLO11n']:.4f} | "
            f"{r['delta_sd']:.4f} | [{r['95%CI_low']:.4f}, {r['95%CI_high']:.4f}] |"
        )
    completed = len(rows)
    expected = len(cfg["seeds"]) * len(cfg["models"])
    md.extend(["", f"完成进度：{completed}/{expected} 个独立训练。", ""])
    (out_dir / "repeated_runs_summary.md").write_text("\n".join(md), encoding="utf-8")

    meta = {
        "completed": completed,
        "expected": expected,
        "complete": completed == expected,
        "raw_csv": str(out_dir / "repeated_runs_raw.csv"),
        "summary_csv": str(out_dir / "repeated_runs_summary.csv"),
        "paired_csv": str(out_dir / "paired_differences.csv"),
        "summary_md": str(out_dir / "repeated_runs_summary.md"),
    }
    (out_dir / "summary_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


if __name__ == "__main__":
    summarize()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计棉花数据集目标密度（高密度小目标）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
SMALL_AREA_RATIO = 0.01  # 相对 640x640 归一化面积 <1% 视为小目标


def parse_label_file(lb_path: Path, img_w: float = 640, img_h: float = 640) -> list[dict]:
    boxes = []
    if not lb_path.exists():
        return boxes
    for line in lb_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = map(float, parts[:5])
        bw, bh = w * img_w, h * img_h
        boxes.append({"w": bw, "h": bh, "area": bw * bh, "area_ratio": (bw * bh) / (img_w * img_h)})
    return boxes


def analyze_split(name: str, img_dir: Path, base: Path, split_rel: str) -> dict:
    imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    label_dir = base / "labels" / Path(split_rel).name
    per_image_counts = []
    areas = []
    small = 0
    total = 0
    for img in imgs:
        lb = label_dir / f"{img.stem}.txt"
        boxes = parse_label_file(lb)
        n = len(boxes)
        per_image_counts.append(n)
        total += n
        for b in boxes:
            areas.append(b["area_ratio"])
            if b["area_ratio"] < SMALL_AREA_RATIO:
                small += 1
    ni = len(imgs)
    return {
        "split": name,
        "images": ni,
        "instances": total,
        "avg_per_image": round(total / ni, 2) if ni else 0,
        "min_per_image": min(per_image_counts) if per_image_counts else 0,
        "max_per_image": max(per_image_counts) if per_image_counts else 0,
        "small_target_ratio": round(small / total, 4) if total else 0,
        "area_ratio_mean": round(sum(areas) / len(areas), 6) if areas else 0,
        "area_ratio_median": round(sorted(areas)[len(areas) // 2], 6) if areas else 0,
    }


def main() -> None:
    data_yaml = ROOT / "data.yaml"
    with open(data_yaml, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    base = Path(d["path"])

    splits = {}
    for key in ("train", "val", "test"):
        img_dir = base / d[key]
        splits[key] = analyze_split(key, img_dir, base, d[key])

    md_lines = [
        "# 棉花苗期数据集密度统计",
        "",
        "## 概述",
        "",
        "本数据集属于**高密度小目标检测**任务：训练图像数量有限，但单图目标实例数极高。",
        "参数设置应优先保证 `max_det`、验证 NMS 与小目标友好增强策略。",
        "",
        "## 分集统计",
        "",
        "| 划分 | 图像数 | 实例总数 | 平均实例/图 | 最少/图 | 最多/图 | 小目标比例 | 面积比均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, s in splits.items():
        md_lines.append(
            f"| {k} | {s['images']} | {s['instances']} | {s['avg_per_image']} | "
            f"{s['min_per_image']} | {s['max_per_image']} | {s['small_target_ratio']:.2%} | {s['area_ratio_mean']} |"
        )

    tr = splits.get("train", {})
    md_lines.extend(
        [
            "",
            "## 结论（用于消融参数说明）",
            "",
            f"- 训练图像约 **{tr.get('images', 0)}** 张，训练实例约 **{tr.get('instances', 0)}** 个。",
            f"- 训练集平均每图约 **{tr.get('avg_per_image', 0)}** 个棉苗，单图最多 **{tr.get('max_per_image', 0)}** 个。",
            "- 因此不能简单按“小样本图像集”处理，而应按“**图像数量有限、实例数量充足、目标密集**”设置 `max_det=1000` 等参数。",
            f"- 小目标（归一化面积<{SMALL_AREA_RATIO:.0%}）占比约 **{tr.get('small_target_ratio', 0):.2%}**。",
            "",
        ]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset_density.json").write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "dataset_density_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(splits, indent=2, ensure_ascii=False))
    print(f"\n报告: {OUT_DIR / 'dataset_density_report.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""消融实验共享工具：配置读取、训练/验证/推理参数字典、实验标志。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "ablation_config.yaml"
# 传入 model.train() 的键（Ultralytics 检测训练支持）
TRAIN_HYP_KEYS = (
    "lrf",
    "weight_decay",
    "warmup_epochs",
    "patience",
    "mosaic",
    "close_mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "fliplr",
    "flipud",
    "erasing",
    "amp",
    "deterministic",
    "val",
    "workers",
)

TRAIN_TOP_KEYS = (
    "imgsz",
    "optimizer",
    "lr0",
    "seed",
    "pretrained",
    "resume",
    "exist_ok",
    "plots",
    "verbose",
    "max_det",
    "conf",
    "iou",
)


def load_config(path: Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_pretrained_weights(cfg: dict, mode: str) -> None:
    pw = cfg.get("pretrained_weights")
    if mode == "random" and pw not in (None, "", "null"):
        raise RuntimeError(
            f"mode=random 时 pretrained_weights 必须为 null/空，当前为: {pw!r}。"
            " 防止误加载预训练权重。"
        )


def apply_mode(cfg: dict, mode: str) -> dict:
    cfg = dict(cfg)
    if mode == "random":
        cfg["init_mode"] = "random"
        cfg["pretrained"] = False
        cfg["resume"] = False
    elif mode == "pretrained":
        cfg["init_mode"] = "pretrained"
        cfg["pretrained"] = True
        cfg["resume"] = False
        pw = cfg.get("pretrained_weights")
        if not pw:
            raise RuntimeError("mode=pretrained 需要在 ablation_config.yaml 中设置 pretrained_weights")
    else:
        raise ValueError(f"未知 mode: {mode}")
    check_pretrained_weights(cfg, mode)
    return cfg


def resolve_data_path(cfg: dict) -> str:
    p = Path(cfg["data"])
    return str(ROOT / p if not p.is_absolute() else p)


def build_train_kwargs(cfg: dict, exp_cfg: dict, args: Any) -> dict:
    """构建 model.train() 参数；训练/验证阶段使用 val_* 与 val_max_det。"""
    kw: dict[str, Any] = {
        "data": resolve_data_path(cfg),
        "imgsz": int(cfg["imgsz"]),
        "epochs": int(getattr(args, "epochs", None) or cfg["epochs"]),
        "batch": int(getattr(args, "batch", None) or cfg["batch"]),
        "optimizer": cfg["optimizer"],
        "lr0": float(cfg["lr0"]),
        "seed": int(cfg["seed"]),
        "pretrained": bool(cfg["pretrained"]),
        "resume": bool(cfg["resume"]),
        "exist_ok": bool(cfg["exist_ok"]),
        "device": str(getattr(args, "device", None) or cfg["device"]),
        "box_loss_type": exp_cfg["box_loss_type"],
        "use_center_guidance": bool(exp_cfg["use_center_guidance"]),
        "center_gain": float(exp_cfg.get("center_gain", cfg["center_gain"])),
        "aux_gain": float(exp_cfg.get("aux_gain", cfg.get("aux_gain", 0.25))),
        "kd_gain": float(exp_cfg.get("kd_gain", cfg.get("kd_gain", 0.0))),
        "plots": not getattr(args, "no_plots", False),
        "verbose": bool(cfg.get("verbose", True)),
        "max_det": int(cfg.get("val_max_det", cfg.get("max_det", 1000))),
        "conf": float(cfg.get("val_conf", 0.001)),
        "iou": float(cfg.get("val_iou", 0.7)),
    }
    for k in TRAIN_HYP_KEYS:
        if k in cfg and cfg[k] is not None:
            kw[k] = cfg[k]
    if cfg.get("auto_augment") is None:
        kw["auto_augment"] = None
    elif "auto_augment" in cfg:
        kw["auto_augment"] = cfg["auto_augment"]
    return kw


def build_val_kwargs(cfg: dict, args: Any | None = None) -> dict:
    return {
        "data": resolve_data_path(cfg),
        "split": str(cfg.get("eval_split", "val")),
        "imgsz": int(cfg["imgsz"]),
        "batch": int(getattr(args, "batch", None) or cfg["batch"]) if args else int(cfg["batch"]),
        "device": str(getattr(args, "device", None) or cfg["device"]) if args else str(cfg["device"]),
        "conf": float(cfg.get("val_conf", 0.001)),
        "iou": float(cfg.get("val_iou", 0.7)),
        "max_det": int(cfg.get("val_max_det", cfg.get("max_det", 1000))),
        "plots": False,
    }


def build_predict_kwargs(cfg: dict, overrides: dict | None = None) -> dict:
    kw = {
        "conf": float(cfg.get("predict_conf", 0.15)),
        "iou": float(cfg.get("predict_iou", 0.5)),
        "max_det": int(cfg.get("predict_max_det", cfg.get("max_det", 1000))),
        "imgsz": int(cfg["imgsz"]),
    }
    if overrides:
        kw.update(overrides)
    return kw


def resolve_model_yaml(model_path: str, use_gcg: bool = False) -> Path:
    """解析模型 YAML；因子 C 启用时由基础 YAML 自动生成 GCGDetect 变体。"""
    p = Path(model_path)
    base = (ROOT / p if not p.is_absolute() else p).resolve()
    if not base.is_file():
        raise FileNotFoundError(f"YAML 不存在: {base}")
    if not use_gcg:
        return base

    text = base.read_text(encoding="utf-8")
    if "GCGDetect" in text:
        return base
    if ", Detect, [nc]" not in text:
        raise RuntimeError(f"无法在 {base} 中将 Detect 替换为 GCGDetect")

    gcg_text = text.replace(", Detect, [nc]", ", GCGDetect, [nc]")
    cache_dir = ROOT / "outputs" / "model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{base.stem}-gcg.yaml"
    if not cached.exists() or cached.read_text(encoding="utf-8") != gcg_text:
        cached.write_text(gcg_text, encoding="utf-8")
    return cached


def exp_display_name(exp_key: str, exp_cfg: dict) -> str:
    mapping = {
        "ablation_A_yolo11n": "A YOLOv11n",
        "ablation_B_yolo11n_p2": "B YOLOv11n + P2",
        "ablation_C_midlite_nop2": "C YOLOv11n + Improved Fusion",
        "ablation_D_yolo11n_gcg_sawiou": "D YOLOv11n + GCG + SA-WIoU v3",
        "ablation_E_midlite_p2safe": "E YOLOv11n + P2 + Improved Fusion",
        "ablation_F_yolo11n_p2_gcg_sawiou": "F YOLOv11n + P2 + GCG + SA-WIoU v3",
        "ablation_G_midlite_nop2_gcg_sawiou": "G YOLOv11n + Improved Fusion + GCG + SA-WIoU v3",
        "ablation_H_midlite_p2safe_gcg_sawiou": "H YOLOv11n + P2 + Improved Fusion + GCG + SA-WIoU v3",
        "baseline": "YOLOv11n",
        "p2": "YOLOv11n + P2",
        "drf_safpn": "YOLOv11n + DRF-SAFPN (3-scale)",
        "sa_wiou_gcg": "YOLOv11n + SA-WIoU v3 + GCG",
        "p2_drf_safpn": "YOLOv11n + P2 + DRF-SAFPN",
        "p2_sa_wiou_gcg": "YOLOv11n + P2 + SA-WIoU v3 + GCG",
        "drf_safpn_sa_wiou_gcg": "YOLOv11n + DRF-SAFPN + SA-WIoU v3 + GCG",
        "full": "Full (P2 + DRF-SAFPN + SA-WIoU + GCG)",
        "full_lite_rmax12": "Lightweight Full + P2 Detect + reg_max=12",
        "full_lite_sharedhead_rmax12": "Lightweight Full + SharedLiteGCGDetect + reg_max=12",
        "midlite_p2safe_r12_v1": "MidLite P2Safe R12-v1",
        "midlite_p2safe_r12_v2": "MidLite P2Safe R12-v2",
        "midlite_p2safe_r12_v3": "MidLite P2Safe R12-v3",
        "full_lite_sharedhead_rmax12_fastadd_v1": "Sharedhead rmax12 + FastAdd-v1",
        "full_lite_sharedhead_rmax12_p2headslim_v1": "Sharedhead rmax12 + P2HeadSlim-v1",
        "full_lite_sharedhead_rmax12_fastadd_p2headslim_v1": "Sharedhead rmax12 + FastAdd + P2HeadSlim-v1",
        "full_lite_sharedhead_rmax12_convfriendly_v1": "Sharedhead rmax12 + ConvFriendly-v1",
        "final_lite_p2_r12_v1": "Final Lite P2 R12-v1",
        "final_lite_p2_r12_v2": "Final Lite P2 R12-v2",
        "p2auxkd_p3p5_r12": "P2-AuxKD P3/P4/P5 R12",
        "p2auxkd_p3inject_r12": "P2-AuxKD P3Inject R12",
        "p2auxkd_distill_r12": "P2-AuxKD Distill R12",
        "p2auxkd_distill_lightneck_r12": "P2-AuxKD Distill LightNeck R12",
        "full_lite_sharedhead_rmax12_p2direct_slimneck": "Sharedhead rmax12 + P2Direct + SlimNeck",
        "full_lite_sharedhead_rmax12_p2direct_semislimneck": "Sharedhead rmax12 + P2Direct + SemiSlimNeck",
        "full_lite_sharedhead_rmax12_p2direct_semislimneck_p2sg": "Sharedhead rmax12 + P2Direct + SemiSlimNeck + P2SG",
        "full_lite_sharedhead_rmax12_p2direct_semislimneck_gcg010": "Sharedhead rmax12 + P2Direct + SemiSlimNeck + GCG0.10",
        "full_lite_sharedhead_rmax12_p2direct_semislimneck_p2sg_gcg010": "Sharedhead rmax12 + P2Direct + SemiSlimNeck + P2SG + GCG0.10",
        "full_lite_sharedhead_rmax12_p2direct_balancedneck_b": "Sharedhead rmax12 + P2Direct + BalancedNeck-B",
        "full_lite_sharedhead_rmax12_p2direct_balancedneck_a": "Sharedhead rmax12 + P2Direct + BalancedNeck-A",
        "full_lite_sharedhead_rmax12_p2direct_balancedneck_c": "Sharedhead rmax12 + P2Direct + BalancedNeck-C",
        "full_lite_sharedhead_rmax12_p2direct_balancedneck_p2sg": "Sharedhead rmax12 + P2Direct + BalancedNeck-B + P2SG",
        "full_lite_sharedhead_rmax12_p2direct_balance_p2sg_v1": "Sharedhead rmax12 + P2Direct + P2SG-v1",
        "full_lite_sharedhead_rmax12_p2direct_balance_p2sg_v2": "Sharedhead rmax12 + P2Direct + P2SG-v2",
        "full_lite_sharedhead_rmax12_p2_fusion_light_v1": "Sharedhead rmax12 + P2 Fusion Light-v1",
        "full_lite_sharedhead_rmax12_p2_fusion_light_v2": "Sharedhead rmax12 + P2 Fusion Light-v2",
    }
    return mapping.get(exp_key, exp_cfg.get("desc", exp_key))


def exp_table_flags(exp_cfg: dict) -> dict[str, str]:
    """2^3 全因子消融表三列标志。"""
    return {
        "P2": "√" if exp_cfg.get("p2") else "×",
        "DRF-SAFPN": "√" if exp_cfg.get("drf_safpn") else "×",
        "SA-WIoU v3 + Gaussian": "√" if exp_cfg.get("sa_wiou_gcg") else "×",
    }


def read_best_epoch(exp_dir: Path, target_epochs: int) -> tuple[int | None, int | None, str]:
    """从 results.csv 读取 best_epoch 与 actual_epochs。"""
    rf = exp_dir / "results.csv"
    if not rf.exists():
        return None, None, "无 results.csv"
    import csv

    with open(rf, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None, "results.csv 为空"
    actual = int(rows[-1].get("epoch", len(rows)))
    # fitness: 优先 mAP50-95
    best_i, best_map = 0, -1.0
    for i, r in enumerate(rows):
        v = float(r.get("metrics/mAP50-95(B)", 0) or 0)
        if v >= best_map:
            best_map = v
            best_i = i
    best_ep = int(rows[best_i].get("epoch", best_i + 1))
    notes = ""
    if actual < target_epochs:
        notes = f"实际训练 {actual}/{target_epochs} epoch"
    return best_ep, actual, notes

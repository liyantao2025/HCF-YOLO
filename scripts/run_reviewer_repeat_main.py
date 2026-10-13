#!/usr/bin/env python3
"""Run the reviewer-requested multi-seed main experiments from scratch."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_CONFIG = ROOT / "configs" / "reviewer_repeat_main.yaml"


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_config(cfg: dict) -> list[str]:
    errors: list[str] = []
    seeds = cfg.get("seeds", [])
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        errors.append("seeds必须包含至少3个互不相同的随机种子")
    if cfg.get("init_mode") != "random":
        errors.append("init_mode必须为random")
    if cfg.get("pretrained") is not False:
        errors.append("pretrained必须为false")
    if cfg.get("resume") is not False:
        errors.append("resume必须为false")
    if cfg.get("pretrained_weights") not in (None, "", "null"):
        errors.append("pretrained_weights必须为空")
    if cfg.get("eval_split") != "test":
        errors.append("最终评价必须使用test集合")
    data_file = ROOT / cfg.get("data", "")
    if not data_file.is_file():
        errors.append(f"数据配置不存在: {data_file}")
    for key, model_cfg in cfg.get("models", {}).items():
        model_file = ROOT / model_cfg.get("model", "")
        if model_file.suffix.lower() != ".yaml":
            errors.append(f"{key}必须从.yaml结构文件初始化，当前为{model_file}")
        if not model_file.is_file():
            errors.append(f"{key}模型配置不存在: {model_file}")
    return errors


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_kwargs(cfg: dict, model_cfg: dict, seed: int, args: argparse.Namespace) -> dict:
    keys = (
        "imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
        "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "patience",
        "box", "cls", "dfl", "workers", "amp", "deterministic", "val",
        "plots", "verbose", "max_det", "mosaic", "close_mosaic", "mixup",
        "cutmix", "copy_paste", "hsv_h", "hsv_s", "hsv_v", "degrees",
        "translate", "scale", "shear", "perspective", "fliplr", "flipud",
        "erasing",
    )
    kw = {k: cfg[k] for k in keys if k in cfg and cfg[k] is not None}
    kw.update({
        "data": str((ROOT / cfg["data"]).resolve()),
        "epochs": int(args.epochs or cfg["epochs"]),
        "batch": int(args.batch or cfg["batch"]),
        "device": str(args.device or cfg["device"]),
        "seed": int(seed),
        "pretrained": False,
        "resume": False,
        "exist_ok": False,
        "conf": float(cfg["val_conf"]),
        "iou": float(cfg["val_iou"]),
        "box_loss_type": model_cfg["box_loss_type"],
        "use_center_guidance": bool(model_cfg["use_center_guidance"]),
        "center_gain": float(model_cfg.get("center_gain", cfg["center_gain"])),
    })
    kw["auto_augment"] = cfg.get("auto_augment")
    if args.no_plots:
        kw["plots"] = False
    return kw


def read_epoch_info(results_csv: Path) -> tuple[int | None, int | None]:
    if not results_csv.exists():
        return None, None
    with results_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0) or 0))
    return int(best["epoch"]), int(rows[-1]["epoch"])


def run_one(cfg: dict, model_key: str, seed: int, args: argparse.Namespace) -> dict:
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    from ablation_utils import resolve_model_yaml

    model_cfg = cfg["models"][model_key]
    project = (ROOT / cfg["project"]).resolve()
    name = f"{model_key}_seed{seed}"
    run_dir = project / name
    result_file = run_dir / "test_metrics.json"

    if result_file.exists() and not args.overwrite:
        print(f"[跳过] 已完成: {name}")
        return json.loads(result_file.read_text(encoding="utf-8"))
    if run_dir.exists():
        if not (args.overwrite and args.yes):
            raise RuntimeError(f"目录已存在但结果不完整: {run_dir}；确认重跑请加 --overwrite --yes")
        shutil.rmtree(run_dir)

    base_model = (ROOT / model_cfg["model"]).resolve()
    if base_model.suffix.lower() != ".yaml":
        raise RuntimeError(f"禁止从权重启动训练: {base_model}")
    model_source = resolve_model_yaml(str(base_model), use_gcg=bool(model_cfg["use_center_guidance"]))
    set_seed(seed)

    print("=" * 72)
    print(f"训练: {model_cfg['display_name']} | seed={seed} | random initialization")
    print(f"结构: {model_source}")
    print("=" * 72)
    model = YOLO(str(model_source))
    kw = train_kwargs(cfg, model_cfg, seed, args)
    kw.update({"project": str(project), "name": name})
    model.train(**kw)

    actual_dir = Path(model.trainer.save_dir)
    best_weights = actual_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"未生成best.pt: {best_weights}")

    test_model = YOLO(str(best_weights))
    metrics = test_model.val(
        data=str((ROOT / cfg["data"]).resolve()),
        split=cfg["eval_split"],
        imgsz=int(cfg["imgsz"]),
        batch=int(args.batch or cfg["batch"]),
        device=str(args.device or cfg["device"]),
        conf=float(cfg["val_conf"]),
        iou=float(cfg["val_iou"]),
        max_det=int(cfg["max_det"]),
        plots=False,
    )
    rd = metrics.results_dict
    best_epoch, actual_epochs = read_epoch_info(actual_dir / "results.csv")
    params = sum(p.numel() for p in test_model.model.parameters())
    try:
        flops = round(float(get_flops(test_model.model, imgsz=int(cfg["imgsz"]))), 3)
    except Exception:
        flops = None
    result = {
        "Model": model_cfg["display_name"],
        "model_key": model_key,
        "seed": seed,
        "Precision": float(rd.get("metrics/precision(B)", 0)),
        "Recall": float(rd.get("metrics/recall(B)", 0)),
        "mAP50": float(rd.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(rd.get("metrics/mAP50-95(B)", 0)),
        "best_epoch": best_epoch,
        "actual_epochs": actual_epochs,
        "Params/M": round(params / 1e6, 3),
        "FLOPs/G": flops,
        "model_source": str(model_source),
        "weights": str(best_weights),
        "eval_split": cfg["eval_split"],
        "pretrained": False,
        "resume": False,
        "init_mode": "random",
        "torch_version": torch.__version__,
    }
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[完成] {name}: mAP50={result['mAP50']:.4f}, mAP50-95={result['mAP50-95']:.4f}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reviewer-requested repeated main experiments")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--all", action="store_true", help="运行全部模型和随机种子")
    p.add_argument("--model", choices=["yolo11n", "hcf_yolo11n"])
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--check", action="store_true", help="只做静态配置检查")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--yes", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    errors = validate_config(cfg)
    if errors:
        raise SystemExit("配置检查失败:\n- " + "\n- ".join(errors))
    print("配置检查通过：3个随机种子、随机初始化、测试集最终评价、两模型同一训练协议。")
    if args.check:
        return
    if not args.all and args.model is None:
        raise SystemExit("请使用 --all，或用 --model 指定模型（可配合 --seed）。")

    model_keys = list(cfg["models"]) if args.all else [args.model]
    seeds = [args.seed] if args.seed is not None else list(cfg["seeds"])
    unknown = [s for s in seeds if s not in cfg["seeds"]]
    if unknown:
        raise SystemExit(f"seed不在预注册列表中: {unknown}")

    for seed in seeds:
        for model_key in model_keys:
            run_one(cfg, model_key, int(seed), args)

    from summarize_reviewer_repeat_main import summarize
    summarize(config_path)


if __name__ == "__main__":
    main()


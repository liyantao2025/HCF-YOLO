#!/usr/bin/env python3
"""集成测试：数据、训练、验证、推理、剪枝、ONNX。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**dict(__import__("os").environ), "KMP_DUPLICATE_LIB_OK": "TRUE"}


def run(cmd: list[str], name: str) -> bool:
    print(f"\n{'='*60}\n[RUN] {name}\n{' '.join(cmd)}\n{'='*60}")
    r = subprocess.run(cmd, cwd=str(ROOT), env=ENV)
    ok = r.returncode == 0
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main():
    py = sys.executable
    results: dict[str, bool] = {}

    results["data_yaml"] = run(
        [py, str(ROOT / "tests" / "check_data.py")],
        "data.yaml + yolo11n.pt",
    )

    for exp in [
        "baseline", "p2", "drf_safpn", "sa_wiou_gcg",
        "p2_drf_safpn", "p2_sa_wiou_gcg", "drf_safpn_sa_wiou_gcg", "full",
    ]:
        results[f"train_{exp}"] = run(
            [
                py,
                str(ROOT / "scripts" / "run_ablation.py"),
                "--exp",
                exp,
                "--epochs",
                "1",
                "--batch",
                "2",
                "--device",
                "cpu",
                "--fresh",
                "--yes",
                "--no-plots",
            ],
            f"run_ablation {exp} 1epoch",
        )

    w_baseline = ROOT / "runs/ablation/yolo11n_baseline/weights/best.pt"
    w_full = ROOT / "runs/ablation/yolo11n_full/weights/best.pt"

    if w_baseline.exists():
        results["predict_baseline"] = run(
            [
                py,
                "predict_drf_safpn.py",
                "--weights",
                str(w_baseline),
                "--source",
                "dataset/images/test_improved",
                "--conf",
                "0.25",
            ],
            "predict baseline",
        )
    else:
        results["predict_baseline"] = False
        print("[SKIP] predict baseline: no weights")

    if w_baseline.exists():
        results["pruning"] = run(
            [
                py,
                "pruning_drf_safpn.py",
                "--weights",
                str(w_baseline),
                "--save",
                "runs/test_all/prune_baseline.json",
            ],
            "pruning baseline",
        )
    else:
        results["pruning"] = False

    if w_full.exists():
        results["pruning_full"] = run(
            [
                py,
                "pruning_drf_safpn.py",
                "--weights",
                str(w_full),
                "--save",
                "runs/test_all/prune_full.json",
            ],
            "pruning full",
        )
    else:
        results["pruning_full"] = False
        print("[SKIP] pruning full")

    if w_baseline.exists():
        results["onnx"] = run(
            [py, "export_onnx.py", "--weights", str(w_baseline), "--imgsz", "640"],
            "export onnx",
        )
        results["export_trt"] = run(
            [py, "export_trt_fp16.py", "--weights", str(w_baseline), "--imgsz", "640"],
            "export onnx+trt script",
        )
    else:
        results["onnx"] = results["export_trt"] = False

    print("\n" + "=" * 60)
    print("集成测试汇总")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:24} {'PASS' if v else 'FAIL'}")
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n失败项: {failed}")
        sys.exit(1)
    print("\n全部集成测试通过")


if __name__ == "__main__":
    main()

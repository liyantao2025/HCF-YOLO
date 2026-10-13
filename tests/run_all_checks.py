#!/usr/bin/env python3
"""一键运行模块级检查（不含完整训练）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "test_saf.py",
    "test_drf_lsk.py",
    "test_sa_wiou.py",
    "test_center_guidance.py",
    "test_model_forward.py",
    "test_detect_scales.py",
]


def main():
    env = {**dict(__import__("os").environ), "KMP_DUPLICATE_LIB_OK": "TRUE"}
    failed = []
    for t in TESTS:
        p = ROOT / "tests" / t
        print(f"\n>>> {t}")
        r = subprocess.run([sys.executable, str(p)], cwd=str(ROOT), env=env)
        if r.returncode != 0:
            failed.append(t)
    print("\n" + "=" * 50)
    if failed:
        print("失败:", failed)
        sys.exit(1)
    print("全部模块测试通过")


if __name__ == "__main__":
    main()

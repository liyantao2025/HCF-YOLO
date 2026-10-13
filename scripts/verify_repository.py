#!/usr/bin/env python3
"""Run lightweight integrity checks without importing PyTorch or Ultralytics."""

from __future__ import annotations

import ast
import argparse
import csv
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def count_files(path: Path, suffixes: set[str] | None = None) -> int:
    files = (p for p in path.rglob("*") if p.is_file())
    if suffixes is not None:
        files = (p for p in files if p.suffix.lower() in suffixes)
    return sum(1 for _ in files)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(relative: str) -> Path:
    path = ROOT / relative
    if not path.exists():
        raise AssertionError(f"missing required path: {relative}")
    return path


def check_dataset(image_root: str, label_root: str, splits: dict[str, int]) -> None:
    for split, expected in splits.items():
        images = require(f"{image_root}/{split}")
        labels = require(f"{label_root}/{split}")
        image_count = count_files(images, {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
        label_count = count_files(labels, {".txt"})
        if image_count != expected or label_count != expected:
            raise AssertionError(
                f"{split}: expected {expected} images/labels, found {image_count}/{label_count}"
            )
        print(f"[ok] {split}: {image_count} images, {label_count} labels")


def check_python_syntax() -> None:
    paths = [ROOT / "scripts", ROOT / "ultralytics", ROOT / "tests"]
    checked = 0
    for base in paths:
        for path in base.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            checked += 1
    print(f"[ok] parsed {checked} Python files")


def check_repeated_results() -> None:
    path = require("results/repeated_training/repeated_runs_raw.csv")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise AssertionError(f"expected six seed-level rows, found {len(rows)}")
    expected = {
        ("YOLO11n", "42"),
        ("YOLO11n", "3407"),
        ("YOLO11n", "2026"),
        ("HCF-YOLO11n", "42"),
        ("HCF-YOLO11n", "3407"),
        ("HCF-YOLO11n", "2026"),
    }
    observed = {(row["Model"], row["seed"]) for row in rows}
    if observed != expected:
        raise AssertionError("repeated-training model/seed rows do not match the retained protocol")
    print("[ok] six repeated-training rows")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify repository files and, when present, restored datasets.")
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Fail unless the authorized datasets have been restored with the expected split counts.",
    )
    args = parser.parse_args()
    require("ultralytics/__init__.py")
    require("models/yolo11n.yaml")
    require("models/yolo11n-drf-safpn-midlite-p2safe-r12-v1-gcg.yaml")
    require("source_weights/hcf_yolo11n_source_best.pt")
    dataset_roots = (
        ROOT / "dataset/images",
        ROOT / "dataset/labels",
        ROOT / "external_validation/dataset/images",
        ROOT / "external_validation/dataset/labels",
    )
    if all(path.exists() for path in dataset_roots):
        check_dataset("dataset/images", "dataset/labels", {"train_improved": 320, "val": 40, "test": 10})
        check_dataset(
            "external_validation/dataset/images",
            "external_validation/dataset/labels",
            {"train": 842, "val": 98, "test": 100},
        )
    elif args.require_data:
        raise AssertionError("authorized datasets have not been restored; see DATASET_CARD.md")
    else:
        print("[skip] third-party datasets are not distributed; use --require-data after restoring them")
    check_repeated_results()
    check_python_syntax()
    source_weight = require("source_weights/hcf_yolo11n_source_best.pt")
    expected_hash = "3dd25ff509777b53f7cceaf17d6b0da3bc30a144eacdc86a5f20f5214033d20c"
    actual_hash = sha256(source_weight)
    if actual_hash != expected_hash:
        raise AssertionError(f"source checkpoint hash mismatch: {actual_hash}")
    print(f"[ok] source checkpoint SHA-256: {actual_hash}")
    print("REPOSITORY CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

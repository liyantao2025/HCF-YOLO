#!/usr/bin/env python3
"""Check the minimal release and load both published checkpoints."""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import torch
from ultralytics import YOLO, __version__ as ultralytics_version


ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "models/hcf_yolo11n.yaml",
    "models/yolo11n.yaml",
    "weights/hcf_yolo11n_best.pt",
    "weights/yolo11n_best.pt",
    "data.yaml",
    "train.py",
    "predict.py",
    "evaluate.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Local Ultralytics: {ultralytics_version}")

    missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
    if missing:
        raise SystemExit("Missing required files:\n- " + "\n- ".join(missing))

    for name in ("hcf_yolo11n_best.pt", "yolo11n_best.pt"):
        checkpoint = ROOT / "weights" / name
        model = YOLO(str(checkpoint))
        parameters = sum(p.numel() for p in model.model.parameters())
        print(f"[ok] {name}: {parameters:,} parameters")
        print(f"     SHA-256: {sha256(checkpoint)}")

    print("PACKAGE CHECK PASSED")


if __name__ == "__main__":
    main()

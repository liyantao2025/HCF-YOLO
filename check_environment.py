from pathlib import Path
import hashlib
import platform

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
try:
    import tensorrt as trt
    print("TensorRT:", trt.__version__)
except Exception as exc:
    print("TensorRT: NOT AVAILABLE", repr(exc))

weights = ROOT / "source_weights" / "hcf_yolo11n_source_best.pt"
print("Source checkpoint:", weights)
print("Source checkpoint SHA-256:", sha256(weights))

for split, expected in (("train_improved", 320), ("val", 40), ("test", 10)):
    images = list((ROOT / "dataset" / "images" / split).glob("*"))
    labels = list((ROOT / "dataset" / "labels" / split).glob("*.txt"))
    print(f"{split}: images={len(images)}, labels={len(labels)}, expected={expected}")
    if len(images) != expected or len(labels) != expected:
        raise SystemExit(f"Dataset count mismatch in {split}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required")
print("PACKAGE CHECK PASSED")

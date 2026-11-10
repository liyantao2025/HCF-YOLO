# HCF-YOLO

Minimal, runnable release for dense cotton-seedling detection. The repository keeps only two detection models:

- `HCF-YOLO11n`: the proposed model.
- `YOLO11n`: the comparison baseline.

The repository includes the local Ultralytics source snapshot required by the custom HCF-YOLO modules, model definitions, selected checkpoints, and simple scripts for setup checking, training, evaluation, and prediction.

For Chinese instructions, see [README_CN.md](README_CN.md).

## 1. Environment

Recommended training environment:

- Python 3.10
- PyTorch 2.1.2 with CUDA 11.8
- NVIDIA GPU with at least 12 GB memory for batch size 16

Create and activate an environment, install a CUDA-compatible PyTorch build from the official PyTorch instructions, and then run:

```bash
python -m pip install -r requirements.txt
python check_setup.py
```

Do not install another `ultralytics` package. This repository uses the included local `ultralytics/` source tree.

## 2. Dataset layout

Third-party images and annotations are not redistributed. Prepare a standard YOLO detection dataset as follows:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image must have a same-stem `.txt` label file. Each label line uses:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`. The included `data.yaml` defines one class, `cotton`.

## 3. Train

Train HCF-YOLO11n from scratch:

```bash
python train.py --model hcf --data data.yaml --epochs 200 --batch 16 --device 0
```

Train the YOLO11n baseline from scratch with the same main settings:

```bash
python train.py --model yolo11n --data data.yaml --epochs 200 --batch 16 --device 0
```

Use `--check` to construct the selected network without starting training:

```bash
python train.py --model hcf --check
python train.py --model yolo11n --check
```

## 4. Predict

HCF-YOLO11n:

```bash
python predict.py --model hcf --source path/to/image_or_folder
```

YOLO11n:

```bash
python predict.py --model yolo11n --source path/to/image_or_folder
```

Predictions are saved under `runs/predict/` by default.

## 5. Evaluate

```bash
python evaluate.py --model hcf --data data.yaml --split test --device 0
python evaluate.py --model yolo11n --data data.yaml --split test --device 0
```

## 6. Files

```text
models/hcf_yolo11n.yaml       HCF-YOLO11n architecture
models/yolo11n.yaml           YOLO11n baseline architecture
weights/hcf_yolo11n_best.pt   selected HCF-YOLO11n checkpoint
weights/yolo11n_best.pt       selected YOLO11n checkpoint
train.py                      training entry point
predict.py                    prediction entry point
evaluate.py                   evaluation entry point
check_setup.py                installation and checkpoint check
```

File hashes are recorded in `SHA256SUMS.txt`. The pre-cleanup complete experimental archive remains available in Git history under tag `full-reproducibility-v1`.

## 7. Data and software notice

Dataset images and annotations remain subject to their original licenses and are not redistributed here. See [DATA_LICENSE_NOTICE.md](DATA_LICENSE_NOTICE.md). The included Ultralytics-derived code retains its upstream AGPL-3.0 notices.

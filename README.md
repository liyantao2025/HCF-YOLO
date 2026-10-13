# HCF-YOLO: reproducibility repository

This repository contains the local Ultralytics source snapshot, model definitions, experiment scripts, dataset split metadata, selected checkpoints, deployment artifacts, and retained result files used for the HCF-YOLO cotton-seedling detection study. Third-party images and annotations are not redistributed in the public repository; see `DATASET_CARD.md` and `DATA_LICENSE_NOTICE.md`.

For Chinese instructions, see [README_CN.md](README_CN.md).

## What is included

| Directory | Contents |
|---|---|
| `ultralytics/` | Modified local Ultralytics 8.3.249 source snapshot, including LMSFF/DRF-SAFPN, GCG, center loss, and SA-WIoU integration. |
| `models/` | YOLO11n baseline, HCF-YOLO11n, ablation, and lightweight model YAML files. |
| `configs/` | Training, ablation, repeat-training, and PTQ-INT8 configurations. |
| `scripts/` | Training, evaluation, localization, counting, pruning, quantization, visualization, and Raspberry Pi benchmarking scripts. |
| `dataset/` | Configuration and metadata for the 320/40/10 split used by the retained pruning/quantization and Raspberry Pi workflows. Images and labels must be restored from authorized sources. |
| `external_validation/` | The 842/98/100 COT-AD/Cotton-BNYQS source-separated validation protocol, manifests, audits, and preparation scripts. Third-party images and labels are excluded. |
| `weights/` | Selected baseline, HCF-YOLO, and matched GCG-ablation checkpoints. |
| `source_weights/` | Source HCF-YOLO11n checkpoint for the structured-pruning pipeline. |
| `deployment/raspberry_pi/` | NCNN models, benchmark script, and deployment package. Benchmark images are excluded pending redistribution permission. |
| `results/` | Retained machine-readable results from repeated training, external validation, localization/counting, pruning/quantization, Raspberry Pi inference, and ablation experiments. |

## Reproduction environments

### RTX 4090 training and TensorRT evaluation

- Python 3.10.8
- PyTorch 2.1.2+cu118
- CUDA runtime 11.8
- TensorRT 10.6.0.post1
- NumPy 1.26.4
- OpenCV 4.10.0
- Local Ultralytics source version 8.3.249

Install the non-PyTorch dependencies:

```bash
python -m pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build and TensorRT separately for the target server. Do not install a second `ultralytics` package; commands should import the included local source tree.

### Raspberry Pi 5 NCNN evaluation

The retained environment record reports Python 3.14.4, PyTorch 2.14.0+cpu, Ultralytics 8.4.146, OpenCV 5.0.0, and aarch64 CPU inference. Exact package availability depends on the Raspberry Pi OS/Python combination. The raw environment JSON files are under `results/raspberry_pi/raw_original_filenames/`.

## Quick checks

From the repository root:

```bash
python scripts/verify_repository.py
python scripts/run_external_source_validation.py --check
```

The first command verifies the public repository without requiring redistributed datasets. After restoring authorized datasets, run `python scripts/verify_repository.py --require-data` for split-count checks. The second command requires CUDA and restored data because the external-validation script is intended for the RTX 4090 server.

## Main experiments

### Three-seed repeated training from scratch

Both models use random initialization (`pretrained=False`) and seeds 42, 3407, and 2026:

```bash
python scripts/run_reviewer_repeat_main.py --all --fresh --yes
python scripts/summarize_reviewer_repeat_main.py
```

Configuration: `configs/reviewer_repeat_main.yaml`.

### Ablation experiments

```bash
python scripts/run_ablation.py --all --fresh --yes --mode random
python scripts/collect_ablation_results.py
```

Configuration: `configs/ablation_config.yaml`.

### Independent source validation

This protocol trains on 842 COT-AD images, validates on 98 COT-AD images, and evaluates once on 100 untouched Cotton-BNYQS images:

```bash
python scripts/run_external_source_validation.py --model all --seed 42 --epochs 200 --batch 16
```

The final retained two-row result is `results/external_validation/external_test_summary_seed42_final.csv`.

### GCG localization diagnostics

```bash
python scripts/compute_gcg_matched_sawiou_localization.py
```

The matched GCG-off/GCG-on checkpoints are in `weights/`. Aggregate 104-image results supplied by the experiment record are in `results/gcg_localization_full104/`; retained per-image diagnostics are available for the ten-image subset only.

### Structured pruning and TensorRT PTQ-INT8

```bash
chmod +x run_all.sh
nohup bash run_all.sh > full_pipeline_console.log 2>&1 &
```

The latest downloaded comparison CSV is `results/pruning_quantization/quantization_compare.csv`. The FP16 and INT8 `.engine` files were generated on the RTX 4090 server but were not downloaded into this local archive; they should be rebuilt because TensorRT engines are tied to the target GPU/TensorRT environment.

### Raspberry Pi NCNN benchmark

```bash
python deployment/raspberry_pi/rpi_ncnn_benchmark.py \
  --model deployment/raspberry_pi/package/models/hcf_yolo11n_ncnn_model \
  --source deployment/raspberry_pi/package/dataset/images/test \
  --output results/raspberry_pi/reproduced_hcf
```

Use the same command with `yolo11n_baseline_ncnn_model` for the baseline.

## Result provenance

- `results/repeated_training/repeated_runs_raw.csv` records the six seed-level values supplied as the completed experiment results.
- `results/external_validation/external_test_summary_seed42_final.csv` is the retained two-model external-source result. The earlier duplicate four-row download is intentionally excluded.
- `results/pruning_quantization/quantization_compare.csv` is the latest full-dataset RTX 4090 comparison downloaded after FP32, TensorRT FP16, and TensorRT INT8 evaluation.
- `results/raspberry_pi/raw_original_filenames/` preserves the original downloaded CSV/JSON filenames. The two raw model labels were known to be interchanged; the corrected workbook is the authoritative combined table.
- `results/gcg_localization_full104/` and `results/counting_full104/` contain author-confirmed aggregate results; the corresponding per-image logs were not located in the local archive.

See [RESULTS.md](RESULTS.md) for a concise index and [DATASET_CARD.md](DATASET_CARD.md) for split details.

## Integrity and version tracking

The historical Git metadata was not retained. The source snapshot identifier reported in the manuscript is:

`515fc50b01b02e0f1e4f2fbeb857cb3cd4dce2a1186382416d1490f7c463eac8`

This is preserved as provenance metadata and is not the checksum of this newly assembled repository. File-level hashes for this release are listed in `SHA256SUMS.txt` and `MANIFEST.csv`.

## Data and software rights

The included code derives from Ultralytics and retains its upstream AGPL-3.0 notices. No new blanket licence has been asserted for third-party datasets, trained weights, or manuscript result files. Before making the GitHub repository public, confirm redistribution permissions for COT-AD, Cotton-BNYQS, and any locally collected imagery; see [DATA_LICENSE_NOTICE.md](DATA_LICENSE_NOTICE.md).

## Citation

Add the final article DOI, repository DOI, author list, and release tag after acceptance. These fields are intentionally not invented in this archive.

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs outputs/quantized outputs/environment

python check_environment.py | tee logs/00_environment_check.log
nvidia-smi -q > outputs/environment/nvidia_smi_q.txt
python -m pip freeze > outputs/environment/pip_freeze.txt

python scripts/classic_bn_gamma_prune_final_egcg.py \
  --stage sparse_train --target final_egcg \
  --source-weights source_weights/hcf_yolo11n_source_best.pt \
  --data data.yaml --epochs 50 --batch 16 --device 0 --workers 8 \
  2>&1 | tee logs/01_sparse_train.log

python scripts/classic_bn_gamma_prune_final_egcg.py \
  --stage analyze --target final_egcg \
  --source-weights source_weights/hcf_yolo11n_source_best.pt \
  --data data.yaml --device 0 \
  2>&1 | tee logs/02_analyze.log

python scripts/classic_bn_gamma_prune_final_egcg.py \
  --stage prune --target final_egcg --ratio 0.05 \
  --source-weights source_weights/hcf_yolo11n_source_best.pt \
  --data data.yaml --batch 16 --device 0 --workers 8 --exist-ok \
  2>&1 | tee logs/03_prune.log

python scripts/classic_bn_gamma_prune_final_egcg.py \
  --stage finetune --target final_egcg --ratio 0.05 \
  --source-weights source_weights/hcf_yolo11n_source_best.pt \
  --data data.yaml --epochs 100 --batch 16 --device 0 --workers 8 \
  2>&1 | tee logs/04_finetune.log

python scripts/quantize_export_eval.py \
  --stage all \
  --model HCF_YOLO11n_Pruned5=runs/prune/classic_bn_gamma_final_EGCG_ratio005/weights/best.pt \
  --formats fp16,int8 --data data.yaml --split test --imgsz 640 --device 0 \
  --export-batch 1 --val-batch 1 --workers 8 --workspace 4 --fraction 1.0 \
  --conf 0.001 --iou 0.70 --max-det 1000 \
  --out-dir outputs/quantized --csv outputs/quantization_compare.csv --strict \
  2>&1 | tee logs/05_quantize_export_eval.log

find outputs runs/prune/classic_bn_gamma_final_EGCG_ratio005 -type f -print0 \
  | sort -z | xargs -0 sha256sum > outputs/SHA256SUMS.txt

du -h outputs/quantized/* > outputs/quantized_engine_sizes.txt
echo "REPRODUCTION PIPELINE COMPLETED"

# Result-file index

| Manuscript evidence | Repository file | Status |
|---|---|---|
| Three-seed YOLO11n/HCF-YOLO11n training | `results/repeated_training/repeated_runs_raw.csv` and `repeated_runs_summary.csv` | Six author-confirmed completed runs; no significance test is claimed. |
| Main ablation study | `results/ablation/` | Retained local CSV exports. |
| External-source generalization | `results/external_validation/external_test_summary_seed42_final.csv` | Final two-row seed-42 result. |
| AP75 and center-location diagnostics | `results/localization_baseline_vs_hcf_10_images/` | Ten-image per-image evidence retained. |
| Matched GCG contribution | `results/gcg_localization_10_images/` and `results/gcg_localization_full104/` | Ten-image raw diagnostics plus author-confirmed 104-image aggregate. |
| Counting MAE/RMSE | `results/counting_10_images/` and `results/counting_full104/` | Ten-image raw diagnostics plus author-confirmed 104-image aggregate. |
| Pruning and TensorRT quantization | `results/pruning_quantization/quantization_compare.csv` | Latest full-dataset FP32/FP16/INT8 comparison downloaded from RTX 4090 server. |
| Raspberry Pi edge deployment | `results/raspberry_pi/` | Raw CSV/JSON and corrected combined workbook. |

The TensorRT `.engine` artifacts and their build log were not present on the local computer when this repository was assembled. The engine should be regenerated with `run_all.sh`; its hardware/runtime dependence makes rebuilding preferable to treating an engine binary as portable.

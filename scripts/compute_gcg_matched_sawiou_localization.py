"""Evaluate GCG with a matched P2+LMSFF+SA-WIoU checkpoint pair."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compute_ap75_center_error import evaluate, local_data_yaml  # noqa: E402


def main() -> None:
    out_dir = ROOT / "outputs" / "gcg_matched_sawiou_localization_10_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(out_dir / "ultralytics_config"))
    data_yaml = local_data_yaml(out_dir)

    checkpoints = [
        (
            "P2+LMSFF+SA-WIoU (without GCG)",
            ROOT / "runs" / "ablation" / "diagnose_E_plus_SAWIoU_only" / "weights" / "best.pt",
        ),
        (
            "P2+LMSFF+SA-WIoU+GCG",
            ROOT
            / "runs"
            / "ablation"
            / "ablation_H_midlite_p2safe_gcg_sawiou_pretrained_samehyp"
            / "weights"
            / "best.pt",
        ),
    ]
    for _, checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    results = [evaluate(name, checkpoint, data_yaml, out_dir) for name, checkpoint in checkpoints]
    summaries = [result["summary"] for result in results]
    details = [row for result in results for row in result["per_image"]]
    without_gcg, with_gcg = summaries

    delta = {
        "comparison": "matched GCG-on minus GCG-off",
        "test_images": with_gcg["test_images"],
        "delta_ap75": with_gcg["ap75"] - without_gcg["ap75"],
        "delta_ap75_percentage_points": (with_gcg["ap75"] - without_gcg["ap75"]) * 100,
        "delta_mean_center_error_px": (
            with_gcg["mean_center_error_px"] - without_gcg["mean_center_error_px"]
        ),
        "center_error_relative_change_percent": (
            (with_gcg["mean_center_error_px"] - without_gcg["mean_center_error_px"])
            / without_gcg["mean_center_error_px"]
            * 100
        ),
        "delta_mean_normalized_center_error_percentage_points": (
            with_gcg["mean_normalized_center_error_percent"]
            - without_gcg["mean_normalized_center_error_percent"]
        ),
        "delta_matched_targets": with_gcg["matched_targets"] - without_gcg["matched_targets"],
        "delta_mean_matched_iou": with_gcg["mean_matched_iou"] - without_gcg["mean_matched_iou"],
    }

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    with (out_dir / "delta.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(delta))
        writer.writeheader()
        writer.writerow(delta)

    payload = {
        "scope": "matched-checkpoint GCG diagnostic on 10 held-out high-density images",
        "controlled_factors": {
            "epochs": 200,
            "seed": 42,
            "initialization": "same yolo11n.pt checkpoint",
            "box_loss": "SA-WIoU v3",
            "architecture": "P2+LMSFF",
            "only_intended_difference": "GCG disabled versus enabled",
        },
        "models": summaries,
        "gcg_delta": delta,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

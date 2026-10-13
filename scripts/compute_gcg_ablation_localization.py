"""Evaluate the isolated localization contribution of GCG on the local test set.

The two checkpoints share the P2 + LMSFF architecture.  The second checkpoint
adds GCG while retaining CIoU, so the reported delta does not mix in SA-WIoU.
"""

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
    out_dir = ROOT / "outputs" / "gcg_ablation_localization_10_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(out_dir / "ultralytics_config"))
    data_yaml = local_data_yaml(out_dir)

    checkpoints = [
        (
            "P2+LMSFF (without GCG)",
            ROOT
            / "runs"
            / "ablation"
            / "yolo11n_drf_safpn_midlite_p2safe_r12_v1_pretrained_samehyp"
            / "weights"
            / "best.pt",
        ),
        (
            "P2+LMSFF+GCG",
            ROOT / "runs" / "ablation" / "diagnose_E_plus_GCG_only" / "weights" / "best.pt",
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
        "comparison": "P2+LMSFF+GCG minus P2+LMSFF",
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
        "scope": "pilot diagnostic on the 10 locally available held-out test images",
        "models": summaries,
        "gcg_delta": delta,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Prepare a source-isolated external-validation split and audit manifests."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_validation"
COT_ROOT = Path(r"D:\Desktop\基于无人机的实时棉花幼苗检测\datasets")
BNYQS_ROOT = Path(r"D:\Desktop\基于无人机的实时棉花幼苗检测\数据集拓展")
EXCLUDED = {"16092023_00000123.JPG"}
VAL_BATCH = "26082023"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
Image.MAX_IMAGE_PIXELS = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, size: int = 8) -> str:
    with Image.open(path) as im:
        gray = im.convert("L").resize((size + 1, size))
        px = list(gray.getdata())
    value = 0
    for y in range(size):
        for x in range(size):
            value = (value << 1) | (px[y * (size + 1) + x] > px[y * (size + 1) + x + 1])
    return f"{value:0{size * size // 4}x}"


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    index = next(i for i, part in enumerate(parts) if part.lower() == "images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def record(image: Path, source: str, split: str, batch: str) -> dict:
    label = label_for(image)
    if not label.is_file():
        raise FileNotFoundError(f"Missing label: {label}")
    with Image.open(image) as im:
        width, height = im.size
    label_lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    target_lines = [line for line in label_lines if line.split()[0] == "0"]
    return {
        "split": split,
        "source_dataset": source,
        "acquisition_batch": batch,
        "file_name": image.name,
        "relative_path": image.as_posix(),
        "label_path": label.as_posix(),
        "width": width,
        "height": height,
        "annotation_count": len(target_lines),
        "all_annotation_count": len(label_lines),
        "ignored_non_target_count": len(label_lines) - len(target_lines),
        "sha256": sha256(image),
        "dhash64": dhash(image),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cot_images = sorted(p for p in (COT_ROOT / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    cot_images = [p for p in cot_images if p.name not in EXCLUDED]

    train_images = [p for p in cot_images if not p.stem.startswith(VAL_BATCH + "_")]
    val_images = [p for p in cot_images if p.stem.startswith(VAL_BATCH + "_")]
    external_images = []
    for split_dir in ("train", "valid", "test"):
        external_images.extend(
            sorted(p for p in (BNYQS_ROOT / split_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
        )

    if (len(train_images), len(val_images), len(external_images)) != (842, 98, 100):
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train_images)}, val={len(val_images)}, external={len(external_images)}"
        )

    rows_by_split = {
        "train": [record(p, "COT-AD", "train", p.stem.split("_")[0]) for p in train_images],
        "val": [record(p, "COT-AD", "val", p.stem.split("_")[0]) for p in val_images],
        "external_test": [record(p, "Cotton-BNYQS", "external_test", "Cotton-BNYQS") for p in external_images],
    }

    for split, rows in rows_by_split.items():
        write_csv(OUT / f"{split}_manifest.csv", rows)
        (OUT / f"{split}_images.txt").write_text(
            "\n".join(r["relative_path"] for r in rows) + "\n", encoding="utf-8"
        )

    all_rows = [r for rows in rows_by_split.values() for r in rows]
    write_csv(OUT / "all_splits_manifest.csv", all_rows)

    # Materialize a YOLO-compatible dataset without duplicating the large image
    # payload. Images are hard-linked on the same D: volume; labels are copied
    # while retaining only class 0 (cotton). Cotton-BNYQS class 1 is weed and is
    # outside the single-class scope of this manuscript.
    dataset_root = OUT / "dataset"
    for split, rows in rows_by_split.items():
        yolo_split = "test" if split == "external_test" else split
        image_dir = dataset_root / "images" / yolo_split
        label_dir = dataset_root / "labels" / yolo_split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source_image = Path(row["relative_path"])
            target_image = image_dir / source_image.name
            if not target_image.exists():
                os.link(source_image, target_image)
            source_label = Path(row["label_path"])
            target_lines = [
                line for line in source_label.read_text(encoding="utf-8").splitlines()
                if line.strip() and line.split()[0] == "0"
            ]
            (label_dir / f"{source_image.stem}.txt").write_text(
                "\n".join(target_lines) + ("\n" if target_lines else ""), encoding="utf-8"
            )

    exact = []
    by_sha: dict[str, list[dict]] = {}
    for row in all_rows:
        by_sha.setdefault(row["sha256"], []).append(row)
    for digest, rows in by_sha.items():
        splits = {r["split"] for r in rows}
        if len(splits) > 1:
            exact.append({"type": "sha256", "distance": 0, "left": rows[0]["relative_path"], "right": rows[1]["relative_path"], "splits": "/".join(sorted(splits))})

    near = []
    split_pairs = (("train", "val"), ("train", "external_test"), ("val", "external_test"))
    for left_name, right_name in split_pairs:
        for left in rows_by_split[left_name]:
            for right in rows_by_split[right_name]:
                distance = hamming(left["dhash64"], right["dhash64"])
                if distance <= 3:
                    near.append({
                        "type": "dhash64", "distance": distance,
                        "left": left["relative_path"], "right": right["relative_path"],
                        "splits": f"{left_name}/{right_name}",
                    })
    audit_rows = exact + near
    audit_path = OUT / "cross_split_duplicate_audit.csv"
    fields = ["type", "distance", "left", "right", "splits"]
    with audit_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    stats = []
    for split, rows in rows_by_split.items():
        stats.append({
            "split": split,
            "source_dataset": rows[0]["source_dataset"],
            "images": len(rows),
            "annotations": sum(int(r["annotation_count"]) for r in rows),
            "acquisition_batches": len({r["acquisition_batch"] for r in rows}),
        })
    write_csv(OUT / "split_statistics.csv", stats)

    excluded = [{
        "file_name": "16092023_00000123.JPG",
        "source_dataset": "COT-AD",
        "annotation_count": 94,
        "reason": "Near-duplicate adjacent frame removed before model development",
    }]
    write_csv(OUT / "excluded_images.csv", excluded)

    manifest_files = sorted(OUT.glob("*_manifest.csv"))
    with (OUT / "split_manifest_sha256.txt").open("w", encoding="utf-8") as f:
        for path in manifest_files:
            f.write(f"{sha256(path)}  {path.name}\n")

    yaml = (
        f"path: {str(dataset_root.resolve()).replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 1\nnames:\n  0: cotton\n"
    )
    (OUT / "external_validation.yaml").write_text(yaml, encoding="utf-8")

    print(f"train={len(train_images)}, val={len(val_images)}, external_test={len(external_images)}")
    print(f"cross_split_exact_duplicates={len(exact)}, dhash_candidates={len(near)}")
    print(OUT)


if __name__ == "__main__":
    main()

# Dataset card

## Task

Single-class object detection of cotton seedlings. Labels use normalized YOLO text format:

```text
class_id x_center y_center width height
```

The retained class mapping is `0: cotton`.

## Protocol A: retained training/pruning set

Expected location after the user restores authorized data: `dataset/`

| Split | Images | Label files |
|---|---:|---:|
| `train_improved` | 320 | 320 |
| `val` | 40 | 40 |
| `test` | 10 | 10 |

This protocol is used by the retained pruning/quantization package and the ten-image localization/counting diagnostics.

## Protocol B: source-separated external validation

Expected location after the user restores authorized data: `external_validation/dataset/`

| Split | Images | Source role |
|---|---:|---|
| `train` | 842 | COT-AD training data from five acquisition batches |
| `val` | 98 | COT-AD held-out acquisition batch |
| `test` | 100 | Untouched Cotton-BNYQS external-source test data |

The external test contains 38,150 cotton boxes. Cotton-BNYQS weed annotations are excluded. Images in this release are the pre-letterboxed 640×640 versions used by the server package. The split manifests, exclusion record, duplicate audit, and manifest hashes are stored alongside the dataset.

## Known exclusion

The image `16092023_00000123.JPG` was removed before the final dataset protocol and did not participate in model training.

## Redistribution

Dataset provenance does not itself establish redistribution permission. Therefore, the public repository excludes third-party images and annotations. Users must obtain the datasets from their authorized sources and reconstruct the directory structure described above. Confirm the original terms before redistributing any reconstructed dataset.

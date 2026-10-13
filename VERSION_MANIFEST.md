# Version and provenance manifest

- Project: HCF-YOLO cotton seedling detection
- Source basis: local Ultralytics 8.3.249 snapshot
- Historical Git commit: unavailable because the original Git metadata was not retained
- Manuscript source-snapshot SHA-256: `515fc50b01b02e0f1e4f2fbeb857cb3cd4dce2a1186382416d1490f7c463eac8`
- Main training initialization: random, `pretrained=False`
- Repeated-training seeds: 42, 3407, 2026
- Main server: NVIDIA GeForce RTX 4090/4090D, 24 GB
- Python: 3.10.8
- PyTorch: 2.1.2+cu118
- CUDA runtime: 11.8
- TensorRT: 10.6.0.post1
- NumPy: 1.26.4
- OpenCV: 4.10.0
- Input size: 640×640
- Main optimizer: Adam
- Main epochs: 200
- Initial learning rate: 0.001
- Final learning-rate factor: 0.01
- Weight decay: 0.0005
- Warm-up: 3 epochs
- Evaluation `max_det`: 1000

The file-level hashes generated for this assembled release are stored separately in `SHA256SUMS.txt` and `MANIFEST.csv`.

# HCF-YOLO 复现仓库说明

本仓库已汇总论文复现所需的主要源码、模型配置、训练与评估脚本、两套数据协议、代表性权重、树莓派 NCNN 部署文件和现有结果记录。

## 目录说明

- `ultralytics/`：本研究使用的 Ultralytics 8.3.249 本地修改版源码。
- `models/`：YOLO11n、HCF-YOLO11n 与消融模型配置。
- `configs/`：消融、三随机种子重复训练和 PTQ-INT8 配置。
- `scripts/`：训练、评估、定位/计数指标、剪枝、量化和树莓派测试脚本。
- `dataset/`：剪枝量化与树莓派流程保留的 320/40/10 数据划分。
- `external_validation/`：842 张训练、98 张验证和 100 张独立来源测试图像，以及划分清单和哈希记录。
- `weights/`：基线、完整模型和 GCG 匹配对照权重。
- `source_weights/`：剪枝流程入口权重。
- `deployment/raspberry_pi/`：两个 NCNN 模型、测试图像和基准测试程序。
- `results/`：论文相关的表格、CSV、JSON 与部署结果。

## 建议上传方式

本目录中单个文件均小于 GitHub 的 100 MB 单文件限制，可以使用 Git 命令直接提交。仓库总体积较大，推荐使用 Git 命令而不是网页逐个拖放。

```bash
git init
git add .
git commit -m "Initial reproducibility release"
git branch -M main
git remote add origin <你的GitHub仓库地址>
git push -u origin main
```

## 上传前必须确认

1. COT-AD、Cotton-BNYQS 及自采图像是否允许二次公开分发。
2. 是否需要公开训练权重和树莓派 NCNN 模型。
3. 在 `README.md` 中补充最终论文 DOI、作者信息和正式仓库版本号。
4. 根据你的授权决定软件、数据和权重各自采用的许可证；当前没有替你虚构许可证。

## 重要结果说明

- 三随机种子结果已按真实训练值整理，不是模拟数据。
- 独立来源验证只保留最终两行结果，已去除先前重复的第二组。
- 最新量化结果来自完整数据集上的 FP32、TensorRT FP16 与 TensorRT INT8 对比。
- 树莓派原始 CSV 文件名曾发生模型标签互换，因此以校正后的中文汇总表为准；原文件仍保留以便追溯。
- 104 张图像的 GCG 定位和计数结果仅保留了最终汇总值，本地没有找到逐图日志，因此已在结果说明中如实标注。

## 完整性校验

```bash
python scripts/verify_repository.py
```

稿件中使用的完整 64 位源码快照标识为：

`515fc50b01b02e0f1e4f2fbeb857cb3cd4dce2a1186382416d1490f7c463eac8`

本次汇总仓库的逐文件 SHA-256 位于 `SHA256SUMS.txt` 和 `MANIFEST.csv`。

# 返修第三项：主模型多随机种子重复实验

## 目标

回应两位审稿人关于单次训练结果、随机初始化波动、均值和标准差缺失的问题。仅重复论文的主要模型对：YOLO11n和HCF-YOLO11n。

## 预注册实验设计

- 随机种子：42、3407、2026。
- 每个种子训练YOLO11n与HCF-YOLO11n，共6次独立训练。
- 两个模型均从YAML结构随机初始化，不加载预训练权重，不断点续训。
- 训练集、验证集和测试集划分保持不变。
- 训练期间使用验证集选择best.pt；最终指标统一在测试集上计算。
- 训练参数与原主实验保持一致：200 epochs、imgsz=640、batch=16、Adam、lr0=0.001、lrf=0.01、weight_decay=0.0005、warmup=3 epochs。
- 数据增强、置信度阈值、NMS IoU阈值和max_det均与原实验一致。
- 报告Precision、Recall、mAP50和mAP50-95的均值±样本标准差，并计算同种子配对差值及其95%置信区间。

## 服务器运行

先检查：

```bash
python scripts/run_reviewer_repeat_main.py --check
```

正式运行全部6组：

```bash
python -u scripts/run_reviewer_repeat_main.py --all --device 0
```

如果需要分开运行，可使用：

```bash
python -u scripts/run_reviewer_repeat_main.py --model yolo11n --device 0
python -u scripts/run_reviewer_repeat_main.py --model hcf_yolo11n --device 0
```

程序会跳过已经生成test_metrics.json的完整实验，不会误删已有结果。仅当某目录不完整且确定需要重跑时，才使用`--overwrite --yes`。

## 输出位置

- 单次训练：`runs/reviewer_repeat_main/<model>_seed<seed>/`
- 原始汇总：`outputs/reviewer_repeat_main/repeated_runs_raw.csv`
- 均值与标准差：`outputs/reviewer_repeat_main/repeated_runs_summary.csv`
- 同种子配对差值：`outputs/reviewer_repeat_main/paired_differences.csv`
- 论文可读表：`outputs/reviewer_repeat_main/repeated_runs_summary.md`


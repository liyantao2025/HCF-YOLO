# DRF-SAFPN-YOLO 消融实验参数总结

> 配置文件：`configs/ablation_config.yaml`  
> 训练入口：`scripts/run_ablation.py`  
> 结果汇总：`scripts/collect_ablation_results.py`  
> 更新说明：正式消融默认 **从零随机初始化**（`pretrained=False`, `resume=False`）

---

## 1. 快速启动

```bash
# 全部消融（200 epoch，从零训练）
python scripts/run_ablation.py --all --fresh --yes

# 单组实验
python scripts/run_ablation.py --exp full --fresh --yes

# 汇总结果
python scripts/collect_ablation_results.py
```

命令行可覆盖项：`--epochs`、`--batch`、`--device`、`--mode`、`--no-plots`。

---

## 2. 初始化与公平性参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `init_mode` | `random` | 初始化模式；正式消融使用随机初始化 |
| `pretrained` | `false` | 不加载 COCO / 历史 `.pt` 权重 |
| `resume` | `false` | 禁止从 `last.pt` 续训 |
| `exist_ok` | `false` | 禁止静默覆盖已有实验目录 |
| `seed` | `42` | 全局随机种子（每组实验开始前重置） |
| `pretrained_weights` | `yolo11n.pt` | 仅 `--mode pretrained` 时生效，默认不使用 |

**模式说明**

| 模式 | 命令 | 用途 |
|------|------|------|
| 模式 1（推荐） | `--mode random` | 全部从 `models/*.yaml` 随机初始化，论文消融公平 |
| 模式 2 | `--mode pretrained` | 统一 COCO 预训练；结构变更模型无法完全一致初始化 |

**约束**

- 模型入口必须为 `models/*.yaml`，禁止 `yolo11n.pt` / `best.pt` / `last.pt`
- 若目录下已有 `weights/best.pt` 或 `last.pt`，须 `--fresh --yes` 或更换实验名

---

## 3. 训练超参数（消融脚本显式设置）

| 参数 | 值 | 来源 |
|------|-----|------|
| `data` | `data.yaml` | `ablation_config.yaml` |
| `project` | `runs/ablation` | `ablation_config.yaml` |
| `epochs` | **200** | `ablation_config.yaml`（CLI 可改） |
| `batch` | **16** | `ablation_config.yaml`（CLI 可改） |
| `imgsz` | **640** | `ablation_config.yaml` |
| `optimizer` | **Adam** | `ablation_config.yaml` |
| `lr0` | **0.001** | `ablation_config.yaml` |
| `device` | **"0"** | `ablation_config.yaml`（CLI 可改） |
| `center_gain` | **0.1** | GCG 中心引导损失权重 λ |
| `plots` | `true` | 训练曲线与标签图（`--no-plots` 关闭） |
| `verbose` | `true` | 打印详细日志 |

---

## 4. Ultralytics 默认参数（脚本未改，八组相同）

以下参数由 `ultralytics/cfg/default.yaml` 提供，`run_ablation.py` 未覆盖：

### 4.1 学习率与优化

| 参数 | 值 |
|------|-----|
| `lrf` | 0.01（最终 LR = lr0 × lrf = 1e-5） |
| `momentum` | 0.937 |
| `weight_decay` | 0.0005 |
| `warmup_epochs` | 3.0 |
| `warmup_momentum` | 0.8 |
| `warmup_bias_lr` | 0.1 |
| `cos_lr` | false |

### 4.2 损失权重

| 参数 | 值 | 说明 |
|------|-----|------|
| `box` | 7.5 | 边界框损失增益 |
| `cls` | 0.5 | 分类损失增益 |
| `dfl` | 1.5 | Distribution Focal Loss 增益 |
| `nbs` | 64 | 名义 batch size（损失归一化） |

> 各实验的 `box_loss_type` 与 `use_center_guidance` 见第 6 节，会覆盖 default 中对应项。

### 4.3 训练策略

| 参数 | 值 |
|------|-----|
| `patience` | 100（早停） |
| `amp` | true（混合精度，GPU 有效） |
| `deterministic` | true |
| `val` | true（每 epoch 验证） |
| `close_mosaic` | 10（最后 10 epoch 关闭 mosaic） |
| `workers` | 8（CLI 未改；CPU 调试可手动改 trainer） |
| `fraction` | 1.0（使用全部训练集） |

### 4.4 数据增强

| 参数 | 值 |
|------|-----|
| `mosaic` | 1.0 |
| `mixup` | 0.0 |
| `cutmix` | 0.0 |
| `fliplr` | 0.5 |
| `flipud` | 0.0 |
| `hsv_h` / `hsv_s` / `hsv_v` | 0.015 / 0.7 / 0.4 |
| `degrees` | 0.0 |
| `translate` | 0.1 |
| `scale` | 0.5 |
| `shear` | 0.0 |
| `perspective` | 0.0 |
| `auto_augment` | randaugment |

### 4.5 验证 / NMS

| 参数 | 值 |
|------|-----|
| `split` | val |
| `iou` | 0.7 |
| `max_det` | 300 |
| `conf`（val） | 0.001 |

---

## 5. 数据集参数（`data.yaml`）

| 项 | 值 |
|----|-----|
| 根路径 | `D:/python_pro/yolov11/dataset` |
| 训练集 | `images/train_improved`（320 张） |
| 验证集 | `images/val`（50 张） |
| 测试集 | `images/test_improved`（40 张） |
| 类别数 `nc` | **1** |
| 类别名 | `cotton` |

---

## 6. 2³ 全因子消融实验对照（8 组）

| 实验键 | 输出目录 | 模型 YAML | P2 (A) | DRF-SAFPN (B) | SA-WIoU+GCG (C) |
|--------|----------|-----------|:------:|:-------------:|:---------------:|
| `baseline` | `yolo11n_baseline` | `models/yolo11n.yaml` | × | × | × |
| `p2` | `yolo11n_p2` | `models/yolo11n-p2.yaml` | √ | × | × |
| `drf_safpn` | `yolo11n_drf_safpn` | `models/yolo11n-drf-safpn-3scale.yaml` | × | √ | × |
| `sa_wiou_gcg` | `yolo11n_sa_wiou_gcg` | `models/yolo11n.yaml` + GCG 头 | × | × | √ |
| `p2_drf_safpn` | `yolo11n_p2_drf_safpn` | `models/yolo11n-drf-safpn.yaml` | √ | √ | × |
| `p2_sa_wiou_gcg` | `yolo11n_p2_sa_wiou_gcg` | `models/yolo11n-p2.yaml` + GCG 头 | √ | × | √ |
| `drf_safpn_sa_wiou_gcg` | `yolo11n_drf_safpn_sa_wiou_gcg` | `models/yolo11n-drf-safpn-3scale.yaml` + GCG 头 | × | √ | √ |
| `full` | `yolo11n_full` | `models/yolo11n-drf-safpn.yaml` + GCG 头 | √ | √ | √ |

> **因子 C**：`sa_wiou_gcg=true` 时由 `resolve_model_yaml()` 将基础 YAML 的 `Detect` 自动替换为 `GCGDetect`（缓存于 `outputs/model_cache/`）。

### 6.1 四尺度特征图尺寸（输入 640×640）

| 层 | Stride | 特征图大小 | 通道（n 缩放后约） |
|----|--------|------------|-------------------|
| P2 | 4 | 160×160 | 32 |
| P3 | 8 | 80×80 | 64 |
| P4 | 16 | 40×40 | 128 |
| P5 | 32 | 20×20 | 256 |

### 6.2 参数量（nc=1，结构检查值）

| 实验 | Params |
|------|--------|
| baseline | 2,590,035 |
| p2 | 2,666,820 |
| drf_safpn (3-scale) | 2,739,387 |
| sa_wiou_gcg | 2,610,709 |
| p2_drf_safpn | 2,762,512 |
| p2_sa_wiou_gcg | 2,672,038 |
| drf_safpn_sa_wiou_gcg | 2,760,061 |
| full | 2,767,730 |

### 6.3 损失函数差异

| 类型 | 适用实验 | 要点 |
|------|----------|------|
| `ciou` | baseline, p2, drf_safpn, p2_drf_safpn | 标准 CIoU 回归损失 |
| `sa_wiou_v3` + GCG | sa_wiou_gcg, p2_sa_wiou_gcg, drf_safpn_sa_wiou_gcg, full | WIoU v3 + 中心引导（因子 C） |

**因子 C（SA-WIoU v3 + Gaussian Center Guidance）**

- C=√ 的 4 组均设 `box_loss_type=sa_wiou_v3` 且 `use_center_guidance=true`
- GCGDetect 头：三尺度组 GCG 作用于 stride 8、16；四尺度组作用于 stride 4、8
- 中心损失：`BCEWithLogitsLoss`，权重 `center_gain=0.1`

---

## 7. 输出路径与日志

| 内容 | 路径 |
|------|------|
| 单组训练结果 | `runs/ablation/<实验名>/` |
| 最佳权重 | `runs/ablation/<实验名>/weights/best.pt` |
| 最后权重 | `runs/ablation/<实验名>/weights/last.pt` |
| 初始化记录 | `runs/ablation/<实验名>/init_settings.txt` |
| 训练参数快照 | `runs/ablation/<实验名>/args.yaml` |
| 指标曲线 | `runs/ablation/<实验名>/results.csv` |
| 汇总 CSV | `outputs/ablation_results.csv` |
| 汇总 Markdown | `outputs/ablation_results.md` |
| 公平性报告 | `outputs/ablation_fairness_report.md` |

---

## 8. 命令行参数一览（`run_ablation.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--config` | `configs/ablation_config.yaml` | 配置文件路径 |
| `--exp` | — | 单组实验键 |
| `--only` | — | 同 `--exp` |
| `--all` | false | 运行全部 8 组 |
| `--skip` | [] | `--all` 时跳过指定组 |
| `--mode` | `random` | `random` / `pretrained` |
| `--fresh` | false | 删除旧实验目录 |
| `--yes` | false | 与 `--fresh` 联用，免确认 |
| `--epochs` | 200 | 覆盖配置 |
| `--batch` | 16 | 覆盖配置 |
| `--device` | `0` | 覆盖配置 |
| `--no-plots` | false | 关闭绘图 |

---

## 9. 论文表格引用建议

**表：消融实验训练设置**

| 项目 | 设置 |
|------|------|
| 输入尺寸 | 640×640 |
| 训练轮数 | 200 |
| Batch size | 16 |
| 优化器 | Adam |
| 初始学习率 | 0.001 |
| 权重初始化 | 随机（YAML 构建，`pretrained=False`） |
| 随机种子 | 42 |
| 数据集 | 棉花苗期，单类别 |

**表：2³ 消融变量**

| 模型 | P2 (A) | DRF-SAFPN (B) | SA-WIoU v3 + Gaussian (C) |
|------|:------:|:-------------:|:-------------------------:|
| baseline | | | |
| +P2 | ✓ | | |
| +DRF-SAFPN | | ✓ | |
| +SA-WIoU+GCG | | | ✓ |
| +P2+DRF-SAFPN | ✓ | ✓ | |
| +P2+SA-WIoU+GCG | ✓ | | ✓ |
| +DRF-SAFPN+SA-WIoU+GCG | | ✓ | ✓ |
| Full | ✓ | ✓ | ✓ |

---

## 10. 修改参数的方法

1. **改全局训练参数**：编辑 `configs/ablation_config.yaml` 中 `epochs`、`batch`、`lr0` 等。
2. **改单组损失 / 结构**：编辑同文件 `experiments.<键>` 下 `box_loss_type`、`model` 等。
3. **临时覆盖**：命令行 `--epochs 1 --batch 2 --device cpu`。
4. **改 Ultralytics 默认增强 / 早停等**：在 `run_ablation.py` 的 `model.train(...)` 中追加对应关键字，或扩展 `ablation_config.yaml` 后传入。

---

## 11. 相关文件索引

```
configs/ablation_config.yaml    # 消融主配置
data.yaml                       # 数据集
models/yolo11n.yaml              # baseline / sa_wiou_gcg（+GCG 头）
models/yolo11n-p2.yaml           # p2 / p2_sa_wiou_gcg（+GCG 头）
models/yolo11n-drf-safpn-3scale.yaml  # drf_safpn / drf_safpn_sa_wiou_gcg（+GCG 头）
models/yolo11n-drf-safpn.yaml    # p2_drf_safpn / full（+GCG 头）
scripts/run_ablation.py
scripts/collect_ablation_results.py
ultralytics/cfg/default.yaml    # 未显式覆盖的默认超参
```

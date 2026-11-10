# HCF-YOLO 简明使用说明

这是一个经过精简并实际核验的棉花幼苗检测项目。仓库只保留两个模型：

- `HCF-YOLO11n`：本文提出的模型；
- `YOLO11n`：用于对比的基线模型。

仓库内已经包含模型结构、最优权重、本地 Ultralytics 源码、训练程序、测试程序和直接预测程序。

## 一、部署环境

推荐服务器环境：

- Python 3.10；
- PyTorch 2.1.2；
- CUDA 11.8；
- NVIDIA GPU；使用 `batch=16` 时建议显存不低于 12 GB。

先根据服务器CUDA版本安装相应的PyTorch，再在项目根目录执行：

```bash
python -m pip install -r requirements.txt
python check_setup.py
```

`check_setup.py`会检查必要文件，并实际加载HCF-YOLO11n和YOLO11n两个权重。仓库已经包含修改后的`ultralytics`源码，不要再安装另一个Ultralytics软件包，以免覆盖自定义模块。

## 二、放置数据集

由于原始公开数据集受各自许可条款约束，仓库不重复上传图像和标注。请将YOLO格式数据按下面的结构放置：

```text
dataset/
├── images/
│   ├── train/    # 训练图像
│   ├── val/      # 验证图像
│   └── test/     # 测试图像
└── labels/
    ├── train/    # 训练标注
    ├── val/      # 验证标注
    └── test/     # 测试标注
```

每张图像都应有一个同名的`.txt`标注文件。每一行的格式为：

```text
类别编号 目标中心x 目标中心y 目标宽度 目标高度
```

后四个数值均归一化到0–1。本项目只有一个类别`cotton`，相关配置已经写入`data.yaml`。

## 三、训练模型

从零开始训练HCF-YOLO11n：

```bash
python train.py --model hcf --data data.yaml --epochs 200 --batch 16 --device 0
```

采用相同主要设置从零开始训练YOLO11n：

```bash
python train.py --model yolo11n --data data.yaml --epochs 200 --batch 16 --device 0
```

如果只想确认模型能否正确构建，而不开始训练：

```bash
python train.py --model hcf --check
python train.py --model yolo11n --check
```

训练结果默认保存在`runs/train/`下。

## 四、直接预测

使用HCF-YOLO11n预测单张图像或整个文件夹：

```bash
python predict.py --model hcf --source 你的图像或文件夹路径
```

使用YOLO11n预测：

```bash
python predict.py --model yolo11n --source 你的图像或文件夹路径
```

例如：

```bash
python predict.py --model hcf --source dataset/images/test
```

预测图像默认保存在`runs/predict/`下。可通过`--conf 0.25`修改置信度阈值，通过`--device cpu`使用CPU预测。

## 五、测试精度

在测试集上评估HCF-YOLO11n：

```bash
python evaluate.py --model hcf --data data.yaml --split test --device 0
```

评估YOLO11n：

```bash
python evaluate.py --model yolo11n --data data.yaml --split test --device 0
```

## 六、核心文件

```text
models/hcf_yolo11n.yaml       HCF-YOLO11n结构
models/yolo11n.yaml           YOLO11n结构
weights/hcf_yolo11n_best.pt   HCF-YOLO11n最优权重
weights/yolo11n_best.pt       YOLO11n基线权重
train.py                      训练程序
predict.py                    直接预测程序
evaluate.py                   精度评估程序
check_setup.py                环境与文件检查程序
```

所有发布文件的SHA-256值记录在`SHA256SUMS.txt`中。精简前的完整实验复现资料仍保留在Git标签`full-reproducibility-v1`中，可用于追溯。

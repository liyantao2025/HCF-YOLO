# 棉花幼苗检测系统

这个系统把改进后的 YOLO11 棉花幼苗检测模型封装成一个本地 Web 应用，并提供批量检测命令行入口。默认权重为：

`ablation/yolo11n_full/weights/best.pt`

默认推理参数来自 `configs/ablation_config.yaml`：

- `conf=0.15`
- `iou=0.5`
- `imgsz=640`
- `max_det=1000`

## 安装依赖

项目已包含本地 `ultralytics` 代码。推理环境还需要 PyTorch、OpenCV、Flask、Pillow、PyYAML 等依赖：

```bash
pip install -r requirements-cotton-system.txt
```

如果需要 GPU 推理，请按当前 CUDA 版本安装匹配的 PyTorch。

## 启动 Web 系统

```bash
python run_cotton_seedling_system.py --host 127.0.0.1 --port 7860
```

打开：

`http://127.0.0.1:7860`

可选参数：

```bash
python run_cotton_seedling_system.py ^
  --weights ablation/yolo11n_full/weights/best.pt ^
  --device 0 ^
  --port 7860
```

也可以用环境变量指定模型：

```bash
set COTTON_MODEL_WEIGHTS=ablation/yolo11n_full/weights/best.pt
set COTTON_DEVICE=0
python run_cotton_seedling_system.py
```

## 功能

- 单张棉田图像上传检测
- 内置测试集示例图快速检测
- 置信度、IoU、输入尺寸、最大检测数在线调整
- 输出幼苗数量、平均置信度、推理耗时
- 保存标注图、检测框 CSV、完整 JSON
- 提供 `/api/detect` REST 接口

结果默认保存到：

`outputs/cotton_seedling_system/`

## REST API

```bash
curl -X POST http://127.0.0.1:7860/api/detect ^
  -F image=@dataset/images/test/result_4_97_png.rf.cee5573126d6a64b0cf87daf44934be2.jpg ^
  -F conf=0.15 ^
  -F iou=0.5 ^
  -F max_det=1000
```

返回字段包括：

- `count`：检测到的棉花幼苗数量
- `avg_confidence`：平均置信度
- `inference_ms`：推理耗时
- `boxes`：每个目标的类别、置信度、`xyxy`、`xywh` 和面积
- `annotated_url`、`csv_url`、`json_url`：结果文件地址

## 批量检测

```bash
python -m cotton_seedling_system.cli ^
  --source dataset/images/test ^
  --weights ablation/yolo11n_full/weights/best.pt ^
  --output outputs/cotton_seedling_system/batch_test ^
  --device 0
```

批量检测会为每张图生成独立的结果目录，并在输出目录生成 `summary.csv`。

## 常见问题

1. 页面提示 `missing inference dependencies: torch, cv2`

   当前 Python 环境缺少推理依赖。执行 `pip install -r requirements-cotton-system.txt`，或切换到已安装 PyTorch/OpenCV 的环境。

2. 页面提示 `model weights not found`

   检查默认权重是否存在，或使用 `--weights` 指定实际 `.pt` 文件。

3. 检测数量明显被截断

   棉花幼苗属于高密度小目标场景，建议保持 `max_det=1000`。

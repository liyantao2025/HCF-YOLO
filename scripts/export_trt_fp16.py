#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 ONNX 并尝试构建 TensorRT FP16 Engine。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ultralytics import YOLO


def try_build_tensorrt(onnx_path: Path, engine_path: Path, imgsz: int = 640):
    try:
        import tensorrt as trt
    except ImportError:
        print("未安装 TensorRT，跳过 engine 构建。可执行: pip install tensorrt")
        print(f"已生成 ONNX: {onnx_path}")
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, (1, 3, imgsz, imgsz), (1, 3, imgsz, imgsz), (1, 3, imgsz, imgsz))
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("TensorRT engine 构建失败")
        return False
    engine_path.write_bytes(engine)
    print(f"TensorRT FP16 engine: {engine_path}")
    return True


def benchmark_onnx(onnx_path: Path, imgsz: int = 640, warmup: int = 10, runs: int = 50):
    try:
        import onnxruntime as ort
    except ImportError:
        print("未安装 onnxruntime，跳过延迟测试")
        return
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp: x})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {inp: x})
    dt = (time.perf_counter() - t0) / runs * 1000
    print(f"ONNX 平均延迟: {dt:.2f} ms | FPS: {1000/dt:.1f} | 输入: {imgsz}x{imgsz}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="runs/detect/drf_safpn_full/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()

    weights = Path(args.weights)
    out_dir = weights.parent
    onnx_path = out_dir / "best.onnx"
    engine_path = out_dir / "best_fp16.engine"

    model = YOLO(str(weights))
    model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, dynamic=False, simplify=True)
    # ultralytics 默认导出到 weights 同目录
    exported = weights.with_suffix(".onnx")
    if exported.exists():
        onnx_path = exported

    print(f"ONNX: {onnx_path}")
    benchmark_onnx(onnx_path, args.imgsz)
    ok = try_build_tensorrt(onnx_path, engine_path, args.imgsz)
    print(f"FP16 Engine: {'已生成' if ok else '未生成（见上方提示）'}")


if __name__ == "__main__":
    main()

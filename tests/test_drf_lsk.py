#!/usr/bin/env python3
"""DRF-LSK 动态感受野模块单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics.nn.modules.drf_safpn import DRF_LSK


def main():
    x = torch.randn(2, 128, 160, 160, requires_grad=True)
    m = DRF_LSK(128, 128)

    print("=== DRF-LSK 单元测试 ===")
    print("输入 shape:", x.shape)

    reduced = m.reduce(x)
    b1, b2, b3 = m.dw1(reduced), m.dw2(reduced), m.dw3(reduced)
    print("分支输出 shapes:", b1.shape, b2.shape, b3.shape)
    assert b1.shape == b2.shape == b3.shape

    merged = m.act(m.bn(torch.cat([b1, b2, b3], dim=1)))
    gates = torch.softmax(m.spatial_gate(merged), dim=1)
    print("attention 权重 shape:", gates.shape)
    print("attention 范围:", float(gates.min()), float(gates.max()))
    print("残差 gamma 初值:", float(m.gamma.item()))

    out = m(x)
    print("输出 shape:", out.shape)
    assert out.shape == x.shape

    loss = out.sum()
    loss.backward()
    print("backward: OK")
    assert not torch.isnan(out).any()
    print("使用残差连接:", m.c2 == m.c1)
    print("=== DRF-LSK 测试通过 ===")


if __name__ == "__main__":
    main()

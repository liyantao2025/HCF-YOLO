#!/usr/bin/env python3
"""SAF 尺度自适应融合模块单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics.nn.modules.drf_safpn import SAF


def main():
    x1 = torch.randn(2, 64, 160, 160, requires_grad=True)
    x2 = torch.randn(2, 128, 80, 80, requires_grad=True)
    x3 = torch.randn(2, 256, 40, 40, requires_grad=True)
    chs = [64, 128, 256]
    out_ch = 128
    saf = SAF(out_ch, n=3, chs=chs)

    print("=== SAF 单元测试 ===")
    print("输入 shapes:", [t.shape for t in [x1, x2, x3]])

    # 手动检查对齐与权重（hook 内部逻辑）
    ref = x3
    from ultralytics.nn.modules.drf_safpn import _align_features

    aligned = _align_features([x1, x2, x3], ref)
    print("对齐后 shapes (以 x3 为参考):", [t.shape for t in aligned])

    out = saf([x1, x2, x3])
    print("输出 shape:", out.shape)
    assert out.shape == (2, out_ch, 40, 40), f"期望 (2,{out_ch},40,40)，得到 {out.shape}"

    # 权重 softmax 检查
    with torch.no_grad():
        projected = []
        for i, f in enumerate(aligned):
            projected.append(saf.proj[i](saf.adapt[i](f)))
        weights = torch.cat([saf.weight_convs[i](projected[i]) for i in range(3)], dim=1)
        w = torch.softmax(weights, dim=1)
        print("融合权重 shape:", w.shape)
        print("权重范围:", float(w.min()), float(w.max()))
        print("各分支权重和 (batch 0):", w[0].flatten().tolist())
        sum_per_spatial = w.sum(dim=1)
        print("Softmax 后和是否≈1:", float(sum_per_spatial.mean()), "(mean over batch)")

    loss = out.sum()
    loss.backward()
    print("backward: OK")
    assert not torch.isnan(out).any(), "输出含 nan"
    assert x1.grad is not None, "输入梯度未回传"
    print("=== SAF 测试通过 ===")


if __name__ == "__main__":
    main()

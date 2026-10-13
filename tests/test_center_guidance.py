#!/usr/bin/env python3
"""Gaussian Center Guidance 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics.nn.modules.head import GCGDetect
from ultralytics.utils.center_loss import build_gaussian_heatmap, center_guidance_loss


def main():
    print("=== Gaussian Center Guidance 单元测试 ===")
    batch, nc = 2, 1
    ch = (64, 128, 256, 512)
    strides = torch.tensor([4.0, 8.0, 16.0, 32.0])

    p2 = torch.randn(batch, ch[0], 160, 160)
    p3 = torch.randn(batch, ch[1], 80, 80)
    p4 = torch.randn(batch, ch[2], 40, 40)
    p5 = torch.randn(batch, ch[3], 20, 20)
    feats = [p2, p3, p4, p5]

    head = GCGDetect(nc=nc, ch=ch)
    head.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])
    head.init_center_branches()
    assert head.center_layers == (0, 1)
    head.train()
    head.export = False
    feats_train = [t.clone() for t in feats]
    _ = head(feats_train)
    assert head.center_preds is not None
    print("训练模式 center_preds:", [p.shape for p in head.center_preds])
    assert head.center_preds[0].shape == (batch, 1, 160, 160)
    assert head.center_preds[1].shape == (batch, 1, 80, 80)

    targets = torch.tensor(
        [
            [0, 0, 0.5, 0.5, 0.05, 0.05],
            [1, 0, 0.25, 0.25, 0.1, 0.1],
        ]
    )
    hm_p2 = build_gaussian_heatmap(targets, batch, 160, 160, 4.0, p2.device, p2.dtype)
    hm_p3 = build_gaussian_heatmap(targets, batch, 80, 80, 8.0, p3.device, p3.dtype)
    print("P2 heatmap:", hm_p2.shape, "max=", float(hm_p2.max()), "min=", float(hm_p2.min()))
    print("P3 heatmap:", hm_p3.shape, "max=", float(hm_p3.max()), "min=", float(hm_p3.min()))
    assert float(hm_p2.max()) <= 1.0 + 1e-5
    assert float(hm_p2.min()) >= 0.0

    cl = center_guidance_loss(head.center_preds, head.center_layers, strides, targets, batch, torch.tensor(640.0))
    cl.backward()
    print("center_loss:", float(cl.item()), "backward: OK")

    head.eval()
    feats_eval = [t.detach().clone() for t in feats]
    _ = head(feats_eval)
    assert head.center_preds is None, "eval 模式应关闭 center 分支"
    print("eval 模式 center_preds: None")
    print("=== GCG 测试通过 ===")


if __name__ == "__main__":
    main()

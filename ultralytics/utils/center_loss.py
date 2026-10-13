# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Gaussian Center Guidance 辅助监督：训练期中心热力图 BCE 损失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_gaussian_heatmap(
    targets: torch.Tensor,
    batch_size: int,
    feat_h: int,
    feat_w: int,
    stride: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """根据 GT 框中心生成 Gaussian 热力图（逐像素取 max 融合多目标）。

    Args:
        targets: (N, 6) -> [batch_idx, cls, x,y,w,h] 归一化 xywh。
        stride: 当前特征层相对原图的下采样步长。
    """
    heatmap = torch.zeros(batch_size, 1, feat_h, feat_w, device=device, dtype=dtype)
    if targets.numel() == 0:
        return heatmap

    for t in targets:
        b = int(t[0].item())
        cx = t[2] * feat_w
        cy = t[3] * feat_h
        bw = t[4] * feat_w
        bh = t[5] * feat_h
        radius = max(1.0, 0.5 * min(bw, bh))
        sigma = radius / 3.0 + 1e-6
        ys = torch.arange(feat_h, device=device, dtype=dtype).view(-1, 1)
        xs = torch.arange(feat_w, device=device, dtype=dtype).view(1, -1)
        gauss = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
        heatmap[b, 0] = torch.maximum(heatmap[b, 0], gauss)
    return heatmap


def center_guidance_loss(
    pred_centers: list[torch.Tensor],
    center_layers: tuple[int, ...],
    strides: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    imgsz: torch.Tensor,
) -> torch.Tensor:
    """多尺度中心引导 BCE 损失（仅训练）。"""
    device = pred_centers[0].device
    dtype = pred_centers[0].dtype
    loss = torch.zeros(1, device=device, dtype=dtype)
    n_terms = 0

    # targets: batch 拼接 [idx, cls, xywh normalized]
    if targets.shape[0] == 0:
        return loss.squeeze()

    for pred, layer_i in zip(pred_centers, center_layers):
        stride = float(strides[layer_i])
        _, _, h, w = pred.shape
        hm_gt = build_gaussian_heatmap(targets, batch_size, h, w, stride, device, dtype)
        loss = loss + F.binary_cross_entropy_with_logits(pred, hm_gt, reduction="mean")
        n_terms += 1
    return loss / max(n_terms, 1)

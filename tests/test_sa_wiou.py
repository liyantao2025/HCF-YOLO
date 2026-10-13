#!/usr/bin/env python3
"""SA-WIoU v3 尺度感知损失单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics.utils.loss import _bbox_iou_loss


def make_boxes(cx, cy, w, h):
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)


def main():
    print("=== SA-WIoU v3 单元测试 ===")
    iou_mean = torch.tensor(0.5)
    eps = 1e-7

    # 相似 IoU 偏差：pred 相对 target 偏移 5px
    small_tgt = make_boxes(100, 100, 20, 20)
    large_tgt = make_boxes(320, 320, 160, 160)
    small_pred = small_tgt.clone()
    small_pred[:, :2] -= 5
    small_pred[:, 2:] += 5
    large_pred = large_tgt.clone()
    large_pred[:, :2] -= 5
    large_pred[:, 2:] += 5

    _, _ = _bbox_iou_loss(small_pred, small_tgt, "wiou_v3", iou_mean, eps)
    loss_small_w, _ = _bbox_iou_loss(small_pred, small_tgt, "sa_wiou_v3", iou_mean, eps)
    loss_large_w, _ = _bbox_iou_loss(large_pred, large_tgt, "sa_wiou_v3", iou_mean, eps)

    # scale_weight 显式计算
    w2, h2 = 20.0, 20.0
    area_s = w2 * h2
    w2l, h2l = 160.0, 160.0
    area_l = w2l * h2l
    mean_area = (area_s + area_l) / 2
    sw_s = min(3.0, max(1.0, mean_area / area_s))
    sw_l = min(3.0, max(1.0, mean_area / area_l))
    print(f"小框 scale_weight: {sw_s:.4f}")
    print(f"大框 scale_weight: {sw_l:.4f}")
    assert sw_s > sw_l, "小框 scale_weight 应大于大框"

    print(f"小框 SA-WIoU loss: {loss_small_w.item():.6f}")
    print(f"大框 SA-WIoU loss: {loss_large_w.item():.6f}")

    # 可微分
    pred = small_pred.clone().requires_grad_(True)
    loss, _ = _bbox_iou_loss(pred, small_tgt, "sa_wiou_v3", iou_mean, eps)
    loss.sum().backward()
    assert pred.grad is not None and not torch.isnan(pred.grad).any()
    print("backward: OK, 无 nan")
    print("=== SA-WIoU 测试通过 ===")


if __name__ == "__main__":
    main()

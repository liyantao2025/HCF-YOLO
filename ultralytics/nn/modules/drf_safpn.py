# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DRF-SAFPN 论文改进模块：SAF 尺度自适应融合 + DRF-LSK 动态感受野选择。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = (
    "SAF",
    "LiteSAF",
    "FastSAF",
    "AlignAddFusion",
    "P2ToP3Adapter",
    "P2SemanticGate",
    "DRF_LSK",
    "LightDRF_LSK",
    "C2f_DRF_LSK",
)


def _align_features(feats: list[torch.Tensor], ref: torch.Tensor) -> list[torch.Tensor]:
    """将多尺度特征对齐到参考特征的空间尺寸。"""
    return [f if f.shape[-2:] == ref.shape[-2:] else F.interpolate(f, size=ref.shape[-2:], mode="nearest") for f in feats]


class SAF(nn.Module):
    """尺度自适应融合模块 (Scale-Adaptive Fusion).

    替代部分 Concat：通过 GAP + 1×1 卷积为各尺度分配 Softmax 权重，自适应融合浅层细节与高层语义，
    减轻传统 Concat 导致的小目标边缘/纹理信息被削弱问题。
    """

    def __init__(self, c2: int, n: int = 2, chs: list | None = None, e: float = 0.5, debug_shapes: bool = False):
        """
        Args:
            c2: 输出通道数（经 width 缩放后由 parse_model 传入）。
            n: 输入分支数量（与 YAML 中 from 列表长度一致）。
            chs: 各输入分支通道列表，由 parse_model 注入。
            e: 中间通道压缩比例。
            debug_shapes: 是否在 forward 中打印 shape（调试用）。
        """
        super().__init__()
        self.c2 = int(c2)
        self.n = int(n)
        self.debug_shapes = debug_shapes
        hidden = max(self.c2 // 2, 16)
        chs = chs or [self.c2] * self.n
        self.adapt = nn.ModuleList(
            nn.Identity() if int(c) == self.c2 else Conv(int(c), self.c2, 1, 1) for c in chs
        )
        self.proj = nn.ModuleList(nn.Conv2d(self.c2, self.c2, 1, bias=False) for _ in range(self.n))
        self.post = Conv(self.c2, self.c2, 1, 1)
        # 每个尺度一条权重分支：GAP -> 1×1 -> 标量权重
        self.weight_convs = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.c2, hidden, 1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 1, 1, bias=True),
            )
            for _ in range(self.n)
        )

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if not isinstance(x, list):
            x = [x]
        assert len(x) == self.n, f"SAF 期望 {self.n} 路输入，实际 {len(x)} 路"
        ref = x[-1]
        aligned = _align_features(x, ref)
        # 若通道不一致，先用 1×1 投影（parse_model 应保证 c2 一致；此处做兜底）
        projected = []
        for i, f in enumerate(aligned):
            projected.append(self.proj[i](self.adapt[i](f)))

        weights = torch.cat([self.weight_convs[i](projected[i]) for i in range(self.n)], dim=1)  # B,n,1,1
        weights = torch.softmax(weights, dim=1)
        out = sum(projected[i] * weights[:, i : i + 1] for i in range(self.n))
        out = self.post(out)
        if self.debug_shapes:
            print(f"[SAF] in={[t.shape for t in x]} -> out={out.shape}")
        return out


class DRF_LSK(nn.Module):
    """动态感受野选择模块 (DRF-LSK).

    多膨胀深度卷积分支 + 空间选择注意力，避免单一超大感受野导致小目标响应过度平滑。
  训练稳定：输出 = x + gamma * fused，gamma 初值为 0。
    """

    def __init__(self, c1: int, c2: int | None = None, debug_shapes: bool = False):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c2 or c1)
        self.debug_shapes = debug_shapes
        mid = max(self.c2 // 2, 16)
        self.reduce = Conv(self.c1, mid, 1, 1) if self.c1 != mid else nn.Identity()
        # 多感受野 depthwise 分支
        self.dw1 = nn.Conv2d(mid, mid, 3, padding=1, dilation=1, groups=mid, bias=False)
        self.dw2 = nn.Conv2d(mid, mid, 5, padding=4, dilation=2, groups=mid, bias=False)
        self.dw3 = nn.Conv2d(mid, mid, 5, padding=6, dilation=3, groups=mid, bias=False)
        self.bn = nn.BatchNorm2d(mid * 3)
        self.act = nn.SiLU(inplace=True)
        # LSK-style 空间选择
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(mid * 3, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 3, 1, bias=True),
        )
        self.expand = Conv(mid, self.c2, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.reduce(x)
        b1, b2, b3 = self.dw1(x), self.dw2(x), self.dw3(x)
        merged = self.act(self.bn(torch.cat([b1, b2, b3], dim=1)))
        gates = torch.softmax(self.spatial_gate(merged), dim=1)
        fused = gates[:, 0:1] * b1 + gates[:, 1:2] * b2 + gates[:, 2:3] * b3
        out = self.expand(fused)
        if self.c2 == self.c1:
            out = identity + self.gamma * out
        if self.debug_shapes:
            print(f"[DRF_LSK] {identity.shape} -> {out.shape}")
        return out


class LiteSAF(nn.Module):
    """Lightweight Scale-Adaptive Fusion.

    Keeps the dynamic scale weighting idea of SAF while removing the per-branch
    projection and post-fusion 1x1 convolutions from the original SAF module.
    """

    def __init__(self, c2: int, n: int = 2, chs: list | None = None, debug_shapes: bool = False):
        super().__init__()
        self.c2 = int(c2)
        self.n = int(n)
        self.debug_shapes = debug_shapes
        chs = chs or [self.c2] * self.n
        self.adapt = nn.ModuleList(
            nn.Identity() if int(c) == self.c2 else Conv(int(c), self.c2, 1, 1) for c in chs
        )
        self.weight_convs = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.c2, 1, 1, bias=True)) for _ in range(self.n)
        )

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if not isinstance(x, list):
            x = [x]
        assert len(x) == self.n, f"LiteSAF expects {self.n} inputs, got {len(x)}"
        ref = x[-1]
        aligned = _align_features(x, ref)
        projected = [self.adapt[i](f) for i, f in enumerate(aligned)]
        weights = torch.cat([self.weight_convs[i](projected[i]) for i in range(self.n)], dim=1)
        weights = torch.softmax(weights, dim=1)
        out = sum(projected[i] * weights[:, i : i + 1] for i in range(self.n))
        if self.debug_shapes:
            print(f"[LiteSAF] in={[t.shape for t in x]} -> out={out.shape}")
        return out


class FastSAF(nn.Module):
    """Deployment-friendly scale fusion with static learnable weights."""

    def __init__(self, c2: int, n: int = 2, chs: list | None = None, debug_shapes: bool = False):
        super().__init__()
        self.c2 = int(c2)
        self.n = int(n)
        self.debug_shapes = debug_shapes
        chs = chs or [self.c2] * self.n
        self.adapt = nn.ModuleList(
            nn.Identity() if int(c) == self.c2 else Conv(int(c), self.c2, 1, 1) for c in chs
        )
        self.weights = nn.Parameter(torch.ones(self.n))

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if not isinstance(x, list):
            x = [x]
        assert len(x) == self.n, f"FastSAF expects {self.n} inputs, got {len(x)}"
        ref = x[-1]
        aligned = _align_features(x, ref)
        projected = [self.adapt[i](f) for i, f in enumerate(aligned)]
        weights = torch.softmax(self.weights, dim=0)
        out = sum(projected[i] * weights[i] for i in range(self.n))
        if self.debug_shapes:
            print(f"[FastSAF] in={[t.shape for t in x]} -> out={out.shape}")
        return out


class AlignAddFusion(nn.Module):
    """Fast channel-aligned add fusion for high-resolution neck paths."""

    def __init__(self, c2: int, n: int = 2, chs: list | None = None, debug_shapes: bool = False):
        super().__init__()
        self.c2 = int(c2)
        self.n = int(n)
        self.debug_shapes = debug_shapes
        chs = chs or [self.c2] * self.n
        self.adapt = nn.ModuleList(
            nn.Identity() if int(c) == self.c2 else Conv(int(c), self.c2, 1, 1) for c in chs
        )

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if not isinstance(x, list):
            x = [x]
        assert len(x) == self.n, f"AlignAddFusion expects {self.n} inputs, got {len(x)}"
        ref = x[-1]
        aligned = _align_features(x, ref)
        out = self.adapt[0](aligned[0])
        for i in range(1, self.n):
            out = out + self.adapt[i](aligned[i])
        out = out / self.n
        if self.debug_shapes:
            print(f"[AlignAddFusion] in={[t.shape for t in x]} -> out={out.shape}")
        return out


class P2ToP3Adapter(nn.Module):
    """Light P2-to-P3 detail injection without producing P2 inference outputs."""

    def __init__(self, c2: int, n: int = 2, chs: list | None = None, debug_shapes: bool = False):
        super().__init__()
        self.c2 = int(c2)
        self.n = int(n)
        self.debug_shapes = debug_shapes
        chs = chs or [self.c2] * self.n
        if self.n != 2:
            raise ValueError(f"P2ToP3Adapter expects 2 inputs [P2, P3], got {self.n}")
        self.p2_down = Conv(int(chs[0]), self.c2, 3, 2)
        self.p3_adapt = nn.Identity() if int(chs[1]) == self.c2 else Conv(int(chs[1]), self.c2, 1, 1)
        self.fuse = Conv(self.c2, self.c2, 1, 1)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if not isinstance(x, list):
            x = [x]
        assert len(x) == self.n, f"P2ToP3Adapter expects {self.n} inputs, got {len(x)}"
        p2 = self.p2_down(x[0])
        p3 = self.p3_adapt(x[1])
        if p2.shape[-2:] != p3.shape[-2:]:
            p2 = F.interpolate(p2, size=p3.shape[-2:], mode="nearest")
        out = self.fuse(p2 + p3)
        if self.debug_shapes:
            print(f"[P2ToP3Adapter] in={[t.shape for t in x]} -> out={out.shape}")
        return out


class P2SemanticGate(nn.Module):
    """Tiny P3-guided semantic gate for direct P2 Detect features."""

    def __init__(self, c2: int, n: int = 2, chs: list | None = None, debug_shapes: bool = False):
        super().__init__()
        self.c2 = int(c2)
        self.n = int(n)
        self.debug_shapes = debug_shapes
        chs = chs or [self.c2] * self.n
        if self.n != 2:
            raise ValueError(f"P2SemanticGate expects 2 inputs [P2, P3], got {self.n}")
        self.p2_adapt = nn.Identity() if int(chs[0]) == self.c2 else Conv(int(chs[0]), self.c2, 1, 1)
        self.gate = nn.Conv2d(int(chs[1]), self.c2, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if not isinstance(x, list):
            x = [x]
        assert len(x) == self.n, f"P2SemanticGate expects {self.n} inputs, got {len(x)}"
        p2 = self.p2_adapt(x[0])
        p3 = x[1] if x[1].shape[-2:] == p2.shape[-2:] else F.interpolate(x[1], size=p2.shape[-2:], mode="nearest")
        out = p2 * torch.sigmoid(self.gate(p3))
        if self.debug_shapes:
            print(f"[P2SemanticGate] in={[t.shape for t in x]} -> out={out.shape}")
        return out


class LightDRF_LSK(nn.Module):
    """Lightweight DRF-LSK with a narrower bottleneck and cheap branch selector."""

    def __init__(self, c1: int, c2: int | None = None, e: float = 0.25, debug_shapes: bool = False):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c2 or c1)
        self.debug_shapes = debug_shapes
        mid = max(int(self.c2 * e), 8)
        self.reduce = Conv(self.c1, mid, 1, 1) if self.c1 != mid else nn.Identity()
        self.dw1 = nn.Conv2d(mid, mid, 3, padding=1, dilation=1, groups=mid, bias=False)
        self.dw2 = nn.Conv2d(mid, mid, 3, padding=2, dilation=2, groups=mid, bias=False)
        self.dw3 = nn.Conv2d(mid, mid, 5, padding=4, dilation=2, groups=mid, bias=False)
        self.bn = nn.BatchNorm2d(mid * 3)
        self.act = nn.SiLU(inplace=True)
        self.spatial_gate = nn.Conv2d(mid * 3, 3, 1, bias=True)
        self.expand = Conv(mid, self.c2, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.reduce(x)
        b1, b2, b3 = self.dw1(x), self.dw2(x), self.dw3(x)
        merged = self.act(self.bn(torch.cat([b1, b2, b3], dim=1)))
        gates = torch.softmax(self.spatial_gate(merged), dim=1)
        fused = gates[:, 0:1] * b1 + gates[:, 1:2] * b2 + gates[:, 2:3] * b3
        out = self.expand(fused)
        if self.c2 == self.c1:
            out = identity + self.gamma * out
        if self.debug_shapes:
            print(f"[LightDRF_LSK] {identity.shape} -> {out.shape}")
        return out


class C2f_DRF_LSK(nn.Module):
    """C2f 结构内嵌 DRF-LSK，用于 Neck 特征精炼。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(DRF_LSK(self.c, self.c) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

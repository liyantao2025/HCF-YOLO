#!/usr/bin/env python3
"""检查 Detect 四尺度特征图空间尺寸。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics.nn.tasks import DetectionModel

EXPECTED = {4: [160, 80, 40, 20], 3: [80, 40, 20]}


def main():
    cfg = "yolo11n-drf-safpn.yaml"
    m = DetectionModel(cfg, nc=1, verbose=False)
    detect = m.model[-1]
    detect_indices = detect.f if isinstance(detect.f, list) else [detect.f]

    shapes = {}

    def hook_fn(idx):
        def _hook(module, inp, out):
            if isinstance(inp[0], list):
                for i, t in enumerate(inp[0]):
                    shapes[f"in_{i}"] = tuple(t.shape)
            else:
                shapes[f"layer_{idx}"] = tuple(out.shape) if torch.is_tensor(out) else None

        return _hook

    detect.register_forward_pre_hook(lambda mod, inp: shapes.update({"detect_in": [tuple(t.shape) for t in inp[0]]}))

    m.eval()
    with torch.no_grad():
        m(torch.randn(1, 3, 640, 640))

    print("=== Detect 输入特征 shapes ===")
    if "detect_in" in shapes:
        for i, sh in enumerate(shapes["detect_in"]):
            h, w = sh[-2], sh[-1]
            print(f"  P{i+2 if detect.nl==4 else i+3}: {sh} -> {w}x{h}")

    stride = detect.stride.tolist()
    print("stride:", stride)
    print("nl:", detect.nl)
    exp_h = EXPECTED.get(detect.nl, [])
    if "detect_in" in shapes:
        got_h = [sh[-2] for sh in shapes["detect_in"]]
        if exp_h:
            ok = got_h == exp_h
            print("尺寸匹配预期", exp_h, ":", "OK" if ok else f"FAIL got {got_h}")


if __name__ == "__main__":
    main()

"""Export official DAV2-Small (relative) to fixed-shape ONNX for on-phone QAIRT/SNPE timing (EXPLORE).

Input: normalized RGB, NCHW float32, upright portrait (H x W multiples of 14). Output: relative
disparity [1, H, W]. Shapes: 686x518 (short side 518, the fusion candidate) and 518x392.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / 'artifacts.local/work/rgb-depth-backbone-compare-dev-20261010/upstream/Depth-Anything-V2'
WEIGHT = ROOT / 'artifacts.local/models/depth-anything--Depth-Anything-V2-Small/depth_anything_v2_vits.pth'
OUT = ROOT / 'artifacts.local/work/dav2s-mobile-20261012'
SHAPES = ((686, 518), (518, 392))

sys.path.insert(0, str(UPSTREAM))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
model.eval()
for h, w in SHAPES:
    x = torch.randn(1, 3, h, w)
    path = OUT / f'dav2s_{h}x{w}.onnx'
    torch.onnx.export(model, x, str(path), input_names=['image'], output_names=['disparity'], opset_version=17, dynamo=False)
    print(path, path.stat().st_size)

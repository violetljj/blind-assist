"""Export UniDepth-V2-Small (known-camera variant: image + unprojected rays) to fixed-shape ONNX for on-phone QAIRT/SNPE
timing (EXPLORE, direction A deployability). Uses the official UniDepthV2ONNXcam wrapper from the pinned source.

Input: normalized RGB NCHW float32 (ImageNet mean/std) and unit camera rays NCHW (from the known K); outputs pts_3d,
confidence, intrinsics. Shapes are landscape H x W, multiples of 14:
  392x518  ~203k px, the model's pixels_min (same pixel count as the DAV2-S 518x392 phone run)
  490x644  ~316k px, what `infer` uses for a 640x480 camera image (the resolution evaluated in A1/A2)
Phone variant (2026-11-17, after QAIRT rejected onnx Acos in DepthHead.embed_rays): the ray embedding depends only on the
camera, so it is computed once in PyTorch with the original function and baked into the graph as a constant (one model per
calibrated camera); only pts_3d and confidence are exported, which prunes the unused intrinsics-prediction branch.
Also writes one real input (a SANPO head frame resized) as raw float32 for the phone run, and the PyTorch FP32 outputs
for the accuracy comparison. Usage: unidepth_mobile_export_dev_20261117.py <out dir>
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
A = ROOT / 'artifacts.local'
SRC = A / 'downloads/unidepth-src-8d8cfe4c-20261117/UniDepth-8d8cfe4c7ee15297099983607febf0d4f32eb3d6'
WEIGHTS = A / 'models/lpiccinelli--unidepth-v2-vits14'
SHAPES = ((392, 518), (490, 644))
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

sys.path.insert(0, str(SRC))
from safetensors.torch import load_file  # noqa: E402
from unidepth.models.unidepthv2.export import UniDepthV2ONNXcam  # noqa: E402


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    config = json.loads((WEIGHTS / 'config.json').read_text(encoding='utf-8'))
    config['model']['pixel_encoder']['pretrained'] = None
    model = UniDepthV2ONNXcam(config)
    info = model.load_state_dict(load_file(str(WEIGHTS / 'model.safetensors')), strict=True)
    model.float().eval()
    print('params', sum(p.numel() for p in model.parameters()), info)
    import cv2
    sess = sorted((A / 'downloads/sanpo-real-head-seg-20261117').iterdir())[0]
    img_p = sorted((sess / 'camera_head/left/video_frames').glob('*.png'))[0]
    desc = json.loads((sess / 'description.json').read_text())
    cp = desc['session_camera_details'][desc['session_camera_location'].index('camera_head')]['left_camera_params']
    bgr = cv2.imread(str(img_p))
    for h, w in SHAPES:
        rgb = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1] / 255.
        x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        sx, sy = w / cp['image_width'], h / cp['image_height']
        K = np.array([[cp['fx'] * sx, 0, cp['cx'] * sx], [0, cp['fy'] * sy, cp['cy'] * sy], [0, 0, 1.]])
        vv, uu = np.mgrid[0:h, 0:w] + .5
        r = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((h, w))], 0)
        rays = (r / np.linalg.norm(r, axis=0, keepdims=True))[None].astype(np.float32)
        tag = f'{h}x{w}'
        dl = model.pixel_decoder.depth_layer
        dl.embed_rays = type(dl).embed_rays.__get__(dl)            # restore the original before computing the constant
        with torch.inference_mode():
            pts, conf, intr = model(torch.from_numpy(x), torch.from_numpy(rays))
            emb = {}
            orig = dl.embed_rays
            dl.embed_rays = lambda r: emb.setdefault('e', orig(r))
            model(torch.from_numpy(x), torch.from_numpy(rays))
        const = emb['e'].clone()
        dl.embed_rays = lambda r: const
        with torch.inference_mode():
            pts_b, _, _ = model(torch.from_numpy(x), torch.from_numpy(rays))
        print(tag, 'baked embedding', tuple(const.shape), 'max |diff| vs original', float((pts_b - pts).abs().max()))
        np.save(out / f'torch_pts_{tag}.npy', pts.numpy())
        x.tofile(out / f'input_rgb_{tag}.raw'); rays.tofile(out / f'input_rays_{tag}.raw')
        path = out / f'unidepth_s_cam_{tag}.onnx'
        class Phone(torch.nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m

            def forward(self, rgbs, rays):
                p, c, _ = self.m(rgbs, rays)
                return p, c
        torch.onnx.export(Phone(model), (torch.from_numpy(x), torch.from_numpy(rays)), str(path), input_names=['rgbs', 'rays'],
                          output_names=['pts_3d', 'confidence'], opset_version=17, dynamo=False)
        print(tag, path.stat().st_size, 'depth median', float(np.median(pts[0, 2].numpy())))
    (out / 'source_frame.txt').write_text(f'{img_p}\n', encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])

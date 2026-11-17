"""UniDepth-V2-S phone model for the calibrated Atom upright camera (480x640, K from the mobile-fusion camera.json), at
the 490x644 phone budget chosen in DIRECTION_A3 (input pixels 315,560). EXPLORE, App integration of direction A.

The research path (evalkit Methods.unidepth_s: official `infer`, pixels_min = pixels_max = 315560) resizes a 640x480
(HxW) image to 658x490 and uses rays of K scaled by the infer resize factor. This script runs that exact path on one real
Atom frame and captures the model-resolution inputs (normalized image, rays), the ray embedding and the model-resolution
points; then exports the known-camera wrapper with that embedding baked in (QAIRT has no Acos), checks the export against
the captured points, and writes fixtures for the phone parity test:
  input_rgb_658x490.raw, input_rays_658x490.raw   NCHW float32 model inputs
  torch_pts_658x490.npy                            research model-resolution points (official infer, CUDA FP16 autocast)
  atom_frame_upright.png, research_depth_640x480.npy   source frame and research depth at 640x480 (HxW)
Usage: unidepth_atom_export_20261118.py <capture rgb jpeg> <out dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HOST = Path(__file__).resolve().parent
ROOT = HOST.parents[3]
A = ROOT / 'artifacts.local'
SRC = A / 'downloads/unidepth-src-8d8cfe4c-20261117/UniDepth-8d8cfe4c7ee15297099983607febf0d4f32eb3d6'
WEIGHTS = A / 'models/lpiccinelli--unidepth-v2-vits14'
CAMERA = A / 'work/mobile-fusion-integration-20261012/camera.json'
PIXELS = 315560
sys.path.insert(0, str(HOST)); sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / 'research/active/dtr-r0/nearfield'))


def main(jpeg, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    from atom_camera import AtomCamera
    from PIL import Image
    from rgb_depth_backbone_adapters import load_model
    from safetensors.torch import load_file
    from unidepth.models.unidepthv2.export import UniDepthV2ONNXcam
    cam = json.loads(CAMERA.read_text(encoding='utf-8')); K = np.array(cam['K'], dtype=np.float64)
    raw = cv2.imread(str(jpeg))[:, :, ::-1]
    up = AtomCamera().undistort_upright(raw)                      # 640x480 (HxW) RGB, same remap as the app
    assert up.shape[:2] == (640, 480), up.shape
    cv2.imwrite(str(out / 'atom_frame_upright.png'), up[:, :, ::-1])
    ud, ud_infer, _ = load_model('unidepth', SRC, WEIGHTS); ud = ud.cuda()
    ud.shape_constraints = dict(ud.shape_constraints, pixels_min=PIXELS, pixels_max=PIXELS)
    cap = {}
    dl = ud.pixel_decoder.depth_layer; orig_embed = dl.embed_rays
    dl.embed_rays = lambda r: cap.setdefault('embed', orig_embed(r))
    orig_ed = ud.encode_decode

    def hook(inputs, image_metas):
        cap['image'] = inputs['image'].detach().float().cpu().numpy()
        res = orig_ed(inputs, image_metas)
        cap['rays'] = inputs['rays'].detach().float().cpu().numpy()
        cap['points'] = res[1]['points'].detach().float().cpu().numpy()
        return res
    ud.encode_decode = hook
    depth = ud_infer(Image.fromarray(np.ascontiguousarray(up)), K.astype(np.float32))
    ud.encode_decode = orig_ed; dl.embed_rays = orig_embed
    h, w = cap['image'].shape[2:]
    print('model input', (h, w), 'rays', cap['rays'].shape, 'embed', tuple(cap['embed'].shape), 'depth median', float(np.median(depth)))
    tag = f'{h}x{w}'
    np.save(out / 'research_depth_640x480.npy', depth.astype(np.float32))
    np.save(out / f'torch_pts_{tag}.npy', cap['points'])
    cap['image'].astype(np.float32).tofile(out / f'input_rgb_{tag}.raw'); cap['rays'].astype(np.float32).tofile(out / f'input_rays_{tag}.raw')

    config = json.loads((WEIGHTS / 'config.json').read_text(encoding='utf-8')); config['model']['pixel_encoder']['pretrained'] = None
    model = UniDepthV2ONNXcam(config); model.load_state_dict(load_file(str(WEIGHTS / 'model.safetensors')), strict=True); model.float().eval()
    const = cap['embed'].detach().float().cpu().clone()
    model.pixel_decoder.depth_layer.embed_rays = lambda r: const

    class Phone(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, rgbs, rays):
            p, c, _ = self.m(rgbs, rays)
            return p, c
    x, r = torch.from_numpy(cap['image']), torch.from_numpy(cap['rays'])
    with torch.inference_mode():
        pts, _ = Phone(model)(x, r)
    z_e, z_r = pts[0, 2].numpy(), cap['points'][0, 2]
    rel = np.abs(z_e - z_r) / np.maximum(z_r, 1e-3)
    print('export (CPU FP32) vs research infer (CUDA FP16) z: median rel', float(np.median(rel)), 'p99', float(np.percentile(rel, 99)))
    path = out / f'unidepth_s_atom_{tag}.onnx'
    torch.onnx.export(Phone(model), (x, r), str(path), input_names=['rgbs', 'rays'], output_names=['pts_3d', 'confidence'],
                      opset_version=17, dynamo=False)
    np.save(out / f'export_pts_{tag}.npy', pts.numpy())
    (out / 'export.json').write_text(json.dumps(dict(source_jpeg=str(jpeg), K=K.tolist(), pixels=PIXELS, model_input_hw=[h, w],
                                                     z_rel_median=float(np.median(rel)), z_rel_p99=float(np.percentile(rel, 99)),
                                                     camera_json=str(CAMERA)), indent=1), encoding='utf-8')
    print('onnx', path, path.stat().st_size)


if __name__ == '__main__':
    main(*sys.argv[1:3])

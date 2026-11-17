"""Dataset adapters for the minimal evaluation kit. Each yields core.Frame objects; nothing here reads evaluator truth
beyond loading the reference depth, which only core.regions / core.depth_metrics use.

  sanpo_real_local  local SANPO-Real coverage subset (10 sessions x 5 frames), CREStereo pseudo-GT, RGB, no usable device
                    gravity (2026-11-17 pose check: 8 deg median), CC BY 4.0
  sanpo_head_seg    SANPO-Real head ZED Mini, 89 sessions with human segmentation (sky removed), 1307 frames every 2 s
                    (host/sanpo_head_seg_fetch_20261117.py), same reference and licence as above
  aria_gen2_walks   Aria Gen 2 Pilot walk_0/walk_1, FoundationStereo pseudo-GT (PNG = optical z, checked 2026-11-17),
                    rectified grayscale, gravity-aligned MPS pose, CC BY-NC 4.0
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from core import Frame

ROOT = Path(__file__).resolve().parents[4]
A = ROOT / 'artifacts.local'
SKY = 27


def _sanpo(data, name, only_cam=None):
    W, H = 1280, 720
    for sess in sorted(os.listdir(data)):
        if not (data / sess).is_dir():
            continue
        cams = [c for c in os.listdir(data / sess) if c.startswith('camera') and (only_cam is None or c == only_cam)]
        if not cams:
            continue
        cam = cams[0]; base = data / sess / cam / 'left'
        desc = json.loads((data / sess / 'description.json').read_text())
        # camera details are listed in `session_camera_location` order (chest ZED 2i, head ZED Mini); v0 took [0] for
        # every session, i.e. ZED 2i intrinsics for head sessions (fx 1974 vs 1368) -> heights ~1.16 m instead of ~1.67 m
        p = desc['session_camera_details'][desc['session_camera_location'].index(cam)]['left_camera_params']
        sc = W / p['image_width']
        K = np.array([[p['fx'] * sc, 0, p['cx'] * sc], [0, p['fy'] * sc, p['cy'] * sc], [0, 0, 1.]])
        for fr in sorted(q.name.split('.')[0] for q in (base / 'depth_maps').glob('*.float16.gz')):
            a = np.frombuffer(gzip.decompress((base / 'depth_maps' / f'{fr}.float16.gz').read_bytes()), '<f2')
            z = a[2:].astype(np.float32).reshape(int(a[0]), int(a[1]))
            seg = cv2.resize(cv2.imread(str(base / 'segmentation_masks' / f'{fr}.png'), cv2.IMREAD_UNCHANGED)[:, :, 2], (W, H),
                             interpolation=cv2.INTER_NEAREST)
            z = np.where(np.isfinite(z) & (seg != SKY), z, np.nan)
            img = cv2.resize(cv2.imread(str(base / 'video_frames' / f'{fr}.png')), (W, H), interpolation=cv2.INTER_AREA)
            yield Frame(name, sess, int(fr), int(fr) / 15., img, K, z, 'pseudo_ground_truth', None, None,
                        'head' if cam == 'camera_head' else 'chest', block=sess)


def sanpo_real_local():
    return _sanpo(A / 'datasets/sanpo-real-coverage-20260927', 'sanpo_real_local')


def sanpo_head_seg():
    return _sanpo(A / 'downloads/sanpo-real-head-seg-20261117', 'sanpo_head_seg', 'camera_head')


def aria_gen2_walks(step=15, block_s=30.):
    for seq in ('walk_0', 'walk_1'):
        base = A / 'downloads/aria-gen2-pilot-20261117' / seq / 'x/depth'
        params = json.loads((base / 'pinhole_camera_parameters.json').read_text()); t0 = params[0]['frameTimestampNs']
        for p in params[::step]:
            fu, fv, u0, v0 = p['camera']['Parameters']; i = p['index']
            z = cv2.imread(str(base / 'depth' / f'depth_{i:08d}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.
            z[z <= 0] = np.nan
            gray = cv2.imread(str(base / 'rectified_images' / f'image_{i:08d}.png'), cv2.IMREAD_UNCHANGED)
            down = Rotation.from_quat(p['T_world_camera']['QuaternionXYZW']).as_matrix().T @ np.array([0, 0, -1.])
            t = (p['frameTimestampNs'] - t0) / 1e9
            yield Frame('aria_gen2_walks', seq, i, t, np.repeat(gray[:, :, None], 3, 2),
                        np.array([[fu, 0, u0], [0, fv, v0], [0, 0, 1.]]), z, 'pseudo_ground_truth', down, None, 'head',
                        block=f'{seq}:{int(t // block_s)}')


ADAPTERS = {'sanpo_real_local': sanpo_real_local, 'aria_gen2_walks': aria_gen2_walks, 'sanpo_head_seg': sanpo_head_seg}

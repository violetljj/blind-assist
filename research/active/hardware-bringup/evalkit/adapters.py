"""Dataset adapters for the minimal evaluation kit. Each yields core.Frame objects; nothing here reads evaluator truth
beyond loading the reference depth, which only core.regions / core.depth_metrics use.

  sanpo_real_local  local SANPO-Real coverage subset (10 sessions x 5 frames), CREStereo pseudo-GT, RGB, no usable device
                    gravity (2026-11-17 pose check: 8 deg median), CC BY 4.0
  sanpo_head_seg    SANPO-Real head ZED Mini, 89 sessions with human segmentation (sky removed), 1307 frames every 2 s
                    (host/sanpo_head_seg_fetch_20261117.py), same reference and licence as above
  sanpo_chest_seg   SANPO-Real chest ZED 2i, 138 sessions with human segmentation, none shared with the head subset or the
                    local coverage subset, 1969 frames every 2 s (same fetch script, --camera camera_chest); confirmation data
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


def _sanpo(data, name, only_cam=None, sky_dir=None):
    """sky_dir: automatic sky masks (host/sanpo_auto_sky_20261117.py, 255 = sky) instead of the human segmentation."""
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
            if sky_dir is None:
                sky = cv2.resize(cv2.imread(str(base / 'segmentation_masks' / f'{fr}.png'), cv2.IMREAD_UNCHANGED)[:, :, 2], (W, H),
                                 interpolation=cv2.INTER_NEAREST) == SKY
            else:
                sky = cv2.imread(str(Path(sky_dir) / sess / f'{fr}.png'), cv2.IMREAD_UNCHANGED) > 0
            if sky.shape != z.shape:
                sky = cv2.resize(sky.astype(np.uint8), (z.shape[1], z.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            z = np.where(np.isfinite(z) & ~sky, z, np.nan)
            img = cv2.resize(cv2.imread(str(base / 'video_frames' / f'{fr}.png')), (W, H), interpolation=cv2.INTER_AREA)
            yield Frame(name, sess, int(fr), int(fr) / 15., img, K, z, 'pseudo_ground_truth', None, None,
                        'head' if cam == 'camera_head' else 'chest', block=sess)


def sanpo_real_local():
    return _sanpo(A / 'datasets/sanpo-real-coverage-20260927', 'sanpo_real_local')


def sanpo_head_seg():
    return _sanpo(A / 'downloads/sanpo-real-head-seg-20261117', 'sanpo_head_seg', 'camera_head')


def sanpo_head_seg_autosky():
    """Seen head subset with automatic instead of human sky (acceptance check only)."""
    return _sanpo(A / 'downloads/sanpo-real-head-seg-20261117', 'sanpo_head_seg', 'camera_head',
                  A / 'work/direction-a-confirm-head-20261117/sky_auto_seen_head_seg')


NOSEG_KEEP = A / 'work/direction-a-confirm-head-20261117/session_filter.json'


def sanpo_head_noseg(keep='file'):
    """Unseen head sessions without human segmentation (host/sanpo_head_noseg_fetch_20261117.py), automatic sky.
    keep='file': only sessions passing the per-session intake check (evalkit/head_noseg_session_filter.py); None: all."""
    keep = set(json.loads(NOSEG_KEEP.read_text())['keep']) if keep == 'file' else None
    for f in _sanpo(A / 'downloads/sanpo-real-head-noseg-20261117', 'sanpo_head_noseg', 'camera_head',
                    A / 'work/direction-a-confirm-head-20261117/sky_auto_head_noseg'):
        if keep is None or f.seq in keep:
            yield f


def _adt_floor_y(base, seq, cam_y, dep, down, K):
    """Scene floor height of one ADT sequence from GT depth: per sampled frame the lowest horizontal surface below the
    camera with >= 2000 points within +-3 cm (1 cm bins along gravity), floor_y = cam_y - that distance; the sequence
    value is the median of the candidates within 0.10 m of their 5th percentile (tables and beds sit above the floor).
    Cached in <seq>.floor.json."""
    cache = base / f'{seq}.floor.json'
    if cache.exists():
        return json.loads(cache.read_text())['floor_y']
    from core import rays
    cand = []
    for i in range(0, len(cam_y), 4):
        z = dep[i].astype(np.float32).reshape(-1) / 1000.; ok = (z > .3) & (z < 10)
        h = (rays(np.asarray(K), *dep[i].shape)[ok] * z[ok, None]) @ down[i].astype(float)
        hist, edges = np.histogram(h[h > .5], bins=np.arange(.5, 3., .01))
        sup = np.convolve(hist, np.ones(7, int), 'same')
        good = np.flatnonzero(sup >= 2000)
        if len(good):
            j = good[-1]; j = max(range(max(0, j - 6), min(len(hist), j + 1)), key=lambda q: hist[q])
            cand.append(float(cam_y[i]) - (edges[j] + .005))
    cand = np.array(cand); low = np.percentile(cand, 5)
    floor_y = float(np.median(cand[cand <= low + .10]))
    cache.write_text(json.dumps(dict(floor_y=floor_y, candidates=len(cand), rule=_adt_floor_y.__doc__.split('Cached')[0].strip())))
    return floor_y


def adt_apartment(block_s=30., min_cam_h=1.3):
    """ADT Apartment sequences (local, previously used by other diagnostics): wide pinhole 512x512 F 256 at 2 Hz from
    host/adt_direction_a_prep_20261118.py, ground-truth depth, gravity from the GT device pose, exact floor (Frame.extra
    ground). Frames with the camera < min_cam_h above the floor (seated) are dropped. Non-commercial licence."""
    base = A / 'work/direction-a-adt-20261118/prep'
    for meta in sorted(p for p in base.glob('*.json') if not p.name.endswith('.floor.json')):
        m = json.loads(meta.read_text()); fx, fy, cx, cy = m['K']
        with np.load(base / f"{m['sequence']}.npz") as d:
            rgb, dep, down, cam_y, ts = d['rgb'], d['depth'], d['down'], d['cam_y'], d['t_ns']
        Kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        floor_y = _adt_floor_y(base, m['sequence'], cam_y, dep, down, Kmat)
        for i in range(len(ts)):
            if min_cam_h is not None and cam_y[i] - floor_y < min_cam_h:      # seated (not the walking use case), rule set before the run
                continue
            z = dep[i].astype(np.float32) / 1000.; z[z <= 0] = np.nan
            t = (int(ts[i]) - int(ts[0])) / 1e9
            yield Frame('adt_apartment', m['sequence'], i, t, cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR),
                        np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), z, 'ground_truth', down[i].astype(float), None,
                        'head', block=f"{m['sequence']}:{int(t // block_s)}",
                        extra=dict(ground=(-down[i].astype(float), float(cam_y[i] - floor_y))))


def sanpo_chest_seg():
    return _sanpo(A / 'downloads/sanpo-real-chest-seg-20261117', 'sanpo_chest_seg', 'camera_chest')


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


ADAPTERS = {'sanpo_real_local': sanpo_real_local, 'aria_gen2_walks': aria_gen2_walks, 'sanpo_head_seg': sanpo_head_seg,
            'sanpo_chest_seg': sanpo_chest_seg, 'sanpo_head_seg_autosky': sanpo_head_seg_autosky,
            'sanpo_head_noseg': sanpo_head_noseg, 'adt_apartment': adt_apartment}

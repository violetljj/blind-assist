"""ASE ladder: how much do a narrower (Atom-like pinhole) view and monocular depth cost the cane-complement decision?
(EXPLORE, 2026-10-15; rules frozen in <work>/PLAN_LADDER.md before any result.)

Arms share GT poses, GT floor height, the truth labels of ase_cane_reach_dev_20261015 (R2) and one decision rule:
  detect  : body band 0.60-1.90 m above floor, |lat| <= 0.30 m, 0.30-2.0 m ahead, solid angle >= 8.3e-4 sr; near d <= 1.5 m
  memory  : own points of the past 3 s (incl. current), 0.05-0.70 m above floor, corridor, 0 .. d + 0.15 m ahead,
            3 cm voxels >= 3 => suppressed; alert = near detection and not suppressed.
  A0 fisheye GT; A1 upright pinhole GT (480x640, f = 425, as the Atom/ADT prep); A2r DAV2 Metric-Hypersim-L raw;
  A2o same with per-frame GT median scale; A3 UniDepth v2 L raw (secondary).
Usage:
  ase_cane_ladder_dev_20261015.py pinhole <scenes> <work>        (projectaria-tools venv)
  ase_cane_ladder_dev_20261015.py mono    <work> dav2l|unidepth  (blindassist-torch-gpu venv, GPU)
  ase_cane_ladder_dev_20261015.py ladder  <scenes> <work> [arms] (blindassist-torch-gpu venv)
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ase_cane_reach_dev_20261015 as A  # noqa: E402

F, W_RAW, H_RAW, STEP, WIN_S = 425.0, 640, 480, 2, 3.


# ---------------------------------------------------------------- pinhole frames (projectaria-tools venv)
def _calib():
    from projectaria_tools.projects import ase
    from projectaria_tools.core import calibration as C
    raw = ase.get_ase_rgb_calibration()
    lin = C.get_linear_camera_calibration(W_RAW, H_RAW, F, 'pinhole', raw.get_transform_device_camera())
    return C, raw, lin, C.rotate_camera_calib_cw90deg(lin)


def _pinhole_scene(args):
    scene, work = Path(args[0]), Path(args[1])
    from PIL import Image
    C, raw, lin, rot = _calib()
    zfac = np.load(work / 'zfac.npy')
    out = work / 'pinhole' / scene.name; out.mkdir(parents=True, exist_ok=True)
    T_dc = raw.get_transform_device_camera().to_matrix()
    n = 0
    for i, _, _, _ in A.load_scene(scene, T_dc):
        if (out / f'{i:07d}.png').exists():
            n += 1; continue
        im = np.array(Image.open(scene / 'rgb' / f'vignette{i:07d}.jpg'))
        d = np.clip(np.round(np.array(Image.open(scene / 'depth' / f'depth{i:07d}.png')).astype(np.float32) * zfac), 0, 65535).astype(np.uint16)
        im = np.ascontiguousarray(np.rot90(C.distort_by_calibration(im, lin, raw), 3))
        dz = np.ascontiguousarray(np.rot90(C.distort_depth_by_calibration(d, lin, raw), 3)).astype(np.uint16)
        Image.fromarray(im).save(out / f'{i:07d}.jpg', quality=95)
        Image.fromarray(dz).save(out / f'{i:07d}.png')
        n += 1
    return scene.name, n


def pinhole(scenes, work):
    work = Path(work)
    C, raw, lin, rot = _calib()
    if not (work / 'zfac.npy').exists():
        n = int(raw.get_image_size()[0]); z = np.zeros((n, n), np.float32)
        for v in range(n):
            for u in range(n):
                r = raw.unproject_no_checks(np.array([u + .5, v + .5])); z[v, u] = r[2] / np.linalg.norm(r)
        np.save(work / 'zfac.npy', z)
    meta = dict(projection=[float(v) for v in rot.get_projection_params()], size=[int(v) for v in rot.get_image_size()],
                T_device_camera=rot.get_transform_device_camera().to_matrix().tolist())
    (work / 'pinhole_meta.json').write_text(json.dumps(meta, indent=1))
    print(meta['projection'], meta['size'])
    todo = sorted((p for p in Path(scenes).iterdir() if (p / 'trajectory.csv').exists()), key=lambda p: int(p.name))
    with ProcessPoolExecutor(4) as ex:
        for name, n in ex.map(_pinhole_scene, [(str(s), str(work)) for s in todo]):
            print('pinhole', name, n, flush=True)


def pinhole_nn(scenes, work):
    """Nearest-neighbour GT z depth for the same pinhole view. projectaria's distort_depth_by_calibration interpolates
    across depth edges (mixed-depth pixels under table edges acted as cane support), so A1 uses this instead.
    The pixel map (<work>/pinhole_nnmap.npz: fisheye pixel per raw-orientation pinhole pixel, |r|/z) comes from
    projectaria-tools lin.unproject -> raw.project."""
    import cv2
    work = Path(work); M = np.load(work / 'pinhole_nnmap.npz')
    mu, mv, nr, ok = np.clip(M['mu'], 0, 703), np.clip(M['mv'], 0, 703), M['nr'], M['ok']
    for sd in sorted((work / 'pinhole').iterdir(), key=lambda p: int(p.name)):
        out = work / 'pinhole_nn' / sd.name; out.mkdir(parents=True, exist_ok=True)
        for f in sorted(sd.glob('*.png')):
            D = cv2.imread(str(Path(scenes) / sd.name / 'depth' / f'depth{f.stem}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32)
            z = np.where(ok, D[mv, mu] / nr, 0.)
            cv2.imwrite(str(out / f.name), np.clip(np.round(np.rot90(z, 3)), 0, 65535).astype(np.uint16))
        print('pinhole_nn', sd.name, flush=True)


# ---------------------------------------------------------------- monocular depth (GPU)
def mono(work, model):
    import cv2
    work = Path(work); meta = json.loads((work / 'pinhole_meta.json').read_text())
    fx, fy, cx, cy = meta['projection']; K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    import adt_backbone_blobs_dev_20261013 as B
    infer = B.load_backbone({'dav2l': 'dav2l_metric', 'unidepth': 'unidepth_l'}[model])
    for sd in sorted((work / 'pinhole').iterdir(), key=lambda p: int(p.name)):
        out = work / f'mono_{model}' / sd.name; out.mkdir(parents=True, exist_ok=True)
        files = sorted(sd.glob('*.jpg'))
        for f in files:
            if (out / f'{f.stem}.png').exists():
                continue
            rgb = cv2.imread(str(f))[:, :, ::-1]
            _, z = infer(np.ascontiguousarray(rgb), K)
            z = np.where(np.isfinite(z) & (z > .05), z, 0.)
            cv2.imwrite(str(out / f'{f.stem}.png'), np.clip(np.round(z * 1000), 0, 65535).astype(np.uint16))
        print(model, sd.name, len(files), flush=True)


def mono_fast(work, limit=None, bench=None):
    """Same DAV2 Metric-Hypersim-L / input 518 / fp16 autocast as mono, but torch SDPA attention and batch 4.
    bench=<scene>: time `limit` frames of that scene both ways and compare with the saved PNGs (no writing)."""
    import time
    import cv2
    import torch
    import torch.nn.functional as Fn
    import adt_backbone_blobs_dev_20261013 as B
    sys.path.insert(0, str(B.UP / 'Depth-Anything-V2/metric_depth'))
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.dinov2_layers import attention as At

    def sdpa_forward(self, x, attn_bias=None):
        Bn, N, C = x.shape
        q, k, v = self.qkv(x).reshape(Bn, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = Fn.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(Bn, N, C)
        return self.proj_drop(self.proj(x))
    work = Path(work)
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], max_depth=20)
    model.load_state_dict(torch.load(B.MODELS / 'depth-anything--Depth-Anything-V2-Metric-Hypersim-Large/depth_anything_v2_metric_hypersim_vitl.pth',
                                     map_location='cpu', weights_only=True), strict=True)
    model = model.cuda().eval()

    def run(files, fast, bs):
        if fast:
            At.Attention.forward = sdpa_forward; At.MemEffAttention.forward = sdpa_forward
        out = []
        for k in range(0, len(files), bs):
            ims = [cv2.imread(str(f)) for f in files[k:k + bs]]
            t = torch.cat([model.image2tensor(im, 518)[0] for im in ims])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t)
            z = Fn.interpolate(dd[:, None].float(), ims[0].shape[:2], mode='bilinear', align_corners=True)[:, 0].cpu().numpy()
            out += list(z)
        torch.cuda.synchronize()
        return out
    if bench:
        files = sorted((work / 'pinhole' / bench).glob('*.jpg'))[:int(limit or 40)]
        from depth_anything_v2.dinov2_layers.attention import Attention
        slow_fwd, mem_fwd = Attention.forward, At.MemEffAttention.forward
        t0 = time.time(); a = run(files, False, 1); t1 = time.time()
        b = run(files, True, 4); t2 = time.time()
        At.Attention.forward, At.MemEffAttention.forward = slow_fwd, mem_fwd
        saved = [cv2.imread(str(work / 'mono_dav2l' / bench / f'{f.stem}.png'), cv2.IMREAD_UNCHANGED) / 1000. for f in files]
        rel = [np.median(np.abs(y - x) / np.maximum(x, .05)) for x, y in zip(a, b)]
        rs = [np.median(np.abs(x - s) / np.maximum(s, .05)) for x, s in zip(a, saved)]
        print(f'old {len(files) / (t1 - t0):.2f} fps, fast {len(files) / (t2 - t1):.2f} fps; '
              f'median |fast-old|/old {np.median(rel):.5f} (p99 frame {np.percentile(rel, 99):.5f}); old vs saved {np.median(rs):.5f}')
        return
    for sd in sorted((work / 'pinhole').iterdir(), key=lambda p: int(p.name)):
        out = work / 'mono_dav2l' / sd.name; out.mkdir(parents=True, exist_ok=True)
        files = [f for f in sorted(sd.glob('*.jpg')) if not (out / f'{f.stem}.png').exists()]
        for k in range(0, len(files), 32):
            chunk = files[k:k + 32]
            for f, z in zip(chunk, run(chunk, True, 4)):
                z = np.where(np.isfinite(z) & (z > .05), z, 0.)
                cv2.imwrite(str(out / f'{f.stem}.png'), np.clip(np.round(z * 1000), 0, 65535).astype(np.uint16))
        print('dav2l fast', sd.name, len(files), flush=True)


# ---------------------------------------------------------------- ladder
def arm_points(arm, scene, work, i, T_fish, T_up, dirs_f, sa_f, dirs_p, sa_p):
    """World points and per-point solid angle seen by one arm at frame i.
    Suffix E<pct> (post-hoc): drop pixels whose depth differs from any 4-neighbour (step-2 grid) by > pct % (edge filter)."""
    import cv2
    edge = None
    if 'E' in arm[1:]:
        arm, edge = arm.split('E')[0], float(arm.split('E')[1]) / 100.
    if arm == 'A0':
        d, _ = A.read(scene, i); ok = (d > 0) & (d < A.MAX_MM)
        return (dirs_f[ok] * (d[ok, None] / 1000.)) @ T_fish[:3, :3].T + T_fish[:3, 3], sa_f[ok], None
    gt = cv2.imread(str(work / 'pinhole_nn' / scene.name / f'{i:07d}.png'), cv2.IMREAD_UNCHANGED)[::STEP, ::STEP].astype(np.float32)
    if arm == 'A1':
        z = gt
    elif arm == 'A1i':                                                # first run: interpolated reprojection (kept for record)
        z = cv2.imread(str(work / 'pinhole' / scene.name / f'{i:07d}.png'), cv2.IMREAD_UNCHANGED)[::STEP, ::STEP].astype(np.float32)
    else:
        src = {'A2r': 'dav2l', 'A2o': 'dav2l', 'A3': 'unidepth'}[arm]
        z = cv2.imread(str(work / f'mono_{src}' / scene.name / f'{i:07d}.png'), cv2.IMREAD_UNCHANGED)[::STEP, ::STEP].astype(np.float32)
        z = np.where(gt > 0, z, 0.)                                   # same valid image area as the pinhole view
        if arm == 'A2o':
            m = (gt > 0) & (gt < A.MAX_MM) & (z > 0)
            z = z * (np.median(gt[m] / z[m]) if m.sum() > 100 else 1.)
    ok = (z > 0) & (z < A.MAX_MM)
    if edge is not None:
        zp = np.pad(z, 1, mode='edge'); jump = np.zeros_like(z)
        for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
            jump = np.maximum(jump, np.abs(zp[dy:dy + z.shape[0], dx:dx + z.shape[1]] - z))
        ok &= jump <= edge * np.maximum(z, 1.)
    stats = None
    if arm not in ('A1', 'A1i'):
        m = ok & (gt > 0) & (gt < A.MAX_MM)
        stats = float(np.median(np.abs(z[m] - gt[m]) / gt[m])) if m.sum() > 100 else None
    return (dirs_p[ok] * (z[ok, None] / 1000.)) @ T_up[:3, :3].T + T_up[:3, 3], sa_p[ok], stats


def ladder_scene(args):
    scene, work, arms = Path(args[0]), Path(args[1]), args[2]
    band_lo = args[3] if len(args) > 3 else A.BAND_LO                  # post-hoc sweep only (PLAN_LADDER addendum)
    S = json.loads((work / 'scenes' / f'{scene.name}.json').read_text()); floor = S['floor']
    truth = {r['frame']: r for r in S['rows']}
    R = np.load(work / 'rays.npz'); dirs_f, sa_f, T_dc = R['dirs'], R['sa'], R['T_dc']
    meta = json.loads((work / 'pinhole_meta.json').read_text()); fx, fy, cx, cy = meta['projection']; W, H = meta['size']
    T_dc_up = np.array(meta['T_device_camera'])
    vv, uu = np.mgrid[0:H:STEP, 0:W:STEP]; x, y = (uu - cx) / fx, (vv - cy) / fy
    dirs_p = np.stack([x, y, np.ones_like(x)], -1); sa_p = STEP * STEP / (fx * fy) / (1 + x * x + y * y) ** 1.5
    frames = A.load_scene(scene, T_dc)
    t = np.array([f[1] for f in frames]); n = len(frames)
    T_up = []
    for i, _, T_fish, _ in frames:
        T_wd = T_fish @ np.linalg.inv(T_dc); T_up.append(T_wd @ T_dc_up)
    res = {}
    for arm in arms:
        keys, body, err = [], [], []
        for j, (i, _, T_fish, _) in enumerate(frames):
            P, w, e = arm_points(arm, scene, work, i, T_fish, T_up[j], dirs_f, sa_f, dirs_p, sa_p)
            h = P[:, 2] - floor
            keys.append(np.unique(A.pack(np.floor(P[(h >= band_lo) & (h <= A.TOPS['R2'])] / A.VOX).astype(np.int64))))
            Tc = T_fish if arm == 'A0' else T_up[j]
            o = Tc[:3, 3]; f = Tc[:3, 2] - A.UP * (Tc[:3, 2] @ A.UP); nf = np.linalg.norm(f)
            if nf < .2:
                body.append(None); continue
            f /= nf; xr = np.cross(f, A.UP)
            lat, fwd = (P - o) @ xr, (P - o) @ f
            m = (np.abs(lat) <= A.X_HALF) & (fwd >= A.Z_RANGE[0]) & (fwd <= A.Z_RANGE[1]) & (h >= A.BODY[0]) & (h <= A.BODY[1])
            om = float(w[m].sum())
            body.append((om, float(np.percentile(fwd[m], 5)) if om >= A.MIN_SR else None, o, f, xr))
            err.append(e)
        rows = []
        for j, (i, _, _, _) in enumerate(frames):
            tr = truth.get(i)
            if tr is None or 'skip' in tr or 'walk_aligned' not in A.strata(tr):
                continue
            if tr['pos']:
                lab = ('MISS' if tr['cat']['R2'] == 'MISS' else 'CANE' if tr['cat']['R2'] == 'CANE' else 'UNOBS') + \
                      ('_near' if tr['d_body'] <= A.NEAR else '_far')
            else:
                lab = 'NEG'
            b = body[j]
            det = b is not None and b[1] is not None and b[1] <= A.NEAR
            sup, where = 0, None
            if det:
                om, d, o, f, xr = b
                win = np.flatnonzero((t >= t[j] - WIN_S) & (t <= t[j]))
                Cc = (A.unpack(np.unique(np.concatenate([keys[k] for k in win]))) + .5) * A.VOX
                a_, b_ = (Cc[:, :2] - o[:2]) @ xr[:2], (Cc[:, :2] - o[:2]) @ f[:2]
                m = (np.abs(a_) <= A.X_HALF) & (b_ >= 0) & (b_ <= d + A.MARGIN); sup = int(m.sum())
                if sup:                                                # descriptive: where the support sits (forward gap to d, height)
                    where = [round(float(np.median(d - b_[m])), 3), round(float(np.median(Cc[m, 2] - floor)), 3)]
            rows.append(dict(frame=i, lab=lab, inst=tr.get('inst'), det=bool(det), d=round(b[1], 3) if det else None,
                             d_truth=tr.get('d_body'), sup=sup, where=where, alert=bool(det and sup < A.MIN_VOX)))
        res[arm] = dict(rows=rows, absrel=float(np.median([e for e in err if e is not None])) if any(e is not None for e in err) else None)
    return scene.name, res


def ladder(scenes, work, arms='A0,A1,A2r,A2o', band_lo=None):
    work = Path(work); arms = arms.split(',')
    band_lo = float(band_lo) if band_lo else A.BAND_LO
    todo = sorted((p for p in Path(scenes).iterdir() if (p / 'trajectory.csv').exists()), key=lambda p: int(p.name))
    with ProcessPoolExecutor(3) as ex:
        res = dict(ex.map(ladder_scene, [(str(s), str(work), arms, band_lo) for s in todo]))
    rep = {}
    for arm in arms:
        rows = [dict(r, scene=s) for s, v in res.items() for r in v[arm]['rows']]
        def cnt(lab, key='alert'): return sum(r[key] for r in rows if r['lab'] == lab)
        def tot(lab): return sum(r['lab'] == lab for r in rows)
        miss = [r for r in rows if r['lab'] == 'MISS_near']
        dd = [abs(r['d'] - r['d_truth']) for r in rows if r['det'] and r['lab'] in ('CANE_near', 'MISS_near')]
        rep[arm] = dict(
            miss_near=tot('MISS_near'), miss_kept=cnt('MISS_near'), miss_not_detected=sum(not r['det'] for r in miss),
            miss_suppressed=sum(r['det'] and not r['alert'] for r in miss),
            miss_objects_kept=len({(r['scene'], r['inst']) for r in miss if r['alert']}),
            miss_objects=len({(r['scene'], r['inst']) for r in miss}),
            cane_near=tot('CANE_near'), cane_alert=cnt('CANE_near'), cane_detected=cnt('CANE_near', 'det'),
            neg=tot('NEG'), neg_alert=cnt('NEG'), neg_detected=cnt('NEG', 'det'),
            far_alert=sum(r['alert'] for r in rows if r['lab'].endswith('_far')),
            d_err_median=round(float(np.median(dd)), 3) if dd else None,
            depth_absrel_median_of_frames=round(float(np.median([v[arm]['absrel'] for v in res.values() if v[arm]['absrel'] is not None])), 4)
            if any(v[arm]['absrel'] is not None for v in res.values()) else None)
    tag = '_'.join(arms) + ('' if band_lo == A.BAND_LO else f'_lo{int(round(band_lo * 100))}')
    (work / f'ladder_{tag}.json').write_text(json.dumps(dict(report=rep, rows=res)), encoding='utf8')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    dict(pinhole=pinhole, pinhole_nn=pinhole_nn, mono=mono, mono_fast=mono_fast, ladder=ladder)[sys.argv[1]](*sys.argv[2:])

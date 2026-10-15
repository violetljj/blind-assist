"""Corridor side-view occupancy net: can learning remove monocular fake cane support? (EXPLORE, 2026-10-15)
Rules frozen in <work>/PLAN_SIDEVIEW.md before training. Test = ladder scenes 0-19; train = 20-54; val = 55-59.

Grid: corridor of the upright pinhole camera (horizontal heading, GT floor), forward 0-2.2 m x height 0-1.9 m, 5 cm cells,
lateral |x| <= 0.30 m collapsed. Target = full-trajectory fisheye GT voxels (amodal). Inputs from DAV2 Metric-Hypersim-Small
(GT median scale, as A2o) over the past 3 s: hit counts (current / past), depth-test see-through and in-view counts
(current / past) at lateral -0.2/0/+0.2 m, plus forward/height coordinates.
Usage:
  ase_cane_sideview_dev_20261015.py mono_s <work>
  ase_cane_sideview_dev_20261015.py grids  <scenes> <work> <first> <last>
  ase_cane_sideview_dev_20261015.py train  <work>
  ase_cane_sideview_dev_20261015.py eval   <work>
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ase_cane_reach_dev_20261015 as A  # noqa: E402

CELL, NF, NH, X_HALF, WIN_S, STEP = .05, 44, 38, .30, 3., 2
TEST, TRAIN, VAL = range(0, 20), range(20, 55), range(55, 60)
BAND_ROWS, BODY_ROWS, BODY_COLS = (1, 14), (12, 38), (6, 40)          # 0.05-0.70 m; 0.60-1.90 m; 0.30-2.0 m


# ---------------------------------------------------------------- DAV2-Small metric depth
def mono_s(work):
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
        return self.proj_drop(self.proj(Fn.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(Bn, N, C)))
    At.Attention.forward = sdpa_forward; At.MemEffAttention.forward = sdpa_forward
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=20)
    model.load_state_dict(torch.load(B.MODELS / 'depth-anything--Depth-Anything-V2-Metric-Hypersim-Small/depth_anything_v2_metric_hypersim_vits.pth',
                                     map_location='cpu', weights_only=True), strict=True)
    model = model.cuda().eval(); work = Path(work)
    for sd in sorted((work / 'pinhole').iterdir(), key=lambda p: int(p.name)):
        out = work / 'mono_dav2s' / sd.name; out.mkdir(parents=True, exist_ok=True)
        files = [f for f in sorted(sd.glob('*.jpg')) if not (out / f'{f.stem}.png').exists()]
        for k in range(0, len(files), 8):
            ims = [cv2.imread(str(f)) for f in files[k:k + 8]]
            t = torch.cat([model.image2tensor(im, 518)[0] for im in ims])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t)
            z = Fn.interpolate(dd[:, None].float(), ims[0].shape[:2], mode='bilinear', align_corners=True)[:, 0].cpu().numpy()
            for f, zz in zip(files[k:k + 8], z):
                zz = np.where(np.isfinite(zz) & (zz > .05), zz, 0.)
                cv2.imwrite(str(out / f'{f.stem}.png'), np.clip(np.round(zz * 1000), 0, 65535).astype(np.uint16))
        print('dav2s', sd.name, len(files), flush=True)


# ---------------------------------------------------------------- grids
def grids_scene(args):
    import cv2
    scene, work = Path(args[0]), Path(args[1])
    out = work / 'grids' / f'{scene.name}.npz'
    if out.exists():
        return scene.name, 'cached'
    R = np.load(work / 'rays.npz'); dirs_f, T_dc = R['dirs'], R['T_dc']
    meta = json.loads((work / 'pinhole_meta.json').read_text()); fx, fy, cx, cy = meta['projection']; W, H = meta['size']
    T_dc_up = np.array(meta['T_device_camera'])
    frames = A.load_scene(scene, T_dc)
    C, I = A.voxel_map(scene, frames, dirs_f)
    cmap = scene / 'object_instances_to_classes.json'
    if cmap.exists():
        classes = {int(a): b for a, b in json.loads(cmap.read_text()).items()}
        floor = float(np.median(C[np.array([classes.get(int(v), '?') for v in I]) == 'floor', 2]))
    else:                                                              # later chunks ship no class map: ASE floor plane is z = 0
        floor = .015                                                   # (scene language a_z = 0; voxel-centre convention as scenes 0-19)
    C = C[(C[:, 2] - floor >= -.05) & (C[:, 2] - floor < NH * CELL + .05)]
    vv, uu = np.mgrid[0:H:STEP, 0:W:STEP]
    dp = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1)
    T_up, Pm, Zm = [], [], []
    for i, _, T_fish, _ in frames:
        Tu = T_fish @ np.linalg.inv(T_dc) @ T_dc_up; T_up.append(Tu)
        gt = cv2.imread(str(work / 'pinhole_nn' / scene.name / f'{i:07d}.png'), cv2.IMREAD_UNCHANGED)[::STEP, ::STEP].astype(np.float32)
        z = cv2.imread(str(work / 'mono_dav2s' / scene.name / f'{i:07d}.png'), cv2.IMREAD_UNCHANGED)[::STEP, ::STEP].astype(np.float32)
        m = (gt > 0) & (gt < A.MAX_MM) & (z > 0)
        z = np.where(gt > 0, z * (np.median(gt[m] / z[m]) if m.sum() > 100 else 1.), 0.) / 1000.
        ok = (z > 0) & (z < A.MAX_MM / 1000.)
        Pm.append(((dp[ok] * z[ok, None]) @ Tu[:3, :3].T + Tu[:3, 3]).astype(np.float32)); Zm.append(z.astype(np.float32))
    t = np.array([f[1] for f in frames]); n = len(frames)
    hc, fc = (np.arange(NH) + .5) * CELL, (np.arange(NF) + .5) * CELL
    lat_s = np.array([-.2, 0., .2])
    X = np.zeros((n, 8, NH, NF), np.float16); Y = np.zeros((n, NH, NF), np.uint8); valid = np.zeros(n, bool)
    gf, gh = np.meshgrid(fc / (NF * CELL), hc / (NH * CELL))
    for j in range(n):
        Tu = T_up[j]; o = Tu[:3, 3]; f = Tu[:3, 2] - A.UP * (Tu[:3, 2] @ A.UP); nf = np.linalg.norm(f)
        if nf < .2:
            continue
        f /= nf; xr = np.cross(f, A.UP); valid[j] = True

        def cells(P):
            d = P - o; lat, fwd, h = d @ xr, d @ f, P[:, 2] - floor
            m = (np.abs(lat) <= X_HALF) & (fwd >= 0) & (fwd < NF * CELL) & (h >= 0) & (h < NH * CELL)
            return np.bincount((h[m] // CELL).astype(int) * NF + (fwd[m] // CELL).astype(int), minlength=NH * NF).reshape(NH, NF)
        Y[j] = cells(C) > 0
        win = np.flatnonzero((t >= t[j] - WIN_S) & (t <= t[j]))
        hp = sum((cells(Pm[k]) for k in win if k != j), np.zeros((NH, NF)))
        L_, F_, H_ = np.meshgrid(lat_s, fc, hc, indexing='ij')                           # (3, NF, NH)
        Wp = (o * (1 - A.UP) + L_[..., None] * xr + F_[..., None] * f + (floor + H_)[..., None] * A.UP).reshape(-1, 3)
        seen = np.zeros((2, NH, NF)); free = np.zeros((2, NH, NF))
        for k in win:
            q = (Wp - T_up[k][:3, 3]) @ T_up[k][:3, :3]
            z = q[:, 2]; zs = np.where(z > .05, z, 1.)
            u = (fx * q[:, 0] / zs + cx) / STEP; v = (fy * q[:, 1] / zs + cy) / STEP
            ins = (z > .05) & (u >= 0) & (u < Zm[k].shape[1]) & (v >= 0) & (v < Zm[k].shape[0])
            zm = np.zeros(len(z)); zm[ins] = Zm[k][v[ins].astype(int), u[ins].astype(int)]
            vis = ins & (zm > 0); fr = vis & (z < zm - .10)
            s = 0 if k == j else 1
            seen[s] += vis.reshape(3, NF, NH).sum(0).T; free[s] += fr.reshape(3, NF, NH).sum(0).T
        X[j] = np.stack([np.log1p(cells(Pm[j])), np.log1p(hp), free[0] / 3, np.log1p(free[1]), seen[0] / 3, np.log1p(seen[1]), gf, gh])
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, X=X, Y=Y, valid=valid, frames=np.array([f[0] for f in frames]), floor=floor)
    return scene.name, f'{n} frames'


def grids(scenes, work, first, last):
    todo = [Path(scenes) / str(k) for k in range(int(first), int(last) + 1) if (Path(scenes) / str(k) / 'trajectory.csv').exists()]
    with ProcessPoolExecutor(3) as ex:
        for name, msg in ex.map(grids_scene, [(str(s), str(work)) for s in todo]):
            print('grids', name, msg, flush=True)


# ---------------------------------------------------------------- network
def make_net():
    import torch.nn as nn

    def blk(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                             nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = blk(8, 32), blk(32, 64), blk(64, 128)
            self.pool = nn.MaxPool2d(2)
            self.u2, self.d2 = nn.ConvTranspose2d(128, 64, 2, 2), blk(128, 64)
            self.u1, self.d1 = nn.ConvTranspose2d(64, 32, 2, 2), blk(64, 32)
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, x):                                    # x: (B, 8, 40, 44) padded
            import torch
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1)); d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)[:, 0]
    return UNet()


def load_split(work, ids):
    Xs, Ys = [], []
    for k in ids:
        p = Path(work) / 'grids' / f'{k}.npz'
        if p.exists():
            Z = np.load(p); v = Z['valid']; Xs.append(Z['X'][v]); Ys.append(Z['Y'][v])
    return np.concatenate(Xs), np.concatenate(Ys)


def pad(x):
    import torch.nn.functional as Fn
    return Fn.pad(x, (0, 0, 0, 40 - NH))


def train(work, epochs=25):
    import torch
    torch.manual_seed(20261015); np.random.seed(20261015)
    Xtr, Ytr = load_split(work, TRAIN); Xva, Yva = load_split(work, VAL)
    print('train', Xtr.shape, 'val', Xva.shape, 'pos rate', float(Ytr.mean()), flush=True)
    dev = 'cuda'
    net = make_net().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(Xtr) // 64 + 1))
    wrow = torch.ones(NH, NF, device=dev); wrow[BAND_ROWS[0]:BAND_ROWS[1]] = 4.
    Xtr_t, Ytr_t = torch.from_numpy(Xtr.astype(np.float32)), torch.from_numpy(Ytr.astype(np.float32))
    Xva_t, Yva_t = torch.from_numpy(Xva.astype(np.float32)).to(dev), torch.from_numpy(Yva.astype(np.float32)).to(dev)
    best = 1e9
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr_t))
        for b in range(0, len(perm), 64):
            idx = perm[b:b + 64]; x, y = Xtr_t[idx].to(dev), Ytr_t[idx].to(dev)
            logit = net(pad(x))[:, :NH]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction='none') * wrow).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            lv, inter, uni = 0., 0., 0.
            for b in range(0, len(Xva_t), 256):
                logit = net(pad(Xva_t[b:b + 256]))[:, :NH]; y = Yva_t[b:b + 256]
                lv += float((torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction='none') * wrow).mean()) * len(y)
                p = (logit > 0)[:, BAND_ROWS[0]:BAND_ROWS[1]]; g = y[:, BAND_ROWS[0]:BAND_ROWS[1]] > .5
                inter += float((p & g).sum()); uni += float((p | g).sum())
            lv /= len(Xva_t)
        print(f'epoch {ep} train {float(loss):.4f} val {lv:.4f} band IoU {inter / max(uni, 1):.3f}', flush=True)
        if lv < best:
            best = lv; torch.save(net.state_dict(), Path(work) / 'sideview_net.pt')


# ---------------------------------------------------------------- decision and test scoring
def decide(occ):
    body = occ[BODY_ROWS[0]:BODY_ROWS[1], BODY_COLS[0]:BODY_COLS[1]]
    cols = np.flatnonzero(body.any(0))
    if not len(cols):
        return False, None, False
    d = (BODY_COLS[0] + cols[0]) * CELL
    sup = occ[BAND_ROWS[0]:BAND_ROWS[1], :int(np.ceil((d + A.MARGIN) / CELL))].any()
    return d <= A.NEAR, d, bool(sup)


def evaluate(work):
    import torch
    work = Path(work); net = make_net().cuda(); net.load_state_dict(torch.load(work / 'sideview_net.pt')); net.eval()
    arms = ['G', 'H', 'N03', 'N05', 'N07']; rows = {a: [] for a in arms}
    for k in TEST:
        Z = np.load(work / 'grids' / f'{k}.npz'); S = json.loads((work / 'scenes' / f'{k}.json').read_text())
        truth = {r['frame']: r for r in S['rows']}
        with torch.no_grad():
            P = torch.sigmoid(net(pad(torch.from_numpy(Z['X'].astype(np.float32)).cuda()))[:, :NH]).cpu().numpy()
        for j, fr in enumerate(Z['frames']):
            tr = truth.get(int(fr))
            if tr is None or 'skip' in tr or not Z['valid'][j] or 'walk_aligned' not in A.strata(tr):
                continue
            lab = (('MISS' if tr['cat']['R2'] == 'MISS' else 'CANE' if tr['cat']['R2'] == 'CANE' else 'UNOBS')
                   + ('_near' if tr['d_body'] <= A.NEAR else '_far')) if tr['pos'] else 'NEG'
            grids_ = dict(G=Z['Y'][j] > 0, H=(Z['X'][j][0] > 0) | (Z['X'][j][1] > 0),
                          N03=P[j] > .3, N05=P[j] > .5, N07=P[j] > .7)
            for a, occ in grids_.items():
                near, d, sup = decide(occ)
                rows[a].append(dict(scene=k, frame=int(fr), lab=lab, inst=tr.get('inst'), det=bool(near), d=d,
                                    alert=bool(near and not sup)))
    rep = {}
    for a, rr in rows.items():
        miss = [r for r in rr if r['lab'] == 'MISS_near']
        rep[a] = dict(miss_kept=sum(r['alert'] for r in miss), miss_near=len(miss),
                      miss_not_detected=sum(not r['det'] for r in miss), miss_suppressed=sum(r['det'] and not r['alert'] for r in miss),
                      miss_objects_kept=len({(r['scene'], r['inst']) for r in miss if r['alert']}),
                      miss_objects=len({(r['scene'], r['inst']) for r in miss}),
                      cane_alert=sum(r['alert'] for r in rr if r['lab'] == 'CANE_near'),
                      cane_detected=sum(r['det'] for r in rr if r['lab'] == 'CANE_near'),
                      neg_alert=sum(r['alert'] for r in rr if r['lab'] == 'NEG'), neg=sum(r['lab'] == 'NEG' for r in rr))
    (work / 'sideview_eval.json').write_text(json.dumps(dict(report=rep, rows=rows)), encoding='utf8')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    dict(mono_s=mono_s, grids=grids, train=train, eval=evaluate)[sys.argv[1]](*sys.argv[2:])

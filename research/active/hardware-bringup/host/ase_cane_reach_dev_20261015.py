"""Cane-reach positives in Aria Synthetic Environments (ASE): can a synthetic egocentric-walk source supply
"the cane will not find it" obstacles at scale? (EXPLORE, 2026-10-15)

Rules and gates are frozen in <work>/PLAN.md before any result. Summary:
  frames  : every 2nd frame (5 Hz) of the 20 ASE train scenes downloaded on 2026-10-13; raw 704x704 fisheye, every 2nd
            pixel ray (projectaria-tools calibration), ray depth in mm; T_wc = T_world_device @ T_device_camera; Z up.
  map     : all sampled frames fused into 3 cm voxels with the majority instance id (static scene, offline truth);
            floor = median height of 'floor' voxels.
  positive: current-frame points 0.60-1.90 m above floor, |lat| <= 0.30 m, 0.30-2.0 m ahead (horizontal optical axis),
            summed solid angle >= 8.3e-4 sr; d_body = 5th percentile forward distance; near = d_body <= 1.5 m.
  support : map voxels in the corridor, 0 .. d_body + 0.15 m ahead, 0.05 m .. top above floor, >= 3 voxels => CANE.
            R2 (primary) top 0.70 m (cane protects waist-down); R1 (legacy ADT/ARKit) top 0.30 m.
            No support and floor cells observed over < 50% of that area => UNOBSERVED, else MISS.
  strata  : all; walking (>= 0.30 m/s over +-2 frames); walking and heading within 30 deg of travel (primary).
Usage:
  ase_cane_reach_dev_20261015.py rays   <work>              (projectaria-tools venv)
  ase_cane_reach_dev_20261015.py label  <scenes> <work>     (blindassist-torch-gpu venv)
  ase_cane_reach_dev_20261015.py sample <scenes> <work>
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STRIDE, PIX = 2, 2
VOX, MARGIN, MIN_VOX = .03, .15, 3
BODY, X_HALF, Z_RANGE, MIN_SR, NEAR = (.60, 1.90), .30, (.30, 2.0), 8.3e-4, 1.5
TOPS = dict(R2=.70, R1=.30); BAND_LO, FLOOR_TOL, FLOOR_COVER = .05, .03, .5
WALK, ALIGN_DEG, MAX_MM = .30, 30., 8000
UP = np.array([0., 0., 1.])


def rays(work):
    from projectaria_tools.projects import ase
    calib = ase.get_ase_rgb_calibration()
    n = int(calib.get_image_size()[0]); g = np.arange(0, n, PIX)
    r = np.zeros((len(g), len(g), 3))
    for a, v in enumerate(g):
        for b, u in enumerate(g):
            q = calib.unproject_no_checks(np.array([float(u), float(v)]))
            r[a, b] = q / np.linalg.norm(q)
    du = np.diff(r, axis=1, append=r[:, -1:] * 2 - r[:, -2:-1]); dv = np.diff(r, axis=0, append=r[-1:] * 2 - r[-2:-1])
    sa = np.linalg.norm(np.cross(du, dv), axis=-1)                       # solid angle of one PIX x PIX sample
    np.savez(Path(work) / 'rays.npz', dirs=r.astype(np.float32), sa=sa.astype(np.float32),
             T_dc=calib.get_transform_device_camera().to_matrix(), image_size=n)
    print('rays', r.shape, 'total sr', float(sa[np.isfinite(sa)].sum()))


def qmat(x, y, z, w):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def load_scene(scene, T_dc):
    rows = list(csv.DictReader(open(scene / 'trajectory.csv')))
    frames = []
    for i in range(0, len(rows), STRIDE):
        if not ((scene / 'depth' / f'depth{i:07d}.png').exists() and (scene / 'instances' / f'instance{i:07d}.png').exists()):
            continue
        r = rows[i]; T = np.eye(4)
        T[:3, :3] = qmat(*(float(r[k]) for k in ('qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device')))
        T[:3, 3] = [float(r[k]) for k in ('tx_world_device', 'ty_world_device', 'tz_world_device')]
        frames.append((i, float(r['tracking_timestamp_us']) / 1e6, T @ T_dc, T[:3, 3].copy()))
    return frames


def read(scene, i):
    import cv2
    d = cv2.imread(str(scene / 'depth' / f'depth{i:07d}.png'), cv2.IMREAD_UNCHANGED)[::PIX, ::PIX]
    k = cv2.imread(str(scene / 'instances' / f'instance{i:07d}.png'), cv2.IMREAD_UNCHANGED)[::PIX, ::PIX]
    return d, k


def points(scene, i, T, dirs):
    d, k = read(scene, i)
    ok = (d > 0) & (d < MAX_MM)
    P = (dirs[ok] * (d[ok, None] / 1000.)) @ T[:3, :3].T + T[:3, 3]
    return P, k[ok].astype(np.int64), ok


def voxel_map(scene, frames, dirs):
    """Majority-instance 3 cm voxel map of all sampled frames: (centers [N,3], instance [N])."""
    OFF, B = 1024, 11                                                   # 11 bits per axis: +-30 m at 3 cm
    codes, counts = [], []
    def compact():
        c = np.concatenate(codes); n = np.concatenate(counts)
        u, inv = np.unique(c, return_inverse=True)
        codes[:] = [u]; counts[:] = [np.bincount(inv, weights=n)]
    for j, (i, _, T, _) in enumerate(frames):
        P, k, _ = points(scene, i, T, dirs)
        q = np.floor(P / VOX).astype(np.int64) + OFF
        assert q.min() >= 0 and q.max() < (1 << B) and k.max() < 4096
        code = (((q[:, 0] << B | q[:, 1]) << B | q[:, 2]) << 12) | k
        u, n = np.unique(code, return_counts=True); codes.append(u); counts.append(n.astype(float))
        if len(codes) > 40:
            compact()
    compact()
    c, n = codes[0], counts[0]
    vox = c >> 12; order = np.lexsort((n, vox))                          # per voxel the highest-count instance last
    vox, inst = vox[order], (c & 4095)[order]
    last = np.r_[vox[1:] != vox[:-1], True]
    vox, inst = vox[last], inst[last]
    mask = (1 << B) - 1
    q = np.stack([(vox >> 2 * B) & mask, (vox >> B) & mask, vox & mask], 1) - OFF
    return ((q + .5) * VOX).astype(np.float32), inst


def label_scene(args):
    scene, work = Path(args[0]), Path(args[1])
    R = np.load(work / 'rays.npz'); dirs, sa, T_dc = R['dirs'], R['sa'], R['T_dc']
    classes = {int(a): b for a, b in json.loads((scene / 'object_instances_to_classes.json').read_text()).items()}
    frames = load_scene(scene, T_dc)
    C, I = voxel_map(scene, frames, dirs)
    cls = np.array([classes.get(int(v), '?') for v in I])
    floor = float(np.median(C[cls == 'floor', 2]))
    H = C[:, 2] - floor
    sup = {r: C[(H >= BAND_LO) & (H <= top)] for r, top in TOPS.items()}
    obst, obst_h = C[(H >= BAND_LO) & (H <= BODY[1])], H[(H >= BAND_LO) & (H <= BODY[1])]
    fl = C[np.abs(H) <= FLOOR_TOL]
    fl = np.unique(np.floor(fl[:, :2] / VOX).astype(np.int64), axis=0)
    flc = (fl + .5) * VOX
    t = np.array([f[1] for f in frames]); pos = np.array([f[3][:2] for f in frames]); n = len(frames)
    lo, hi = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
    vel = (pos[hi] - pos[lo]) / np.maximum(t[hi] - t[lo], 1e-3)[:, None]; speed = np.linalg.norm(vel, axis=1)
    rows = []
    for j, (i, _, T, _) in enumerate(frames):
        o = T[:3, 3]; f = T[:3, 2] - UP * (T[:3, 2] @ UP); nf = np.linalg.norm(f)
        row = dict(frame=i, speed=round(float(speed[j]), 3), cam_h=round(float(o[2] - floor), 3))
        if nf < .2:
            rows.append(dict(row, skip='vertical')); continue
        f /= nf; xr = np.cross(f, UP)
        row['align_deg'] = round(float(np.degrees(np.arccos(np.clip(vel[j] @ f[:2] / max(speed[j], 1e-6), -1, 1)))), 1)
        P, k, ok = points(scene, i, T, dirs)
        lat, fwd, hgt = (P - o) @ xr, (P - o) @ f, P[:, 2] - floor
        body = (np.abs(lat) <= X_HALF) & (fwd >= Z_RANGE[0]) & (fwd <= Z_RANGE[1]) & (hgt >= BODY[0]) & (hgt <= BODY[1])
        w = sa[ok][body]; omega = float(w.sum())
        if omega < MIN_SR:
            rows.append(dict(row, pos=False)); continue
        d_body = float(np.percentile(fwd[body], 5))
        kw = np.bincount(k[body], weights=w); inst = int(kw.argmax())
        reach = d_body + MARGIN
        def region(Q):
            a, b = (Q[:, :2] - o[:2]) @ xr[:2], (Q[:, :2] - o[:2]) @ f[:2]
            return (np.abs(a) <= X_HALF) & (b >= 0) & (b <= reach)
        cover = region(np.c_[flc, np.zeros(len(flc))]).sum() * VOX * VOX / (2 * X_HALF * reach)
        lowest = obst_h[region(obst)]
        cat = {}
        for r, S in sup.items():
            nv = int(region(S).sum())
            cat[r] = 'CANE' if nv >= MIN_VOX else ('UNOBSERVED' if cover < FLOOR_COVER else 'MISS')
        rows.append(dict(row, pos=True, omega=round(omega, 5), d_body=round(d_body, 3), inst=inst,
                         cls=classes.get(inst, '?'), cat=cat, floor_cover=round(float(cover), 3),
                         lowest=round(float(lowest.min()), 3) if len(lowest) else None,
                         body_h=[round(float(np.percentile(hgt[body], q)), 3) for q in (5, 95)]))
    out = dict(scene=scene.name, floor=floor, frames=n, voxels=len(C), rows=rows)
    (work / 'scenes').mkdir(exist_ok=True)
    (work / 'scenes' / f'{scene.name}.json').write_text(json.dumps(out), encoding='utf8')
    return scene.name, n, len(C)


def strata(row):
    s = ['all']
    if row['speed'] >= WALK:
        s.append('walk')
        if row.get('align_deg', 180) <= ALIGN_DEG:
            s.append('walk_aligned')
    return s


def summarize(work):
    work = Path(work); stats, miss = {}, []
    def add(k, v=1): stats[k] = stats.get(k, 0) + v
    for p in sorted((work / 'scenes').glob('*.json'), key=lambda p: int(p.stem)):
        S = json.loads(p.read_text())
        for row in S['rows']:
            for s in strata(row):
                add(f'{s}|frames')
                if 'skip' in row:
                    add(f'{s}|skip'); continue
                if not row['pos']:
                    add(f'{s}|NONPOS'); continue
                band = 'near' if row['d_body'] <= NEAR else 'far'
                for r, c in row['cat'].items():
                    add(f'{s}|{r}|{c}|{band}')
            if row.get('pos') and 'walk_aligned' in strata(row) and row['d_body'] <= NEAR and row['cat']['R2'] == 'MISS':
                miss.append(dict(scene=S['scene'], **row))
    objs = {(m['scene'], m['inst']) for m in miss}
    by_cls, obj_cls = {}, {}
    for m in miss:
        by_cls[m['cls']] = by_cls.get(m['cls'], 0) + 1
    for sc, inst in objs:
        c = next(m['cls'] for m in miss if m['scene'] == sc and m['inst'] == inst)
        obj_cls[c] = obj_cls.get(c, 0) + 1
    lows = [m['lowest'] for m in miss if m['lowest'] is not None]
    hb = dict(zip(['0.70-1.00', '1.00-1.40', '1.40+'], np.histogram(lows, [.70, 1.0, 1.4, 9])[0].tolist())) if lows else {}
    per_scene = {}
    for m in miss:
        per_scene[m['scene']] = per_scene.get(m['scene'], 0) + 1
    rep = dict(stats=stats, primary_miss_near=len(miss), scenes_with_miss=len(per_scene), per_scene=per_scene,
               distinct_objects=len(objs), class_frames=dict(sorted(by_cls.items(), key=lambda x: -x[1])),
               class_objects=dict(sorted(obj_cls.items(), key=lambda x: -x[1])), lowest_height_frames=hb,
               lowest_below_070=sum(l < .70 for l in lows))
    (work / 'summary.json').write_text(json.dumps(rep, indent=1), encoding='utf8')
    (work / 'miss_primary.json').write_text(json.dumps(miss), encoding='utf8')
    print(json.dumps(rep, indent=1))


def label(scenes, work):
    scenes = sorted((p for p in Path(scenes).iterdir() if (p / 'trajectory.csv').exists()), key=lambda p: int(p.name))
    with ProcessPoolExecutor(3) as ex:
        for name, n, nv in ex.map(label_scene, [(str(s), work) for s in scenes]):
            print('scene', name, 'frames', n, 'voxels', nv, flush=True)
    summarize(work)


def sample(scenes, work, k=16, seed=20261015):
    """Fixed-seed near-MISS (primary, R2) sample: RGB with body points (red) and corridor points below 0.70 m (green)."""
    import cv2
    scenes, work = Path(scenes), Path(work)
    miss = json.loads((work / 'miss_primary.json').read_text())
    rng = np.random.default_rng(seed)
    pick = [miss[i] for i in sorted(rng.choice(len(miss), min(k, len(miss)), replace=False))]
    R = np.load(work / 'rays.npz'); dirs, T_dc = R['dirs'], R['T_dc']
    tiles = []
    for m in pick:
        sc = scenes / m['scene']; frames = {f[0]: f for f in load_scene(sc, T_dc)}
        _, _, T, _ = frames[m['frame']]
        rgb = cv2.imread(str(sc / 'rgb' / f'vignette{m["frame"]:07d}.jpg'))
        d, _ = read(sc, m['frame']); ok = (d > 0) & (d < MAX_MM)
        P = np.zeros(d.shape + (3,)); P[ok] = (dirs[ok] * (d[ok, None] / 1000.)) @ T[:3, :3].T + T[:3, 3]
        floor = T[2, 3] - m['cam_h']; o = T[:3, 3]
        f = T[:3, 2] - UP * (T[:3, 2] @ UP); f /= np.linalg.norm(f); xr = np.cross(f, UP)
        lat, fwd, hgt = (P - o) @ xr, (P - o) @ f, P[..., 2] - floor
        body = ok & (np.abs(lat) <= X_HALF) & (fwd >= Z_RANGE[0]) & (fwd <= Z_RANGE[1]) & (hgt >= BODY[0]) & (hgt <= BODY[1])
        low = ok & (np.abs(lat) <= X_HALF) & (fwd >= 0) & (fwd <= m['d_body'] + MARGIN) & (hgt < TOPS['R2']) & (hgt > -FLOOR_TOL)
        ov = np.zeros(d.shape + (3,), np.uint8); ov[body] = (0, 0, 255); ov[low & ~body] = (0, 200, 0)
        ov = cv2.resize(ov, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        tile = np.ascontiguousarray(np.rot90(cv2.addWeighted(rgb, 1., ov, .5, 0), 3))
        tile = cv2.resize(tile, (352, 352))
        txt = f'{m["scene"]}#{m["frame"]} {m["cls"]} d={m["d_body"]:.2f} low={m["lowest"]} cov={m["floor_cover"]:.2f}'
        cv2.putText(tile, txt, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
    cv2.imwrite(str(work / 'miss_near_sample.jpg'), sheet)
    (work / 'miss_near_sample.json').write_text(json.dumps(dict(pool=len(miss), seed=seed, picked=pick), indent=1),
                                                encoding='utf8')
    print(len(miss), 'primary near MISS frames; sheet written')


def pack(q, OFF=1024, B=11):
    q = q + OFF
    return (q[:, 0] << B | q[:, 1]) << B | q[:, 2]


def unpack(c, OFF=1024, B=11):
    m = (1 << B) - 1
    return np.stack([(c >> 2 * B) & m, (c >> B) & m, c & m], 1) - OFF


def memory_scene(args):
    """Causal 3 s ground memory on GT points (PLAN_MEMORY.md): does the past alone show the cane support?"""
    scene, work = Path(args[0]), Path(args[1])
    S = json.loads((work / 'scenes' / f'{scene.name}.json').read_text()); floor = S['floor']
    R = np.load(work / 'rays.npz'); dirs, T_dc = R['dirs'], R['T_dc']
    frames = load_scene(scene, T_dc); rows = {r['frame']: r for r in S['rows']}
    sup, flo = [], []
    for i, _, T, _ in frames:
        P, _, _ = points(scene, i, T, dirs); h = P[:, 2] - floor
        sup.append(np.unique(pack(np.floor(P[(h >= BAND_LO) & (h <= TOPS['R2'])] / VOX).astype(np.int64))))
        flo.append(np.unique(pack(np.floor(P[np.abs(h) <= FLOOR_TOL] / VOX).astype(np.int64))))
    t = np.array([f[1] for f in frames]); out = []
    for j, (i, _, T, _) in enumerate(frames):
        row = rows[i]
        if not row.get('pos') or row['d_body'] > NEAR or 'walk_aligned' not in strata(row):
            continue
        win = np.flatnonzero((t >= t[j] - 3.) & (t <= t[j]))
        o = T[:3, 3]; f = T[:3, 2] - UP * (T[:3, 2] @ UP); f /= np.linalg.norm(f); xr = np.cross(f, UP)
        reach = row['d_body'] + MARGIN
        def region(Q):
            a, b = (Q[:, :2] - o[:2]) @ xr[:2], (Q[:, :2] - o[:2]) @ f[:2]
            return (np.abs(a) <= X_HALF) & (b >= 0) & (b <= reach)
        C = (unpack(np.unique(np.concatenate([sup[k] for k in win]))) + .5) * VOX
        inr = region(C); h = C[:, 2] - floor
        F = (unpack(np.unique(np.concatenate([flo[k] for k in win]))) + .5) * VOX
        cells = len(np.unique(np.floor(F[region(F), :2] / VOX).astype(np.int64), axis=0)) if len(F) else 0
        out.append(dict(frame=i, cls=row['cls'], truth=row['cat'], mem=dict(R2=int(inr.sum()), R1=int((inr & (h <= TOPS['R1'])).sum())),
                        cover=round(cells * VOX * VOX / (2 * X_HALF * reach), 3)))
    return scene.name, out


def memory(scenes, work):
    work = Path(work)
    scenes = sorted((p for p in Path(scenes).iterdir() if (p / 'trajectory.csv').exists()), key=lambda p: int(p.name))
    with ProcessPoolExecutor(3) as ex:
        res = dict(ex.map(memory_scene, [(str(s), str(work)) for s in scenes]))
    rep = {}
    for r in TOPS:
        cane = [x for v in res.values() for x in v if x['truth'][r] == 'CANE']
        miss = [x for v in res.values() for x in v if x['truth'][r] == 'MISS']
        resid = [x for x in cane if x['mem'][r] < MIN_VOX]
        by = {}
        for x in resid:
            by[x['cls']] = by.get(x['cls'], 0) + 1
        allpos = [x for v in res.values() for x in v if x['truth'][r] in ('CANE', 'MISS')]
        share = {}
        for x in allpos:
            s = share.setdefault(x['cls'], [0, 0]); s[0] += x['truth'][r] == 'CANE'; s[1] += 1
        standing = sorted(c for c, (a, b) in share.items() if a / b >= .95)
        rep[r] = dict(cane=len(cane), miss=len(miss), cane_suppressed=len(cane) - len(resid), residual=len(resid),
                      residual_frac=round(len(resid) / max(len(cane), 1), 3),
                      miss_suppressed=sum(x['mem'][r] >= MIN_VOX for x in miss),
                      residual_by_class=dict(sorted(by.items(), key=lambda kv: -kv[1])),
                      residual_cover_lt_half=sum(x['cover'] < FLOOR_COVER for x in resid),
                      class_share={c: f'{a}/{b}' for c, (a, b) in sorted(share.items(), key=lambda kv: -kv[1][1])},
                      standing=standing,
                      prior_removes_residual=sum(x['cls'] in standing for x in resid),
                      prior_suppresses_miss=sum(x['cls'] in standing for x in miss))
    (work / 'memory_summary.json').write_text(json.dumps(rep, indent=1), encoding='utf8')
    (work / 'memory_rows.json').write_text(json.dumps(res), encoding='utf8')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    dict(rays=rays, label=label, sample=sample, summarize=summarize, memory=memory)[sys.argv[1]](*sys.argv[2:])

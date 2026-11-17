"""Head-on overhang insertion on REAL outdoor images (Oxford Spires cam0 + TLS geometry), scored with the ASE-trained D8
without retraining (EXPLORE, 2026-11-17). Plan: artifacts.local/work/spires-cane-reach-20261117/PLAN_INSERT.md

  python spires_insert_dev_20261117.py runs   <spires> <work>   select straight outdoor walking segments, place objects
  python spires_insert_dev_20261117.py fetch  <spires> <work>   read only the needed cam0 frames from the remote images.zip
  python spires_insert_dev_20261117.py check  <spires> <work>   projection overlays + insertion sheet (self-check)
  python spires_insert_dev_20261117.py render <spires> <work>   pinhole + TLS depth + insertion + DAV2-S + grids (+ empty runs)
  python spires_insert_dev_20261117.py score  <ase_work> <work>
<work> = artifacts.local/work/spires-cane-reach-20261117 (vox/ from spires_cane_reach_dev_20261117.py voxels).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import spires_cane_reach_dev_20261117 as S  # noqa: E402
import ase_cane_reach_dev_20261015 as A  # noqa: E402
import ase_cane_sideview_dev_20261015 as V  # noqa: E402
import ase_cane_insert_dev_20261015 as I  # noqa: E402

from scipy.spatial.transform import Rotation as Rot, Slerp  # noqa: E402

HF = 'datasets/ori-drs/oxford_spires_dataset/sequences/{}/raw/images.zip'
W, H, F, CX, CY = 480, 640, 425., 239.5, 319.5
K0 = (698.1692094914263, 698.5147584873237, 712.7646767255006, 560.5751284820508)
D0 = (-0.04114225009462773, -0.002421765984053725, 0.003400355152679783, -0.0018518822347983275)
PITCH = 6.0
MAX_RUNS, SEED, GAP_S, PRE = 36, 20261117, 3.0, 15


def _T(t, q):
    M = np.eye(4); M[:3, :3] = Rot.from_quat(q).as_matrix(); M[:3, 3] = t; return M


T_BL = _T([0, 0, .124], [0, 0, 1, 0])
T_CL = _T([0.00035060884033447846, -0.079574053851818, -0.053826061545697405],
          [0.5034725628949727, 0.5003337192107279, -0.4947539264483171, 0.5013981452864594])
c, s = np.cos(np.radians(-PITCH)), np.sin(np.radians(-PITCH))
R_CV = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])                    # virtual pinhole in cam0 axes (pitched down)
T_BV = T_BL @ np.linalg.inv(T_CL); T_BV[:3, :3] = T_BV[:3, :3] @ R_CV   # base -> virtual camera


class Traj:
    def __init__(self, path):
        a = np.loadtxt(path); self.t = a[:, 0]; self.P = a[:, 1:4]; self.sl = Slerp(self.t, Rot.from_quat(a[:, 4:8]))

    def T(self, ts):
        ts = np.clip(np.atleast_1d(ts), self.t[0], self.t[-1]); out = np.tile(np.eye(4), (len(ts), 1, 1))
        out[:, :3, :3] = self.sl(ts).as_matrix(); out[:, :3, 3] = np.stack([np.interp(ts, self.t, self.P[:, k]) for k in range(3)], 1)
        return out @ T_BV                                                # world <- virtual camera


def heading(T):
    f = T[:3, 2].copy(); f[2] = 0; n = np.linalg.norm(f)
    return f / n if n > .2 else None


# ---------------------------------------------------------------- segment selection
def runs(spires, work):
    spires, work = Path(spires), Path(work); out = []; n = 0; grids = {}
    rng = np.random.default_rng(SEED)
    for sd in sorted((spires / 'seq').iterdir()):
        site = S.site_of(sd.name)
        if site is None or not (work / 'vox' / f'{site}.npz').exists():
            continue
        if site not in grids:
            Z = np.load(work / 'vox' / f'{site}.npz'); grids = {site: S.Grid(Z['keys'], Z['counts'])}
        G = grids[site]; tr = Traj(sd / 'gt-tum.txt')
        ts = np.arange(tr.t[0], tr.t[-1], 1 / S.HZ); T = tr.T(ts); o = T[:, :3, 3]; m = len(ts)
        lo, hi = np.clip(np.arange(m) - 2, 0, m - 1), np.clip(np.arange(m) + 2, 0, m - 1)
        v = (o[hi, :2] - o[lo, :2]) / (ts[hi] - ts[lo])[:, None]; sp = np.linalg.norm(v, axis=1)
        hd = [heading(T[i]) for i in range(m)]
        gz = np.array([S.ground_at(G, o[i, :2], o[i, 2]) for i in range(m)])
        ok = []
        for i in range(m):
            good = sp[i] >= S.WALK and hd[i] is not None and np.isfinite(gz[i])
            if good:
                good = np.degrees(np.arccos(np.clip(hd[i][:2] @ v[i] / sp[i], -1, 1))) <= 30
            ok.append(bool(good))
        j = PRE; last_end = -1e9
        while j < m:
            if not ok[j] or ts[j] - last_end < GAP_S:
                j += 1; continue
            run = [j]; hs = [hd[j]]
            while run[-1] + 1 < m and ok[run[-1] + 1]:
                mu = np.mean(hs, 0); mu /= np.linalg.norm(mu)
                if np.degrees(np.arccos(np.clip(hd[run[-1] + 1] @ mu, -1, 1))) > 15:
                    break
                run.append(run[-1] + 1); hs.append(hd[run[-1]])
            j0, j1 = run[0], run[-1]; disp = float(np.linalg.norm(o[j1, :2] - o[j0, :2]))
            if disp < 2.0:
                j = j1 + 1; continue
            u = np.mean(hs, 0); u /= np.linalg.norm(u); r = np.cross(u, A.UP)
            floor = float(np.median(gz[j0:j1 + 1]))
            near = o[j1, :2] + .35 * u[:2]
            g_obj = S.ground_at(G, near + .3 * u[:2], o[j1, 2])
            gs = np.r_[gz[j0:j1 + 1], g_obj]
            # outdoor: nothing 2.2-40 m above ground within 1.5 m of the device at the segment start and end
            # (self-check fix before scoring: the planned 1.0 m / 12 m let two Blenheim hall segments through)
            above = 0
            for jj in (j0, j1):
                cc = G.c[G.near(o[jj, :2], 1.5)]
                above += ((np.hypot(*(cc[:, :2] - o[jj, :2]).T) <= 1.5) & (cc[:, 2] - floor >= 2.2) & (cc[:, 2] - floor <= 40)).sum()
            # empty ahead from the segment start to the object's far end (+ margin)
            far = disp + .35 + .6 + .15
            cc = G.c[G.near(o[j0, :2] + u[:2] * far / 2, far / 2 + .5)]
            d = cc[:, :2] - o[j0, :2]; fw, la, hg = d @ u[:2], d @ r[:2], cc[:, 2] - floor
            ahead = ((np.abs(la) <= .35) & (fw >= 0) & (fw <= far) & (hg >= .05) & (hg <= 2.0)).sum()
            if np.isfinite(gs).all() and np.ptp(gs) <= .08 and above == 0 and ahead < 3 and n < MAX_RUNS:
                kind = I.TYPES[n % len(I.TYPES)]
                lat = rng.uniform(-.1, .1)
                out.append(dict(id=n, seq=sd.name, kind=kind, j0=int(j0), j1=int(j1), t_frames=ts[j0 - PRE:j1 + 1].tolist(),
                                frames=list(range(j1 + 1 - (j0 - PRE))), run_frames=list(range(PRE, PRE + j1 - j0 + 1)),
                                near=(near + lat * r[:2]).tolist(), u=u.tolist(), r=r.tolist(), floor=floor,
                                boxes=I.boxes_for(kind, rng), disp=round(disp, 2)))
                n += 1; last_end = ts[j1]; j = j1 + 1; continue
            j = j1 + 1
    (work / 'insert').mkdir(exist_ok=True)
    (work / 'insert' / 'runs.json').write_text(json.dumps(out, indent=1), encoding='utf8')
    from collections import Counter
    print(len(out), 'runs', Counter(o['kind'] for o in out), Counter(o['seq'][11:] for o in out))


# ---------------------------------------------------------------- images
def cam0_times(seq, work):
    p = Path(work) / 'insert' / 'cam0' / f'{seq}.json'
    if p.exists():
        return json.loads(p.read_text())
    import zipfile
    from huggingface_hub import HfFileSystem
    z = zipfile.ZipFile(HfFileSystem().open(HF.format(seq), 'rb', block_size=1 << 20))
    names = sorted(x for x in z.namelist() if x.startswith('cam0/') and x.endswith('.jpg'))
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(names)); return names


def image_times(run, work):
    names = cam0_times(run['seq'], work); tt = np.array([float(x[5:-4]) for x in names])
    out = []
    for t in run['t_frames']:
        i = int(np.argmin(np.abs(tt - t)))
        out.append(names[i] if abs(tt[i] - t) <= .025 else None)
    return out


def fetch(spires, work):
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfFileSystem
    work = Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    need = {}
    for run in R:
        for nm in image_times(run, work):
            if nm:
                need.setdefault(run['seq'], set()).add(nm)

    def one(seq):
        od = work / 'insert' / 'img' / seq; od.mkdir(parents=True, exist_ok=True)
        todo = [x for x in sorted(need[seq]) if not (od / x[5:]).exists()]
        if todo:
            z = zipfile.ZipFile(HfFileSystem().open(HF.format(seq), 'rb', block_size=1 << 20))
            for x in todo:
                (od / x[5:]).write_bytes(z.read(x))
        return seq, len(need[seq])
    with ThreadPoolExecutor(3) as ex:
        for seq, k in ex.map(one, sorted(need)):
            print('fetched', seq, k, flush=True)


_MAP = None


def undistort_map():
    global _MAP
    if _MAP is None:
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        d = np.stack([(uu - CX) / F, (vv - CY) / F, np.ones_like(uu)], -1) @ R_CV.T     # rays in cam0 axes
        r = np.hypot(d[..., 0], d[..., 1]); th = np.arctan2(r, d[..., 2])
        k1, k2, k3, k4 = D0; thd = th * (1 + k1 * th ** 2 + k2 * th ** 4 + k3 * th ** 6 + k4 * th ** 8)
        sc = np.where(r > 1e-9, thd / np.maximum(r, 1e-9), 1.)
        mx = (K0[0] * d[..., 0] * sc + K0[2]).astype(np.float32); my = (K0[1] * d[..., 1] * sc + K0[3]).astype(np.float32)
        valid = (mx >= 0) & (mx <= 1439) & (my >= 0) & (my <= 1079) & (th < np.radians(80))
        _MAP = (mx, my, valid)
    return _MAP


def pinhole_rgb(path):
    import cv2
    mx, my, valid = undistort_map()
    im = cv2.imread(str(path)); out = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out[~valid] = 0; return out


def map_crop(spires, site, centres, rad=9.0):
    acc = [[] for _ in centres]; cen = np.asarray(centres)
    for ch in S.read_pcd_xyz(Path(spires) / 'map' / f'{site}.pcd'):
        for k, c0 in enumerate(cen):
            m = (np.abs(ch[:, 0] - c0[0]) < rad) & (np.abs(ch[:, 1] - c0[1]) < rad)
            if m.any():
                acc[k].append(ch[m])
    return [np.concatenate(a) if a else np.zeros((0, 3), np.float32) for a in acc]


def tls_depth(P, T):
    """Optical-axis depth (mm) of TLS points in the virtual pinhole, z-buffer with distance-adaptive square splats."""
    q = (P - T[:3, 3]) @ T[:3, :3]; z = q[:, 2]; m = z > .1; q, z = q[m], z[m]
    u = F * q[:, 0] / z + CX; v = F * q[:, 1] / z + CY
    m = (u > -8) & (u < W + 8) & (v > -8) & (v < H + 8); u, v, z = u[m], v[m], z[m]
    buf = np.full(H * W, np.inf, np.float32)
    rad = np.clip(np.ceil(.01 * F / z / 2).astype(int), 0, 4)
    for rr in np.unique(rad):
        s_ = rad == rr; ui, vi, zz = np.round(u[s_]).astype(int), np.round(v[s_]).astype(int), z[s_]
        for du in range(-rr, rr + 1):
            for dv in range(-rr, rr + 1):
                a, b = ui + du, vi + dv; k = (a >= 0) & (a < W) & (b >= 0) & (b < H)
                np.minimum.at(buf, b[k] * W + a[k], zz[k])
    d = buf.reshape(H, W); d[~np.isfinite(d)] = 0
    d[~undistort_map()[2]] = 0
    return d * 1000.


def grid_Y(G, T, floor):
    o = T[:3, 3]; f = heading(T); xr = np.cross(f, A.UP)
    cc = G.c[G.near(o[:2], 2.6)]; d = cc - o; lat, fwd, h = d @ xr, d @ f, cc[:, 2] - floor
    m = (np.abs(lat) <= V.X_HALF) & (fwd >= 0) & (fwd < V.NF * V.CELL) & (h >= 0) & (h < V.NH * V.CELL)
    Y = np.zeros((V.NH, V.NF), np.uint8); Y[(h[m] // V.CELL).astype(int), (fwd[m] // V.CELL).astype(int)] = 1
    return Y


def _prepare(spires, work, R):
    """Per run: poses (world <- virtual cam), image paths, TLS depth, TLS grid Y."""
    spires, work = Path(spires), Path(work); out = {}
    by_site = {}
    for run in R:
        by_site.setdefault(S.site_of(run['seq']), []).append(run)
    for site, rr in by_site.items():
        Z = np.load(work / 'vox' / f'{site}.npz'); G = S.Grid(Z['keys'], Z['counts'])
        trs = {run['seq']: Traj(spires / 'seq' / run['seq'] / 'gt-tum.txt') for run in rr}
        info = []
        for run in rr:
            names = image_times(run, work); tim = [float(x[5:-4]) if x else t for x, t in zip(names, run['t_frames'])]
            T = trs[run['seq']].T(np.array(tim)); info.append((run, names, tim, T))
        cents = [T[:, :3, 3].mean(0)[:2] for _, _, _, T in info]
        crops = map_crop(spires, site, cents)
        for (run, names, tim, T), P in zip(info, crops):
            gts = [tls_depth(P, Tk) for Tk in T]
            Ys = np.array([grid_Y(G, Tk, run['floor']) for Tk in T])
            out[run['id']] = dict(names=names, tim=tim, T=T, gts=gts, Y=Ys)
            print('prepared', run['id'], run['seq'][11:], len(names), flush=True)
    return out


def check(spires, work):
    import cv2
    work = Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    pick = R[::max(1, len(R) // 6)][:6]; prep = _prepare(spires, work, pick); tiles = []
    for run in pick:
        p = prep[run['id']]; k = run['run_frames'][-1]
        im = pinhole_rgb(work / 'insert' / 'img' / run['seq'] / p['names'][k][5:])
        gt = p['gts'][k] / 1000.
        edge = cv2.Canny(np.clip(np.log(np.maximum(gt, .3)) * 60, 0, 255).astype(np.uint8), 20, 60) > 0
        ov = im.copy(); ov[edge] = (0, 0, 255)
        boxes = I.world_boxes(run); t_, _ = I.raycast(p['T'][k], (F, F, CX, CY), H, W, boxes)
        ins = im.copy(); mm = np.isfinite(t_) & ((gt == 0) | (t_ < gt)); ins[mm] = (ins[mm] * .3 + np.array(I.COLORS[run['kind']]) * .7).astype(np.uint8)
        cv2.putText(ov, f"{run['id']} {run['seq'][11:]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.putText(ins, run['kind'], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        tiles.append(np.hstack([ov, ins]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(work / 'insert' / 'check_overlay.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    # nearest visible floor distance (row 1 of the current-frame seen channel) is reported after render
    print('wrote check_overlay.jpg')


DAV2_SRC = Path('E:/linnan/linnan/artifacts.local/work/corridor-depth-e1-20260917/upstream/metric_depth')


def _dav2s():
    """Same model and weights as ase_cane_sideview._dav2s; its source copy was removed in storage cleanup, so load the
    surviving Depth-Anything-V2 source copy (state dict must load strictly)."""
    import adt_backbone_blobs_dev_20261013 as B
    B.UP = DAV2_SRC.parent.parent / '_dav2_alias'
    alias = B.UP / 'Depth-Anything-V2'
    if not alias.exists():
        import subprocess
        alias.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(alias), str(DAV2_SRC.parent)], check=True, capture_output=True)
    return V._dav2s()


def render(spires, work):
    import cv2
    import torch
    import torch.nn.functional as Fn
    work = Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    meta = dict(projection=[F, F, CX, CY], size=[W, H])
    L = np.array([.4, .3, .86]); L /= np.linalg.norm(L)
    model = _dav2s(); rng = np.random.default_rng(7)
    prep = _prepare(spires, work, R)

    def dav2(ims):
        zs = []
        for b in range(0, len(ims), 8):
            t_ = torch.cat([model.image2tensor(im, 518)[0] for im in ims[b:b + 8]])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t_)
            zs += list(Fn.interpolate(dd[:, None].float(), (H, W), mode='bilinear', align_corners=True)[:, 0].cpu().numpy())
        return zs
    scales = []
    for run in R:
        p = prep[run['id']]; od = work / 'insert' / str(run['id']); od.mkdir(exist_ok=True)
        boxes = I.world_boxes(run); col = np.array(I.COLORS[run['kind']], float)
        frames = [(k, t) for k, t in zip(run['frames'], p['tim'])]; jx = {k: k for k in run['frames']}
        T_up = list(p['T']); Z = dict(Y=p['Y'], frames=np.array(run['frames']))
        base, ims, gts = [], [], []
        for k in run['frames']:
            nm = p['names'][k]
            rgb = pinhole_rgb(work / 'insert' / 'img' / run['seq'] / nm[5:]) if nm else np.zeros((H, W, 3), np.uint8)
            gt = p['gts'][k].astype(np.float32).copy(); base.append((rgb.copy(), gt.copy()))
            t, n = I.raycast(T_up[k], meta['projection'], H, W, boxes)
            m = np.isfinite(t) & ((gt == 0) | (t * 1000 < gt)) & undistort_map()[2]
            shade = (.35 + .65 * np.clip(n @ L, 0, 1))[..., None] * col + rng.normal(0, 6, (H, W, 1))
            rgb[m] = np.clip(shade[m], 0, 255).astype(np.uint8); gt[m] = t[m] * 1000
            if k == run['run_frames'][-1] or k == run['run_frames'][len(run['run_frames']) // 2]:
                cv2.imwrite(str(od / f'{k:03d}.jpg'), rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
            ims.append(rgb); gts.append(gt)
        zs = dav2(ims)
        for gt, z in zip(gts, zs):
            m = (gt > 0) & (gt < A.MAX_MM) & (z > 0)
            if m.sum() > 100:
                scales.append(float(np.median(gt[m] / 1000. / z[m])))
        X, Yg, lab, dd_ = I.inputs_and_labels(run, frames, T_up, Z, jx, meta, gts, zs)
        np.savez_compressed(od / 'grid.npz', X=X, Y=Yg, d=np.array([np.nan if v is None else v for v in dd_]),
                            lab=np.array(lab), frames=np.array(run['run_frames']))
        # empty run: same frames, no object (dummy box far above, never in any grid)
        e = dict(run, boxes=[((0., 0., 100.), (.01, .01, .01))])
        ze = dav2([b[0] for b in base])
        Xe, Ye, _, _ = I.inputs_and_labels(e, frames, T_up, Z, jx, meta, [b[1] for b in base], ze)
        np.savez_compressed(od / 'grid_empty.npz', X=Xe, Y=Ye, frames=np.array(run['run_frames']))
        print('run', run['id'], run['kind'], run['seq'][11:], 'labels', dict(zip(*np.unique(lab, return_counts=True))), flush=True)
    (work / 'insert' / 'render_stats.json').write_text(json.dumps(dict(
        dav2_scale_pct=np.percentile(scales, [10, 25, 50, 75, 90]).round(3).tolist(), n=len(scales))), encoding='utf8')


def score(ase_work, work):
    import torch
    ase_work, work = Path(ase_work), Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    D = I.D
    net = D.make_head(8).cuda(); net.load_state_dict(torch.load(ase_work / 'decision_head8_all.pt')); net.eval()
    dec = lambda occ: (lambda r: bool(r[0] and not r[2]))(V.decide(occ))
    per, empty, floor_seen = [], dict(frames=0, alert_frames=0, episodes=0, seconds=0.), []
    for run in R:
        od = work / 'insert' / str(run['id']); g = np.load(od / 'grid.npz'); ge = np.load(od / 'grid_empty.npz')
        with torch.no_grad():
            s = torch.sigmoid(net(V.pad(torch.from_numpy(g['X']).cuda().float()))).cpu().numpy()
            se = torch.sigmoid(net(V.pad(torch.from_numpy(ge['X']).cuda().float()))).cpu().numpy()
        al = dict(D8=s > I.THETA, H=np.array([dec((x[0] > 0) | (x[1] > 0)) for x in g['X']]), G=np.array([dec(y) for y in g['Y']]))
        per.append(dict(id=run['id'], kind=run['kind'], seq=run['seq'], lab=g['lab'].tolist(), d=g['d'].tolist(), s=s.tolist(),
                        s_empty=se.tolist(), **{a: v.tolist() for a, v in al.items()}))
        ae = se > I.THETA; empty['frames'] += len(ae); empty['alert_frames'] += int(ae.sum())
        empty['episodes'] += int(ae[0]) + int(((~ae[:-1]) & ae[1:]).sum()); empty['seconds'] += len(ae) / S.HZ
        for x in g['X']:
            a = np.flatnonzero(x[4][1] > 0)
            if len(a):
                floor_seen.append((a[0] + .5) * V.CELL)

    def summarize(alerts_of):
        out = {}
        for grp, kinds in (('miss', I.MISS_TYPES), ('control', set(I.TYPES) - I.MISS_TYPES)):
            caught = t07 = t10 = disturbed = nnear = 0
            for p in [p for p in per if p['kind'] in kinds]:
                a = alerts_of(p); near = [q for q, l in enumerate(p['lab']) if l.endswith('_near')]
                if not near:
                    continue
                nnear += 1; hit = [q for q in near if a[q]]
                if grp == 'miss' and hit:
                    caught += 1; t07 += p['d'][hit[0]] >= .7; t10 += p['d'][hit[0]] >= 1.
                if grp == 'control':
                    disturbed += bool(hit)
            out[grp] = dict(runs=nnear, caught=caught, timely_07=t07, timely_10=t10) if grp == 'miss' else dict(runs=nnear, disturbed=disturbed)
        return out
    rep = {a: summarize(lambda p, a=a: p[a]) for a in ('D8', 'H', 'G')}
    rep['curve_D8'] = {str(t): summarize(lambda p, t=t: np.array(p['s']) > t) for t in (.1, .2, .3, .4, .5, .6, .7, .8, .9)}
    rep['by_kind'] = {}
    for kk in I.TYPES:
        rr = [p for p in per if p['kind'] == kk]; nr = [p for p in rr if any(l.endswith('_near') for l in p['lab'])]
        rep['by_kind'][kk] = dict(runs=len(rr), near=len(nr), **{a: sum(any(p[a][q] for q, l in enumerate(p['lab']) if l.endswith('_near'))
                                                                      for p in nr) for a in ('D8', 'H', 'G')})
    # miss runs not caught by D8: never detected in the body band vs suppressed (decomposition via H grid for context)
    empty['alerts_per_min'] = round(empty['episodes'] / max(empty['seconds'], 1e-6) * 60, 2)
    rep['empty_runs_D8'] = empty
    rep['floor_seen_m_pct'] = np.percentile(floor_seen, [10, 25, 50, 75, 90]).round(2).tolist() if floor_seen else None
    (work / 'insert' / 'score.json').write_text(json.dumps(dict(report=rep, runs=per), indent=1,
                                                         default=lambda o: o.item() if hasattr(o, 'item') else str(o)), encoding='utf8')
    print(json.dumps({a: rep[a] for a in ('D8', 'H', 'G')}, indent=1))
    print(json.dumps(rep['by_kind'], indent=1)); print('empty', empty, 'floor_seen', rep['floor_seen_m_pct'])
    for t, v in rep['curve_D8'].items():
        print('curve', t, v)


if __name__ == '__main__':
    dict(runs=runs, fetch=fetch, check=check, render=render, score=score)[sys.argv[1]](*sys.argv[2:])

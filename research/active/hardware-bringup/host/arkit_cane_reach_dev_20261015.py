"""Cane-reach positives mined from real homes: ARKitScenes raw (iPad LiDAR depth + ARKit poses) (EXPLORE, 2026-10-15).

Question and gate are frozen in <work>/PLAN.md before any download. Rules mirror sanpo/musohu_cane_reach (body 0.60-1.90 m
above ground, cane band 0.05-0.30 m, corridor +-0.30 m x 0.30-2.0 m, +0.15 m margin, 3 cm voxels, >=3 voxels => CANE).
ARKitScenes specifics fixed before results:
  frames   : lowres_depth + confidence, every 24th 60 Hz frame (2.5 Hz); pose = lowres_wide.traj (10 Hz world->camera
             angle-axis; slerp/linear interpolation, inverted); intrinsics = median of 5 pincam files.
  world    : ARKitScenes traj world is Z-up (checked on the first capture: camera image-down axis -> -Z, floor at min Z).
  positive : body-band points with confidence 2 whose summed solid angle >= 8.3e-4 sr (Atom 150 px threshold).
  support  : confidence >= 1 points (generous => conservative MISS), fused over +-10 s.
  ground   : per frame, +-10 s window points within 3 m horizontally; 1 cm height histogram peak within 0.15 m above
             the 1st percentile.
Usage:
  arkit_cane_reach_dev_20261015.py select <work>
  arkit_cane_reach_dev_20261015.py fetch  <work>
  arkit_cane_reach_dev_20261015.py label  <work>
  arkit_cane_reach_dev_20261015.py sample <work>
"""
from __future__ import annotations

import csv
import glob
import io
import json
import struct
import sys
import threading
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw'
N_VISITS, STRIDE, LIMIT = 40, 24, 1_500_000_000
VOX, WIN_S, BAND, MARGIN, MIN_VOX = .03, 10., (.05, .30), .15, 3
BODY, X_HALF, Z_RANGE, MIN_SR, WALK, NEAR = (.60, 1.90), .30, (.30, 2.0), 8.3e-4, .30, 1.5
WORKROOT = Path(__file__).resolve().parents[4] / 'artifacts.local' / 'work'

_lock = threading.Lock()
_bytes = [0]


def _get(session, url, a=None, b=None):
    import requests  # noqa: F401
    h = {'Accept-Encoding': 'identity'}
    if a is not None:
        h['Range'] = f'bytes={a}-{b}'
    for attempt in range(4):
        try:
            r = session.get(url, headers=h, timeout=(10, 60))
            if r.status_code not in ((206,) if a is not None else (200,)):
                raise ValueError(f'HTTP {r.status_code} {url}')
            with _lock:
                _bytes[0] += len(r.content)
                if _bytes[0] > LIMIT:
                    raise RuntimeError('download limit exceeded')
            return r.content
        except (OSError, ValueError) as exc:
            if attempt == 3:
                raise
            err = exc
    raise err


def _size(session, url):
    r = session.head(url, timeout=(10, 30))
    if r.status_code != 200:
        raise ValueError(f'HEAD {r.status_code} {url}')
    return int(r.headers['Content-Length'])


def central_directory(session, url):
    """name -> (local header offset, compressed size, method). Handles zip64 via zipfile on a ranged tail buffer."""
    import zipfile
    size = _size(session, url)
    tail_len = min(size, 4 << 20)
    while True:
        tail = _get(session, url, size - tail_len, size - 1)

        class Tail(io.RawIOBase):
            pos = 0
            def seekable(self): return True
            def readable(self): return True
            def tell(self): return self.pos
            def seek(self, n, w=0):
                self.pos = n if w == 0 else self.pos + n if w == 1 else size + n
                return self.pos
            def read(self, n=-1):
                off = self.pos - (size - tail_len)
                if off < 0:
                    raise EOFError('need longer tail')
                n = size - self.pos if n < 0 else n
                out = tail[off:off + n]; self.pos += len(out); return out
        try:
            z = zipfile.ZipFile(Tail())
            return {i.filename: (i.header_offset, i.compress_size, i.compress_type) for i in z.infolist()}
        except EOFError:
            if tail_len == size:
                raise
            tail_len = min(size, tail_len * 4)


def member(session, url, entry):
    off, csize, method = entry
    raw = _get(session, url, off, off + 30 + 512 + csize - 1)
    if raw[:4] != b'PK\x03\x04':
        raise ValueError('bad local header')
    n, e = struct.unpack('<HH', raw[26:30])
    data = raw[30 + n + e:30 + n + e + csize]
    if len(data) != csize:
        data += _get(session, url, off + len(raw), off + 30 + n + e + csize - 1)
    return data if method == 0 else zlib.decompress(data, -15)


def select(work):
    work = Path(work)
    rows = list(csv.DictReader((WORKROOT / 'rgb-backbone-confirm-dev-20261010/official/raw_raw_train_val_splits.csv')
                               .read_text(encoding='utf-8-sig').splitlines()))
    caps, visits = set(), set()
    for f in glob.glob(str(WORKROOT / '*' / 'consumed_inventory.json')):
        d = json.loads(Path(f).read_text(encoding='utf-8-sig'))
        caps |= set(map(str, d.get('capture_ids', []))); visits |= set(map(str, d.get('visit_ids', [])))
    chosen, seen = [], set()
    for r in rows:
        if r['fold'] != 'Training' or r['visit_id'] in ('', 'NA') or r['video_id'] in caps or r['visit_id'] in visits:
            continue
        if r['visit_id'] in seen:
            continue
        seen.add(r['visit_id']); chosen.append(dict(capture=r['video_id'], visit=r['visit_id']))
        if len(chosen) == N_VISITS:
            break
    (work / 'selection.json').write_text(json.dumps(dict(excluded_captures=len(caps), excluded_visits=len(visits),
                                                         chosen=chosen), indent=1), encoding='utf8')
    print(len(chosen), 'captures selected')


def fetch_one(work, cap):
    import requests
    out = work / 'frames' / f'{cap}.npz'
    if out.exists():
        return cap, 'cached'
    s = requests.Session()
    root = f'{BASE}/Training/{cap}'
    traj = _get(s, f'{root}/lowres_wide.traj').decode()
    cds = {a: central_directory(s, f'{root}/{a}.zip') for a in ('lowres_depth', 'confidence', 'lowres_wide_intrinsics')}
    stem = lambda n: n.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    dnames = sorted((n for n in cds['lowres_depth'] if n.endswith('.png')), key=lambda n: float(stem(n).rsplit('_', 1)[1]))
    cnames = {stem(n): n for n in cds['confidence'] if n.endswith('.png')}
    picked = [n for n in dnames[::STRIDE] if stem(n) in cnames]
    knames = sorted(n for n in cds['lowres_wide_intrinsics'] if n.endswith('.pincam'))
    K = np.median([np.array(member(s, f'{root}/lowres_wide_intrinsics.zip', cds['lowres_wide_intrinsics'][knames[int(i)]])
                            .decode().split(), float) for i in np.linspace(0, len(knames) - 1, 5)], 0)
    import cv2
    depth, conf = [], []
    for n in picked:
        d = cv2.imdecode(np.frombuffer(member(s, f'{root}/lowres_depth.zip', cds['lowres_depth'][n]), np.uint8), cv2.IMREAD_UNCHANGED)
        c = cv2.imdecode(np.frombuffer(member(s, f'{root}/confidence.zip', cds['confidence'][cnames[stem(n)]]), np.uint8),
                         cv2.IMREAD_UNCHANGED)
        depth.append(d); conf.append(c)
    ts = np.array([float(stem(n).rsplit('_', 1)[1]) for n in picked])
    tr = np.array([[float(x) for x in line.split()] for line in traj.splitlines() if line.strip()])
    from scipy.spatial.transform import Rotation, Slerp
    ok = (ts >= tr[0, 0]) & (ts <= tr[-1, 0])                    # traj is 10 Hz world->camera: interpolate (slerp + linear)
    Rwc = Slerp(tr[:, 0], Rotation.from_rotvec(tr[:, 1:4]))(ts[ok]).as_matrix()
    tw = np.stack([np.interp(ts[ok], tr[:, 0], tr[:, 4 + a]) for a in range(3)], 1)
    T = np.zeros((len(ts), 4, 4)); T[:, 3, 3] = 1
    T[ok, :3, :3] = np.transpose(Rwc, (0, 2, 1)); T[ok, :3, 3] = -np.einsum('kji,kj->ki', Rwc, tw)   # camera -> world
    np.savez_compressed(out, depth=np.stack(depth)[ok], conf=np.stack(conf)[ok], ts=ts[ok], T_wc=T[ok], K=K,
                        names=np.array(picked)[ok])
    return cap, f'{ok.sum()}/{len(ts)} frames'


def fetch(work):
    work = Path(work); (work / 'frames').mkdir(parents=True, exist_ok=True)
    caps = [c['capture'] for c in json.loads((work / 'selection.json').read_text())['chosen']]
    log = {}
    with ThreadPoolExecutor(8) as ex:
        for cap, msg in ex.map(lambda c: _safe_fetch(work, c), caps):
            log[cap] = msg; print(cap, msg, f'{_bytes[0] / 1e6:.0f} MB', flush=True)
    (work / 'fetch_log.json').write_text(json.dumps(dict(log=log, bytes=_bytes[0]), indent=1), encoding='utf8')


def _safe_fetch(work, cap):
    try:
        return fetch_one(work, cap)
    except Exception as exc:                                      # record and continue; failures stay in the log
        return cap, f'FAILED {exc!r}'


def unproject(depth, K, step=1):
    w, h, fx, fy, cx, cy = K
    sx, sy = depth.shape[1] / w, depth.shape[0] / h
    fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
    vv, uu = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    x, y = (uu - cx) / fx, (vv - cy) / fy
    dirs = np.stack([x, y, np.ones_like(x)], -1).reshape(-1, 3)
    sr = (1 / (fx * fy) / (1 + x * x + y * y) ** 1.5).reshape(-1)   # pixel solid angle
    return dirs, sr


def label_capture(path):
    Z = np.load(path)
    depth, conf, ts, T, K = Z['depth'], Z['conf'], Z['ts'], Z['T_wc'], Z['K']
    dirs, sr = unproject(depth[0], K)
    n = len(ts); pts, cfs = [], []
    for j in range(n):
        z = depth[j].reshape(-1) / 1000.; ok = (z > .1) & (z < 5.)
        P = (T[j, :3, :3] @ (dirs[ok] * z[ok, None]).T).T + T[j, :3, 3]
        pts.append((P.astype(np.float32), np.flatnonzero(ok))); cfs.append(conf[j].reshape(-1)[ok])
    up = np.array([0., 0., 1.])
    t = ts - ts[0]
    pos = T[:, [0, 1], 3]; lo, hi = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
    speed = np.linalg.norm(pos[hi] - pos[lo], axis=1) / np.maximum(t[hi] - t[lo], 1e-3)
    rows = []
    for i in range(n):
        o = T[i, :3, 3]
        win = np.flatnonzero(np.abs(t - t[i]) <= WIN_S)
        W = np.concatenate([pts[j][0] for j in win]); Wc = np.concatenate([cfs[j] for j in win])
        near = np.hypot(W[:, 0] - o[0], W[:, 1] - o[1]) <= 3.
        hy = W[near, 2]
        if len(hy) < 500:
            rows.append(None); continue
        p1 = np.percentile(hy, 1); hist, edges = np.histogram(hy, bins=np.arange(p1 - .02, p1 + .15, .01))
        floor = float(edges[hist.argmax()] + .005)
        f = T[i, :3, 2] - up * (T[i, :3, 2] @ up); nf = np.linalg.norm(f)
        if nf < .2:                                               # looking straight down/up: no defined corridor
            rows.append(None); continue
        f /= nf; xr = np.cross(f, up)
        P, idx = pts[i]; c = cfs[i]
        lat, fwd, hgt = (P - o) @ xr, (P - o) @ f, P[:, 2] - floor
        body = (np.abs(lat) <= X_HALF) & (fwd >= Z_RANGE[0]) & (fwd <= Z_RANGE[1]) & (hgt >= BODY[0]) & (hgt <= BODY[1]) & (c == 2)
        omega = float(sr[idx[body]].sum())
        cam_h = float(o[2] - floor); pitch = float(np.degrees(np.arcsin(np.clip(T[i, :3, 2] @ up, -1, 1))))
        if omega < MIN_SR:
            rows.append(dict(pos=False, omega=omega, speed=float(speed[i]), cam_h=cam_h, pitch=pitch)); continue
        d_body = float(np.percentile(fwd[body], 5))
        gh = W[:, 2] - floor
        g = W[(gh >= BAND[0]) & (gh <= BAND[1]) & (Wc >= 1)]
        gl, gf = (g - o) @ xr, (g - o) @ f
        sup = (np.abs(gl) <= X_HALF) & (gf >= 0) & (gf <= d_body + MARGIN)
        nvox = len(np.unique(np.floor(g[sup] / VOX).astype(np.int64), axis=0)) if sup.any() else 0
        rows.append(dict(pos=True, omega=omega, d_body=d_body, support_vox=int(nvox), speed=float(speed[i]), cam_h=cam_h,
                         pitch=pitch, body_h=[float(np.percentile(hgt[body], 5)), float(np.percentile(hgt[body], 95))]))
    return path.stem, dict(rows=rows, ts=ts.tolist(), names=Z['names'].tolist())


def label(work):
    work = Path(work)
    files = sorted((work / 'frames').glob('*.npz'))
    with ProcessPoolExecutor(7) as ex:
        res = dict(ex.map(label_capture, files))
    sel = {c['capture']: c['visit'] for c in json.loads((work / 'selection.json').read_text())['chosen']}
    stats, per_visit = {}, {}
    def add(k, v=1): stats[k] = stats.get(k, 0) + v
    for cap, r in res.items():
        for row in r['rows']:
            if row is None:
                add('no_corridor_or_ground'); continue
            subs = ('all', 'moving') if row['speed'] >= WALK else ('all',)
            for sub in subs:
                add(f'{sub}|frames')
                if not row['pos']:
                    add(f'{sub}|NONPOS'); continue
                cat = 'CANE' if row['support_vox'] >= MIN_VOX else 'MISS'
                band = 'near' if row['d_body'] <= NEAR else 'far'
                add(f'{sub}|{cat}|{band}')
                if sub == 'all' and cat == 'MISS' and band == 'near':
                    per_visit[sel[cap]] = per_visit.get(sel[cap], 0) + 1
    # contiguous MISS-near runs (events) to show autocorrelation
    runs = 0
    for r in res.values():
        prev = False
        for row in r['rows']:
            cur = bool(row and row['pos'] and row['support_vox'] < MIN_VOX and row['d_body'] <= NEAR)
            runs += cur and not prev; prev = cur
    rep = dict(stats=stats, miss_near_per_visit=per_visit, miss_near_visits=len(per_visit), miss_near_runs=runs,
               captures=len(res))
    (work / 'labels.json').write_text(json.dumps(dict(report=rep, labels=res)), encoding='utf8')
    print(json.dumps(rep, indent=1))


def sample(work, k=16, seed=20261015):
    """Fixed-seed sample of near MISS frames; fetch their RGB and draw body points (red) + support voxels (green)."""
    import cv2
    import requests
    work = Path(work); L = json.loads((work / 'labels.json').read_text())['labels']
    pool = [(cap, i) for cap, r in sorted(L.items()) for i, row in enumerate(r['rows'])
            if row and row['pos'] and row['support_vox'] < MIN_VOX and row['d_body'] <= NEAR]
    rng = np.random.default_rng(seed); pick = [pool[i] for i in sorted(rng.choice(len(pool), min(k, len(pool)), replace=False))]
    s = requests.Session(); tiles = []
    cds = {}
    for cap, i in pick:
        root = f'{BASE}/Training/{cap}'
        if cap not in cds:
            cds[cap] = central_directory(s, f'{root}/lowres_wide.zip')
        name = L[cap]['names'][i].rsplit('/', 1)[-1]
        rgb = cv2.imdecode(np.frombuffer(member(s, f'{root}/lowres_wide.zip', cds[cap][f'lowres_wide/{name}']), np.uint8), 1)
        Z = np.load(work / 'frames' / f'{cap}.npz'); d = Z['depth'][i] / 1000.; T, K = Z['T_wc'][i], Z['K']
        dirs, _ = unproject(Z['depth'][i], K); P = (T[:3, :3] @ (dirs * d.reshape(-1, 1)).T).T + T[:3, 3]
        row = L[cap]['rows'][i]; up = np.array([0., 0., 1.]); o = T[:3, 3]
        floor = o[2] - row['cam_h']; f = T[:3, 2] - up * (T[:3, 2] @ up); f /= np.linalg.norm(f); xr = np.cross(f, up)
        lat, fwd, hgt = (P - o) @ xr, (P - o) @ f, P[:, 2] - floor
        body = (np.abs(lat) <= X_HALF) & (fwd >= Z_RANGE[0]) & (fwd <= Z_RANGE[1]) & (hgt >= BODY[0]) & (hgt <= BODY[1]) & (Z['conf'][i].reshape(-1) == 2) & (d.reshape(-1) > .1)
        corr = (np.abs(lat) <= X_HALF) & (fwd >= 0) & (fwd <= row['d_body'] + MARGIN) & (hgt < BODY[0]) & (d.reshape(-1) > .1)
        m = np.zeros(d.shape + (3,), np.uint8); m[..., 2] = body.reshape(d.shape) * 255; m[..., 1] = corr.reshape(d.shape) * 160
        m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        tile = cv2.addWeighted(rgb, 1., m, .55, 0)
        cv2.putText(tile, f'{cap}#{i} d={row["d_body"]:.2f} h={row["body_h"][0]:.2f}-{row["body_h"][1]:.2f} cam={row["cam_h"]:.2f}',
                    (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .33, (255, 255, 255), 1)
        tiles.append(cv2.resize(tile, (320, 240)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
    cv2.imwrite(str(work / 'miss_near_sample.jpg'), sheet)
    (work / 'miss_near_sample.json').write_text(json.dumps(dict(pool=len(pool), seed=seed, picked=pick), indent=1), encoding='utf8')
    print(len(pool), 'near MISS frames; sheet written')


if __name__ == '__main__':
    dict(select=select, fetch=fetch, label=label, sample=sample)[sys.argv[1]](*sys.argv[2:])

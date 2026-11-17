"""Oxford Spires cane-reach label pilot (EXPLORE, 2026-11-17).

Labels come from the TLS ground-truth map plus the TLS-registered trajectory, so they do not depend on what the
device camera sees. Rule = ADT review L2 / ASE v2 (body 0.60-1.90 m, +-0.25 m, 0.30-2.0 m; cane band 0.05-0.70 m,
+-0.35 m, 0 .. d_body + 0.15 m, >= 3 voxels of 3 cm). Plan: artifacts.local/work/spires-cane-reach-20261117/PLAN.md

  python spires_cane_reach_dev_20261117.py voxels <spires_dir> <work>   # map -> 3 cm voxels near trajectories
  python spires_cane_reach_dev_20261117.py label  <spires_dir> <work>   # self-check + labels + summary
"""
import json
import sys
from pathlib import Path

import numpy as np

VOX = .03
BODY_H, BODY_HALF, Z_RANGE, MIN_BODY_VOX = (.60, 1.90), .25, (.30, 2.0), 10
BAND, CANE_HALF, MARGIN, MIN_VOX = (.05, .70), .35, .15, 3
NEAR, WALK = 1.5, .30
HZ, GROUND_R, GROUND_MIN = 5., .35, 50
KEEP_R = 3.0  # keep map voxels within this xy distance of any trajectory sample
SITES = {'keble-college': 'keble-college', 'observatory-quarter': 'observatory-quarter',
         'blenheim-palace': 'blenheim-palace', 'christ-church': 'christ-church'}


def site_of(seq):
    for s in SITES:
        if s in seq:
            return s
    return None


def read_tum(p):
    a = np.loadtxt(p)
    from scipy.spatial.transform import Rotation as Rot
    return a[:, 0], a[:, 1:4], Rot.from_quat(a[:, 4:8]).as_matrix()


def resample(t, P, R, hz=HZ):
    from scipy.spatial.transform import Rotation as Rot, Slerp
    ts = np.arange(t[0], t[-1], 1 / hz)
    Pi = np.stack([np.interp(ts, t, P[:, k]) for k in range(3)], 1)
    Ri = Slerp(t, Rot.from_matrix(R))(ts).as_matrix()
    return ts, Pi, Ri


def read_pcd_xyz(path, chunk=8_000_000):
    """Yield float32 xyz chunks from a binary PCD (fields parsed from the header)."""
    with open(path, 'rb') as f:
        hdr = {}
        while True:
            line = f.readline().decode('ascii').strip()
            k, *v = line.split()
            hdr[k] = v
            if k == 'DATA':
                break
        off = f.tell()
    assert hdr['DATA'][0] == 'binary', hdr['DATA']
    names, sizes, types, counts = hdr['FIELDS'], list(map(int, hdr['SIZE'])), hdr['TYPE'], list(map(int, hdr['COUNT']))
    fields = []
    for i, (n, s, ty, c) in enumerate(zip(names, sizes, types, counts)):
        dt = {'F': 'f', 'U': 'u', 'I': 'i'}[ty] + str(s)
        fields.append((n if n != '_' else f'pad{i}', dt, (c,)) if c > 1 else (n if n != '_' else f'pad{i}', dt))
    dtype = np.dtype(fields)
    n = int(hdr['POINTS'][0])
    mm = np.memmap(path, dtype=dtype, mode='r', offset=off, shape=(n,))
    for a in range(0, n, chunk):
        b = mm[a:a + chunk]
        yield np.stack([b['x'], b['y'], b['z']], 1).astype(np.float32)


def voxels(spires, work, *only):
    spires, work = Path(spires), Path(work); (work / 'vox').mkdir(parents=True, exist_ok=True)
    for site in (only or SITES):
        out = work / 'vox' / f'{site}.npz'
        mp = spires / 'map' / f'{site}.pcd'
        if out.exists() or not mp.exists():
            continue
        xy = np.concatenate([read_tum(s / 'gt-tum.txt')[1][::10, :2] for s in sorted((spires / 'seq').iterdir())
                             if site_of(s.name) == site])
        cell = .5; g0 = np.floor(xy.min(0) / cell).astype(int) - 8
        keep = np.zeros(tuple(np.floor(xy.max(0) / cell).astype(int) - g0 + 9), bool)
        ij = np.floor(xy / cell).astype(int) - g0
        r = int(np.ceil(KEEP_R / cell))
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di * di + dj * dj <= r * r:
                    keep[np.clip(ij[:, 0] + di, 0, keep.shape[0] - 1), np.clip(ij[:, 1] + dj, 0, keep.shape[1] - 1)] = True
        keys, cnts, total = [], [], 0
        for P in read_pcd_xyz(mp):
            total += len(P)
            q = np.floor(P[:, :2] / cell).astype(int) - g0
            ok = (q >= 0).all(1) & (q[:, 0] < keep.shape[0]) & (q[:, 1] < keep.shape[1])
            ok[ok] = keep[q[ok, 0], q[ok, 1]]
            k = np.floor(P[ok] / VOX).astype(np.int64)
            u, c = np.unique(k, axis=0, return_counts=True)
            keys.append(u); cnts.append(c)
        k = np.concatenate(keys); c = np.concatenate(cnts)
        u, inv = np.unique(k, axis=0, return_inverse=True)
        cc = np.bincount(inv.ravel(), weights=c).astype(np.int32)
        np.savez_compressed(out, keys=u.astype(np.int32), counts=cc, total=total)
        print(site, 'map points', total, 'voxels kept', len(u), flush=True)


class Grid:
    def __init__(self, keys, counts, tile=1.0):
        self.c = (keys + .5) * VOX; self.n = counts
        t = np.floor(self.c[:, :2] / tile).astype(np.int64); self.tile = tile
        tid = (t[:, 0] << 32) + (t[:, 1] & 0xffffffff)
        o = np.argsort(tid); self.c, self.n, tid = self.c[o], self.n[o], tid[o]
        u, s = np.unique(tid, return_index=True)
        self.idx = dict(zip(u.tolist(), zip(s.tolist(), np.r_[s[1:], len(tid)].tolist())))

    def near(self, xy, r):
        a = np.floor((xy - r) / self.tile).astype(int); b = np.floor((xy + r) / self.tile).astype(int)
        parts = []
        for i in range(a[0], b[0] + 1):
            for j in range(a[1], b[1] + 1):
                s = self.idx.get((i << 32) + (j & 0xffffffff))
                if s:
                    parts.append(np.arange(*s))
        return np.concatenate(parts) if parts else np.zeros(0, int)


def ground_at(G, xy, zdev):
    ii = G.near(xy, GROUND_R)
    c = G.c[ii]; m = (np.hypot(*(c[:, :2] - xy).T) <= GROUND_R) & (c[:, 2] < zdev - .8)
    if not m.any():
        return np.nan
    z = c[m, 2]; w = G.n[ii][m]
    lo = np.floor(z.min() / VOX); h = np.bincount((np.floor(z / VOX) - lo).astype(int), weights=w)
    k = np.flatnonzero(h >= GROUND_MIN)
    return float((lo + k[0] + .5) * VOX) if len(k) else np.nan


def label(spires, work):
    spires, work = Path(spires), Path(work)
    grids, out, check = {}, {}, {}
    for s in sorted((spires / 'seq').iterdir()):
        site = site_of(s.name)
        if site is None or not (work / 'vox' / f'{site}.npz').exists():
            continue
        if site not in grids:
            Z = np.load(work / 'vox' / f'{site}.npz'); grids[site] = Grid(Z['keys'], Z['counts'])
        G = grids[site]
        t, P, R = resample(*read_tum(s / 'gt-tum.txt'))
        n = len(t); lo, hi = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
        v = (P[hi, :2] - P[lo, :2]) / np.maximum(t[hi] - t[lo], 1e-3)[:, None]; speed = np.linalg.norm(v, axis=1)
        walk = speed >= WALK
        # forward axis self-check: which base axis (+-x,+-y,+-z) projected horizontally follows the walking direction
        vd = v[walk] / speed[walk, None]; best = None
        for ax in range(3):
            for sg in (1, -1):
                f = sg * R[walk][:, :2, ax]; nf = np.linalg.norm(f, axis=1)
                ang = np.degrees(np.arccos(np.clip((f * vd).sum(1) / np.maximum(nf, 1e-6), -1, 1)))
                med = float(np.median(ang))
                if best is None or med < best[0]:
                    best = (med, ax, sg)
        med_ang, ax, sg = best
        gz = np.array([ground_at(G, P[i, :2], P[i, 2]) for i in range(n)])
        ok = np.isfinite(gz)
        from scipy.ndimage import median_filter
        gs = gz.copy(); gs[ok] = median_filter(gz[ok], size=int(2 * HZ) + 1, mode='nearest')
        dev_h = P[:, 2] - gs
        # gravity self-check: PCA plane normal of ground-layer voxels within 1 m at 5 spread-out frames
        tilt = []
        for i in np.flatnonzero(ok)[np.linspace(0, ok.sum() - 1, 5).astype(int)]:
            c = G.c[G.near(P[i, :2], 1.0)]
            c = c[(np.hypot(*(c[:, :2] - P[i, :2]).T) <= 1.0) & (np.abs(c[:, 2] - gs[i]) <= .06)]
            if len(c) >= 200:
                nrm = np.linalg.svd(c - c.mean(0), full_matrices=False)[2][2]
                tilt.append(float(np.degrees(np.arccos(abs(nrm[2])))))
        check_tilt = round(float(np.median(tilt)), 2) if tilt else None
        check[s.name] = dict(ground_tilt_deg_median=check_tilt,n=n, walk_frac=round(float(walk.mean()), 3), fwd_axis=f'{"+" if sg > 0 else "-"}{"xyz"[ax]}',
                             fwd_vs_walk_deg_median=round(med_ang, 1), ground_found_walk=round(float(ok[walk].mean()), 3),
                             device_height_walk_median=round(float(np.nanmedian(dev_h[walk])), 3),
                             path_m=round(float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum()), 1),
                             minutes=round(float((t[-1] - t[0]) / 60), 2))
        rows = []
        for i in range(n):
            if not np.isfinite(gs[i]):
                rows.append(None); continue
            f = sg * R[i, :2, ax]; f = f / np.linalg.norm(f); l = np.array([-f[1], f[0]])
            ii = G.near(P[i, :2], 2.6); c = G.c[ii]
            d = c[:, :2] - P[i, :2]; fw, lat, hg = d @ f, d @ l, c[:, 2] - gs[i]
            body = (np.abs(lat) <= BODY_HALF) & (fw >= Z_RANGE[0]) & (fw <= Z_RANGE[1]) & (hg >= BODY_H[0]) & (hg <= BODY_H[1])
            if body.sum() < MIN_BODY_VOX:
                rows.append(dict(neg=True)); continue
            db = float(np.percentile(fw[body], 5))
            sup = (np.abs(lat) <= CANE_HALF) & (fw >= 0) & (fw <= db + MARGIN) & (hg >= BAND[0]) & (hg <= BAND[1])
            rows.append(dict(d_body=round(db, 3), support_vox=int(sup.sum()), body_vox=int(body.sum()),
                             body_hmin=round(float(hg[body].min()), 2)))
        out[s.name] = dict(t=np.round(t, 3).tolist(), speed=np.round(speed, 3).tolist(), dev_h=np.round(dev_h, 3).tolist(),
                           xy=np.round(P[:, :2], 2).tolist(), rows=rows)
        print(s.name, check[s.name], flush=True)
    (work / 'labels.json').write_text(json.dumps(out), encoding='utf8')
    (work / 'selfcheck.json').write_text(json.dumps(check, indent=1), encoding='utf8')
    summary(work)


def summary(work):
    work = Path(work); L = json.loads((work / 'labels.json').read_text(encoding='utf8'))
    rep, misses = {}, []
    for s, v in L.items():
        c = dict(frames=len(v['rows']), walk=0, none=0, NEG=0, CANE=0, MISS=0, CANE_near=0, MISS_near=0)
        for k, r in enumerate(v['rows']):
            if v['speed'][k] < WALK:
                continue
            c['walk'] += 1
            if r is None:
                c['none'] += 1; continue
            if r.get('neg'):
                c['NEG'] += 1; continue
            cat = 'CANE' if r['support_vox'] >= MIN_VOX else 'MISS'; c[cat] += 1
            if r['d_body'] <= NEAR:
                c[cat + '_near'] += 1
                if cat == 'MISS':
                    misses.append(dict(seq=s, k=k, t=v['t'][k], d_body=r['d_body'], body_hmin=r['body_hmin'],
                                       body_vox=r['body_vox'], xy=v['xy'][k]))
        rep[s] = c
    # group near-MISS frames into events (same sequence, gap <= 1 s)
    ev = []
    for m in misses:
        if ev and ev[-1]['seq'] == m['seq'] and m['t'] - ev[-1]['t1'] <= 1.0:
            ev[-1]['t1'] = m['t']; ev[-1]['frames'] += 1; ev[-1]['d_min'] = min(ev[-1]['d_min'], m['d_body'])
        else:
            ev.append(dict(seq=m['seq'], t0=m['t'], t1=m['t'], frames=1, d_min=m['d_body'], body_hmin=m['body_hmin'], xy=m['xy']))
    tot = {k: sum(r[k] for r in rep.values()) for k in next(iter(rep.values()))}
    res = dict(per_seq=rep, total=tot, near_miss_frames=misses, near_miss_events=ev,
               seqs_with_near_miss=sorted({m['seq'] for m in misses}))
    (work / 'summary.json').write_text(json.dumps(res, indent=1), encoding='utf8')
    print(json.dumps(tot), '\nevents', len(ev), 'seqs', len(res['seqs_with_near_miss']))
    for e in ev:
        print(e)



# ---------------------------------------------------------------- audit renders (map only, no camera images)
def audit(spires, work, n_pick='30'):
    """Render, for sampled near-MISS events, the coloured TLS map from the device pose (front view) and a side view
    of the corridor with the body and cane bands. Seeded sampling of events, one frame (minimum d_body) per event."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    spires, work = Path(spires), Path(work)
    S = json.loads((work / 'summary.json').read_text(encoding='utf8'))
    L = json.loads((work / 'labels.json').read_text(encoding='utf8'))
    ev = S['near_miss_events']; rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(ev), min(int(n_pick), len(ev)), replace=False).tolist())
    frames = []
    for j in pick:
        e = ev[j]; v = L[e['seq']]
        ks = [k for k in range(len(v['t'])) if e['t0'] - 1e-3 <= v['t'][k] <= e['t1'] + 1e-3 and v['rows'][k] and 'd_body' in v['rows'][k]]
        k = min(ks, key=lambda k: v['rows'][k]['d_body']); frames.append((j, e['seq'], k))
    # pose at the picked samples (recompute exactly as in label)
    poses = {}
    for seq in {f[1] for f in frames}:
        t, P, R = resample(*read_tum(spires / 'seq' / seq / 'gt-tum.txt')); poses[seq] = (P, R)
    by_site = {}
    for f in frames:
        by_site.setdefault(site_of(f[1]), []).append(f)
    crops = {}
    for site, fs in by_site.items():
        cen = np.array([poses[f[1]][0][f[2], :2] for f in fs]); acc = {f: [] for f in fs}
        for chunk in read_pcd_rgb(spires / 'map' / f'{site}.pcd'):
            for f, c in zip(fs, cen):
                m = (np.abs(chunk[:, 0] - c[0]) < 6) & (np.abs(chunk[:, 1] - c[1]) < 6)
                if m.any():
                    acc[f].append(chunk[m])
        for f in fs:
            crops[f] = np.concatenate(acc[f]) if acc[f] else np.zeros((0, 6), np.float32)
    rows = []
    out = work / 'audit'; out.mkdir(exist_ok=True)
    for f in frames:
        j, seq, k = f; P, R = poses[seq]; o = P[k]; r = L[seq]['rows'][k]
        g = o[2] - L[seq]['dev_h'][k]
        fwd = R[k, :2, 0] / np.linalg.norm(R[k, :2, 0]); lat = np.array([-fwd[1], fwd[0]])
        Q = crops[f]; d = Q[:, :2] - o[:2]; fw, la, up = d @ fwd, d @ lat, Q[:, 2] - o[2]
        col = np.clip(Q[:, 3:6], 0, 1)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
        m = fw > .15
        x, y = -la[m] / fw[m], -up[m] / fw[m]; vis = (np.abs(x) < 1.2) & (np.abs(y) < .9)
        o_ = np.argsort(-fw[m][vis])
        ax[0].scatter(x[vis][o_], y[vis][o_], c=col[m][vis][o_], s=.4, marker='s', linewidths=0)
        ax[0].set_xlim(-1.2, 1.2); ax[0].set_ylim(.9, -.9); ax[0].set_aspect('equal'); ax[0].set_xticks([]); ax[0].set_yticks([])
        ax[0].set_title(f'#{j} {seq[11:]} t+{L[seq]["t"][k] - L[seq]["t"][0]:.1f}s  front view (map)', fontsize=8)
        hg = Q[:, 2] - g; m2 = (np.abs(la) <= .35) & (fw > -.3) & (fw < 2.6) & (hg > -.2) & (hg < 2.3)
        inb = np.abs(la[m2]) <= .25
        ax[1].scatter(fw[m2][~inb], hg[m2][~inb], c='0.75', s=.6, linewidths=0)
        ax[1].scatter(fw[m2][inb], hg[m2][inb], c=col[m2][inb], s=.8, linewidths=0)
        ax[1].add_patch(plt.Rectangle((.3, .6), 1.7, 1.3, fill=False, ec='r', lw=.8))
        ax[1].add_patch(plt.Rectangle((0, .05), r['d_body'] + .15, .65, fill=False, ec='b', lw=.8))
        ax[1].axvline(r['d_body'], color='r', ls=':', lw=.8)
        ax[1].set_xlim(-.3, 2.6); ax[1].set_ylim(-.2, 2.3); ax[1].set_aspect('equal')
        ax[1].set_title(f'side view |lat|<=0.35 m (colour = within 0.25 m); d={r["d_body"]} hmin={r["body_hmin"]}', fontsize=8)
        fig.tight_layout(); p = out / f'ev{j:02d}.png'; fig.savefig(p, dpi=90); plt.close(fig)
        rows.append(dict(event=j, seq=seq, k=k, t=L[seq]['t'][k], d_body=r['d_body'], body_hmin=r['body_hmin'], png=p.name))
    (out / 'audit_index.json').write_text(json.dumps(rows, indent=1), encoding='utf8')
    print('rendered', len(rows))


def read_pcd_rgb(path, chunk=8_000_000):
    """Yield (x, y, z, r, g, b) float32 chunks; colour from the packed rgb field, else grey from intensity."""
    with open(path, 'rb') as f:
        hdr = {}
        while True:
            k, *v = f.readline().decode('ascii').strip().split()
            hdr[k] = v
            if k == 'DATA':
                break
        off = f.tell()
    names, sizes, types, counts = hdr['FIELDS'], list(map(int, hdr['SIZE'])), hdr['TYPE'], list(map(int, hdr['COUNT']))
    fields = [((n if n != '_' else f'pad{i}'), {'F': 'f', 'U': 'u', 'I': 'i'}[ty] + str(s)) + (((c,),) if c > 1 else ())
              for i, (n, s, ty, c) in enumerate(zip(names, sizes, types, counts))]
    mm = np.memmap(path, dtype=np.dtype(fields), mode='r', offset=off, shape=(int(hdr['POINTS'][0]),))
    for a in range(0, len(mm), chunk):
        b = mm[a:a + chunk]; xyz = np.stack([b['x'], b['y'], b['z']], 1).astype(np.float32)
        if 'rgb' in names:
            u = b['rgb'].view(np.uint32)
            rgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1).astype(np.float32) / 255
        else:
            q = np.full(len(b), .45, np.float32); rgb = np.stack([q, q * 1.1, q * 1.3], 1)  # no colour: flat tint
        yield np.concatenate([xyz, rgb], 1)


if __name__ == '__main__':
    {'voxels': voxels, 'label': label, 'audit': audit, 'summary': lambda *a: summary(a[-1])}[sys.argv[1]](*sys.argv[2:])

"""BlindAssist near-field evaluation kit, minimal version (see EVALKIT.md). Geometry, intake self-check, regions, metrics.

Conventions: camera frame x right, y down, z forward; depth = optical z in metres; `down` = gravity in the camera frame.
Task definition (chosen for the walking user, independent of the App's camera-relative boxes):
  corridor   lateral +-0.35 m (half shoulder width 0.25 m + 0.10 m margin), forward 0.30-2.50 m along the horizontal heading
  bands      height ABOVE GROUND: LOW 0.15-0.50 m (tripping), BODY 0.50-1.30 m, HEAD 1.30-2.00 m (overhanging)
  occupied   obstacle surface area in the band >= 0.005 m^2 (about 5 x 10 cm), distance-invariant, counted after a 3x3
             morphological opening of the band mask (drops isolated stereo 'flying pixels'), same for truth and predictions
  empty      no reference point at all in the band BEFORE the opening (strict); flying-pixel-only frames are ambiguous
Version history: v0 (LOW from 0.10 m, no opening); v1 2026-11-17 (LOW from 0.15 m, opening, strict empty);
v1.1 2026-11-17 (method comparisons on common output frames via `compare`, no-output frames reported) - see EVALKIT.md.
The ground plane and gravity axes used for evaluation come from the reference depth (RANSAC, normal within 15 deg of the
device gravity when available). Reference grade and the independence unit (`block`) travel with every frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

PROTOCOL_VERSION = 'v1.1'

X_HALF, Z_NEAR, Z_FAR = .35, .30, 2.50
BANDS = {'LOW': (.15, .50), 'BODY': (.50, 1.30), 'HEAD': (1.30, 2.00)}
MIN_AREA = .005
DIST_BINS = ((.30, 1.0), (1.0, 2.0), (2.0, 2.5))
NEAR = (.5, 3.)
REF_GRADES = ('ground_truth', 'pseudo_ground_truth', 'sparse_geometry')


@dataclass
class Frame:
    dataset: str
    seq: str
    index: int
    t: float                       # seconds from sequence start
    image: np.ndarray              # HxWx3 uint8 BGR (grayscale replicated if mono)
    K: np.ndarray                  # 3x3 pinhole intrinsics shared by image and ref_depth
    ref_depth: np.ndarray          # HxW optical z (m), NaN where invalid
    ref_grade: str                 # one of REF_GRADES
    down: np.ndarray | None        # gravity in the camera frame from the DEVICE (IMU/VIO); None if unavailable
    eye_height: float | None       # known sensor height above ground (m), sequence-level
    mount: str                     # 'head' or 'chest'
    block: str = ''                # independence / aggregation unit
    extra: dict = field(default_factory=dict)


def rays(K, h, w):
    vv, uu = np.mgrid[0:h, 0:w]
    return np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((h, w))], -1).reshape(-1, 3)


def pixel_area(K, h, w, z):
    """Surface area (m^2) a pixel covers at optical depth z, for a surface facing the camera: z^2 / (fx fy |d|)."""
    d = rays(K, h, w)
    return z ** 2 / (K[0, 0] * K[1, 1] * np.linalg.norm(d, axis=1))


def ground_plane(pts, down, rng, tries=300, tol=.03, min_inliers=2000, max_deg=15.):
    """RANSAC plane below the camera with normal within max_deg of `down`; returns (up normal, sensor height) or None."""
    below = pts[(pts @ down) > .5]
    if len(below) < min_inliers:
        return None
    sel = below[rng.choice(len(below), min(30000, len(below)), replace=False)]
    best = None
    for _ in range(tries):
        p = sel[rng.choice(len(sel), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0]); nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        n = -n if n @ down > 0 else n
        if np.degrees(np.arccos(np.clip(n @ -down, -1, 1))) > max_deg:
            continue
        d = -n @ p[0]
        if d <= .3:
            continue
        inl = np.abs(sel @ n + d) < tol
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), n, float(d))
    if best is None:
        return None
    _, n, d = best
    return (n, d) if (np.abs(below @ n + d) < tol).sum() >= min_inliers else None


def _largest_plane(pts, down, rng, tries=300, tol=.03):
    below = pts[(pts @ down) > .5]
    if len(below) < 2000:
        return None
    sel = below[rng.choice(len(below), min(20000, len(below)), replace=False)]
    best = None
    for _ in range(tries):
        q = sel[rng.choice(len(sel), 3, replace=False)]
        n = np.cross(q[1] - q[0], q[2] - q[0]); nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn; n = -n if n @ down > 0 else n; d = -n @ q[0]
        inl = np.abs(sel @ n + d) < tol
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), n, d)
    return best[0] / len(sel), float(np.degrees(np.arccos(np.clip(best[1] @ -down, -1, 1)))), float(best[2])


def self_check(frames, rng, height_range=None):
    """Intake checks run BEFORE any method on a sample of frames: units, depth meaning (optical z vs range), ground
    planarity, sensor height for the declared mount, and device-gravity consistency."""
    height_range = height_range or {'head': (1.2, 1.9), 'chest': (.9, 1.6)}
    med, q_z, q_r, hts, ang = [], [], [], [], []
    for f in frames:
        z = f.ref_depth.reshape(-1); ok = np.isfinite(z) & (z > .3) & (z < 10)
        if ok.sum() < 2000:
            continue
        med.append(float(np.median(z[ok])))
        d = rays(f.K, *f.ref_depth.shape); dn = np.linalg.norm(d, axis=1)
        down = f.down if f.down is not None else np.array([0, 1., 0])
        a = _largest_plane(d[ok] * z[ok, None], down, rng)
        b = _largest_plane(d[ok] * (z[ok] / dn[ok])[:, None], down, rng)    # as if the file held range instead
        if a and b:
            q_z.append(a[0]); q_r.append(b[0])
            hts.append((f.extra['ground'][1] if 'ground' in f.extra else a[2], f.mount))    # exact floor when the dataset has one
            if f.down is not None:
                ang.append(a[1])
    mounts = sorted({m for _, m in hts})
    in_range = [height_range[m][0] <= h <= height_range[m][1] for h, m in hts]
    # v1 tooling fix: per-mount MEDIAN height must lie in range (walkers may pass raised surfaces, e.g. Aria walk 210-270 s
    # sees a level plane ~1.15 m below the eye); still catches the v0 SANPO intrinsics bug (head median ~1.16 m)
    mount_median = {m: float(np.median([h for h, mm in hts if mm == m])) for m in mounts}
    res = dict(frames=len(q_z), mounts=mounts, sensor_height_in_range_share=float(np.mean(in_range)) if in_range else None,
               sensor_height_median_by_mount=mount_median, depth_median_m=float(np.median(med)) if med else None,
               plane_share_as_optical_z=float(np.median(q_z)) if q_z else None,
               plane_share_if_range=float(np.median(q_r)) if q_r else None,
               sensor_height_m=float(np.median([h for h, _ in hts])) if hts else None,
               device_gravity_vs_ground_deg=float(np.median(ang)) if ang else None)
    checks = dict(units=res['depth_median_m'] is not None and .5 <= res['depth_median_m'] <= 50,
                  meaning=bool(q_z) and res['plane_share_as_optical_z'] >= res['plane_share_if_range'] - .02,
                  planar=bool(q_z) and res['plane_share_as_optical_z'] >= .4,
                  height=bool(mount_median) and all(height_range[m][0] <= v <= height_range[m][1] for m, v in mount_median.items()),
                  gravity=res['device_gravity_vs_ground_deg'] is None or res['device_gravity_vs_ground_deg'] <= 5)
    res['checks'] = checks; res['pass'] = all(checks.values())
    return res


def _open(mask, h, w):
    """3x3 morphological opening of a flat pixel mask (removes isolated flying pixels)."""
    return cv2.morphologyEx(mask.reshape(h, w).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).reshape(-1).astype(bool)


def _corridor(pts, ok, axes, h0):
    dw, fwd, right = axes
    above = h0 - pts @ dw; along = pts @ fwd
    planar = ok & (np.abs(pts @ right) <= X_HALF) & (along >= Z_NEAR) & (along <= Z_FAR)
    return above, along, planar


def regions(f, rng):
    """Evaluator-side regions and truth from the reference depth: near / near_nonground masks, corridor masks per band,
    truth occupancy per band (1 occupied, 0 empty, -1 ambiguous) and the nearest occupied distance."""
    h, w = f.ref_depth.shape
    d = rays(f.K, h, w); z = f.ref_depth.reshape(-1); valid = np.isfinite(z) & (z > .3) & (z < 10)
    pts = d * np.where(valid, z, 0)[:, None]
    # device gravity: normal within 15 deg; without it, the camera y axis is only a loose prior (35 deg, camera pitch)
    if 'ground' in f.extra:          # dataset with an exact floor (ADT: scene floor + GT pose), DIRECTION_A_INDOOR_ADT_DEV_20261118
        gp = f.extra['ground']
    else:
        gp = (ground_plane(pts[valid], f.down, rng) if f.down is not None else ground_plane(pts[valid], np.array([0, 1., 0]), rng, max_deg=35.))
    out = dict(valid=valid, near=valid & (z >= NEAR[0]) & (z <= NEAR[1]), ground=gp)
    if gp is None:
        return out
    n, h0 = gp; dw = -n
    fwd = np.array([0, 0, 1.]) - dw * dw[2]; fwd /= np.linalg.norm(fwd)
    out['axes'] = (dw, fwd, np.cross(dw, fwd))
    above, along, planar = _corridor(pts, valid, out['axes'], h0)
    out['near_nonground'] = out['near'] & (above > BANDS['LOW'][0])
    area = pixel_area(f.K, h, w, np.where(valid, z, 0))
    for b, (lo, hi) in BANDS.items():
        raw = planar & (above >= lo) & (above < hi); m = _open(raw, h, w)
        out[f'corridor_{b}'] = m
        a = float(area[m].sum())
        out[f'truth_{b}'] = 1 if a >= MIN_AREA else (0 if not raw.any() else -1)          # strict empty (v1)
        out[f'truth_opened_{b}'] = 1 if a >= MIN_AREA else (0 if a == 0 else -1)         # empty after opening (reported alongside)
        out[f'truth_dist_{b}'] = float(along[m].min()) if m.any() else None
    return out


def band_alert(pred, f, reg):
    """Occupancy alert per band from a predicted optical-z map, on the evaluator's axes and ground height."""
    if pred is None or 'axes' not in reg:
        return {b: None for b in BANDS}
    h, w = f.ref_depth.shape
    d = rays(f.K, h, w); p = pred.reshape(-1); ok = np.isfinite(p) & (p > .05)
    pts = d * np.where(ok, p, 0)[:, None]
    above, _, planar = _corridor(pts, ok, reg['axes'], reg['ground'][1])
    area = pixel_area(f.K, h, w, np.where(ok, p, 0))
    return {b: bool(area[_open(planar & (above >= lo) & (above < hi), h, w)].sum() >= MIN_AREA) for b, (lo, hi) in BANDS.items()}


def depth_metrics(pred, ref, mask, min_px=200):
    if pred is None or mask.sum() < min_px:
        return None
    p, g = pred.reshape(-1)[mask], ref.reshape(-1)[mask]
    ok = np.isfinite(p) & (p > 0)
    if ok.mean() < .5:
        return dict(absrel=None, delta1=0., valid=float(ok.mean()))
    r = np.where(ok, p / g, np.inf)
    return dict(absrel=float(np.median(np.abs(np.where(ok, p, 1e9) - g) / g)),
                delta1=float(np.mean(np.maximum(r, 1 / r) < 1.25)), valid=float(ok.mean()))


def evaluable(r):
    """Frame on which the evaluator had a ground plane, i.e. a method could be scored. Rows written from v1.1 carry the
    flag; for older rows, a frame counts as evaluable when any method produced an alert there."""
    return r['evaluable'] if 'evaluable' in r else any(v.get('LOW') is not None for v in r['alert'].values())


def has_output(r, method):
    return r['alert'][method].get('LOW') is not None


def summarize(rows, method, keep=None):
    """Block-median depth metrics per region; per band frame-level recall (also by distance bin), frame FA rate and
    FA episodes (consecutive alerted truth-negative frames merged) per minute. `keep(row)` restricts the frames (used by
    `compare` for common output frames). `no_output` = evaluable frames on which the method returned nothing; those frames
    drop out of this method's own denominators, so a comparison between methods must use `compare`."""
    no_output = sum(1 for r in rows if evaluable(r) and not has_output(r, method))
    if keep is not None:
        rows = [r for r in rows if keep(r)]
    out = {}
    blocks = sorted({r['block'] for r in rows})
    for region in ['near', 'near_nonground'] + [f'corridor_{b}' for b in BANDS]:
        per = []
        for b in blocks:
            ms = [r['depth'][method].get(region) for r in rows if r['block'] == b]
            v = [m['absrel'] for m in ms if m and m['absrel'] is not None]
            if v:
                per.append((float(np.median(v)), float(np.median([m['delta1'] for m in ms if m])), len(v)))
        out[region] = dict(blocks=len(per), frames=sum(p[2] for p in per),
                           absrel=float(np.median([p[0] for p in per])) if per else None,
                           delta1=float(np.median([p[1] for p in per])) if per else None)
    for band in BANDS:
        fpo = tno = 0
        for r in rows:
            to, a = r.get('truth_opened', {}).get(band), r['alert'][method].get(band)
            if a is not None and to == 0:
                fpo += a; tno += (not a)
        tp = fn = fp = tn = 0; eps = 0; minutes = 0.; bins = {f'{lo}-{hi}': [0, 0] for lo, hi in DIST_BINS}
        for seq in sorted({r['seq'] for r in rows}):
            rs = sorted((r for r in rows if r['seq'] == seq), key=lambda r: r['t']); prev = False
            for i, r in enumerate(rs):
                if i:
                    minutes += min(rs[i]['t'] - rs[i - 1]['t'], 2.) / 60
                t, a = r['truth'].get(band), r['alert'][method].get(band)
                if a is None or t is None or t < 0:
                    prev = False; continue
                if t == 1:
                    tp += a; fn += (not a); prev = False
                    dist = r['truth_dist'].get(band)
                    for lo, hi in DIST_BINS:
                        if dist is not None and lo <= dist < hi + (1e-9 if hi == Z_FAR else 0):
                            bins[f'{lo}-{hi}'][0] += a; bins[f'{lo}-{hi}'][1] += 1
                else:
                    fp += a; tn += (not a)
                    eps += a and not prev; prev = a
        out[f'detect_{band}'] = dict(positives=tp + fn, negatives=fp + tn, recall=tp / max(tp + fn, 1),
                                     recall_by_distance={k: (v[0] / v[1] if v[1] else None, v[1]) for k, v in bins.items()},
                                     fa_frame_rate=fp / max(fp + tn, 1), fa_episodes=int(eps),
                                     fa_episodes_per_min=eps / max(minutes, 1e-9), minutes=minutes,
                                     negatives_opened=fpo + tno, fa_frame_rate_opened=fpo / max(fpo + tno, 1))
    out['no_output'] = no_output; out['frames'] = len(rows)
    return out


def _binom_two_sided(k, n):
    """Exact two-sided sign test (McNemar on discordant pairs): P(|X - n/2| >= |k - n/2|), X ~ Bin(n, 1/2)."""
    from math import comb
    if n == 0:
        return 1.
    d = abs(k - n / 2)
    return min(1., sum(comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= d - 1e-12) / 2 ** n)


def compare(rows, a, b):
    """Paired comparison of method b against method a (v1.1). Detection: per band, only frames where BOTH have output;
    reports both confusion counts, discordant pairs on positives (a hit / b miss and the reverse) and on negatives, with
    the exact sign-test p-value. Depth: per region, block medians over frames where both AbsRel values exist.
    Also reports each method's no-output frames and the system view where no output counts as no alert."""
    both = lambda r: has_output(r, a) and has_output(r, b)
    sa, sb = summarize(rows, a, both), summarize(rows, b, both)
    out = dict(methods=(a, b), common_frames=sa['frames'], no_output={a: sa['no_output'], b: sb['no_output']}, detect={}, depth={})
    for band in BANDS:
        pos = [r for r in rows if both(r) and r['truth'].get(band) == 1]
        neg = [r for r in rows if both(r) and r['truth'].get(band) == 0]
        hit_a = sum(1 for r in pos if r['alert'][a][band] and not r['alert'][b][band])
        hit_b = sum(1 for r in pos if r['alert'][b][band] and not r['alert'][a][band])
        fa_a = sum(1 for r in neg if r['alert'][a][band] and not r['alert'][b][band])
        fa_b = sum(1 for r in neg if r['alert'][b][band] and not r['alert'][a][band])
        sys_view = {}
        for m in (a, b):
            ev = [r for r in rows if evaluable(r) and r['truth'].get(band) in (0, 1)]
            sys_view[m] = dict(tp=sum(1 for r in ev if r['truth'][band] == 1 and r['alert'][m].get(band)),
                               positives=sum(1 for r in ev if r['truth'][band] == 1),
                               fp=sum(1 for r in ev if r['truth'][band] == 0 and r['alert'][m].get(band)),
                               negatives=sum(1 for r in ev if r['truth'][band] == 0))
        out['detect'][band] = dict(
            positives=len(pos), negatives=len(neg),
            tp={a: sum(1 for r in pos if r['alert'][a][band]), b: sum(1 for r in pos if r['alert'][b][band])},
            fp={a: sum(1 for r in neg if r['alert'][a][band]), b: sum(1 for r in neg if r['alert'][b][band])},
            recall={a: sa[f'detect_{band}']['recall'], b: sb[f'detect_{band}']['recall']},
            fa_frame_rate={a: sa[f'detect_{band}']['fa_frame_rate'], b: sb[f'detect_{band}']['fa_frame_rate']},
            fa_episodes_per_min={a: sa[f'detect_{band}']['fa_episodes_per_min'], b: sb[f'detect_{band}']['fa_episodes_per_min']},
            discordant_pos={'only_' + a: hit_a, 'only_' + b: hit_b, 'p': _binom_two_sided(hit_b, hit_a + hit_b)},
            discordant_neg={'only_' + a: fa_a, 'only_' + b: fa_b, 'p': _binom_two_sided(fa_b, fa_a + fa_b)},
            no_output_as_no_alert=sys_view)
    for region in ['near', 'near_nonground'] + [f'corridor_{b}' for b in BANDS] + ['nng_in_fov', 'nng_out_fov']:
        per_a, per_b = [], []
        for blk in sorted({r['block'] for r in rows}):
            va, vb = [], []
            for r in rows:
                if r['block'] != blk:
                    continue
                ma, mb = r['depth'][a].get(region), r['depth'][b].get(region)
                if ma and mb and ma['absrel'] is not None and mb['absrel'] is not None:
                    va.append(ma['absrel']); vb.append(mb['absrel'])
            if va:
                per_a.append(float(np.median(va))); per_b.append(float(np.median(vb)))
        if per_a:
            out['depth'][region] = dict(blocks=len(per_a), absrel={a: float(np.median(per_a)), b: float(np.median(per_b))},
                                        blocks_b_better=int(sum(y < x for x, y in zip(per_a, per_b))))
    return out


def min_resolvable_frames(tolerance, positives):
    """A recall tolerance expressed in frames. Rules should state tolerances in frames (or discordant pairs) and treat a
    tolerance below 3 frames as unresolvable on that data (v1.1 lesson from the A1 SANPO recheck: 0.02 of 58 = 1.2 frames)."""
    return tolerance * positives

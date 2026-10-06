"""Real-head confirmation batch (CNH_RHC_20261007): see CNH_REAL_HEAD_CONFIRM_PROTOCOL_20261007.md.

Fresh units 400000-400479 (calibration 400000-400095, evaluation 400096-400479), 40 configs each.
Arms per sequence (one render each): SYN = original scripted motion; NAT = sensor follows a real
HEADS-UP unconstrained head window (yaw minus recording median offset, horizontal position,
speed-normalised to 0.16 m/frame, pitch -10 deg); ALN = same real path with head yaw = travel.
Queries: SYN{exact}; NAT{exact, E1, E1+k, E1-k, EMA}; ALN{exact, E1}. Frozen M3 / smoothing /
alarm threshold / r3 gate / tau grid / event and control definitions.
Stages: windows | smoke (dev unit 98000 only) | freeze | run | analyze.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_active_scan_dev as A  # noqa: E402
import cnh_heads_up_heading as H  # noqa: E402

OUT = A.ROOT/'artifacts.local/work/cnh-real-head-confirm-20261007'
CAL = list(range(400000, 400096))
EVAL = list(range(400096, 400480))
SEED = 2026100795
K = 8.060857413547966  # union-clear half-width: dev unconstrained E1 RMS (PLAN_STEP2 k_deg.u1)
STEP = .16
QUERIES = dict(SYN=('exact',), NAT=('exact', 'E1', 'E1p', 'E1m', 'EMA'), ALN=('exact', 'E1'))


def wrap(a):
    return (np.asarray(a)+180.) % 360.-180.


# ---------------------------------------------------------------- real head windows
def windows():
    """Window bank from HEADS-UP unconstrained walking: 16 samples at 5 Hz (every 6 frames of 30 Hz)."""
    p, fwd = H.load('unconstrained'); d = H.series('unconstrained'); t, ok = d['t'], d['ok']
    head = np.degrees(np.arctan2(fwd[:, 1], fwd[:, 0]))
    offset = wrap(head[t]-d['T'])[ok]; bias = float(np.median(offset))
    ps = np.stack([np.convolve(p[:, k], np.ones(30)/30, 'same') for k in (0, 1)], 1)
    c = np.r_[0, np.cumsum(ok)]; starts = np.flatnonzero(c[96:]-c[:-96] == 96)
    split = int(np.median(t[starts]))
    rows = []
    for i in starts:
        f = t[i]+6*np.arange(16)
        rows.append(np.concatenate([[t[i]], head[f]-bias, d['T'][i+6*np.arange(16)], p[f, 0], p[f, 1], ps[f, 0], ps[f, 1]]))
    W = np.array(rows)
    half = np.where(W[:, 0]+95 < split, 0, np.where(W[:, 0] >= split, 1, -1))
    return dict(W=W, half=half, bias=bias, split=split)


def nat_poses(win, sign, aligned=False):
    """Window row -> sensor[16], travel[16], exact query[16] (sim frame: y up, z forward, x lateral)."""
    import cnh_cvr_pilot as CP
    hy, ty = win[1:17], win[17:33]
    hp = np.stack([win[33:49], win[49:65]], 1); sp = np.stack([win[65:81], win[81:97]], 1)
    phi = np.deg2rad(ty[-1]); u = np.array([np.cos(phi), np.sin(phi)]); l = np.array([-np.sin(phi), np.cos(phi)])
    scale = STEP/np.mean(np.linalg.norm(np.diff(sp, axis=0), axis=1))
    def to_sim(xy):
        r = (xy-sp[-1])*scale
        return np.stack([sign*(r@l), np.zeros(16), r@u], 1)
    th_t = sign*wrap(ty-ty[-1]); th_h = th_t if aligned else sign*wrap(hy-ty[-1])
    hpos, tpos = to_sim(hp), to_sim(sp)
    sensor = np.repeat(np.eye(4)[None], 16, 0); travel = sensor.copy(); query = sensor.copy()
    for f in range(16):
        sensor[f, :3, :3] = CP.rotation(th_h[f], 'y')@CP.rotation(-10, 'x'); sensor[f, :3, 3] = hpos[f]
        travel[f, :3, :3] = CP.rotation(th_t[f], 'y'); travel[f, :3, 3] = tpos[f]
        query[f, :3, :3] = CP.rotation(wrap(th_h[f]-th_t[f]), 'y')@CP.rotation(-10, 'x')
    return sensor, travel, query


def noisy_for(sensor, unit, config):
    from cnh_track_a_readout import noisy_poses
    seed = int(np.random.SeedSequence([A.NOISE_PREFIX, unit, config, 1]).generate_state(1)[0])
    return noisy_poses(sensor, seed, dt=.2)


def rel_query(rel):
    import cnh_cvr_pilot as CP
    q = np.eye(4); q[:3, :3] = CP.rotation(float(rel), 'y')@CP.rotation(-10., 'x')
    return q


def e1_rel(noisy):
    """Existing deployable estimate (cnh_query_direction_dev.est_query): noisy yaw minus 1 s displacement."""
    out = np.zeros(16)
    for f in range(16):
        a, b = noisy[max(0, f-5), [0, 2], 3], noisy[f, [0, 2], 3]; d = b-a
        out[f] = wrap(A_yaw(noisy[f])-np.degrees(np.arctan2(d[0], d[1])))
    return out


def A_yaw(R):
    fw = R[:3, 2]
    return np.degrees(np.arctan2(fw[0], fw[2]))


def ema_rel(noisy):
    """Adaptive EMA of noisy horizontal velocity at 5 Hz: tau 0.5 s, 0.25 s when its direction
    changed >= 10 deg/s over the last 0.4 s. Frame 0 falls back to E1 convention (zero history)."""
    P = noisy[:, [0, 2], 3]; out = np.zeros(16); m = None; dirs = np.full(16, np.nan); tau = .5
    for f in range(1, 16):
        v = P[f]-P[f-1]
        a = 1-np.exp(-.2/tau)
        m = v.copy() if m is None else (1-a)*m+a*v
        dirs[f] = np.degrees(np.arctan2(m[0], m[1]))
        if f >= 3:
            tau = .25 if abs(wrap(dirs[f]-dirs[f-2]))/.4 >= 10. else .5
    for f in range(16):
        out[f] = wrap(A_yaw(noisy[f])-(dirs[f] if np.isfinite(dirs[f]) else 0.))
    return out


# ---------------------------------------------------------------- per-unit computation
def draw(bank, unit, config):
    rng = np.random.default_rng([SEED, unit, config])
    idx = np.flatnonzero(bank['half'] == unit % 2)
    return int(idx[rng.integers(len(idx))]), (1. if rng.random() < .5 else -1.)


def geometry(boxes, travel):
    import cnh_extrinsic_aug_evaluate as EV
    rows = EV.geometry_scene(boxes, travel)
    return dict(category=np.array([r['ref_category'] for r in rows]), clear_all=np.array([r['clear_all'] for r in rows]),
                covered=np.array([r['covered'] for r in rows]), fraction=float(rows[0]['reference_fraction']),
                censor=str(rows[0]['censor_reason']))


def run_unit(rn, bank, unit, folder, deadline=None):
    import cnh_margin_confirm as MC
    import cnh_extrinsic_aug_data as D
    dest = folder/f'unit{unit}.npz'
    if dest.exists():
        return
    tick = time.monotonic(); flag = A.mirrored(unit); mode = unit % 3
    scenes = MC.scenes_for(unit); C = len(scenes)
    boxes = [D.mirror_boxes(s['boxes'], flag) for s in scenes]
    res = dict(unit=unit, mode=mode, mirror=flag)
    arms = {}
    # SYN: original construction (shared motion within the unit, mirrored by unit)
    head0 = A.scripted_head(mode)
    s_, t_, _, q_ = (A.mirror(x, flag) for x in A.poses_for(unit, 0, head0))
    noisy = np.stack([A.mirror(A.poses_for(unit, c, head0)[2], flag) for c in range(C)])
    arms['SYN'] = dict(sensor=np.repeat(s_[None], C, 0), travel=np.repeat(t_[None], C, 0), noisy=noisy,
                       queries=dict(exact=np.repeat(q_[None], C, 0)))
    # NAT / ALN: per-sequence real window
    picks = [draw(bank, unit, c) for c in range(C)]; res['window'] = np.array([p[0] for p in picks]); res['sign'] = np.array([p[1] for p in picks])
    for arm, aligned in (('NAT', False), ('ALN', True)):
        S, T, Q, N = [], [], [], []
        for c, (i, sg) in enumerate(picks):
            s, t, q = nat_poses(bank['W'][i], sg, aligned); S.append(s); T.append(t); Q.append(q); N.append(noisy_for(s, unit, c))
        S, T, Q, N = map(np.stack, (S, T, Q, N))
        e1 = np.stack([e1_rel(n) for n in N]); qs = dict(exact=Q, E1=np.stack([[rel_query(r) for r in row] for row in e1]))
        res[f'{arm}_e1_err'] = wrap(e1-np.array([[A_yaw(q) for q in row] for row in Q])).astype(np.float32)
        if arm == 'NAT':
            qs['E1p'] = np.stack([[rel_query(r+K) for r in row] for row in e1]); qs['E1m'] = np.stack([[rel_query(r-K) for r in row] for row in e1])
            em = np.stack([ema_rel(n) for n in N]); qs['EMA'] = np.stack([[rel_query(r) for r in row] for row in em])
            res['NAT_ema_err'] = wrap(em-np.array([[A_yaw(q) for q in row] for row in Q])).astype(np.float32)
        arms[arm] = dict(sensor=S, travel=T, noisy=N, queries=qs)
    for arm, a in arms.items():
        if deadline and time.time() > deadline:
            raise TimeoutError('real-head confirm wall budget reached')
        z = np.stack([rn.eng.render_sequence(boxes[c], a['sensor'][c], A.ANGLES, unit, c)[0] for c in range(C)], 1)
        if arm == 'SYN':
            res['SYN_z_sha'] = __import__('hashlib').sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
        for qn in QUERIES[arm]:
            raw = np.empty((3, C, 13, 2), np.float32)
            for s, ang in enumerate(A.ANGLES):
                ex = rn.eng.N.extrinsic(ang)
                raw[s] = rn.raw(z[s], a['noisy']@ex, a['queries'][qn]@ex)
            res[f'{arm}_{qn}_raw'] = raw
        res[f'{arm}_gate'] = np.stack([A.gate_seq(a['noisy'][c]) for c in range(C)])
        if arm != 'ALN':  # ALN shares NAT travel and truth
            g = [geometry(boxes[c], a['travel'][c]) for c in range(C)]
            res[f'{arm}_category'] = np.stack([x['category'] for x in g]); res[f'{arm}_clear_all'] = np.stack([x['clear_all'] for x in g])
            res[f'{arm}_covered'] = np.stack([x['covered'] for x in g]); res[f'{arm}_fraction'] = np.array([x['fraction'] for x in g])
            res[f'{arm}_censor'] = np.array([x['censor'] for x in g])
    res['seconds'] = time.monotonic()-tick
    folder.mkdir(parents=True, exist_ok=True); tmp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **res); os.replace(tmp, dest)
    print('unit', unit, round(res['seconds'], 1), 's', flush=True)


# ---------------------------------------------------------------- stages
def bank_cached():
    path = OUT/'windows.npz'
    if not path.exists():
        b = windows(); OUT.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=b['W'], half=b['half'], bias=b['bias'], split=b['split'])
    with np.load(path) as z:
        return dict(W=z['W'], half=z['half'], bias=float(z['bias']), split=int(z['split']))


def smoke():
    import cnh_heading_uncertainty_dev as HU
    bank = bank_cached(); rn = HU.Runner(); unit = 98000
    folder = OUT/'smoke'; run_unit(rn, bank, unit, folder)
    z = dict(np.load(folder/f'unit{unit}.npz'))
    obs, sc = A.stored(unit)
    out = dict(unit=unit, seconds=float(z['seconds']),
               syn_raw_vs_stored_max_abs=float(np.abs(z['SYN_exact_raw']-sc['reference']).max()),
               windows=int(len(bank['W'])), per_half=[int((bank['half'] == h).sum()) for h in (0, 1)], mount_bias_deg=bank['bias'])
    for arm in ('SYN', 'NAT'):
        cats = z[f'{arm}_category']; cov = z[f'{arm}_covered']
        contact = (cov & np.isin(cats, ['contact0-2cm', 'contact2-5cm', 'contact>5cm'])).any(1)
        out[arm] = dict(contacts=int(contact.sum()), controls=int(z[f'{arm}_clear_all'].all(1).sum()), covered=int(cov[:, 0].sum()))
    for arm in ('NAT', 'ALN'):
        e = z[f'{arm}_e1_err'][:, 3:]; out[f'{arm}_E1_err_rms'] = float(np.sqrt((e**2).mean()))
    e = z['NAT_ema_err'][:, 3:]; out['NAT_EMA_err_rms'] = float(np.sqrt((e**2).mean()))
    for arm in ('SYN', 'NAT', 'ALN'):
        g = z[f'{arm}_gate']; out[f'{arm}_gate_pass_single_dual'] = [float(g[..., 0].mean()), float(g[..., 1].mean())]
    print(json.dumps(out, indent=1))
    (folder/'smoke.json').write_text(json.dumps(out, indent=1)+'\n', encoding='utf8')


def freeze(hours):
    bank = bank_cached(); now = time.time()
    A.save(OUT/'PLAN.json', dict(
        task='CNH_RHC_20261007', lane='CONFIRM (fresh simulation units, frozen protocol)',
        authorization='User "同意" (2026-10-07) to real-head-driven confirmation batch; protocol approved with "ok"',
        protocol='research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_PROTOCOL_20261007.md',
        calibration_units=[CAL[0], CAL[-1]], evaluation_units=[EVAL[0], EVAL[-1]], configs_per_unit=40,
        hashes={p: A.sha(A.ROOT/p) for p in (
            'research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_PROTOCOL_20261007.md',
            'research/active/dtr-r0/nearfield/cnh_real_head_confirm.py',
            'research/active/dtr-r0/nearfield/cnh_real_head_confirm_analyze.py',
            'research/active/dtr-r0/nearfield/cnh_heads_up_heading.py',
            'artifacts.local/work/cnh-heads-up-heading-20261007/raw/unconstrained_camera_poses.csv',
            'artifacts.local/work/cnh-real-head-confirm-20261007/windows.npz')},
        windows=dict(count=int(len(bank['W'])), per_half=[int((bank['half'] == h).sum()) for h in (0, 1)],
                     mount_bias_deg=bank['bias'], split_frame=bank['split']),
        union_clear_k_deg=K, budget_wall_hours=hours, started_unix=now, deadline_unix=now+hours*3600))


def run():
    import cnh_heading_uncertainty_dev as HU
    plan = A.read(OUT/'PLAN.json')
    for p, h in plan['hashes'].items():
        assert A.sha(A.ROOT/p) == h, f'frozen input changed: {p}'
    bank = bank_cached(); rn = HU.Runner()
    for u in CAL+EVAL:
        run_unit(rn, bank, u, OUT/'units', plan['deadline_unix'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['windows', 'smoke', 'freeze', 'run', 'analyze'])
    ap.add_argument('--hours', type=float, default=4.); a = ap.parse_args()
    if a.stage == 'windows':
        b = bank_cached(); print(len(b['W']), [int((b['half'] == h).sum()) for h in (0, 1)], b['bias'], b['split'])
    elif a.stage == 'smoke':
        smoke()
    elif a.stage == 'freeze':
        freeze(a.hours)
    elif a.stage == 'run':
        run()
    else:
        import cnh_real_head_confirm_analyze as AN
        AN.main()


if __name__ == '__main__':
    main()

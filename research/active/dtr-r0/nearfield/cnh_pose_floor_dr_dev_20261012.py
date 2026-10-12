"""Pose/floor-randomized ToF stack: render at varied true pitch and floor depth (EXPLORE).

Real phone captures (hardware-bringup REAL_PHONE_IMU_REPLAY_DEV_20261012) showed false alarms
whenever the floor is within ~3 m: the simulator always rendered the true pitch at -10 deg with
the floor slab 1.60-1.65 m below the sensor, so near floor returns never occurred in training.

Changes versus the robust stack (cnh_robust_stack_dev_20261012):
  * every training episode is re-rendered with true pitch -10+delta (delta~U(-15,15), 20% 0)
    and floor depth U(1.40,1.80) m (20% unchanged); instrument DR draws as the no-bin0 retrain;
  * public transforms use the true pose plus an IMU-like error eps~U(-5,5) (20% exact);
  * per-row nuisance (crosstalk tail/DC) as before;
  * M3 (5 seeds) and ordinary (3 seeds) retrained with the original recipes, bf16 autocast;
  * cal / S-train / pooled negatives / hold re-rendered under a deploy mixture of
    (delta, floor, eps); hold also under fixed conditions; S refit and cuts calibrated as before.
LOCAL is disabled in the strong grade (as in the robust stack) and is not computed when scoring.
Labels are unchanged (body-volume truth does not depend on sensor pitch or floor depth).
Development only; every evaluated set is consumed.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import cnh_no_bin0_dr_train_20261011 as TR
import cnh_nuisance_pitch_dr_m3_dev_20261012 as Q
from cnh_cvr_pilot import rotation

ROOT = TR.ROOT
SRC = TR.OUT
OUT = ROOT / 'artifacts.local/work/cnh-pose-floor-dr-dev-20261012'
BASE = OUT                     # renders (training histories, eval sets) live here
DATA, EVAL = BASE / 'inputs/train', BASE / 'eval'
NOISE = None                   # (low, high) extra iid normalized noise SD, set by --noise
ROBUST = ROOT / 'artifacts.local/work/cnh-robust-stack-dev-20261012'
ROBUST_M3 = ROOT / 'artifacts.local/work/cnh-nuisance-pitch-dr-m3-dev-20261012/models/M3'
POSE_SEED, FEAT_SEED = 2026101231, 2026101232
P_NOMINAL, TRUE_DELTA, FLOOR, EPS, P_EXACT = .2, (-15., 15.), (1.40, 1.80), 5., .2
SEEDS = (2026100955, 2026100956, 2026100957)
BIN_NOISE_SCALE = np.r_[np.ones(14), .75, .75].astype(np.float32)  # real bins 14-15 carry ~0.75x
DEPLOY_DELTA = (-15, -10, -5, 0, 5, 10, 15)
DEPLOY_FLOOR = (1.45, 1.60, 1.75)
DEPLOY_EPS = (-4, 0, 4)
# name -> (true delta, floor or None, public eps, p_clean); None delta = deploy mixture
HOLD_CONDITIONS = {
    'nominal_clean': (0, None, 0, 1.), 'nominal_nuis': (0, None, 0, 0.),
    'down15': (-15, 1.60, 0, 0.), 'down5': (-5, 1.60, 0, 0.), 'up5': (5, 1.60, 0, 0.), 'up15': (15, 1.60, 0, 0.),
    'low_floor_down10': (-10, 1.45, 0, 0.), 'deploy': (None, None, None, .25),
}


def rot(delta):
    r = np.eye(4); r[:3, :3] = rotation(float(delta), 'x'); return r


def set_floor(boxes, y):
    """Move the floor slab (full-width horizontal slab below the sensor) to depth y."""
    out = copy.deepcopy(boxes)
    if y is None:
        return out
    moved = 0
    for b in out:
        if b['lo'][0] <= -7.9 and b['hi'][0] >= 7.9 and b['lo'][2] <= -7.9 and b['lo'][1] > 1.2 and b['hi'][1] - b['lo'][1] < .5:
            thickness = b['hi'][1] - b['lo'][1]
            b['lo'] = [b['lo'][0], float(y), b['lo'][2]]; b['hi'] = [b['hi'][0], float(y) + thickness, b['hi'][2]]
            moved += 1
    if moved != 1:
        raise ValueError(f'Expected exactly one floor slab, found {moved}')
    return out


# ------------------------------------------------------------------ training data
def render_train():
    import cnh_location_reference_gpu as G
    folder = OUT / 'inputs/train'; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'histories.npy').exists():
        raise FileExistsError(folder / 'histories.npy')
    src = SRC / 'inputs/train'
    for name in ('transforms', 'length', 'labels', 'weights', 'mask', 'm3_indices', 'm3_labels'):
        shutil.copyfile(src / (name + '.npy'), folder / (name + '.npy'))
    n = len(np.load(folder / 'length.npy', mmap_mode='r'))
    hist = np.lib.format.open_memmap(folder / 'histories.npy', mode='w+', dtype=np.float16, shape=(n, 8, 8, 8, 16))
    delta = np.zeros(n, np.float32); floor = np.full(n, np.nan, np.float32)
    cursor = 0; began = time.monotonic()
    TR.configure()
    for index, seq in enumerate(TR.sequences()):
        p = TR.draw_parameters(seq['key'])
        rng = np.random.default_rng(np.random.SeedSequence([POSE_SEED, *map(int, seq['key'])]))
        d = 0. if rng.random() < P_NOMINAL else float(rng.uniform(*TRUE_DELTA))
        fl = None if rng.random() < P_NOMINAL else float(rng.uniform(*FLOOR))
        renderer = G.ExpectedRenderer(np.asarray(seq['sensor']) @ rot(d), set_floor(seq['boxes'], fl))
        try:
            raw = TR.raw_expected(renderer); counts = renderer.params.signal_counts
        finally:
            renderer.close()
        seed = int(np.random.SeedSequence([TR.DR_SEED + 1, *seq['key']]).generate_state(1)[0])
        z = TR.sample_response(raw, p, seed, counts)
        for f in TR.FRAMES:
            begin = max(0, f - 7); le = f - begin + 1
            hist[cursor] = 0; hist[cursor, 8 - le:] = z[begin:f + 1]
            delta[cursor] = d; floor[cursor] = np.nan if fl is None else fl
            cursor += 1
        if (index + 1) % 256 == 0:
            print('RENDER_TRAIN', index + 1, cursor, round(time.monotonic() - began, 1), flush=True)
    if cursor != n:
        raise ValueError(f'Row identity differs: {cursor} vs {n}')
    hist.flush()
    np.save(folder / 'true_delta_deg.npy', delta); np.save(folder / 'floor_y.npy', floor)
    receipt = dict(stage='render_train', rows=n, episodes=index + 1, seconds=time.monotonic() - began,
                   pose_seed=POSE_SEED, p_nominal=P_NOMINAL, true_delta_range=TRUE_DELTA, floor_range=FLOOR,
                   code_sha256=TR.sha(Path(__file__)))
    (OUT / 'render_train_receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf8')
    return receipt


def features():
    """One pass: per-row nuisance + public pose (true + eps); ordinary tokens (all rows), M3 voxels (M3 rows)."""
    import cnh_boundary_token_model as M
    from cnh_cvr_projection import Projector
    data = DATA
    for name in ('ordinary.npy', 'm3.npy'):
        if (OUT / name).exists():
            raise FileExistsError(OUT / name)
    z = np.load(data / 'histories.npy', mmap_mode='r'); t = np.load(data / 'transforms.npy', mmap_mode='r')
    le = np.load(data / 'length.npy', mmap_mode='r'); ids = np.load(data / 'm3_indices.npy')
    delta = np.load(data / 'true_delta_deg.npy')
    position = np.full(len(z), -1, np.int64); position[ids] = np.arange(len(ids))
    rng = np.random.default_rng(FEAT_SEED)
    ordinary = np.lib.format.open_memmap(OUT / 'ordinary.npy', mode='w+', dtype=np.float16, shape=(len(z), 2, 16, 9, 8, 8))
    vox = np.lib.format.open_memmap(OUT / 'm3.npy', mode='w+', dtype=np.float16, shape=(len(ids), 3, 24, 17, 33))
    projector = Projector(); began = time.monotonic(); eps_all = np.zeros(len(z), np.float32)
    with torch.no_grad():
        for begin in range(0, len(z), 64):
            end = min(begin + 64, len(z)); n = end - begin
            zz = np.array(z[begin:end]).astype(np.float32); ll = np.array(le[begin:end])
            add, _ = Q.nuisance(rng, n)
            valid = np.arange(8)[None, :] >= (8 - ll)[:, None]
            zz += add[:, None] * valid[:, :, None, None, None]
            if NOISE:
                # Training renders carry ~0.1 normalized noise; real and nominal eval CNH carry ~1.
                sd = rng.uniform(*NOISE, n)[:, None, None, None, None] * BIN_NOISE_SCALE
                zz += rng.standard_normal(zz.shape).astype(np.float32) * sd * valid[:, :, None, None, None]
            zz[..., 0] = 0
            eps = np.where(rng.random(n) < P_EXACT, 0., rng.uniform(-EPS, EPS, n)); eps_all[begin:end] = eps
            mats = np.stack([rot(a) for a in delta[begin:end] + eps])
            tt = np.array(t[begin:end]) @ mats[:, None]
            zt = torch.as_tensor(zz, device='cuda').half().float()
            tg = torch.as_tensor(tt, device='cuda', dtype=torch.float32); lt = torch.as_tensor(ll, device='cuda')
            geometry = M.feature_geometry(tg, lt, 'center')
            ordinary[begin:end] = M.build_features(zt, geometry, lt)[:, :, :, list(TR.CHANNELS)].half().cpu().numpy()
            sel = np.flatnonzero(position[begin:end] >= 0)
            if len(sel):
                v = Q.fast_project(zz[sel].astype(np.float16), tt[sel], ll[sel], projector)
                vox[position[begin + sel]] = v.half().cpu().numpy()
            if begin % (64 * 128) == 0:
                print('FEATURES', begin, '/', len(z), round(time.monotonic() - began, 1), flush=True)
    ordinary.flush(); vox.flush(); np.save(OUT / 'public_eps_deg.npy', eps_all)
    receipt = dict(stage='features', rows=int(len(z)), m3_rows=int(len(ids)), seconds=time.monotonic() - began, seed=FEAT_SEED,
                   eps_range=EPS, p_exact=P_EXACT, nuisance='as cnh_nuisance_pitch_dr_m3_dev_20261012', code_sha256=TR.sha(Path(__file__)))
    (OUT / 'features_receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf8')
    return receipt


def train_m3(seeds=(0, 1, 2, 3, 4)):
    import cnh_nuisance_dr_m3_dev_20261011 as N
    N.OUT = OUT
    N.train(list(seeds), amp=True)


def train_ordinary():
    import cnh_robust_stack_dev_20261012 as RS
    RS.OUT = OUT
    RS.train_ordinary()


# ------------------------------------------------------------------ evaluation sets
def _ev():
    import cnh_no_bin0_dr_eval_20261011 as EV
    return EV


def eval_sets():
    EV = _ev()
    sets = {'cal': (EV.F.OLD / 'data/cal', None), 'strain': (EV.OUT.parent / 'cnh-task-cost-retrain-dev-20261010/data/train', None)}
    for dataset in EV.POOL:
        rows = EV.C.read(EV.OUT.parent / dataset / 'scene_rows.json')
        for split in ('cal', 'hold'):
            ids = np.array([i for i, r in enumerate(rows[split]) if r['placement'] in ('pass', 'clear')])
            sets[f'pool_{dataset}_{split}'] = (EV.OUT.parent / dataset / 'data' / split, ids)
    return sets


def _variants(n, rng, condition):
    delta, floor, eps, _ = condition
    if delta is None:
        return (rng.choice(DEPLOY_DELTA, n).astype(float), rng.choice(DEPLOY_FLOOR, n).astype(float), rng.choice(DEPLOY_EPS, n).astype(float))
    return np.full(n, float(delta)), np.full(n, np.nan if floor is None else float(floor)), np.full(n, float(eps))


def render_eval(name, folder, ids, condition, seed):
    """Render the source rows under per-scene (true delta, floor); store hist/ambient/geometry."""
    EV = _ev(); C, D = EV.C, EV.D
    import cnh_task_cost_source_20261010 as TS
    target = EVAL / name
    if (target / 'geometry.npz').exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    rows = C.read(Path(folder).parent.parent / 'scene_rows.json')[Path(folder).name]
    with np.load(Path(folder) / 'geometry.npz') as g:
        geom = {k: g[k] for k in g.keys()}
    k_rep = np.load(Path(folder) / 'hist.npy', mmap_mode='r').shape[1]
    if ids is None:
        ids = np.arange(len(rows))
    rows = [rows[i] for i in ids]; n = len(rows)
    rng = np.random.default_rng(seed)
    delta, floor, eps = _variants(n, rng, condition)
    _, R, Renderer, _ = D.frozen_imports()
    base = geom['sensor']
    hist = np.lib.format.open_memmap(target / 'hist.npy', mode='w+', dtype=np.int32, shape=(n, k_rep, 16, 8, 8, 16))
    ambient = np.zeros((n, k_rep, 16, 8, 8), np.float32)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row['background_id'], delta[i], floor[i])].append(i)
    began = time.monotonic()
    for (bg, d, fl), members in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], -1 if np.isnan(kv[0][2]) else kv[0][2])):
        engine = Renderer.ExpectedRenderer(base @ rot(d), set_floor(rows[members[0]]['background_boxes'], None if np.isnan(fl) else fl))
        try:
            amb = engine.ambient.copy()
            for start, endpoints in engine.iter_render([rows[i]['target_box'] for i in members], candidate_batch=4, pose_batch=16):
                for j in range(len(endpoints)):
                    i = members[start + j]; row = rows[i]
                    mean = endpoints[j, 0] + row['rho'] / Renderer.ENDPOINT_RHO * (endpoints[j, 1] - endpoints[j, 0])
                    for k in range(k_rep):
                        s = int(np.random.SeedSequence([seed, int(ids[i]), k]).generate_state(1)[0])
                        hist[i, k] = R.sample(mean, amb, s)[0]
                        ambient[i, k] = amb
        finally:
            engine.close()
    hist.flush(); np.save(target / 'ambient.npy', ambient)
    out = {key: (value[ids] if value.ndim and len(value) == len(geom['category']) else value) for key, value in geom.items()}
    if 'target_support' in geom:   # visibility depends on the true pitch
        support = np.zeros_like(out['target_support']); visible = np.zeros_like(out['target_visible_now']); positive = np.zeros_like(out['label_positive'])
        for d in np.unique(delta):
            sel = np.flatnonzero(delta == d)
            s, v, p = TS.target_support([rows[i] for i in sel], base @ rot(d))
            support[sel], visible[sel], positive[sel] = s, v, p
        out.update(target_support=support, target_visible_now=visible, label_positive=positive)
    np.savez_compressed(target / 'geometry.npz', **out, true_delta=delta, floor_y=floor, public_eps=eps, source_ids=ids)
    C.save(target / 'scene_rows.json', rows)
    print('RENDER_EVAL', name, n, len(groups), 'groups', round(time.monotonic() - began, 1), flush=True)
    return target


def _no_local():
    import cnh_bar_local_readout_dev as L
    L.patches = lambda: ([], [])
    L.scan = lambda feature, qs, norms, locations: (np.zeros((len(feature), 2)),)


def score(name, stack, p_clean, seed, out_name=None):
    """Score a rendered eval folder; nuisance per scene-replica; public pose = true + eps (grouped)."""
    EV = _ev()
    target = OUT / 'scores' / f'{out_name or name}_{stack}.npz'
    if target.exists():
        return dict(np.load(target))
    folder = EVAL / name
    _no_local()
    z = np.asarray(EV.normalized(folder)).astype(np.float32)
    n, k = z.shape[:2]
    with np.load(folder / 'geometry.npz') as g:
        sensor, query, delta, eps = g['sensor'], g['public_query'], g['true_delta'], g['public_eps']
    rng = np.random.default_rng(seed)
    if p_clean < 1:
        import cnh_robust_stack_dev_20261012 as RS
        z = z + RS._nuisance_scene(rng, n, k, p_clean)
    z = z.astype(np.float16)
    public = delta + eps
    if stack == 'new':
        m3 = sorted((OUT / 'models/M3').glob('model_seed*.pt')); ordinary = [OUT / 'models' / f'ordinary_seed{s}.pt' for s in SEEDS]
    else:
        m3 = sorted(ROBUST_M3.glob('model_seed*.pt')); ordinary = [ROBUST / 'models' / f'ordinary_seed{s}.pt' for s in SEEDS]
    original = EV.paths; EV.paths = lambda arm: (list(m3), list(ordinary))
    result = None
    try:
        for value in np.unique(public):
            sel = np.flatnonzero(public == value); r = rot(value)
            part = EV.infer(z[sel], sensor @ r, query @ r, arms=('no_bin0',))['no_bin0']
            if result is None:
                result = {key: np.empty((3, n) + v.shape[2:] if key == 'ordinary_raw' else (n,) + v.shape[1:], v.dtype) for key, v in part.items()}
            for key, v in part.items():
                if key == 'ordinary_raw':
                    result[key][:, sel] = v
                else:
                    result[key][sel] = v
    finally:
        EV.paths = original
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **result)
    return result


def prepare_eval():
    TR.configure()  # cuBLAS DLL directory for the CuPy renderer
    jobs = []
    for i, (name, (folder, ids)) in enumerate(eval_sets().items()):
        jobs.append((name, folder, ids, HOLD_CONDITIONS['deploy'], 100 + i))
    EV = _ev()
    for i, (cond, spec) in enumerate(HOLD_CONDITIONS.items()):
        if cond == 'nominal_nuis':
            continue  # same render as nominal_clean
        jobs.append((f'hold_{cond}', EV.OUT / 'data/hold', None, spec, 200 + i))
    for name, folder, ids, cond, seed in jobs:
        render_eval(name, folder, ids, cond, seed)


def score_all(stacks=('new', 'robust')):
    began = time.monotonic()
    for stack in stacks:
        names = [n for n in eval_sets()] if stack == 'new' else []
        for name in names:
            score(name, stack, .25, 11)
            print('SCORED', name, stack, round(time.monotonic() - began, 1), flush=True)
        for i, (cond, spec) in enumerate(HOLD_CONDITIONS.items()):
            folder = 'hold_nominal_clean' if cond == 'nominal_nuis' else f'hold_{cond}'
            key = f'hold_{cond}'
            if not (OUT / 'scores' / f'{key}_{stack}.npz').exists():
                if cond == 'nominal_nuis':
                    score(folder, stack, 0., 41, out_name=key)  # shared nominal render, with nuisance
                else:
                    score(folder, stack, spec[3], 40 + i)
            print('SCORED', key, stack, round(time.monotonic() - began, 1), flush=True)


# ------------------------------------------------------------------ fit / evaluate
def _strong(scores, cuts):
    EV = _ev(); E = EV.E
    return (2 * (E.smooth(scores['m3_raw']) >= cuts['raised_tau'])).astype(np.int8)


def fit():
    EV = _ev(); C, E, B, T, V, CM, FM = EV.C, EV.E, EV.B, EV.T, EV.V, EV.CM, EV.FM
    cal = dict(np.load(OUT / 'scores/cal_new.npz'))
    with np.load(EV.F.OLD / 'baselines/cal_ideal.npz') as a:
        oldm, oldl = E.smooth(a['m3_raw']), E.smooth(a['local_raw'])
    with np.load(EVAL / 'cal/geometry.npz') as g:
        category = g['category']
    clear = (category == 'clear').all(1)
    target = int(E.old_fusion(oldm, oldl)[clear].any(-1).sum())
    m = E.smooth(cal['m3_raw'])
    raised, count = B.nearest(m[clear].max(-1), target)
    single = [float(E.threshold(E.calibrate(oldm, oldl, E.smooth(cal['ordinary_raw'][si]), category)['standalone'])) for si in range(3)]
    cuts = dict(m3_tau=raised, raised_tau=raised, local_tau=1e308, single=single, strong_clear_target=target, strong_clear_actual=count,
                rule='pose/floor DR M3-only strong; raised matched to original old5 clear count on deploy-render cal; LOCAL disabled')
    strain = dict(np.load(OUT / 'scores/strain_new.npz'))
    with np.load(EVAL / 'strain/geometry.npz') as g:
        scat, support = g['category'], g['target_support']
    rows = C.read(EVAL / 'strain/scene_rows.json')
    labels, mask, weights, counts = T.supervision(scat, support, _strong(strain, cuts) > 0, [r['lateral_gap_m'] for r in rows])
    out = OUT / 'S'; out.mkdir(exist_ok=True)
    T.train_s(out, EV.s_features(strain, cuts), labels, weights)
    blocks, cats, olds, prow = [], [], [], []
    for dataset in EV.POOL:
        for split in ('cal', 'hold'):
            name = f'pool_{dataset}_{split}'
            blocks.append(dict(np.load(OUT / 'scores' / f'{name}_new.npz')))
            with np.load(EVAL / name / 'geometry.npz') as g:
                cats.append(g['category']); ids = g['source_ids']
            with np.load(EV.OUT.parent / dataset / 'data' / split / 'scores.npz') as o:
                olds.append((2 * E.old_fusion(E.smooth(o['m3_raw'][ids]), E.smooth(o['local_raw'][ids]))).astype(np.int8))
            prow.extend(C.read(EVAL / name / 'scene_rows.json'))
    pool = {key: np.concatenate([b[key] for b in blocks], axis=1 if key == 'ordinary_raw' else 0) for key in blocks[0]}
    pcat = np.concatenate(cats); old5 = np.concatenate(olds)
    masks, _, _, _ = CM.negative_masks(pcat, prow)
    den = int(masks['far_pass_plus_clear'].sum() * old5.shape[1])
    base = int((FM.replay_gap1(old5)[masks['far_pass_plus_clear']] > 0).any((2, 3)).sum())
    light = T.predict_s(out, EV.s_features(pool, cuts)).mean(0)
    strong = _strong(pool, cuts)
    taus, stats, ties = V.sweep(strong, light, pcat, prow, lambda: None, True)
    feasible = np.flatnonzero(stats[:, 8] <= base + .03 * den)
    chosen = int(feasible[0]) if len(feasible) else len(taus) - 1
    cuts['s_tau'] = float(taus[chosen]) if np.isfinite(taus[chosen]) else (1e308 if taus[chosen] > 0 else -1e308)
    cuts['s_calibration'] = dict(status='SEALED' if len(feasible) else 'STRONG_ALREADY_EXCEEDS_POOLED_CAP', old5_far_clear_clips=base,
                                 far_clear_denominator=den, selected_far_clear_clips=int(stats[chosen, 8]), strong_only_far_clear_clips=int(stats[-1, 8]))
    (OUT / 'cuts.json').write_text(json.dumps(dict(cuts=cuts, supervision=counts), indent=1, default=float), encoding='utf8')
    manifest = dict(candidate='pose_floor_dr_stack', drop_bins=[0],
                    m3_paths=[dict(path=p.relative_to(ROOT).as_posix(), sha256=TR.sha(p)) for p in sorted((OUT / 'models/M3').glob('model_seed*.pt'))],
                    ordinary_paths=[dict(path=(OUT / 'models' / f'ordinary_seed{s}.pt').relative_to(ROOT).as_posix(), sha256=TR.sha(OUT / 'models' / f'ordinary_seed{s}.pt')) for s in SEEDS],
                    s_paths=[[dict(path=(out / 'models' / f'S_{s}_{h}.pickle').relative_to(ROOT).as_posix(), sha256=TR.sha(out / 'models' / f'S_{s}_{h}.pickle')) for h in ('HEAD', 'BODY')] for s in SEEDS],
                    cuts={k: cuts[k] for k in ('m3_tau', 'raised_tau', 'local_tau', 'single', 's_tau')}, development_only=True)
    (OUT / 'real_manifest.json').write_text(json.dumps(manifest, indent=1, default=float), encoding='utf8')
    print(json.dumps(cuts, default=float, indent=1))


def evaluate():
    EV = _ev(); C, E, T, TM = EV.C, EV.E, EV.T, EV.TM
    rows = C.read(EV.OUT / 'scene_rows.json')['hold']
    with np.load(EV.OUT / 'data/hold/geometry.npz') as a:
        category = a['category']
    with np.load(EV.OUT / 'hold_scores/old_scores.npz') as a:
        baseline = (2 * E.old_fusion(E.smooth(a['m3_raw']), E.smooth(a['local_raw']))).astype(np.int8)
    ncuts = json.loads((OUT / 'cuts.json').read_text(encoding='utf8'))['cuts']
    rcuts = json.loads((ROBUST / 'cuts.json').read_text(encoding='utf8'))['cuts']
    table = {}
    for cond in HOLD_CONDITIONS:
        for stack, cuts, sdir in (('robust', rcuts, ROBUST / 'S'), ('new', ncuts, OUT / 'S')):
            sc = dict(np.load((OUT if stack == 'new' else BASE) / 'scores' / f'hold_{cond}_{stack}.npz'))
            strong = _strong(sc, cuts)
            light = T.predict_s(sdir, EV.s_features(sc, cuts)).mean(0)
            grade = TM.apply_gate(strong, light, cuts['s_tau'])
            st, _ = EV.describe(grade, baseline, category, rows)
            ss, _ = EV.describe(strong, baseline, category, rows)
            r = dict(timely=st['HEAD']['timely'] + st['BODY']['timely'], denominator=st['HEAD']['denominator'] + st['BODY']['denominator'],
                     far_clear_rate=st['far_clear']['rate'], weighted_cost=st['weighted_cost'],
                     strong_timely=ss['HEAD']['timely'] + ss['BODY']['timely'], strong_far_clear_rate=ss['far_clear']['rate'])
            table[f'{stack}|{cond}'] = r
            print(f"{stack:7s} {cond:17s} timely {r['timely']}/{r['denominator']} ({100*r['timely']/r['denominator']:.1f}%)  far+clear {100*r['far_clear_rate']:.1f}%"
                  f"  | strong-only timely {100*r['strong_timely']/r['denominator']:.1f}% far+clear {100*r['strong_far_clear_rate']:.1f}%", flush=True)
    (OUT / 'evaluation.json').write_text(json.dumps(table, indent=1), encoding='utf8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('render_train', 'features', 'train_m3', 'train_ordinary', 'prepare_eval', 'score_all', 'fit', 'evaluate'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--stacks', nargs='+', default=['new', 'robust'])
    p.add_argument('--noise', type=float, nargs=2, default=None, help='extra iid normalized noise SD range; writes a separate output')
    a = p.parse_args()
    if a.noise:
        NOISE = tuple(a.noise)
        OUT = ROOT / 'artifacts.local/work/cnh-pose-floor-noise-dr-dev-20261012'; OUT.mkdir(parents=True, exist_ok=True)
        FEAT_SEED += 1
    if a.stage == 'train_m3':
        train_m3(a.seeds)
    elif a.stage == 'score_all':
        score_all(tuple(a.stacks))
    else:
        print(globals()[a.stage]())

"""Robust ToF stack: nuisance+pitch DR for ordinary and S, on top of the pitch-DR M3 (EXPLORE).

Stages
  ordinary_features / train_ordinary : ordinary token features from the no-bin0 DR training
      histories with the same per-row nuisance and assumed-pitch error as the pitch-DR M3;
      original ordinary recipe (3 seeds, 32 epochs, 7488 steps), bf16 autocast.
  score  : frozen robust M3 + new ordinary scores (and LOCAL, current-peak S features) on the
      backbone-cal, S-train, pooled-negative and hold sets, each under a 'deploy' condition:
      per scene nuisance (p_clean .25, tail t1~U(0,3), zone U(.5,1.5), DC U(-.3,.8)) and an
      assumed-pitch error from {-10,-5,0,5,10} deg. The hold is additionally scored clean and
      under fixed pitch errors for evaluation.
  fit    : strong grade = raised robust M3 only (LOCAL disabled); raised cut and ordinary
      single cuts calibrated on the deploy-condition cal set to the original clear-slot targets;
      S refit on deploy-condition S-train scores; S tau sealed on deploy pooled negatives
      (lowest whole-tie tau with far+clear notified clips <= original old5 + 3pp).
  evaluate : hold grades vs the current default candidate (no_bin0 S) under each condition.
Development only; every evaluated set is consumed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cnh_no_bin0_dr_train_20261011 as TR
import cnh_nuisance_pitch_dr_m3_dev_20261012 as Q

ROOT = TR.ROOT
SRC = TR.OUT
OUT = ROOT / 'artifacts.local/work/cnh-robust-stack-dev-20261012'
M3_DIR = ROOT / 'artifacts.local/work/cnh-nuisance-pitch-dr-m3-dev-20261012/models/M3'
SEEDS = (2026100955, 2026100956, 2026100957)
ORD_SEED = 2026101207
CHANNELS = TR.CHANNELS
DELTAS = (-10, -5, 0, 5, 10)


def ordinary_features():
    import cnh_boundary_token_model as M
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'ordinary.npy'
    if target.exists():
        raise FileExistsError(target)
    z = np.load(SRC / 'inputs/train/histories.npy', mmap_mode='r')
    t = np.load(SRC / 'inputs/train/transforms.npy', mmap_mode='r')
    le = np.load(SRC / 'inputs/train/length.npy', mmap_mode='r')
    rng = np.random.default_rng(ORD_SEED)
    out = np.lib.format.open_memmap(target, mode='w+', dtype=np.float16, shape=(len(z), 2, 16, 9, 8, 8))
    began = time.monotonic()
    with torch.no_grad():
        for begin in range(0, len(z), 64):
            end = min(begin + 64, len(z)); n = end - begin
            zz = np.array(z[begin:end]).astype(np.float32)
            add, _ = Q.nuisance(rng, n)
            ll = np.array(le[begin:end])
            valid = np.arange(8)[None, :] >= (8 - ll)[:, None]
            zz += add[:, None] * valid[:, :, None, None, None]
            mats, _ = Q.pitch_error(rng, n)
            tt = np.array(t[begin:end]) @ mats[:, None]
            zt = torch.as_tensor(zz, device='cuda').half().float(); zt[..., 0] = 0
            tt = torch.as_tensor(tt, device='cuda', dtype=torch.float32)
            lt = torch.as_tensor(ll, device='cuda')
            geometry = M.feature_geometry(tt, lt, 'center')
            out[begin:end] = M.build_features(zt, geometry, lt)[:, :, :, list(CHANNELS)].half().cpu().numpy()
            if begin % (64 * 128) == 0:
                print('ORD_FEATURES', begin, '/', len(z), round(time.monotonic() - began, 1), flush=True)
    out.flush()
    receipt = dict(stage='ordinary_features', rows=int(len(z)), seconds=time.monotonic() - began, seed=ORD_SEED,
                   nuisance='as cnh_nuisance_pitch_dr_m3_dev_20261012', code_sha256=TR.sha(Path(__file__)))
    (OUT / 'ordinary_features_receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf8')
    return receipt


def train_ordinary():
    import cnh_boundary_token_model as M
    from cnh_counterfactual_train_dev import restore_features, slot_schedule
    TR.configure()
    x = TR.load_gpu(OUT / 'ordinary.npy')
    y = torch.as_tensor(np.load(SRC / 'inputs/train/labels.npy'), device='cuda')
    w = torch.as_tensor(np.load(SRC / 'inputs/train/weights.npy'), device='cuda')
    le = torch.as_tensor(np.load(SRC / 'inputs/train/length.npy'), device='cuda')
    for seed in SEEDS:
        target = OUT / 'models' / f'ordinary_seed{seed}.pt'; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        torch.manual_seed(seed); net = M.BoundaryTokenReadout().cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001); steps = 0; history = []
        for epoch in range(32):
            t0 = time.monotonic(); order, _ = slot_schedule(seed, epoch, 39936, 19968); ws = ms = 0.
            for begin in range(0, len(x), 256):
                ix = torch.as_tensor(order[begin:begin + 256], device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    raw = net(restore_features(x[ix]), le[ix])
                lossraw = F.binary_cross_entropy_with_logits(raw.float(), y[ix], reduction='none')
                mass = w[ix].sum(); weighted = (lossraw * w[ix]).sum(); loss = weighted / mass
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); steps += 1
                ws += float(weighted.detach()); ms += float(mass)
            history.append(dict(epoch=epoch + 1, loss=ws / ms, seconds=time.monotonic() - t0))
            print('TRAIN_ORD', seed, epoch + 1, round(history[-1]['loss'], 6), round(history[-1]['seconds'], 1), flush=True)
        if steps != 7488:
            raise ValueError('Original ordinary schedule differs')
        torch.save(dict(state_dict=net.cpu().state_dict(), arm='robust', seed=seed, epochs=32, steps=steps, drop_bins=[0]), target)
        target.with_suffix('.json').write_text(json.dumps(dict(seed=seed, steps=steps, amp_bf16=True, sha256=TR.sha(target), history=history), indent=1), encoding='utf8')
        del net, opt
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- scoring / fit / evaluate
def _ev():
    import cnh_no_bin0_dr_eval_20261011 as EV
    return EV


def _rot(delta):
    from cnh_cvr_pilot import rotation
    r = np.eye(4); r[:3, :3] = rotation(delta, 'x'); return r


def _nuisance_scene(rng, n, k, p_clean):
    """Per scene-replica nuisance in normalized units [n,k,1,8,8,16] (held over frames)."""
    clean = rng.random((n, k)) < p_clean
    t1 = np.where(clean, 0., rng.uniform(0, 3, (n, k)))[..., None, None]
    zone = rng.uniform(.5, 1.5, (n, k, 8, 8))
    dc = np.where(clean, 0., rng.uniform(-.3, .8, (n, k)))
    add = np.zeros((n, k, 8, 8, 16), np.float32)
    for b, f in ((1, 1.), (2, .43), (3, .30)):
        add[..., b] = t1 * f * zone
    add[..., 1:] += dc[..., None, None, None]
    return add[:, :, None]


def score(folder, ids, stack, condition, seed, out_name):
    """condition: ('deploy',) random per-scene pitch; ('fixed', delta, p_clean) one pitch for all."""
    EV = _ev()
    target = OUT / 'scores' / f'{out_name}.npz'
    if target.exists():
        return {k: v for k, v in np.load(target).items()}
    z = np.asarray(EV.normalized(folder, ids)).astype(np.float32)
    n, k = z.shape[:2]
    with np.load(Path(folder) / 'geometry.npz') as g:
        sensor, query = g['sensor'], g['public_query']
    rng = np.random.default_rng(seed)
    p_clean = .25 if condition[0] == 'deploy' else condition[2]
    if p_clean < 1:
        z = z + _nuisance_scene(rng, n, k, p_clean)
    z = z.astype(np.float16)
    delta = rng.choice(DELTAS, n) if condition[0] == 'deploy' else np.full(n, condition[1])
    if stack == 'robust':
        m3 = sorted(M3_DIR.glob('model_seed*.pt')); ordinary = [OUT / 'models' / f'ordinary_seed{s}.pt' for s in SEEDS]
    else:
        m3, ordinary = EV.paths('no_bin0')
    original = EV.paths; EV.paths = lambda arm: (list(m3), list(ordinary))
    result = None
    try:
        for d in np.unique(delta):
            sel = np.flatnonzero(delta == d); r = _rot(float(d))
            part = EV.infer(z[sel], sensor @ r, query @ r, arms=('no_bin0',))['no_bin0']
            if result is None:
                result = {key: np.empty((3, n) + v.shape[2:] if key == 'ordinary_raw' else (n,) + v.shape[1:], v.dtype)
                          for key, v in part.items()}
            for key, v in part.items():
                if key == 'ordinary_raw':
                    result[key][:, sel] = v
                else:
                    result[key][sel] = v
    finally:
        EV.paths = original
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **result, pitch_delta=delta)
    return result


def score_all(part='all'):
    EV = _ev(); C = EV.C; F_ = EV.F
    jobs = [(F_.OLD / 'data/cal', None, 'robust', ('deploy',), 11, 'cal_robust_deploy'),
            (OUT.parent / 'cnh-task-cost-retrain-dev-20261010/data/train', None, 'robust', ('deploy',), 12, 'strain_robust_deploy')]
    for di, dataset in enumerate(EV.POOL):
        rows = C.read(OUT.parent / dataset / 'scene_rows.json')
        for si, split in enumerate(('cal', 'hold')):
            ids = np.array([i for i, r in enumerate(rows[split]) if r['placement'] in ('pass', 'clear')])
            jobs.append((OUT.parent / dataset / 'data' / split, ids, 'robust', ('deploy',), 20 + 2 * di + si, f'pool_{dataset}_{split}_robust_deploy'))
    hold = EV.OUT / 'data/hold'
    for stack in ('robust', 'default'):
        jobs.append((hold, None, stack, ('fixed', 0, 1.), 40, f'hold_{stack}_clean_0'))
        for d in DELTAS:
            jobs.append((hold, None, stack, ('fixed', d, 0.), 41, f'hold_{stack}_nuis_{d}'))
    began = time.monotonic()
    hold_jobs = [j for j in jobs if j[5].startswith('hold_')]
    jobs = hold_jobs if part == 'hold' else [j for j in jobs if j not in hold_jobs] if part == 'train' else jobs
    for folder, ids, stack, cond, seed, name in jobs:
        score(folder, ids, stack, cond, seed, name)
        print('SCORED', name, round(time.monotonic() - began, 1), flush=True)


def _strong(scores, cuts):
    EV = _ev(); E = EV.E
    return (2 * ((E.smooth(scores['m3_raw']) >= cuts['raised_tau']) | (E.smooth(scores['local_raw']) >= cuts['local_tau']))).astype(np.int8)


def fit():
    EV = _ev(); C, E, B, T, V, CM, FM = EV.C, EV.E, EV.B, EV.T, EV.V, EV.CM, EV.FM
    cal = dict(np.load(OUT / 'scores/cal_robust_deploy.npz'))
    with np.load(EV.F.OLD / 'baselines/cal_ideal.npz') as a:
        oldm, oldl = E.smooth(a['m3_raw']), E.smooth(a['local_raw'])
    with np.load(EV.F.OLD / 'data/cal/geometry.npz') as g:
        category = g['category']
    clear = (category == 'clear').all(1)
    target = int(E.old_fusion(oldm, oldl)[clear].any(-1).sum())
    m = E.smooth(cal['m3_raw'])
    raised, count = B.nearest(m[clear].max(-1), target)
    single = [float(E.threshold(E.calibrate(oldm, oldl, E.smooth(cal['ordinary_raw'][si]), category)['standalone'])) for si in range(3)]
    cuts = dict(m3_tau=raised, raised_tau=raised, local_tau=1e308, single=single, strong_clear_target=target, strong_clear_actual=count,
                rule='robust M3-only strong; raised matched to original old5 clear count on deploy-condition cal; LOCAL disabled')
    src = OUT.parent / 'cnh-task-cost-retrain-dev-20261010'
    strain = dict(np.load(OUT / 'scores/strain_robust_deploy.npz'))
    with np.load(src / 'data/train/geometry.npz') as g:
        scat, support = g['category'], g['target_support']
    rows = C.read(src / 'scene_rows.json')['train']
    labels, mask, weights, counts = T.supervision(scat, support, _strong(strain, cuts) > 0, [r['lateral_gap_m'] for r in rows])
    feats = EV.s_features(strain, cuts)
    out = OUT / 'S'; out.mkdir(exist_ok=True)
    T.train_s(out, feats, labels, weights)
    blocks, cats, olds, prow = [], [], [], []
    for dataset in EV.POOL:
        allrows = C.read(OUT.parent / dataset / 'scene_rows.json')
        for split in ('cal', 'hold'):
            folder = OUT.parent / dataset / 'data' / split
            ids = np.array([i for i, r in enumerate(allrows[split]) if r['placement'] in ('pass', 'clear')])
            blocks.append(dict(np.load(OUT / 'scores' / f'pool_{dataset}_{split}_robust_deploy.npz')))
            with np.load(folder / 'geometry.npz') as g:
                cats.append(g['category'][ids])
            with np.load(folder / 'scores.npz') as o:
                olds.append((2 * E.old_fusion(E.smooth(o['m3_raw'][ids]), E.smooth(o['local_raw'][ids]))).astype(np.int8))
            prow.extend(allrows[split][i] for i in ids)
    pool = {key: np.concatenate([b[key] for b in blocks], axis=1 if key == 'ordinary_raw' else 0) for key in blocks[0] if key != 'pitch_delta'}
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
    print(json.dumps(cuts, default=float, indent=1))


def evaluate():
    EV = _ev(); C, E, T, TM = EV.C, EV.E, EV.T, EV.TM
    rows = C.read(EV.OUT / 'scene_rows.json')['hold']
    with np.load(EV.OUT / 'data/hold/geometry.npz') as a:
        category = a['category']
    with np.load(EV.OUT / 'hold_scores/old_scores.npz') as a:
        baseline = (2 * E.old_fusion(E.smooth(a['m3_raw']), E.smooth(a['local_raw']))).astype(np.int8)
    rcuts = json.loads((OUT / 'cuts.json').read_text(encoding='utf8'))['cuts']
    dcuts = json.loads((EV.OUT / 'main_real_manifest.json').read_text(encoding='utf8'))['cuts']
    table = {}
    for cond in ['clean_0'] + [f'nuis_{d}' for d in DELTAS]:
        for stack, cuts, sdir in (('default', dcuts, EV.OUT / 'no_bin0'), ('robust', rcuts, OUT / 'S')):
            sc = dict(np.load(OUT / 'scores' / f'hold_{stack}_{cond}.npz'))
            strong = _strong(sc, cuts)
            light = T.predict_s(sdir, EV.s_features(sc, cuts)).mean(0)
            grade = TM.apply_gate(strong, light, cuts['s_tau'])
            st, _ = EV.describe(grade, baseline, category, rows)
            ss, _ = EV.describe(strong, baseline, category, rows)
            r = dict(timely=st['HEAD']['timely'] + st['BODY']['timely'], denominator=st['HEAD']['denominator'] + st['BODY']['denominator'],
                     far_clear_rate=st['far_clear']['rate'], weighted_cost=st['weighted_cost'],
                     strong_timely=ss['HEAD']['timely'] + ss['BODY']['timely'], strong_far_clear_rate=ss['far_clear']['rate'])
            table[f'{stack}|{cond}'] = r
            print(f"{stack:8s} {cond:9s} timely {r['timely']}/{r['denominator']} ({100*r['timely']/r['denominator']:.1f}%)  far+clear {100*r['far_clear_rate']:.1f}%  cost {r['weighted_cost']:.0f}  | strong-only timely {100*r['strong_timely']/r['denominator']:.1f}% far+clear {100*r['strong_far_clear_rate']:.1f}%", flush=True)
    (OUT / 'evaluation.json').write_text(json.dumps(table, indent=1), encoding='utf8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('ordinary_features', 'train_ordinary', 'score_all', 'fit', 'evaluate'))
    p.add_argument('--part', default='all', choices=('all', 'train', 'hold'))
    a = p.parse_args()
    print(score_all(a.part) if a.stage == 'score_all' else globals()[a.stage]())

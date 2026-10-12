"""Consumed Development noise-only ablation; original robust RNG/calibration retained."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import cnh_no_bin0_dr_train_20261011 as TR
import cnh_nuisance_pitch_dr_m3_dev_20261012 as Q
import cnh_nuisance_dr_m3_dev_20261011 as N
import cnh_robust_stack_dev_20261012 as RS
import cnh_pose_floor_dr_dev_20261012 as P

ROOT = TR.ROOT
OUT = ROOT / 'artifacts.local/work/cnh-noise-only-ablation-dev-20261012'
OLD_Q, OLD_RS, OLD_P = Q.OUT, RS.OUT, P.OUT
JOINT = OUT.parent / 'cnh-pose-floor-noise-dr-dev-20261012'
NOISE_SEED = 2026101251


def save(path, value):
    path.write_text(json.dumps(value, indent=1, default=float), encoding='utf8')


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = OUT / 'PLAN.json'
    if not plan.exists():
        save(plan, dict(lane='EXPLORE consumed Development', budget='one complete stack: five M3 and three ordinary models, original schedules; original S refit',
            goal='isolate additive training noise versus original robust stack', adjustable='independent Gaussian noise only; no pose/floor rerender; no threshold tuning from hold or real',
            noise_sd=[.85, 1.05], bin_noise_scale=P.BIN_NOISE_SCALE.tolist(), noise_seed=NOISE_SEED,
            m3_rng=Q.SEED, ordinary_rng=RS.ORD_SEED, training_source=str(TR.OUT / 'inputs/train'),
            calibration='RS original cal/strain/six pools, same seeds, nuisance, pitch, fit and pooled cap',
            evaluation='eight already consumed posefloor conditions, paired scene/replica/query timely and far-clear; real replay by parent',
            decision='report counts, rescue/loss and costs; no promotion gate, no superiority assumption; joint arm is practical unmatched comparator',
            resume='complete feature receipts/models/scores reused; incomplete feature/model outputs rejected, no silent overwrite; at most current model training lost on interruption',
            local_skip='EV.s_features reads ordinary_raw/current/current_valid only; LOCAL disabled at 1e308, so zero LOCAL is equivalent',
            python=sys.executable, torch=torch.__version__, device=torch.cuda.get_device_name(), code_sha256=TR.sha(Path(__file__))))
    TR.configure()


def features(kind):
    import cnh_boundary_token_model as M
    from cnh_cvr_projection import Projector
    receipt = OUT / f'{kind}_features_receipt.json'
    if receipt.exists():
        return
    target = OUT / f'{kind}.npy'
    if target.exists():
        raise FileExistsError(f'Incomplete feature file retained: {target}')
    src = TR.OUT / 'inputs/train'
    z = np.load(src / 'histories.npy', mmap_mode='r')
    t = np.load(src / 'transforms.npy', mmap_mode='r')
    le = np.load(src / 'length.npy', mmap_mode='r')
    ids = np.load(src / 'm3_indices.npy') if kind == 'm3' else np.arange(len(z))
    rng = np.random.default_rng(Q.SEED if kind == 'm3' else RS.ORD_SEED)
    noise_rng = np.random.default_rng(NOISE_SEED + (kind == 'ordinary'))
    shape = (len(ids), 3, 24, 17, 33) if kind == 'm3' else (len(ids), 2, 16, 9, 8, 8)
    projector = Projector() if kind == 'm3' else None
    old = np.load((OLD_Q if kind == 'm3' else OLD_RS) / f'{kind}.npy', mmap_mode='r')
    began = time.monotonic(); deltas = []
    def project(zz, tt, ll):
        if kind == 'm3':
            return Q.fast_project(zz.astype(np.float16), tt, ll, projector).half().cpu().numpy()
        zt = torch.as_tensor(zz, device='cuda').half().float(); zt[..., 0] = 0
        tg = torch.as_tensor(tt, device='cuda', dtype=torch.float32)
        lt = torch.as_tensor(ll, device='cuda')
        return M.build_features(zt, M.feature_geometry(tg, lt, 'center'), lt)[:, :, :, list(TR.CHANNELS)].half().cpu().numpy()
    out = None
    with torch.no_grad():
        for begin in range(0, len(ids), 64):
            ix = ids[begin:begin + 64]; ll = np.array(le[ix]); n = len(ix)
            zz = np.array(z[ix]).astype(np.float32)
            add, _ = Q.nuisance(rng, n)
            valid = (np.arange(8)[None] >= (8 - ll)[:, None])[:, :, None, None, None]
            zz += add[:, None] * valid
            mats, delta = Q.pitch_error(rng, n); deltas.extend(delta.tolist())
            tt = np.array(t[ix]) @ mats[:, None]
            if begin == 0:
                zero = zz.copy(); zero[..., 0] = 0
                actual = project(zero, tt, ll)
                expected = np.array(old[:n]); same = np.array_equal(actual, expected)
                parity = dict(rows=n, exact=same, max_abs=float(np.max(np.abs(actual.astype(float)-expected.astype(float)))), differing=int(np.count_nonzero(actual != expected)))
                save(OUT / f'{kind}_zero_noise_parity.json', parity)
                if not same:
                    raise AssertionError(f'{kind} zero-noise parity failed: {parity}')
                out = np.lib.format.open_memmap(target, mode='w+', dtype=np.float16, shape=shape)
            sd = noise_rng.uniform(.85, 1.05, n)[:, None, None, None, None] * P.BIN_NOISE_SCALE
            zz += noise_rng.standard_normal(zz.shape).astype(np.float32) * sd * valid
            zz[..., 0] = 0
            out[begin:begin+n] = project(zz, tt, ll)
            if begin % 8192 == 0:
                print('FEATURES', kind, begin, len(ids), round(time.monotonic()-began, 1), flush=True)
    out.flush()
    np.save(OUT / f'{kind}_pitch_delta_deg.npy', np.asarray(deltas, np.float32))
    if kind == 'm3':
        np.testing.assert_array_equal(np.asarray(deltas, np.float32), np.load(OLD_Q / 'pitch_delta_deg.npy'))
    save(receipt, dict(rows=len(ids), seconds=time.monotonic()-began, parity=parity, nuisance_pitch_rng_unchanged=True, code_sha256=TR.sha(Path(__file__))))


def train_m3():
    N.OUT = OUT
    seeds = []
    for s in range(5):
        p = OUT / f'models/M3/model_seed{s}.pt'
        if p.exists():
            assert p.with_suffix('.json').exists(), f'Incomplete model: {p}'
        else:
            seeds.append(s)
    if seeds:
        N.train(seeds, amp=True)


def train_ordinary():
    RS.OUT = OUT
    pending = []
    for s in P.SEEDS:
        p = OUT / f'models/ordinary_seed{s}.pt'
        if p.exists():
            assert p.with_suffix('.json').exists(), f'Incomplete model: {p}'
        else:
            pending.append(s)
    RS.SEEDS = tuple(pending)
    if pending:
        RS.train_ordinary()
    RS.SEEDS = P.SEEDS


def score_calibration():
    RS.OUT = OUT; RS.M3_DIR = OUT / 'models/M3'
    P._no_local()
    RS.score_all('train')


def fit():
    RS.OUT = OUT
    if not (OUT / 'cuts.json').exists():
        if (OUT / 'S/models').exists():
            raise FileExistsError('Partial S fit retained; inspect before resume')
        RS.fit()
    cuts = json.loads((OUT / 'cuts.json').read_text())['cuts']
    def entry(p):
        return dict(path=p.relative_to(ROOT).as_posix(), sha256=TR.sha(p))
    manifest = dict(candidate='noise_only_robust_ablation', development_only=True, drop_bins=[0],
        m3_paths=[entry(p) for p in sorted((OUT / 'models/M3').glob('model_seed*.pt'))],
        ordinary_paths=[entry(OUT / f'models/ordinary_seed{s}.pt') for s in P.SEEDS],
        s_paths=[[entry(OUT / f'S/models/S_{s}_{h}.pickle') for h in ('HEAD','BODY')] for s in P.SEEDS],
        cuts={k: cuts[k] for k in ('m3_tau','raised_tau','local_tau','single','s_tau')})
    save(OUT / 'real_manifest.json', manifest)


def score_hold():
    P.OUT = OUT
    for i, (cond, spec) in enumerate(P.HOLD_CONDITIONS.items()):
        folder = 'hold_nominal_clean' if cond == 'nominal_nuis' else f'hold_{cond}'
        P.score(folder, 'new', spec[3], 41 if cond == 'nominal_nuis' else 40+i, out_name=f'hold_{cond}')
        print('HOLD_SCORED', cond, flush=True)


def evaluate():
    EV = RS._ev()
    rows = EV.C.read(EV.OUT / 'scene_rows.json')['hold']
    category = np.load(EV.OUT / 'data/hold/geometry.npz')['category']
    arms = [('robust', OLD_RS, OLD_P, 'robust'), ('noise_only', OUT, OUT, 'new'), ('joint', JOINT, JOINT, 'new')]
    results = {}
    for cond in P.HOLD_CONDITIONS:
        grades = {}
        for arm, folder, score_folder, suffix in arms:
            cuts = json.loads((folder / 'cuts.json').read_text())['cuts']
            sc = dict(np.load(score_folder / f'scores/hold_{cond}_{suffix}.npz'))
            strong = P._strong(sc, cuts)
            light = EV.T.predict_s(folder / 'S', EV.s_features(sc, cuts)).mean(0)
            grade = EV.TM.apply_gate(strong, light, cuts['s_tau'])
            grades[arm] = dict(all=grade, strong=strong, light_only=(grade == 1).astype(np.int8))
        results[cond] = {}
        for arm, layers in grades.items():
            results[cond][arm] = {layer: EV.describe(g, grades['robust'][layer], category, rows)[0] for layer, g in layers.items()}
            print('RESULT', cond, arm, json.dumps(results[cond][arm]['all']), flush=True)
    save(OUT / 'evaluation_detailed.json', results)


def cross_calibration():
    """Posthoc 2x2 strong-only calibration diagnostic, separately authorized after real replay."""
    EV = RS._ev()
    receipt = OUT / 'cross_calibration.json'
    if receipt.exists():
        return
    addendum = OUT / 'cross_calibration_PLAN.json'
    if not addendum.exists():
        save(addendum, dict(reason='noise-only real quiet segments remain strong; distinguish training from calibration coverage',
            scope='two existing calibration sets, frozen weights, same old5 clear80 whole-tie rule; strong-only 2x2 posthoc',
            budget='two additional calibration inferences; no training, no S refit, no new data',
            prohibition='do not overwrite sealed cuts or interpret posthoc as independent confirmation',
            code_sha256=TR.sha(Path(__file__))))
    P.OUT = OUT
    noise_pose = P.score('cal', 'new', .25, 11, out_name='crosscal_posefloor')
    target = OUT / 'scores/crosscal_original_joint.npz'
    if target.exists():
        joint_original = dict(np.load(target))
    else:
        folder = EV.F.OLD / 'data/cal'
        z = np.asarray(EV.normalized(folder)).astype(np.float32)
        n, k = z.shape[:2]
        with np.load(folder / 'geometry.npz') as g:
            sensor, query = g['sensor'], g['public_query']
        rng = np.random.default_rng(11)
        z = (z + RS._nuisance_scene(rng, n, k, .25)).astype(np.float16)
        delta = rng.choice(RS.DELTAS, n)
        original_paths = EV.paths
        EV.paths = lambda arm: (sorted((JOINT / 'models/M3').glob('model_seed*.pt')),
                               [JOINT / f'models/ordinary_seed{s}.pt' for s in P.SEEDS])
        joint_original = None
        P._no_local()
        try:
            for d in np.unique(delta):
                sel = np.flatnonzero(delta == d); r = RS._rot(float(d))
                part = EV.infer(z[sel], sensor @ r, query @ r, arms=('no_bin0',))['no_bin0']
                if joint_original is None:
                    joint_original = {key: np.empty((3, n) + v.shape[2:] if key == 'ordinary_raw' else (n,) + v.shape[1:], v.dtype) for key, v in part.items()}
                for key, value in part.items():
                    if key == 'ordinary_raw':
                        joint_original[key][:, sel] = value
                    else:
                        joint_original[key][sel] = value
        finally:
            EV.paths = original_paths
        np.savez_compressed(target, **joint_original, pitch_delta=delta)
    category = np.load(EV.F.OLD / 'data/cal/geometry.npz')['category']
    pose_category = np.load(P.EVAL / 'cal/geometry.npz')['category']
    np.testing.assert_array_equal(category, pose_category)
    clear = (category == 'clear').all(1)
    with np.load(EV.F.OLD / 'baselines/cal_ideal.npz') as old:
        count_target = int(EV.E.old_fusion(EV.E.smooth(old['m3_raw']), EV.E.smooth(old['local_raw']))[clear].any(-1).sum())
    new_cuts = {}
    for key, scores in (('noise_only|posefloor', noise_pose), ('joint|original', joint_original)):
        tau, count = EV.B.nearest(EV.E.smooth(scores['m3_raw'])[clear].max(-1), count_target)
        new_cuts[key] = dict(raised_tau=float(tau), clear_target=count_target, clear_actual=int(count))
    for key, folder in (('noise_only|original', OUT), ('joint|posefloor', JOINT)):
        cuts = json.loads((folder / 'cuts.json').read_text())['cuts']
        new_cuts[key] = dict(raised_tau=cuts['raised_tau'], clear_target=cuts['strong_clear_target'], clear_actual=cuts['strong_clear_actual'])
    rows = EV.C.read(EV.OUT / 'scene_rows.json')['hold']
    hold_category = np.load(EV.OUT / 'data/hold/geometry.npz')['category']
    metrics = {}
    for cond in P.HOLD_CONDITIONS:
        baseline_scores = dict(np.load(OLD_P / f'scores/hold_{cond}_robust.npz'))
        baseline = P._strong(baseline_scores, json.loads((OLD_RS / 'cuts.json').read_text())['cuts'])
        metrics[cond] = {}
        for key, cut in new_cuts.items():
            folder = OUT if key.startswith('noise_only') else JOINT
            scores = dict(np.load(folder / f'scores/hold_{cond}_new.npz'))
            grade = P._strong(scores, cut)
            metrics[cond][key] = EV.describe(grade, baseline, hold_category, rows)[0]
    save(receipt, dict(posthoc=True, training_unchanged=True, strong_only=True, cuts=new_cuts, hold=metrics))
    print('CROSS_CALIBRATION', json.dumps(new_cuts), flush=True)


def run(stages):
    setup()
    for stage in stages:
        began = time.monotonic()
        save(OUT / 'status.json', dict(status='running', stage=stage, pid=os.getpid(), started=time.time()))
        print('STAGE_START', stage, flush=True)
        if stage in ('m3_features','ordinary_features'):
            features(stage.split('_')[0])
        else:
            globals()[stage]()
        torch.cuda.empty_cache()
        save(OUT / f'{stage}_complete.json', dict(stage=stage, seconds=time.monotonic()-began, finished=time.time()))
        print('STAGE_COMPLETE', stage, round(time.monotonic()-began, 1), flush=True)
    save(OUT / 'status.json', dict(status='complete', stages=stages, pid=os.getpid(), finished=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stages', nargs='*', default=['m3_features','ordinary_features','train_m3','train_ordinary','score_calibration','fit','score_hold','evaluate'])
    args = p.parse_args()
    try:
        run(args.stages)
    except BaseException:
        if OUT.exists():
            save(OUT / f'failure_{int(time.time())}.json', dict(traceback=traceback.format_exc(), pid=os.getpid()))
        raise

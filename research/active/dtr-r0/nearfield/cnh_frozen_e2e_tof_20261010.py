"""Frozen models on new physical ToF scenes, then causal notification evaluation.

prepare creates geometry only; run requires the controller's sealed PLAN.
No training, protected outcomes, native RGB substitution or validation tuning.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
import cnh_counterfactual_common_dev as C
import cnh_counterfactual_data_dev as D
import cnh_counterfactual_eval_dev as E
import cnh_graded_evidence_dev as G
import cnh_frozen_e2e_source_20261010 as S

ROOT = C.ROOT
OLD = C.OUT
JOINT = ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
THRESHOLDS = ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json'
DEFAULT_OUT = ROOT/'artifacts.local/work/cnh-frozen-e2e-20261010'
HERE = Path(__file__).resolve().parent


def save_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Preserve evidence: {path}')
    C.save(path, value)


def prepare(out):
    began = time.monotonic()
    if (out/'prepare_receipt.json').exists():
        raise FileExistsError('Preserve prepared cohort')
    _, _, _, B = D.frozen_imports()
    sensor, query = B.poses(-10.)
    rows = S.scenes()
    old_rows = C.read(OLD/'scene_rows.json')
    old_keys = {r['physical_key'] for split in ('ordinary_train', 'cf_train', 'cal', 'validation')
                for r in old_rows[split]}
    info = {}
    for split, values in rows.items():
        keys = {r['physical_key'] for r in values}
        if keys & old_keys:
            raise ValueError('Fresh physical worlds overlap consumed train/cal/validation')
        cat = S.categories(values, sensor)
        folder = out/'data'/split
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/'geometry.npz', category=cat, sensor=sensor, public_query=query,
            scene_ids=np.arange(len(values)), scene_uids=np.array([r['scene_uid'] for r in values]))
        info[split] = dict(scenes=len(values), unique_physical=len(keys), old_physical_overlap=0,
            contact_per_height=(S.K*(cat == 'contact').sum(0)).tolist(),
            clear_scenes=int((cat == 'clear').all(1).sum()),
            pass_scenes=int(((cat == 'pass').any(1)&~(cat == 'contact').any(1)).sum()),
            families=sorted({r['background_family'] for r in values}), geometry_sha256=C.sha(folder/'geometry.npz'))
    save_new(out/'scene_rows.json', rows)
    model_paths = [ROOT/f'artifacts.local/work/cnh-margin-labels-20261002/models/M3/model_seed{s}.pt' for s in range(5)]
    model_paths += [OLD/f'models/ordinary_seed{s}.pt' for s in G.SEEDS]
    model_paths += [JOINT/f'models/{s}_score_current_{height}.pickle' for s in G.SEEDS for height in E.HEIGHTS]
    # Materialized input declarations import the actual frozen geometry chain.
    import cnh_bar_local_readout_dev  # noqa: F401
    import cnh_boundary_token_model  # noqa: F401
    import cnh_graded_peak_joint_features_dev  # noqa: F401
    import cnh_graded_corridor_eval_dev  # noqa: F401
    import cnh_frozen_e2e_metrics_20261010  # noqa: F401
    paths = [Path(__file__), THRESHOLDS, JOINT/'calibrations.json', *model_paths]
    paths += [Path(m.__file__).resolve() for n, m in list(sys.modules.items())
              if n.startswith('cnh_') and getattr(m, '__file__', None)]
    hashes = {str(p): C.sha(p) for p in sorted(set(paths), key=str)}
    save_new(out/'input_hashes.json', hashes)
    save_new(out/'prepare_receipt.json', dict(status='COMPLETE', seconds=time.monotonic()-began,
        scene_rows_sha256=C.sha(out/'scene_rows.json'), inputs_sha256=C.sha(out/'input_hashes.json'),
        source_sha256=C.sha(Path(__file__)), source_author_sha256=C.sha(Path(S.__file__)),
        cohorts=info, replicas=S.K, model_seeds=list(G.SEEDS), main_seed=G.SEEDS[0],
        new_raw=0, predictions=0, protected_access=0, training=0,
        old_geometry_identity_source=str(OLD/'scene_rows.json'), old_geometry_sha256=C.sha(OLD/'scene_rows.json'),
        independence='No physical-world duplicates within/across fresh splits; no overlap with consumed old train/cal/validation; disjoint new background families and target size/rho. Same simulator; K is correlated noise replication.'))
    print('PREPARED', json.dumps(info), flush=True)


def render(out, rows, check):
    import torch
    dll_handle = None
    dll = Path(torch.__file__).parent.parent/'nvidia/cublas/bin'
    if os.name == 'nt' and dll.is_dir():
        dll_handle = os.add_dll_directory(str(dll))
        os.environ['PATH'] = str(dll)+os.pathsep+os.environ.get('PATH', '')
    engines = []
    maps = {}
    began = time.monotonic()
    try:
        _, R, Renderer, _ = D.frozen_imports()
        by_bg = defaultdict(list)
        for split, values in rows.items():
            with np.load(out/'data'/split/'geometry.npz') as a:
                sensor = a['sensor']
            path = out/'data'/split/'hist.npy'
            if path.exists():
                raise FileExistsError('Partial histogram retained')
            maps[split] = np.lib.format.open_memmap(path, mode='w+', dtype=np.int32,
                shape=(len(values), S.K, 16, 8, 8, 16))
            for row in values:
                by_bg[row['background_id']].append((split, row))
        parity, metadata = [], []
        ambient = None
        for bgid, items in sorted(by_bg.items()):
            check()
            engine = Renderer.ExpectedRenderer(sensor, items[0][1]['background_boxes'])
            engines.append(engine)
            try:
                amb = engine.ambient.copy()
                if ambient is None:
                    ambient = amb
                else:
                    np.testing.assert_array_equal(ambient, amb)
                targets = [row['target_box'] for _, row in items]
                metadata.append(dict(background_id=bgid, backend=engine.metadata))
                for start, endpoints in engine.iter_render(targets, candidate_batch=4, pose_batch=16, deadline_check=check):
                    for j in range(len(endpoints)):
                        split, row = items[start+j]
                        mean = endpoints[j, 0]+row['rho']/Renderer.ENDPOINT_RHO*(endpoints[j, 1]-endpoints[j, 0])
                        # One new world per background checks CUDA/CPU physics parity.
                        if start+j == 0:
                            ref = R.expected(dict(poses=sensor[[0, 3, 13, 15]], boxes=row['boxes']))
                            np.testing.assert_allclose(mean[[0, 3, 13, 15]], ref['expectation'], atol=1e-8, rtol=1e-11)
                            np.testing.assert_array_equal(amb[[0, 3, 13, 15]], ref['ambient'])
                            parity.append(dict(background_id=bgid, max_abs=float(np.abs(mean[[0, 3, 13, 15]]-ref['expectation']).max())))
                        for k in range(S.K):
                            maps[split][row['scene_id'], k] = R.sample(mean, ambient, S.sampling_seed(row, k))[0]
            finally:
                engine.close()
                engines.remove(engine)
            print('FRESH_RENDER', bgid, len(items), round(time.monotonic()-began, 2), flush=True)
        for split, a in maps.items():
            a.flush()
            np.save(out/'data'/split/'ambient.npy', ambient)
        return dict(seconds=time.monotonic()-began, raw_draws=sum(len(r)*S.K for r in rows.values()),
            parity=parity, backend=metadata)
    finally:
        maps.clear()
        for engine in engines:
            engine.close()
        if dll_handle is not None:
            dll_handle.close()


def scientific(out, rows, check):
    # Observation-only model input path. Authoring geometry is used by renderer,
    # never by the three inference mechanisms.
    if (out/'render_receipt.json').exists():
        render_receipt = C.read(out/'render_receipt.json')
        for path, digest in render_receipt['outputs_sha256'].items():
            if C.sha(out/path) != digest:
                raise ValueError('Completed rendering changed: '+path)
    else:
        render_receipt = render(out, rows, check)
        render_receipt['outputs_sha256'] = {f'data/{split}/{name}': C.sha(out/'data'/split/name)
            for split in S.SPLITS for name in ('hist.npy', 'ambient.npy')}
        save_new(out/'render_receipt.json', render_receipt)
    import cnh_aligned_boundary_dev as B
    import cnh_bar_local_readout_dev as L
    import cnh_counterfactual_train_dev as T
    import cnh_boundary_token_model as M
    import cnh_graded_peak_tracks_dev as P
    import cnh_graded_peak_joint_features_dev as F
    A, _, _, normal = B.imports()
    A.OUT = out/'runtime'
    engine = None
    began = time.monotonic()
    try:
        engine = A.Engine()
        torch = engine.torch
        qs, locations = L.patches()
        for split in S.SPLITS:
            check()
            folder = out/'data'/split
            if (folder/'scores_receipt.json').exists():
                record = C.read(folder/'scores_receipt.json')
                if C.sha(folder/'scores.npz') != record['sha256']:
                    raise ValueError('Completed score branch changed')
                continue
            if (folder/'scores.npz').exists():
                raise FileExistsError('Incomplete score branch binding preserved')
            hist = np.load(folder/'hist.npy', mmap_mode='r')
            amb = np.load(folder/'ambient.npy')
            with np.load(folder/'geometry.npz') as a:
                sensor, query = a['sensor'], a['public_query']
            count = len(hist)*S.K
            z = normal(hist.reshape(count, 16, 8, 8, 16), np.broadcast_to(amb, (count, *amb.shape)))
            m3 = np.empty((count, 13, 2), np.float32)
            local = np.empty_like(m3, dtype=float)
            raw = np.empty((len(G.SEEDS), count, 13, 2), np.float32)
            ordinary_models = []
            for seed in G.SEEDS:
                model = M.BoundaryTokenReadout().cuda().eval()
                state = torch.load(OLD/f'models/ordinary_seed{seed}.pt', map_location='cpu', weights_only=True)
                if state['arm'] != 'ordinary' or state['seed'] != seed or state['steps'] != 7488:
                    raise ValueError('Frozen ordinary checkpoint identity changed')
                model.load_state_dict(state['state_dict'])
                ordinary_models.append(model)
            with torch.inference_mode():
                for fj, f in enumerate(D.FRAMES):
                    check()
                    start = max(0, int(f)-7)
                    tr = (query[f]@np.linalg.inv(sensor[f]))[None]@sensor[start:f+1]
                    projection = L.Projection(engine, tr)
                    norms = []
                    for patch in qs:
                        mapped = patch@projection.p
                        variance = np.asarray(mapped.multiply(mapped).sum(1)).ravel()
                        norms.append(np.where(variance > 0, np.sqrt(variance), np.inf))
                    if f in (3, 13):
                        np.testing.assert_array_equal(projection(z[:1, start:f+1]).cpu().numpy(),
                            engine.project(z[:1, start:f+1], tr[None]).cpu().numpy())
                    length = len(tr)
                    padded = np.repeat(np.eye(4)[None], 8, 0)
                    padded[8-length:] = tr
                    for offset in range(0, count, 64):
                        check()
                        stop = min(offset+64, count)
                        feat = projection(z[offset:stop, start:f+1]).cpu().numpy().astype(np.float16)
                        m3[offset:stop, fj] = engine.predict(feat)
                        local[offset:stop, fj] = L.scan(feat, qs, norms, locations)[0]
                        history = np.zeros((stop-offset, 8, 8, 8, 16), np.float16)
                        history[:, 8-length:] = z[offset:stop, start:f+1]
                        histories = torch.as_tensor(history, device='cuda')
                        transforms = torch.as_tensor(np.broadcast_to(padded, (stop-offset, 8, 4, 4)).copy(), device='cuda', dtype=torch.float32)
                        lengths = torch.full((stop-offset,), length, device='cuda', dtype=torch.int64)
                        geometry = M.feature_geometry(transforms, lengths, 'center')
                        compact = M.build_features(histories, geometry, lengths)[:, :, :, list(T.KEPT_CHANNELS)].half()
                        features = T.restore_features(compact)
                        for si, model in enumerate(ordinary_models):
                            raw[si, offset:stop, fj] = model(features, lengths).cpu().numpy()
                    del projection
                    print('FRESH_INFER', split, int(f), round(time.monotonic()-began, 2), flush=True)
            ordinary_models.clear()
            candidates = P.current_candidates(z, P.native_geometry(sensor, query))
            current, valid = F.build(dict(candidate_amplitude=candidates['amplitude'], candidate_valid=candidates['valid'],
                candidate_inner_share=candidates['inner_share'], candidate_xyz=candidates['xyz'],
                candidate_support_low_xyz=candidates['support_low_xyz'], candidate_support_high_xyz=candidates['support_high_xyz']))
            np.savez_compressed(folder/'scores.npz', m3_raw=m3.reshape(len(hist), S.K, 13, 2),
                local_raw=local.reshape(len(hist), S.K, 13, 2), ordinary_raw=raw.reshape(len(G.SEEDS), len(hist), S.K, 13, 2),
                current=current.reshape(len(hist), S.K, 13, 2, 22), current_valid=valid.reshape(len(hist), S.K, 13, 2, 22),
                current_names=np.array(F.CURRENT_NAMES), seeds=np.array(G.SEEDS))
            save_new(folder/'scores_receipt.json', dict(status='COMPLETE', sha256=C.sha(folder/'scores.npz'),
                plan_sha256=C.sha(out/'PLAN.json'), scenes=len(hist), replicas=S.K,
                source_sha256=C.sha(Path(__file__)), label_access=False))
            del z, hist, raw, candidates, current, valid, m3, local
            torch.cuda.empty_cache()
        return dict(render=render_receipt, inference_seconds=time.monotonic()-began,
            backend='FP64 CuPy physics / CPU Skellam; CUDA frozen M3+ordinary, CPU exact local sparse scan and current peaks',
            training=0, protected_access=0, seeds=list(G.SEEDS), rows_each=768*S.K*13)
    finally:
        if engine is not None:
            engine.nets = []
            engine.projector = None
            engine.torch.cuda.empty_cache()
            if hasattr(engine, '_dll'):
                engine._dll.close()


def emit_fast(grade):
    """Vectorized state machine used only for calibration tie enumeration."""
    grade = np.asarray(grade, np.int8)
    flat = np.moveaxis(grade, -2, -1).reshape(-1, 13)
    peak = np.zeros(len(flat), np.int8)
    quiet = np.zeros_like(peak)
    answer = np.zeros_like(flat)
    for f in range(13):
        g = flat[:, f]
        positive = g > 0
        quiet = np.where(positive, 0, quiet+1)
        peak = np.where(quiet > 1, 0, peak)
        answer[:, f] = np.where(positive&(g > peak), g, 0)
        peak = np.where(positive, np.maximum(peak, g), peak)
    return np.moveaxis(answer.reshape(*grade.shape[:-2], 2, 13), -1, -2)


def matching(both, old5, margin, category, check):
    """All whole-score tie cutoffs, exact nonmonotone notification cost replay.

    Only negative-scene labels/costs select tau. Contact margins/utility never
    enter the sweep. Masked contact slots cannot create threshold candidates.
    """
    clear = (category == 'clear').all(1)
    passed = (category == 'pass').any(1)&~(category == 'contact').any(1)
    negative = clear|passed
    neggrade = both[negative].copy()
    negold = old5[negative]
    negmargin = margin[negative]
    negclear = np.repeat(clear[negative], both.shape[1])
    negative_grade = neggrade.reshape(-1, 13, 2)
    negative_old = negold.reshape(-1, 13, 2)
    negative_margin = negmargin.reshape(-1, 13, 2)
    oldemit = emit_fast(2*negative_old.astype(np.int8))
    counts = lambda e: ((e > 0).any(-1).sum(-1))
    oldcount = counts(oldemit)
    caps = np.array([oldcount[negclear].sum(), oldcount[~negclear].sum()], np.int64)
    emitted = emit_fast(negative_grade)
    current_count = counts(emitted)
    cost = np.array([current_count[negclear].sum(), current_count[~negclear].sum()], np.int64)
    eligible = (negative_grade > 0)&~negative_old&np.isfinite(negative_margin)
    clips, frames, queries = np.where(eligible)
    values = negative_margin[eligible]
    order = np.argsort(values, kind='stable')
    values, clips, frames, queries = (a[order] for a in (values, clips, frames, queries))
    bounds = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)] if len(values) else np.array([0])
    if len(values) and values.min() < 0:
        raise ValueError('Original both additional grades require nonnegative evidence margin')
    curve = [dict(tau=0., tau_kind='finite', clear=int(cost[0]), pass_=int(cost[1]), feasible=bool(np.all(cost <= caps)))]
    chosen = 0. if np.all(cost <= caps) else None
    for b, end in zip(bounds[:-1], bounds[1:]):
        check()
        affected = np.unique(clips[b:end])
        old_counts = current_count[affected].copy()
        negative_grade[clips[b:end], frames[b:end], queries[b:end]] = 0
        newemit = emit_fast(negative_grade[affected])
        emitted[affected] = newemit
        current_count[affected] = counts(newemit)
        delta = current_count[affected]-old_counts
        cost += [delta[negclear[affected]].sum(), delta[~negclear[affected]].sum()]
        tau = float(np.nextafter(values[b], np.inf))
        feasible = bool(np.all(cost <= caps))
        curve.append(dict(tau=tau, tau_kind='finite', clear=int(cost[0]), pass_=int(cost[1]), feasible=feasible))
        if chosen is None and feasible:
            chosen = tau
    # All finite negative candidate margins were removed: exact original 5-grid.
    np.testing.assert_array_equal(negative_grade, 2*negative_old.astype(np.int8))
    np.testing.assert_array_equal(cost, caps)
    curve.append(dict(tau=None, tau_kind='positive_infinity', clear=int(caps[0]), pass_=int(caps[1]), feasible=True))
    if chosen is None:
        chosen = np.inf
    grade = np.where(old5, 2, np.where((both > 0)&(margin >= chosen), both, 0)).astype(np.int8)
    result = dict(tau=float(chosen) if np.isfinite(chosen) else None,
        tau_kind='finite' if np.isfinite(chosen) else 'positive_infinity' if chosen > 0 else 'negative_infinity',
        clear_cap=int(caps[0]), pass_cap=int(caps[1]), thresholds=len(curve), negative_margin_ties=len(bounds)-1,
        contact_utility_access=False, algorithm='Ascending complete negative score ties with local exact episode replay; all points enumerated, no monotonicity assumption')
    return grade, result, curve


def score_split(out, split):
    import cnh_graded_corridor_eval_dev as Corr
    with np.load(out/'data'/split/'scores.npz', allow_pickle=False) as a:
        m3, local, raw = E.smooth(a['m3_raw']), E.smooth(a['local_raw']), a['ordinary_raw']
        current, valid = a['current'].astype(float), a['current_valid']
    ths, cuts = C.read(THRESHOLDS), C.read(JOINT/'calibrations.json')
    old5 = E.old_fusion(m3, local)
    boths, margins, joints = [], [], []
    valid = valid & np.isfinite(current)
    spatial = np.concatenate((np.where(valid, current, np.nan), (~valid).astype(float)), -1)
    for si, seed in enumerate(G.SEEDS):
        th = ths[str(seed)]
        ordinary = E.smooth(raw[si])
        features = np.concatenate((Corr.build_score_features(raw[si], ordinary, th['single']), spatial), -1)
        joint = np.empty_like(ordinary, dtype=float)
        for q, height in enumerate(E.HEIGHTS):
            with (JOINT/f'models/{seed}_score_current_{height}.pickle').open('rb') as stream:
                model = pickle.load(stream)
            x = features[..., q, :]
            joint[..., q] = model.decision_function(x.reshape(-1, 47)).reshape(x.shape[:-1])
        cutoff = cuts[f'{seed}/score_current/c15_p64']['theta']
        strong = old5|(ordinary >= th['addition'])
        base = np.where(strong, 2, np.where(ordinary >= th['single'], 1, 0)).astype(np.int8)
        both = np.where((base == 0)&np.isfinite(joint)&(joint >= cutoff), 1, base).astype(np.int8)
        margin = np.maximum(ordinary-th['single'], joint-cutoff)
        assert np.all(both[old5] == 2)
        boths.append(both)
        margins.append(margin)
        joints.append(joint)
    return dict(m3=(2*(m3 >= E.M3_THETA)).astype(np.int8), old5=(2*old5).astype(np.int8),
        both=np.array(boths), margin=np.array(margins), joint_scores=np.array(joints))


def evaluate(out, rows, check):
    import cnh_frozen_e2e_metrics_20261010 as Metrics
    began = time.monotonic()
    data = {}
    for split in S.SPLITS:
        with np.load(out/'data'/split/'geometry.npz') as a:
            data[split] = dict(category=a['category'])
    cal = score_split(out, 'cal')
    calibrations, curves, matched = {}, [], []
    for si, seed in enumerate(G.SEEDS):
        grade, record, curve = matching(cal['both'][si], cal['old5'] > 0, cal['margin'][si], data['cal']['category'], check)
        calibrations[str(seed)] = record
        matched.append(grade)
        curves += [dict(seed=seed, **r) for r in curve]
    cal['matched'] = np.array(matched)
    # Bind the three cost-selected cutoffs before opening holdout score readout.
    save_new(out/'sealed_calibration.json', dict(calibrations=calibrations, plan_sha256=C.sha(out/'PLAN.json'),
        cal_scores_sha256=C.sha(out/'data/cal/scores.npz'), selection='Negative costs only; hold metrics unopened'))
    G.write_csv(out/'calibration_curve.csv', curves)
    hold = score_split(out, 'hold')
    held_matched = []
    for si, seed in enumerate(G.SEEDS):
        record = calibrations[str(seed)]
        tau = record['tau'] if record['tau_kind'] == 'finite' else np.inf if record['tau_kind'] == 'positive_infinity' else -np.inf
        held_matched.append(np.where(hold['old5'] > 0, 2,
            np.where((hold['both'][si] > 0)&(hold['margin'][si] >= tau), hold['both'][si], 0)).astype(np.int8))
    hold['matched'] = np.array(held_matched)
    reports, comparisons, event_rows = {}, {}, []
    for split, scores in (('cal', cal), ('hold', hold)):
        check()
        keys, grades, emitted = [], [], []
        category = data[split]['category']
        for si, seed in enumerate(G.SEEDS):
            for arm in ('m3', 'old5', 'both', 'matched'):
                grade = scores[arm] if arm in ('m3', 'old5') else scores[arm][si]
                key = f'{seed}/{arm}'
                notice = Metrics.replay_gap1(grade)
                np.testing.assert_array_equal(notice, emit_fast(grade))
                reports[f'{split}/{key}'] = Metrics.summarize(grade, category, rows[split])
                if arm in ('both', 'matched'):
                    comparisons[f'{split}/{key}'] = Metrics.compare(scores['old5'], grade, category, rows[split])
                clocks = dict(first_any=G.first(notice > 0), first_timely=G.first(notice > 0, 11),
                    first_strong=G.first(notice == 2), first_strong_timely=G.first(notice == 2, 11),
                    old5_first_timely=G.first(Metrics.replay_gap1(scores['old5']) > 0, 11))
                for n, row in enumerate(rows[split]):
                    for k in range(S.K):
                        for q, height in enumerate(E.HEIGHTS):
                            event_rows.append(dict(split=split, seed=seed, arm=arm, scene=n, replica=k,
                                scene_uid=row['scene_uid'], height=height, category=category[n, q],
                                shape_family=row['shape_family'], size_variant=row['size_variant'],
                                background_family=row['background_family'], background_id=row['background_id'],
                                **{name:int(value[n, k, q]) for name, value in clocks.items()},
                                light_notifications=int((notice[n, k, :, q] == 1).sum()),
                                strong_notifications=int((notice[n, k, :, q] == 2).sum())))
                keys.append(key)
                grades.append(grade)
                emitted.append(notice)
        np.savez_compressed(out/f'{split}_grades_notifications.npz', keys=np.array(keys), grades=np.array(grades),
            notifications=np.array(emitted), category=category, scene_ids=np.arange(len(rows[split])),
            joint_scores=scores['joint_scores'], margin=scores['margin'], seeds=np.array(G.SEEDS))
    G.write_csv(out/'event_ledger.csv', event_rows)
    save_new(out/'metrics.json', dict(reports=reports, comparisons=comparisons, calibrations=calibrations,
        evidence='Fresh physical AABB backgrounds and targets, same simulator; ToF-to-notification only; synchronized RGB not evaluable'))
    return dict(seconds=time.monotonic()-began, cells=len(reports), comparisons=len(comparisons),
        sealed_calibration_sha256=C.sha(out/'sealed_calibration.json'), training=0, source_sha256=C.sha(Path(__file__)))


def run(out):
    start = time.monotonic()
    plan = C.read(out/'PLAN.json')
    prep = C.read(out/'prepare_receipt.json')
    if (out/'receipt.json').exists():
        raise FileExistsError('Preserve completed result')
    for path, digest in C.read(out/plan.get('runtime_input_hashes', 'input_hashes.json')).items():
        if C.sha(path) != digest:
            raise ValueError(f'Frozen input changed: {path}')
    if C.sha(out/'scene_rows.json') != prep['scene_rows_sha256']:
        raise ValueError('Prepared scene identity changed')
    rows = C.read(out/'scene_rows.json')
    for split in S.SPLITS:
        if C.sha(out/'data'/split/'geometry.npz') != prep['cohorts'][split]['geometry_sha256']:
            raise ValueError('Prepared evaluator geometry changed')
    for name in ('TEMP', 'TMP', 'CUPY_CACHE_DIR'):
        folder = out/'runtime'/('cupy-cache' if name == 'CUPY_CACHE_DIR' else 'tmp')
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(folder)
    previous = sum(C.read(p)['seconds'] for p in (out/'stage_receipts').glob('scientific_*.json'))
    def science_check():
        if previous+time.monotonic()-start >= 600:
            raise TimeoutError('Frozen scientific command-wall600s reached')
    phase = 'scientific'
    try:
        if not (out/'scientific_receipt.json').exists():
            result = scientific(out, rows, science_check)
            scientific_seconds = time.monotonic()-start
            save_new(out/'scientific_receipt.json', dict(status='COMPLETE', seconds=scientific_seconds,
                plan_sha256=C.sha(out/'PLAN.json'), detail=result,
                outputs_sha256={f'data/{split}/{name}': C.sha(out/'data'/split/name)
                    for split in S.SPLITS for name in ('scores.npz', 'scores_receipt.json')}))
        else:
            receipt = C.read(out/'scientific_receipt.json')
            if receipt['plan_sha256'] != C.sha(out/'PLAN.json'):
                raise ValueError('Scientific PLAN changed on resume')
            for path, digest in receipt['outputs_sha256'].items():
                if C.sha(out/path) != digest:
                    raise ValueError('Completed scientific output changed: '+path)
        evalstart = time.monotonic()
        evalprevious = sum(C.read(p)['phase_seconds'] for p in (out/'stage_receipts').glob('evaluation_failure_*.json'))
        phase = 'evaluation'
        def eval_check():
            if evalprevious+time.monotonic()-evalstart >= 180:
                raise TimeoutError('Frozen evaluation180s reached')
        with threadpool_limits(limits=2):
            eval_result = evaluate(out, rows, eval_check)
        save_new(out/'evaluation_receipt.json', dict(status='COMPLETE', **eval_result))
        save_new(out/'receipt.json', dict(status='COMPLETE', seconds=time.monotonic()-start,
            plan_sha256=C.sha(out/'PLAN.json'), scientific_seconds=C.read(out/'scientific_receipt.json')['seconds'],
            evaluation_seconds=eval_result['seconds'], protected_access=0, training=0, resource_release=True,
            command=[sys.executable, *sys.argv]))
        print('FROZEN_E2E_COMPLETE', round(time.monotonic()-start, 3), flush=True)
    except BaseException as error:
        record = dict(status='FAILED', phase=phase, seconds=time.monotonic()-start,
            phase_seconds=time.monotonic()-(evalstart if phase == 'evaluation' else start),
            error=repr(error), traceback=traceback.format_exc())
        save_new(out/f'failure_{time.time_ns()}.json', record)
        if phase == 'scientific':
            save_new(out/'stage_receipts'/f'scientific_failure_{time.time_ns()}.json', record)
        else:
            save_new(out/'stage_receipts'/f'evaluation_failure_{time.time_ns()}.json', record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Canonical artifacts.local required')
    (prepare if args.stage == 'prepare' else run)(out)

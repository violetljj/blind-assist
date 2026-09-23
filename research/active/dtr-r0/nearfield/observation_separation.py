"""Matched public observation distances, followed by separate truth evaluation."""
import argparse
from collections import defaultdict
import itertools
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

from query_occupancy_data import read, write, sha, observation_tokens
from tof_fov45_core import boxes45, simulate
from ba_camera_corridor import sample_native

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = REPO / 'artifacts.local/evidence/ba-observation-separation-20260923'
COHORTS = {'stability': ('ba-local-stability-20260923', 'local-stability/'),
           'rescue': ('ba-local-rescue-fresh-20260923', 'local-rescue-fresh/')}
DRAWS = 16


def matched_poses(spec):
    groups = defaultdict(dict)
    for i, case in enumerate(spec['cases']):
        if case['trajectory'] == 'approach_dwell_return' and case['frame_in_clip'] in range(4, 9):
            groups[case['base_group_id']].setdefault(case['layout_relation'], []).append(i)
    for group, poses in groups.items():
        assert set(poses) == {'INSIDE', 'BOUNDARY', 'OUTSIDE'}, group
        anchor = spec['cases'][poses['OUTSIDE'][0]]
        for relation, indices in poses.items():
            assert [spec['cases'][i]['frame_in_clip'] for i in indices] == list(range(4, 9))
            c = spec['cases'][indices[0]]
            for i in indices:
                r = spec['cases'][i]
                assert r['camera'] == c['camera'] and r['objects'] == c['objects']
            assert c['camera'] == anchor['camera']
            assert c['nominal_front_m'] == anchor['nominal_front_m']
            for a, b in zip(c['objects'], anchor['objects'], strict=True):
                assert {k: v for k, v in a.items() if k != 'center_m'} == {
                    k: v for k, v in b.items() if k != 'center_m'}
                if a['name'] == 'background':
                    assert a == b
                else:
                    assert a['center_m'][0::2] == b['center_m'][0::2]
    return dict(groups)


def rgb_matrices_numpy(images):
    x = np.asarray(images, np.float32) / 255
    delta = np.abs(x[:, None] - x[None, :])
    full = delta.mean((2, 3, 4))
    patch = delta.reshape(len(x), len(x), 3, 9, 20, 16, 20).mean((2, 4, 6)).max((2, 3))
    return np.stack([full, patch])


def rgb_matrices_torch(images):
    import torch
    x = torch.as_tensor(np.asarray(images), device='cuda', dtype=torch.float32) / 255
    delta = (x[:, None] - x[None, :]).abs()
    full = delta.mean((2, 3, 4))
    patch = delta.reshape(len(x), len(x), 3, 9, 20, 16, 20).mean((2, 4, 6)).amax((2, 3))
    # Include result transfer in the equivalent-work benchmark.
    return torch.stack([full, patch]).cpu().numpy()


def tof_matrices(tokens):
    t = np.asarray(tokens, np.float64)
    z, valid = t[:, :, 0] * 8, t[:, :, 1] == 1
    common = valid[:, None] & valid[None, :]
    counts = common.sum(2)
    delta2 = (z[:, None] - z[None, :]) ** 2
    variance = (.01 + .02*z[:, None])**2 + (.01 + .02*z[None, :])**2
    denom = np.maximum(counts, 1)
    rms = np.sqrt(np.where(common, delta2, 0).sum(2) / denom)
    standardized = np.sqrt(np.where(common, delta2/variance, 0).sum(2) / denom)
    rms[counts == 0] = np.nan
    standardized[counts == 0] = np.nan
    validity = (valid[:, None] != valid[None, :]).mean(2)
    return np.stack([rms, standardized, validity]), counts


def separation(matrix, n, skip_first=False):
    left = list(range(1 if skip_first else 0, n))
    right = list(range(n + (1 if skip_first else 0), 2*n))
    within = np.asarray([matrix[a, b] for ids in (left, right)
                         for a, b in itertools.combinations(ids, 2)], float)
    cross = matrix[np.ix_(left, right)].ravel().astype(float)
    w, c = within[np.isfinite(within)], cross[np.isfinite(cross)]
    missing = int(len(within)-len(w)+len(cross)-len(c))
    if not len(w) or not len(c):
        return dict(within_count=len(within), cross_count=len(cross), missing=missing,
                    separated=False, ratio=None, auc=None, cross_above_noise_fraction=None)
    noise = float(np.quantile(w, .95)); median = float(np.median(c))
    above = int((c > noise).sum())
    auc = float(((c[:, None] > w).sum() + .5*(c[:, None] == w).sum())/(len(c)*len(w)))
    return dict(within_count=len(within), cross_count=len(cross), missing=missing,
                within_p95=noise, cross_median=median, cross_min=float(c.min()),
                ratio=median/noise if noise else None, auc=auc,
                cross_above_noise_fraction=above/len(cross),
                separated=bool(missing == 0 and above/len(cross) >= .9))


def prepare():
    paths = [Path(__file__), HERE/'OBSERVATION_SEPARATION_PROTOCOL_20260923.md',
             HERE/'tof_fov45_core.py', HERE/'ba_camera_corridor.py',
             HERE/'query_occupancy_data.py']
    write(ROOT/'plan/seal.json', dict(files={p.relative_to(REPO).as_posix(): sha(p) for p in paths}))
    for p in paths:
        shutil.copy2(p, ROOT/'plan'/p.name)
    inputs = [dict(alias='plan', path=str(ROOT/'plan'), role='configuration', purpose='fixed-observation-diagnostic')]
    for cohort, (stem, _) in COHORTS.items():
        base = ROOT.parent/stem
        for alias, path, role in [
            ('spec', base/'plan/spec.json', 'configuration'),
            ('observations', base.with_name(stem+'-prepared')/'observations', 'observation'),
            ('manifest', base.with_name(stem+'-prepared')/'materialization.json', 'configuration'),
            ('native', base.with_name(stem+'-capture')/'evaluator', 'evaluator'),
            ('labels', base.with_name(stem+'-prepared')/'labels/evaluation.npz', 'evaluator')]:
            inputs.append(dict(alias=cohort+'_'+alias, path=str(path), role=role,
                               purpose='consumed-matched-repeat-observation-separation'))
    write(ROOT/'run-spec.json', dict(schema='blindassist-asset-run-v1', id='observation-separation-20260923-v1',
        route='ue-observation-separation', question='Do opposite corridor poses differ beyond repeated-pose observation variation?',
        evaluator=Path(__file__).relative_to(REPO).as_posix(),
        evidence_boundary='Consumed controlled matched-pose diagnostic; not hardware noise or generalizable decoding',
        reuse=dict(mode='diagnostic', query='LOCAL stability rescue same pose dwell RGB ToF noise'), inputs=inputs,
        outputs=[dict(alias='result', path=str(ROOT.with_name(ROOT.name+'-run')/'result.json'), role='result', required=True)],
        result_output='result', command=[sys.executable, str(Path(__file__)), 'execute', '--result', '{{output:result}}']))


def execute(result):
    assert read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])['state'] == 'running'
    start = time.perf_counter(); out = result.parent
    seal = read(ROOT/'plan/seal.json')
    for f, h in seal['files'].items():
        assert sha(REPO/f) == h
    sys.path.insert(0, str(REPO))
    from tools.research_backend import BackendCandidate, DeviceObservation, select_backend
    import torch
    rgb_fn = None
    pending = []; parity = 0; source_receipts = {}
    for cohort, (stem, prefix) in COHORTS.items():
        base = ROOT.parent/stem; prep = base.with_name(stem+'-prepared'); cap = base.with_name(stem+'-capture')
        spec = read(base/'plan/spec.json'); manifest = read(prep/'materialization.json')
        geo = read(cap/'evaluator/geometry.json'); groups = matched_poses(spec)
        for f in ('tof.npy', 'rgb.npy', 'identities.json'):
            assert sha(prep/'observations'/f) == manifest['hashes']['observations/'+f]
        rgb = np.load(prep/'observations/rgb.npy', mmap_mode='r')
        tof = np.load(prep/'observations/tof.npy')
        if rgb_fn is None:
            first = next(iter(groups.values()))
            sample = np.asarray(rgb[first['BOUNDARY'] + first['OUTSIDE']])
            cpu = BackendCandidate('numpy-rgb-pair-matrix', 'cpu', lambda: rgb_matrices_numpy(sample),
                lambda _: DeviceObservation('cpu', platform.processor(), 'NumPy '+np.__version__))
            gpu = None
            if torch.cuda.is_available():
                gpu = BackendCandidate('torch-rgb-pair-matrix', 'cuda', lambda: rgb_matrices_torch(sample),
                    lambda _: DeviceObservation('cuda', torch.cuda.get_device_name(), 'Torch '+torch.__version__),
                    torch.cuda.synchronize)
                assert np.allclose(rgb_matrices_numpy(sample), rgb_matrices_torch(sample), atol=2e-6, rtol=2e-6)
            backend = select_backend('batch-tensor', cpu=cpu, gpu=gpu, cpu_reason='ACCELERATOR_UNAVAILABLE',
                                     record_path=out/'backend.json')
            rgb_fn = rgb_matrices_torch if backend['selected_device_type'] == 'cuda' else rgb_matrices_numpy
        repeats = {}; hashes = {}; poses = []
        for group, slots in groups.items():
            for relation, ids in slots.items():
                anchor_depth = None
                for i in ids:
                    g = geo[i]; p = cap/'evaluator'/g['native_path']
                    h = sha(p); assert h == g['native_sha256']; hashes[str(i)] = h
                    native = np.load(p)
                    values, _ = simulate(sample_native(native), prefix+spec['cases'][i]['sensor_noise_key'], boxes45())
                    assert np.array_equal(observation_tokens(values, boxes45()), tof[i]); parity += 1
                    if i == ids[0]: anchor_depth = sample_native(native)
                draws = []
                for draw in range(DRAWS):
                    key = f'observation-separation/{cohort}/{group}/{relation}/{draw}'
                    values, _ = simulate(anchor_depth, key, boxes45())
                    draws.append(observation_tokens(values, boxes45()))
                repeats[(group, relation)] = np.stack(draws)
                poses.append(dict(group=group, relation=relation, indices=ids,
                                  depth_m=spec['cases'][ids[0]]['nominal_front_m']))
            for relation in ('BOUNDARY', 'INSIDE'):
                ids = slots[relation] + slots['OUTSIDE']
                rm = rgb_fn(rgb[ids]); tm, common = tof_matrices(np.concatenate([
                    repeats[group, relation], repeats[group, 'OUTSIDE']]))
                name = f'{cohort}-{group}-{relation.lower()}'
                np.savez_compressed(out/(name+'.npz'), rgb=rm, tof=tm, common=common,
                    draws=np.concatenate([repeats[group, relation], repeats[group, 'OUTSIDE']]),
                    indices=ids)
                pending.append(dict(cohort=cohort, group=group, relation=relation,
                    file=name+'.npz', indices=ids, depth_m=spec['cases'][ids[0]]['nominal_front_m']))
            print('OBSERVATIONS', cohort, group, flush=True)
        source_receipts[cohort] = dict(native_hashes=hashes, poses=poses,
                                      spec_sha256=sha(base/'plan/spec.json'))
    write(out/'public-manifest.json', dict(pairs=pending, sources=source_receipts,
        matrices={p['file']: sha(out/p['file']) for p in pending}, labels_joined=False))
    write(out/'public-seal.json', dict(sha256=sha(out/'public-manifest.json'), labels_joined=False))
    # All public matrices are now sealed. Truth only evaluates pair eligibility.
    rows = []; unknown = {}
    for cohort, (stem, _) in COHORTS.items():
        prep = ROOT.parent/(stem+'-prepared'); manifest = read(prep/'materialization.json')
        p = prep/'labels/evaluation.npz'; assert sha(p) == manifest['hashes']['labels/evaluation.npz']
        labels = np.load(p)
        for pair in [r for r in pending if r['cohort'] == cohort]:
            ids = pair['indices']; valid = bool(labels['valid'][ids][:, [1, 4]].all())
            truth = (labels['classes'][ids][:, [1, 4]] < 6).any(1)
            eligible = valid and truth.tolist() == [True]*5+[False]*5
            with np.load(out/pair['file']) as m:
                metrics = {}
                for k, name in enumerate(('rgb_full_L1', 'rgb_patch_max_L1')):
                    metrics[name] = separation(m['rgb'][k], 5)
                    metrics[name+'_without_arrival'] = separation(m['rgb'][k], 5, True)
                for k, name in enumerate(('tof_rms_m', 'tof_standardized_rms', 'tof_validity_disagreement')):
                    metrics[name] = separation(m['tof'][k], DRAWS)
                counts = m['common'][:DRAWS, DRAWS:]
                coverage = dict(min_common_zones=int(counts.min()), median_common_zones=float(np.median(counts)))
            rows.append(dict(**pair, eligible=eligible, valid=valid, truth=truth.tolist(),
                             metrics=metrics, coverage=coverage))
        # UNKNOWN from the original observed baseline remains source metadata.
        obsmeta = read(prep/'observations/identities.json')
        selected = sorted({i for r in pending if r['cohort'] == cohort for i in r['indices']})
        unknown[cohort] = dict(frames=len(selected),
            unknown=sum(bool(obsmeta[i]['baseline']['unknown']) for i in selected),
            unknown_silent=sum(bool(obsmeta[i]['baseline']['unknown']) and not obsmeta[i]['baseline']['alert']
                               for i in selected))
    keys = list(rows[0]['metrics'])
    summaries = {}
    for cohort in COHORTS:
        summaries[cohort] = {}
        for relation in ('BOUNDARY', 'INSIDE'):
            rr = [r for r in rows if r['cohort'] == cohort and r['relation'] == relation]
            summaries[cohort][relation] = {k: dict(pairs=len(rr),
                eligible=sum(r['eligible'] for r in rr),
                separated=sum(r['eligible'] and r['metrics'][k]['separated'] for r in rr),
                median_ratio=float(np.median([r['metrics'][k]['ratio'] for r in rr
                    if r['metrics'][k]['ratio'] is not None])) if any(r['metrics'][k]['ratio'] is not None for r in rr) else None,
                missing=sum(r['metrics'][k]['missing'] for r in rr)) for k in keys}
    all_rgb = all(r['eligible'] and r['metrics']['rgb_patch_max_L1']['separated'] and
                  r['metrics']['rgb_patch_max_L1_without_arrival']['separated'] for r in rows)
    decision = 'MATCHED_RGB_DIFFERENCES_SURVIVE_VARIATION' if all_rgb else 'PARTIAL_MATCHED_OBSERVATION_SEPARATION'
    write(out/'pairs.json', rows)
    write(result, dict(status='PASS', decision=decision, summary=summaries,
        pairs=len(rows), groups=len({(r['cohort'], r['group']) for r in rows}),
        poses=sum(len(s['poses']) for s in source_receipts.values()),
        original_ToF_parity_frames=parity, noise_draws_per_pose=DRAWS,
        invalid_or_nonopposite_pairs=sum(not r['eligible'] for r in rows),
        source_unknown=unknown,
        fits=0, new_capture_frames=0, elapsed_s=time.perf_counter()-start,
        boundary='Known matched scenes and repeated rendering; no hardware-noise or transferable-decoder claim'))
    for f, h in seal['files'].items(): assert sha(REPO/f) == h
    write(out/'output-seal.json', dict(files={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(decision, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'execute'))
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else execute(args.result)

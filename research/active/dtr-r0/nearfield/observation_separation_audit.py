"""Independent saved-input and exported-matrix audit; no sensor regeneration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO/'artifacts.local/evidence/ba-observation-separation-audit-20260923'
SOURCE = ROOT.parent/'ba-observation-separation-20260923-run'
SOURCE_PLAN = ROOT.parent/'ba-observation-separation-20260923/plan'
COHORTS = {'stability': 'ba-local-stability-20260923',
           'rescue': 'ba-local-rescue-fresh-20260923'}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def prepare():
    write(ROOT/'plan/seal.json', dict(script_sha256=sha(Path(__file__))))
    inputs = [dict(alias='plan', path=str(ROOT/'plan'), role='configuration', purpose='independent-audit'),
              dict(alias='source', path=str(SOURCE), role='evaluator', purpose='exported-public-matrices-and-results'),
              dict(alias='source_plan', path=str(SOURCE_PLAN), role='configuration', purpose='frozen-code-input-seal')]
    for cohort, stem in COHORTS.items():
        base = ROOT.parent/stem
        for suffix, p, role in [
            ('spec', base/'plan/spec.json', 'configuration'),
            ('manifest', ROOT.parent/(stem+'-prepared')/'materialization.json', 'configuration'),
            ('observations', ROOT.parent/(stem+'-prepared')/'observations', 'observation'),
            ('labels', ROOT.parent/(stem+'-prepared')/'labels/evaluation.npz', 'evaluator'),
            ('native', ROOT.parent/(stem+'-capture')/'evaluator', 'evaluator')]:
            inputs.append(dict(alias=cohort+'_'+suffix, path=str(p), role=role,
                               purpose='saved-input-geometry-label-and-UNKNOWN-audit'))
    write(ROOT/'run-spec.json', dict(schema='blindassist-asset-run-v1', id='observation-separation-audit-20260923-v1',
        route='ue-observation-separation', question='Do independent matrix replay, geometry and counts reproduce the separation diagnostic?',
        evaluator=Path(__file__).relative_to(REPO).as_posix(),
        evidence_boundary='Consumed controlled Development audit; no sensor simulation replay or physical noise claim',
        reuse=dict(mode='diagnostic', query='observation separation independent saved inputs audit'), inputs=inputs,
        outputs=[dict(alias='result', path=str(ROOT.with_name(ROOT.name+'-run')/'result.json'), role='result', required=True)],
        result_output='result', command=[sys.executable, str(Path(__file__)), 'execute', '--result', '{{output:result}}']))


def metric(matrix, n, start=0):
    within, cross = [], []
    for base in (0, n):
        for i in range(start, n):
            for j in range(i+1, n):
                within.append(float(matrix[base+i, base+j]))
    for i in range(start, n):
        for j in range(start, n):
            cross.append(float(matrix[i, n+j]))
    w = sorted(v for v in within if np.isfinite(v))
    c = sorted(v for v in cross if np.isfinite(v))
    missing = len(within)+len(cross)-len(w)-len(c)
    result = dict(within_count=len(within), cross_count=len(cross), missing=missing,
                  separated=False, ratio=None, auc=None, cross_above_noise_fraction=None)
    if not w or not c:
        return result
    # Explicit linear percentile interpolation, independent of np.quantile.
    position = .95*(len(w)-1)
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    noise = w[lo]+(w[hi]-w[lo])*(position-lo)
    median = (c[(len(c)-1)//2]+c[len(c)//2])/2
    above = sum(v > noise for v in c)
    wins = sum(1 if v > u else .5 if v == u else 0 for v in c for u in w)
    result.update(within_p95=noise, cross_median=median, cross_min=c[0],
                  ratio=median/noise if noise else None, auc=wins/(len(w)*len(c)),
                  cross_above_noise_fraction=above/len(cross),
                  separated=missing == 0 and above/len(cross) >= .9)
    return result


def replay_rgb(images):
    result = np.zeros((2, len(images), len(images)), np.float64)
    for i in range(len(images)):
        for j in range(len(images)):
            delta = np.abs(images[i].astype(np.float64)-images[j].astype(np.float64))/255
            result[0, i, j] = delta.sum()/delta.size
            result[1, i, j] = max(float(delta[:, y:y+20, x:x+20].mean())
                for y in range(0, 180, 20) for x in range(0, 320, 20))
    return result


def replay_tof(draws):
    result = np.zeros((3, len(draws), len(draws)), np.float64)
    counts = np.zeros((len(draws), len(draws)), np.int64)
    for i, a in enumerate(draws):
        for j, b in enumerate(draws):
            va, vb = a[:, 1] == 1, b[:, 1] == 1
            common = np.flatnonzero(va & vb)
            counts[i, j] = len(common)
            result[2, i, j] = np.count_nonzero(va != vb)/64
            if not len(common):
                result[:2, i, j] = np.nan
                continue
            squares, scaled = [], []
            for k in common:
                za, zb = float(a[k, 0])*8, float(b[k, 0])*8
                d = (za-zb)**2
                squares.append(d)
                scaled.append(d/((.01+.02*za)**2+(.01+.02*zb)**2))
            result[0, i, j] = (sum(squares)/len(common))**.5
            result[1, i, j] = (sum(scaled)/len(common))**.5
    return result, counts


def execute(result_path):
    started = time.perf_counter()
    assert read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])['state'] == 'running'
    assert sha(Path(__file__)) == read(ROOT/'plan/seal.json')['script_sha256']
    checks = 0
    def check(condition):
        nonlocal checks
        assert condition
        checks += 1
    def equal(actual, expected):
        if isinstance(expected, dict):
            check(set(actual) == set(expected))
            for key in expected:
                equal(actual[key], expected[key])
        elif isinstance(expected, float):
            check(bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-12)))
        else:
            check(actual == expected)
    for f, h in read(SOURCE/'output-seal.json')['files'].items():
        check(sha(SOURCE/f) == h)
    for f, h in read(SOURCE_PLAN/'seal.json')['files'].items():
        check(sha(REPO/f) == h)
        check(sha(SOURCE_PLAN/Path(f).name) == h)
    public = read(SOURCE/'public-manifest.json')
    check(not public['labels_joined'])
    check(read(SOURCE/'public-seal.json') == dict(sha256=sha(SOURCE/'public-manifest.json'), labels_joined=False))
    rows = read(SOURCE/'pairs.json'); report = read(SOURCE/'result.json')
    check(len(rows) == len(public['pairs']) == 32)
    check(len({r['file'] for r in rows}) == 32)
    recomputed = []; unknown = {}; max_rgb_error = 0.; max_tof_error = 0.
    selected_frames = 0
    for cohort, stem in COHORTS.items():
        base = ROOT.parent/stem; prepared = ROOT.parent/(stem+'-prepared')
        cap = ROOT.parent/(stem+'-capture')/'evaluator'
        spec = read(base/'plan/spec.json'); cases = spec['cases']
        manifest = read(prepared/'materialization.json')
        check(sha(base/'plan/spec.json') == public['sources'][cohort]['spec_sha256'])
        for f in ('rgb.npy', 'tof.npy', 'identities.json'):
            check(sha(prepared/'observations'/f) == manifest['hashes']['observations/'+f])
        check(sha(prepared/'labels/evaluation.npz') == manifest['hashes']['labels/evaluation.npz'])
        rgb = np.load(prepared/'observations/rgb.npy', mmap_mode='r')
        identities = read(prepared/'observations/identities.json')
        geometry = read(cap/'geometry.json')
        with np.load(prepared/'labels/evaluation.npz') as loaded:
            valid, classes = loaded['valid'], loaded['classes']
        relevant = [r for r in rows if r['cohort'] == cohort]
        check(len(relevant) == 16)
        expected = {}
        for i, case in enumerate(cases):
            if case['trajectory'] == 'approach_dwell_return' and 4 <= case['frame_in_clip'] <= 8:
                expected.setdefault(case['base_group_id'], {}).setdefault(case['layout_relation'], []).append(i)
        check(len(expected) == 8)
        check(len(public['sources'][cohort]['poses']) == 24)
        seen = set()
        for row in relevant:
            key = (row['group'], row['relation']); check(key not in seen); seen.add(key)
            check(row['relation'] in ('INSIDE', 'BOUNDARY'))
            ids = expected[row['group']][row['relation']]+expected[row['group']]['OUTSIDE']
            check(row['indices'] == ids)
            anchor = cases[ids[0]]; outside = cases[ids[5]]
            check(anchor['camera'] == outside['camera'])
            check(anchor['nominal_front_m'] == outside['nominal_front_m'] == row['depth_m'])
            check(all(anchor['camera'][k] == 0 for k in ('pitch', 'yaw', 'roll')))
            for a, b in zip(anchor['objects'], outside['objects'], strict=True):
                check({k:v for k,v in a.items() if k != 'center_m'} == {k:v for k,v in b.items() if k != 'center_m'})
                check(a == b if a['name'] == 'background' else a['center_m'][::2] == b['center_m'][::2])
            truths = []
            for pos, i in enumerate(ids):
                case, geo = cases[i], geometry[i]
                ref = anchor if pos < 5 else outside
                check(case['camera'] == ref['camera'] and case['objects'] == ref['objects'])
                check(case['frame_in_clip'] == 4+pos%5)
                check(geo['declared_camera'] == case['camera'] and geo['id'] == case['name'])
                check(sha(cap/geo['native_path']) == geo['native_sha256'] == public['sources'][cohort]['native_hashes'][str(i)])
                check(max(abs(geo['actual_camera_location_m'][j]-case['camera'][k]) for j,k in enumerate(('x','y','z'))) <= .002)
                positive = False
                for obj, planned in zip(geo['objects'], case['objects'], strict=True):
                    center, half = obj['render_bounds_center_m'], obj['render_bounds_extent_m']
                    check(obj['name'] == planned['name'])
                    check(max(abs(center[j]-planned['center_m'][j]) for j in range(3)) <= .002)
                    check(max(abs(2*half[j]-planned['size_m'][j]) for j in range(3)) <= .002)
                    x = center[1]-case['camera']['y']; y = case['camera']['z']-center[2]; z = center[0]-case['camera']['x']
                    positive |= (x+half[1] >= -.3 and x-half[1] <= .3 and
                                 y+half[2] >= -.2 and y-half[2] <= .9 and
                                 z+half[0] >= .3 and z-half[0] <= 3.)
                truths.append(bool(positive))
            check(truths == (classes[ids][:, [1,4]] < 6).any(1).tolist())
            is_valid = bool(valid[ids][:, [1,4]].all())
            eligible = is_valid and truths == [True]*5+[False]*5
            equal(row['valid'], is_valid); equal(row['truth'], truths); equal(row['eligible'], eligible)
            check(sha(SOURCE/row['file']) == public['matrices'][row['file']])
            match = [p for p in public['pairs'] if p['file'] == row['file']]
            check(len(match) == 1)
            for k,v in match[0].items():
                equal(row[k], v)
            with np.load(SOURCE/row['file']) as loaded:
                rm, tm, common, draws = [loaded[k] for k in ('rgb','tof','common','draws')]
                check(loaded['indices'].tolist() == ids)
            check(rm.shape == (2,10,10) and tm.shape == (3,32,32) and draws.shape == (32,64,6))
            check(np.isfinite(draws).all() and np.isin(draws[:,:,1], [0,1]).all())
            check(np.all(draws[:,:,0][draws[:,:,1] == 0] == 0))
            live = draws[:,:,0][draws[:,:,1] == 1]*8
            check(bool(np.all((live >= .1) & (live < 8))))
            check(bool(np.all(draws[:,:,2:] == draws[0:1,:,2:])))
            rr = replay_rgb(rgb[ids]); tt, cc = replay_tof(draws)
            check(bool(np.allclose(rr, rm, atol=2e-6, rtol=2e-6)))
            check(bool(np.allclose(tt, tm, atol=1e-12, rtol=1e-12, equal_nan=True)))
            check(np.array_equal(cc, common))
            max_rgb_error = max(max_rgb_error, float(np.max(np.abs(rr-rm))))
            max_tof_error = max(max_tof_error, float(np.nanmax(np.abs(tt-tm))))
            metrics = {}
            for k, name in enumerate(('rgb_full_L1','rgb_patch_max_L1')):
                metrics[name] = metric(rm[k], 5)
                metrics[name+'_without_arrival'] = metric(rm[k], 5, 1)
                check(metric(rr[k],5)['separated'] == metrics[name]['separated'])
                check(metric(rr[k],5,1)['separated'] == metrics[name+'_without_arrival']['separated'])
            for k, name in enumerate(('tof_rms_m','tof_standardized_rms','tof_validity_disagreement')):
                metrics[name] = metric(tm[k],16)
            equal(row['metrics'], metrics)
            equal(row['coverage'], dict(min_common_zones=int(cc[:16,16:].min()), median_common_zones=float(np.median(cc[:16,16:]))))
            recomputed.append(dict(cohort=cohort, relation=row['relation'], group=row['group'], eligible=eligible, metrics=metrics))
        selected = sorted({i for row in relevant for i in row['indices']})
        check(len(selected) == 120); selected_frames += len(selected)
        unknown[cohort] = dict(frames=len(selected), unknown=sum(bool(identities[i]['baseline']['unknown']) for i in selected),
            unknown_silent=sum(bool(identities[i]['baseline']['unknown']) and not identities[i]['baseline']['alert'] for i in selected))
    summary = {}
    for cohort in COHORTS:
        summary[cohort] = {}
        for relation in ('BOUNDARY','INSIDE'):
            rr = [r for r in recomputed if r['cohort'] == cohort and r['relation'] == relation]
            summary[cohort][relation] = {}
            for key in rr[0]['metrics']:
                ratios = [r['metrics'][key]['ratio'] for r in rr if r['metrics'][key]['ratio'] is not None]
                summary[cohort][relation][key] = dict(pairs=len(rr), eligible=sum(r['eligible'] for r in rr),
                    separated=sum(r['eligible'] and r['metrics'][key]['separated'] for r in rr),
                    median_ratio=float(np.median(ratios)) if ratios else None,
                    missing=sum(r['metrics'][key]['missing'] for r in rr))
    equal(report['summary'],summary); equal(report['source_unknown'],unknown)
    equal(report['pairs'],32); equal(report['groups'],16); equal(report['poses'],48)
    equal(report['original_ToF_parity_frames'],selected_frames); equal(report['noise_draws_per_pose'],16)
    equal(report['invalid_or_nonopposite_pairs'],sum(not r['eligible'] for r in recomputed))
    equal(report['fits'],0); equal(report['new_capture_frames'],0)
    all_rgb = all(r['eligible'] and r['metrics']['rgb_patch_max_L1']['separated'] and
                  r['metrics']['rgb_patch_max_L1_without_arrival']['separated'] for r in recomputed)
    equal(report['decision'], 'MATCHED_RGB_DIFFERENCES_SURVIVE_VARIATION' if all_rgb else 'PARTIAL_MATCHED_OBSERVATION_SEPARATION')
    check(sha(Path(__file__)) == read(ROOT/'plan/seal.json')['script_sha256'])
    write(result_path, dict(status='PASS', assertions=checks, pairs_replayed=32, source_frames=selected_frames,
        max_rgb_matrix_absolute_error=max_rgb_error, max_tof_matrix_absolute_error=max_tof_error,
        source_decision=report['decision'], elapsed_s=time.perf_counter()-started,
        backend='CPU_FROZEN_PROTOCOL_CPU_ONLY_INDEPENDENT_REPLAY',
        scope='All exported metrics/counts/seals, RGB CPU replay, ToF matrices from exported draws, source geometry/labels/UNKNOWN. '
              'Does not independently rerun sensor simulation, reproduce noise draws or regenerate visible-valid labels; '
              '240 original-ToF parity is checked as frame accounting, not independently measured sensor parity.',
        fits=0, new_capture_frames=0))
    print('INDEPENDENT_OBSERVATION_SEPARATION_AUDIT_PASS', checks, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare','execute'))
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else execute(args.result)

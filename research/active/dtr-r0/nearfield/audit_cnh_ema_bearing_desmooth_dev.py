"""Independent CPU audit of the sole k1 label desmoothing comparison.

Reads cached raw observations and evaluator masks; never imports candidate
decision/metric implementations. All five conditions remain separate and the
decision uses the four challenges within each sensor configuration. This audit
accepts frozen evaluator geometry, rather than independently proving it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-ema-bearing-desmooth-dev-20261008'
SECTOR = ROOT/'artifacts.local/work/cnh-sector-notice-dev-20261008'
NATIVE = ROOT/'artifacts.local/work/cnh-torso-bias-dev-20261007/native_motion'
EMA = NATIVE/'head_yaw/ema_compare_20261008'
OPPORTUNITY = ROOT/'artifacts.local/work/cnh-ema-bearing-opportunity-dev-20261008'
NAMES = ('zero', 'const_pos', 'const_neg', 'pulse_pos', 'pulse_neg')
SENSORS = ('single', 'dual')
CAP_SECONDS = 90.


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def equal(actual, expected, key, floating=False):
    if floating:
        np.testing.assert_allclose(actual, expected, atol=0, rtol=0, err_msg=key)
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=key)


def argmax_and_margin(score):
    """LEFT-first exact ties, independently implemented strict comparisons."""
    labels = np.zeros(score.shape[:-1], np.int8)
    best = score[..., 0].copy()
    for label in (1, 2):
        update = score[..., label] > best
        labels[update] = label
        best[update] = score[..., label][update]
    ordered = np.sort(score, axis=-1)
    return labels, ordered[..., 2]-ordered[..., 1]


def outcomes(labels, first, truth, contact):
    fields = {key: np.zeros(len(contact), bool) for key in ('clean', 'unique', 'wrong', 'abstain')}
    for row in np.flatnonzero(contact & (first >= 0)):
        frame = int(first[row]); label = int(labels[row, frame])
        legal = truth[row, frame]
        if label < 0:
            fields['abstain'][row] = True
        elif legal[label]:
            fields['clean'][row] = True
            fields['unique'][row] = int(legal.sum()) == 1
        else:
            fields['wrong'][row] = True
    return fields


def pair(new, old, mask):
    rescued = int(np.count_nonzero(mask & new & ~old))
    lost = int(np.count_nonzero(mask & ~new & old))
    return dict(rescued=rescued, lost=lost, diff=rescued-lost)


def row_summary(mask, old, new):
    result = dict(timely=int(mask.sum()), old_clean=int(old['clean'][mask].sum()),
                  k1_clean=int(new['clean'][mask].sum()), old_unique=int(old['unique'][mask].sum()),
                  k1_unique=int(new['unique'][mask].sum()))
    result.update({key: value for key, value in pair(new['clean'], old['clean'], mask).items()
                   if key in ('rescued', 'lost')})
    return result


def run():
    tick = time.monotonic()
    audit_path = OUT/'audit.json'
    # Preserve every prior attempt and its consumed wall time in the same receipt.
    previous = read(audit_path) if audit_path.exists() else None
    attempts = [] if previous is None else previous.get('attempts', [])
    if previous is not None and previous.get('status') == 'PASS':
        raise FileExistsError('Completed audit is immutable; no redundant rerun')
    spent = sum(float(item['seconds']) for item in attempts)
    def check():
        if spent+time.monotonic()-tick >= CAP_SECONDS:
            raise TimeoutError('Independent audit cumulative CPU wall cap90 seconds')
    status = 'FAIL'; error = None; evidence = {}; verified = []
    try:
        check()
        plan = read(OUT/'PLAN.json'); result = read(OUT/'result.json')
        assert result['status'] == 'COMPLETE', 'Incomplete result cannot be ranked'
        candidate = arrays(OUT/'candidate.npz'); ledger = arrays(OUT/'ledger.npz')
        source = arrays(SECTOR/'ledger.npz'); opportunity = arrays(OPPORTUNITY/'ledger.npz')
        baseline = arrays(EMA/'ledger.npz'); parent_result = read(SECTOR/'result.json')
        for key in ('unit', 'config', 'contact', 'control', 'deadline'):
            equal(ledger[key], source[key], key)
            equal(ledger[key], baseline[key], 'EMA '+key)
        for key in ('unit', 'config'):
            equal(candidate[key], source[key], 'candidate '+key)
        uid, config, contact, control, deadline = [source[key] for key in ('unit','config','contact','control','deadline')]
        assert (len(uid), int(contact.sum()), int(control.sum())) == (3840, 229, 384)
        assert list(sorted(set(uid))) == list(range(99000, 99096))
        raw_score = np.empty((5, 2, 3840, 13, 3), float)
        old_score = np.empty_like(raw_score)
        yaw_change = np.empty((5, 3840, 13), float)
        raw_arrays = 0
        for unit in range(99000, 99096):
            check(); rows = np.flatnonzero(uid == unit)
            equal(config[rows], np.arange(40), f'config order {unit}')
            with np.load(SECTOR/'units'/f'unit{unit}.npz') as sector, \
                    np.load(NATIVE/'units'/f'unit{unit}.npz') as native, \
                    np.load(NATIVE/'head_yaw/units'/f'unit{unit}.npz') as physical:
                for ni, name in enumerate(NAMES):
                    raw = np.asarray(sector[name+'/raw'], float)
                    assert raw.shape == (3, 3, 40, 13, 2)
                    assert np.isfinite(raw).all()
                    smooth = np.zeros_like(raw)
                    for frame in range(13):
                        start = max(0, frame-4); denominator = 0.
                        for past in range(start, frame+1):
                            weight = float(2**(past-start)); denominator += weight
                            smooth[..., frame, :] += raw[..., past, :]*weight
                        smooth[..., frame, :] /= denominator
                    # Fusion follows per-query/per-branch/per-height smoothing.
                    for si, branches in enumerate(((0,), (1, 2))):
                        direct = raw[:, branches].max(axis=(1, 4)).transpose(1, 2, 0)
                        old = smooth[:, branches].max(axis=(1, 4)).transpose(1, 2, 0)
                        raw_score[ni, si, rows] = direct
                        old_score[ni, si, rows] = old
                    noisy = native['noisy'] if name == 'zero' else physical[name+'/noisy']
                    yaw = np.degrees(np.arctan2(noisy[:, :, 0, 2], noisy[:, :, 2, 2]))[:, 3:]
                    for frame in range(13):
                        delta = yaw[:, frame, None]-yaw[:, max(0, frame-4):frame+1]
                        yaw_change[ni, rows, frame] = np.abs((delta+180.)%360.-180.).max(-1)
                    raw_arrays += 1
        check()
        equal(old_score, source['score'], 'independently rebuilt cached S score', floating=True)
        equal(old_score, candidate['old_score'], 'candidate old_score', floating=True)
        equal(raw_score, candidate['raw_score'], 'candidate raw_score', floating=True)
        equal(yaw_change, candidate['yaw_change'], 'candidate yaw_change', floating=True)
        verified.extend(['96-unit raw and noisy coordinate reconstruction','per-query branch height smoothing before fusion','cached old-score parity','yaw max wrapped deviation over causal five-output window'])
        clean_diffs = {sensor: [] for sensor in SENSORS}
        unique_diffs = {sensor: [] for sensor in SENSORS}
        strata_rows = 0
        for ni, name in enumerate(NAMES):
            for si, sensor in enumerate(SENSORS):
                check(); key = name+'/'+sensor; cell = result['metrics'][key]
                k1_labels, k1_margin = argmax_and_margin(raw_score[ni, si])
                old_labels, old_margin = argmax_and_margin(old_score[ni, si])
                stable = np.zeros((3840, 13), bool)
                for frame in range(13):
                    stable[:, frame] = (k1_labels[:, max(0,frame-4):frame+1] == k1_labels[:, frame,None]).all(-1)
                for arm, labels, margin in (('k1',k1_labels,k1_margin), ('L2',old_labels,old_margin)):
                    equal(candidate[key+'/'+arm+'/labels'], labels, key+' '+arm+' labels')
                    equal(candidate[key+'/'+arm+'/margin'], margin, key+' '+arm+' margin', floating=True)
                equal(candidate[key+'/k1/stable'], stable, key+' stability')
                theta = parent_result['metrics'][key]['ema']['working_point']['threshold']
                alarm = baseline['score'][ni, si, 4] >= theta
                assert cell['threshold'] == theta
                assert cell['total_notices'] == int(alarm.sum())
                first = np.full(3840, -1, int)
                for row in np.flatnonzero(contact):
                    frames = np.flatnonzero(alarm[row, 2:int(deadline[row])+1])
                    if len(frames): first[row] = int(frames[0])+2
                equal(ledger[key+'/alarm'], alarm, key+' preserved alarm')
                equal(ledger[key+'/first'], first, key+' preserved first')
                equal(first, opportunity[key+'/first'], key+' prior EMA first')
                valid = contact & (first >= 0)
                vals = dict(k1=k1_labels, L2=old_labels, L1=opportunity[key+'/L1/labels'])
                reconstructed = {}
                for arm, labels in vals.items():
                    equal(ledger[key+'/'+arm+'/labels'], labels, key+' '+arm+' ledger labels')
                    fields = outcomes(labels, first, source['truth'][ni], contact)
                    reconstructed[arm] = fields
                    rec = cell['arms'][arm]
                    assert int(rec['timely']) == int(valid.sum())
                    for metric, arr in fields.items():
                        equal(ledger[key+'/'+arm+'/'+metric], arr, key+' '+arm+' '+metric)
                        assert int(rec['first_'+metric]) == int(arr.sum()), (key,arm,metric)
                    assert int(fields['clean'].sum()+fields['wrong'].sum()+fields['abstain'].sum()) == int(valid.sum())
                for old_arm in ('L2','L1'):
                    for metric in ('clean','unique','wrong'):
                        expected = pair(reconstructed['k1'][metric], reconstructed[old_arm][metric], contact)
                        actual = cell['k1_against_'+old_arm][metric]
                        for count, value in expected.items(): assert actual[count] == value, (key,old_arm,metric,count)
                rr = np.flatnonzero(valid); ff = first[rr]
                yy = yaw_change[ni, rr, ff]; mm = old_margin[rr, ff]; ss = stable[rr, ff]
                masks = dict(yaw=(yy <= 2., (yy > 2.) & (yy <= 10.), yy > 10.),
                             old_margin=(mm == 0., (mm > 0.) & (mm <= .1), (mm > .1) & (mm <= .5), mm > .5),
                             k1_stability=(ss, ~ss))
                labels_by_group = dict(yaw=['<=2','(2,10]','>10'],
                                       old_margin=['=0','(0,.1]','(.1,.5]','>.5'],
                                       k1_stability=['stable','changing'])
                for group, parts in masks.items():
                    group_rows = cell['strata'][group]
                    if isinstance(group_rows, dict): group_rows = list(group_rows.values())
                    assert len(group_rows) == len(parts), (key,group,'strata length')
                    assert [row['label'] for row in group_rows] == labels_by_group[group]
                    assert sum(int(part.sum()) for part in parts) == len(rr)
                    for actual, part in zip(group_rows, parts):
                        selected = np.zeros(3840, bool); selected[rr[part]] = True
                        expected = row_summary(selected, reconstructed['L2'], reconstructed['k1'])
                        for metric, value in expected.items(): assert actual[metric] == value, (key,group,actual['label'],metric)
                        strata_rows += 1
                for arm, data in (('k1',raw_score),('L2',old_score)):
                    at_first = data[ni, si, rr, ff]
                    tie_count = int(((at_first == at_first.max(-1)[:,None]).sum(-1) > 1).sum())
                    assert cell['first_exact_max_ties'][arm] == tie_count
                if ni > 0:
                    clean_diffs[sensor].append(pair(reconstructed['k1']['clean'], reconstructed['L2']['clean'], contact)['diff'])
                    unique_diffs[sensor].append(pair(reconstructed['k1']['unique'], reconstructed['L2']['unique'], contact)['diff'])
        for sensor in SENSORS:
            dc, du = clean_diffs[sensor], unique_diffs[sensor]
            decision = ('PRIORITIZE_DESMOOTH_FOLLOWUP' if all(d >= 0 for d in dc)
                        and all(d >= 0 for d in du) and any(d > 0 for d in dc)
                        else 'RETAIN_ORIGINAL_L2')
            actual = result['decisions'][sensor]
            assert actual['diffs_clean'] == dc
            assert actual['diffs_unique'] == du
            assert actual['decision'] == decision
        check()
        verified.extend(['fixed EMA alarm and first-notice timing','k1 L2 L1 outcomes and paired counts','90 frozen strata rows','separate single and dual predeclared decisions'])
        evidence = dict(events=229, controls=384, windows=3840, units=96,
                        raw_arrays=raw_arrays, condition_sensor_cells=10, strata_rows=strata_rows,
                        decisions=result['decisions'], candidate_implementation_functions_called=False,
                        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in
                                      (OUT/'PLAN.json',OUT/'candidate.npz',OUT/'ledger.npz',OUT/'result.json',Path(__file__))},
                        limits='Independent recount using accepted cached truth; not new geometric validation, confirmation, hardware or user benefit. Decision concerns only frozen k1 extra-output-smoothing removal; it does not reject all temporal or spatial mechanisms.')
        check(); status = 'PASS'
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        elapsed = time.monotonic()-tick
        attempts.append(dict(status=status, seconds=elapsed, error=error))
        receipt = dict(status=status, cumulative_seconds=spent+elapsed, budget_seconds=CAP_SECONDS,
                       attempts=attempts, verified=verified, evidence=evidence)
        OUT.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(dict(status=status,seconds=elapsed,cumulative_seconds=spent+elapsed,error=error)))


if __name__ == '__main__':
    run()

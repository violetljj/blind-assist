"""Independent replay of additive peak alerts and their cost accounting.

Uses the earlier independent cache/metric implementation, never producer alert
or grading helpers. Peak association itself has its own separate audit.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import audit_cnh_graded_corridor_eval_dev as A

ROOT = A.ROOT
OUT = ROOT / 'artifacts.local/work/cnh-graded-peak-track-dev-20261010/alert'
TRACKS = OUT.parent / 'tracks'


def cutoff(values, cap):
    """Lowest cutoff that admits no partial tied score group above the cap."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if len(values) <= cap:
        return -np.inf
    levels, count = np.unique(values, return_counts=True)
    remaining = len(values) - np.cumsum(count)
    first = np.flatnonzero(remaining <= cap)[0]
    return float(np.nextafter(levels[first], np.inf))


def evidence(split, ids):
    with np.load(TRACKS / f'{split}_tracks.npz', allow_pickle=False) as a:
        np.testing.assert_array_equal(a['scene_ids'], ids)
        np.testing.assert_array_equal(a['frames'], np.arange(3, 16))
        np.testing.assert_array_equal(a['queries'], ['HEAD', 'BODY'])
        amp = a['candidate_amplitude'][..., 0]
        valid = a['candidate_valid'][..., 0]
        names = a['names'].tolist()
        f = a['features']
        fv = a['valid']
        selected = a['best_track_id']
        sums = np.zeros(amp.shape, np.float64)
        available = np.zeros(amp.shape, int)
        for t in range(13):
            # Extractor top1 sum was formed from FP32 current amplitudes.
            lo = max(0, t - 4)
            sums[..., t, :] = amp[..., lo:t+1, :].sum(-2)
            available[..., t, :] = valid[..., lo:t+1, :].sum(-2)
        track_sum = f[..., names.index('tracked_sum')].astype(np.float64)
        track_valid = fv[..., names.index('tracked_sum')]
        hits = f[..., names.index('hits')]
        assert np.all((~track_valid) | (selected >= 0))
        return dict(frame_peak=np.where(valid, amp.astype(np.float64), -np.inf),
                    bag5_sum=np.where((available >= 2) & valid, sums, -np.inf),
                    tracked_sum=np.where(track_valid & (hits >= 2), track_sum, -np.inf))


def added_costs(flags, category):
    pure_clear = (category == 'clear').all(-1)
    pure_pass = (category == 'pass').any(-1) & ~(category == 'contact').any(-1)
    joint = flags.any(-1)
    return {name: A.costs(joint[mask]) for name, mask in
            (('clear', pure_clear), ('pass', pure_pass))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    began = time.monotonic()
    target = OUT / 'audit'
    target.mkdir(exist_ok=True)
    if (target / 'result.json').exists():
        raise FileExistsError('Preserve prior independent audit')
    plan = A.read(OUT / 'PLAN.json')
    cuts = A.read(OUT / 'calibrations.json')
    metrics = A.read(OUT / 'metrics.json')
    for path, digest in plan['inputs_sha256'].items():
        A.eq(sha(ROOT / path), digest, 'input_sha256')
    producer = Path(__file__).with_name('cnh_graded_peak_alert_dev.py')
    A.eq(sha(producer), plan['source_sha256'], 'producer_sha256')
    ds = A.load()
    cal = ds['cal']
    families = {}
    for row in cal['rows']:
        families.setdefault(row['background_family'], set()).add(row['background_id'])
    assert all(len(ids) == 2 for ids in families.values())
    fit_ids = {min(ids) for ids in families.values()}
    cutoff_ids = {row['background_id'] for row in cal['rows']} - fit_ids
    validation_ids = {row['background_id'] for row in ds['validation']['rows']}
    assert not (cutoff_ids & fit_ids or validation_ids & (cutoff_ids | fit_ids))
    A.eq(A.read(OUT / 'cal_partition.json'),
         dict(fit_background_ids=sorted(fit_ids), calibrate_background_ids=sorted(cutoff_ids),
              fit='NOT_RUN; only cal9/11 used for new raw-evidence cutoffs'), 'cal partition')
    selected = np.array([row['background_id'] in cutoff_ids for row in cal['rows']])
    scores = {split: evidence(split, d['ids']) for split, d in ds.items()}
    thresholds = A.read(ROOT / 'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
    saved = {}
    for split in ds:
        with np.load(OUT / f'{split}_grades.npz', allow_pickle=False) as archive:
            saved[split] = dict(zip(archive['keys'].tolist(), archive['grades']))
    reports, bases, firsts, details = {}, {}, {}, {}
    for seed in A.SEEDS:
        refs, baseline = {}, {}
        for split, d in ds.items():
            ordinary = A.smooth(d['raw'][seed])
            th = thresholds[str(seed)]
            old = (d['m3'] >= .9404184587540165) | (d['local'] >= 4.625390338985158)
            strong = old | (ordinary >= th['addition'])
            prior = strong | (ordinary >= th['single'])
            baseline[split] = np.where(strong, 2, np.where(prior, 1, 0)).astype(np.int8)
            refs[split] = dict(prior_light=prior, ordinary_OR=strong,
                               M3=d['m3'] >= .8557642486787612, old_fusion=old)
            bases[(split, seed)] = A.describe(baseline[split], d, refs[split])
        for rule in plan['rules']:
            for clear_cap, pass_cap in plan['extra_budgets_clear_pass_slots']:
                key = f'{seed}/{rule}/c{clear_cap}_p{pass_cap}'
                score = scores['cal'][rule][selected]
                eligible = (baseline['cal'][selected] == 0) & np.isfinite(score)
                candidate = np.where(eligible, score, -np.inf).max(-1)
                category = cal['category'][selected]
                clear = (category == 'clear').all(-1)
                passed = (category == 'pass').any(-1) & ~(category == 'contact').any(-1)
                theta = max(cutoff(candidate[clear], clear_cap), cutoff(candidate[passed], pass_cap), 0.)
                record = cuts[key]
                A.eq(record['theta'], theta, 'cutoff/' + key)
                additions = eligible & (score >= theta)
                ac = int(additions[clear].any(-1).sum())
                ap = int(additions[passed].any(-1).sum())
                A.eq(record, dict(theta=theta, clear_cap=clear_cap, pass_cap=pass_cap,
                                  actual_extra_candidate_clear_slots=ac,
                                  actual_extra_candidate_pass_slots=ap,
                                  clear_unused=clear_cap-ac, pass_unused=pass_cap-ap,
                                  clear_slot_denominator=int(clear.sum())*4*13,
                                  pass_slot_denominator=int(passed.sum())*4*13), 'cal/' + key)
                assert ac <= clear_cap and ap <= pass_cap
                for split, d in ds.items():
                    score = scores[split][rule]
                    added = (baseline[split] == 0) & np.isfinite(score) & (score >= theta)
                    grade = np.where(added, 1, baseline[split]).astype(np.int8)
                    np.testing.assert_array_equal(grade, saved[split][key])
                    np.testing.assert_array_equal(grade == 2, baseline[split] == 2)
                    assert np.all(grade >= baseline[split])
                    report = A.describe(grade, d, refs[split])
                    A.eq(report, metrics[split + '/' + key], 'metrics/' + split + '/' + key)
                    p = report['paired_any']['prior_light']
                    assert all(row['loss'] == 0 and row['later'] == 0 for row in p)
                    reports[(split, key)] = report
                    details[(split, key)] = added_costs(added, d['category'])
                    if split == 'validation':
                        firsts[key] = (A.first(baseline[split] > 0), A.first(grade > 0))
    seen = set()
    with (OUT / 'summary.csv').open(encoding='utf8', newline='') as stream:
        for row in csv.DictReader(stream):
            seed = int(row['seed']);split = row['split']
            key = f"{seed}/{row['rule']}/c{row['clear_cap']}_p{row['pass_cap']}"
            identity = (split, key)
            assert identity not in seen
            seen.add(identity)
            r, b = reports[identity], bases[(split, seed)]
            p = r['paired_any']['prior_light']
            a = details[identity]
            expected = dict(HEAD=r['any']['counts'][0], BODY=r['any']['counts'][1],
                            HEAD_rescue=p[0]['rescue'], BODY_rescue=p[1]['rescue'],
                            HEAD_loss=p[0]['loss'], BODY_loss=p[1]['loss'],
                            HEAD_earlier=p[0]['earlier'], BODY_earlier=p[1]['earlier'],
                            HEAD_later=p[0]['later'], BODY_later=p[1]['later'],
                            clear_slots=r['any']['clear_slots'], pass_clips=r['any']['pass_clips'],
                            extra_total_clear_slots=r['any']['clear_slots']-b['any']['clear_slots'],
                            extra_total_pass_clips=r['any']['pass_clips']-b['any']['pass_clips'],
                            extra_candidate_clear_slots=a['clear']['slots'],
                            extra_candidate_pass_slots=a['pass']['slots'],
                            light_pass_slots=r['joint_costs']['pass']['light']['slots'],
                            light_pass_longest_frames=r['joint_costs']['pass']['light']['longest_run_frames'])
            for name, value in expected.items():
                A.eq(int(row[name]), value, 'summary/' + name)
            assert expected['extra_total_clear_slots'] <= expected['extra_candidate_clear_slots']
            assert expected['extra_total_pass_clips'] <= a['pass']['clips']
    A.eq(len(seen), 54, 'summary count')
    ledger_seen = set();lookup = {int(v):i for i,v in enumerate(ds['validation']['ids'])}
    with (OUT / 'event_ledger.csv').open(encoding='utf8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['key'];i = lookup[int(row['scene'])];k = int(row['replica'])
            q = ('HEAD', 'BODY').index(row['height'])
            identity = (key, i, k, q)
            assert identity not in ledger_seen
            ledger_seen.add(identity)
            before, after = firsts[key]
            A.eq(int(row['before_first']), int(before[i,k,q]), 'ledger/before')
            A.eq(int(row['after_first']), int(after[i,k,q]), 'ledger/after')
            A.eq(row['category'], ds['validation']['category'][i,q], 'ledger/category')
            A.eq(row['family'], ds['validation']['rows'][i]['shape_family'], 'ledger/family')
            assert before[i,k,q] < 0 or (0 <= after[i,k,q] <= before[i,k,q])
    A.eq(len(ledger_seen), 82944, 'ledger count')
    seconds = time.monotonic() - began
    assert seconds <= 60, ('audit budget', seconds)
    result = dict(status='PASS', cells=54, calibrations=27, event_rows=len(ledger_seen),
                  scalar_checks=A.CHECKS, CPU_seconds=seconds, CPU_seconds_cap=60, GPU_seconds=0,
                  cutoff_background_ids=sorted(cutoff_ids), unused_fit_background_ids=sorted(fit_ids),
                  validation_background_ids=sorted(validation_ids),
                  source_sha256=sha(Path(__file__)), producer_source_sha256=sha(producer),
                  shared_independent_metrics_source_sha256=sha(Path(A.__file__)),
                  independence='Independent raw cache smoothing, rank0 window reconstruction, tie-group threshold enumeration, grades, all metrics/groups/timing/segments/longest runs, candidate versus total costs, and ledger. Track geometry/association audited separately.')
    (target / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    (target / 'added_candidate_duration.json').write_text(json.dumps({split+'/'+key:value for (split,key),value in details.items()}, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()

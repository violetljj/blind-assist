"""Independent public-interval oracle audit; no model, simulator or native depth replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

QUERIES = [(x-.3, x+.3, lo, hi) for lo, hi in ((.42, .9), (-.2, .42)) for x in (-.3, 0., .3)]
ARMS = ('A_current', 'local', 'exact_interval', 'sampled_interval', 'native_witness')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_governed(repo, result):
    artifact = (repo/'artifacts.local').resolve()
    path = Path(os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL', '')).resolve()
    if not path.is_relative_to(artifact) or not path.is_file():
        raise ValueError('Use an active research-ue governed RunSpec')
    journal = read(path)
    owned = [(Path(o['path']) if Path(o['path']).is_absolute() else artifact/o['path']).resolve()
             for o in journal.get('outputs', [])]
    if journal.get('schema') != 'blindassist-asset-run-journal-v1' or journal.get('state') != 'running' or result not in owned:
        raise ValueError('Active governed journal does not own this result')
    if result.exists():
        raise FileExistsError('Preserve prior audit result')
    return dict(path=str(path), run_id=journal['id'], sha256=sha(path))


def possible(ax, ay, low, high, query):
    enter, leave = np.maximum(low, .3).copy(), np.minimum(high, 3.).copy()
    for slope, left, right in ((ax, query[0], query[1]), (ay, query[2], query[3])):
        zero = slope == 0
        safe = np.where(zero, 1., slope)
        first, last = left/safe, right/safe
        enter = np.maximum(enter, np.where(zero, -np.inf if left <= 0 <= right else np.inf, np.minimum(first, last)))
        leave = np.minimum(leave, np.where(zero, np.inf if left <= 0 <= right else -np.inf, np.maximum(first, last)))
    return enter <= leave+1e-12


def sample_grid():
    focal = 320/math.tan(math.radians(50))
    slope = np.linspace(-math.tan(math.radians(22.5)), math.tan(math.radians(22.5)), 9)
    xs = np.floor((320+focal*slope)*256/640+.5).astype(int)
    ys = np.floor((180+focal*slope)*192/360+.5).astype(int)
    return np.asarray([y*256+x for r in range(8) for c in range(8)
        for y in np.rint(np.linspace(ys[r], ys[r+1]-1, 4)).astype(int)
        for x in np.rint(np.linspace(xs[c], xs[c+1]-1, 4)).astype(int)])


def recount(rows, arm):
    counts = {key: 0 for key in ('TP', 'FP', 'FN', 'prediction_unknown')}
    clips = defaultdict(list)
    for r in rows:
        truth, pred = r['truth'], r['predictions'][arm]
        if truth is not None and (truth or pred['alert']):
            counts['TP' if truth and pred['alert'] else 'FN' if truth else 'FP'] += 1
        counts['prediction_unknown'] += pred['unknown']
        clips[r['clip_id']].append(r)
    segments, events = 0, []
    for name, clip in sorted(clips.items()):
        clip.sort(key=lambda r: r['frame_in_clip'])
        previous_false, active = False, None
        for r in clip:
            false = r['truth'] is False and r['predictions'][arm]['alert']
            segments += false and not previous_false
            previous_false = false
            if r['truth'] is True:
                if active is None:
                    active = dict(clip_id=name, start_frame=r['frame_in_clip'], first_alert_time_s=None)
                    events.append(active)
                if active['first_alert_time_s'] is None and r['predictions'][arm]['alert']:
                    active['first_alert_time_s'] = r['time_s']
            else:
                active = None
    return dict(counts=counts, false_alert_segment_count=int(segments),
                false_alert_sampled_duration_s=counts['FP']*.2, events=events,
                event_count=len(events), detected_events=sum(e['first_alert_time_s'] is not None for e in events))


def audit(repo: Path, result: Path):
    repo, result = Path(repo).resolve(), Path(result).resolve()
    governance, start = require_governed(repo, result), time.perf_counter()
    evidence = repo/'artifacts.local/evidence'
    support, primary = evidence/'ba-local-support-20260922-run', evidence/'ba-return-correspondence-20260923-run'
    source = evidence/'ba-local-transfer-20260922-evaluated/frame-results.json'
    tof_path = evidence/'ba-local-transfer-20260922-prepared/observations/tof.npy'
    paths = [source, tof_path, support/'frame-diagnostics.json', support/'input-seal.json', support/'output-seal.json',
             primary/'frame-results.json', primary/'metrics.json', primary/'output-seal.json']
    inputs = {str(p.relative_to(repo)): sha(p) for p in paths}
    old, native = read(source), read(support/'frame-diagnostics.json')
    rows, saved = read(primary/'frame-results.json'), read(primary/'metrics.json')
    tof, sealed = np.load(tof_path, allow_pickle=False, mmap_mode='r'), read(support/'output-seal.json')['files']
    checks = 0
    def check(ok, message):
        nonlocal checks
        if not bool(ok):
            raise ValueError(message)
        checks += 1
    check(len(rows) == len(old) == len(native) == 576 and tof.shape == (576, 64, 6), 'Complete frozen source required')
    for p in (source, tof_path):
        check(inputs[str(p.relative_to(repo))] == read(support/'input-seal.json')['input_file_hashes'][p.relative_to(repo).as_posix()], 'Original source hash')
    check(sha(support/'frame-diagnostics.json') == sealed['frame-diagnostics.json'], 'Native-count seal')
    check(sha(support/'input-seal.json') == read(support/'output-seal.json')['input_seal_sha256'], 'Parent input-seal anchor')
    for name, digest in read(primary/'output-seal.json')['files'].items():
        check(sha(primary/name) == digest, 'Primary output seal '+name)
    grid, focal = sample_grid(), 320/math.tan(math.radians(50))
    check(len(grid) == len(np.unique(grid)) == 1024, 'Fixed 4x4 per-zone grid')
    for i, (r, before, prior) in enumerate(zip(rows, old, native)):
        check(r['id'] == before['id'] == prior['id'] and r['index'] == i, 'Frame identity')
        for key, value in before.items():
            check(r[key] == value if key != 'predictions' else all(r[key][a] == p for a, p in value.items()), 'Preserved original '+key)
        path = support/'lineage'/f'frame-{i:04}.npz'
        inputs[str(path.relative_to(repo))] = sha(path)
        check(inputs[str(path.relative_to(repo))] == sealed[f'lineage/frame-{i:04}.npz'], 'Lineage seal')
        with np.load(path, allow_pickle=False) as line:
            native_index, sample = line['native_indices'], line['sampled_indices']
            zones = np.repeat(np.arange(64), np.diff(line['zone_offsets']))
            check(np.array_equal(native_index, (((sample//256*2+1)*360)//384)*640+((sample%256*2+1)*640)//512), 'Independent sample projection')
            public = tof[i].astype(np.float64); distance = public[:, 0]*8
            valid = (public[:, 1] == 1) & (distance >= .1) & (distance < 8) & np.isfinite(distance)
            distance = np.where(valid, distance, 0.)
            radius = .1+3*(.01+.02*distance)
            low, high = np.where(valid, np.maximum(.1, distance-radius), 0.), np.where(valid, distance+radius, 0.)
            check(np.allclose(low, line['interval_low'], atol=1e-12, rtol=0) and np.allclose(high, line['interval_high'], atol=1e-12, rtol=0), 'Original public interval')
            check(valid[zones].all(), 'No unobserved return contributors')
        ax, ay = (native_index%640+.5-320)/focal, (native_index//640+.5-180)/focal
        selected = np.isin(sample, grid)
        possible_by_query = [possible(ax, ay, low[zones], high[zones], query) for query in QUERIES]
        exact = [int(p.sum()) for p in possible_by_query]
        sampled = [int((p & selected).sum()) for p in possible_by_query]
        native_counts = [q['observed_contributor_points'] for q in prior['queries']]
        scores = before['query_probabilities']['local']; winner = 1 if scores[1] >= scores[4] else 4
        check(r['correspondence'] == dict(winner_query=winner, exact_counts=exact, sampled_counts=sampled, native_counts=native_counts), 'Independent correspondence counts')
        for arm, counts in (('exact_interval', exact), ('sampled_interval', sampled), ('native_witness', native_counts)):
            a = before['predictions']['A_current']; alert = bool(a['alert'] or (before['predictions']['local_standalone']['alert'] and counts[winner] > 0))
            check(r['predictions'][arm] == dict(alert=alert, unknown=a['unknown'], ambiguous=alert and a['unknown']), 'Frozen support readout '+arm)
    metrics = {a: recount(rows, a) for a in ARMS}
    report = saved.get('metrics', saved)
    check(report['frames'] == 576 and report['clips'] == 48, 'Full frame/clip denominator')
    for arm, measured in metrics.items():
        target = report['arms'][arm]
        for key, value in measured['counts'].items():
            check(target['frames']['all_known'][key] == value, 'Frame recount '+arm+'/'+key)
        for key in ('false_alert_segment_count', 'event_count', 'detected_events'):
            check(target[key] == measured[key], 'Event recount '+arm+'/'+key)
        check(math.isclose(target['false_alert_sampled_duration_s'], measured['false_alert_sampled_duration_s'], abs_tol=1e-9), 'Inclusive false duration')
        check(len(target['events']) == len(measured['events']) and all(all(a[k] == v for k, v in b.items()) for a, b in zip(target['events'], measured['events'])), 'Onset recount '+arm)
    rescues = [r for r in rows if r['truth'] and r['predictions']['local']['alert'] and not r['predictions']['A_current']['alert']]
    added = [r for r in rows if r['truth'] is False and r['predictions']['local']['alert'] and not r['predictions']['A_current']['alert']]
    check(len(rescues) == 36 and len(added) == 2, 'Frozen rescue/cost denominator')
    check([metrics['native_witness']['counts'][k] for k in ('TP','FP','FN')] == [224,4,32], 'Previous native-XYZ gate parity')
    gate = {a: dict(original_rescues=36, retained=sum(r['predictions'][a]['alert'] for r in rescues),
        original_added_FP=2, removed=sum(not r['predictions'][a]['alert'] for r in added),
        A_preserved=all(not r['predictions']['A_current']['alert'] or r['predictions'][a]['alert'] for r in rows)) for a in ARMS[2:]}
    for arm, value in gate.items():
        value['A_preserved'] &= all(a['first_alert_time_s'] is None or (b['first_alert_time_s'] is not None
            and b['first_alert_time_s'] <= a['first_alert_time_s']+1e-9)
            for a, b in zip(metrics['A_current']['events'], metrics[arm]['events']))
        value['status'] = 'PASS' if value['retained'] >= 32 and value['removed'] == 2 and value['A_preserved'] else 'FAIL'
        check(saved['opportunity'][arm] == value, 'Independent opportunity gate '+arm)
    answer = dict(status='PASS', checks=checks, frames=576, metrics=metrics, gate=gate, governance=governance,
        input_seal=inputs, source_sha256=sha(Path(__file__)), elapsed_s=time.perf_counter()-start,
        backend='CPU NumPy; TASK_NOT_GPU_SUITABLE; no models or simulator',
        limits='Independent ray-slab and fixed-grid recount. Native XYZ witness reuses previously audited counts; no depth replay. No physical ownership or RGB learnability claim.')
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.open('x', encoding='utf-8') as stream:
        json.dump(answer, stream, indent=2, allow_nan=False)
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True); parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args(); audit(args.repo, args.result)

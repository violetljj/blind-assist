"""Consumed depth-frontend comparison; native mode is evaluator-only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import mz101_spatial as m
from run_mz101_spatial import observation_contract, ground_truth, metrics, sha

ROOT = Path(__file__).resolve().parents[4]
TASK = ROOT / 'artifacts.local/work/mz103-depth-frontend-20260912'
PANELS = {
    'mz101': ('mz101-stereo-tof-spatial-20260912', 'comparison-v1', 'stereo-depth', 'common_'),
    'mz102': ('mz102-stereo-support-20260912', 'fresh-v1', 'depth', ''),
}


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def event_compare(candidate, baseline):
    key = lambda e: (e['episode'], e['part'], e['start_s'])
    a = {key(e): e for e in candidate['event_details']}
    lost, delays = [], []
    for b in baseline['event_details']:
        e = a[key(b)]
        if b['detected'] and not e['detected']:
            lost.append(key(b))
        if b['detected'] and e['detected']:
            delays.append(e['first_correct_delay_s'] - b['first_correct_delay_s'])
    return dict(lost_events=lost, max_extra_delay_s=max(delays) if delays else None)


def run(mode, output, frontend_root=None):
    if output.exists() or output.resolve().parent != TASK.resolve():
        raise ValueError('Fresh output directory directly under task root required')
    if mode != 'native' and frontend_root is None:
        raise ValueError('Candidate requires sealed RGB-only frontend root')
    output.mkdir(parents=True)
    start = time.perf_counter()
    frontend = None
    if frontend_root is not None:
        frontend = json.loads((frontend_root / 'receipt.json').read_text())
        assert frontend['status'] == 'PASS'
        for name, digest in frontend['hashes'].items():
            assert sha(frontend_root / name) == digest, name
    results = {}
    for panel, (task, old_dir, depth_dir, prefix) in PANELS.items():
        folder = ROOT / 'artifacts.local/work' / task
        capture, old = folder / 'capture-v1', folder / old_dir
        spec_path = capture / 'spec.json'
        spec = json.loads(spec_path.read_text())
        receipt = json.loads((capture / 'receipt.json').read_text())
        assert receipt['status'] == 'PASS' and receipt['frames'] == 288
        assert spec['rig'] == m.RIG and sha(spec_path) == receipt['spec_sha256']
        for name, digest in receipt['hashes'].items():
            assert sha(capture / name) == digest, name
        obs = observation_contract(spec)
        ids = [o['episode'] for o in obs]
        sealed = np.load(old / 'predictions.npz')
        support = {n: [] for n in ('tof', 'sgbm', 'replacement')}
        hashes, coverage = {}, []
        for o in obs:
            raw = capture / 'frame' / o['id']
            tof = m.tof_points(np.load(raw / 'tof-range.npy'), np.load(raw / 'tof-valid.npy'))
            sgbm = np.load(old / depth_dir / (o['id'] + '.npy'))
            path = (raw / 'native-left-depth.npy' if mode == 'native' else
                    frontend_root / panel / 'depth' / (o['id'] + '.npy'))
            depth = np.load(path)
            assert depth.shape == (m.RIG['height'], m.RIG['width'])
            valid = np.isfinite(depth) & (depth >= m.MIN_DEPTH) & (depth <= m.MAX_DEPTH)
            depth = np.where(valid, depth, np.nan)
            hashes[str(path.relative_to(ROOT))] = sha(path)
            coverage.append(int(valid.sum()))
            for name, points in [('tof', tof), ('sgbm', m.depth_points(sgbm)),
                                 ('replacement', m.depth_points(depth))]:
                support[name].append(m.readout(points, o['pose'], common_fov=True)[0])
        support = {k: np.asarray(v) for k, v in support.items()}
        support['sgbm_union'] = support['tof'] + support['sgbm']
        support['replacement_union'] = support['tof'] + support['replacement']
        pred = {k: m.hysteresis(v, ids) for k, v in support.items()}
        for name, saved in [('tof', 'tof'), ('sgbm', 'stereo'), ('sgbm_union', 'union')]:
            np.testing.assert_array_equal(pred[name], sealed[prefix + saved])
            np.testing.assert_array_equal(support[name], sealed[prefix + saved + '_support'])
        dest = output / panel
        dest.mkdir()
        write(dest / 'observations.json', obs)
        np.savez_compressed(dest / 'predictions.npz', **pred,
                            **{k + '_support': v for k, v in support.items()})
        write(dest / 'prediction-seal.json', dict(mode=mode,
              authority='EVALUATOR_ONLY_REFERENCE' if mode == 'native' else 'FROZEN_RGB_FRONTEND',
              prediction_sha256=sha(dest / 'predictions.npz'), input_hashes=hashes,
              baseline_parity=True, spec_sha256=sha(spec_path)))
        # Evaluation starts only after predictions are fixed. Native mode remains
        # an explicitly evaluator-informed intervention, never a sensor result.
        gt = ground_truth(spec)
        np.testing.assert_array_equal(gt, np.load(old / 'truth.npy'))
        np.save(dest / 'truth.npy', gt)
        scores = {k: metrics(v, gt, obs) for k, v in pred.items()}
        baseline, replacement = scores['sgbm_union'], scores['replacement_union']
        events = event_compare(replacement, baseline)
        gates = dict(fewer_fp=replacement['FP'] < baseline['FP'],
                     no_more_fn=replacement['FN'] <= baseline['FN'],
                     no_lost_events=not events['lost_events'],
                     no_more_false_sessions=replacement['false_sessions'] <= baseline['false_sessions'],
                     timing_within_one_frame=events['max_extra_delay_s'] is None or events['max_extra_delay_s'] <= .25)
        transitions = dict(retained_tp=int((pred['replacement_union'] & pred['sgbm_union'] & gt).sum()),
                           lost_tp=int((~pred['replacement_union'] & pred['sgbm_union'] & gt).sum()),
                           new_tp=int((pred['replacement_union'] & ~pred['sgbm_union'] & gt).sum()))
        slices = {family: {k: metrics(v, gt, obs, [f['family'] == family for f in spec['frames']])
                  for k, v in pred.items()} for family in dict.fromkeys(f['family'] for f in spec['frames'])}
        detail = dict(frames=len(obs), results=scores, gates=gates, event_compare=events,
            transitions=transitions, slices=slices,
            per_part={part: {k: dict(TP=int((v[:,j] & gt[:,j]).sum()),
                FP=int((v[:,j] & ~gt[:,j]).sum()), FN=int((~v[:,j] & gt[:,j]).sum()))
                for k, v in pred.items()} for j, part in enumerate(m.PARTS)},
            support_fp={k: dict(current=int((v & ~gt & (support[k] > 0)).sum()),
                                held=int((v & ~gt & (support[k] == 0)).sum())) for k, v in pred.items()},
            unknown_query_frames={k: int((v == 0).sum()) for k, v in support.items()},
            mean_eligible_replacement_pixels=float(np.mean(coverage)))
        write(dest / 'summary.json', detail)
        results[panel] = detail
        print(panel, json.dumps({k: {x: v[x] for x in ('TP','FP','FN','F1','missed_events')}
              for k, v in scores.items()}), flush=True)
    pooled = {k: {x: sum(p['results'][k][x] for p in results.values())
              for x in ('TP', 'FP', 'FN', 'missed_events', 'false_sessions')}
              for k in ('tof', 'sgbm_union', 'replacement_union')}
    for value in pooled.values():
        value['F1'] = 2 * value['TP'] / (2 * value['TP'] + value['FP'] + value['FN'])
    headroom = (pooled['replacement_union']['FP'] < pooled['sgbm_union']['FP'] and
                pooled['replacement_union']['FN'] <= pooled['sgbm_union']['FN'] and
                all(not p['event_compare']['lost_events'] for p in results.values()))
    summary = dict(mode=mode, frames=576, evidence='CONSUMED_DEVELOPMENT',
        pooled_descriptive=pooled, headroom=headroom,
        candidate_pass=all(all(p['gates'].values()) for p in results.values()),
        panels={k: dict(gates=v['gates'], transitions=v['transitions']) for k,v in results.items()},
        backend=dict(device='CPU', reason='TASK_NOT_GPU_SUITABLE', scope='fixed geometry and evaluator'),
        frontend=frontend, elapsed_s=time.perf_counter()-start,
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(output / 'summary.json', summary)
    write(output / 'receipt.json', dict(status='PASS', hashes={str(p.relative_to(output)): sha(p)
          for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['native', 'foundation'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frontend-root', type=Path)
    args = parser.parse_args()
    run(args.mode, args.output, args.frontend_root)

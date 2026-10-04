"""CPU-only recency proxy audit of sealed, already-computed dose results.

Read only after the new coverage-cue PLAN is frozen. No model, renderer, Torch,
raw-score calibration, or new inference is imported. Geometric target first-ray
counts are evaluator truth, never inputs to a cue policy.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / 'artifacts.local/work/cnh-head-recenter-dose-20261005'
OUTPUT = ROOT / 'artifacts.local/work/cnh-coverage-cue-geometry-20261005'
DOSES = ('none', '2.5', '2.1', '1.7', '1.3', '1.0', 'zero')
ARMS = ('M3', 'VD')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')


def counts(rows):
    return {arm: dict(timely=sum(r[arm]['timely'] for r in rows),
                      n=sum(r[arm]['n'] for r in rows)) for arm in ARMS}


def run(input_dir=INPUT, output_dir=OUTPUT):
    tick = time.monotonic()
    plan_path = output_dir / 'PLAN.json'
    # The PLAN existence and hash are recorded before numerical payload access.
    plan = read(plan_path)
    plan_hash = sha(plan_path)
    assert isinstance(plan, dict) and plan, 'Nonempty frozen PLAN required'
    input_paths = [input_dir / name for name in ('geometry.json', 'sequences.json')]
    geometry, sequences = [read(p) for p in input_paths]
    geoms = {(g['unit'], g['dose']): g for g in geometry if g['group'] == 'opposite'}
    units = sorted({u for u, _ in geoms})
    assert len(units) == 11
    assert set(geoms) == {(u, d) for u in units for d in DOSES}
    records = {(s['unit'], s['dose'], s['variant'], s['replica'], s['arm']): s
               for s in sequences if s['group'] == 'opposite' and s['variant'] < 2}
    assert len(records) == 11 * 7 * 2 * 4 * 2
    trajectories = []
    for unit in units:
        for variant in (0, 1):
            for dose in DOSES:
                geom = geoms[unit, dose]
                ranges = np.asarray(geom['ranges_m'], dtype=float)
                ray_counts = np.asarray(geom['counts'], dtype=int)
                assert ranges.shape == (16,) and ray_counts.shape == (7, 16)
                assert np.all(ray_counts >= 0)
                window = (ranges >= .9 - 1e-9) & (ranges <= 1.4 + 1e-9)
                visible = ray_counts[variant] > 0
                n_vis = int(np.count_nonzero(window & visible))
                row = dict(unit=unit, variant=variant, dose=dose, n_vis=n_vis,
                           visible_window_frames=np.flatnonzero(window & visible).tolist(),
                           visible_window_ranges_m=ranges[window & visible].tolist(),
                           exposure_ray_frames=int(ray_counts[variant, window].sum()))
                for arm in ARMS:
                    replicas = [records[unit, dose, variant, k, arm] for k in range(4)]
                    for seq in replicas:
                        frame = seq['first_frame']
                        assert seq['stopped'] == (frame is not None)
                        reconstructed = frame is not None and ranges[frame] >= .9 - 1e-9
                        assert bool(seq['timely']) == reconstructed
                        if frame is not None:
                            assert abs(ranges[frame] - seq['first_range_m']) < 1e-9
                            assert bool(seq['visible_at_first']) == bool(visible[frame])
                    row[arm] = dict(timely=sum(s['timely'] for s in replicas), n=4,
                                    timely_by_replica=[s['timely'] for s in replicas])
                trajectories.append(row)
    by_n_vis = []
    for n in sorted({r['n_vis'] for r in trajectories}):
        rows = [r for r in trajectories if r['n_vis'] == n]
        by_n_vis.append(dict(n_vis=n, trajectory_dose_n=len(rows),
                              scenes=len({r['unit'] for r in rows}), arms=counts(rows)))
    by_dose = []
    for dose in DOSES:
        rows = [r for r in trajectories if r['dose'] == dose]
        by_dose.append(dict(dose=dose, trajectory_n=len(rows),
                            n_vis_histogram={str(n): sum(r['n_vis'] == n for r in rows)
                                             for n in sorted({r['n_vis'] for r in rows})},
                            arms=counts(rows)))
    gate_reasons = []
    gate_groups = {}
    for label, rows in [('n_vis_ge3', [r for r in trajectories if r['n_vis'] >= 3]),
                        ('n_vis_0', [r for r in trajectories if r['n_vis'] == 0])]:
        gate_groups[label] = counts(rows)
        for arm, cell in gate_groups[label].items():
            cell['rate'] = cell['timely'] / cell['n'] if cell['n'] else None
            if not cell['n']:
                gate_reasons.append(f'{label} {arm}: no evaluable denominator')
            elif label == 'n_vis_ge3' and cell['rate'] < .9:
                gate_reasons.append(f'{arm} n_vis>=3 timely rate <0.9')
            elif label == 'n_vis_0' and cell['rate'] >= .8:
                gate_reasons.append(f'{arm} n_vis=0 timely rate >=0.8')
    status = 'STOP_PROXY_INCONSISTENT' if gate_reasons else 'PASS_PROXY_LIMITED'
    result = dict(status=status, run='CNH_COVERAGE_CUE_GEOMETRY_20261005_STEP1',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Consumed simulated Development; true target first-ray visibility proxy audit',
                  definition='n_vis: target first-hit ray count >0 at sampled front ranges in inclusive [0.9,1.4]m; one geometry shared by K4; timely iff stored first alarm front range >=0.9m.',
                  statistical_unit='22 trajectories / 11 scenes, repeated across 7 doses; K4 is photon AND pose replicas, not independent geometry.',
                  limitations='This target-level first-ray proxy is not a calibration of generic edge-band cell visibility, occlusion, exposure strength, or all head-motion regimes. No two-frame efficacy claim without samples.',
                  plan_sha256=plan_hash, source_sha256=sha(Path(__file__)),
                  input_sha256={p.name: sha(p) for p in input_paths},
                  n_per_arm=616, gate_definition='STOP iff either arm timely rate n_vis>=3 <0.9, or n_vis=0 >=0.8; absent gate denominator is not evaluable.',
                  gate_groups=gate_groups, gate_reasons=gate_reasons,
                  by_n_vis=by_n_vis, by_dose=by_dose, trajectories=trajectories,
                  two_frame_trajectory_dose_n=sum(r['n_vis'] == 2 for r in trajectories),
                  elapsed_s=time.monotonic() - tick)
    assert sha(plan_path) == plan_hash, 'PLAN changed during gate'
    save(output_dir / 'step1_gate.json', result)
    lines = ['| n_vis | 轨迹×剂量 | M3及时/分母 | VD及时/分母 |',
             '|---|---:|---:|---:|']
    for cell in by_n_vis:
        m, v = cell['arms']['M3'], cell['arms']['VD']
        lines.append(f"| {cell['n_vis']} | {cell['trajectory_dose_n']} | {m['timely']}/{m['n']} | {v['timely']}/{v['n']} |")
    lines += ['', f"Gate: {status}; 两帧样本轨迹×剂量数: {result['two_frame_trajectory_dose_n']}。", '',
              '统计单位是22条几何轨迹/11个场景；同一几何的K4包含光子和位姿噪声，7种剂量为重复观测。', '',
              '| 剂量 | n_vis:轨迹数 | M3及时/88 | VD及时/88 |', '|---|---|---:|---:|']
    for cell in by_dose:
        m, v = cell['arms']['M3'], cell['arms']['VD']
        hist = ', '.join(f'{k}:{n}' for k, n in cell['n_vis_histogram'].items())
        lines.append(f"| {cell['dose']} | {hist} | {m['timely']}/{m['n']} | {v['timely']}/{v['n']} |")
    (output_dir / 'step1_gate_table.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.input, args.output)

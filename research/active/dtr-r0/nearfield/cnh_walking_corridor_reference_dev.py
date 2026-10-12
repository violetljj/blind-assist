"""Cached BlindWays pelvis corridor references, with a declared nominal 60Hz clock.

No clock/orientation/online position is recovered. Files remain separate segments;
future paths are evaluation-only. Source data is not redistributed by this script.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cnh_direction_corridor_reference_dev import corridor_reference


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/downloads/blindways-20261007/Motion'
OUT = ROOT/'artifacts.local/work/cnh-walking-corridor-dev-20261010'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def metrics(columns, mask):
    def count(field, value):
        return int(np.count_nonzero(mask & (columns[field] == value)))
    geometry = mask & (columns['future_status'] == 'AVAILABLE')
    paired = mask & (columns['status'] == 'AVAILABLE')
    error = columns['wrapped_error_deg'][paired]
    deviations = columns['max_future_center_distance_to_chord_m'][geometry]
    return dict(rows=int(mask.sum()), past_missing=count('past_status', 'NOT_AVAILABLE_PAST_WINDOW'),
                past_low_motion=count('past_status', 'LOW_MOTION_PAST'),
                future_missing=count('future_status', 'NOT_AVAILABLE_FUTURE_WINDOW'),
                future_low_motion=count('future_direction_status', 'LOW_MOTION_FUTURE_CHORD'),
                future_geometry=int(geometry.sum()), paired_directions=int(paired.sum()),
                error_rms_deg=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                error_abs_median_deg=float(np.median(np.abs(error))) if len(error) else None,
                error_abs_p95_deg=float(np.percentile(np.abs(error), 95)) if len(error) else None,
                center_chord_deviation_p95_m=float(np.percentile(deviations, 95)) if len(deviations) else None,
                center_chord_deviation_max_m=float(deviations.max()) if len(deviations) else None,
                center_chord_deviation_gt_030m=int(np.count_nonzero(deviations > .30)))


def summarize(columns):
    rows = []
    participants = ['ALL'] + sorted(set(columns['participant'].tolist()))
    for participant in participants:
        participant_mask = np.ones(len(columns['horizon_s']), bool) if participant == 'ALL' else columns['participant'] == participant
        for horizon in (.5, 1., 1.5):
            base = participant_mask & (columns['horizon_s'] == horizon)
            for subset in ('all', 'past_displacement_ge_030m'):
                mask = base if subset == 'all' else base & (columns['past_displacement_m'] >= .30)
                rows.append(dict(participant=participant, horizon_s=horizon, subset=subset, **metrics(columns, mask)))
    return rows


def main(source, out, max_seconds):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'reference_pack.npz').exists():
        raise FileExistsError('Preserve prior reference pack')
    plan = json.loads((out/'PLAN.json').read_text(encoding='utf-8-sig'))
    assert plan['fixed']['sample_hz'] == 60 and plan['fixed']['horizons_s'] == [.5, 1, 1.5]
    manifest = []; blocks = []; paths = []; xyz = []; times = []; offsets = [0]; audits = []
    files = sorted(source.glob('*.npy'))
    if not files:
        raise ValueError('No cached source files')
    nominal_times = np.rint(np.arange(600)*1e9/60).astype(np.int64)
    for path in files:
        if time.monotonic()-started >= max_seconds:
            write_json(out/'run_receipt.json', dict(status='BUDGET_STOP', processed=len(manifest), seconds=time.monotonic()-started))
            raise TimeoutError('Frozen runner wall time limit')
        raw = path.read_bytes()
        motion = np.load(path, allow_pickle=False)
        if motion.shape != (600, 24, 3) or not np.isfinite(motion).all():
            raise ValueError(f'Unexpected source shape or nonfinite positions: {path.name}')
        result = corridor_reference(nominal_times, motion[:, 0, :], horizontal_axes=(0, 1), sample_hz=60)
        c = result['columns']; n = len(c['status'])
        c['clip'] = np.full(n, path.stem)
        c['participant'] = np.full(n, path.stem.split('_')[0])
        blocks.append(c)
        assert len(result['segments']) == 1 and len(result['invalid_edges']['edge_after_index']) == 0
        assert n == 30 and np.array_equal(np.unique(c['anchor_index']), np.arange(0, 600, 60))
        local_offsets = result['future_path_offsets']
        for index in range(n):
            lo, hi = local_offsets[index:index+2]
            offsets.append(offsets[-1]+int(hi-lo))
            if hi > lo:
                assert result['future_path_timestamps_ns'][lo] == c['anchor_timestamp_ns'][index]
                assert result['future_path_timestamps_ns'][hi-1] == c['future_window_end_ns'][index]
                assert np.array_equal(result['future_path_positions'][lo], motion[c['anchor_index'][index], 0])
            else:
                assert c['future_status'][index] == 'NOT_AVAILABLE_FUTURE_WINDOW'
        paths.append(result['future_path_xy']); xyz.append(result['future_path_positions']); times.append(result['future_path_timestamps_ns'])
        manifest.append(dict(file=path.name, participant=path.stem.split('_')[0], bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), shape=list(motion.shape), dtype=str(motion.dtype)))
        audits.append(dict(clip=path.stem, rows=n, anchors=10, missing_future=int(np.sum(c['future_status'] != 'AVAILABLE'))))
    columns = {name: np.concatenate([block[name] for block in blocks]) for name in blocks[0]}
    pack = dict(columns, future_path_offsets=np.array(offsets, np.int64), future_path_xy=np.concatenate(paths),
                future_path_positions=np.concatenate(xyz), future_path_timestamps_ns=np.concatenate(times))
    np.savez_compressed(out/'reference_pack.npz', **pack)
    with (out/'reference_windows.csv').open('w', encoding='utf8', newline='') as handle:
        writer = csv.writer(handle); names = list(columns); writer.writerow(names)
        writer.writerows(zip(*(columns[name] for name in names)))
    write_json(out/'source_manifest.json', dict(source=str(source.relative_to(ROOT)).replace('\\', '/'), files=manifest,
        receipt_sha256=hashlib.sha256((source.parent/'RECEIPT.json').read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(corridor_reference.__code__.co_filename).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256((out/'PLAN.json').read_bytes()).hexdigest(),
        source_units='metres assumed consistently with previous analysis; primary metric unit statement not verified; no independent physical calibration',
        coordinate_contract='joint0 Pelvis per author visualizer; rawXY plane and Z-up inherited cached contract, no independent gravity calibration',
        timestamps='nominal relative60Hz rounded frame clock; no recorded acquisition/arrival timestamps',
        config=result['configuration']))
    summary = dict(clips=len(files), participants=sorted(set(columns['participant'].tolist())),
                   frames=len(files)*600, anchors=len(files)*10, windows=len(columns['status']),
                   future_path_points=int(offsets[-1]), summaries=summarize(columns))
    write_json(out/'reference_summary.json', summary)
    write_json(out/'structure_check.json', dict(status='PASS', clips=len(audits), windows=summary['windows'],
        assertions=['source600x24x3 finite', 'separate clip segments', '30 rows and10 native anchors per clip', 'full path offsets and exact endpoints', 'tail censor retained']))
    write_json(out/'run_receipt.json', dict(status='PASS', seconds=time.monotonic()-started, max_seconds=max_seconds, clips=len(files),
        gpu_seconds=0, training=0, inference=0, source_changed=False))
    print(json.dumps(dict(clips=summary['clips'], windows=summary['windows'], seconds=time.monotonic()-started)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--max-seconds', type=float, default=120)
    args = parser.parse_args()
    main(args.source, args.output, args.max_seconds)

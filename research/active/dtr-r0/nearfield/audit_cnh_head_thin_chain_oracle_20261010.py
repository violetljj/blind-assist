"""Cheap independent cached-score/retention/projection and continuous-FOV audit."""
import argparse
import ast
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work'
FROZEN = WORK/'cnh-frozen-e2e-20261010'
MAPS = WORK/'cnh-bar-local-readout-dev-20261009'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def projection_ast(path):
    parsed = ast.parse(Path(path).read_text(encoding='utf8'))
    return ast.dump(next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Projection'))


def audit(out):
    start = time.monotonic()
    physics = read(out/'audit_receipt.json')
    assert physics['status'] == 'PASS'
    chain = read(out/'chain_receipt.json')
    assert chain['status'] == 'COMPLETE'
    assert sha(Path(__file__).with_name('cnh_head_thin_chain_oracle_20261010.py')) == chain['source_sha256']
    for path, expected in chain['inputs_sha256'].items():
        assert sha(path) == expected, ('chain input SHA', path)
    binding = chain['map_configuration_source_sha256']
    for path, record in binding.items():
        if isinstance(record, str):
            assert sha(ROOT/path) == record
        else:
            assert sha(MAPS/'source_snapshot.py') == record['archived_sha256']
            assert sha(ROOT/path) == record['current_sha256']
            assert projection_ast(ROOT/path) == projection_ast(MAPS/'source_snapshot.py')
            assert record['Projection_class_AST_equal']
    with np.load(out/'chain_expectations.npz', allow_pickle=False) as a:
        ids, sensor, query, ambient = a['scene_ids'], a['sensor'], a['public_query'], a['ambient']
        target, present, direct = a['expected_target'], a['expected_present'], a['direct']
    metadata = read(FROZEN/'scene_rows.json')['hold']
    selected = [metadata[int(i)] for i in ids]
    assert len(ids) == 32 and len(set(ids.tolist())) == 32
    assert all(r['group'] == 0 and r['placement'] == 'contact' and r['shape_family'] == 'horizontal' for r in selected)
    with np.load(FROZEN/'data/hold/scores.npz', allow_pickle=False) as a:
        raw = a['m3_raw'][ids]
    smoothed = np.empty_like(raw, dtype=float)
    for t in range(13):
        smoothed[..., t, :] = np.average(raw[..., max(0, t-4):t+1, :].astype(float),
            axis=-2, weights=np.exp2(np.arange(min(t+1, 5))))
    records = csv_read(out/'chain_frames.csv')
    events = csv_read(out/'chain_events.csv')
    accum = csv_read(out/'chain_accumulation.csv')
    assert len(records) == 32*2*13 and len(events) == 64
    lookup = {(int(r['scene']), int(r['replica']), int(r['frame'])): r for r in records}
    sizecounts = {}
    for size in (0, 1):
        keep = np.asarray([r['size_variant'] == size for r in selected])
        scores = smoothed[keep, :, :, 0]
        timely, anyhit = (scores[:, :, :11] >= .8557642486787612).any(2), (scores >= .8557642486787612).any(2)
        current = [r for r in events if int(r['size_variant']) == size]
        outcomes = dict(timely=int(timely.sum()), late=int((anyhit & ~timely).sum()), silent=int((~anyhit).sum()))
        assert len(current) == scores.shape[0]*scores.shape[1] == 32
        assert all(sum(r['outcome'] == name for r in current) == count for name, count in outcomes.items())
        support = sum(any(int(lookup[int(r['scene']), int(r['replica']), f]['direct_top8_bins']) > 0
                          for f in range(3, 14)) for r in current)
        assert support == sum(int(r['target_compatible_top8_timely_frames']) > 0 for r in current)
        sizecounts[str(size)] = dict(denominator=32, unique_scenes=int(keep.sum()),
            target_compatible_top8_timely=support, **outcomes)
    assert sizecounts['0']['timely'] == 3 and sizecounts['0']['target_compatible_top8_timely'] == 27
    assert sizecounts['1']['timely'] == 21
    for row in records:
        def number(name):
            return float(row[name]) if row[name] else None
        chain_fraction = [number(n) for n in ('top8_energy_fraction', 'local_peak_energy_fraction',
                                             'positive_gate_energy_fraction', 'gate_energy_fraction')]
        if all(v is not None for v in chain_fraction):
            assert all(x <= y+1e-9 for x, y in zip(chain_fraction, chain_fraction[1:]))
        for name in ('direct_top8_bins', 'direct_local_peak_bins', 'direct_positive_bins', 'direct_gated_bins'):
            assert int(row[name]) <= int(row['direct_bins'])
    from cnh_cvr_projection import query_masks
    mask = query_masks()[0].ravel().astype(float)
    bias = np.load(WORK/'cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy').astype(np.float32)
    denominator = np.sqrt(np.maximum(16*ambient.astype(np.float32)[..., None]+np.maximum(bias, 0), 1e-9)).reshape(16, 1024)
    t = target.reshape(32, 16, 1024)/denominator
    projected_verified = 0
    # Prespecified one thin and one thick scene, early and timely end frames.
    scene_indices = [int(np.flatnonzero([r['size_variant'] == size for r in selected])[0]) for size in (0, 1)]
    for frame in (3, 13):
        begin = max(0, frame-7)
        matrix = sparse.load_npz(MAPS/f'map_f{frame}.npz').tocsr()
        sensitivity = np.asarray(mask@matrix).reshape(frame-begin+1, 1024)
        for i in scene_indices:
            row = lookup[int(ids[i]), 0, frame]
            expected_current = float(np.dot(t[i, frame], sensitivity[-1]))
            expected_history = float(np.sum(t[i, begin:frame+1]*sensitivity))
            expected_via_voxels = float(np.dot(np.asarray(matrix@t[i, begin:frame+1].ravel()).ravel(), mask))
            np.testing.assert_allclose(expected_history, expected_via_voxels, rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(float(row['projected_target_current_mass']), expected_current, rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(float(row['projected_target_past8_mass']), expected_history, rtol=1e-12, atol=1e-10)
            assert all(float(lookup[int(ids[i]), k, frame]['projected_target_past8_mass']) == float(row['projected_target_past8_mass']) for k in (0, 1))
            for window in (1, 2, 4, 8):
                selectedrow = next(r for r in accum if int(r['scene']) == int(ids[i]) and int(r['frame']) == frame and int(r['requested_window']) == window)
                wbegin = max(begin, frame-window+1)
                expected = np.sum(t[i, wbegin:frame+1]*sensitivity[wbegin-begin:])
                np.testing.assert_allclose(float(selectedrow['target_projected_query_mass']), expected, rtol=1e-12, atol=1e-10)
                assert float(selectedrow['target_count_channel_delta']) == 0
                projected_verified += 1
    # Continuous square-cone exclusion, independent of subray midpoint sampling.
    geometries = read(out/'geometries.json')
    with np.load(out/'physical.npz', allow_pickle=False) as a:
        occupancy, endpoints = a['occupancy'], a['endpoints']
    fov_rows = []
    halfcone = np.tan(np.deg2rad(22.5))
    for g in geometries:
        corners = np.asarray(list(itertools.product(*zip(g['target_box']['lo'], g['target_box']['hi']))))
        camera = (corners-sensor[15, :3, 3])@sensor[15, :3, :3]
        plane_exclusion = max(float((camera[:, 0]-halfcone*camera[:, 2]).min()),
                              float((-camera[:, 0]-halfcone*camera[:, 2]).min()))
        if g['thickness_m'] in (.013, .017):
            assert plane_exclusion > 0
            assert occupancy[g['geometry_id'], 15].sum() == 0
            np.testing.assert_array_equal(endpoints[g['geometry_id'], 0, 15], endpoints[g['geometry_id'], 1, 15])
        fov_rows.append(dict(geometry_id=g['geometry_id'], thickness_m=g['thickness_m'], frame=15,
            continuous_horizontal_exclusion_mm=plane_exclusion*1000,
            min_horizontal_angle_deg=float(np.degrees(np.arctan(np.abs(camera[:, 0])/camera[:, 2])).min()),
            sub16_zone_occupancy_sum=float(occupancy[g['geometry_id'], 15].sum())))
    seconds = time.monotonic()-start
    assert seconds+physics['seconds'] < 120
    result = dict(status='PASS', seconds=seconds, physics_audit_seconds=physics['seconds'],
        cumulative_audit_internal_seconds=seconds+physics['seconds'], source_sha256=sha(Path(__file__)),
        cached_m3_and_top8_counts=sizecounts, fraction_chain_rows_verified=len(records),
        independent_projection_window_pairs_verified=projected_verified,
        f15_continuous_fov=fov_rows,
        source_binding='Fixed cvr/diagnostic hashes; original local map snapshot hash retained; current Projection class AST identical. Whole local file differs due subsequent scan fix; not all-current-file hash identity.',
        limits=['Target-compatible bin may include background/noise; no exact target attribution to observed peaks',
            'Expected sparse map FP16 and signedlog descriptors are not actual CUDA voxel reconstruction or M3 network attribution',
            'Reference-normalized target z and native observation SNR have distinct denominators; learned M3 logits/local patch scores have other units',
            'Thin vs thick original groups also differ width/placement; chain is not an isolated thickness experiment',
            'At f15 1.3/1.7cm boxes provably leave continuous FOV; 3cm sub16 misses tiny continuous overlap; 8.5cm remains nonzero'],
        model_forward=0, training=0, new_noise=0, protected_access=0)
    save_new(out/'chain_audit_receipt.json', result)
    print('CHAIN_AUDIT_PASS', json.dumps(sizecounts), 'seconds', round(seconds, 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=WORK/'cnh-head-thin-signal-oracle-dev-20261010')
    args = parser.parse_args()
    audit(args.out.resolve())

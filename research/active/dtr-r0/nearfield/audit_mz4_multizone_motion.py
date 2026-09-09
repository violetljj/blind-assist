"""Independent NumPy geometry and integer-bitset audit of MZ4, no Torch inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def physical_labels(centers, pairs):
    # Independent rectangle intersection against the frozen physical query bands.
    per_patch = []
    for y, z in centers:
        y0, y1 = [2.5 * np.tan(np.radians(v)) for v in (y - 1, y + 1)]
        z0, z1 = [1.7 + 2.5 * np.tan(np.radians(v)) for v in (z - 1, z + 1)]
        per_patch.append([y0 <= .28 and y1 >= -.28 and z0 <= 1.4 and z1 >= .65,
                          y0 <= .18 and y1 >= -.18 and z0 <= 1.85 and z1 >= 1.4])
    per_patch = np.array(per_patch)
    return np.concatenate((np.zeros((1, 2), bool), per_patch,
                            per_patch[pairs[:, 0]] | per_patch[pairs[:, 1]]))


def verify_geometry(data, geometry):
    y, x = np.indices((360, 640))
    f = 320 / np.tan(np.radians(50))
    rays = np.stack((np.ones_like(x), (x - 319.5) / f, -(y - 179.5) / f), -1)
    weights = (rays ** 2).sum(-1) ** -1.5
    angles = np.degrees(np.arctan(rays[..., 1:]))
    patch_pixels = [np.flatnonzero(((angles >= c - 1) & (angles < c + 1)).all(-1))
                    for c in data['centers']]
    perturb = np.random.default_rng(23).uniform(-.25, .25, (25, 2))
    times = np.linspace(0, 2, 25)
    pitches = 2 * np.sin(4 * np.pi * times) + perturb[:, 0]
    yaws = np.sin(2 * np.pi * times) + perturb[:, 1]
    max_error, checks = 0., 0
    for bias in (0., .5):
        prefix = '' if bias == 0 else 'biased_'
        for step in (0, 4, 8, 12, 20, 24):
            p, a = np.radians([pitches[step], yaws[step] + bias])
            # Sensor basis vectors expressed independently in the fixed body frame.
            forward = [np.cos(a) * np.cos(p), np.sin(a) * np.cos(p), np.sin(p)]
            right = [-np.sin(a), np.cos(a), 0.]
            up = [-np.cos(a) * np.sin(p), -np.sin(a) * np.sin(p), np.cos(p)]
            sensor = np.stack([np.einsum('...i,i->...', rays, axis)
                               for axis in (forward, right, up)], -1)
            angular = np.degrees(np.arctan2(sensor[..., 1:], sensor[..., :1]))
            inside = ((angular >= -22.5) & (angular < 22.5)).all(-1) & (sensor[..., 0] > 0)
            bins = np.floor((angular + 22.5) / 5.625).astype(int)
            zones = np.where(inside, 8 * bins[..., 1] + bins[..., 0], -1)
            assert np.array_equal(zones, geometry[prefix + 'zone_maps'][step])
            denominator = np.bincount(zones[inside], weights=weights[inside], minlength=64)
            assert np.allclose(denominator, geometry[prefix + 'denominators'][step], atol=1e-10, rtol=1e-12)
            coverage = np.stack([np.bincount(zones.ravel()[pixels],
                        weights=weights.ravel()[pixels], minlength=64) / denominator
                        for pixels in patch_pixels], 1)
            error = float(np.max(np.abs(coverage - geometry[prefix + 'single_coverage'][step])))
            max_error = max(error, max_error)
            assert error < 1e-10
            pairs = data['pairs']
            all_coverage = np.concatenate((np.zeros((64, 1)), coverage,
                                             coverage[:, pairs[:, 0]] + coverage[:, pairs[:, 1]]), 1)
            expected = data['codes' if bias == 0 else 'biased_codes'][step]
            assert np.array_equal(all_coverage >= .02, expected)
            checks += expected.size
    return dict(poses_per_orientation_model=6, code_checks=checks, max_coverage_error=max_error)


class BitsetInference:
    def __init__(self, codes, labels):
        flat = codes.reshape(-1, codes.shape[-1])
        self.full = (1 << len(labels)) - 1
        self.positive = [int.from_bytes(np.packbits(labels[:, q], bitorder='little').tobytes(), 'little')
                         for q in range(2)]
        self.bits = [int.from_bytes(row.tobytes(), 'little') for row in
                      np.packbits(flat, axis=1, bitorder='little')]

    def run(self, observations, valid, selected):
        obs, keep = observations.reshape(len(selected), -1), valid.reshape(len(selected), -1)
        states, counts, included = [], [], []
        for case, row in enumerate(obs):
            feasible = self.full
            for k in np.flatnonzero(keep[case]):
                feasible &= self.bits[k] if row[k] else self.full ^ self.bits[k]
                if not feasible:
                    break
            n = feasible.bit_count()
            count = [(feasible & p).bit_count() for p in self.positive]
            states.append([2 if n == 0 or 0 < c < n else int(c == n) for c in count])
            counts.append(n)
            included.append(bool(feasible & (1 << int(selected[case]))))
        return np.array(states), np.array(counts), np.array(included)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    run = args.run
    result = json.loads((run / 'result.json').read_text())
    for name, expected in result['artifact_hashes'].items():
        assert digest(run / name) == expected
    root = Path(__file__).resolve().parents[4]
    for name, expected in result['input_hashes'].items():
        assert digest(root / name) == expected
    data, geo = np.load(run / 'observations.npz'), np.load(run / 'geometry.npz')
    labels = physical_labels(data['centers'], data['pairs'])
    assert np.array_equal(labels, data['labels'])
    geometry_checks = verify_geometry(data, geo)
    selected, valid, codes = data['selected'], data['valid'], data['codes']
    truth = labels[selected]
    for law in ('nearest_supported', 'background_only'):
        candidate = codes if law == 'nearest_supported' else np.zeros_like(codes)
        stationary = np.broadcast_to(candidate[:1], candidate.shape)
        obs_moving = np.moveaxis(candidate[..., selected], -1, 0)
        obs_static = np.moveaxis(stationary[..., selected], -1, 0)
        methods = dict(single=(candidate[:1], obs_static[:, :1], valid[:, :1]),
                       stationary=(stationary, obs_static, valid),
                       moving_joint=(candidate, obs_moving, valid))
        if law == 'nearest_supported':
            methods.update(unaligned=(stationary, obs_moving, valid),
                           biased_yaw=(data['biased_codes'], obs_moving, valid))
        for method, (template, obs, keep) in methods.items():
            state, count, included = BitsetInference(template, labels).run(obs, keep, selected)
            key = law + '__' + method
            assert np.array_equal(state, data['states_' + key]), key
            assert np.array_equal(count, data['counts_' + key]), key
            assert np.array_equal(included, data['included_' + key]), key
            reported = result['results'][law + '/' + method]
            exact = sum(all(s != 2 and bool(s) == bool(t) for s, t in zip(row, target))
                        for row, target in zip(state, truth))
            assert exact == reported['exact_known']
            assert sum(n == 0 for n in count) == reported['empty']
            assert sum(2 in row for row in state) == reported['any_unknown']
            assert float(np.median(count)) == reported['median_feasible']
            for q, name in enumerate(('BODY', 'HEAD')):
                pairs = list(zip(state[:, q], truth[:, q]))
                expected = dict(positive_total=sum(bool(t) for _, t in pairs),
                    resolved_positive=sum(s == 1 and t for s, t in pairs),
                    false_positive=sum(s == 1 and not t for s, t in pairs),
                    false_negative=sum(s == 0 and t for s, t in pairs),
                    unknown=sum(s == 2 for s, _ in pairs))
                assert expected == reported[name], (key, name)
            print('Verified', law, method, flush=True)
        if law == 'nearest_supported':
            fs, fc = [], []
            for step in range(25):
                state, count, inc = BitsetInference(candidate[step:step + 1], labels).run(
                    obs_moving[:, step:step + 1], valid[:, step:step + 1], selected)
                assert inc.all()
                fs.append(state)
                fc.append(count)
            frames, counts = np.stack(fs, 1), np.stack(fc, 1)
            assert np.array_equal(frames, data['frame_states'])
            assert np.array_equal(counts, data['frame_counts'])
            yes, no = (frames == 1).any(1), (frames == 0).any(1)
            assert not (yes & no).any()
            state = np.where(yes, 1, np.where(no, 0, 2))
            assert np.array_equal(state, data['states_nearest_supported__moving_framewise'])
            reported = result['results']['nearest_supported/moving_framewise']
            assert int(((state != 2).all(1) & (state == truth).all(1)).sum()) == reported['exact_known']
    joint = data['states_nearest_supported__moving_joint']
    for name, expected in result['moving_joint_changes'].items():
        baseline = data['states_nearest_supported__' + name]
        assert expected == dict(
            new_resolved_queries=int(((baseline == 2) & (joint != 2)).sum()),
            lost_resolved_queries=int(((baseline != 2) & (joint == 2)).sum()),
            newly_both_known=int(((baseline == 2).any(1) & (joint != 2).all(1)).sum()))
    audit = dict(status='PASS', method='Independent NumPy geometry and Python integer bitsets; no Torch import',
                  geometry=geometry_checks, audited_sequence_rows=8 * 500, framewise_rows=25 * 500,
                  query_labels_checked=len(labels) * 2, seconds=time.perf_counter() - start,
                  result_sha256=digest(run / 'result.json'), audit_operator_sha256=digest(Path(__file__)),
                  scope='Engineering/consistency verification, not independent scene or hardware validation')
    (run / 'independent-audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()

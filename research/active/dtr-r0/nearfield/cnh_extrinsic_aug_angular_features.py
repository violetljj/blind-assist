"""Post-pilot-only conversion of retained angular photons to shared V voxels.

No fresh scene or photon generation. Caller supplies frozen angular evaluation
units. New models must consume these features; old logits cannot stand in.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import cnh_extrinsic_aug_data as D
from cnh_temporal_readout_data import normalized_z

ROOT = D.ROOT
OUT = D.OUT
PILOT = ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
DUAL = ROOT/'artifacts.local/work/cnh-dual-sensor-alarm-20261005'
ANGULAR = ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005/angular'
PSI = [0, 10, 15, 20, 25, 30]
AXES = sorted({float(psi+offset) for psi in PSI for offset in (0, -15, 15)})


def axis_name(axis):
    return ('m' if axis < 0 else 'p')+f'{abs(axis):g}'.replace('.', 'p')


def observations(unit):
    """Only invoke after PASS; keep these old observation reads label blind."""
    paths = [PILOT/'observations/evaluation'/f'unit{unit}.npz',
             DUAL/'observations'/f'unit{unit}.npz',
             ANGULAR/'observations'/f'unit{unit}.npz']
    originals = [np.load(path, allow_pickle=False) for path in paths]
    try:
        base, dual, angular = originals
        out = {}
        out[15.] = (normalized_z(base['hist'], base['ambient'][None, None]),
                    np.linalg.inv(base['travel'])@base['sensor'], base['noisy'])
        for index, axis in enumerate((0., 30.)):
            out[axis] = (normalized_z(dual['hist'][index], dual['ambient'][index][None, None]),
                         dual['query'][index], dual['noisy'][index])
        for axis in AXES:
            if axis in out:
                continue
            where = np.flatnonzero(angular['angles_deg'] == axis)
            assert len(where) == 1, (unit, axis, 'Missing retained physical axis')
            index = int(where[0])
            out[axis] = (normalized_z(angular['hist'][index], angular['ambient'][index][None, None]),
                         angular['query'][index], angular['noisy'][index])
        for z, query, noisy in out.values():
            assert z.shape == (7, 4, 16, 8, 8, 16)
            assert query.shape == (16, 4, 4) and noisy.shape == (4, 16, 4, 4)
        return out, {str(path): D.sha(path) for path in paths}
    finally:
        for data in originals:
            data.close()


def run(units):
    D.setup()
    D.require_evaluation()
    units = [int(unit) for unit in units]
    assert units and len(set(units)) == len(units)
    assert all(unit % 3 == 0 for unit in units), 'Angular evaluation is retained mode0 only'
    pilot_plan = D.read(DUAL/'PLAN.json')
    envelope_plan = D.read(ANGULAR.parent/'PLAN.json')
    expected_units = sorted(set(pilot_plan['evaluation_units']) & set(envelope_plan['A']['units']))
    assert units == expected_units, 'Only the frozen retained evaluation split is admissible'
    folder = OUT/'features/angular'
    folder.mkdir(parents=True, exist_ok=True)
    contract_path = folder/'CONTRACT.json'
    if contract_path.exists():
        contract = D.read(contract_path)
        assert contract['units'] == units and contract['axes_degrees'] == AXES
    else:
        D.save(contract_path, dict(status='FROZEN_BEFORE_ANGULAR_FEATURES', units=units,
           axes_degrees=AXES, psi_deg=PSI,
           feature_shape=[12, 28, 13, 3, 24, 17, 33],
           observations='Retained original photons/query/noisy pose at each absolute optical axis; no re-render',
           plan_sha256=D.sha(OUT/'PLAN.json'), source_sha256=D.sha(__file__)))
    projector = D.CachedProjector()
    tick = time.monotonic()
    receipts = []
    try:
        for unit in units:
            dest = folder/f'unit{unit}.npy'
            receipt_path = dest.with_suffix('.json')
            if receipt_path.exists():
                old = D.read(receipt_path)
                assert old['feature_sha256'] == D.sha(dest)
                receipts.append(old)
                continue
            retained, inputs = observations(unit)
            partial = dest.with_suffix('.partial.npy')
            assert not dest.exists() and not partial.exists(), 'Inspect partial feature evidence before restarting'
            output = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float16,
                                              shape=(12, 28, 13, 3, 24, 17, 33))
            for axis_index, axis in enumerate(AXES):
                D.plan()
                z, query, noisy = retained[axis]
                nn = np.broadcast_to(noisy[None], (7, 4, 16, 4, 4)).reshape(28, 16, 4, 4)
                qq = np.broadcast_to(query, (28, 16, 4, 4))
                features = projector.features(z.reshape(28, 16, 8, 8, 16), nn, qq)
                output[axis_index] = features.reshape(28, 13, 3, 24, 17, 33)
            output.flush()
            del output
            partial.replace(dest)
            r = dict(status='COMPLETE', unit=unit, axes_degrees=AXES,
                     feature_sha256=D.sha(dest), samples=len(AXES)*7*4*13,
                     input_sha256=inputs, source_sha256=D.sha(__file__))
            D.save(receipt_path, r)
            receipts.append(r)
            print('angular cached', unit, len(receipts), round(time.monotonic()-tick, 2), flush=True)
    finally:
        projector.close()
    rows = {name: [] for name in ('unit', 'axis_index', 'config', 'variant', 'replica', 'frame')}
    for unit in units:
        for axis_index in range(len(AXES)):
            for variant in range(7):
                for replica in range(4):
                    for frame in D.FRAMES:
                        values = (unit, axis_index, variant*4+replica, variant, replica, int(frame))
                        for key, value in zip(rows, values):
                            rows[key].append(value)
    metadata = OUT/'inputs/angular/rows.npz'
    metadata.parent.mkdir(parents=True, exist_ok=True)
    assert not metadata.exists()
    np.savez_compressed(metadata, **{key:np.asarray(value) for key,value in rows.items()})
    D.save(folder/'receipt.json', dict(status='COMPLETE', units=units, axes_degrees=AXES,
         samples=len(units)*len(AXES)*7*4*13, seconds=time.monotonic()-tick,
         variants=np.repeat(np.arange(7),4).tolist(), replicas=np.tile(np.arange(4),7).tolist(),
         configs=list(range(28)), rows_sha256=D.sha(metadata),
         source_sha256=D.sha(__file__), plan_sha256=D.sha(OUT/'PLAN.json')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--units', required=True, help='Frozen retained angular evaluation unit IDs, comma separated')
    args = ap.parse_args()
    run([int(value) for value in args.units.split(',')])

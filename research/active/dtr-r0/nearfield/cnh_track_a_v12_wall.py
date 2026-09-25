"""Reproduce the original static-wall diagnostic and apply frozen r² transport."""
import argparse
import json
from pathlib import Path
import numpy as np
from cnh_route_sensor import angular_rays, SensorParameters, synthesize_response, derive_readout, H3
from cnh_track_a_v12_readout import accumulate


def run(output):
    original = Path(__file__).resolve().parents[4] / 'artifacts.local/evidence/cnh-track-a-fov-20260925-v1/b1-static-wall-diagnostic.json'
    receipt = json.loads(original.read_text())
    params = SensorParameters(**receipt['sensor_parameters'])
    rays, weights = angular_rays(16)
    hist = np.array([derive_readout(synthesize_response(z/rays[..., 2], .5,
        rays[..., 2], weights, params=params), H3)['histogram'] for z in (2., 1.8)])
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 2, 3] = .2
    values = {}
    for enabled in (False, True):
        acc = accumulate(hist, np.zeros_like(hist), poses, 4, range_compensation=enabled)
        current, accumulated = hist[-1], acc['mean'][-1]
        l1 = float(np.abs(accumulated-current).sum())
        values['r2' if enabled else 'uncompensated'] = dict(
            absolute_L1_counts=l1, relative_L1=l1/float(np.abs(current).sum()),
            current_total_counts=float(current.sum()), accumulated_total_counts=float(accumulated.sum()))
    if abs(values['uncompensated']['relative_L1'] - receipt['relative_L1']) > 1e-12:
        raise AssertionError('Original static-wall diagnostic did not reproduce')
    result = dict(kind='SYNTHETIC_ENGINEERING_DIAGNOSTIC_NOT_DATASET', scene=receipt['scene'],
        sensor_parameters=receipt['sensor_parameters'], history_frames=2, **values,
        interpretation='History-only radial gain; no expected equality or acceptance threshold. Angular/H3 discretization persists.')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    run(p.parse_args().output)

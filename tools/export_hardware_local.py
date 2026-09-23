"""Export frozen LOCAL HGB for Android; no fitting, selection, or research mutation.

The optional parity payload is engineering regression using synthetic/consumed
public observations only. It is not fresh performance evaluation.
"""
from pathlib import Path
import argparse
import hashlib
import gzip
import io
import json
import pickle
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/active/dtr-r0/nearfield'))
from inherit_spatial_model import extract, FEATURE_SIZE
from local_transfer_inference import frozen_bundle, FROZEN_MODELS, FROZEN_SELECTION


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen', type=Path, default=ROOT / 'artifacts.local/evidence/ba-inherit-spatial-20260922-run')
    parser.add_argument('--asset', type=Path, default=ROOT / 'app/src/main/assets/hardware_local/local-v1.bin')
    parser.add_argument('--fixtures', type=Path, default=ROOT / 'core/assist/src/test/resources/hardware_local/local-parity.bin.gz')
    parser.add_argument('--observations', type=Path)
    args = parser.parse_args()
    _, thresholds = frozen_bundle(args.frozen)
    with (args.frozen / 'local.pkl').open('rb') as stream:
        model = pickle.load(stream)
    assert list(model.classes_) == [0, 1] and model.n_features_in_ == FEATURE_SIZE
    args.asset.parent.mkdir(parents=True, exist_ok=True)
    with args.asset.open('wb') as stream:
        stream.write(struct.pack('>iiddi', 0x42414c31, FEATURE_SIZE, thresholds['local'], float(model._baseline_prediction[0, 0]), len(model._predictors)))
        for iteration in model._predictors:
            assert len(iteration) == 1
            nodes = iteration[0].nodes
            assert not nodes['is_categorical'].any()
            stream.write(struct.pack('>i', len(nodes)))
            for node in nodes:
                stream.write(struct.pack('>didiibb', float(node['value']), int(node['feature_idx']), float(node['num_threshold']), int(node['left']), int(node['right']), int(node['missing_go_to_left']), int(node['is_leaf'])))
    meta = dict(schema='blindassist-hardware-local-v1', source_model_sha256=FROZEN_MODELS['local'],
        selection_sha256=FROZEN_SELECTION, threshold=thresholds['local'], asset_sha256=digest(args.asset),
        features=FEATURE_SIZE, trees=len(model._predictors), fits=0, selections=0,
        extractor_sha256=digest(ROOT / 'research/active/dtr-r0/nearfield/inherit_spatial_model.py'),
        scope='Frozen controlled-simulation LOCAL model; nominal physical camera/ToF adaptation is experimental, not calibrated or validated safety performance')
    args.asset.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8', newline='\n')
    rng = np.random.default_rng(20260923)
    boxes = np.array([[r / 8, c / 8, (r + 1) / 8, (c + 1) / 8] for r in range(8) for c in range(8)])
    cases = []
    for k in range(4):
        rgb = rng.integers(0, 256, (3, 180, 320), dtype=np.uint8) if k != 0 else np.zeros((3, 180, 320), np.uint8)
        tof = np.column_stack([rng.uniform(.0125, .999, 64), rng.integers(0, 2, 64), boxes]).astype(np.float32)
        if k == 0: tof[:, 1] = 0
        if k == 1: tof[:, 1] = 1; tof[:, 0] = .2
        cases.append((rgb, tof))
    if args.observations:
        rgbs = np.load(args.observations / 'rgb.npy', mmap_mode='r')
        tofs = np.load(args.observations / 'tof.npy', mmap_mode='r')
        for index in (0, len(rgbs) // 3, len(rgbs) // 2, len(rgbs) - 1):
            cases.append((rgbs[index], tofs[index]))
    args.fixtures.parent.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    stream.write(struct.pack('>ii', 0x42414631, len(cases)))
    for rgb, tof in cases:
        features = extract(rgb, tof)[:, 1]
        probabilities = model.predict_proba(features)[:, 1]
        stream.write(rgb.transpose(1, 2, 0).tobytes())
        stream.write(tof.astype('>f8').tobytes())
        stream.write(features.astype('>f4').tobytes())
        stream.write(probabilities.astype('>f8').tobytes())
    payload = stream.getvalue()
    args.fixtures.write_bytes(gzip.compress(payload, mtime=0) if args.fixtures.suffix == '.gz' else payload)
    fixture_meta = dict(schema='blindassist-hardware-local-parity-v1', cases=len(cases),
        synthetic_cases=4, synthetic_seed=20260923,
        consumed_public_cases=4 if args.observations else 0,
        observations=args.observations.relative_to(ROOT).as_posix() if args.observations and args.observations.is_absolute() else str(args.observations),
        indices=[0, len(rgbs)//3, len(rgbs)//2, len(rgbs)-1] if args.observations else [],
        inputs_sha256={name: digest(args.observations / name) for name in ('rgb.npy', 'tof.npy')} if args.observations else {},
        fixture_sha256=digest(args.fixtures), payload_sha256=hashlib.sha256(payload).hexdigest(),
        model=meta, scope='Engineering regression only; reused public observations, no evaluation labels or fresh performance claims')
    args.fixtures.with_suffix('.json').write_text(json.dumps(fixture_meta, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({**meta, 'fixtures': str(args.fixtures), 'cases': len(cases), 'fixture_sha256': digest(args.fixtures)}))


if __name__ == '__main__':
    main()

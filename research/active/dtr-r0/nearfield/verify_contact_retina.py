"""Verify all captured NF-G7 boxes and model/training partition boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from verify_temporal_structure import rays, intersect


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--capture', type=Path, required=True)
    args = p.parse_args()
    root = args.capture.resolve()
    target = root/'verification.json'
    assert not target.exists()
    started = time.perf_counter()
    read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
    receipt, release = read(root/'receipt.json'), read(root/'process-release.json')
    data, spec = read(root/'model/dataset.json'), read(root/'evaluator/spec.json')
    assert receipt['status'] == 'PASS' and receipt['source_unchanged'] and release['released']
    assert receipt['spec_sha256'] == sha(root.parent/'spec.json')
    assert len(data['frames']) == 768 and len(data['samples']) == 432
    assert data['sampling'] == spec['sampling'] and data['calibration']['pose_translation_unit'] == 'metres'
    assert not list((root/'model').rglob('*.npy'))
    training = read(root/'training/labels.json')['targets']
    assert set(training) == {s['sample_id'] for s in data['samples'] if s['split'] != 'test'}
    assert training == {k: v for k, v in spec['labels']['targets'].items() if k in training}
    rows, tiles, failures = [], [], []
    for i, (frame, case) in enumerate(zip(data['frames'], spec['cases'])):
        assert frame['sample_index'] == i and frame['clip_id'] == case['clip_id']
        assert frame['camera_transform'] == case['camera'] and frame['wearer_transform'] == case['wearer']
        assert frame['time_s'] == case['time_s'] and frame['speed_m_s'] == case['speed_m_s']
        image_path = root/'model'/frame['rgb_path']
        image = cv2.imread(str(image_path))
        native_path = root/f'evaluator/native/{i:04d}.npy'
        native = np.load(native_path, allow_pickle=False)
        assert image.shape == (360, 640, 3) and native.shape == (360, 640)
        origin = np.array([case['camera'][k] for k in ('x','y','z')])
        direction = rays(case['camera'])
        obj = case['objects'][-1]
        expected = intersect(origin, direction, obj)
        projected = np.isfinite(expected)
        visible = projected & (native > 0) & (np.abs(native-expected) < .03)
        if not visible.any():
            failures.append(dict(index=i, reason='No native-visible task target', projected=int(projected.sum())))
        floor = (native > 0) & (np.abs(origin[2]+native*direction[:,:,2]-.12) < .02)
        if floor.sum() < 500:
            failures.append(dict(index=i, reason='Floor reference absent'))
        rows.append(dict(sample_index=i, rgb_sha256=sha(image_path), native_sha256=sha(native_path),
                         target_projected_pixels=int(projected.sum()), target_visible_pixels=int(visible.sum()),
                         target_visible_fraction=float(visible.sum()/max(1,projected.sum())),
                         floor_reference_pixels=int(floor.sum())))
        if frame['frame_in_clip'] == 8:
            tile = np.zeros((207,320,3), np.uint8)
            tile[:180] = cv2.resize(image,(320,180))
            cv2.putText(tile, frame['clip_id'], (5,198), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            tiles.append(tile)
    sheet = np.concatenate([np.concatenate(tiles[i:i+6],axis=1) for i in range(0,len(tiles),6)],axis=0)
    cv2.imwrite(str(root/'contact-sheet.jpg'),sheet)
    result = dict(status='FAIL' if failures else 'PASS', failures=failures, frames=len(rows),
                  source_unchanged=True, owned_processes_released=True, model_excludes_native_truth=True,
                  trainval_excludes_test_targets=True, dataset_sha256=sha(root/'model/dataset.json'),
                  spec_sha256=sha(root/'evaluator/spec.json'), rows=rows,
                  elapsed_s=time.perf_counter()-started, code_sha256=sha(Path(__file__)),
                  scope='Rendered box/floor consistency; simulated trajectory time, not wall-clock cadence or human mesh')
    target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    if failures:
        raise RuntimeError('Acquisition verification failed; do not train')


if __name__ == '__main__':
    main()

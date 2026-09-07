"""CUDA native-visibility reference construction; only train/val labels cross boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from contact_retina_spec import BODY_BOXES


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def query_overlap(case, box, distance=3.):
    low, high = box
    origin = [case['wearer'][k] for k in ('x', 'y', 'z')]
    obj = case['objects'][-1]
    a = [c - s / 2 for c, s in zip(obj['center_m'], obj['size_m'])]
    b = [c + s / 2 for c, s in zip(obj['center_m'], obj['size_m'])]
    qlow = [origin[0] + high[0], origin[1] + low[1], origin[2] + low[2]]
    qhigh = [qlow[0] + distance, origin[1] + high[1], origin[2] + high[2]]
    return all(a[k] <= qhigh[k] and b[k] >= qlow[k] for k in range(3))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--stream', action='store_true', help='Verify completed frames while owned capture runs')
    args = ap.parse_args()
    root = args.capture.resolve()
    assert not (root / 'verification.json').exists()
    started = time.perf_counter()
    spec = read(root.parent / 'spec.json')
    expected_frames = [dict(sample_index=i, rgb_path=f'sample/{i:04d}.png',
                            clip_id=c['clip_id'], frame_in_clip=c['frame_in_clip'], time_s=c['time_s'])
                       for i,c in enumerate(spec['cases'])]
    data = dict(frames=expected_frames, samples=spec['samples']) if args.stream else read(root / 'model/dataset.json')
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    assert len(data['frames']) == len(spec['cases']) == 6144
    assert len(data['samples']) == 4096
    assert not list((root / 'model').rglob('*.npy'))
    allowed = {'sample_index', 'rgb_path', 'clip_id', 'frame_in_clip', 'time_s'}
    assert all(set(f) == allowed for f in data['frames'])
    sample_at = {s['frame_indices'][-1]: s for s in data['samples']}
    yy, xx = torch.meshgrid(torch.arange(360, device=device), torch.arange(640, device=device), indexing='ij')
    f = 320 / np.tan(np.radians(50))
    rx, ry = (xx - 319.5) / f, (yy - 179.5) / f
    labels, supports, training_supports, rows, tiles = {}, {}, {}, [], []
    for i, (frame, case) in enumerate(zip(data['frames'], spec['cases'])):
        if args.stream:
            deadline = time.monotonic() + 180
            while True:
                if (root/'receipt.json').exists():
                    assert read(root/'receipt.json')['status']=='PASS', 'Capture failed'
                try:
                    ready = read(root/'progress.json')['frames'] > i
                except (FileNotFoundError, json.JSONDecodeError):
                    ready = False
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f'No new captured frame {i} within180seconds')
                time.sleep(1.)
        assert frame['sample_index'] == i and frame['clip_id'] == case['clip_id']
        image_path = root / 'model' / frame['rgb_path']
        image = cv2.imread(str(image_path))
        assert image is not None and image.shape == (360, 640, 3)
        native_path = root / f'evaluator/native/{i:04d}.npy'
        native = torch.as_tensor(np.load(native_path), device=device)
        assert native.shape == (360, 640)
        pose = case['camera']
        p, y = np.radians([pose['pitch'], pose['yaw']])
        forward = torch.tensor([np.cos(p)*np.cos(y), np.cos(p)*np.sin(y), np.sin(p)], device=device, dtype=torch.float32)
        right = torch.tensor([-np.sin(y), np.cos(y), 0], device=device, dtype=torch.float32)
        up = torch.tensor([-np.sin(p)*np.cos(y), -np.sin(p)*np.sin(y), np.cos(p)], device=device, dtype=torch.float32)
        direction = forward + rx[..., None]*right - ry[..., None]*up
        origin = torch.tensor([pose[k] for k in ('x','y','z')], device=device)
        obj = case['objects'][-1]
        center = torch.tensor(obj['center_m'], device=device)
        half = torch.tensor(obj['size_m'], device=device) / 2
        a, b = (center-half-origin)/direction, (center+half-origin)/direction
        near = torch.minimum(a,b).amax(-1)
        far = torch.maximum(a,b).amin(-1)
        projected = (far >= near) & (near > 0)
        visible = projected & (native > 0) & ((native-near).abs() < .03)
        points = origin + native[...,None]*direction
        floor = (native > 0) & ((points[...,2]-.12).abs() < .02)
        nvisible = int(visible.sum())
        assert int(floor.sum()) > 500, f'No expected floor at frame {i}'
        masks, target = [], []
        for box in BODY_BOXES:
            low, high = box
            body = case['wearer']
            mask = visible & (points[...,0] >= body['x'] + high[0]) & (points[...,0] <= body['x'] + high[0] + 3.)
            mask &= (points[...,1] >= body['y'] + low[1]) & (points[...,1] <= body['y'] + high[1])
            mask &= (points[...,2] >= body['z'] + low[2]) & (points[...,2] <= body['z'] + high[2])
            geometric = query_overlap(case, box)
            target.append(1 if int(mask.sum()) >= 3 else -1 if geometric or nvisible < 3 else 0)
            masks.append(mask)
        # At the exact stop instant the last observed interval still moves.
        # Current zero velocity is not inferable from these causal RGB frames.
        stop_boundary = (case['phase'] == 'stop' and case['frame_in_clip'] == 2)
        target += [-1 if v < 0 or (v == 1 and stop_boundary) else int(v == 1 and case['speed_m_s'] > .1) for v in target[:2]]
        if i in sample_at:
            sample = sample_at[i]
            sid = sample['sample_id']
            labels[sid] = target
            pooled = F.adaptive_max_pool2d(torch.stack(masks)[None].float(), (18,32))[0].byte().cpu().numpy()
            supports[sid] = pooled
            if sample['split'] != 'test':
                training_supports[sid] = pooled
        rows.append(dict(sample_index=i, rgb_sha256=sha(image_path), native_sha256=sha(native_path),
                         target_visible_pixels=nvisible, projected_pixels=int(projected.sum()),
                         floor_pixels=int(floor.sum()), targets=target))
        if i % 192 == 5:
            tile = np.zeros((202,320,3), np.uint8)
            tile[:180] = cv2.resize(image,(320,180))
            cv2.putText(tile, frame['clip_id'],(3,196),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
            tiles.append(tile)
        if i % 256 == 0:
            print(json.dumps(dict(stage='verify',frames=i+1,total=len(data['frames']))),flush=True)
    deadline = time.monotonic()+60
    while not (root/'process-release.json').exists() and time.monotonic()<deadline:
        time.sleep(1.)
    receipt = read(root/'receipt.json')
    assert receipt['status']=='PASS' and receipt['source_unchanged']
    assert read(root/'process-release.json')['released']
    actual = read(root/'model/dataset.json')
    assert actual['frames']==expected_frames and actual['samples']==spec['samples']
    assert sha(root.parent/'spec.json')==receipt['spec_sha256']==sha(root/'evaluator/spec.json')
    assert all(set(f)==allowed for f in actual['frames'])
    train_ids = {s['sample_id'] for s in data['samples'] if s['split'] != 'test'}
    (root/'training').mkdir(exist_ok=True)
    order = ['BODYnear','HEADnear','BODYapproaching','HEADapproaching']
    for folder, ids in (('evaluator',set(labels)),('training',train_ids)):
        (root/folder/'labels.json').write_text(json.dumps(dict(target_order=order, targets={k:labels[k] for k in labels if k in ids}),indent=2),encoding='utf-8')
    np.savez_compressed(root/'training/support.npz', **training_supports)
    np.savez_compressed(root/'evaluator/support.npz', **supports)
    if tiles:
        sheet = np.concatenate([np.concatenate(tiles[j:j+4],axis=1) for j in range(0,len(tiles),4)],axis=0)
        cv2.imwrite(str(root/'contact-sheet.jpg'),sheet)
    result = dict(status='PASS',frames=len(rows),samples=len(labels),rows=rows,
                  unknown_cells=int((np.asarray(list(labels.values()))<0).sum()),
                  source_unchanged=True,model_excludes_native_pose_speed_objects=True,
                  trainval_excludes_test_targets=True,backend='CUDA',device=torch.cuda.get_device_name(),
                  dataset_sha256=sha(root/'model/dataset.json'),spec_sha256=sha(root/'evaluator/spec.json'),
                  code_sha256=sha(Path(__file__)),elapsed_s=time.perf_counter()-started,
                  scope='Visible task-object proxy queries; negatives do not certify all-scene free space; no real camera cadence')
    (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__ == '__main__':
    main()

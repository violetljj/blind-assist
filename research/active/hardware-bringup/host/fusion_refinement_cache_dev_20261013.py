"""One-pass DAV2 cache on consumed October 12 captures; no frozen eval mutation."""
from __future__ import annotations
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import cv2
import torch
from atom_camera import AtomCamera, ROOT
import real_rgb_small_model_dev_20261012 as S
import real_rgb_tof_fusion_dev_20261012 as U
import real_rgb_query_feasibility_dev_20261012 as F
import real_phone_imu_replay as R
import pose_envelope as PE
from atom_imu import load_mounting

OUT = ROOT / 'artifacts.local/work/fusion-refinement-dev-20261013'

def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cam = AtomCamera()
    rotation = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    labels = json.loads((ROOT / 'artifacts.local/work/frozen-fusion-eval-v1-smoke-20261012/labels_dev.json').read_text())
    smoke = json.loads((ROOT / 'artifacts.local/work/frozen-fusion-eval-v1-smoke-20261012/frozen_fusion_eval_v1.json').read_text())
    weights = ROOT / 'artifacts.local/models/depth-anything--Depth-Anything-V2-Small/depth_anything_v2_vits.pth'
    torch.set_num_threads(4)
    model, _ = S.load_dav2_small(str(weights))
    manifest = dict(height=smoke['head_height_m'], height_source=smoke['head_height_source'],
                    weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), sessions={}, evidence='consumed Development')
    try:
        for d in sorted(F.RAW.iterdir()):
            key = d.name[17:23]
            if key not in U.SCENES:
                continue
            x = R.load(d, rotation)
            tof_t = np.array([r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')], np.int64)
            rgb = list(F.lines(d / 'rgb.jsonl'))
            rgb_t = np.array([r['received_elapsed_ns'] for r in rgb], np.int64)
            old = np.load(U.NEW / 'joint_replay' / f'{d.name}_imu.npz')
            grades, pose_ok = PE.apply(old['S_grade'], x['pitch'], x['roll'])
            ready = np.flatnonzero(old['ready'])
            assert len(grades) == len(tof_t)
            dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
            folder = OUT / 'cache' / key
            folder.mkdir(parents=True, exist_ok=True)
            meta = []
            for i in ready:
                if time.perf_counter() - start > 1150:
                    raise TimeoutError('GPU command-wall budget reserve reached')
                j = int(np.argmin(np.abs(rgb_t - tof_t[i])))
                match = bool(abs(rgb_t[j] - tof_t[i]) <= 100_000_000)
                row = dict(i=int(i), t_ns=int(tof_t[i]), rgb_t_ns=int(rgb_t[j]), rgb_index=j,
                           seq=int(x['seq'][i]), device_ms=int(x['ms'][i]), pitch=float(x['pitch'][i]), roll=float(x['roll'][i]),
                           pose_ok=bool(pose_ok[i]), matched=match, tof_strong=(grades[i] == 2).tolist(), tof_grade=grades[i].tolist(), dist=dist[i].tolist())
                path = folder / f'{int(i):04d}.npz'
                if match and pose_ok[i]:
                    up = cam.undistort_upright(cv2.imread(str(d / rgb[j]['file']))[:, :, ::-1])
                    with torch.inference_mode():
                        disp = model.infer_image(np.ascontiguousarray(up[:, :, ::-1]), 518)
                    np.savez_compressed(path, disp=disp.astype(np.float32), valid=(up.sum(axis=2) > 0))
                    row['cache'] = str(path.relative_to(OUT))
                meta.append(row)
            label = labels.get(d.name, dict(scene=U.SCENES[key], **{'class': 'unknown'}))
            manifest['sessions'][key] = dict(session=d.name, label=label, frames=meta,
                                              tof_cache_sha256=hashlib.sha256((U.NEW / 'joint_replay' / f'{d.name}_imu.npz').read_bytes()).hexdigest())
            (OUT / 'cache_manifest.partial.json').write_text(json.dumps(manifest, indent=1), encoding='utf8')
            print(key, len(meta), 'ready', sum('cache' in r for r in meta), 'RGB', round(time.perf_counter()-start, 1), 'sec', flush=True)
        manifest['cache_wall_s'] = time.perf_counter() - start
        manifest['cache_bytes'] = sum(p.stat().st_size for p in (OUT / 'cache').rglob('*.npz'))
        (OUT / 'cache_manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf8')
    finally:
        del model
        torch.cuda.empty_cache()
        (OUT / 'cache_runtime.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-start)), encoding='utf8')

if __name__ == '__main__':
    main()

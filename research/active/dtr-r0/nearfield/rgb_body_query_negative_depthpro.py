"""Frozen native Depth Pro on additional_cal RGB/public K only.

No evaluator/reference files, training, downloads, calibration fit or K remap.
One manifest owner; partial outputs are resumable within the cumulative budget.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from rgb_body_query_input_diagnostic import sha, write

WEIGHT_SHA = '3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce'


def run(repo, observations, output, budget_s):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    lock = output/'writer.json'
    # Never replace an active/in-doubt writer automatically.
    with lock.open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), script=str(Path(__file__).resolve())), f)
    manifest_path = output/'predictions.json'
    model = transform = torch = x = pred = None
    previous_wall = 0.
    times = []
    result = dict(status='STARTING', rows=[], arm='raw-native', scale_fit=False,
                  evaluator_inputs=False, precision='float16', budget_s=budget_s,
                  principal_point='Depth Pro supplied public fx only; native RGB; no centered remap')
    try:
        obs = json.loads(observations.read_text('utf-8-sig'))
        selected = [r for r in obs['rows'] if r['split'] == 'additional_cal']
        excluded = [r for r in obs['rows'] if r['split'] in ('train', 'cal')]
        identity = lambda r: (r['scan'], r['frame'])
        if len(selected) != 240 or len({identity(r) for r in selected}) != 240:
            raise ValueError('Expected 240 unique additional_cal frames')
        if ({identity(r) for r in selected} & {identity(r) for r in excluded} or
                {r['rgb_sha256'] for r in selected} & {r['rgb_sha256'] for r in excluded}):
            raise ValueError('additional_cal overlaps train/cal')
        source_sha = sha(observations)
        selected_obs = dict(rows=selected, queries=obs['queries'],
                            source_observations_sha256=source_sha,
                            dataset_manifest_sha256=obs['dataset_manifest_sha256'])
        selected_path = output/'observations.json'
        if selected_path.exists():
            if json.loads(selected_path.read_text('utf-8-sig')) != selected_obs:
                raise ValueError('Selected observation identity differs')
        else:
            write(selected_path, selected_obs)
        script_sha = sha(Path(__file__))
        if manifest_path.exists():
            result = json.loads(manifest_path.read_text('utf-8-sig'))
            if (result['observations_sha256'] != sha(selected_path) or
                    result['script_sha256'] != script_sha or result['budget_s'] != budget_s):
                raise ValueError('Resume input/config identity differs')
            if result['status'] == 'COMPLETE':
                return True
            previous_wall = result['allocation_wall_s']
            by_key = {identity(r): r for r in selected}
            for p in result['rows']:
                if (p['rgb_sha256'] != by_key[identity(p)]['rgb_sha256'] or
                        sha(Path(p['path'])) != p['sha256']):
                    raise ValueError('Resume output identity differs')
        result.update(status='STARTING', observations_sha256=sha(selected_path),
                      source_observations_sha256=source_sha, script_sha256=script_sha,
                      pid=os.getpid(), total_frames=240)
        write(manifest_path, result)
        base = repo/'artifacts.local/work/ba-nfo-depthpro-20260919'
        backend_path = base/'backend.json'
        backend = json.loads(backend_path.read_text('utf-8-sig'))
        weight = base/'depth_pro.pt'
        if sha(weight) != WEIGHT_SHA:
            raise ValueError('Weight identity differs')
        upstream = base/'upstream/src'
        code_rows = [dict(path=p.relative_to(upstream).as_posix(), sha256=sha(p))
                     for p in sorted((upstream/'depth_pro').rglob('*.py'))]
        code_fingerprint = hashlib.sha256(json.dumps(code_rows, sort_keys=True).encode()).hexdigest()
        write(output/'model_identity.json', dict(weight_sha256=WEIGHT_SHA,
              backend_sha256=sha(backend_path), upstream_python_files=code_rows,
              upstream_code_sha256=code_fingerprint))
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(4)
        if (not torch.cuda.is_available() or backend['selected_framework'] != f'torch-{torch.__version__}' or
                backend['selected_device_name'] != torch.cuda.get_device_name(0)):
            raise RuntimeError('Inherited measured backend differs')
        result.update(device=torch.cuda.get_device_name(0), framework=f'torch-{torch.__version__}',
                      backend='torch-cuda', weight_sha256=WEIGHT_SHA,
                      backend_reference_sha256=sha(backend_path), upstream_code_sha256=code_fingerprint)
        sys.path.insert(0, str(upstream))
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(weight))
        model, transform = depth_pro.create_model_and_transforms(cfg, device=torch.device('cuda'), precision=torch.float16)
        model.eval().requires_grad_(False)
        result['startup_s'] = time.perf_counter()-started
        done = {identity(p) for p in result['rows']}
        print(f'START {len(done)}/240 startup={result["startup_s"]:.2f}s remaining_budget={budget_s-previous_wall-result["startup_s"]:.2f}s', flush=True)
        for row in selected:
            if identity(row) in done:
                continue
            remaining = budget_s-previous_wall-(time.perf_counter()-started)
            if remaining < max(times[-5:], default=4.)+3.:
                result['status'] = 'PARTIAL_BUDGET'
                break
            rgb_path = Path(row['rgb_path'])
            if sha(rgb_path) != row['rgb_sha256']:
                raise ValueError('RGB identity differs')
            with Image.open(rgb_path) as image:
                rgb = image.convert('RGB')
            if [rgb.height, rgb.width] != row['color_shape']:
                raise ValueError('RGB shape differs')
            x = transform(rgb)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode():
                pred = model.infer(x, f_px=torch.tensor(row['color_K'][0][0], device='cuda'))
            if pred['depth'].device.type != 'cuda':
                raise RuntimeError('Prediction did not run on CUDA')
            torch.cuda.synchronize()
            times.append(time.perf_counter()-begin)
            array = pred['depth'].float().cpu().numpy()
            if list(array.shape) != row['color_shape']:
                raise ValueError('Prediction shape differs')
            out = output/f'{row["scan"]}_{row["frame"]:06d}.npz'
            if out.exists():
                raise FileExistsError('Unmanifested prediction: inspect before resume')
            np.savez_compressed(out, depth=array)
            result['rows'].append(dict(environment=row['environment'], scan=row['scan'],
                frame=row['frame'], split=row['split'], rgb_sha256=row['rgb_sha256'],
                path=str(out), sha256=sha(out), inference_s=times[-1]))
            result.update(status='RUNNING', completed_calls=len(result['rows']),
                          elapsed_s=time.perf_counter()-started,
                          allocation_wall_s=previous_wall+time.perf_counter()-started)
            write(manifest_path, result)
            if len(result['rows']) % 8 == 0:
                print(f'{len(result["rows"])}/240 call={times[-1]:.3f}s elapsed={result["elapsed_s"]:.2f}s', flush=True)
            del array, rgb
            x = pred = None
        else:
            result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='FAILED_PARTIAL' if result['rows'] else 'FAILED', error=repr(error))
    finally:
        result.update(inference_s=sum(r['inference_s'] for r in result['rows']),
                      completed_calls=len(result['rows']), attempt_calls=len(times))
        if torch is not None and torch.cuda.is_available():
            result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            x = pred = model = transform = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            result['allocated_bytes_after_cleanup'] = torch.cuda.memory_allocated()
        result['allocation_wall_s'] = previous_wall+time.perf_counter()-started
        result['resource_release'] = 'model/tensors released; CUDA cache cleared; process exits'
        write(manifest_path, result)
        write(output/'terminal.json', {k: v for k, v in result.items() if k != 'rows'})
        lock.unlink()
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
    return result['status'] == 'COMPLETE'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=900.)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.repo.resolve(), args.observations.resolve(), args.output.resolve(), args.budget_s) else 1)

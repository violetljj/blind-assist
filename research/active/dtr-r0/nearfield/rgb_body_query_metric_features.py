"""Observation-only frozen Depth Pro features on public calibrated sensor rays.

Reads RGB/public calibration and a prior prediction cache for reproducibility,
never sensor-reference depth, labels, evaluation, or dataset manifests.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_3rscan import color_coordinates, sample_prediction


def sampling_grid(row):
    """Public identity-extrinsic depth rays -> RGB -> internal feature grid.

    align_corners=False preserves pixel centres through the official square
    resize: normalized coordinates equal 2*(RGB pixel+.5)/RGB dimension-1.
    This is independent of reference depth and model outputs.
    """
    mx, my = color_coordinates(np.asarray(row['depth_K']),
                               np.asarray(row['color_K']), row['depth_shape'])
    ch, cw = row['color_shape']
    grid = np.stack([2*(mx+.5)/cw-1, 2*(my+.5)/ch-1], -1)
    inside = (mx >= 0) & (mx <= cw-1) & (my >= 0) & (my <= ch-1)
    return grid.astype(np.float32), inside, mx, my


def run(repo, sensor, output, budget_s=240.):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'feature_manifest.json'
    if manifest_path.exists() or (output/'plan.json').exists():
        raise FileExistsError('Preserve evidence; no implicit restart')
    obs_path = sensor/'observations.json'
    prior_path = sensor/'depthpro/predictions.json'
    observations = json.loads(obs_path.read_text('utf-8-sig'))
    previous = json.loads(prior_path.read_text('utf-8-sig'))
    rows = observations['rows']
    priors = {(r['scan'], r['frame']): r for r in previous['rows']}
    plan = dict(status='PLANNED', frames=len(rows), gpu_allocation_budget_s=budget_s,
                download_bytes=0, input_contract='Full native RGB + public color/depth K; no reference reads',
                model='Frozen cached Depth Pro FP16 raw-native with public fx; head[3] ReLU 32-channel output',
                sampling='float32 grid_sample, bilinear, zero padding, align_corners=False; public RGB-normalized sensor ray coordinates',
                output_contract='Per-frame float16 [H,W,32] features; float32 sampled predicted depth and full native predicted depth; source/output SHA256',
                observation_sha256=sha(obs_path), previous_prediction_manifest_sha256=sha(prior_path),
                script_sha256=sha(Path(__file__)))
    write(output/'plan.json', plan)
    receipt = {**plan, 'rows': [], 'status': 'STARTING', 'hook_calls': 0}
    torch = model = transform = handle = None
    capture = {}
    start = time.perf_counter()
    times = []
    try:
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(4)
        base = repo/'artifacts.local/work/ba-nfo-depthpro-20260919'
        backend = json.loads((base/'backend.json').read_text('utf-8-sig'))
        if (not torch.cuda.is_available() or
                backend['selected_framework'] != f'torch-{torch.__version__}' or
                backend['selected_device_name'] != torch.cuda.get_device_name(0)):
            raise RuntimeError('Inherited measured backend differs')
        torch.cuda.reset_peak_memory_stats()
        weight = base/'depth_pro.pt'
        weight_sha = sha(weight)
        if weight_sha != '3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce':
            raise ValueError('Frozen weight identity differs')
        sys.path.insert(0, str(base/'upstream/src'))
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        source_root = base/'upstream/src/depth_pro'
        source_sha = {str(p.relative_to(source_root)): sha(p) for p in sorted(source_root.rglob('*.py'))}
        cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(weight))
        model, transform = depth_pro.create_model_and_transforms(cfg, device=torch.device('cuda'), precision=torch.float16)
        model.eval().requires_grad_(False)
        if not isinstance(model.head[3], torch.nn.ReLU) or model.head[2].out_channels != 32:
            raise ValueError('Unexpected official head feature contract')
        def hook(module, inputs, value):
            capture['value'] = value
            receipt['hook_calls'] += 1
        handle = model.head[3].register_forward_hook(hook)
        receipt.update(framework=f'torch-{torch.__version__}', device=torch.cuda.get_device_name(0),
                       backend_reference_sha256=sha(base/'backend.json'), weight_sha256=weight_sha,
                       source_sha256=source_sha, head_modules=[str(m) for m in model.head],
                       internal_image_size=model.img_size, startup_s=time.perf_counter()-start)
        for index, row in enumerate(rows):
            if budget_s-(time.perf_counter()-start) < max(times[-5:], default=3.)+4.:
                receipt['status'] = 'PARTIAL_BUDGET'
                break
            if sha(row['rgb_path']) != row['rgb_sha256']:
                raise ValueError('RGB source identity changed')
            prior = priors[row['scan'], row['frame']]
            if prior['rgb_sha256'] != row['rgb_sha256'] or sha(prior['path']) != prior['sha256']:
                raise ValueError('Prior prediction identity differs')
            rgb = Image.open(row['rgb_path']).convert('RGB')
            if [rgb.height, rgb.width] != row['color_shape']:
                raise ValueError('Unexpected native RGB shape')
            grid, inside, mx, my = sampling_grid(row)
            grid_gpu = torch.as_tensor(grid, device='cuda', dtype=torch.float32)[None]
            x = transform(rgb)
            torch.cuda.synchronize()
            call_start = time.perf_counter()
            with torch.inference_mode():
                pred = model.infer(x, f_px=torch.tensor(row['color_K'][0][0], device='cuda'))
                feature = capture.pop('value')
                if feature.shape != (1,32,model.img_size,model.img_size):
                    raise ValueError('Unexpected feature shape')
                if index == 0:
                    receipt['first_hook_shape'] = list(feature.shape)
                    receipt['first_input_shape'] = list(x.shape)
                    receipt['first_feature_dtype'] = str(feature.dtype)
                sampled = F.grid_sample(feature.float(), grid_gpu, mode='bilinear',
                                        padding_mode='zeros', align_corners=False)
                feat = sampled[0].permute(1,2,0).half().cpu().numpy()
                depth = pred['depth'].float().cpu().numpy()
            torch.cuda.synchronize()
            times.append(time.perf_counter()-call_start)
            if list(depth.shape) != row['color_shape'] or not np.isfinite(feat).all():
                raise ValueError('Invalid feature/depth output')
            with np.load(prior['path'], allow_pickle=False) as old:
                old_depth = old['depth']
            if old_depth.shape != depth.shape:
                raise ValueError('Prior output shape differs')
            diff = np.abs(old_depth.astype(np.float32)-depth)
            stem = f'{row["scan"]}_{row["frame"]:06d}'
            feature_path = output/f'{stem}.npz'
            depth_path = output/f'{stem}_depth.npz'
            sampled_depth = sample_prediction(depth, mx, my)
            # Uncompressed writes avoid unnecessary CPU compression inside the
            # GPU allocation budget; durable payloads remain in canonical F:.
            np.savez(feature_path, features=feat, predicted_depth=sampled_depth,
                     feature_in_color=inside, color_K=np.asarray(row['color_K']),
                     depth_K=np.asarray(row['depth_K']))
            np.savez(depth_path, depth=depth)
            receipt['rows'].append(dict(row=index, environment=row['environment'], scan=row['scan'],
                frame=row['frame'], split=row['split'], rgb_path=row['rgb_path'], rgb_sha256=row['rgb_sha256'],
                color_K=row['color_K'], depth_K=row['depth_K'], color_shape=row['color_shape'], depth_shape=row['depth_shape'],
                feature_path=str(feature_path), feature_sha256=sha(feature_path), feature_shape=list(feat.shape),
                depth_path=str(depth_path), depth_sha256=sha(depth_path), feature_in_color_pixels=int(inside.sum()),
                prior_prediction_sha256=prior['sha256'], previous_depth_abs_max_m=float(diff.max()),
                previous_depth_abs_mean_m=float(diff.mean()), inference_and_feature_sample_s=times[-1]))
            receipt.update(completed_frames=len(times), elapsed_s=time.perf_counter()-start, status='RUNNING')
            write(manifest_path, receipt)
            if len(times)%8 == 0:
                print(f'features {len(times)}/{len(rows)} elapsed={receipt["elapsed_s"]:.3f}s', flush=True)
            del x, pred, feature, sampled, grid_gpu, feat, depth, old_depth, diff
        else:
            receipt['status'] = 'COMPLETE'
    except Exception as error:
        receipt.update(status='FAILED', error=repr(error))
        raise
    finally:
        if handle is not None:
            handle.remove()
        capture.clear()
        receipt['hook_removed'] = True
        if torch is not None and torch.cuda.is_available():
            receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        del handle, model, transform
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            receipt['allocated_bytes_after_cleanup'] = torch.cuda.memory_allocated()
        receipt.update(allocation_wall_s=time.perf_counter()-start,
                       inference_and_feature_sample_s=sum(times), completed_frames=len(times))
        write(manifest_path, receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('rows','source_sha256','head_modules')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--sensor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=240.)
    args = parser.parse_args()
    run(args.repo.resolve(), args.sensor.resolve(), args.output.resolve(), args.budget_s)

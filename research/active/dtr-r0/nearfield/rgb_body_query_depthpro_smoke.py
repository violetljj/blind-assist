"""Observation-only paired native/low RGB Depth Pro execution on real train data.

No stereo/depth/panoptic/pose labels are read. Outputs are estimated depth, not
body-query truth, metric accuracy, event recall, or mobile latency evidence.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from rgb_body_query_input_diagnostic import sha, write


def run(repo, output, count, budget_s):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch.set_num_threads(4)
        base = repo / 'artifacts.local/work/ba-nfo-depthpro-20260919'
        # Same installed model/framework/device and fixed internal inference
        # lattice as the earlier measured CPU/GPU probe. Reuse its placement,
        # then time actual native images here; not a repeated frozen-run call.
        backend = json.loads((base / 'backend.json').read_text('utf-8-sig'))
        if (backend['selected_framework'] != f'torch-{torch.__version__}' or
                not torch.cuda.is_available() or
                backend['selected_device_name'] != torch.cuda.get_device_name(0)):
            raise RuntimeError('Earlier backend measurement no longer matches; placement needs refresh')
        weight = base / 'depth_pro.pt'
        weight_sha = sha(weight)
        if weight_sha != '3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce':
            raise ValueError('Reference weight identity changed')
        sys.path.insert(0, str(base / 'upstream/src'))
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(weight))
        model, transform = depth_pro.create_model_and_transforms(
            cfg, device=torch.device('cuda'), precision=torch.float16)
        model.eval().requires_grad_(False)
        observations = json.loads((output / 'observations.json').read_text('utf-8-sig'))
        selected, seen = [], set()
        for row in observations:
            if row['session'] not in seen:
                selected.append(row)
                seen.add(row['session'])
        selected = selected[:count]
        write(output / 'baseline_selection.json', dict(
            method='First manifest frame of each session in existing manifest order, no model/label selection',
            rows=selected, weight_sha256=weight_sha,
            note='same source image, focal length scaled with actual input width; no GT scale fitting'))
        results, timing, displays = [], [], []
        for index, row in enumerate(selected):
            image_path, description_path = repo / row['rgb'], repo / row['description']
            if sha(image_path) != row['rgb_sha256'] or sha(description_path) != row['description_sha256']:
                raise ValueError('Input identity changed after diagnostic')
            rgb = Image.open(image_path).convert('RGB')
            desc = json.loads(description_path.read_text('utf-8-sig'))
            camera = desc['session_camera_details'][desc['session_camera_location'].index(row['camera'])]
            params = camera['left_camera_params']
            if any(float(x) != 0 for x in params['distortion']):
                raise ValueError('This smoke expects official rectified pinhole RGB')
            native_fx = params['fx'] * rgb.width / params['image_width']
            depths, times = {}, {}
            for name, image in (('native', rgb), ('low128', rgb.resize((128, 72), Image.Resampling.BILINEAR))):
                if time.perf_counter() - started >= budget_s:
                    raise TimeoutError('Baseline startup/inference wall-time budget reached')
                fx = native_fx * image.width / rgb.width
                x = transform(image)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.inference_mode():
                    pred = model.infer(x, f_px=torch.tensor(fx, device='cuda'))
                if pred['depth'].device.type != 'cuda':
                    raise RuntimeError('Inference did not execute on CUDA')
                torch.cuda.synchronize()
                seconds = time.perf_counter()-begin
                array = pred['depth'].float().cpu().numpy()
                depths[name] = array
                times[name] = seconds
                timing.append(seconds)
                np.savez_compressed(output / f'depthpro_{index:02d}_{name}.npz', depth=array)
                write(output / 'baseline_progress.json', dict(completed=len(timing), total=2*len(selected),
                                                             stage='inference', seconds=time.perf_counter()-started))
                print(f'{len(timing)}/{2*len(selected)} {name} {seconds:.3f}s', flush=True)
                del pred, x
            native, low = depths['native'], depths['low128']
            if native.shape != (rgb.height, rgb.width) or low.shape != (72, 128):
                raise ValueError('Unexpected official output shape')
            low_up = np.asarray(Image.fromarray(low).resize(rgb.size, Image.Resampling.BILINEAR))
            valid = np.isfinite(native) & (native > 0) & np.isfinite(low_up) & (low_up > 0)
            results.append(dict(session=row['session'], camera=row['camera'], frame=row['frame'],
                                rgb_sha256=row['rgb_sha256'], native_shape=list(native.shape),
                                fx_native=native_fx, fx_low128=native_fx*128/rgb.width,
                                valid_paired_fraction=float(valid.mean()), inference_seconds=times,
                                median_relative_output_change=float(np.median(np.abs(low_up[valid]-native[valid])/native[valid])) if valid.any() else None,
                                native_median_depth=float(np.median(native[valid])) if valid.any() else None,
                                low128_median_depth=float(np.median(low_up[valid])) if valid.any() else None))
            if index < 3:
                displays.append((rgb, native, low_up))
        torch.cuda.synchronize()
        result = dict(status='COMPLETE', frames=len(results), calls=len(timing),
                      backend='torch-cuda', device=torch.cuda.get_device_name(0), framework=f'torch-{torch.__version__}',
                      backend_reference='artifacts.local/work/ba-nfo-depthpro-20260919/backend.json',
                      weight_sha256=weight_sha, precision='float16', allocation_wall_seconds=time.perf_counter()-started,
                      measured_inference_seconds=sum(timing), median_call_seconds=float(np.median(timing)),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      event_recall='NOT_EVALUABLE_NO_EVENT_LABELS', metric_accuracy='NOT_EVALUATED',
                      scale_fit=False, body_transform='NOT_AVAILABLE', rows=results)
        write(output / 'depthpro_smoke.json', result)
        sheet = Image.new('RGB', (1440, 300*len(displays)), '#111111')
        for j, (rgb, native, low) in enumerate(displays):
            images = [rgb]
            for depth in (native, low):
                v = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
                gray = np.uint8(np.clip(v/10, 0, 1)*255)
                images.append(Image.fromarray(gray).convert('RGB'))
            for k, image in enumerate(images):
                sheet.paste(image.resize((480, 270)), (480*k, 300*j+30))
                ImageDraw.Draw(sheet).text((480*k+8, 300*j+8), ('Real RGB', 'Native-input predicted depth 0-10m', '128x72-input predicted depth 0-10m')[k], fill='white')
        sheet.save(output / 'depthpro_comparison.png')
        print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)
    except Exception as error:
        write(output / 'depthpro_failure.json', dict(status='FAILED', error=repr(error),
                                                    seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--budget-s', type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.frames <= 10:
        parser.error('This smoke permits 1-10 real frames')
    run(args.repo.resolve(), args.output.resolve(), args.frames, args.budget_s)

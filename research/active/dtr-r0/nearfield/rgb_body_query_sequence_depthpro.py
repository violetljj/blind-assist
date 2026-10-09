"""Fixed 45-frame observation-only Depth Pro baseline; no evaluator inputs.

Centered arms remap public rectified K to a centered equal-focal pinhole,
preserving original camera rays and optical Z. All outputs share the original
1280x720 ray grid. Raw arms retain the fx-only principal-point limitation.
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

from rgb_body_query_geometry import resize_intrinsics
from rgb_body_query_input_diagnostic import sha, write

ARMS = ('raw-native', 'raw128x72', 'K-centered-native', 'K-centered-low128')
WEIGHT_SHA = '3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce'


def centered_camera(k, shape):
    """Enclose full original pixel-cell ray extent, using only public K.

    Integer centers; pixel cells extend from -.5 to width-.5. No rotation,
    no image-dependent crop, no truth. Focal chosen as public original fx.
    """
    k = np.asarray(k, np.float64)
    h, w = shape
    if k[0, 1] != 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('Only zero-skew rectified pinhole supported')
    f = float(k[0, 0])
    extent_x = max(abs((-.5-k[0, 2])/k[0, 0]), abs((w-.5-k[0, 2])/k[0, 0]))
    extent_y = max(abs((-.5-k[1, 2])/k[1, 1]), abs((h-.5-k[1, 2])/k[1, 1]))
    cw, ch = int(np.ceil(2*f*extent_x)), int(np.ceil(2*f*extent_y))
    ck = np.array([[f, 0, (cw-1)/2], [0, f, (ch-1)/2], [0, 0, 1]], np.float64)
    return ck, (ch, cw)


def coordinate_map(source_k, target_k, target_shape):
    """Map target integer centers to original source integer centers."""
    yy, xx = np.indices(target_shape, dtype=np.float32)
    sx = (xx-target_k[0, 2])/target_k[0, 0]*source_k[0, 0]+source_k[0, 2]
    sy = (yy-target_k[1, 2])/target_k[1, 1]*source_k[1, 1]+source_k[1, 2]
    return sx.astype(np.float32), sy.astype(np.float32)


def valid_cells(x, y, shape):
    h, w = shape
    return (x >= -.5) & (x <= w-.5) & (y >= -.5) & (y <= h-.5)


def run(repo, output, budget_s):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'prediction_manifest.json'
    if manifest_path.exists():
        raise FileExistsError('Preserve prior run; no implicit restart')
    manifest = dict(status='STARTING', arms=list(ARMS), total_calls=180, rows=[],
                    coordinate_contract='original camera right/down/forward; optical Z metres',
                    pixel_convention='integer centers; resize u_new=(u_old+.5)*scale-.5',
                    scale_fit=False, evaluator_inputs=False, future_frames=False,
                    budget_s=budget_s, model_internal_shape=[1536, 1536],
                    model_input_contract='RGB and public description K only; acquisition metadata selects fixed paths',
                    centered_padding='constant black; invalid original mapping excluded; full public ray extent retained')
    torch, model, transform = None, None, None
    try:
        import cv2
        import torch
        torch.set_num_threads(4)
        base = repo/'artifacts.local/work/ba-nfo-depthpro-20260919'
        backend_path = base/'backend.json'
        backend = json.loads(backend_path.read_text('utf-8-sig'))
        if (backend['selected_framework'] != f'torch-{torch.__version__}' or
                not torch.cuda.is_available() or
                backend['selected_device_name'] != torch.cuda.get_device_name(0)):
            raise RuntimeError('Inherited CPU/GPU placement no longer matches')
        manifest.update(backend='torch-cuda', framework=f'torch-{torch.__version__}',
                        device=torch.cuda.get_device_name(0), precision='float16',
                        backend_reference='artifacts.local/work/ba-nfo-depthpro-20260919/backend.json',
                        backend_reference_sha256=sha(backend_path), inherited_backend_measurement=True)
        weight = base/'depth_pro.pt'
        if sha(weight) != WEIGHT_SHA:
            raise ValueError('Model hash mismatch')
        manifest['weight_sha256'] = WEIGHT_SHA
        sys.path.insert(0, str(base/'upstream/src'))
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(weight))
        model, transform = depth_pro.create_model_and_transforms(cfg, device=torch.device('cuda'), precision=torch.float16)
        model.eval().requires_grad_(False)
        sequence = repo/'artifacts.local/work/rgb-body-query-dev-20261009/sequence'
        receipt = json.loads((sequence/'acquisition_receipt.json').read_text('utf-8-sig'))
        plan = json.loads((sequence/'planned_files.json').read_text('utf-8-sig'))
        manifest.update(acquisition_receipt_sha256=sha(sequence/'acquisition_receipt.json'),
                        plan_sha256=sha(sequence/'planned_files.json'), session=receipt['session'], camera=plan['camera'])
        refs = [r for r in plan['references'] if Path(r['path']).name == 'description.json']
        if len(refs) != 1:
            raise ValueError('Expected unique public description reference')
        ref = refs[0]
        desc_path = Path(ref['path'])
        if sha(desc_path) != ref['sha256']:
            raise ValueError('Description identity changed')
        desc = json.loads(desc_path.read_text('utf-8-sig'))
        params = desc['session_camera_details'][desc['session_camera_location'].index(plan['camera'])]['left_camera_params']
        if any(float(d) != 0 for d in params['distortion']):
            raise ValueError('Zero public distortion required')
        public_k = np.array([[params['fx'], 0, params['cx']], [0, params['fy'], params['cy']], [0, 0, 1]], np.float64)
        native_shape, grid_shape = (1242, 2208), (720, 1280)
        nk = resize_intrinsics(public_k, (params['image_height'], params['image_width']), native_shape)
        gk = resize_intrinsics(nk, native_shape, grid_shape)
        ck, centered_shape = centered_camera(nk, native_shape)
        low_centered_shape = (round(128*centered_shape[0]/centered_shape[1]), 128)
        arm_shapes = dict(zip(ARMS, (native_shape, (72, 128), centered_shape, low_centered_shape)))
        arm_ks = {arm: resize_intrinsics(ck if arm.startswith('K-') else nk,
                                       centered_shape if arm.startswith('K-') else native_shape, arm_shapes[arm]) for arm in ARMS}
        manifest.update(description_path=str(desc_path), description_sha256=ref['sha256'],
                        public_K=public_k.tolist(), original_K=nk.tolist(), output_K=gk.tolist(),
                        output_shape=list(grid_shape), centered_K=ck.tolist(), centered_shape=list(centered_shape),
                        arms_config={arm: dict(shape=list(arm_shapes[arm]), K=arm_ks[arm].tolist(),
                                             focal_px=float(arm_ks[arm][0, 0]),
                                             focal_width_ratio=float(arm_ks[arm][0, 0]/arm_shapes[arm][1]),
                                             principal_point_correction=arm.startswith('K-')) for arm in ARMS})
        files = sorted((r for r in receipt['verified_files'] if r['kind'] == 'video_frames'), key=lambda r: r['frame'])
        if [r['frame'] for r in files] != list(range(392, 437)):
            raise ValueError('Fixed 45-frame identity differs')
        input_map = coordinate_map(nk, ck, centered_shape)
        input_valid = valid_cells(*input_map, native_shape)
        np.savez_compressed(output/'centered_mapping.npz', valid_mask=input_valid, K=ck)
        output_maps = {arm: coordinate_map(arm_ks[arm], gk, grid_shape) for arm in ARMS}
        manifest['startup_seconds'] = time.perf_counter()-started
        write(manifest_path, manifest)
        times = []
        for r in files:
            path = Path(r['path'])
            if sha(path) != r['sha256']:
                raise ValueError('RGB identity changed')
            with Image.open(path) as image:
                rgb = image.convert('RGB')
            if (rgb.height, rgb.width) != native_shape:
                raise ValueError('Unexpected source dimensions')
            centered = cv2.remap(np.asarray(rgb), *input_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            centered[~input_valid] = 0
            images = {ARMS[0]: rgb, ARMS[1]: rgb.resize((128, 72), Image.Resampling.BILINEAR),
                      ARMS[2]: Image.fromarray(centered),
                      ARMS[3]: Image.fromarray(centered).resize((128, low_centered_shape[0]), Image.Resampling.BILINEAR)}
            for arm in ARMS:
                # Leave room for the next call and terminal receipt. Failure/startup counts.
                remaining = budget_s-(time.perf_counter()-started)
                reserve = max(times[-5:], default=2.0)+2.0
                if remaining < reserve:
                    manifest['status'] = 'PARTIAL_BUDGET'
                    raise TimeoutError('Stop before next inference at fixed budget boundary')
                x = transform(images[arm])
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.inference_mode():
                    pred = model.infer(x, f_px=torch.tensor(arm_ks[arm][0, 0], device='cuda', dtype=torch.float32))
                if pred['depth'].device.type != 'cuda':
                    raise RuntimeError('Prediction must be CUDA')
                torch.cuda.synchronize()
                call_seconds = time.perf_counter()-begin
                times.append(call_seconds)
                source_depth = pred['depth'].float().cpu().numpy()
                if source_depth.shape != arm_shapes[arm]:
                    raise ValueError('Depth output does not match input rays')
                mx, my = output_maps[arm]
                valid = valid_cells(mx, my, arm_shapes[arm])
                depth = cv2.remap(source_depth, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                valid &= np.isfinite(depth) & (depth > 0)
                depth[~valid] = np.nan
                out = output/f'{r["frame"]:06d}_{arm}.npz'
                np.savez_compressed(out, depth=depth, valid_mask=valid, K=gk)
                manifest['rows'].append(dict(frame=r['frame'], arm=arm, session=receipt['session'], camera=plan['camera'],
                                             rgb_path=str(path), rgb_sha256=r['sha256'], path=str(out),
                                             depth_path=str(out), sha256=sha(out), depth_sha256=sha(out),
                                             K=gk.tolist(), public_K=gk.tolist(), shape=list(grid_shape),
                                             invalid_mask_key='valid_mask (invert for invalid)', invalid_pixels=int((~valid).sum()),
                                             input_shape=list(arm_shapes[arm]), input_K=arm_ks[arm].tolist(),
                                             inference_seconds=call_seconds, call_kind='cold' if len(times)==1 else 'steady'))
                manifest.update(status='IN_PROGRESS', completed_calls=len(times), elapsed_s=time.perf_counter()-started)
                write(manifest_path, manifest)
                print(f'{len(times)}/180 {r["frame"]} {arm} {call_seconds:.3f}s elapsed={manifest["elapsed_s"]:.2f}', flush=True)
                del x, pred, source_depth, depth
        manifest['status'] = 'COMPLETE'
    except Exception as error:
        if manifest['status'] != 'PARTIAL_BUDGET':
            manifest['status'] = 'FAILED_PARTIAL' if manifest['rows'] else 'FAILED'
        manifest['error'] = repr(error)
    finally:
        manifest['allocation_wall_seconds'] = time.perf_counter()-started
        timings = [r['inference_seconds'] for r in manifest['rows']]
        manifest.update(completed_calls=len(timings), completed_frames=len({r['frame'] for r in manifest['rows']}),
                        measured_inference_seconds=sum(timings), cold_call_seconds=timings[0] if timings else None,
                        steady_median_seconds=float(np.median(timings[1:])) if len(timings)>1 else None,
                        script_sha256=sha(Path(__file__)))
        if torch is not None and torch.cuda.is_available():
            manifest['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            del model, transform
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        manifest['resource_release'] = 'model deleted; CUDA cache cleared; owning process exits on return'
        write(manifest_path, manifest)
        print(json.dumps({k: v for k, v in manifest.items() if k != 'rows'}), flush=True)
    return manifest['status'] == 'COMPLETE'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=240)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.repo.resolve(), args.output.resolve(), args.budget_s) else 1)

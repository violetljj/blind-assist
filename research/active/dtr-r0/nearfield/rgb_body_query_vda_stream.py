"""Official metric Video Depth Anything, observation-only causal streaming.

Each call reads the current left RGB; temporal state contains past frames only.
Public K is retained for downstream geometry, not supplied to the VDA API.
Depth, mask, pose and evaluator labels are never loaded. No GT scale fitting.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

from rgb_body_query_geometry import resize_intrinsics, visible_readout

SOURCE_REVISION = '4f5ae23172ba60fd7bc11ef671cca678842c7072'
WEIGHT_SHA256 = '3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33'
WEIGHT_BYTES = 116444063


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def select_rgb(receipt):
    rows = sorted((r for r in receipt['verified_files'] if r['kind'] == 'video_frames'),
                  key=lambda r: r['frame'])
    if [r['frame'] for r in rows] != list(range(392, 437)):
        raise ValueError('Need exactly the previously acquired 45 consecutive left RGB frames')
    return rows


def run(sequence, output, budget_s=240, input_size=518):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    torch = model = original_features = None
    predictions = []
    result = dict(status='RUNNING', rows=rows, budget_s=budget_s)
    try:
        receipt_path = sequence/'acquisition_receipt.json'
        selected = select_rgb(json.loads(receipt_path.read_text('utf-8-sig')))
        plan = json.loads((sequence/'planned_files.json').read_text('utf-8-sig'))
        description_ref = next(r for r in plan['references'] if Path(r['path']).name == 'description.json')
        if sha(description_ref['path']) != description_ref['sha256']:
            raise ValueError('Public description input identity changed')
        description = json.loads(Path(description_ref['path']).read_text('utf-8-sig'))
        params = description['session_camera_details'][description['session_camera_location'].index(plan['camera'])]['left_camera_params']
        if any(float(v) != 0 for v in params['distortion']):
            raise ValueError('This shared readout requires rectified public pinhole RGB')
        public_shape = (params['image_height'], params['image_width'])
        public_k = np.array([[params['fx'], 0., params['cx']], [0., params['fy'], params['cy']], [0., 0., 1.]])
        common_shape = (720, 1280)
        common_k = resize_intrinsics(public_k, public_shape, common_shape)
        upstream, weight = output/'upstream', output/'metric_video_depth_anything_vits.pth'
        revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != SOURCE_REVISION or sha(weight) != WEIGHT_SHA256 or weight.stat().st_size != WEIGHT_BYTES:
            raise ValueError('Official source/checkpoint identity changed')
        metadata = json.loads((output/'model_metadata.json').read_text('utf-8-sig'))
        if metadata.get('cardData', {}).get('license') != 'apache-2.0':
            raise ValueError('Checkpoint licence differs from actual official model card')
        source_files = {str(p.relative_to(upstream)).replace('\\', '/'): sha(p)
                        for p in [upstream/'run_streaming.py', upstream/'video_depth_anything/video_depth_stream.py',
                                  upstream/'video_depth_anything/dpt_temporal.py',
                                  upstream/'video_depth_anything/motion_module/motion_module.py']}
        sys.path[:0] = [str(output/'dependencies'), str(upstream)]
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; this GPU-suitable run cannot silently fall back')
        from video_depth_anything.video_depth_stream import VideoDepthAnything
        model = VideoDepthAnything(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
        model.to('cuda').eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()
        shapes = []
        original_features = model.forward_features

        def checked_features(tensor):
            if tensor.device.type != 'cuda' or tensor.shape[1] != 1:
                raise RuntimeError('Streaming feature input must be exactly one current CUDA RGB frame')
            shapes.append(list(tensor.shape))
            return original_features(tensor)

        model.forward_features = checked_features
        result.update(backend='torch-cuda', framework=f'torch-{torch.__version__}',
                      device=torch.cuda.get_device_name(0), precision='autocast-float16',
                      source_revision=revision, source_files_sha256=source_files,
                      depth_coordinate='optical-Z metres: official run.py metric point-cloud sets z=depth, points=(x*z,y*z,z)',
                      weight_sha256=WEIGHT_SHA256, weight_bytes=WEIGHT_BYTES,
                      weight_model_revision=metadata['sha'], actual_weight_license='Apache-2.0',
                      checkpoint_training='Official single metric model trained on Virtual KITTI and IRS; no outdoor-only variant',
                      sequence_receipt_sha256=sha(receipt_path), input_size=input_size,
                      K_input='NOT_SUPPORTED_BY_OFFICIAL_VDA_API; public K available to shared downstream geometry',
                      public_K_common=common_k.tolist(), common_shape=list(common_shape),
                      GT_scale_fit=False, future_frames_read=False,
                      causality='Official infer_video_depth_one; one RGB current frame + previously computed attention hidden states; first-frame cache replicated for warmup',
                      scope='Already-consumed Development real RGB; metric estimates are not independent physical truth, body collision or mobile latency evidence')
        for index, selected_row in enumerate(selected):
            elapsed = time.perf_counter()-start
            if elapsed >= budget_s:
                raise TimeoutError('Cumulative VDA startup/inference wall budget reached')
            path = Path(selected_row['path'])
            if sha(path) != selected_row['sha256']:
                raise ValueError('RGB hash differs from existing acquisition receipt')
            with Image.open(path) as image:
                frame = np.asarray(image.convert('RGB'))
            torch.cuda.synchronize()
            infer_start = time.perf_counter()
            depth = model.infer_video_depth_one(frame, input_size=input_size, device='cuda', fp32=False)
            torch.cuda.synchronize()
            infer_seconds = time.perf_counter()-infer_start
            if depth.shape != frame.shape[:2] or not np.isfinite(depth).all():
                raise ValueError('Unexpected official native output shape or nonfinite prediction')
            if model.id != index or max(model.frame_id_list) > index:
                raise ValueError('Temporal cache includes a future frame identity')
            destination = output/f'vda_{selected_row["frame"]:06d}.npz'
            # Lossless uncompressed arrays keep streaming wall cost visible and low.
            np.savez(destination, depth=depth.astype(np.float32, copy=False))
            common_depth = np.asarray(Image.fromarray(depth).resize((1280, 720), Image.Resampling.BILINEAR))
            common_destination = output/f'vda_{selected_row["frame"]:06d}_common.npz'
            np.savez(common_destination, depth=common_depth.astype(np.float32, copy=False))
            predictions.append(dict(session=plan['session'], camera=plan['camera'], frame=selected_row['frame'],
                                    arm='vda_metric_causal', depth_path=str(common_destination),
                                    depth_sha256=sha(common_destination), rgb_sha256=selected_row['sha256'],
                                    shape=list(common_shape), public_K=common_k.tolist(),
                                    valid_fraction=float((np.isfinite(common_depth) & (common_depth > 0)).mean()),
                                    readout=visible_readout(common_depth, common_k, common_shape)))
            row = dict(frame=selected_row['frame'], index=index, rgb_path=str(path),
                       rgb_sha256=selected_row['sha256'], depth_path=str(destination),
                       depth_sha256=sha(destination), depth_shape=list(depth.shape),
                       inference_seconds=infer_seconds, nominal_time_s=index/15.,
                       finite_fraction=float(np.isfinite(depth).mean()),
                       positive_fraction=float((depth > 0).mean()),
                       median_depth_m=float(np.median(depth)),
                       stream_cache_frame_ids=list(model.frame_id_list), feature_input_shape=shapes[-1])
            rows.append(row)
            write(output/'progress.json', dict(completed=len(rows), total=45, stage='causal_streaming',
                                              seconds=time.perf_counter()-start))
            print(f'{len(rows)}/45 frame={selected_row["frame"]} infer={infer_seconds:.3f}s', flush=True)
            if len(rows) == 3:
                elapsed = time.perf_counter()-start
                estimated_remaining = np.median([r['inference_seconds'] for r in rows])*42
                write(output/'prefix_smoke.json', dict(status='PASS', frames=3, elapsed_s=elapsed,
                                                     projected_remaining_inference_s=float(estimated_remaining),
                                                     all_feature_calls_single_frame=True,
                                                     cache_ids_all_past_or_present=True))
                if elapsed + estimated_remaining >= budget_s:
                    raise TimeoutError('Prefix measured remaining inference cannot fit budget; no full run')
        result.update(status='COMPLETE', frames=len(rows), calls=len(shapes),
                      all_feature_calls_single_frame=all(s[1] == 1 for s in shapes),
                      measured_inference_seconds=sum(r['inference_seconds'] for r in rows),
                      median_call_seconds=float(np.median([r['inference_seconds'] for r in rows])),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      wall_seconds=time.perf_counter()-start)
    except Exception as error:
        result.update(status='FAILED', error=repr(error), traceback=traceback.format_exc(),
                      frames=len(rows), wall_seconds=time.perf_counter()-start)
        print(result['traceback'], flush=True)
    finally:
        if model is not None:
            model.frame_cache_list.clear()
            model.frame_id_list.clear()
            model.forward_features = None
            model = None
        original_features = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            result['allocated_bytes_after_cleanup'] = torch.cuda.memory_allocated()
        result['terminal_wall_seconds'] = time.perf_counter()-start
        write(output/'vda_stream_receipt.json', result)
        write(output/'prediction_manifest.json', dict(status=result['status'], arm='vda_metric_causal',
                                                     rows=predictions, common_shape=[720,1280],
                                                     resize='PIL BILINEAR depth; integer pixel centers half-pixel K',
                                                     source_receipt_sha256=sha(output/'vda_stream_receipt.json')))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows', 'traceback']}), flush=True)
    return result['status'] == 'COMPLETE'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=240)
    parser.add_argument('--input-size', type=int, default=518)
    args = parser.parse_args()
    if not 0 < args.budget_s <= 240 or args.input_size != 518:
        parser.error('This phase permits <=240 GPU wall seconds and fixed official 518 input')
    sys.exit(0 if run(args.sequence, args.output, args.budget_s, args.input_size) else 1)

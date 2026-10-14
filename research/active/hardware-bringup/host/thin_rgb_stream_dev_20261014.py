"""Full RGB-clock, label-blind thin-structure input stream, consumed Development.

References original JPEGs without copying them. Old depth reuse is exact
session/RGB-index identity, never nearest-frame filling. Only candidates are
produced here; track association, metric projection and decisions are downstream.
"""
from __future__ import annotations

import hashlib
import json
import argparse
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from atom_camera import AtomCamera, ROOT
import real_rgb_query_feasibility_dev_20261012 as F
from thin_structure_depth_ridge_dev_20261013 import propose

OUT = ROOT / 'artifacts.local/work/thin-continuity-dev-20261014/input'
OLD = ROOT / 'artifacts.local/work/fusion-refinement-dev-20261013'
KEYS = ('073554', '073635', '073740', '073821')
EXPECTED = dict(zip(KEYS, (102, 70, 104, 378)))
WEIGHTS = ROOT / 'artifacts.local/models/depth-anything--Depth-Anything-V2-Small/depth_anything_v2_vits.pth'
GPU_LIMIT = 450.
CPU_LIMIT = 600.
BYTE_LIMIT = 600 * 1024 ** 2
RIDGE = dict(side_offsets=[4, 8, 12], normal_directions=4, minimum_prominence_iqr=.02,
             hough_threshold=20, min_length=25, max_gap=8, min_support=.6, top_k=40)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def write_json(name, value):
    (OUT / name).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf8')


def identify_frames(rgb, key, session):
    """Use capture clock when present, separately flag receipt and index issues."""
    seen = {}
    prev = None
    rows = []
    for j, original in enumerate(rgb):
        device_ns = original.get('camera_timestamp_ns')
        device_us = original.get('device_capture_timestamp_us')
        if device_us is not None:
            assert device_ns in (None, int(device_us) * 1000)
            device_ns = int(device_us) * 1000
        domain = original.get('camera_clock_domain') if device_ns is not None else 'ANDROID_RECEIPT_MONOTONIC'
        t = int(device_ns) if device_ns is not None else int(original['received_elapsed_ns'])
        boot = original.get('boot_id')
        source = original.get('source_id')
        capture_id = (boot if boot is not None else source, domain, t)
        duplicate = capture_id in seen
        reasons = []
        gap = None
        receipt_gap = None
        if prev is None:
            reasons.append('FIRST_FRAME')
        else:
            receipt_gap = (int(original['received_elapsed_ns']) - prev['received_elapsed_ns']) / 1e9
            if (boot, source, domain) != (prev['boot_id'], prev['source_id'], prev['time_domain']):
                reasons.append('BOOT_SOURCE_OR_CLOCK_CHANGE')
            else:
                gap = (t - prev['time_ns']) / 1e9
                if gap < 0:
                    reasons.append('CAPTURE_CLOCK_BACKWARDS')
                elif gap == 0:
                    reasons.append('DUPLICATE_CAPTURE_TIME')
                elif gap > .5:
                    reasons.append('CAPTURE_GAP_GT_0_5S')
            if receipt_gap <= 0:
                reasons.append('RECEIPT_NONINCREASING')
        row = dict(key=key, session=session, rgb_index=j, index_record=original['index'],
                   rgb_file=str(F.RAW / session / original['file']), rgb_sha256=original['sha256'],
                   bytes=original['bytes'], device_capture_us=device_us, device_capture_ns=device_ns,
                   received_elapsed_ns=original['received_elapsed_ns'], boot_id=boot, source_id=source,
                   source_frame_id=original.get('source_frame_id'), time_ns=t, time_domain=domain,
                   clock_gap_s=gap, receipt_gap_s=receipt_gap, duplicate_capture=duplicate,
                   duplicate_of_rgb_index=seen.get(capture_id), clock_events=reasons,
                   temporal_reset=any(reason != 'DUPLICATE_CAPTURE_TIME' for reason in reasons),
                   capture_time_semantics=original.get('camera_timestamp_semantics', 'UNKNOWN_EXPOSURE_SEMANTICS'),
                   coordinate_system='undistorted upright x-right/y-down 480x640; AtomCamera transpose mapping',
                   metric_distance_verified=False, safety_decision=None)
        seen.setdefault(capture_id, j)
        rows.append(row)
        prev = row
    assert len(rows) == EXPECTED[key]
    return rows


def prepare_manifest():
    old_path = OLD / 'cache_manifest.json'
    old = json.loads(old_path.read_text())
    assert sha(WEIGHTS) == old['weights_sha256']
    camera = AtomCamera()
    sessions = {}
    frame_rows = []
    multiple_checks = []
    verified_cache_arrays = {}
    for key in KEYS:
        s = old['sessions'][key]
        index_path = F.RAW / s['session'] / 'rgb.jsonl'
        rgb = list(F.lines(index_path))
        rows = identify_frames(rgb, key, s['session'])
        for row in rows:
            assert Path(row['rgb_file']).stat().st_size == row['bytes']
            assert sha(row['rgb_file']) == row['rgb_sha256']
        links = defaultdict(list)
        caches = defaultdict(list)
        for r in s['frames']:
            assert r['rgb_t_ns'] == rgb[r['rgb_index']]['received_elapsed_ns']
            links[r['rgb_index']].append(dict(tof_index=r['i'], matched=r['matched'], pose_ok=r['pose_ok'],
                                             nearest_receipt_delta_ms=(r['rgb_t_ns'] - r['t_ns']) / 1e6,
                                             existing_depth=bool(r.get('cache'))))
            if r.get('cache'):
                caches[r['rgb_index']].append(OLD / r['cache'])
        for row in rows:
            j = row['rgb_index']
            row['old_tof_links'] = links[j]
            row['old_tof_ready_link_available'] = bool(links[j])
            row['old_exact_cache_paths'] = [str(p) for p in caches[j]]
            row['depth_reuse_status'] = 'EXACT_RGB_INDEX' if caches[j] else 'NEW_FROZEN_MODEL_REQUIRED'
            if caches[j]:
                reference = None
                array_sha = None
                for path in caches[j]:
                    with np.load(path) as z:
                        a = z['disp'].copy()
                    assert a.shape == (640, 480) and a.dtype == np.float32 and np.isfinite(a).all()
                    if reference is None:
                        reference = a
                        array_sha = hashlib.sha256(a.tobytes()).hexdigest()
                    else:
                        assert np.array_equal(a.view(np.uint32), reference.view(np.uint32)), (key, j, 'multiple old caches disagree')
                row['old_exact_cache_array_sha256'] = array_sha
                if len(caches[j]) > 1:
                    multiple_checks.append(dict(key=key, rgb_index=j, old_cache_count=len(caches[j]), raw_bits_equal=True))
                verified_cache_arrays[key, j] = reference
        sessions[key] = dict(session=s['session'], frames=len(rows), raw_index=str(index_path), raw_index_sha256=sha(index_path),
                             old_ready_frames=len(s['frames']), exact_old_cache_rgb_indices=sum(bool(paths) for paths in caches.values()),
                             duplicate_captures=sum(r['duplicate_capture'] for r in rows),
                             clock_event_counts=dict(Counter(e for r in rows for e in r['clock_events'])))
        frame_rows.extend(rows)
    assert len(frame_rows) == 654
    manifest = dict(frames=frame_rows, sessions=sessions, complete_original_rgb_manifest=True,
                    annotation_files_read=False, old_session_label_fields_used=False,
                    weights_sha256=old['weights_sha256'], old_manifest_sha256=sha(old_path),
                    K_upright=camera.K_upright.tolist(), coordinate_system='undistorted upright 480x640',
                    camera_calibration_sha256=sha(ROOT / 'artifacts.local/work/atom-camera-calibration-20261012/calibration.json'),
                    multiple_old_cache_checks=multiple_checks, candidate_config=RIDGE,
                    timestamp_contract='device camera-reported capture preferred within its boot; unmapped, exposure timing unverified; no ToF timing gate',
                    frames_jsonl=str(OUT / 'frames.jsonl'))
    write_json('manifest.json', manifest)
    return manifest, verified_cache_arrays, camera


def validate_completed(rows, manifest):
    expected = {(r['key'], r['rgb_index']) for r in manifest['frames']}
    observed = {(r['key'], r['rgb_index']) for r in rows}
    assert len(rows) == 654 and observed == expected
    assert all(r['annotation_files_read'] is False for r in rows)
    assert all(r['metric_distance_verified'] is False and r['safety_decision'] is None for r in rows)
    assert all(r['depth_available'] for r in rows)
    for r in rows:
        if r['depth_source'] == 'EXACT_OLD_RGB_INDEX':
            assert r['depth_cache'] in r['old_exact_cache_paths']
        assert all(0 <= c['x1'] < 480 and 0 <= c['x2'] < 480 and 0 <= c['y1'] < 640 and 0 <= c['y2'] < 640 for c in r['candidates'])
        assert all('track_id' not in c and 'hits' not in c for c in r['candidates'])
    return dict(all_654_original_indices=True, exact_index_reuse_only=True, no_annotation_access=True,
                all_frames_depth_available=True, all_candidate_endpoints_in_frame=True,
                independent_observation_no_metric_or_alarm=True, no_producer_track_ids=True)


def verify_existing():
    """A narrow identity/pairing check; no model loading or annotation access."""
    start = time.perf_counter()
    manifest = json.loads((OUT / 'manifest.json').read_text())
    rows = [json.loads(line) for line in (OUT / 'frames.jsonl').read_text().splitlines()]
    checks = validate_completed(rows, manifest)
    lookup = {(r['key'], r['rgb_index']): r for r in rows}
    old = json.loads((OLD / 'cache_manifest.json').read_text())
    old_rows = {}
    path = ROOT / 'artifacts.local/work/thin-structure-dev-20261013/detection_ridge_frames.jsonl'
    for line in path.open(encoding='utf8'):
        r = json.loads(line)
        if r['key'] in KEYS:
            old_rows[r['key'], r['i']] = r
    compared = 0
    for key in KEYS:
        for meta in old['sessions'][key]['frames']:
            if not meta.get('cache'):
                continue
            prior = old_rows[key, meta['i']]['candidates']
            current = lookup[key, meta['rgb_index']]['candidates']
            assert len(prior) == len(current), (key, meta['i'], 'candidate count')
            for a, b in zip(prior, current):
                assert all(a[field] == value for field, value in b.items()), (key, meta['i'], 'candidate values')
            compared += 1
    for r in rows:
        with np.load(r['depth_cache']) as z:
            a = z['disp']
            assert a.dtype == np.float32 and a.shape == (640, 480) and np.isfinite(a).all()
            assert hashlib.sha256(a.tobytes()).hexdigest() == r['disparity_sha256']
    checks.update(old_same_image_candidate_rows_exact=compared, all_654_depth_locator_shape_and_hash=True)
    summary = dict(checks=checks, verification_wall_s=time.perf_counter() - start,
                   source_sha256=sha(__file__), frames_sha256=sha(OUT / 'frames.jsonl'),
                   source_generation_receipt=str(OUT / 'receipt.json'),
                   annotation_files_read=False, no_new_gpu_inference=True,
                   sessions={key: dict(frames=sum(r['key'] == key for r in rows),
                               candidates=sum(len(r['candidates']) for r in rows if r['key'] == key),
                               exact_old_depth=sum(r['key'] == key and r['depth_source'] == 'EXACT_OLD_RGB_INDEX' for r in rows),
                               new_depth=sum(r['key'] == key and r['depth_source'] == 'NEW_FROZEN_MODEL_SAME_RGB' for r in rows),
                               duplicate_captures=sum(r['key'] == key and r['duplicate_capture'] for r in rows),
                               capture_gaps_gt_0_5s=[dict(rgb_index=r['rgb_index'], gap_s=r['clock_gap_s']) for r in rows if r['key'] == key and 'CAPTURE_GAP_GT_0_5S' in r['clock_events']]) for key in KEYS})
    write_json('verification-final.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-preflight', action='store_true')
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    if args.verify_existing:
        verify_existing()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'frames.jsonl').exists():
        raise FileExistsError('Existing producer output retained; no implicit overwrite')
    prior = dict(gpu_command_wall_s=0., wall_s=0., model_inference_s=0.)
    prior_benchmark = None
    if args.resume_preflight:
        prior = json.loads((OUT / 'receipt.json').read_text())
        assert prior['complete'] is False and prior['completed_frames'] == 0
        prior_benchmark = json.loads((OUT / 'benchmark.json').read_text())
        for name in ('receipt.json', 'benchmark.json', 'progress.json', 'manifest.json'):
            retained = OUT / (Path(name).stem + '-initial-preflight.json')
            if not retained.exists():
                shutil.copy2(OUT / name, retained)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(0)
    manifest, old_arrays, camera = prepare_manifest()
    missing = [r for r in manifest['frames'] if not r['old_exact_cache_paths']]
    print(json.dumps(dict(event='MANIFEST_READY', frames=654, old_exact=len(old_arrays), new_required=len(missing), sessions=manifest['sessions'])), flush=True)
    import torch
    import real_rgb_small_model_dev_20261012 as S
    torch.set_num_threads(2)
    model = None
    gpu_start = None
    gpu_command_wall = 0.
    inference_s = 0.
    generated_bytes = 0
    new_outputs = {}
    completed = []
    failures = []
    benchmark = []
    counts = Counter()
    backend = {}

    def infer(row):
        nonlocal inference_s, generated_bytes
        if time.perf_counter() - gpu_start + prior['gpu_command_wall_s'] > GPU_LIMIT - 8:
            raise TimeoutError('GPU budget reserve reached; no implicit frame subsampling')
        raw = cv2.imread(row['rgb_file'])
        if raw is None:
            raise ValueError(f'Raw JPEG decode failed: {row["rgb_file"]}')
        assert sha(row['rgb_file']) == row['rgb_sha256']
        upright = camera.undistort_upright(raw)
        t = time.perf_counter()
        with torch.inference_mode():
            disparity = model.infer_image(upright, 518)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        inference_s += elapsed
        assert disparity.shape == (640, 480) and disparity.dtype == np.float32 and np.isfinite(disparity).all()
        path = OUT / 'depth' / row['key'] / f'{row["rgb_index"]:04d}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, disp=disparity)
        generated_bytes += path.stat().st_size
        if generated_bytes > BYTE_LIMIT - 20 * 1024 ** 2:
            raise RuntimeError('Payload reserve reached; retain explicitly incomplete output')
        new_outputs[row['key'], row['rgb_index']] = (disparity, str(path), elapsed)
        return disparity, str(path), elapsed

    try:
        gpu_start = time.perf_counter()
        model, params = S.load_dav2_small(str(WEIGHTS))
        torch.cuda.synchronize()
        backend = dict(torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                       device=torch.cuda.get_device_name(), device_index=torch.cuda.current_device(),
                       parameter_dtype=str(next(model.parameters()).dtype), parameters=params,
                       output_dtype='float32', infer_image_input_size=518,
                       allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                       cudnn_benchmark=torch.backends.cudnn.benchmark)
        load_s = time.perf_counter() - gpu_start
        if prior_benchmark is not None:
            benchmark = prior_benchmark['samples']
            for b in benchmark:
                row = next(r for r in missing if r['key'] == b['key'] and r['rgb_index'] == b['rgb_index'])
                path = OUT / 'depth' / row['key'] / f'{row["rgb_index"]:04d}.npz'
                with np.load(path) as z:
                    a = z['disp'].copy()
                assert a.shape == (640, 480) and np.isfinite(a).all()
                new_outputs[row['key'], row['rgb_index']] = (a, str(path), b['inference_s'])
        else:
            for key in ('073554', '073635', '073821'):
                row = next(r for r in missing if r['key'] == key)
                t = time.perf_counter()
                disparity, path, execution = infer(row)
                cs, n = propose(disparity)
                benchmark.append(dict(key=key, rgb_index=row['rgb_index'], inference_s=execution,
                                      full_infer_save_and_propose_s=time.perf_counter() - t,
                                      payload_bytes=Path(path).stat().st_size, candidates=len(cs), proposal_count=n))
        # CUDA first execute is a once-per-residency cost, not per-frame steady state.
        warm = benchmark[1:]
        cost = max(b['full_infer_save_and_propose_s'] for b in warm)
        cold_extra = max(0., benchmark[0]['full_infer_save_and_propose_s'] - cost)
        old_cost = max(.06, max(b['full_infer_save_and_propose_s'] - b['inference_s'] for b in warm))
        estimated = prior['gpu_command_wall_s'] + load_s + cold_extra + len(missing) * cost + len(old_arrays) * old_cost
        write_json('benchmark.json', dict(samples=benchmark, load_s=load_s, backend=backend,
                     missing_count=len(missing), old_exact_count=len(old_arrays), projected_gpu_residency_wall_s=estimated,
                     gpu_budget_s=GPU_LIMIT, one_model_residency_per_attempt=True,
                     prior_preflight_gpu_wall_s=prior['gpu_command_wall_s'], cold_execute_separate=True,
                     resumed_without_reinferring_three_benchmark_images=prior_benchmark is not None))
        print(json.dumps(dict(event='BENCHMARK', projected_wall_s=estimated, samples=benchmark, backend=backend)), flush=True)
        if estimated > GPU_LIMIT - 10:
            raise RuntimeError('Projected complete stream exceeds GPU budget; report plan to root before selecting another completeness strategy')
        with (OUT / 'frames.jsonl').open('w', encoding='utf8') as output:
            for meta in manifest['frames']:
                if time.perf_counter() - start + prior['wall_s'] > CPU_LIMIT - 10:
                    raise TimeoutError('CPU command-wall reserve reached; full manifest retained')
                key = meta['key'], meta['rgb_index']
                t = time.perf_counter()
                if meta['old_exact_cache_paths']:
                    disparity = old_arrays[key]
                    path = meta['old_exact_cache_paths'][0]
                    source = 'EXACT_OLD_RGB_INDEX'
                    execution = 0.
                else:
                    if key in new_outputs:
                        disparity, path, execution = new_outputs.pop(key)
                    else:
                        disparity, path, execution = infer(meta)
                        new_outputs.pop(key)
                    source = 'NEW_FROZEN_MODEL_SAME_RGB'
                cs, n = propose(disparity)
                row = dict(meta, depth_available=True, depth_cache=path, depth_source=source,
                           disparity_sha256=hashlib.sha256(disparity.tobytes()).hexdigest(),
                           depth_inference_s=execution, candidates=cs, raw_ridge_proposals=n,
                           annotation_files_read=False, candidate_parameters_changed=False,
                           depth_from_previous_preflight=prior_benchmark is not None and any(b['key'] == key[0] and b['rgb_index'] == key[1] for b in benchmark),
                           producer_work_s=time.perf_counter() - t)
                output.write(json.dumps(clean(row), allow_nan=False) + '\n')
                output.flush()
                completed.append(row)
                counts[source] += 1
                counts['candidate_observations'] += len(cs)
                if meta['rgb_index'] == EXPECTED[meta['key']] - 1 or len(completed) % 100 == 0:
                    write_json('progress.json', dict(completed_frames=len(completed), original_frames=654,
                                wall_s=time.perf_counter() - start, gpu_residency_wall_s=time.perf_counter() - gpu_start,
                                model_inference_s=inference_s, counts=counts, last_key=key, complete=False))
                    print(f'progress {len(completed)}/654 wall {time.perf_counter()-start:.2f}s gpu {time.perf_counter()-gpu_start:.2f}s', flush=True)
        checks = validate_completed(completed, manifest)
        write_json('verification.json', dict(checks=checks, frames=654, multiple_cache_checks=manifest['multiple_old_cache_checks']))
    except Exception as error:
        failures.append(dict(type=type(error).__name__, message=str(error), completed_frames=len(completed)))
        print(json.dumps(dict(event='FAILURE', failure=failures[-1])), flush=True)
        raise
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()
        gpu_command_wall = time.perf_counter() - gpu_start if gpu_start is not None else 0.
        receipt = dict(complete=len(completed) == 654 and not failures, completed_frames=len(completed), original_frames=654,
                       wall_s=time.perf_counter() - start, gpu_command_wall_s=gpu_command_wall,
                       model_inference_s=inference_s, GPU_released=True, backend=backend,
                       gpu_command_wall_s_including_prior=gpu_command_wall + prior['gpu_command_wall_s'],
                       cpu_command_wall_s_including_prior=time.perf_counter() - start + prior['wall_s'],
                       model_inference_s_including_prior=inference_s + prior['model_inference_s'],
                       failure_history_retained='initial-preflight files and original run.log' if args.resume_preflight else None,
                       source_path=str(Path(__file__).resolve()), source_sha256=sha(__file__),
                       candidate_source_sha256=sha(Path(__file__).parent / 'thin_structure_depth_ridge_dev_20261013.py'),
                       no_label_inputs=True, no_training=True, no_download=True, counts=counts, failures=failures,
                       payload_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
                       original_manifest=str(OUT / 'manifest.json'), frames_jsonl=str(OUT / 'frames.jsonl'))
        write_json('receipt.json', receipt)
        write_json('progress.json', receipt)


if __name__ == '__main__':
    main()

"""Consumed local Development stage audit; weak labels enter evaluation only.

No detection/track/scale/alert policy is changed. The independent producer chooses
the old six uniform positions in four sessions, with no annotation access.
Centerline/tube observations are oracle diagnostics, never shaft masks or truth.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from atom_camera import AtomCamera, ROOT
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_query_feasibility_dev_20261012 as F
from thin_structure_evaluate_dev_20261013 import matches
from thin_structure_geometry_dev_20261013 import inside_boxes

OUT = ROOT / 'artifacts.local/work/thin-information-dev-20261014/stage'
CACHE = ROOT / 'artifacts.local/work/fusion-refinement-dev-20261013'
THIN = ROOT / 'artifacts.local/work/thin-structure-dev-20261013'
TRACK = ROOT / 'artifacts.local/work/thin-local-geometry-dev-20261013'
CONF = ROOT / 'artifacts.local/work/thin-confidence-dev-20261013'
UPSTREAM = ROOT / 'artifacts.local/work/rgb-depth-backbone-compare-dev-20261010/upstream/Depth-Anything-V2'
KEYS = ('073554', '073635', '073740', '073821')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, (np.bool_, np.integer)):
            return item.item()
        return item
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf8')


def selected_rows(manifest):
    # Matches the existing pre-annotation sampling exactly, without labels.
    out = []
    for key in KEYS:
        session = manifest['sessions'][key]
        frames = session['frames']
        picks = np.rint(np.linspace(.05, .95, 6) * (len(frames) - 1)).astype(int)
        for order, j in enumerate(picks):
            out.append(dict(key=key, session=session['session'], order=order, **frames[j]))
    return out


def read_selected(path, keys):
    selected = {}
    with path.open(encoding='utf8') as f:
        for line in f:
            row = json.loads(line)
            ident = row['key'], row['i']
            if ident in keys:
                selected[ident] = row
    assert selected.keys() == keys, (str(path), keys - selected.keys())
    return selected


def load_transform():
    path = UPSTREAM / 'depth_anything_v2/util/transform.py'
    spec = importlib.util.spec_from_file_location('thin_stage_official_transform', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    resize = mod.Resize(518, 518, resize_target=False, keep_aspect_ratio=True,
                        ensure_multiple_of=14, resize_method='lower_bound',
                        image_interpolation_method=cv2.INTER_CUBIC)
    return resize, mod


def rgb_lines(image, target_shape):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if lines is None:
        return []
    sy, sx = image.shape[0] / target_shape[0], image.shape[1] / target_shape[1]
    result = []
    for line in lines[:, 0]:
        p = line.reshape(2, 2).astype(float)
        p[:, 0] = (p[:, 0] + .5) / sx - .5
        p[:, 1] = (p[:, 1] + .5) / sy - .5
        p = np.clip(p, [0, 0], [target_shape[1] - 1, target_shape[0] - 1])
        length = float(np.linalg.norm(p[1] - p[0]))
        if length >= 25:
            result.append(dict(x1=p[0, 0], y1=p[0, 1], x2=p[1, 0], y2=p[1, 1], length=length))
    return sorted(result, key=lambda c: -c['length'])


def generate(no_native=False, reuse_native=False):
    """Seal label-blind observations before any weak-label read."""
    import torch
    import torch.nn.functional as fn
    import real_rgb_small_model_dev_20261012 as S
    start = time.perf_counter()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    manifest = json.loads((CACHE / 'cache_manifest.json').read_text())
    sampled = selected_rows(manifest)
    idents = {(r['key'], r['i']) for r in sampled}
    inherited_paths = dict(ridge=THIN / 'detection_ridge_frames.jsonl',
                           chain=TRACK / 'camera_chain_frames.jsonl',
                           geometry=TRACK / 'geometry_frames.jsonl',
                           distance=CONF / 'distance_states.jsonl')
    inherited = {name: read_selected(path, idents) for name, path in inherited_paths.items()}
    readout = json.loads((CACHE / 'robust_readout.json').read_text())
    fits = {(k, r['i']): r for k, s in readout['sessions'].items()
            for r in s['arms']['robust_replace'] if (k, r['i']) in idents}
    weights = ROOT / 'artifacts.local/models/depth-anything--Depth-Anything-V2-Small/depth_anything_v2_vits.pth'
    assert sha(weights) == manifest['weights_sha256']
    cam = AtomCamera()
    resize, mod = load_transform()
    model = None
    gpu_start = None
    execution_s = 0.
    rows = []
    failures = []
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        if not no_native and not reuse_native:
            gpu_start = time.perf_counter()
            model, _ = S.load_dav2_small(str(weights))
            torch.cuda.synchronize()
        rgb_indices = {k: list(F.lines(F.RAW / manifest['sessions'][k]['session'] / 'rgb.jsonl')) for k in KEYS}
        for meta in sampled:
            if time.perf_counter() - start > 780:
                raise TimeoutError('CPU reserve: stop before 900 command-wall seconds')
            key, i = meta['key'], meta['i']
            ident = key, i
            filename = F.RAW / meta['session'] / rgb_indices[key][meta['rgb_index']]['file']
            raw = cv2.imread(str(filename))
            if raw is None:
                raise ValueError(f'Raw decode failed: {filename}')
            up = cam.undistort_upright(raw)
            # The real infer_image transforms BGR -> RGB float64 -> official cubic.
            rgb_float = cv2.cvtColor(up, cv2.COLOR_BGR2RGB) / 255.
            resized = resize({'image': rgb_float})['image']
            prepared_bgr = np.clip(resized[..., ::-1] * 255, 0, 255).astype(np.uint8)
            assert resized.shape == (686, 518, 3)
            normalized = mod.NormalizeImage([.485, .456, .406], [.229, .224, .225])({'image': resized})
            tensor = mod.PrepareForNet()(normalized)['image']
            assert tensor.shape == (3, 686, 518) and tensor.dtype == np.float32
            old_depth = valid = native = returned = None
            if meta.get('cache'):
                with np.load(CACHE / meta['cache']) as payload:
                    old_depth, valid = payload['disp'].copy(), payload['valid'].copy()
            arraypath = OUT / f'blind_{key}_{i:04d}.npz'
            if reuse_native:
                with np.load(arraypath) as saved:
                    assert np.array_equal(saved['upright'], up)
                    assert np.array_equal(saved['prepared'], prepared_bgr)
                    native, returned = saved['native_disparity'].copy(), saved['returned_disparity'].copy()
                    if old_depth is not None:
                        assert np.array_equal(saved['cached_disparity'], old_depth)
            row = dict(key=key, i=i, order=meta['order'], meta=meta,
                       raw_file=str(filename), raw_sha256=sha(filename),
                       cache_file=str(CACHE / meta['cache']) if meta.get('cache') else None,
                       cache_sha256=sha(CACHE / meta['cache']) if meta.get('cache') else None,
                       prepared_shape=list(resized.shape), input_source_shape=list(up.shape),
                       rgb_raw_lines=rgb_lines(up, up.shape), rgb_prepared_lines=rgb_lines(prepared_bgr, up.shape),
                       fit= fits[ident], inherited={name: data[ident] for name, data in inherited.items()},
                       annotations_read=False, native_status='NOT_RUN')
            if model is not None:
                if time.perf_counter() - gpu_start > 135:
                    raise TimeoutError('GPU reserve: stop before 150 command-wall seconds')
                t0 = time.perf_counter()
                with torch.inference_mode():
                    y = model(torch.from_numpy(tensor).unsqueeze(0).cuda())
                    z = fn.interpolate(y[:, None], (640, 480), mode='bilinear', align_corners=True)[0, 0]
                torch.cuda.synchronize()
                execution_s += time.perf_counter() - t0
                native, returned = y[0].cpu().numpy(), z.cpu().numpy()
            if native is not None:
                row.update(native_status='AVAILABLE', native_shape=list(native.shape),
                           cache_parity=dict(status='NOT_EVALUABLE_NO_OLD_CACHE'))
                if old_depth is not None:
                    iqr = max(float(np.percentile(old_depth, 75) - np.percentile(old_depth, 25)), 1e-5)
                    delta = np.abs(returned - old_depth)
                    row['cache_parity'] = dict(status='AVAILABLE', raw_bits_equal=bool(np.array_equal(returned.view(np.uint32), old_depth.view(np.uint32))),
                                               max_abs=float(delta.max()), max_abs_over_old_iqr=float(delta.max() / iqr),
                                               within_1e_5_iqr=bool(delta.max() / iqr <= 1e-5))
            payload = dict(upright=up, prepared=prepared_bgr, prepared_gray_float=np.dot(resized, [.299, .587, .114]))
            if old_depth is not None:
                payload.update(cached_disparity=old_depth, valid=valid)
            if native is not None:
                payload.update(native_disparity=native, returned_disparity=returned)
            if not reuse_native:
                np.savez_compressed(arraypath, **payload)
            row['blind_payload'] = str(arraypath)
            rows.append(row)
            print(f'blind {key}/{i}: raw lines {len(row["rgb_raw_lines"])} -> {len(row["rgb_prepared_lines"])}; native {row["native_status"]}', flush=True)
    except Exception as error:
        failures.append(dict(type=type(error).__name__, message=str(error), completed=len(rows)))
        raise
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()
        gpu_wall = time.perf_counter() - gpu_start if gpu_start is not None else 0.
        sources = {name: dict(path=str(path), sha256=sha(path)) for name, path in inherited_paths.items()}
        result = dict(rows=rows, generation_wall_s=time.perf_counter() - start, gpu_command_wall_s=gpu_wall,
                      gpu_model_execution_s=execution_s, annotations_read=False, failures=failures,
                      sources=sources, weights_sha256=manifest['weights_sha256'],
                      transform_sha256=sha(UPSTREAM / 'depth_anything_v2/util/transform.py'),
                      unknown_serialization='nonfinite inherited fields preserved as explicit null; no missing values become negatives',
                      native_payload_reused_without_new_inference=reuse_native,
                      first_attempt_failure='JSON serialization encountered inherited NaN after all 24 native payloads were saved; original run.log retained' if reuse_native else None,
                      native_output_scope='same immutable model and old selection; not fresh detector validation')
        write_json(OUT / 'blind_features.json', result)
    return result


def line_points(segment, count=40):
    p = np.asarray(segment, float).reshape(2, 2)
    q = p[0] + np.linspace(.1, .9, count)[:, None] * (p[1] - p[0])
    direction = p[1] - p[0]
    direction /= np.linalg.norm(direction)
    return q, np.array([-direction[1], direction[0]])


def profile(array, segment, coord='original'):
    q, normal = line_points(segment)
    offsets = np.arange(-8, 9)
    positions = q[:, None] + offsets[None, :, None] * normal
    if coord == 'input':
        positions = (positions + .5) * [518 / 480, 686 / 640] - .5
    elif coord == 'native':
        positions *= [(518 - 1) / (480 - 1), (686 - 1) / (640 - 1)]
    bounded = ((positions[..., 0] >= 0) & (positions[..., 0] < array.shape[1] - 1) &
               (positions[..., 1] >= 0) & (positions[..., 1] < array.shape[0] - 1)).all(axis=1)
    if bounded.sum() < 8:
        return dict(status='UNKNOWN_BORDER', valid_sample_count=int(bounded.sum()))
    positions = positions[bounded]
    values = cv2.remap(np.asarray(array, np.float32), positions[..., 0].astype(np.float32),
                       positions[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    bg = np.maximum(values[:, 2], values[:, 14])  # inherited +/-6 pixel surroundings
    iqr = max(float(np.percentile(array, 75) - np.percentile(array, 25)), 1e-5)
    median = np.median(values - bg[:, None], axis=0) / iqr
    best = int(np.argmax(median[5:12]) + 5)
    background = (values[:, 0] + values[:, 16]) / 2
    contrast = np.max(np.abs(values[:, 5:12] - background[:, None]), axis=1)
    return dict(status='AVAILABLE', valid_sample_count=int(bounded.sum()), center_prominence_iqr=float(median[8]),
                best_3px_prominence_iqr=float(median[best]), best_offset_px=int(offsets[best]),
                center_positive_002=bool(median[8] >= .02), best_3px_positive_002=bool(median[best] >= .02),
                line_vs_8px_background_contrast=float(np.median(contrast)), iqr=iqr)


def tube_mask(shape, segment, radius):
    p = np.asarray(segment, float).reshape(2, 2)
    y, x = np.mgrid[:shape[0], :shape[1]]
    xy = np.stack([x, y], axis=-1)
    v = p[1] - p[0]
    t = np.clip(((xy - p[0]) @ v) / (v @ v), 0, 1)
    return np.linalg.norm(xy - (p[0] + t[..., None] * v), axis=-1) <= radius + .5


def project_masks(disparity, valid, fit, rays):
    if fit is None:
        return None
    inv = (disparity.ravel() - fit[1]) / fit[0]
    good = valid.ravel() & np.isfinite(inv) & (inv > 1e-3)
    points = rays[good] / inv[good, None]
    masks = np.zeros((len(inv), 2), bool)
    masks[good] = inside_boxes(points)
    return masks.reshape(*disparity.shape, 2)


def evaluate():
    start = time.perf_counter()
    blind_path = OUT / 'blind_features.json'
    sealed_hash = sha(blind_path)
    blind = json.loads(blind_path.read_text())
    labels_path = THIN / 'annotations.json'
    labels = json.loads(labels_path.read_text())
    selected = {(r['key'], r['i']): r for r in blind['rows']}
    cam = AtomCamera()
    k = cam.K_upright
    vv, uu = np.mgrid[:640, :480]
    dirs = np.stack([(uu - k[0, 2]) / k[0, 0], (vv - k[1, 2]) / k[1, 1], np.ones_like(uu)], -1).reshape(-1, 3)
    rct = np.asarray(json.loads((X.OUT / 'camera_tof_extrinsic.json').read_text())['rotation'])
    rows = []
    parity = []
    for label in labels['samples']:
        ident = label['session'], label['i']
        if ident not in selected:
            continue
        observation = selected[ident]
        inherited = observation['inherited']
        segment = label['segment']
        row = dict(key=ident[0], i=ident[1], original_label_status=label['label_status'],
                   raw_file=observation['raw_file'], cache_file=observation['cache_file'],
                   blind_payload=observation['blind_payload'], distance_truth='UNKNOWN',
                   original_visible=label['label_status'] == 'visible_segment', target_pixel_mask='UNKNOWN',
                   visibility_label_coordinate_system='undistorted upright; prior weak annotations',
                   sensor_to_upright_information_loss='NOT_EVALUABLE: no sensor-coordinate target mask',
                   loss_causal_identification='UNKNOWN: centerline is not target mask, metric GT or exposure truth')
        if not segment:
            rows.append(row)
            continue
        row['evaluator_only_segment'] = segment
        with np.load(observation['blind_payload']) as z:
            up, prep = z['upright'], z['prepared']
            rawgray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            row['input'] = dict(original=profile(rawgray, segment),
                                prepared=profile(z['prepared_gray_float'], segment, 'input'),
                                raw_line_match_10px=any(matches(segment, c) for c in observation['rgb_raw_lines']),
                                prepared_line_match_10px=any(matches(segment, c) for c in observation['rgb_prepared_lines']),
                                raw_line_match_3px=any(matches(segment, c, 3) for c in observation['rgb_raw_lines']),
                                prepared_line_match_3px=any(matches(segment, c, 3) for c in observation['rgb_prepared_lines']))
            row['depth'] = dict(cached=profile(z['cached_disparity'], segment) if 'cached_disparity' in z else dict(status='UNKNOWN_NO_OLD_DEPTH_CACHE'),
                                native_status=observation['native_status'],
                                old_cache_available='cached_disparity' in z,
                                native_missing_cache_role='new diagnostic only, not frozen readiness or detection rescue')
            if 'native_disparity' in z:
                row['depth'].update(native=profile(z['native_disparity'], segment, 'native'), cache_parity=observation['cache_parity'])
            candidates = inherited['chain']['candidates']
            indices = [j for j, c in enumerate(candidates) if matches(segment, c)]
            assert [(c['x1'], c['y1'], c['x2'], c['y2']) for c in candidates] == [
                (c['x1'], c['y1'], c['x2'], c['y2']) for c in inherited['ridge']['candidates']]
            row['automatic'] = dict(located=bool(indices), stable=any(candidates[j]['stable'] for j in indices),
                                    matched_candidate_indices=indices,
                                    matched_candidates=[dict(candidate=candidates[j],
                                                             geometry=inherited['geometry']['candidates'][j]['geometry']['robust_replace'],
                                                             distance=inherited['distance']['candidates'][j]) for j in indices])
            row['automatic']['inside_any'] = any(any(c['geometry']['query']) for c in row['automatic']['matched_candidates'])
            row['automatic']['stable_inside_any'] = any(c['candidate']['stable'] and any(c['geometry']['query']) for c in row['automatic']['matched_candidates'])
            meta = observation['meta']
            rays = dirs @ (F.rot(meta['pitch'], meta['roll']) @ rct.T).T if 'cached_disparity' in z else None
            masks = project_masks(z['cached_disparity'], z['valid'], observation['fit']['fit'], rays) if 'cached_disparity' in z else None
            row['readout'] = dict(source_anchor=observation['fit']['anchor'], fit=observation['fit']['fit'],
                                  old_full_frame_px=observation['fit']['px'], old_full_frame_rgb=observation['fit']['rgb'],
                                  tube_radius_px={}, target_contribution='UNKNOWN: tube contains unlabelled surroundings; not true shaft pixels')
            if masks is not None:
                counts = masks.sum(axis=(0, 1)).tolist()
                assert counts == observation['fit']['px'], (ident, counts, observation['fit']['px'])
                parity.append(dict(key=ident[0], i=ident[1], full_image_counts=counts))
                for radius in (0, 1, 3):
                    tube = tube_mask(z['cached_disparity'].shape, segment, radius)
                    n = masks[tube].sum(axis=0).tolist()
                    row['readout']['tube_radius_px'][str(radius)] = dict(total_tube_pixels=int(tube.sum()),
                                                                       HEAD_BODY_inside=n, HEAD_BODY_150px=[v >= 150 for v in n])
            # Overview plus crop; labels used only as overlay/evaluation.
            image = up.copy()
            for c in candidates:
                cv2.line(image, (round(c['x1']), round(c['y1'])), (round(c['x2']), round(c['y2'])), (80, 80, 80), 1)
            for j in indices:
                c = candidates[j]
                cv2.line(image, (round(c['x1']), round(c['y1'])), (round(c['x2']), round(c['y2'])), (0, 255, 0), 2)
            cv2.line(image, tuple(segment[:2]), tuple(segment[2:]), (255, 0, 255), 1)
            if observation['order'] in (0, 3, 5):
                depth = z['cached_disparity'] if 'cached_disparity' in z else z['returned_disparity']
                shown = cv2.applyColorMap(np.clip((depth - np.percentile(depth, 2)) / max(np.percentile(depth, 98) - np.percentile(depth, 2), 1e-5) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                cv2.line(shown, tuple(segment[:2]), tuple(segment[2:]), (255, 255, 255), 1)
                prepview = cv2.resize(prep, (480, 640), interpolation=cv2.INTER_AREA)
                header = np.zeros((35, 1440, 3), np.uint8)
                cv2.putText(header, f'{ident[0]}/{ident[1]} original/lines | actual input viewed at source size | cached disparity; located={bool(indices)}', (8, 23), 0, .55, (255, 255, 255), 1)
                panel = np.vstack([header, np.hstack([image, prepview, shown])])
                panelpath = OUT / f'overlay_{ident[0]}_{ident[1]:04d}.jpg'
                cv2.imwrite(str(panelpath), panel)
                row['overlay'] = str(panelpath)
            if observation['order'] == 0 and ident[0] in ('073554', '073821'):
                raw_sensor = cv2.imread(observation['raw_file'])
                mapx, mapy = cv2.convertMaps(cam.map1, cam.map2, cv2.CV_32FC1)
                q, _ = line_points(segment, 80)
                # Upright(x,y) = undistorted raw(y,x); lookup maps in row/column order.
                xy = np.rint(q).astype(int)
                mapped = np.stack([mapx[xy[:, 0], xy[:, 1]], mapy[xy[:, 0], xy[:, 1]]], axis=1)
                sensor_overlay = raw_sensor.copy()
                cv2.polylines(sensor_overlay, [np.rint(mapped).astype(np.int32)], False, (255, 0, 255), 2)
                sensorpath = OUT / f'sensor_correspondence_{ident[0]}_{ident[1]:04d}.jpg'
                cv2.imwrite(str(sensorpath), np.hstack([raw_sensor, sensor_overlay]))
                row['sensor_correspondence_overlay'] = str(sensorpath)
                row['sensor_overlay_role'] = 'evaluation-only inverse map of upright weak segment; raw visual QA, not a new pixel mask'
        rows.append(row)
    targets = [r for r in rows if r['original_visible']]
    def count(predicate):
        return sum(bool(predicate(r)) for r in targets)
    summary = dict(selected=len(rows), evaluable_targets=len(targets), not_evaluable=len(rows) - len(targets),
                   input_raw_line_10px=count(lambda r: r['input']['raw_line_match_10px']),
                   input_prepared_line_10px=count(lambda r: r['input']['prepared_line_match_10px']),
                   input_raw_line_3px=count(lambda r: r['input']['raw_line_match_3px']),
                   input_prepared_line_3px=count(lambda r: r['input']['prepared_line_match_3px']),
                   cached_center_positive_002=count(lambda r: r['depth']['cached'].get('center_positive_002')),
                   cached_best3_positive_002=count(lambda r: r['depth']['cached'].get('best_3px_positive_002')),
                   native_center_positive_002=count(lambda r: r['depth'].get('native', {}).get('center_positive_002')),
                   native_best3_positive_002=count(lambda r: r['depth'].get('native', {}).get('best_3px_positive_002')),
                   automatic_located=count(lambda r: r['automatic']['located']),
                   automatic_stable=count(lambda r: r['automatic']['stable']),
                   automatic_inside=count(lambda r: r['automatic']['inside_any']),
                   automatic_stable_inside=count(lambda r: r['automatic']['stable_inside_any']),
                   native_output_evaluable=count(lambda r: r['depth']['native_status'] == 'AVAILABLE'),
                   old_cached_depth_evaluable=count(lambda r: r['depth']['old_cache_available']),
                   native_cache_parity_pass=count(lambda r: r['depth'].get('cache_parity', {}).get('within_1e_5_iqr')),
                   tube_150px_frames={str(radius): count(lambda r: any(r['readout']['tube_radius_px'].get(str(radius), {}).get('HEAD_BODY_150px', []))) for radius in (0, 1, 3)})
    contrast_ratios = [r['input']['prepared']['line_vs_8px_background_contrast'] / max(r['input']['original']['line_vs_8px_background_contrast'], 1e-6) for r in targets if r['input']['prepared']['status'] == r['input']['original']['status'] == 'AVAILABLE']
    summary['input_contrast_ratio_prepared_over_original'] = dict(n=len(contrast_ratios), min=float(min(contrast_ratios)), median=float(np.median(contrast_ratios)), max=float(max(contrast_ratios)))
    paired = [r for r in targets if r['depth']['old_cache_available'] and r['depth']['native_status'] == 'AVAILABLE']
    summary['paired_native_to_cached'] = dict(n=len(paired),
        native_center_positive=sum(r['depth']['native'].get('center_positive_002', False) for r in paired),
        cached_center_positive=sum(r['depth']['cached'].get('center_positive_002', False) for r in paired),
        center_positive_to_negative=sum(r['depth']['native'].get('center_positive_002', False) and not r['depth']['cached'].get('center_positive_002', False) for r in paired),
        native_best3_positive=sum(r['depth']['native'].get('best_3px_positive_002', False) for r in paired),
        cached_best3_positive=sum(r['depth']['cached'].get('best_3px_positive_002', False) for r in paired),
        best3_positive_to_negative=sum(r['depth']['native'].get('best_3px_positive_002', False) and not r['depth']['cached'].get('best_3px_positive_002', False) for r in paired),
        returned_raw_bits_equal_old_cache=sum(r['depth']['cache_parity']['raw_bits_equal'] for r in paired))
    summary['native_fixed_sample_scope'] = dict(selected_frames=len(blind['rows']),
        new_native_outputs=sum(r['native_status'] == 'AVAILABLE' for r in blind['rows']),
        old_cached_frames=sum(bool(r['cache_file']) for r in blind['rows']),
        all_old_cache_raw_bits_parity=sum(r.get('cache_parity', {}).get('raw_bits_equal', False) for r in blind['rows']))
    stage_classes = Counter()
    for r in targets:
        category = ('OLD_DEPTH_UNAVAILABLE' if not r['depth']['old_cache_available'] else
                    'NO_MATCHED_AUTOMATIC_CANDIDATE' if not r['automatic']['located'] else
                    'MATCHED_BUT_TRACK_NOT_STABLE' if not r['automatic']['stable'] else
                    'STABLE_BUT_NO_MATCHED_CANDIDATE_GEOMETRY_INSIDE' if not r['automatic']['stable_inside_any'] else
                    'STABLE_MATCHED_CANDIDATE_GEOMETRY_INSIDE')
        r['first_descriptive_stage_limitation'] = category
        stage_classes[category] += 1
    summary['descriptive_mutually_exclusive_classes'] = dict(stage_classes)
    missing = []
    for r in targets:
        if not r['depth']['old_cache_available']:
            m = selected[r['key'], r['i']]['meta']
            missing.append(dict(key=r['key'], i=r['i'], pose_ok=m['pose_ok'], rgb_matched=m['matched'],
                                rgb_minus_tof_receipt_ms=(m['rgb_t_ns'] - m['t_ns']) / 1e6,
                                inferred_reason='old nearest-receipt match outside +/-100ms; exposure synchronization UNKNOWN'))
    summary['old_depth_unavailability'] = missing
    assert len(targets) == 22 and len(rows) == 24
    assert summary['automatic_located'] == 17 and summary['automatic_stable'] == 13
    assert sha(blind_path) == sealed_hash
    result = dict(summary=summary, rows=rows, projection_parity=parity, weak_annotation_sha256=sha(labels_path),
                  sealed_blind_sha256=sealed_hash, evaluation_wall_s=time.perf_counter() - start,
                  denominator='22 assistant-visible conservative shaft centerlines in 24 old fixed sample positions; 4 sessions, not independent events',
                  causal_limits=['No measured pole distance/width or human mask', 'input lines and contrast are descriptive, not target truth',
                                 'native-versus-resampled disparity comparison cannot separate encoder/decoder causal mechanisms',
                                 'matched candidate/track path is separate from the full-frame v1-v3 RGB pixel readout',
                                 'tube >=150 is width sensitivity, not contribution of true shaft', 'outside box does not establish a missed threat'],
                  checks=dict(fixed_old_sampling=True, producer_labels_read=False, blind_output_unmodified_after_evaluation=True,
                              immutable_candidate_endpoint_identity=True, reproduced_old_17_located_13_stable=True,
                              full_frame_projection_parity=len(parity), synthetic_checks=checks()))
    write_json(OUT / 'stages.json', result)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return result


def checks():
    # Check coordinate transforms, native sampling and 150px behavior at boundary.
    segment = [20, 20, 20, 220]
    a = np.zeros((640, 480), np.float32)
    a[:, 20] = 1
    native = cv2.resize(a, (518, 686), interpolation=cv2.INTER_LINEAR)
    p = profile(a, segment)
    q = profile(native, segment, 'native')
    assert p['center_positive_002'] and q['center_positive_002']
    masks = [tube_mask(a.shape, segment, radius) for radius in (0, 1, 3)]
    assert all(np.all(x <= y) for x, y in zip(masks, masks[1:]))
    assert masks[0].sum() == 201
    assert matches(segment, dict(x1=20, y1=20, x2=20, y2=220), 3)
    # Missing scale remains unavailable; box counting excludes estimates <0.3m.
    rays = np.stack([np.zeros(a.size), np.ones(a.size) * .6, np.ones(a.size)], 1)
    assert project_masks(np.ones_like(a), np.ones_like(a, bool), None, rays) is None
    inside = project_masks(np.ones_like(a), np.ones_like(a, bool), [1., 0.], rays)
    assert inside.sum(axis=(0, 1)).tolist() == [0, 640 * 480]
    near = project_masks(np.ones_like(a), np.ones_like(a, bool), [.02, 0.], rays)
    assert not near.any()
    return dict(native_coordinate_ridge=True, tube_nested_and_single_pixel_201=True,
                label_match=True, missing_scale_unknown=True, old_box_boundary_semantics=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--no-native', action='store_true')
    parser.add_argument('--reuse-native-payloads', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    checks()
    if not args.evaluate_only:
        if (OUT / 'blind_features.json').exists():
            raise FileExistsError('Existing blind output retained; use --evaluate-only for a verified evaluator repair')
        generate(args.no_native, args.reuse_native_payloads)
    result = evaluate()
    write_json(OUT / 'completion.json', dict(complete=True, wall_s=time.perf_counter() - start,
                                            source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
                                            payload_bytes=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()),
                                            checks=result['checks'], no_training=True, no_download=True, no_app_change=True))


if __name__ == '__main__':
    main()

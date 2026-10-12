"""Label-blind selection of unused cached 3RScan environments for transfer.

This is Development, not official held-out test or body-clearance evidence.
Selection uses official metadata, cache presence and paired frame names only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile

import cv2
import numpy as np
from PIL import Image

from rgb_body_query_3rscan import (color_coordinates, optical_z, parse_info,
                                   sensor_labels, task_queries)
from rgb_body_query_input_diagnostic import sha, write


def select(root, old_selection, count):
    meta = json.loads((root/'3RScan.json').read_text('utf-8-sig'))
    old = json.loads(old_selection.read_text('utf-8-sig'))
    used_ids = {s for g in old['groups'] for s in (g['environment'], g['scan'])}
    excluded, candidates = [], []
    for g in meta:
        if g['type'] != 'train':
            continue
        ids = [g['reference']] + [s['reference'] for s in g['scans']]
        cached = sorted(s for s in ids if (root/s/'sequence.zip').is_file())
        if not cached:
            continue
        row = dict(environment=g['reference'],
                   scan=g['reference'] if g['reference'] in cached else cached[0],
                   official_split='train', split='validation',
                   group_sha256=hashlib.sha256(g['reference'].encode()).hexdigest(),
                   group_scan_ids=ids, cached_scan_ids=cached)
        (excluded if used_ids.intersection(ids) else candidates).append(row)
    candidates.sort(key=lambda r: r['group_sha256'])
    if len(candidates) < count:
        raise ValueError('Insufficient unused official-train cached environments')
    chosen = candidates[:count]
    for g in chosen:
        with zipfile.ZipFile(root/g['scan']/'sequence.zip') as archive:
            names = set(archive.namelist())
        paired = sorted(int(m.group(1)) for n in names if
                        (m := re.fullmatch(r'frame-(\d{6})\.color\.jpg', n)) and
                        f'frame-{int(m.group(1)):06d}.depth.pgm' in names)
        indices = np.linspace(0, len(paired)-1, 8).round().astype(int)
        if len(set(indices)) != 8:
            raise ValueError('Need 8 distinct paired frames; do not replace selected group')
        g.update(frames=[paired[i] for i in indices], paired_frame_count=len(paired))
    return chosen, excluded, candidates


def independent_labels(depth, k, query, observed):
    """Independent XYZ membership and three-axis slab check, no shared interval helper."""
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    uv1 = np.stack([xx, yy, np.ones_like(xx)], -1)
    directions = uv1 @ np.linalg.inv(k).T
    points = directions * depth[..., None]
    lo, hi = np.array(query['low']), np.array(query['high'])
    near, far = np.full(depth.shape, -np.inf), np.full(depth.shape, np.inf)
    parallel_out = np.zeros(depth.shape, bool)
    for axis in range(3):
        direction = directions[..., axis]
        nonzero = np.abs(direction) > 1e-12
        a = np.divide(lo[axis], direction, out=np.zeros_like(direction), where=nonzero)
        b = np.divide(hi[axis], direction, out=np.zeros_like(direction), where=nonzero)
        near = np.maximum(near, np.where(nonzero, np.minimum(a, b), -np.inf))
        far = np.minimum(far, np.where(nonzero, np.maximum(a, b), np.inf))
        parallel_out |= ~nonzero & ((lo[axis] > 0) | (hi[axis] < 0))
    reachable = ~parallel_out & (near <= far+1e-12)
    valid = np.isfinite(depth) & (depth > 0) & observed
    inside = ((points >= lo-1e-12) & (points <= hi+1e-12)).all(-1)
    result = np.full(depth.shape, 255, np.uint8)
    result[reachable] = 2
    result[reachable & valid & inside] = 1
    result[reachable & valid & (depth > far+1e-12)] = 0
    return result


def prepare(repo, output, old_selection, count=8, budget_s=240):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Preserve prior attempt; output must be empty')
    root = repo/'artifacts.local/datasets/3rscan'
    groups, excluded, candidates = select(root, old_selection, count)
    queries = task_queries()
    selection = dict(status='SELECTED_BEFORE_REFERENCE_READ', groups=groups,
                     queries=queries, method='exclude old environment groups including rescans; reference SHA order; one reference-preferred scan; 8 uniform paired frames',
                     cached_official_train_environments=len(candidates)+len(excluded),
                     unused_cached_environments=len(candidates), excluded_groups=excluded,
                     old_selection_sha256=sha(old_selection), metadata_sha256=sha(root/'3RScan.json'),
                     budget_cpu_wall_s=budget_s, gpu_s=0, download_bytes=0,
                     role='Development frozen transfer; all validation; no training/calibration')
    write(output/'selection.json', selection)
    rows, receipt = [], dict(status='RUNNING', independent_xyz_checks=0)
    try:
        for g in groups:
            archive_path = root/g['scan']/'sequence.zip'
            with zipfile.ZipFile(archive_path) as archive:
                info_bytes = archive.read('_info.txt')
                info = parse_info(info_bytes)
                mx, my = color_coordinates(info['depth_K'], info['color_K'], info['depth_shape'])
                ch, cw = info['color_shape']
                observed = (mx >= 0) & (mx <= cw-1) & (my >= 0) & (my <= ch-1)
                for frame in g['frames']:
                    if time.perf_counter()-started >= budget_s:
                        raise TimeoutError('Fixed preparation CPU wall budget reached')
                    prefix = f'{g["scan"]}_{frame:06d}'
                    rgb_bytes = archive.read(f'frame-{frame:06d}.color.jpg')
                    depth_bytes = archive.read(f'frame-{frame:06d}.depth.pgm')
                    rgb = Image.open(io.BytesIO(rgb_bytes)).convert('RGB')
                    raw = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    if (rgb.height, rgb.width) != info['color_shape'] or raw.shape != info['depth_shape'] or raw.dtype != np.uint16:
                        raise ValueError('Source dimensions/type disagree with public calibration')
                    depth = optical_z(raw, info['shift'])
                    labels, qs = zip(*(sensor_labels(depth, info['depth_K'], q, observed) for q in queries))
                    for label, q in zip(labels, queries):
                        if not np.array_equal(label, independent_labels(depth, info['depth_K'], q, observed)):
                            raise ValueError('Independent XYZ/slab label mismatch')
                        receipt['independent_xyz_checks'] += 1
                    rgb_path = output/f'{prefix}.jpg'; rgb_path.write_bytes(rgb_bytes)
                    ref_path = output/f'{prefix}.npz'
                    np.savez_compressed(ref_path, labels=np.stack(labels), depth=depth,
                                        depth_K=info['depth_K'], color_K=info['color_K'],
                                        map_x=mx, map_y=my, observed=observed)
                    rows.append(dict(environment=g['environment'], scan=g['scan'], split='validation',
                                     official_split='train', group_sha256=g['group_sha256'], frame=frame,
                                     rgb_path=str(rgb_path), rgb_sha256=sha(rgb_path), reference_path=str(ref_path), reference_sha256=sha(ref_path),
                                     source_zip=str(archive_path), source_rgb_sha256=hashlib.sha256(rgb_bytes).hexdigest(),
                                     source_depth_sha256=hashlib.sha256(depth_bytes).hexdigest(), calibration_sha256=hashlib.sha256(info_bytes).hexdigest(),
                                     color_K=info['color_K'].tolist(), depth_K=info['depth_K'].tolist(), color_shape=list(info['color_shape']),
                                     depth_shape=list(info['depth_shape']), depth_shift=info['shift'], queries=list(qs)))
        counts = dict(Counter(q['state'] for r in rows for q in r['queries']))
        units = {key: sum(q[key] for r in rows for q in r['queries']) for key in
                 ('positive_pixels', 'free_ray_pixels', 'unknown_pixels', 'domain_pixels')}
        manifest = dict(status='REAL_SENSOR_SPARSE_QUERY_READY', rows=rows, groups=groups, queries=queries,
                        state_counts_by_split={'train': {}, 'cal': {}, 'validation': counts}, frames=len(rows),
                        query_ray_units=units, elapsed_s=time.perf_counter()-started,
                        label_contract='0=FREE_RAY first measured surface after box;1=measured surface in box;2=missing/occluded;255=ray misses box',
                        sensor='Tango calibrated sensor depth, mm/depthShift, optical-Z; not metrology GT',
                        query_contract='sampled first-return rays only; FREE_ON_SAMPLED_RAYS not whole-volume clearance',
                        observation_contract='full RGB and public K/query; depth/labels/evaluator masks excluded',
                        official_data_role='official train cached unseen-by-old-run environments; consumed Development transfer, not independent confirmation',
                        license='existing research cache, no redistribution; toolkit license not dataset license',
                        selection_sha256=sha(output/'selection.json'))
        write(output/'dataset_manifest.json', manifest)
        write(output/'observations.json', dict(rows=[{k: r[k] for k in
              ('environment', 'scan', 'split', 'frame', 'rgb_path', 'rgb_sha256', 'color_K', 'color_shape', 'depth_K', 'depth_shape')} for r in rows],
              queries=queries, dataset_manifest_sha256=sha(output/'dataset_manifest.json')))
        used = {i for g in excluded for i in g['group_scan_ids']}
        selected = {i for g in groups for i in g['group_scan_ids']}
        if used & selected or len(rows) != count*8 or len({r['environment'] for r in rows}) != count:
            raise ValueError('Environment exclusion/count mismatch')
        for r in rows:
            if sha(r['rgb_path']) != r['rgb_sha256'] or sha(r['reference_path']) != r['reference_sha256']:
                raise ValueError('Output identity mismatch')
        receipt.update(status='COMPLETE', groups=len(groups), frames=len(rows), query_cells=len(rows)*len(queries),
                       query_states=counts, query_ray_units=units, environment_and_rescan_overlap=[],
                       observations_evaluator_fields_absent=True, output_identities_verified=True,
                       source_sha256=sha(Path(__file__)), selection_sha256=sha(output/'selection.json'),
                       manifest_sha256=sha(output/'dataset_manifest.json'), observations_sha256=sha(output/'observations.json'))
    except Exception as error:
        receipt.update(status='FAILED', error=repr(error), completed_frames=len(rows)); raise
    finally:
        receipt.update(cpu_wall_s=time.perf_counter()-started, gpu_s=0, download_bytes=0)
        write(output/'preparation_receipt.json', receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old-selection', type=Path, required=True)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--budget-s', type=float, default=240)
    args = parser.parse_args()
    prepare(args.repo.resolve(), args.output.resolve(), args.old_selection.resolve(), args.count, args.budget_s)

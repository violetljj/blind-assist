"""Real sensor-ray supervision from official-train 3RScan; no clearance GT.

RGB models receive the full image and public calibration. Depth is used only
for supervision/scoring, on its native calibrated rays, without hole filling.
Occluding first returns before a query are UNKNOWN, not negative occupancy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from rgb_body_query_geometry import camera_queries
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import rays, ray_interval, confusion, ratios

IGNORE, UNKNOWN, FREE_RAY, POSITIVE = 255, 2, 0, 1


def task_queries():
    result = camera_queries()
    for name, a, b in [('close', .3, .8), ('middle', .8, 1.5), ('upper_near', 1.5, 3.)]:
        for lateral, lo, hi in [('center', -.3, .3), ('left', -.9, -.3), ('right', .3, .9)]:
            result.append(dict(name=f'{lateral}_{name}', low=[lo, -.55, a], high=[hi, .55, b]))
    return result


def parse_info(payload):
    text = payload.decode('utf-8') if isinstance(payload, bytes) else payload
    fields = dict(line.split(' = ', 1) for line in text.splitlines() if ' = ' in line)
    def matrix(key):
        return np.asarray(fields[key].split(), np.float64).reshape(4, 4)
    if not all(np.allclose(matrix(key), np.eye(4), atol=1e-8) for key in
               ('m_calibrationColorExtrinsic', 'm_calibrationDepthExtrinsic')):
        raise ValueError('Only calibrated identity sensor extrinsics supported')
    shift = float(fields['m_depthShift'])
    if not np.isfinite(shift) or shift <= 0:
        raise ValueError('Invalid depthShift')
    return dict(color_K=matrix('m_calibrationColorIntrinsic')[:3, :3],
                depth_K=matrix('m_calibrationDepthIntrinsic')[:3, :3], shift=shift,
                color_shape=(int(fields['m_colorHeight']), int(fields['m_colorWidth'])),
                depth_shape=(int(fields['m_depthHeight']), int(fields['m_depthWidth'])))


def optical_z(raw, depth_shift):
    if not np.isfinite(depth_shift) or depth_shift <= 0:
        raise ValueError('Invalid depthShift')
    z = np.array(raw, np.float32)/depth_shift
    z[(~np.isfinite(z)) | (z <= 0)] = np.nan
    return z


def color_coordinates(depth_k, color_k, depth_shape):
    rx, ry = rays(depth_k, depth_shape)
    h = np.stack([rx, ry, np.ones_like(rx)], -1) @ np.asarray(color_k).T
    return (h[..., 0]/h[..., 2]).astype(np.float32), (h[..., 1]/h[..., 2]).astype(np.float32)


def sample_prediction(depth, mx, my):
    """Only prediction values are interpolated; never mutate or fill reference."""
    depth = np.asarray(depth, np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    val = cv2.remap(np.where(valid, depth, 0), mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
    weight = cv2.remap(valid.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    val[weight < 1.-1e-6] = np.nan
    return val


def sensor_labels(depth, k, query, observed=None, min_support=16):
    rx, ry = rays(k, depth.shape)
    entry, exit, reachable = ray_interval(rx, ry, query)
    valid = np.isfinite(depth) & (depth > 0)
    if observed is not None:
        valid &= observed
    occupied = valid & reachable & (depth >= entry-1e-12) & (depth <= exit+1e-12)
    free = valid & reachable & (depth > exit+1e-12)
    label = np.full(depth.shape, IGNORE, np.uint8)
    label[reachable] = UNKNOWN
    label[occupied] = POSITIVE
    label[free] = FREE_RAY
    counts = {name: int((label == value).sum()) for name, value in
              [('positive_pixels', POSITIVE), ('free_ray_pixels', FREE_RAY), ('unknown_pixels', UNKNOWN)]}
    state = 'POSITIVE' if counts['positive_pixels'] >= min_support else (
        'FREE_ON_SAMPLED_RAYS' if counts['free_ray_pixels'] >= min_support and
        counts['unknown_pixels'] == 0 and counts['positive_pixels'] == 0 else 'UNKNOWN')
    return label, dict(name=query['name'], state=state, domain_pixels=int(reachable.sum()), **counts)


def selected_groups(root, count=12):
    meta_path = root/'3RScan.json'
    meta = json.loads(meta_path.read_text('utf-8-sig'))
    groups = []
    for g in meta:
        if g['type'] != 'train':
            continue
        ids = [g['reference']] + [s['reference'] for s in g['scans']]
        cached = sorted(s for s in ids if (root/s/'sequence.zip').is_file())
        if cached:
            group = g['reference']
            groups.append(dict(environment=group, scan=group if group in cached else cached[0],
                               official_split='train', group_sha256=hashlib.sha256(group.encode()).hexdigest()))
    groups.sort(key=lambda g: g['group_sha256'])
    if len(groups) < count:
        raise ValueError('Insufficient train environments')
    chosen = groups[:count]
    for i, g in enumerate(chosen):
        g['split'] = 'train' if i < 7 else ('cal' if i < 9 else 'validation')
    return chosen


def prepare(repo, output, budget_s=600):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'dataset_manifest.json').exists():
        raise FileExistsError('Preserve completed dataset, do not implicitly restart')
    root = repo/'artifacts.local/datasets/3rscan'
    groups = selected_groups(root)
    write(output/'selection.json', dict(groups=groups, method='12 official-train environments by reference SHA; one scan; 8 uniform paired frames',
                                        metadata_sha256=sha(root/'3RScan.json'), queries=task_queries()))
    rows, preview = [], []
    for g in groups:
        archive_path = root/g['scan']/'sequence.zip'
        with zipfile.ZipFile(archive_path) as archive:
            info_bytes = archive.read('_info.txt'); info = parse_info(info_bytes)
            names = set(archive.namelist())
            paired = sorted(int(m.group(1)) for name in names if
                            (m := re.fullmatch(r'frame-(\d{6})\.color\.jpg', name)) and
                            f'frame-{int(m.group(1)):06d}.depth.pgm' in names)
            indices = np.linspace(0, len(paired)-1, 8).round().astype(int)
            if len(set(indices)) != 8:
                raise ValueError('Need 8 distinct paired frames')
            mx, my = color_coordinates(info['depth_K'], info['color_K'], info['depth_shape'])
            ch, cw = info['color_shape']
            observed = (mx >= 0) & (mx <= cw-1) & (my >= 0) & (my <= ch-1)
            for frame in (paired[i] for i in indices):
                if time.perf_counter()-start >= budget_s:
                    raise TimeoutError('Reference preparation budget reached')
                prefix = f'{g["scan"]}_{frame:06d}'
                rgb_bytes = archive.read(f'frame-{frame:06d}.color.jpg')
                depth_bytes = archive.read(f'frame-{frame:06d}.depth.pgm')
                rgb = Image.open(io.BytesIO(rgb_bytes)).convert('RGB')
                raw = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                if (rgb.height, rgb.width) != info['color_shape'] or raw.shape != info['depth_shape'] or raw.dtype != np.uint16:
                    raise ValueError('Source dimensions/type disagree with calibration')
                depth = optical_z(raw, info['shift'])
                rgb_path = output/f'{prefix}.jpg'; rgb_path.write_bytes(rgb_bytes)
                labels, qs = zip(*(sensor_labels(depth, info['depth_K'], q, observed) for q in task_queries()))
                ref_path = output/f'{prefix}.npz'
                np.savez_compressed(ref_path, labels=np.stack(labels), depth=depth, depth_K=info['depth_K'],
                                    color_K=info['color_K'], map_x=mx, map_y=my, observed=observed)
                rows.append(dict(**g, frame=frame, rgb_path=str(rgb_path), rgb_sha256=sha(rgb_path),
                                 reference_path=str(ref_path), reference_sha256=sha(ref_path),
                                 source_zip=str(archive_path), source_rgb_sha256=hashlib.sha256(rgb_bytes).hexdigest(),
                                 source_depth_sha256=hashlib.sha256(depth_bytes).hexdigest(),
                                 calibration_sha256=hashlib.sha256(info_bytes).hexdigest(),
                                 color_K=info['color_K'].tolist(), depth_K=info['depth_K'].tolist(),
                                 color_shape=list(info['color_shape']), depth_shape=list(info['depth_shape']),
                                 depth_shift=info['shift'], queries=list(qs)))
                if frame == paired[indices[0]]:
                    preview.append((g, rgb.copy(), depth.copy(), mx, my))
    counts = {split: dict(Counter(q['state'] for r in rows if r['split'] == split for q in r['queries']))
              for split in ('train', 'cal', 'validation')}
    manifest = dict(status='REAL_SENSOR_SPARSE_QUERY_READY', rows=rows, groups=groups, queries=task_queries(),
                    state_counts_by_split=counts, frames=len(rows), elapsed_s=time.perf_counter()-start,
                    label_contract='0=FREE_RAY first measured surface after box;1=measured surface in box;2=missing/occluded;255=ray misses box',
                    sensor='Tango calibrated sensor depth, mm/depthShift, optical-Z backprojection; not metrology GT',
                    query_contract='POSITIVE or FREE_ON_SAMPLED_RAYS or UNKNOWN; sampled first-return rays only, no whole-volume clearance',
                    observation_contract='full RGB, public K, public query; depth/labels/evaluator masks excluded',
                    official_data_role='official train, consumed local Development; 7/2/3 environment groups, not independent confirmation',
                    license='existing local research cache; do not redistribute data; toolkit MIT is not data license')
    write(output/'dataset_manifest.json', manifest)
    write(output/'observations.json', dict(rows=[{k: r[k] for k in
          ('environment', 'scan', 'split', 'frame', 'rgb_path', 'rgb_sha256', 'color_K', 'color_shape', 'depth_K', 'depth_shape')} for r in rows],
          queries=task_queries(), dataset_manifest_sha256=sha(output/'dataset_manifest.json')))
    sheet = Image.new('RGB', (960, 240*len(preview)), '#101010')
    for i, (g, rgb, depth, mx, my) in enumerate(preview):
        sheet.paste(rgb.resize((480, 270)).crop((0, 15, 480, 240)), (0, i*240+15))
        d = np.uint8(np.clip(np.nan_to_num(depth)/6, 0, 1)*255)
        sheet.paste(Image.fromarray(d).convert('RGB').resize((240, 185)), (480, i*240+30))
        projected = np.array(rgb)
        valid = np.isfinite(depth)
        u, v = np.rint(mx[valid]).astype(int), np.rint(my[valid]).astype(int)
        inside = (u >= 0) & (u < rgb.width) & (v >= 0) & (v < rgb.height)
        projected[v[inside], u[inside]] = [255, 0, 255]
        sheet.paste(Image.fromarray(projected).resize((240, 135)), (720, i*240+30))
        ImageDraw.Draw(sheet).text((5, i*240), g['environment']+' '+g['split'], fill='white')
    sheet.save(output/'rgb_depth_contact.png')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('rows', 'groups', 'queries')}, ensure_ascii=False), flush=True)


def baseline(repo, output, budget_s=600):
    import dataclasses
    import gc
    import sys
    import torch
    start = time.perf_counter()
    obs_path = output/'observations.json'
    obs = json.loads(obs_path.read_text('utf-8-sig'))
    dest = output/'depthpro'; dest.mkdir(exist_ok=True)
    manifest_path = dest/'predictions.json'
    if manifest_path.exists():
        raise FileExistsError('No implicit model rerun')
    model = transform = None
    result = dict(status='STARTING', rows=[], scale_fit=False, evaluator_inputs=False, arm='raw-native',
                  observations_sha256=sha(obs_path), budget_s=budget_s,
                  principal_point='Depth Pro supplied fx only; raw image, no centered remap')
    times = []
    try:
        torch.set_num_threads(4)
        base = repo/'artifacts.local/work/ba-nfo-depthpro-20260919'
        backend = json.loads((base/'backend.json').read_text('utf-8-sig'))
        if not torch.cuda.is_available() or backend['selected_framework'] != f'torch-{torch.__version__}' or backend['selected_device_name'] != torch.cuda.get_device_name(0):
            raise RuntimeError('Inherited measured backend differs')
        weight = base/'depth_pro.pt'
        if sha(weight) != '3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce':
            raise ValueError('Weight identity differs')
        sys.path.insert(0, str(base/'upstream/src'))
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(weight))
        model, transform = depth_pro.create_model_and_transforms(cfg, device=torch.device('cuda'), precision=torch.float16)
        model.eval().requires_grad_(False)
        result.update(device=torch.cuda.get_device_name(0), framework=f'torch-{torch.__version__}', weight_sha256=sha(weight),
                      backend_reference_sha256=sha(base/'backend.json'), startup_s=time.perf_counter()-start)
        for row in obs['rows']:
            if budget_s-(time.perf_counter()-start) < max(times[-5:], default=2.)+3.:
                result['status'] = 'PARTIAL_BUDGET'; break
            path = Path(row['rgb_path'])
            if sha(path) != row['rgb_sha256']:
                raise ValueError('Changed RGB identity')
            rgb = Image.open(path).convert('RGB')
            if [rgb.height, rgb.width] != row['color_shape']:
                raise ValueError('Changed RGB shape')
            x = transform(rgb)
            torch.cuda.synchronize(); begin = time.perf_counter()
            with torch.inference_mode():
                pred = model.infer(x, f_px=torch.tensor(row['color_K'][0][0], device='cuda'))
            if pred['depth'].device.type != 'cuda':
                raise RuntimeError('No CUDA inference')
            torch.cuda.synchronize(); times.append(time.perf_counter()-begin)
            array = pred['depth'].float().cpu().numpy()
            if list(array.shape) != row['color_shape']:
                raise ValueError('Wrong prediction shape')
            out = dest/f'{row["scan"]}_{row["frame"]:06d}.npz'
            np.savez_compressed(out, depth=array)
            result['rows'].append(dict(environment=row['environment'], scan=row['scan'], frame=row['frame'], split=row['split'],
                                       rgb_sha256=row['rgb_sha256'], path=str(out), sha256=sha(out), inference_s=times[-1]))
            result.update(status='RUNNING', completed_calls=len(times), elapsed_s=time.perf_counter()-start)
            write(manifest_path, result)
            if len(times) % 8 == 0:
                print(f'{len(times)}/{len(obs["rows"])} elapsed={result["elapsed_s"]:.2f}s', flush=True)
            del pred, x, array
        else:
            result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='FAILED', error=repr(error)); raise
    finally:
        result.update(inference_s=sum(times), peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
        del model, transform
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()
        result['allocation_wall_s'] = time.perf_counter()-start
        write(manifest_path, result)
        print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


def evaluate(output, budget_s=600):
    start = time.perf_counter()
    manifest = json.loads((output/'dataset_manifest.json').read_text('utf-8-sig'))
    preds = json.loads((output/'depthpro/predictions.json').read_text('utf-8-sig'))
    source = {(r['scan'], r['frame']): r for r in manifest['rows']}
    records = []
    for p in preds['rows']:
        if time.perf_counter()-start >= budget_s:
            raise TimeoutError('Evaluation budget reached')
        r = source[p['scan'], p['frame']]
        if r['rgb_sha256'] != p['rgb_sha256'] or sha(p['path']) != p['sha256'] or sha(r['reference_path']) != r['reference_sha256']:
            raise ValueError('Prediction/reference identity differs')
        with np.load(p['path']) as npz:
            depth = npz['depth']
        with np.load(r['reference_path']) as ref:
            sampled = sample_prediction(depth, ref['map_x'], ref['map_y'])
            labels, k = ref['labels'], ref['depth_K']
        rx, ry = rays(k, sampled.shape)
        for i, q in enumerate(manifest['queries']):
            entry, exit, reachable = ray_interval(rx, ry, q)
            predicted = np.isfinite(sampled) & reachable & (sampled >= entry) & (sampled <= exit)
            c = confusion(labels[i], predicted)
            records.append(dict(environment=r['environment'], scan=r['scan'], frame=r['frame'], split=r['split'],
                                query=q['name'], query_family='legacy6' if i < 6 else 'new_distance_bands9', reference_state=r['queries'][i]['state'],
                                predicted_positive=int(predicted.sum()) >= 16, predicted_support=int(predicted.sum()), **c))
    summaries = {}
    for scope in ('all', 'train', 'cal', 'validation'):
        rows = [r for r in records if scope == 'all' or r['split'] == scope]
        c = {key: sum(r[key] for r in rows) for key in ('tp', 'fn', 'fp', 'tn')}
        positive = [r for r in rows if r['reference_state'] == 'POSITIVE']
        free = [r for r in rows if r['reference_state'] == 'FREE_ON_SAMPLED_RAYS']
        summaries[scope] = dict(**c, **ratios(c), query_positive_hits=sum(r['predicted_positive'] for r in positive),
                               query_positive_total=len(positive), sampled_free_false_support=sum(r['predicted_positive'] for r in free),
                               sampled_free_total=len(free), unknown_query_total=sum(r['reference_state'] == 'UNKNOWN' for r in rows),
                               frames=len({(r['scan'], r['frame']) for r in rows}))
    result = dict(status='COMPLETE', records=records, summary=summaries, elapsed_s=time.perf_counter()-start,
                  scope='full RGB inference, public calibrated ray-grid geometry; known measured occupied/free rays only; no GT localization mask',
                  limitations='sensor resolution/holes and occlusion; FREE_ON_SAMPLED_RAYS not entire box clearance; no thin/body/event claim',
                  reference_manifest_sha256=sha(output/'dataset_manifest.json'), prediction_manifest_sha256=sha(output/'depthpro/predictions.json'))
    write(output/'evaluation.json', result)
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'baseline', 'evaluate'])
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=600)
    args = parser.parse_args()
    if args.action == 'evaluate':
        evaluate(args.output.resolve(), args.budget_s)
    else:
        globals()[args.action](args.repo.resolve(), args.output.resolve(), args.budget_s)

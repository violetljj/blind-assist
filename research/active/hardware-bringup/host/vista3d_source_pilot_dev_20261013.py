"""Small pinned public-data capability check; no model, threshold, or eval changes.

Object GT is a registered scan reprojection. Background and ORB-SLAM poses are
not external metrology. Samples are selected by filename order before decoding.
"""
from __future__ import annotations

import concurrent.futures
import csv
from datetime import datetime
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/data-potential-dev-20261013/vista3d'
REV = '6a24d271559ed378f8f0649fc6652a37e323c405'
REPO = 'shaniaguo/VisTa3D'
SCENES = ['hallway/wires_hallway', 'hallway/bluecable_hallway']
MODALITIES = ['image', 'depth', 'ground_truth_depth', 'pcd_mask']


def get(url):
    with urllib.request.urlopen(url, timeout=40) as response:
        return response.read()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding='utf8')


def tree(path):
    url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/' + urllib.parse.quote(path, safe='/')
    # Each selected modality has fewer than the API's 1000-entry page limit.
    with urllib.request.urlopen(url + '?limit=1000', timeout=40) as response:
        if 'rel="next"' in response.headers.get('Link', ''):
            raise ValueError('Pagination required; do not silently select a truncated sequence')
        return json.loads(response.read())


def local_path(remote):
    return OUT / 'files' / remote.replace(':', '_')


def fetch_file(item):
    remote = item['path']
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/' + urllib.parse.quote(remote, safe='/')
    path = local_path(remote)
    data = path.read_bytes() if path.exists() else get(url)
    if 'size' in item and len(data) != item['size']:
        raise ValueError(f'Size mismatch: {remote}')
    if len(data) > 5_000_000:
        raise ValueError('Per-file pilot limit exceeded')
    digest = hashlib.sha256(data).hexdigest()
    expected = item.get('lfs', {}).get('oid')
    if expected and expected != digest:
        raise ValueError(f'LFS hash mismatch: {remote}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return dict(remote=remote, local=str(path.relative_to(OUT)), url=url,
                bytes=len(data), sha256=digest, official_lfs_sha_checked=bool(expected))


def quantiles(x):
    return np.quantile(x, [.1, .5, .9]).tolist() if len(x) else None


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [{'path': 'README.md'}]
    selections = []
    for scene in SCENES:
        entries = {m: tree(f'{scene}/data/{m}') for m in MODALITIES}
        maps = {m: {Path(x['path']).name: x for x in rows if x['type'] == 'file'}
                for m, rows in entries.items()}
        common = sorted(set.intersection(*(set(x) for x in maps.values())))
        indices = np.linspace(0, len(common) - 1, 6).astype(int).tolist()
        names = [common[i] for i in indices]
        selections.append(dict(scene=scene, modality_counts={m: len(v) for m, v in maps.items()},
                               common_frames=len(common), selected_indices=indices, selected_names=names))
        for name in names:
            inputs.extend(maps[m][name] for m in MODALITIES)
        inputs.append(next(x for x in tree(f'{scene}/data') if x['path'].endswith('intrinsics.npy')))
        top = tree(scene)
        inputs.extend(x for x in top if Path(x['path']).name in ('CameraTrajectory.txt', 'frames.csv'))
    assert sum(x.get('size', 0) for x in inputs) < 100_000_000
    save('selection_before_decode.json', dict(revision=REV, role='Development source-capability pilot',
                                             rule='six equally spaced common filenames per named scene',
                                             scenes=selections, files=inputs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(fetch_file, inputs))
    save('download_receipt.json', dict(revision=REV, files=receipts,
                                       bytes=sum(x['bytes'] for x in receipts)))
    rows = []
    montage = Image.new('RGB', (3 * 640, 4 * 280), 'white')
    draw = ImageDraw.Draw(montage)
    for selection in selections:
        scene = selection['scene']
        K = np.load(local_path(f'{scene}/data/intrinsics.npy'), allow_pickle=False)
        assert K.shape == (3, 3) and np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0
        poses = np.loadtxt(local_path(f'{scene}/CameraTrajectory.txt'))
        assert poses.ndim == 2 and poses.shape[1] == 8 and np.isfinite(poses).all()
        quaternion_norm_error = float(np.max(np.abs(np.linalg.norm(poses[:, 4:], axis=1) - 1)))
        assert quaternion_norm_error < 1e-4
        with local_path(f'{scene}/frames.csv').open(encoding='utf8') as f:
            frame_names = {Path(r['color_path']).name for r in csv.DictReader(f)}
        selection['pose_rows'] = len(poses)
        selection['quaternion_norm_max_error'] = quaternion_norm_error
        selection['selected_frame_csv_matches'] = 0
        selection['selected_pose_timestamp_matches'] = 0
        for name in selection['selected_names']:
            assert name in frame_names
            dt = datetime.fromisoformat(name.removesuffix('.png'))
            seconds_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
            assert np.min(np.abs(poses[:, 0] - seconds_of_day)) < 1e-4
            selection['selected_frame_csv_matches'] += 1
            selection['selected_pose_timestamp_matches'] += 1
            paths = {m: local_path(f'{scene}/data/{m}/{name}') for m in MODALITIES}
            rgb = Image.open(paths['image']).convert('RGB')
            raw = np.asarray(Image.open(paths['depth'])).astype(float) / 1000
            gt = np.asarray(Image.open(paths['ground_truth_depth'])).astype(float) / 256
            mask_array = np.asarray(Image.open(paths['pcd_mask']))
            mask = mask_array > 0
            if mask.ndim == 3:
                mask = mask.any(axis=2)
            assert mask.shape == gt.shape == raw.shape == (rgb.height, rgb.width)
            obj = mask & (gt > 0)
            paired = obj & (raw > 0)
            abs_error = np.abs(raw[paired] - gt[paired])
            row = dict(scene=scene, name=name, shape=list(gt.shape), intrinsics=K.tolist(),
                       object_mask_pixels=int(mask.sum()), object_valid_gt_pixels=int(obj.sum()),
                       raw_valid_on_object_pixels=int(paired.sum()),
                       raw_missing_on_object_pixels=int((obj & (raw == 0)).sum()),
                       object_reference_depth_p10_p50_p90_m=quantiles(gt[obj]),
                       raw_absolute_error_p10_p50_p90_m=quantiles(abs_error),
                       role='source capability; no model output or hardware accuracy',
                       gt_kind='registered structured-light object scan reprojected with estimated poses')
            rows.append(row)
            i = len(rows) - 1; x = (i % 3) * 640; y = (i // 3) * 280
            thumb = rgb.copy(); thumb.thumbnail((320, 240))
            overlay = np.asarray(rgb).copy()
            overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([255, 40, 40])).astype('uint8')
            marked = Image.fromarray(overlay); marked.thumbnail((320, 240))
            montage.paste(thumb, (x, y + 35)); montage.paste(marked, (x + 320, y + 35))
            draw.text((x + 5, y + 2), f'{Path(scene).name} | ordinal {selection["selected_names"].index(name)}', fill='black')
            draw.text((x + 5, y + 17), f'object GT={int(obj.sum())} px; raw missing={int((obj & (raw == 0)).sum())} px', fill='black')
    montage.save(OUT / 'sample_contact_sheet.jpg', quality=90)
    summary = dict(revision=REV, scenes=selections, frames=len(rows),
                   download_bytes=sum(x['bytes'] for x in receipts),
                   object_valid_gt_pixels=sum(x['object_valid_gt_pixels'] for x in rows),
                   raw_missing_on_object_pixels=sum(x['raw_missing_on_object_pixels'] for x in rows),
                   checks=dict(dimensions=True, positive_finite_intrinsics=True,
                               selected_file_sizes=True, selected_timestamps_match_csv_and_pose=True,
                               pose_finite_unit_quaternions=True,
                               official_lfs_sha=sum(x['official_lfs_sha_checked'] for x in receipts)),
                   wall_s=time.perf_counter() - start, rows=rows,
                   limitation='Twelve sampled frames from two scenes; object scan registration is not perfect ground truth; no independent head-worn validation.')
    save('summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'scenes')}))


if __name__ == '__main__':
    main()

"""Bounded ASE download and camera-anchored BODY/HEAD coverage pilot.

Official sources and limitations are in ASE_BODY_QUERY_PILOT_20260909.md.
Never imports model weights or trains. Download URLs stay in ignored artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
import requests
from scipy.spatial.transform import Rotation

from ase_body_query_geometry import ray_depth_coverage

ROOT = Path(__file__).resolve().parents[4]


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def artifact_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to((ROOT / 'artifacts.local').resolve()):
        raise ValueError('Outputs must be under canonical artifacts.local')
    return path


def download(args):
    """One official ten-scene chunk, SHA1 checked before selective extraction."""
    output = artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    filename = 'train_chunk_0000000.zip'
    entries = json.loads(Path(args.cdn_file).read_text(encoding='utf-8-sig'))
    entry = next(x for x in entries if x['filename'] == filename)
    if not entry['cdn'].startswith('https://') or len(entry['sha']) != 40:
        raise ValueError('Expected official HTTPS CDN and SHA1 metadata')
    archive = output / filename
    partial = archive.with_suffix('.zip.partial')
    started = time.monotonic()
    if not archive.exists():
        if partial.exists():
            raise FileExistsError(f'Retain incomplete evidence; use another output: {partial}')
        if shutil.disk_usage(output).free < args.max_bytes * 2:
            raise RuntimeError('Insufficient free artifact space for archive and extraction')
        # This operation downloads an authorized URL; TLS verification stays on.
        with requests.get(entry['cdn'], stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            length = int(response.headers.get('Content-Length', 0))
            if length > args.max_bytes:
                raise ValueError('Chunk exceeds the predeclared download cap')
            received = 0
            last = 0.
            with partial.open('xb') as stream:
                for block in response.iter_content(1024 * 1024):
                    received += len(block)
                    if received > args.max_bytes:
                        raise ValueError('Download exceeded byte cap; retained .partial')
                    stream.write(block)
                    elapsed = time.monotonic() - started
                    if elapsed - last >= 20:
                        print(f'downloaded_bytes={received} seconds={elapsed:.1f}', flush=True)
                        last = elapsed
        if digest(partial, 'sha1') != entry['sha']:
            raise ValueError('Official SHA1 mismatch; retained .partial')
        partial.rename(archive)
    if digest(archive, 'sha1') != entry['sha']:
        raise ValueError('Existing archive SHA1 mismatch')
    selected = {str(x) for x in args.scenes}
    extracted = []
    with ZipFile(archive) as zipped:
        total = 0
        for item in zipped.infolist():
            parts = Path(item.filename.replace('\\', '/')).parts
            if item.is_dir() or not any(p in selected for p in parts):
                continue
            # Keep only the geometry pilot inputs, not dense full trajectories.
            if not (any(p in ('rgb', 'depth') for p in parts)
                    or parts[-1] in ('trajectory.csv', 'trajectory.txt', 'ase_scene_language.txt')):
                continue
            target = (output / 'scenes' / item.filename).resolve()
            if not target.is_relative_to((output / 'scenes').resolve()):
                raise ValueError('Unsafe ZIP member path')
            total += item.file_size
            if total > args.max_bytes:
                raise ValueError('Selected extraction exceeded cap')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError('Extraction exists; do not overwrite source evidence')
            with zipped.open(item) as source, target.open('xb') as dest:
                shutil.copyfileobj(source, dest)
            extracted.append(dict(path=str(target.relative_to(output)), bytes=item.file_size))
    receipt = dict(status='DOWNLOADED_VERIFIED', archive=filename,
                   sha1=entry['sha'], sha256=digest(archive), bytes=archive.stat().st_size,
                   cdn_manifest_sha256=digest(args.cdn_file), scene_ids=args.scenes,
                   extracted=extracted, elapsed_seconds=time.monotonic()-started)
    dump(output / 'download-receipt.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k != 'extracted'}))


def load_trajectory(path):
    """Official ASE CSV field positions; stop on ambiguity instead of guessing."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) < 10 or 'tx' not in header[3].lower() or 'qx' not in header[6].lower():
            raise ValueError(f'Unsupported ASE trajectory header: {header[:10]}')
        rows = list(reader)
    transforms = []
    for row in rows:
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(np.asarray(row[6:10], float)).as_matrix()
        transform[:3, 3] = np.asarray(row[3:6], float)
        transforms.append(transform)
    return transforms


def preview(rgb, coverage, path, title):
    rgb = np.array(rgb, copy=True)
    membership = coverage['membership'].reshape(2, 6, *rgb.shape[:2]).any(axis=1)
    for mask, color in zip(membership, ([0, 200, 255], [255, 80, 120])):
        rgb[mask] = (rgb[mask] * .45 + np.asarray(color) * .55).astype(np.uint8)
    image = Image.fromarray(rgb).rotate(-90, expand=True)
    image.thumbnail((640, 640))
    canvas = Image.new('RGB', (image.width, image.height + 44), '#171b22')
    canvas.paste(image, (0, 44))
    ImageDraw.Draw(canvas).text((8, 8), title, fill='white')
    canvas.save(path)


def audit(args):
    from ase_camera import AseCalibration
    output = artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows, input_hashes = [], []
    started = time.monotonic()
    for scene_id in args.scenes:
        candidates = [p for p in Path(args.source).rglob(str(scene_id))
                      if p.is_dir() and (p/'rgb').is_dir() and (p/'depth').is_dir()]
        if len(candidates) != 1:
            raise ValueError(f'Expected one scene directory for {scene_id}, got {len(candidates)}')
        scene = candidates[0]
        trajectories = [p for p in [scene/'trajectory.csv', scene/'trajectory.txt'] if p.exists()]
        if len(trajectories) != 1:
            raise ValueError('Missing or ambiguous trajectory')
        transforms = load_trajectory(trajectories[0])
        images = sorted((scene/'rgb').glob('vignette*.jpg'))
        if len(images) != len(transforms):
            raise ValueError('Cannot use frame-index trajectory alignment: count mismatch')
        chosen = np.unique(np.linspace(0, len(images)-1, args.frames, dtype=int))
        for position, index in enumerate(chosen):
            rgb_path = images[index]
            frame = int(rgb_path.stem.removeprefix('vignette'))
            if frame != index:
                raise ValueError('Non-contiguous image IDs; frame-index alignment unverified')
            depth_path = scene/'depth'/f'depth{frame:07d}.png'
            rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
            depth_mm = np.asarray(Image.open(depth_path))
            if depth_mm.shape != rgb.shape[:2] or depth_mm.dtype.kind not in 'iu':
                raise ValueError('Expected native integer depth with matching RGB shape')
            if position == 0:
                calibration = AseCalibration(rgb.shape[1])
                if rgb.shape[0] != rgb.shape[1]:
                    raise ValueError('Expected unrotated square ASE camera image')
                yy, xx = np.indices(depth_mm.shape)
                pixels = np.stack([xx, yy], axis=-1)
                rays = calibration.unproject(pixels)
                recovered = calibration.project(rays)
                residual = np.linalg.norm(recovered - pixels, axis=-1)
                if not np.isfinite(residual).any() or np.nanmax(residual) > 1e-4:
                    raise ValueError('Calibration roundtrip check failed')
            coverage = ray_depth_coverage(rays, depth_mm.astype(float)/1000.,
                        transforms[index] @ calibration.T_device_camera,
                        [0., 0., -1.], camera_height_m=1.70)
            record = dict(scene_id=scene_id, frame_id=frame,
                          four_way=coverage['four_way'], raw_counts=coverage['raw_counts'].tolist(),
                          near=coverage['near'].tolist(), unknown_pixels=coverage['unknown_count'],
                          valid_pixels=coverage['valid_count'], image_pixels=int(depth_mm.size))
            rows.append(record)
            for source in [rgb_path, depth_path]:
                input_hashes.append(dict(path=str(source), sha256=digest(source)))
            if position in (0, len(chosen)//2, len(chosen)-1):
                preview(rgb, coverage, output/f'scene-{scene_id}-frame-{frame}.jpg',
                        f'Scene {scene_id} / {frame}: {record["four_way"]} (proxy)')
        input_hashes.append(dict(path=str(trajectories[0]), sha256=digest(trajectories[0])))
        print(f'scene={scene_id} frames={len(chosen)} complete', flush=True)
    counts = Counter(r['four_way'] for r in rows)
    head_scenes = sorted({r['scene_id'] for r in rows if r['four_way']=='HEAD_ONLY'})
    # This screening criterion admits a follow-up; it is not a method success gate.
    useful = counts['HEAD_ONLY'] >= 8 and len(head_scenes) >= 2 and counts['BODY_ONLY'] >= 8
    result = dict(status='COVERAGE_USEFUL_FOR_FOLLOWUP' if useful else 'COVERAGE_INSUFFICIENT_FOR_HEAD_ONLY_TEST',
                  frames=len(rows), scene_ids=args.scenes, four_way=dict(counts),
                  head_only_scene_ids=head_scenes, frame_records=rows,
                  elapsed_seconds=time.monotonic()-started,
                  backend='NumPy CPU; bounded image decode and geometry ETL',
                  height_authority='ASSUMED_CAMERA_ANCHORED_PROXY_NOT_MEASURED_FLOOR',
                  gravity_authority='ASE_WORLD_Z_UP_ASSUMPTION_CHECK_VISUALS',
                  count_authority='NATIVE_PIXELS_NOT_V1_RESOLUTION_EQUIVALENT',
                  model_evaluation='NOT_RUN_REQUIRES_CALIBRATION_AND_COVERAGE_REVIEW',
                  evidence_scope='DEVELOPMENT_SYNTHETIC_VISIBLE_SURFACES_ONLY')
    dump(output/'inputs.json', input_hashes)
    dump(output/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='frame_records'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    dl = sub.add_parser('download')
    dl.add_argument('--cdn-file', required=True)
    dl.add_argument('--output', required=True)
    dl.add_argument('--scenes', type=int, nargs='+', default=[0, 4, 8], choices=range(10))
    dl.add_argument('--max-bytes', type=int, default=4*1024**3)
    check = sub.add_parser('audit')
    check.add_argument('--source', required=True)
    check.add_argument('--output', required=True)
    check.add_argument('--scenes', type=int, nargs='+', default=[0, 4, 8])
    check.add_argument('--frames', type=int, default=24)
    args = parser.parse_args()
    if args.command == 'download':
        download(args)
    else:
        audit(args)


if __name__ == '__main__':
    main()

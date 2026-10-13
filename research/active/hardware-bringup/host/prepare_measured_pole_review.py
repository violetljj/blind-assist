"""Prepare offline three-view line annotation without guessing metric measurements.

Real Atom input is fisheye-undistorted then transposed exactly as AtomCamera does.
That image chart is distinct from physical camera axes: image_ray_to_camera stays
null until independently supplied, so preparation alone cannot admit geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from atom_camera import AtomCamera, CALIBRATION, ROOT

ORDER = ('baseline_0cm', 'baseline_20cm', 'validation_10cm')
TOOL_VERSION = 'prepare-measured-pole-review-1.0'
TRANSPOSE_NOTE = (
    'raw 640x480; OpenCV fisheye remap with R=I then transpose to 480x640; '
    'q_upright=S q_raw_undistorted, S=[[0,1,0],[1,0,0],[0,0,1]], det=-1; '
    'physical sensor parity not established; image_ray_to_camera must be supplied '
    'from an independently established physical axis mapping before geometry evaluation'
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_picture(view: dict, index: int) -> tuple[np.ndarray, list]:
    image = view.get('image', {})
    width, height = image.get('width'), image.get('height')
    points = view.get('line_endpoints_px')
    if not isinstance(width, int) or not isinstance(height, int) or width < 16 or height < 16:
        raise ValueError('Synthetic template needs explicit image width/height.')
    points_array = np.asarray(points, dtype=float)
    if points_array.shape != (2, 2) or not np.isfinite(points_array).all():
        raise ValueError('Synthetic template needs projected line_endpoints_px for drawing.')
    if np.any(points_array < 0) or np.any(points_array[:, 0] >= width) or np.any(points_array[:, 1] >= height):
        raise ValueError('Synthetic endpoints must lie inside the image.')
    picture = np.full((height, width, 3), (238, 241, 235), np.uint8)
    for x in range(0, width, 40):
        cv2.line(picture, (x, 0), (x, height-1), (212, 221, 212), 1)
    for y in range(0, height, 40):
        cv2.line(picture, (0, y), (width-1, y), (212, 221, 212), 1)
    p1, p2 = (tuple(np.rint(p).astype(int)) for p in points_array)
    cv2.line(picture, p1, p2, (28, 49, 36), 2, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(picture, p, 4, (47, 87, 63), 1, cv2.LINE_AA)
    cv2.putText(picture, 'SYNTHETIC EXAMPLE', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (43, 78, 112), 1, cv2.LINE_AA)
    cv2.putText(picture, f'{index+1}: {view["view_id"]}', (12, 45), cv2.FONT_HERSHEY_SIMPLEX, .45, (59, 78, 61), 1, cv2.LINE_AA)
    cv2.putText(picture, 'Click the two small endpoint rings', (12, height-14), cv2.FONT_HERSHEY_SIMPLEX, .43, (59, 78, 61), 1, cv2.LINE_AA)
    return picture, points_array.tolist()


def prepare(args: argparse.Namespace) -> dict:
    source_template = args.template.resolve(strict=True)
    model = json.loads(source_template.read_text(encoding='utf-8-sig'))
    if model.get('schema_version') != 'measured-pole-geometry-v1':
        raise ValueError('Expected measured-pole-geometry-v1 schema.')
    views = model.get('views', [])
    if len(views) != 3 or tuple(v.get('view_id') for v in views) != ORDER:
        raise ValueError('Expected three views in 0cm, 20cm, 10cm order.')
    if [v.get('role') for v in views] != ['FIT', 'FIT', 'VALIDATION']:
        raise ValueError('First two views must be FIT and the third VALIDATION.')
    if args.synthetic_images and model.get('evidence_role') != 'SYNTHETIC_EXAMPLE':
        raise ValueError('--synthetic-images requires explicit SYNTHETIC_EXAMPLE evidence role.')
    if args.raw and model.get('evidence_role') != 'MEASURED_INPUT':
        raise ValueError('Real image preparation requires a MEASURED_INPUT template.')

    artifact_root = (ROOT / 'artifacts.local').resolve(strict=True)
    out = args.out.resolve()
    if not out.is_relative_to(artifact_root):
        raise ValueError('Output must resolve inside the canonical artifacts.local tree.')
    names = [f'{i:02d}_{view["view_id"]}_upright.png' for i, view in enumerate(views)]
    expected = names + ['review-template.json', 'preprocessing-receipt.json', 'USAGE.txt']
    if any((out / name).exists() for name in expected):
        raise FileExistsError('Output already contains a preparation artifact; choose a new output directory.')

    calibration = args.calibration.resolve(strict=True) if args.raw else None
    camera = AtomCamera(calibration) if args.raw else None
    records = []
    pictures = []
    for i, view in enumerate(views):
        original_endpoints = view.get('line_endpoints_px')
        if args.raw:
            raw_path = args.raw[i].resolve(strict=True)
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if raw is None:
                raise ValueError(f'Cannot decode source image: {raw_path}')
            if (raw.shape[1], raw.shape[0]) != tuple(camera.size):
                raise ValueError(f'Expected original Atom raw dimensions {camera.size}, got {raw.shape[1::-1]}: {raw_path}')
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            picture = cv2.cvtColor(camera.undistort_upright(rgb), cv2.COLOR_RGB2BGR)
            view['K'] = camera.K_upright.tolist()
            previous_q = view.get('image_ray_to_camera')
            view['image_ray_to_camera'] = None
            pixel_domain = 'atom_undistorted_upright_transpose'
            note = TRANSPOSE_NOTE
            record = dict(view_id=view['view_id'], source_path=str(raw_path), source_sha256=sha(raw_path),
                          source_size_px=list(camera.size), calibration_ref=str(calibration),
                          calibration_sha256=sha(calibration), recipe='AtomCamera.undistort_upright(balance=0)',
                          image_ray_to_camera_status='MISSING_PHYSICAL_AXIS_MAPPING',
                          previous_template_image_ray_to_camera=previous_q)
            view.setdefault('image', {}).update(calibration_ref=str(calibration), source_path=str(raw_path), source_sha256=record['source_sha256'])
        else:
            picture, expected_points = synthetic_picture(view, i)
            pixel_domain = view.get('image', {}).get('pixel_domain') or 'synthetic_pinhole'
            note = 'Synthetic pinhole image generated from the geometry tool projected endpoints; no real sensor capture or metric accuracy claim.'
            record = dict(view_id=view['view_id'], source_template=str(source_template),
                          source_template_sha256=sha(source_template), recipe='synthetic_endpoint_drawing',
                          synthetic_reference_endpoints_px=expected_points,
                          evidence_role='SYNTHETIC_EXAMPLE')
            view.setdefault('image', {}).setdefault('calibration_ref', 'synthetic-explicit-K')
        view['image'].update(path=names[i], width=int(picture.shape[1]), height=int(picture.shape[0]),
                             pixel_domain=pixel_domain, undistortion_note=note,
                             preprocessing_receipt='preprocessing-receipt.json')
        view['line_endpoints_px'] = None
        record['processed_path'] = names[i]
        record['cleared_template_endpoints'] = original_endpoints
        records.append(record); pictures.append(picture)

    out.mkdir(parents=True, exist_ok=True)
    for name, picture, view, record in zip(names, pictures, views, records):
        target = out / name
        if not cv2.imwrite(str(target), picture):
            raise IOError(f'Failed to write {target}')
        digest = sha(target)
        view['image']['processed_sha256'] = digest
        record['processed_sha256'] = digest
    receipt = dict(tool_version=TOOL_VERSION, created_at=datetime.now(timezone.utc).isoformat(),
                   evidence_role=model['evidence_role'], source_template=str(source_template),
                   source_template_sha256=sha(source_template), generator_sha256=sha(Path(__file__)),
                   atom_camera_sha256=sha(Path(__file__).with_name('atom_camera.py')),
                   original_metric_measurement_fields_preserved=True, metric_numbers_invented=False,
                   real_preparation_does_not_establish_image_ray_to_physical_camera_mapping=bool(args.raw),
                   views=records)
    model['preparation'] = {k: v for k, v in receipt.items() if k != 'views'}
    model['annotation'] = dict(tool_version='measured-pole-review-1.0', completed_at=None,
                               annotated_by=None, note=None, export_kind='DRAFT')
    (out / 'review-template.json').write_text(json.dumps(model, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    (out / 'preprocessing-receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    usage = (
        'Open research/active/hardware-bringup/host/measured_pole_review.html directly in Chrome or Edge.\n'
        'Import review-template.json, then select the three generated PNGs in order 0cm / 20cm / 10cm.\n'
        'The first two views reconstruct; the third is validation only.\n'
        'Fill real measurement fields and image_ray_to_camera from actual sources; null is never zero.\n'
        'Point to the two endpoints of the same identifiable straight edge in every image.\n'
        'Save a draft at any time; geometry export requires complete metadata and matching image hashes.\n'
        'SYNTHETIC_EXAMPLE remains synthetic in every export.\n'
    )
    (out / 'USAGE.txt').write_text(usage, encoding='utf-8')
    return dict(output=str(out), template=str(out / 'review-template.json'),
                evidence_role=model['evidence_role'], images=names,
                physical_axis_mapping_status='MISSING' if args.raw else 'FROM_SYNTHETIC_TEMPLATE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True, help='Geometry v1 template; measurements are preserved, never inferred.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--raw', type=Path, nargs=3, metavar=('RAW_0', 'RAW_20', 'RAW_10'), help='Three original Atom JPEGs in FIT/FIT/VALIDATION order.')
    mode.add_argument('--synthetic-images', action='store_true', help='Draw the geometry agent synthetic-example projected endpoints.')
    parser.add_argument('--calibration', type=Path, default=CALIBRATION)
    parser.add_argument('--out', type=Path, required=True, help='New case directory under artifacts.local; prior outputs are not overwritten.')
    args = parser.parse_args()
    try:
        result = prepare(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'Preparation stopped: {error}\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

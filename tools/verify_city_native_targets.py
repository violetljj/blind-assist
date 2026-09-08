"""CUDA native visible-surface labels and exact-instance render agreement.

Fixed before outcomes: 3cm agreement, >=3 visible pixels, no unexplained
nearer-clone pixels for a reliable target/frame. No threshold fitting.
Absent floor or unreliable target geometry remains UNKNOWN.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/active/dtr-r0/nearfield'
sys.path.insert(0, str(SOURCE))
from contact_retina_spec import BODY_BOXES
from worlds_verify import camera_points, corridor_masks

POLICY = dict(agreement_m=.03, minimum_visible_pixels=3,
    maximum_unexplained_nearer_clone_pixels=0, query_range_m=3.,
    negative_semantics='NO_VISIBLE_IN_QUERY_SUPPORT_NOT_FREE_SPACE',
    missing_depth='UNKNOWN', missing_floor='UNKNOWN',
    target_identity='Declared exact original mesh/instance transform plus isolated/full render agreement',
    independent_geometry='Only optional supplied raycheck; missing is an explicit evidence gap')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_depth(path, device='cuda'):
    array = np.load(path, allow_pickle=False)
    if array.shape != (360, 640) or array.dtype != np.dtype('<f4'):
        raise ValueError('Expected native float32 360x640 axial metres')
    if not np.isfinite(array).all() or not ((array >= 0) & (array < 100)).all():
        raise ValueError('Invalid native depth values')
    return torch.from_numpy(array).to(device)


def in_query(depth, camera, floor):
    """Translate and undo wearer yaw before projection to avoid world-float loss.

    This is algebraically the full UE camera-to-world followed by inverse wearer
    yaw; wearer pitch remains zero. Native axial depth retains camera pitch.
    """
    if any(not np.isfinite(float(camera[k])) for k in ('x', 'y', 'z', 'pitch', 'yaw', 'roll')):
        raise ValueError('Invalid camera pose')
    local = dict(x=0., y=0., z=float(camera['z']) - floor,
                 yaw=0., pitch=float(camera['pitch']), roll=float(camera['roll']))
    points = camera_points(depth, local)
    valid = torch.isfinite(depth) & (depth > 0) & (depth < 100)
    wearer = dict(x=0., y=0., z=0., pitch=0., yaw=0., roll=0.)
    return corridor_masks(points, valid, wearer, POLICY['query_range_m']), valid


def scene_labels(native, camera, floor):
    if floor is None:
        return torch.full((2, *native.shape), -1, device=native.device, dtype=torch.int8), [-1, -1]
    query, valid = in_query(native, camera, floor)
    mask = torch.where(valid[None], query.to(torch.int8), -1).to(torch.int8)
    near = (query.sum((-2, -1)) >= POLICY['minimum_visible_pixels']).to(torch.int8)
    if not bool(valid.any()):
        near[:] = -1
    return mask, near.tolist()


def target_label(native, isolated, camera, floor, ray_status=None):
    known = (native > 0) & (native < 100)
    hit = (isolated > 0) & (isolated < 100)
    comparable = known & hit
    agree = comparable & ((native - isolated).abs() <= POLICY['agreement_m'])
    occluded = comparable & (isolated > native + POLICY['agreement_m'])
    closer = comparable & (isolated < native - POLICY['agreement_m'])
    row = dict(projected_pixels=int(hit.sum()), native_agree_pixels=int(agree.sum()),
        occluded_pixels=int(occluded.sum()), unexplained_nearer_clone_pixels=int(closer.sum()),
        unknown_native_under_clone_pixels=int((hit & ~known).sum()), raycheck=ray_status or 'NOT_SUPPLIED')
    reasons = []
    if floor is None:
        reasons.append('MISSING_FLOOR')
    if row['native_agree_pixels'] < POLICY['minimum_visible_pixels']:
        reasons.append('INSUFFICIENT_RENDER_AGREEMENT')
    if row['unexplained_nearer_clone_pixels'] > POLICY['maximum_unexplained_nearer_clone_pixels']:
        reasons.append('UNEXPLAINED_NEARER_CLONE')
    if ray_status not in (None, 'PASS'):
        reasons.append('INDEPENDENT_RAYCHECK_UNRESOLVED')
    row['uncertainty_reasons'] = reasons
    reliable = not reasons
    mask = torch.full((2, *native.shape), -1, dtype=torch.int8, device=native.device)
    if reliable:
        query, _ = in_query(native, camera, floor)
        positive = query & agree[None]
        mask = torch.where(known[None] & ~closer[None], positive.to(torch.int8), -1).to(torch.int8)
        # Fewer than 3 native in-query pixels cannot be a target alert opportunity.
        for h in range(2):
            if int(positive[h].sum()) < POLICY['minimum_visible_pixels']:
                mask[h][positive[h]] = -1
    row.update(status='EVALUABLE' if reliable else 'UNKNOWN',
        reliable_in_query_pixels=(mask == 1).sum((-2, -1)).tolist())
    return mask, row


def overlay(rgb_path, mask, out, title):
    """Scientific support boundaries over unchanged RGB, no image generation."""
    with Image.open(rgb_path) as source:
        canvas = source.convert('RGB')
    for h, color in enumerate(((0, 255, 130), (255, 90, 220))):
        positive = Image.fromarray(((mask[h] == 1) * 255).astype(np.uint8))
        interior = positive.filter(ImageFilter.MinFilter(3))
        boundary = np.array(positive) - np.array(interior)
        canvas.paste(color, (0, 0), Image.fromarray(boundary))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 640, 36), fill=(0, 0, 0))
    draw.text((5, 4), title, fill='white')
    draw.text((5, 20), 'Green BODY / magenta HEAD; positive visible in-query support only', fill='white')
    canvas.save(out)


@torch.inference_mode()
def run(args):
    started = time.perf_counter()
    root, out = args.capture.resolve(strict=True), args.output.resolve()
    artifacts = (ROOT / 'artifacts.local').resolve()
    if not root.is_relative_to(artifacts) or not out.is_relative_to(artifacts) or out == artifacts or out.exists():
        raise ValueError('Canonical capture and fresh artifact output required')
    receipt_path, spec_path = root / 'receipt.json', root / 'source/spec.json'
    receipt, spec = read(receipt_path), read(spec_path)
    if receipt.get('status') != 'PASS' or receipt.get('source_unchanged') is not True:
        raise ValueError('Completed source-preserving capture required')
    if sha(spec_path) != receipt['spec_sha256']:
        raise ValueError('Spec identity mismatch')
    frames = read(root / 'model/dataset.json')['frames']
    ids = [int(f['sample_index']) for f in frames]
    if ids != list(range(len(spec['cases']))) or receipt['frame_count'] != len(ids):
        raise ValueError('Capture frame/case identity mismatch')
    probes = {p['case']: p for p in receipt.get('native_floor_probes', [])}
    targets = spec.get('native_targets', [])
    target_ids = [t['target_id'] for t in targets]
    if len(set(target_ids)) != len(target_ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', t) for t in target_ids):
        raise ValueError('Unique simple target IDs required')
    ray_rows = {}
    if args.raycheck:
        rays = read(args.raycheck)
        if rays.get('schema') != 'city-native-target-raycheck-v1':
            raise ValueError('Raycheck schema mismatch')
        ray_rows = {(r['target_id'], int(r['sample_index'])): r for r in rays['rows']}
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for dense native projection')
    out.mkdir(parents=True)
    (out / 'targets').mkdir()
    (out / 'overlays').mkdir()
    report = dict(schema='city-native-target-validation-v1', status='BUILDING', policy=POLICY,
        actual_backend='cuda', device=torch.cuda.get_device_name(), rows=[], targets={},
        spec_sha256=sha(spec_path), receipt_sha256=sha(receipt_path),
        raycheck_sha256=sha(args.raycheck) if args.raycheck else None,
        source_sha256={p.name: sha(p) for p in (Path(__file__), SOURCE / 'worlds_verify.py', SOURCE / 'contact_retina_spec.py')})
    write(out / 'protocol.json', report)
    n = len(ids)
    near = np.full((n, 2), -1, np.int8)
    support = np.lib.format.open_memmap(out / 'support.npy', mode='w+', dtype=np.int8, shape=(n, 2, 360, 640))
    target_arrays = {t: np.lib.format.open_memmap(out / 'targets' / (t + '.npy'), mode='w+', dtype=np.int8,
        shape=(n, 2, 360, 640)) for t in target_ids}
    try:
        for i, case in enumerate(spec['cases']):
            probe = probes.get(case['name'], {})
            floor = None
            if probe.get('hit') is True and probe.get('point_m') is not None:
                p = probe['point_m']
                if (len(p) == 3 and all(np.isfinite(float(v)) for v in p)
                    and float(p[2]) < float(case['camera']['z'])
                    and abs(float(p[0]) - float(case['camera']['x'])) <= .05
                    and abs(float(p[1]) - float(case['camera']['y'])) <= .05):
                    floor = float(p[2])
            native_path = root / f'evaluator/native/{i:04d}.npy'
            native = load_depth(native_path)
            mask, near[i] = scene_labels(native, case['camera'], floor)
            support[i] = mask.cpu().numpy()
            row = dict(sample_index=i, floor_probe=probe, floor_z_m=floor, native_sha256=sha(native_path),
                near=near[i].tolist(), positive_pixels=(mask == 1).sum((-2, -1)).tolist(),
                unknown_pixels=(mask == -1).sum((-2, -1)).tolist(), status='EVALUABLE' if floor is not None else 'UNKNOWN')
            report['rows'].append(row)
            rgb = root / 'model' / frames[i]['rgb_path']
            if i in (0, n // 2, n - 1):
                overlay(rgb, support[i], out / f'overlays/scene-{i:04d}.png', f'Scene frame {i} | floor {floor}')
            for target in targets:
                tid = target['target_id']
                isolated_path = root / f'evaluator/isolated/{tid}/{i:04d}.npy'
                target_arrays[tid][i] = -1
                if not isolated_path.exists():
                    evidence = dict(sample_index=i, status='UNKNOWN', reason='Missing isolated original-instance render')
                else:
                    ray_status = ray_rows.get((tid, i), {}).get('status', 'UNKNOWN') if args.raycheck else None
                    tm, evidence = target_label(native, load_depth(isolated_path), case['camera'], floor, ray_status)
                    target_arrays[tid][i] = tm.cpu().numpy()
                    evidence.update(sample_index=i, isolated_sha256=sha(isolated_path))
                    if args.raycheck:
                        ray_evidence = ray_rows.get((tid, i), {})
                        evidence['raycheck_source_component'] = ray_evidence.get('source_component')
                        evidence['raycheck_unresolved_rays'] = [
                            r for r in ray_evidence.get('rows', []) if r.get('status') == 'UNKNOWN']
                    if (tm == 1).any():
                        overlay(rgb, target_arrays[tid][i], out / f'overlays/{tid}-{i:04d}.png',
                                f'{tid} frame {i} | {evidence["status"]} | native-render agreement')
                report['targets'].setdefault(tid, []).append(evidence)
        support.flush()
        for array in target_arrays.values():
            array.flush()
        np.save(out / 'near.npy', near, allow_pickle=False)
        entries = []
        for target in targets:
            tid = target['target_id']
            evaluable = any(r['status'] == 'EVALUABLE' for r in report['targets'][tid])
            query_coverage = any(any(v >= 3 for v in r.get('reliable_in_query_pixels', []))
                                 for r in report['targets'][tid])
            report.setdefault('target_coverage', {})[tid] = dict(
                status='EVALUABLE' if query_coverage else 'NOT_EVALUABLE',
                reason='Reliable visible in-query support' if query_coverage else 'No reliable in-query category opportunity',
                native_identity_agreement_frames=sum(r.get('native_agree_pixels', 0) >= 3 and
                    r.get('unexplained_nearer_clone_pixels', 1) == 0 for r in report['targets'][tid]),
                collision_raycheck_frames=sum(r.get('raycheck') == 'PASS' for r in report['targets'][tid]))
            entries.append(dict(**target, frame_indices=ids, masks=f'targets/{tid}.npy',
                status='EVALUABLE' if evaluable else 'UNKNOWN',
                authority='Isolated declared original instance/full native axial-depth agreement; measured floor; '
                          + ('supplied independent raycheck gate' if args.raycheck else 'NO_INDEPENDENT_RAYCHECK')))
        manifest = dict(schema='city-native-route-labels-v1', sample_indices=ids, near='near.npy', support='support.npy',
            targets=entries, provenance=dict(policy=POLICY, body_boxes=BODY_BOXES,
                spec_sha256=report['spec_sha256'], receipt_sha256=report['receipt_sha256'],
                floor='Per-frame native downward collision hit; not uniform planar-floor truth',
                semantics='Visible in-query native surfaces only; unknown/occluded/unloaded geometry is not certified clear',
                independent_raycheck='SUPPLIED' if args.raycheck else 'MISSING'))
        write(out / 'native-route-labels.json', manifest)
        report.update(status='PASS', execution='PASS_NOT_UNIVERSAL_LABEL_ACCEPTANCE', elapsed_seconds=time.perf_counter()-started,
            near_sha256=sha(out / 'near.npy'), support_sha256=sha(out / 'support.npy'),
            target_mask_sha256={tid: sha(out / 'targets' / (tid + '.npy')) for tid in target_ids},
            label_manifest_sha256=sha(out / 'native-route-labels.json'))
        write(out / 'label-validation.json', report)
        print(json.dumps(dict(status='PASS', frames=n, targets=len(targets), output=str(out))))
    except Exception as error:
        write(out / 'failure.json', dict(status='FAIL', error=repr(error)))
        raise
    finally:
        del support
        target_arrays.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raycheck', type=Path)
    run(parser.parse_args())

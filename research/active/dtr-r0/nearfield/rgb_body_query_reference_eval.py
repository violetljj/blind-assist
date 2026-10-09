"""Reference-domain intrusion evaluation, never full scene clearance truth.

Model predictions are produced separately from RGB/K. Panoptic target masks
enter only this evaluator: query classification is thus a localization-oracle
geometry diagnostic. Pixel metrics measure the same annotated visible domain.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from rgb_body_query_geometry import camera_queries, resize_intrinsics
from rgb_body_query_input_diagnostic import sha, write

TARGET_CLASSES = (4, 9, 22, 24, 28)
IGNORE, UNKNOWN, NEGATIVE, POSITIVE = 255, 2, 0, 1
MIN_SUPPORT = 16


def load_depth(path):
    values = np.frombuffer(gzip.decompress(Path(path).read_bytes()), dtype='<f2')
    if values.size < 2 or not np.isfinite(values[:2]).all():
        raise ValueError('Invalid depth header')
    h, w = map(int, values[:2])
    if h <= 0 or w <= 0 or values.size != 2+h*w:
        raise ValueError('Depth header/payload mismatch')
    return values[2:].reshape(h, w).astype(np.float32)


def rays(k, shape):
    yy, xx = np.indices(shape, dtype=np.float64)
    inv = np.linalg.inv(np.asarray(k, np.float64))
    return inv[0, 0]*xx + inv[0, 1]*yy + inv[0, 2], inv[1, 0]*xx + inv[1, 1]*yy + inv[1, 2]


def ray_interval(rx, ry, query):
    """Closed axial-Z interval in which a ray intersects a public box."""
    low = np.full(rx.shape, query['low'][2], dtype=np.float64)
    high = np.full(rx.shape, query['high'][2], dtype=np.float64)
    for ray, lo, hi in zip((rx, ry), query['low'][:2], query['high'][:2]):
        nz = np.abs(ray) > 1e-12
        a = np.divide(lo, ray, out=np.zeros_like(ray), where=nz)
        b = np.divide(hi, ray, out=np.zeros_like(ray), where=nz)
        low = np.maximum(low, np.where(nz, np.minimum(a, b), -np.inf))
        high = np.minimum(high, np.where(nz, np.maximum(a, b), np.inf))
        high[(~nz) & ((lo > 0) | (hi < 0))] = -np.inf
    return low, high, low <= high+1e-12


def interval_labels(cres, zed, target, rx, ry, query):
    """Two estimated depths bound a reference interval, not physical accuracy."""
    entry, exit, reachable = ray_interval(rx, ry, query)
    domain = target & reachable
    paired = np.isfinite(cres) & np.isfinite(zed) & (cres > 0) & (zed > 0) & (cres <= 80) & (zed <= 80)
    good = paired & (np.abs(cres-zed) <= np.maximum(.25, .25*np.minimum(cres, zed)))
    zlo, zhi = np.minimum(cres, zed), np.maximum(cres, zed)
    inside = good & (zlo >= entry-1e-12) & (zhi <= exit+1e-12)
    outside = good & ((zhi < entry-1e-12) | (zlo > exit+1e-12))
    labels = np.full(cres.shape, IGNORE, dtype=np.uint8)
    labels[domain] = UNKNOWN
    labels[domain & outside] = NEGATIVE
    labels[domain & inside] = POSITIVE
    pos, unk = int((labels == POSITIVE).sum()), int((labels == UNKNOWN).sum())
    state = 'POSITIVE' if pos >= MIN_SUPPORT else ('NEGATIVE_VISIBLE_TARGET_SET' if pos == 0 and unk == 0 else 'UNKNOWN')
    return labels, dict(name=query['name'], state=state, positive_pixels=pos,
                       unknown_pixels=unk, domain_pixels=int(domain.sum()),
                       vacuous_negative=state.startswith('NEGATIVE') and not domain.any())


def reference(repo, output, budget_s):
    started = time.perf_counter()
    source = repo/'artifacts.local/work/rgb-body-query-dev-20261009/sequence'
    receipt = json.loads((source/'acquisition_receipt.json').read_text('utf-8-sig'))
    plan = json.loads((source/'planned_files.json').read_text('utf-8-sig'))
    if receipt['status'] != 'COMPLETE':
        raise ValueError('Need complete sequence source')
    files = {(f['frame'], f['kind']): f for f in receipt['verified_files']}
    desc_ref = next(r for r in plan['references'] if r['path'].endswith('description.json'))
    if sha(desc_ref['path']) != desc_ref['sha256']:
        raise ValueError('Changed public calibration')
    desc = json.loads(Path(desc_ref['path']).read_text('utf-8-sig'))
    c = desc['session_camera_details'][desc['session_camera_location'].index(plan['camera'])]['left_camera_params']
    native_k = np.array([[c['fx'], 0., c['cx']], [0., c['fy'], c['cy']], [0., 0., 1.]])
    meta = {r['frame']: r for r in plan['frames_metadata']}
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in plan['frames']:
        if time.perf_counter()-started > budget_s:
            raise TimeoutError('Reference preparation budget reached')
        consumed = []
        for kind in ('depth_maps', 'zed_depth_maps', 'segmentation_masks'):
            item = files[frame, kind]
            if sha(item['path']) != item['sha256']:
                raise ValueError(f'Changed reference {frame} {kind}')
            consumed.append(dict(kind=kind, sha256=item['sha256']))
        cres = load_depth(files[frame, 'depth_maps']['path'])
        zed_native = load_depth(files[frame, 'zed_depth_maps']['path'])
        zed = np.asarray(Image.fromarray(zed_native).resize((cres.shape[1], cres.shape[0]), Image.Resampling.NEAREST))
        with Image.open(files[frame, 'segmentation_masks']['path']) as image:
            mask = np.asarray(image.resize((cres.shape[1], cres.shape[0]), Image.Resampling.NEAREST))
        target = np.isin(mask[..., 0], TARGET_CLASSES)  # Includes ID0 spatial masks.
        k = resize_intrinsics(native_k, (c['image_height'], c['image_width']), cres.shape)
        rx, ry = rays(k, cres.shape)
        labels, qs = [], []
        for q in camera_queries():
            label, record = interval_labels(cres, zed, target, rx, ry, q)
            labels.append(label); qs.append(record)
        path = output/f'reference_{frame:06d}.npz'
        np.savez(path, labels=np.stack(labels), target=target, semantic=mask[..., 0], K=k)
        rows.append(dict(**meta[frame], path=str(path.resolve()), sha256=sha(path),
                         rgb_sha256=files[frame, 'video_frames']['sha256'], reference_inputs=consumed,
                         queries=qs))
    counts = Counter(q['state'] for r in rows for q in r['queries'])
    manual = Counter(q['state'] for r in rows if r['annotation_type'] == 'HUMAN_ANNOTATED' for q in r['queries'])
    result = dict(status='COMPLETE', frames=len(rows), query_frames=6*len(rows), state_counts=dict(counts),
                  manual_state_counts=dict(manual), vacuous_negatives=sum(q['vacuous_negative'] for r in rows for q in r['queries']),
                  target_classes=list(TARGET_CLASSES), minimum_positive_pixels=MIN_SUPPORT,
                  agreement_rule='abs(CRES-ZED)<=max(.25m,.25*min(CRES,ZED)); estimated reference agreement only',
                  scope='visible annotated target-set intrusion, not full occupancy or body clearance',
                  depth_contract='expected optical-Z metres; public stereo disparity convention; no world-pose use',
                  resize='CRES native grid; nearest ZED/mask; half-pixel public K',
                  source_receipt_sha256=sha(source/'acquisition_receipt.json'),
                  seconds=time.perf_counter()-started, rows=rows)
    write(output/'reference_manifest.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


def confusion(label, predicted):
    p, n = label == POSITIVE, label == NEGATIVE
    return dict(tp=int((p & predicted).sum()), fn=int((p & ~predicted).sum()),
                fp=int((n & predicted).sum()), tn=int((n & ~predicted).sum()))


def ratios(c):
    def divide(a, b): return a/b if b else None
    return dict(iou=divide(c['tp'], c['tp']+c['fp']+c['fn']),
                recall=divide(c['tp'], c['tp']+c['fn']),
                fpr=divide(c['fp'], c['fp']+c['tn']))


def summarize_records(rr):
    c = {k: sum(r[k] for r in rr) for k in ('tp', 'fn', 'fp', 'tn')}
    # Empty target angular domains force an oracle-negative for every model.
    # Keep their reference state, but exclude them from discrimination scores.
    scoreable = [r for r in rr if r['reference_state'] != 'UNKNOWN'
                 and not r.get('vacuous_negative', False)]
    qc = dict(tp=0, fn=0, fp=0, tn=0)
    for r in scoreable:
        key = ('tp' if r['oracle_region_prediction'] else 'fn') if r['reference_state'] == 'POSITIVE' else ('fp' if r['oracle_region_prediction'] else 'tn')
        qc[key] += 1
    manual = [r for r in rr if r['annotation_type'] == 'HUMAN_ANNOTATED']
    mc = {k: sum(r[k] for r in manual) for k in ('tp', 'fn', 'fp', 'tn')}
    return dict(frames=len({r['frame'] for r in rr}), query_frames=len(rr),
                pixel_confusion=c, pixel_metrics=ratios(c),
                manual_pixel_confusion=mc, manual_pixel_metrics=ratios(mc),
                excluded_vacuous_query_negatives=sum(r.get('vacuous_negative', False) for r in rr),
                scoreable_query_frames=len(scoreable), oracle_region_query_confusion=qc)


def evaluate(reference_dir, manifests, output):
    started = time.perf_counter()
    ref = json.loads((reference_dir/'reference_manifest.json').read_text('utf-8-sig'))
    by_frame = {r['frame']: r for r in ref['rows']}
    pred_by_frame = defaultdict(list)
    sources = []
    for path in manifests:
        m = json.loads(path.read_text('utf-8-sig'))
        sources.append(dict(path=str(path.resolve()), sha256=sha(path), status=m['status']))
        for row in m['rows']:
            pred_by_frame[row['frame']].append(row)
    records = []
    for frame in sorted(pred_by_frame):
        r = by_frame[frame]
        if sha(r['path']) != r['sha256']:
            raise ValueError('Changed reference cache')
        with np.load(r['path'], allow_pickle=False) as a:
            labels, target, k = a['labels'], a['target'], a['K']
        rx, ry = rays(k, target.shape)
        intervals = [ray_interval(rx, ry, q) for q in camera_queries()]
        for pred in pred_by_frame[frame]:
            path = pred.get('depth_path', pred.get('path'))
            digest = pred.get('depth_sha256', pred.get('sha256'))
            if sha(path) != digest or pred['rgb_sha256'] != r['rgb_sha256']:
                raise ValueError('Prediction/source identity differs')
            with np.load(path, allow_pickle=False) as a:
                z = a['depth']
                valid = a['valid_mask'] if 'valid_mask' in a else np.isfinite(z) & (z > 0)
            public_k = np.asarray(pred.get('public_K', pred.get('K')), dtype=float)
            if z.shape != target.shape or not np.allclose(public_k, k, rtol=1e-10, atol=1e-10):
                raise ValueError('Shared ray grid or public K differs')
            for i, (entry, exit, reachable) in enumerate(intervals):
                projected = valid & np.isfinite(z) & (z > 0) & reachable & (z >= entry-1e-12) & (z <= exit+1e-12)
                # Reference target region is an evaluator-only localization oracle.
                oracle_count = int((projected & target).sum())
                # Also preserve full observation readout; never score its false alerts
                # against incomplete target-set labels as though those were full truth.
                observed_count = int(projected.sum())
                c = confusion(labels[i], projected)
                records.append(dict(frame=frame, arm=pred.get('arm', 'VDA-metric-streaming'),
                                    query=camera_queries()[i]['name'], reference_state=r['queries'][i]['state'],
                                    vacuous_negative=r['queries'][i].get('vacuous_negative', False),
                                    annotation_type=r['annotation_type'], **c,
                                    oracle_region_prediction=oracle_count >= MIN_SUPPORT,
                                    oracle_region_support_pixels=oracle_count,
                                    observation_support_pixels=observed_count,
                                    unknown_reference_pixels=int((labels[i] == UNKNOWN).sum()),
                                    labeled_reference_pixels=sum(c.values())))
    summaries = {}
    for arm in sorted({r['arm'] for r in records}):
        rr = [r for r in records if r['arm'] == arm]
        summaries[arm] = summarize_records(rr)
    arm_frames = [{r['frame'] for r in records if r['arm'] == arm} for arm in summaries]
    common_frames = sorted(set.intersection(*arm_frames)) if arm_frames else []
    matched = {arm: summarize_records([r for r in records if r['arm'] == arm and r['frame'] in common_frames])
               for arm in summaries}
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='COMPLETE', reference_manifest_sha256=sha(reference_dir/'reference_manifest.json'),
                  prediction_sources=sources, summaries=summaries, rows=records,
                  matched_comparison=dict(frames=common_frames, summaries=matched),
                  metric_scope='intrusion pixels on paired-consistent annotated target domain; query confusion uses GT localization oracle',
                  event_metrics='NOT_EVALUABLE: no verified NEG-to-POS onset; partial DepthPro excludes near-positive suffix',
                  interpretation='Different frame sets are not matched arm superiority; no full RGB detection or real collision metric',
                  seconds=time.perf_counter()-started)
    write(output, result)
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('reference', 'evaluate'))
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--prediction-manifest', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path)
    p.add_argument('--budget-s', type=float, default=240)
    args = p.parse_args()
    if args.stage == 'reference':
        reference(args.repo.resolve(), args.reference.resolve(), args.budget_s)
    else:
        if not args.prediction_manifest or args.output is None:
            p.error('evaluate requires prediction manifests and output')
        evaluate(args.reference.resolve(), args.prediction_manifest, args.output.resolve())

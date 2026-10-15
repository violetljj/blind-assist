"""Independent native-label segmentation/point evaluator (validation by default).

Input JSONL: source_id, split, gt_path, prediction_path, native_class_ids,
supervision_scope. Prediction .npy is C,H,W logits or native-size H,W IDs.
Optional metadata JSONL joins source_id and supplies objects/source_group.
No physical role, safe-grasp, or instance-exclusive truth is inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def identifier(row):
    return str(row.get('source_id') or row.get('source_image_id') or row.get('sample_id') or '')


def load_mask(path):
    p = Path(path)
    if p.suffix == '.npy':
        a = np.load(p, allow_pickle=False)
    elif p.suffix == '.txt':
        a = np.loadtxt(p, dtype=np.int64)
    else:
        with Image.open(p) as image:
            a = np.asarray(image)
    if a.ndim != 2 or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError(f'GT must be a finite 2D integer native label map: {p}')
    return a.astype(np.int64)


def native_prediction(array, shape, class_ids, point_scores='softmax'):
    if array.ndim == 2:
        if array.shape != shape or not np.equal(array, np.floor(array)).all():
            raise ValueError('Discrete predictions must be integer and already at native GT resolution')
        return array.astype(np.int64), None
    if array.ndim != 3 or len(class_ids) != array.shape[0] or not np.isfinite(array).all():
        raise ValueError('Prediction must be finite C,H,W with one native class ID per channel')
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
    if tuple(array.shape[1:]) != tuple(shape):
        t = F.interpolate(t, size=shape, mode='bilinear', align_corners=False)
    logits = t[0].numpy()
    prediction = np.asarray(class_ids, dtype=np.int64)[logits.argmax(0)]
    if point_scores == 'softmax':
        logits = logits - logits.max(0, keepdims=True)
        logits = np.exp(logits)
        logits /= logits.sum(0, keepdims=True)
    return prediction, logits


def metrics_from_confusion(cm, foreground_ids):
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.divide(tp, union, out=np.full(len(tp), np.nan), where=union > 0)
    fg = iou[foreground_ids]
    return {'mIoU_foreground': float(np.nanmean(fg)) if np.isfinite(fg).any() else None,
            'mIoU_with_background': float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
            'pixel_accuracy': float(tp.sum()/cm.sum()) if cm.sum() else None,
            'per_class_IoU': {str(i): float(v) if np.isfinite(v) else None for i,v in enumerate(iou)},
            'per_class_GT_pixels': cm.sum(1).tolist(), 'per_class_predicted_pixels': cm.sum(0).tolist()}


def score_arrays(gt, prediction, scores, class_ids, ignore_id=255, small_fraction=.01):
    n = max(class_ids)+1
    valid = gt != ignore_id
    if not valid.any() or not np.isin(gt[valid], class_ids).all():
        raise ValueError('Empty valid GT or native label not included in class IDs')
    if not np.isin(prediction, class_ids).all():
        raise ValueError('Prediction has unknown native class ID')
    cm = np.bincount((gt[valid]*n+prediction[valid]).ravel(), minlength=n*n).reshape(n,n)
    points = []
    if scores is not None:
        for c in class_ids:
            area = int((gt == c).sum())
            if c == 0 or area == 0:
                continue
            score = scores[class_ids.index(c)].copy()
            # Do not use GT validity to choose a point. A point landing in an ignore
            # region is explicitly not evaluable rather than selected elsewhere.
            y,x = np.unravel_index(int(score.argmax()), score.shape)
            actual = int(gt[y,x])
            evaluable = actual != ignore_id
            points.append({'class_id': c, 'point_xy': [int(x), int(y)],
                           'native_label_at_point': actual,
                           'hit': bool(actual == c) if evaluable else None,
                           'native_label_mixup': bool(actual not in [0,c]) if evaluable else None,
                           'target_pixels': area, 'target_fraction': area/int(valid.sum()),
                           'small_region': area/int(valid.sum()) <= small_fraction})
    pairs = [{'class_ids': [a['class_id'], b['class_id']],
              'joint_success': bool(a['hit'] and b['hit']) if a['hit'] is not None and b['hit'] is not None else None}
             for a,b in itertools.combinations(points, 2)]
    return {'confusion': cm.tolist(), 'points': points, 'pairs': pairs,
            'valid_pixels': int(valid.sum()), 'ignored_pixels': int((~valid).sum())}


def aggregate(rows, foreground_ids):
    if not rows:
        return {'images': 0}
    cm = np.sum([r['confusion'] for r in rows], axis=0)
    points = [p for r in rows for p in r['points'] if p['hit'] is not None]
    pairs = [p for r in rows for p in r['pairs'] if p['joint_success'] is not None]
    by_class = {}
    for c in foreground_ids:
        ps = [p for p in points if p['class_id'] == c]
        by_class[str(c)] = {'positive_image_queries': len(ps), 'hits': sum(p['hit'] for p in ps),
                            'point_hit_rate': float(np.mean([p['hit'] for p in ps])) if ps else None}
    return {'images': len(rows), **metrics_from_confusion(cm, foreground_ids),
            'confusion_GT_rows_prediction_columns': cm.tolist(),
            'point_queries': len(points), 'point_hits': sum(p['hit'] for p in points),
            'point_hit_rate': float(np.mean([p['hit'] for p in points])) if points else None,
            'native_label_mixups': sum(p['native_label_mixup'] for p in points),
            'native_label_mixup_rate': float(np.mean([p['native_label_mixup'] for p in points])) if points else None,
            'same_image_positive_class_pairs': len(pairs),
            'joint_pair_successes': sum(p['joint_success'] for p in pairs),
            'joint_pair_success_rate': float(np.mean([p['joint_success'] for p in pairs])) if pairs else None,
            'per_class_point': by_class}


def bootstrap(rows, foreground_ids, draws=1000, seed=20261015, compare=None):
    """Cluster resample entire source groups; no invented physical-instance IDs."""
    groups = defaultdict(list)
    for r in rows:
        groups[r['bootstrap_group']].append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    metric_names = ['mIoU_foreground', 'point_hit_rate', 'joint_pair_success_rate']
    values = {k: [] for k in metric_names}
    delta = {k: [] for k in metric_names}
    comparison = {r['source_id']: r for r in compare} if compare is not None else None
    if comparison is not None and set(comparison) != {r['source_id'] for r in rows}:
        raise ValueError('Paired bootstrap requires identical source image IDs')
    for _ in range(draws):
        chosen = [r for i in rng.integers(0, len(keys), len(keys)) for r in groups[keys[i]]]
        stats = aggregate(chosen, foreground_ids)
        ref = aggregate([comparison[r['source_id']] for r in chosen], foreground_ids) if comparison is not None else None
        for k in metric_names:
            if stats.get(k) is not None:
                values[k].append(stats[k])
                if ref is not None and ref.get(k) is not None:
                    delta[k].append(stats[k]-ref[k])
    interval = lambda d: {k: {'CI95_percentile': np.percentile(v,[2.5,97.5]).tolist(), 'valid_draws':len(v)} if v else None for k,v in d.items()}
    return {'draws':draws, 'seed':seed, 'groups':len(keys),
            'grouping': 'source_group_when_provided_else_image_ID; not known physical instance grouping',
            'image_ID_fallback_count':sum(r['source_group_known'] is False for r in rows),
            'intervals':interval(values), 'paired_difference_vs_reference':interval(delta) if compare is not None else None}


def strata(rows, foreground_ids):
    partitions = defaultdict(list)
    for r in rows:
        partitions['supervision_scope/'+r['supervision_scope']].append(r)
        partitions['image_source/'+r['image_source']].append(r)
        count = r.get('annotated_object_count')
        partitions['annotated_object_count/'+('unknown' if count is None else 'multiple' if count>1 else 'single' if count==1 else 'zero')].append(r)
        partitions['positive_native_classes/'+('multiple' if len(r['points'])>1 else 'single_or_none')].append(r)
    result = {k:aggregate(v,foreground_ids) for k,v in sorted(partitions.items())}
    for small in [True,False]:
        ps = [p for r in rows for p in r['points'] if p['small_region']==small and p['hit'] is not None]
        result['point_target_area/'+('small' if small else 'not_small')] = {
            'queries':len(ps), 'hits':sum(p['hit'] for p in ps),
            'hit_rate':float(np.mean([p['hit'] for p in ps])) if ps else None,
            'scope':'native semantic class union, not instance-exclusive parts'}
    return result


def evaluate(args):
    manifest = Path(args.predictions).resolve()
    inputs = read_rows(manifest)
    if not inputs:
        raise ValueError('Empty prediction manifest')
    metadata = {identifier(r):r for r in read_rows(args.metadata)} if args.metadata else {}
    # Validate every split before loading any GT file, including mixed manifests.
    for r in inputs:
        if r.get('split') not in ['val','validation'] and not (r.get('split')=='test' and args.allow_test):
            raise ValueError('Only val/validation permitted by default; test requires explicit --allow-test after authorization')
    results, seen = [], set()
    expected_classes = None
    for r in inputs:
        sid = identifier(r)
        if not sid or sid in seen:
            raise ValueError('Missing or repeated source ID')
        seen.add(sid)
        meta = metadata.get(sid,{})
        ids = list(r.get('native_class_ids',range(10)))
        if len(set(ids)) != len(ids) or sorted(ids) != list(range(max(ids)+1)):
            raise ValueError('Native class IDs must cover contiguous 0..C-1 once')
        if expected_classes is not None and ids != expected_classes:
            raise ValueError('Mixed class mappings')
        expected_classes = ids
        scope = r.get('supervision_scope')
        if scope not in ['full_image','oracle_object']:
            raise ValueError('Declare supervision_scope full_image or oracle_object')
        gt_path = r.get('gt_path') or meta.get('mask') or meta.get('mask_path')
        if not gt_path:
            raise ValueError('Missing GT path')
        gt = load_mask(gt_path)
        pp = Path(r['prediction_path'])
        pp = pp if pp.is_absolute() else manifest.parent/pp
        pred,scores = native_prediction(np.load(pp,allow_pickle=False), gt.shape, ids, args.point_scores)
        scored = score_arrays(gt,pred,scores,ids,args.ignore_id,args.small_fraction)
        group = r.get('source_group') or meta.get('source_group')
        objects = meta.get('objects')
        results.append({'source_id':sid,'split':r['split'],'supervision_scope':scope,
            'image_source':str(r.get('image_source') or meta.get('image_source') or meta.get('source') or 'unknown'),
            'annotated_object_count':len(objects) if isinstance(objects,list) else r.get('instance_count'),
            'bootstrap_group':str(group) if group is not None else 'image:'+sid,
            'source_group_known':group is not None, **scored})
    scopes = {r['supervision_scope'] for r in results}
    if len(scopes)>1:
        raise ValueError('Do not aggregate oracle-object and full-image predictions in one result')
    fg=[c for c in expected_classes if c != 0]
    compare = read_rows(args.compare_per_image) if args.compare_per_image else None
    summary={'input_manifest':str(manifest),'input_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
             'metric_scope':'native semantic labels; union of all instances per class; not physical role or safe-grasp truth',
             'point_score_definition':args.point_scores+'_per_class_spatial_argmax_no_GT_conditioning',
             'logit_resize':'bilinear float32 align_corners=False to native mask',
             'small_region_fraction_of_valid_image_pixels':args.small_fraction,
             'overall':aggregate(results,fg),'strata':strata(results,fg),
             'bootstrap':bootstrap(results,fg,args.bootstrap,args.seed,compare)}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'per_image.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary['overall'],ensure_ascii=False))


def selftest():
    ids=[0,1,2]
    gt=np.array([[1,1],[2,0]])
    logits=np.array([[[0,0],[0,9]],[[9,8],[0,0]],[[0,0],[9,0]]],dtype=np.float32)
    pred,score=native_prediction(logits,gt.shape,ids)
    perfect=score_arrays(gt,pred,score,ids)
    assert perfect['confusion']==[[1,0,0],[0,2,0],[0,0,1]]
    assert all(p['hit'] for p in perfect['points']) and perfect['pairs'][0]['joint_success']
    swapped=logits[[0,2,1]]
    pred,score=native_prediction(swapped,gt.shape,ids)
    wrong=score_arrays(gt,pred,score,ids)
    assert all(p['native_label_mixup'] for p in wrong['points'])
    assert not any(p['hit'] for p in wrong['points'])
    assert metrics_from_confusion(wrong['confusion'],[1,2])['mIoU_foreground']==0
    pred,score=native_prediction(logits,(4,4),ids)
    assert pred.shape==(4,4) and np.allclose(score.sum(0),1)
    a={'source_id':'a','bootstrap_group':'a','source_group_known':False,**perfect}
    b={'source_id':'b','bootstrap_group':'b','source_group_known':False,**perfect}
    boot=bootstrap([a,b],[1,2],20,7,[a,b])
    assert boot['intervals']['mIoU_foreground']['CI95_percentile']==[1,1]
    assert boot['paired_difference_vs_reference']['point_hit_rate']['CI95_percentile']==[0,0]
    print('PASS: native confusion, perfect and swapped points/pairs, resizing, image bootstrap and paired zero difference')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('selftest')
    s=sub.add_parser('score')
    s.add_argument('--predictions',required=True);s.add_argument('--metadata')
    s.add_argument('--output',required=True);s.add_argument('--allow-test',action='store_true')
    s.add_argument('--point-scores',choices=['softmax','raw_logits'],default='softmax')
    s.add_argument('--ignore-id',type=int,default=255);s.add_argument('--small-fraction',type=float,default=.01)
    s.add_argument('--bootstrap',type=int,default=1000);s.add_argument('--seed',type=int,default=20261015)
    s.add_argument('--compare-per-image')
    args=p.parse_args()
    if args.command=='selftest':selftest()
    else:evaluate(args)


if __name__=='__main__':
    main()

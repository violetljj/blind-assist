#!/usr/bin/env python3
"""Joint-model RGB export and separately invoked native-mask scoring.

``export`` calls the trainer's SAME infer_points function used for validation
checkpoint selection. It sees RGB plus path metadata, never GT pixels. Only a
native uint8 all-ten-class prediction and fixed query points are persisted, not
37-channel floating maps. ``score`` has no Torch/model/trainer import and reads
GT only after immutable export. Both commands stream atomic per-image chunks.
Run CPU synthetic checks on the worker: ``python full_joint_evaluate.py selftest``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

SHARED = (1, 2, 5, 7, 8, 9)
UMD_TO_IIT = {0: 0, 1: 5, 2: 2, 3: 255, 4: 1, 5: 7, 6: 8, 7: 9}
COUNT_FIELDS = ('queries', 'hits', 'evaluable', 'unknown', 'small_queries',
                'small_hits', 'small_evaluable', 'small_unknown')
METHODS = ('raw', 'stable', 'interior', 'quality')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def index_rows(path):
    result = {}
    for row in read_rows(path):
        sid = str(row.get('source_id', ''))
        if not sid or sid in result:
            raise ValueError(f'Missing or repeated source_id in {path}: {sid!r}')
        result[sid] = row
    if not result:
        raise ValueError(f'Empty manifest: {path}')
    return result


def relocate(value, data_root=None):
    path = Path(value)
    if path.is_file():
        return path.resolve()
    normalized = str(value).replace('\\', '/')
    if data_root and '/data/' in normalized:
        candidate = Path(data_root) / 'data' / normalized.rsplit('/data/', 1)[1]
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Missing declared input, no guessed file matching: {value}')


def chunks_path(output, sid):
    return Path(output) / 'chunks' / (hashlib.sha256(sid.encode()).hexdigest() + '.json')


class Run:
    """Exclusive lock; stale locks require explicit owner inspection, not guessing."""
    def __init__(self, output, config, resume):
        self.output, self.config = Path(output), config
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = self.output / 'owner.lock'
        self.token = uuid.uuid4().hex
        self.start = time.monotonic()
        target = self.output / 'config.json'
        if target.exists():
            if not resume or json.loads(target.read_text(encoding='utf-8')) != config:
                raise ValueError('Existing output requires --resume and identical frozen inputs/source')
        else:
            write(target, config)

    def __enter__(self):
        fd = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(dict(pid=os.getpid(), started_utc=utc(), token=self.token,
                           argv=sys.argv, config_sha256=sha(self.output / 'config.json')), stream)
        (self.output / 'chunks').mkdir(exist_ok=True)
        return self

    def progress(self, completed, total, stage):
        write(self.output / 'progress.json', dict(stage=stage, completed=completed,
              total=total, wall_seconds=time.monotonic() - self.start, updated_utc=utc()))

    def __exit__(self, kind, error, trace):
        if error is not None:
            failure = dict(status='FAILED', error=repr(error), traceback=''.join(
                traceback.format_exception(kind, error, trace)), failed_utc=utc(),
                wall_seconds=time.monotonic() - self.start)
            write(self.output / ('failure_' + self.token + '.json'), failure)
            write(self.output / 'terminal.json', failure)
        if self.lock.exists() and json.loads(self.lock.read_text()).get('token') == self.token:
            self.lock.unlink()


def completed(output, config):
    path = Path(output) / 'terminal.json'
    if not path.exists():
        return False
    terminal = json.loads(path.read_text(encoding='utf-8'))
    if terminal.get('status') != 'COMPLETE':
        return False
    for name, digest in terminal.get('output_sha256', {}).items():
        if sha(Path(output) / name) != digest:
            raise ValueError(f'Completed output changed: {name}')
    if json.loads((Path(output) / 'config.json').read_text()) != config:
        raise ValueError('Completed output input/source seal differs')
    return True


def resources():
    result = dict(process_cpu_seconds=time.process_time())
    try:
        import psutil
        info = psutil.Process().memory_info()
        result.update(resident_bytes=int(info.rss),
                      peak_resident_bytes=int(info.peak_wset) if hasattr(info, 'peak_wset') else None)
    except ImportError:
        result['resident_bytes'] = None
    return result


def extra_points(native_logits, prediction):
    """GT-free original NumPy float32 softmax and fixed 4-connected interior."""
    import numpy as np
    from scipy import ndimage
    value = native_logits - native_logits.max(0, keepdims=True)
    value = np.exp(value)
    value /= value.sum(0, keepdims=True)
    raw, interior = [], []
    for cls in range(1, 10):
        y, x = np.unravel_index(int(value[cls].argmax()), prediction.shape)
        raw.append([int(x), int(y)])
        labels, _ = ndimage.label(prediction == cls)  # fixed 4-connectivity
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        if sizes.max() == 0:
            interior.append(None)
            continue
        selected = labels == int(sizes.argmax())  # first component on area ties
        distance = ndimage.distance_transform_edt(np.pad(selected, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(int(distance.argmax()), prediction.shape)
        interior.append([int(x), int(y)])
    return dict(raw=raw, interior=interior)


def export(args):
    # All configuration/COMPLETE checks happen before any model or CUDA import.
    source = Path(__file__).resolve()
    checkpoint = Path(args.checkpoint)
    expected_name = dict(point='best.pt', budget='best_budget.pt', miou='best_miou.pt')[args.selection]
    if checkpoint.name != expected_name:
        raise ValueError(f'--selection {args.selection} requires immutable {expected_name}')
    training_terminal = checkpoint.parent / 'terminal.json'
    if json.loads(training_terminal.read_text(encoding='utf-8')).get('status') != 'COMPLETE':
        raise ValueError('Finish source training before exporting validation-selected checkpoints')
    config = dict(command='export', source_sha256=sha(source),
                  trainer_source_sha256=sha(source.with_name('full_joint_train.py')),
                  model_source_sha256=sha(source.with_name('full_joint_models.py')),
                  checkpoint_sha256=sha(args.checkpoint), records_sha256=sha(args.records),
                  training_terminal_sha256=sha(training_terminal),
                  dataset=args.dataset, selection=args.selection, batch=args.batch,
                  device=args.device, cpu_threads=args.cpu_threads,
                  rgb_only=True, retained_class_ids=list(range(10)), gt_pixels_read=False,
                  persisted_float_maps=False, selectors='trainer.infer_points + fixed raw/interior')
    out = Path(args.output)
    if args.resume and completed(out, config):
        print(json.dumps(dict(status='ALREADY_COMPLETE', output=str(out))), flush=True)
        return
    observations = index_rows(args.records)
    if args.dataset == 'iit':
        observations = {sid: row for sid, row in observations.items()
                        if row.get('split') in ('val', 'validation')}
    if not observations:
        raise ValueError('No validation observations; test/export split not guessed')
    for row in observations.values():
        if args.dataset == 'umd' and row.get('manual_annotation_status') != 'AUTHOR_SELECTED_MANUAL':
            raise ValueError('UMD export requires author-selected manual observations')
        if not (row.get('image') or row.get('image_path')):
            raise ValueError('RGB observation path missing')
        if not (row.get('mask') or row.get('mask_path') or row.get('gt_path')):
            raise ValueError('GT path metadata missing; never decode it in export')
    with Run(out, config, args.resume) as run:
        import numpy as np
        from PIL import Image
        import torch
        import full_joint_train as trainer
        torch.set_num_threads(args.cpu_threads)
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if saved.get('config', {}).get('source_sha256') != config['trainer_source_sha256']:
            raise ValueError('Checkpoint trainer-source SHA differs')
        if saved['config'].get('model_source_sha256') != config['model_source_sha256']:
            raise ValueError('Checkpoint model-source SHA differs')
        if args.dataset == 'iit' and saved['config'].get('records_sha256') != config['records_sha256']:
            raise ValueError('IIT export requires the same frozen train/cleanval manifest used for selection')
        # The caller chooses an immutable validation-selected checkpoint. Record
        # both its actual checkpoint cursor and requested selection family.
        model = trainer.make_model(saved['config'], pretrained=False)
        model.load_state_dict(saved['model'], strict=True)
        device = torch.device(args.device)
        model.to(device).eval()
        size = int(saved['config'].get('size', saved['config'].get('input_size', 448)))
        manifest = out / 'predictions.jsonl'
        temporary_manifest = manifest.with_suffix('.jsonl.tmp')
        source_rows = list(observations.items())
        timings, forwarded, reused = Counter(), 0, 0
        with temporary_manifest.open('w', encoding='utf-8') as stream, torch.inference_mode():
            for base in range(0, len(source_rows), args.batch):
                pending = []
                for sid, row in source_rows[base:base + args.batch]:
                    chunk = chunks_path(out, sid)
                    if chunk.exists():
                        item = json.loads(chunk.read_text())
                        if item['source_id'] != sid or sha(item['prediction_path']) != item['prediction_sha256']:
                            raise ValueError(f'Corrupt reusable export chunk: {sid}')
                        if sha(relocate(row.get('image') or row['image_path'], args.data_root)) != item['image_sha256']:
                            raise ValueError(f'RGB input changed since export: {sid}')
                        reused += 1
                        continue
                    tick = time.monotonic()
                    path = relocate(row.get('image') or row['image_path'], args.data_root)
                    raw = path.read_bytes()
                    image_digest = hashlib.sha256(raw).hexdigest()
                    if row.get('image_sha256') and image_digest != row['image_sha256']:
                        raise ValueError(f'RGB differs from source manifest hash: {sid}')
                    with Image.open(io.BytesIO(raw)) as image:
                        image = image.convert('RGB')
                        shape = (image.height, image.width)
                        array = np.array(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
                    x = torch.from_numpy(array).permute(2, 0, 1) / 255.
                    x = (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor([.229, .224, .225])[:, None, None]
                    pending.append((sid, row, x, shape, image_digest))
                    timings['rgb_decode_seconds'] += time.monotonic() - tick
                if pending:
                    tick = time.monotonic()
                    xs = torch.stack([entry[2] for entry in pending]).to(device)
                    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                        batch = model(xs)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    timings['model_forward_seconds'] += time.monotonic() - tick
                    for index, (sid, row, _, shape, image_sha) in enumerate(pending):
                        tick = time.monotonic()
                        outputs = {key: tensor[index:index + 1] if tensor is not None else None
                                   for key, tensor in batch.items()}
                        points, native = trainer.infer_points(outputs, shape)
                        if native.shape[1] != 10 or not torch.isfinite(native).all():
                            raise ValueError('Expected finite native float32 all-ten-class logits')
                        # Argmax is evaluated before persistence; never float16.
                        native_array = native[0].float().cpu().numpy()
                        prediction = native_array.argmax(0).astype(np.uint8)
                        selected = {key: value[0].cpu().tolist() for key, value in points.items()}
                        selected.update(extra_points(native_array, prediction))
                        if 'stable' not in selected or not set(selected) <= set(METHODS):
                            raise ValueError('Unexpected shared selector interface')
                        target = chunks_path(out, sid).with_suffix('.npz')
                        with target.with_suffix('.npz.tmp').open('wb') as handle:
                            np.savez_compressed(handle, prediction=prediction)
                        os.replace(target.with_suffix('.npz.tmp'), target)
                        item = dict(source_id=sid, dataset=args.dataset, split='val' if args.dataset == 'iit' else 'external_transfer',
                            prediction_path=str(target.resolve()), prediction_sha256=sha(target),
                            original_rgb_shape=list(shape), image_sha256=image_sha,
                            gt_path=str(row.get('mask') or row.get('mask_path') or row['gt_path']),
                            gt_label_space='umd_native_0_7' if args.dataset == 'umd' else 'iit_native_0_9',
                            source_group=row.get('source_group'), image_source=row.get('image_source'),
                            annotated_object_count=len(row['objects']) if isinstance(row.get('objects'), list) else None,
                            points=selected, native_class_ids=list(range(10)), gt_pixels_read_during_export=False)
                        write(chunks_path(out, sid), item)
                        forwarded += 1
                        timings['native_readout_and_persistence_seconds'] += time.monotonic() - tick
                    del batch, xs
                for sid, _ in source_rows[base:base + args.batch]:
                    item = json.loads(chunks_path(out, sid).read_text())
                    stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                run.progress(min(base + args.batch, len(source_rows)), len(source_rows), 'rgb_export')
        os.replace(temporary_manifest, manifest)
        receipt = dict(status='COMPLETE', images=len(observations), forwarded_this_attempt=forwarded,
            reused_images=reused, arm=saved['config']['arm'], checkpoint_epoch=saved.get('epoch_index'),
            checkpoint_updates=saved.get('updates'), native_channels_retained=10,
            gt_pixels_read=False, timing_seconds=dict(timings), wall_seconds=time.monotonic() - run.start,
            resource_observation=resources(),
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None,
            output_sha256={'predictions.jsonl': sha(manifest)})
        write(out / 'terminal.json', receipt)
        print(json.dumps(receipt), flush=True)


def load_gt(row, data_root):
    import numpy as np
    path = relocate(row['gt_path'], data_root)
    raw_bytes = path.read_bytes()
    if row['dataset'] == 'umd':
        from scipy.io import loadmat
        raw = np.asarray(loadmat(io.BytesIO(raw_bytes), variable_names=['gt_label'])['gt_label'])
        if raw.ndim != 2 or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all() or not np.isin(raw, (-1, *UMD_TO_IIT)).all():
            raise ValueError('Unexpected native UMD GT, no silent remapping')
        gt = np.full(raw.shape, 255, dtype=np.uint8)
        for old, new in UMD_TO_IIT.items():
            gt[raw == old] = new
        reasons = dict(native_void_minus1=raw == -1, unsupported_scoop3=raw == 3)
    else:
        from PIL import Image
        with Image.open(io.BytesIO(raw_bytes)) as image:
            gt = np.array(image)
        if gt.ndim != 2 or not np.isin(gt, range(10)).all():
            raise ValueError('IIT GT must have original native IDs0..9')
        reasons = {}
    return gt, reasons, hashlib.sha256(raw_bytes).hexdigest()


def point_record(point, gt, prediction, cls, cm, valid_pixels, reasons):
    import numpy as np
    area = int((gt == cls).sum())
    pred_area, overlap = int(cm[:, cls].sum()), int(cm[cls, cls])
    union = area + pred_area - overlap
    support = ('no_predicted_class' if pred_area == 0 else 'zero_overlap' if overlap == 0
               else 'overlap_iou_below_half' if overlap / union < .5 else 'overlap_iou_at_least_half')
    result = dict(class_id=cls, point_xy=point, target_pixels=area,
                  target_fraction=area / valid_pixels, small_region=area / valid_pixels <= .01,
                  segmentation_support=support, native_label_at_point=None,
                  predicted_class_at_point=None, ignore_reason=None)
    if point is None:
        return dict(result, hit=False, category='empty_candidate')
    if len(point) != 2 or any(int(x) != x for x in point):
        raise ValueError('Expected integer xy point or explicit empty candidate')
    x, y = map(int, point)
    if not (0 <= x < gt.shape[1] and 0 <= y < gt.shape[0]):
        raise ValueError('Point outside native RGB; never silently clamp in scorer')
    actual = int(gt[y, x])
    result.update(native_label_at_point=actual, predicted_class_at_point=int(prediction[y, x]))
    if actual == 255:
        found = [name for name, mask in reasons.items() if mask[y, x]]
        if len(found) != 1:
            raise AssertionError('An unknown UMD point must have one ignore reason')
        return dict(result, hit=None, category='unknown_ignore', ignore_reason=found[0])
    return dict(result, hit=actual == cls, category='hit' if actual == cls else 'background' if actual == 0 else 'other_foreground')


def counts_for(points, methods):
    import numpy as np
    count = np.zeros((len(methods), 10, len(COUNT_FIELDS)), dtype=np.int64)
    for m, method in enumerate(methods):
        for p in points[method]:
            values = (1, p['hit'] is True, p['hit'] is not None, p['hit'] is None)
            count[m, p['class_id'], :4] += values
            if p['small_region']:
                count[m, p['class_id'], 4:] += values
    return count


def point_summary(count):
    value = {field: int(number) for field, number in zip(COUNT_FIELDS, count)}
    value['hit_rate_evaluable'] = value['hits'] / value['evaluable'] if value['evaluable'] else None
    value['hits_over_all_positive_queries'] = value['hits'] / value['queries'] if value['queries'] else None
    value['small_hit_rate_evaluable'] = value['small_hits'] / value['small_evaluable'] if value['small_evaluable'] else None
    value['small_hits_over_all_positive_queries'] = value['small_hits'] / value['small_queries'] if value['small_queries'] else None
    return value


def segmentation_summary(cm, classes):
    import numpy as np
    union = cm.sum(0) + cm.sum(1) - cm.diagonal()
    iou = np.divide(cm.diagonal(), union, out=np.full(10, np.nan), where=union > 0)
    selected = iou[list(classes)]
    return dict(mIoU=float(np.nanmean(selected)) if np.isfinite(selected).any() else None,
                per_class_iou={str(c): float(iou[c]) if np.isfinite(iou[c]) else None for c in range(10)},
                confusion_GT_rows_prediction_columns=cm.tolist(),
                class_GT_pixels=cm.sum(1).tolist(), class_predicted_pixels=cm.sum(0).tolist())


def bootstrap(group_cm, group_counts, classes, methods, draws, seed, reference=None):
    """Compact group sufficient statistics, no image/point expansion per draw."""
    import numpy as np
    keys = sorted(group_cm)
    matrices = np.stack([group_cm[k] for k in keys])
    values = np.stack([group_counts[k] for k in keys])[:, :, list(classes)].sum(2)
    if reference is not None:
        ref_cm = np.stack([reference[0][k] for k in keys])
        ref_count = np.stack([reference[1][k] for k in keys])[:, :, list(classes)].sum(2)
    rng = np.random.default_rng(seed)
    distributions = {'mIoU': []}
    for method in methods:
        for metric in ('hit_rate_evaluable', 'hits_over_all_positive_queries', 'small_hits_over_all_positive_queries'):
            distributions[method + '/' + metric] = []
    differences = {name: [] for name in distributions} if reference is not None else None
    for _ in range(draws):
        indices = rng.integers(0, len(keys), len(keys))
        current_cm, current_count = matrices[indices].sum(0), values[indices].sum(0)
        current = {'mIoU': segmentation_summary(current_cm, classes)['mIoU']}
        for i, method in enumerate(methods):
            stats = point_summary(current_count[i])
            current.update({method + '/' + name: stats[name] for name in ('hit_rate_evaluable', 'hits_over_all_positive_queries', 'small_hits_over_all_positive_queries')})
        ref = None
        if reference is not None:
            ref = dict(mIoU=segmentation_summary(ref_cm[indices].sum(0), classes)['mIoU'])
            stats = point_summary(ref_count[indices].sum(0)[0])
            ref.update({method + '/' + name: stats[name] for method in methods for name in ('hit_rate_evaluable', 'hits_over_all_positive_queries', 'small_hits_over_all_positive_queries')})
        for name, number in current.items():
            if number is not None:
                distributions[name].append(number)
                if ref is not None and ref[name] is not None:
                    differences[name].append(number - ref[name])
    def intervals(data):
        return {name: dict(CI95_percentile=np.percentile(value, [2.5, 97.5]).tolist(), valid_draws=len(value)) if value else None for name, value in data.items()}
    return dict(draws=draws, seed=seed, groups=len(keys), intervals=intervals(distributions),
                difference_vs_reference=intervals(differences) if differences is not None else None)


def score(args):
    import numpy as np  # no Torch/model/trainer imports on this command path
    observations = index_rows(args.predictions)
    datasets = {row['dataset'] for row in observations.values()}
    if len(datasets) != 1 or not datasets <= {'iit', 'umd'}:
        raise ValueError('Exactly one declared data source required')
    dataset = next(iter(datasets))
    classes = SHARED if dataset == 'umd' else tuple(range(1, 10))
    methods = sorted(next(iter(observations.values()))['points'])
    if not methods or not set(methods) <= set(METHODS):
        raise ValueError('Unknown exported point policy')
    reference_rows = index_rows(args.reference) if args.reference else None
    if reference_rows is not None and set(reference_rows) != set(observations):
        raise ValueError('Reference must have exactly the same source IDs; no subset matching')
    config = dict(command='score', source_sha256=sha(__file__), manifest_sha256=sha(args.predictions),
                  reference_sha256=sha(args.reference) if args.reference else None,
                  dataset=dataset, foreground_ids=list(classes), methods=methods,
                  bootstrap=args.bootstrap, seed=args.seed, small_semantic_union_fraction=.01,
                  unknown_policy='ignore landing stays UNKNOWN; conservative denominator separately reported')
    out = Path(args.output)
    if args.resume and completed(out, config):
        print(json.dumps(dict(status='ALREADY_COMPLETE', output=str(out))), flush=True)
        return
    with Run(out, config, args.resume) as run:
        group_cm, group_counts, ref_cm, ref_counts = {}, {}, {}, {}
        ignore_pixels, ignore_points, categories, support, input_gt_sha = Counter(), Counter(), Counter(), Counter(), {}
        all_ignored = []
        total_cm = np.zeros((10, 10), dtype=np.int64)
        total_count = np.zeros((len(methods), 10, 8), dtype=np.int64)
        manifest, queries = out / 'per_image.jsonl', out / 'per_query.jsonl'
        with manifest.with_suffix('.jsonl.tmp').open('w', encoding='utf-8') as images_stream, queries.with_suffix('.jsonl.tmp').open('w', encoding='utf-8') as query_stream:
            for index, (sid, row) in enumerate(observations.items()):
                if row.get('native_class_ids') != list(range(10)) or row.get('gt_pixels_read_during_export') is not False or sorted(row['points']) != methods:
                    raise ValueError('Invalid observation-only all-ten-channel export contract')
                group = row.get('source_group') if dataset == 'umd' else 'rgb_sha256:' + row['image_sha256']
                if not group:
                    raise ValueError('UMD author tool/scene source group missing')
                chunk = chunks_path(out, sid)
                if chunk.exists():
                    result = json.loads(chunk.read_text())
                    if result['source_id'] != sid or result['prediction_sha256'] != row['prediction_sha256'] or result['bootstrap_group'] != group:
                        raise ValueError('Scored chunk source/group changed')
                    if sha(relocate(row['gt_path'], args.data_root)) != result['gt_sha256']:
                        raise ValueError('GT input changed since scoring')
                    if sha(row['prediction_path']) != row['prediction_sha256']:
                        raise ValueError('Native prediction changed since scoring')
                else:
                    gt, reasons, gt_sha = load_gt(row, args.data_root)
                    if sha(row['prediction_path']) != row['prediction_sha256']:
                        raise ValueError('Persisted native prediction changed')
                    with np.load(row['prediction_path'], allow_pickle=False) as arrays:
                        pred = arrays['prediction']
                    if gt.shape != pred.shape or not np.isin(pred, range(10)).all():
                        raise ValueError('Native GT/RGB/prediction geometry or class IDs differ')
                    valid = gt != 255
                    cm = np.bincount((gt[valid].astype(np.int64) * 10 + pred[valid]).ravel(), minlength=100).reshape(10, 10)
                    scored = {method: [point_record(row['points'][method][cls - 1], gt, pred, cls, cm, int(valid.sum()), reasons)
                                      for cls in range(1, 10) if (gt == cls).any()] for method in methods}
                    result = dict(source_id=sid, dataset=dataset, bootstrap_group=group,
                        gt_sha256=gt_sha, prediction_sha256=row['prediction_sha256'],
                        valid_pixels=int(valid.sum()), all_pixels_ignored=not valid.any(),
                        ignored_pixels_by_reason={name: int(mask.sum()) for name, mask in reasons.items()},
                        image_source=row.get('image_source'), annotated_object_count=row.get('annotated_object_count'),
                        confusion=cm.tolist(), points=scored)
                    write(chunk, result)
                cm = np.asarray(result['confusion'], dtype=np.int64)
                counts = counts_for(result['points'], methods)
                total_cm += cm
                total_count += counts
                group_cm.setdefault(group, np.zeros((10, 10), dtype=np.int64))[:] += cm
                group_counts.setdefault(group, np.zeros_like(total_count))[:] += counts
                ignore_pixels.update(result['ignored_pixels_by_reason'])
                input_gt_sha[sid] = result['gt_sha256']
                if result['all_pixels_ignored']:
                    all_ignored.append(sid)
                for method in methods:
                    for p in result['points'][method]:
                        if p['class_id'] in classes:
                            categories[method + '/' + p['category']] += 1
                            if p['ignore_reason']:
                                ignore_points[method + '/' + p['ignore_reason']] += 1
                            if p['hit'] is False:
                                support[method + '/' + p['segmentation_support']] += 1
                                if p['small_region']:
                                    support[method + '/small/' + p['segmentation_support']] += 1
                        query_stream.write(json.dumps(dict(source_id=sid, bootstrap_group=group, method=method, **p)) + '\n')
                images_stream.write(json.dumps(result) + '\n')
                if reference_rows is not None:
                    old = reference_rows[sid]
                    old_points = old['points'] if isinstance(old['points'], list) else old['points'][args.reference_method]
                    old_cm = np.asarray(old['confusion'], dtype=np.int64)
                    if old_cm.shape != (10, 10) or not np.array_equal(old_cm.sum(1), cm.sum(1)):
                        raise ValueError('Reference GT class/ignore denominators differ from this scored image')
                    ref_cm.setdefault(group, np.zeros((10, 10), dtype=np.int64))[:] += old_cm
                    ref_counts.setdefault(group, np.zeros((1, 10, 8), dtype=np.int64))[:] += counts_for({'reference': old_points}, ['reference'])
                if index % 25 == 0 or index + 1 == len(observations):
                    run.progress(index + 1, len(observations), 'native_gt_score')
        os.replace(manifest.with_suffix('.jsonl.tmp'), manifest)
        os.replace(queries.with_suffix('.jsonl.tmp'), queries)
        methods_summary = {method: dict(primary=point_summary(total_count[m, list(classes)].sum(0)),
            per_class={str(cls): point_summary(total_count[m, cls]) for cls in range(1, 10)}) for m, method in enumerate(methods)}
        summary = dict(status='COMPLETE', dataset=dataset, input_images=len(observations),
            bootstrap_groups=len(group_cm), primary_foreground_ids=list(classes),
            all_pixels_ignored_source_ids=all_ignored,
            segmentation=segmentation_summary(total_cm, classes),
            shared6_segmentation=segmentation_summary(total_cm, SHARED), methods=methods_summary,
            ignored_pixels_by_reason=dict(ignore_pixels), ignored_points_by_reason=dict(ignore_points),
            point_categories=dict(categories),
            failed_point_segmentation_support=dict(support),
            bootstrap=bootstrap(group_cm, group_counts, classes, methods, args.bootstrap, args.seed,
                                reference=(ref_cm, ref_counts) if reference_rows is not None else None),
            scope='Native semantic unions; not instance-role truth, existence/rejection or user safety',
            grouping='UMD author tool/scene; IIT exact RGB SHA (not proven independent scene)',
            predictions_policy='Full ten-class native float32 argmax before uint8 persistence; no GT masking or renormalization',
            selectors_read_GT=False, training_or_checkpoint_selection_on_UMD=False,
            resource_observation=resources(),
            wall_seconds=time.monotonic() - run.start)
        write(out / 'gt_input_sha256.json', input_gt_sha)
        write(out / 'summary.json', summary)
        write(out / 'terminal.json', dict(status='COMPLETE', images=len(observations),
            wall_seconds=time.monotonic() - run.start,
            output_sha256={name: sha(out / name) for name in ('summary.json', 'per_image.jsonl', 'per_query.jsonl', 'gt_input_sha256.json')}))
        print(json.dumps(dict(status='COMPLETE', dataset=dataset, images=len(observations),
                             groups=len(group_cm), mIoU=summary['segmentation']['mIoU'])), flush=True)


def selftest():
    import numpy as np
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[1:3, 1:3] = 5  # exactly1% small semantic union
    gt[18, 18] = 255
    pred = gt.copy()
    pred[pred == 255] = 3  # unused class kept; GT ignore cannot select points
    valid = gt != 255
    cm = np.bincount((gt[valid].astype(np.int64) * 10 + pred[valid]).ravel(), minlength=100).reshape(10, 10)
    reasons = {'native_void_minus1': gt == 255}
    hit = point_record([1, 1], gt, pred, 5, cm, int(valid.sum()), reasons)
    unknown = point_record([18, 18], gt, pred, 5, cm, int(valid.sum()), reasons)
    empty = point_record(None, gt, pred, 5, cm, int(valid.sum()), reasons)
    assert hit['hit'] is True and unknown['hit'] is None and empty['hit'] is False
    counts = counts_for({'stable': [hit, unknown, empty]}, ['stable'])
    assert counts[0, 5, :4].tolist() == [3, 1, 2, 1]
    stats = point_summary(counts[0, 5])
    assert stats['hit_rate_evaluable'] == .5 and stats['hits_over_all_positive_queries'] == 1 / 3
    # Distinguish valid-image area from total area;4/399 is above1%.
    assert not hit['small_region']
    assert segmentation_summary(cm, SHARED)['mIoU'] == 1.
    interval = bootstrap({'a': cm}, {'a': counts}, SHARED, ['stable'], 20, 20261015)
    assert interval['intervals']['stable/hit_rate_evaluable']['CI95_percentile'] == [.5, .5]
    np_logits = np.zeros((10, 20, 20), dtype=np.float32)
    np_logits[5, 1:3, 1:3] = 5
    policies = extra_points(np_logits, np_logits.argmax(0))
    assert policies['raw'][4] == [1, 1] and policies['interior'][4] == [1, 1]
    print(json.dumps(dict(status='PASS', synthetic_only=True, GPU_or_model_run=False,
        checks=['all10_argmax', 'unknown_not_failure', 'empty_candidate_miss',
                'valid_area_small_definition', 'cluster_bootstrap', 'fixed_raw_and_interior'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('export')
    sub.add_argument('--checkpoint', required=True)
    sub.add_argument('--records', required=True)
    sub.add_argument('--dataset', choices=('iit', 'umd'), required=True)
    sub.add_argument('--selection', choices=('point', 'miou', 'budget'), default='point',
                     help='Frozen checkpoint family chosen on IIT validation, never UMD')
    sub.add_argument('--device', default='cuda')
    sub.add_argument('--batch', type=int, default=4)
    sub.add_argument('--cpu-threads', type=int, default=1)
    sub = commands.add_parser('score')
    sub.add_argument('--predictions', required=True)
    sub.add_argument('--bootstrap', type=int, default=1000)
    sub.add_argument('--seed', type=int, default=20261015)
    sub.add_argument('--reference', help='Optional complete original per_image.jsonl for paired context')
    sub.add_argument('--reference-method', default='stable')
    commands.add_parser('selftest')
    for name in ('export', 'score'):
        sub = commands.choices[name]
        sub.add_argument('--output', required=True)
        sub.add_argument('--data-root', help='Legacy full baseline data root for declared-path relocation')
        sub.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(getattr(args, 'cpu_threads', 1))
    if args.command == 'export' and (args.batch < 1 or args.cpu_threads < 1):
        parser.error('Batch and CPU threads must be positive')
    if args.command == 'score' and args.bootstrap < 1:
        parser.error('Bootstrap draws must be positive')
    {'export': export, 'score': score, 'selftest': lambda _: selftest()}[args.command](args)


if __name__ == '__main__':
    main()

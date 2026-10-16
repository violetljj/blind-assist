#!/usr/bin/env python3
"""Matched-data IIT joint point experiments; run heavy commands on the worker.

The manifest is filtered before image access. No UMD/test labels are consumed.
All arms use the same deterministic image exposure/augmentation and point rule.
`precompute` caches train-mask geometry once; `profile` measures real updates;
`train` resumes only committed optimizer updates including RNG/sampler state.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import socket
import tempfile
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import ColorJitter

from full_joint_models import DinoJoint, JointConfig

SCHEMA = 'iit_joint_update_resume_v1'
CACHE_SCHEMA = 'iit_joint_geometry_v1'
ARMS = ('plain', 'plain_quality', 'detail', 'detail_quality', 'candidate_quality', 'joint')
CLASSES = ['background', 'contain', 'cut', 'display', 'engine', 'grasp', 'hit', 'pound', 'support', 'wrap-grasp']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def append(path, value):
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def save_checkpoint(path, value):
    temporary = Path(str(path) + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_records(path, check_sizes=True):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len({str(row['source_id']) for row in rows}) != len(rows):
        raise ValueError('Duplicate source_id in manifest')
    train = [row for row in rows if row['split'] == 'train']
    val = [row for row in rows if row['split'] == 'val']
    if check_sizes and (len(train), len(val)) != (4417, 1757):
        raise ValueError(f'Frozen IIT train4417/cleanval1757 required, got {len(train)}/{len(val)}')
    if not train or not val:
        raise ValueError('Official train and cleanval must both be nonempty')
    for row in train + val:
        if not Path(row['image']).is_file() or not Path(row['mask']).is_file():
            raise FileNotFoundError(row['source_id'])
    return train, val


def mask_array(path):
    suffix = Path(path).suffix.lower()
    if suffix in ('.txt', '.csv'):
        result = np.loadtxt(path, dtype=np.int64)
    elif suffix == '.npy':
        result = np.load(path, allow_pickle=False).astype(np.int64)
    else:
        with Image.open(path) as image:
            result = np.asarray(image).astype(np.int64)
    if result.ndim != 2 or not np.isin(result, np.arange(10)).all():
        raise ValueError(f'Expected native two-dimensional IDs0..9: {path}')
    return result


def resized_mask(row, size):
    result = mask_array(row['mask'])
    return np.asarray(Image.fromarray(result.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)).copy()


def geometry(mask, grid):
    """One byte own-class margin and two flip-specific nearest-positive grids.

    The nearest coordinates are derived from the full 448 mask, not a downsampled
    class mask, so a valid thin foreground class cannot vanish in target creation.
    Arrays store XY integer pixels, with 65535 denoting an absent foreground class.
    """
    from scipy.ndimage import distance_transform_edt
    height, width = mask.shape
    ys = np.minimum(((np.arange(grid) + .5) * height / grid).astype(np.int64), height - 1)
    xs = np.minimum(((np.arange(grid) + .5) * width / grid).astype(np.int64), width - 1)
    margin = np.zeros_like(mask, dtype=np.uint8)
    nearest = np.full((2, 9, 2, grid, grid), 65535, dtype=np.uint16)
    for cls in range(1, 10):
        foreground = mask == cls
        if not foreground.any():
            continue
        # Padding gives an explicit image-border boundary, including all-one masks.
        inside = distance_transform_edt(np.pad(foreground, 1))[1:-1, 1:-1]
        margin[foreground] = np.rint(np.minimum(inside[foreground], 8.) / 8. * 255).astype(np.uint8)
        indices = distance_transform_edt(~foreground, return_distances=False, return_indices=True)
        for flip in (0, 1):
            query_x = width - 1 - xs if flip else xs
            target_y = indices[0][ys[:, None], query_x[None, :]]
            target_x = indices[1][ys[:, None], query_x[None, :]]
            nearest[flip, cls - 1, 0] = width - 1 - target_x if flip else target_x
            nearest[flip, cls - 1, 1] = target_y
    return margin, nearest


def cache_key(source_id):
    return hashlib.sha256(str(source_id).encode('utf-8')).hexdigest()[:24] + '.npz'


def precompute(args):
    torch.set_num_threads(1)
    train, val = read_records(args.records)
    out = Path(args.output)
    begin, cpu_begin = time.perf_counter(), time.process_time()
    metadata = {'schema': CACHE_SCHEMA, 'records_sha256': sha(args.records), 'size': args.size,
                'grid': args.size // 4, 'train_rows': len(train), 'validation_rows_not_cached': len(val),
                'target_source': 'train_native_masks_only', 'source_sha256': sha(__file__)}
    if (out / 'index.json').exists():
        old = json.loads((out / 'index.json').read_text(encoding='utf-8'))
        if any(old.get(key) != metadata[key] for key in metadata):
            raise ValueError('Existing geometry cache input/schema changed')
    index = []
    for number, row in enumerate(train):
        target = out / cache_key(row['source_id'])
        mask_sha = sha(row['mask'])
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                if str(saved['mask_sha256']) != mask_sha or saved['mask'].shape != (args.size, args.size):
                    raise ValueError(f'Geometry cache mismatch {row["source_id"]}')
        else:
            mask = resized_mask(row, args.size)
            margin, nearest = geometry(mask, args.size // 4)
            temporary = Path(str(target) + '.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, mask=mask, margin=margin, nearest=nearest,
                                    mask_sha256=np.array(mask_sha))
            os.replace(temporary, target)
        index.append({'source_id': row['source_id'], 'file': target.name, 'sha256': sha(target),
                      'bytes': target.stat().st_size, 'mask_sha256': mask_sha})
        if (number + 1) % 100 == 0:
            write(out / 'progress.json', dict(status='RUNNING', completed=number + 1,
                  total=len(train), wall_seconds=time.perf_counter() - begin))
    write(out / 'index.json', dict(metadata, files=index))
    result = dict(status='COMPLETE', **metadata, cached_files=len(index), bytes=sum(row['bytes'] for row in index),
                  wall_seconds=time.perf_counter() - begin, cpu_seconds=time.process_time() - cpu_begin)
    write(out / 'terminal.json', result)
    print(json.dumps(result), flush=True)


class IITJoint(Dataset):
    def __init__(self, rows, size, seed, epoch=0, cache=None):
        self.rows, self.size, self.seed, self.epoch = rows, size, seed, epoch
        self.cache = Path(cache) if cache else None
        self.jitter = ColorJitter(.15, .15, .1, .02)
        if any(row['split'] not in ('train', 'val') for row in rows):
            raise ValueError('This training harness never opens test image/mask pixels')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['image']) as image:
            image = image.convert('RGB')
        if self.cache is not None:
            with np.load(self.cache / cache_key(row['source_id']), allow_pickle=False) as stored:
                mask, margin = stored['mask'].copy(), stored['margin'].copy()
                # Independent per-image augmentation makes mid-epoch restart exact.
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(self.seed + self.epoch * len(self.rows) + index)
                    flip = int(torch.rand(()).item() < .5)
                    if flip:
                        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        mask, margin = mask[:, ::-1].copy(), margin[:, ::-1].copy()
                    image = self.jitter(image)
                nearest = stored['nearest'][flip].copy()
        else:
            mask = mask_array(row['mask'])
            if (image.height, image.width) != mask.shape:
                raise ValueError(f'RGB/mask dimensions differ: {row["source_id"]}')
            margin, nearest = None, None
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.
        x = (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor([.229, .224, .225])[:, None, None]
        return x, torch.from_numpy(mask.copy()).long(), margin, nearest, row


def collate(batch):
    xs, masks, margins, nearest, rows = zip(*batch)
    return (torch.stack(xs), list(masks), torch.from_numpy(np.stack(margins)) if margins[0] is not None else None,
            torch.from_numpy(np.stack(nearest).astype(np.int32)) if nearest[0] is not None else None, list(rows))


class EpochSampler(Sampler):
    def __init__(self, length, seed, epoch, start=0):
        self.order = torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()[start:]

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


def segmentation_loss(logits, masks):
    """Per-image Dice is invariant to physical microbatch partitioning."""
    z = F.interpolate(logits.float(), size=masks.shape[-2:], mode='bilinear', align_corners=False)
    ce = F.cross_entropy(z, masks, reduction='none').mean((1, 2))
    probability = z.softmax(1)
    target = F.one_hot(masks, 10).permute(0, 3, 1, 2).float()
    mass = target.sum((2, 3))[:, 1:]
    dice = (2 * (probability * target).sum((2, 3))[:, 1:] + 1) / (probability.sum((2, 3))[:, 1:] + mass + 1)
    present = mass > 0
    loss = ((1 - dice) * present).sum(1) / present.sum(1).clamp_min(1)
    return (ce + loss).mean()


def base_coordinates(height, width, device):
    ys = (torch.arange(height, device=device, dtype=torch.float32) + .5) / height
    xs = (torch.arange(width, device=device, dtype=torch.float32) + .5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack((xx, yy), 0)


def endpoint_targets(masks, margin, offsets, target_shape):
    """Detached candidate endpoint GT, only called by the training loss."""
    batch, height, width = masks.shape
    grid_h, grid_w = target_shape
    coordinates = base_coordinates(grid_h, grid_w, masks.device)[None, None].expand(batch, 9, -1, -1, -1)
    if offsets is not None:
        movement = F.interpolate(offsets.detach().reshape(batch, 18, *offsets.shape[-2:]).float(),
                                 size=target_shape, mode='bilinear', align_corners=False).reshape(batch, 9, 2, grid_h, grid_w)
        coordinates = coordinates + movement
    pixel_x = (coordinates[:, :, 0].clamp(0, 1 - 1e-7) * width).long()
    pixel_y = (coordinates[:, :, 1].clamp(0, 1 - 1e-7) * height).long()
    index = (pixel_y * width + pixel_x).reshape(batch, 9, -1)
    labels = masks.reshape(batch, 1, -1).expand(-1, 9, -1).gather(2, index)
    inside_margin = margin.reshape(batch, 1, -1).expand(-1, 9, -1).gather(2, index).float() / 255.
    classes = torch.arange(1, 10, device=masks.device)[None, :, None]
    return ((labels == classes).float() * (.5 + .5 * inside_margin)).reshape(batch, 9, grid_h, grid_w)


def _sampled_binary_loss(logits, target, sample_count):
    """Equal positive/negative groups; hard and uniform points include wrong parts.

    Work at native448 target resolution: a one-pixel valid region remains a
    positive. An absent class contributes negatives without inventing a positive.
    """
    # Fixed-shape masked top-k avoids a CUDA synchronization for every class's
    # nonzero list. Random priorities sample uniformly without replacement.
    z, y = logits.flatten(2), target.flatten(2)
    count = min(sample_count, z.shape[-1])
    random_priority = torch.rand_like(z)
    losses, available = [], []
    for positive in (True, False):
        eligible = y > 0 if positive else y == 0
        hard_scores = -z.detach() if positive else z.detach()
        hard = hard_scores.masked_fill(~eligible, -torch.inf).topk(count, dim=-1, sorted=False).indices
        uniform = random_priority.masked_fill(~eligible, -torch.inf).topk(count, dim=-1, sorted=False).indices
        selected = torch.cat((hard, uniform), -1)
        valid = eligible.gather(2, selected)
        pointwise = F.binary_cross_entropy_with_logits(z.gather(2, selected), y.gather(2, selected), reduction='none')
        losses.append((pointwise * valid).sum(-1) / valid.sum(-1).clamp_min(1))
        available.append(valid.any(-1))
    loss = torch.stack(losses, -1)
    valid_group = torch.stack(available, -1)
    return ((loss * valid_group).sum(-1) / valid_group.sum(-1).clamp_min(1)).mean()


def offset_loss(offsets, nearest, masks, radius):
    batch, _, _, height, width = offsets.shape
    if nearest.shape[-2:] != (height, width):
        raise ValueError('Cached nearest grid must equal the offset output grid')
    base = base_coordinates(height, width, offsets.device)[None, None]
    target = nearest.float()
    exists = (nearest[:, :, 0] != 65535) & (nearest[:, :, 1] != 65535)
    target = (target + .5) / target.new_tensor([masks.shape[-1], masks.shape[-2]])[None, None, :, None, None] - base
    reachable = exists & (target.abs().amax(2) <= radius)
    pointwise = F.smooth_l1_loss(offsets.float(), target, reduction='none', beta=.02).mean(2)
    inside = target.abs().amax(2) < 1. / max(masks.shape[-2:])
    groups = torch.stack((reachable & inside, reachable & ~inside), -1)
    count = groups.sum((2, 3))
    group_loss = (pointwise[..., None] * groups).sum((2, 3)) / count.clamp_min(1)
    class_loss = (group_loss * (count > 0)).sum(-1) / (count > 0).sum(-1).clamp_min(1)
    valid_class = count.sum(-1) > 0
    return ((class_loss * valid_class).sum(-1) / valid_class.sum(-1).clamp_min(1)).mean()


def joint_loss(outputs, masks, margin, nearest, quality_weight=.5, offset_weight=2., radius=.125, samples=32, coarse_weight=.2):
    segmentation = segmentation_loss(outputs['seg_logits'], masks)
    coarse = segmentation * 0
    if outputs.get('coarse_seg_logits') is not None:
        coarse_logits = F.interpolate(outputs['coarse_seg_logits'].float(), size=masks.shape[-2:], mode='bilinear', align_corners=False)
        coarse = F.cross_entropy(coarse_logits, masks)
    quality = segmentation * 0
    relocation = segmentation * 0
    if outputs.get('quality_logits') is not None and quality_weight:
        quality_map = F.interpolate(outputs['quality_logits'].float(), size=masks.shape[-2:], mode='bilinear', align_corners=False)
        target = endpoint_targets(masks, margin, outputs.get('offsets'), masks.shape[-2:])
        quality = _sampled_binary_loss(quality_map, target, samples)
    if outputs.get('offsets') is not None and offset_weight:
        relocation = offset_loss(outputs['offsets'], nearest, masks, radius)
    total = segmentation + coarse_weight * coarse + quality_weight * quality + offset_weight * relocation
    return total, {'segmentation': segmentation.detach(), 'coarse': coarse.detach(),
                   'quality': quality.detach(), 'offset': relocation.detach()}


def infer_points(outputs, native_size):
    """Image/model-only point production. Returns all nine queries, never GT."""
    height, width = native_size
    logits = F.interpolate(outputs['seg_logits'].float(), size=native_size, mode='bilinear', align_corners=False)
    z = logits.double()
    stable = torch.stack([z[:, cls] - torch.logsumexp(torch.cat((z[:, :cls], z[:, cls + 1:]), 1), 1)
                          for cls in range(1, 10)], 1)
    peak = stable.flatten(2).argmax(2)
    result = {'stable': torch.stack((peak % width, peak // width), -1)}
    if outputs.get('quality_logits') is not None:
        quality = F.interpolate(outputs['quality_logits'].float(), size=native_size, mode='bilinear', align_corners=False)
        chosen = quality.flatten(2).argmax(2)
        xy = torch.stack(((chosen % width).float() + .5, (chosen // width).float() + .5), -1)
        if outputs.get('offsets') is not None:
            movement = F.interpolate(outputs['offsets'].reshape(logits.shape[0], 18, *outputs['offsets'].shape[-2:]).float(),
                                     size=native_size, mode='bilinear', align_corners=False).reshape(logits.shape[0], 9, 2, -1)
            delta = movement.gather(3, chosen[:, :, None, None].expand(-1, -1, 2, 1)).squeeze(-1)
            xy = xy + delta * xy.new_tensor([width, height])
        integer = xy.floor().long()
        integer[..., 0].clamp_(0, width - 1)
        integer[..., 1].clamp_(0, height - 1)
        result['quality'] = integer
    return result, logits


def metrics(cm, point_counts, small_counts):
    cm = np.asarray(cm, dtype=np.int64)
    union = cm.sum(0) + cm.sum(1) - cm.diagonal()
    valid = union > 0
    fg = valid.copy()
    fg[0] = False
    iou = np.divide(cm.diagonal(), union, out=np.zeros(10, dtype=float), where=valid)
    return {'miou_foreground': float(iou[fg].mean()) if fg.any() else 0., 'per_class_iou': [float(iou[i]) if valid[i] else None for i in range(10)],
            'pixel_confusion': cm.tolist(), 'point_counts': point_counts, 'small_point_counts': small_counts,
            'point_hit': {key: value['hits'] / value['queries'] if value['queries'] else 0. for key, value in point_counts.items()}}


@torch.inference_mode()
def validate(model, loader, device, amp, export=None):
    model.eval()
    cm = np.zeros((10, 10), dtype=np.int64)
    counts, small, images = {}, {}, 0
    if export:
        (export / 'outputs').mkdir(parents=True, exist_ok=True)
        if (export / 'predictions.jsonl').exists():
            raise FileExistsError('Never append a second pass to an existing prediction manifest')
    for xs, masks, _, _, rows in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            batch = model(xs.to(device, non_blocking=True))
        for index, (gt, row) in enumerate(zip(masks, rows)):
            outputs = {key: value[index:index + 1] if value is not None else None for key, value in batch.items()}
            points, resized_logits = infer_points(outputs, tuple(gt.shape))
            pred = resized_logits.argmax(1)[0].cpu().numpy()
            target = gt.numpy()
            cm += np.bincount((target * 10 + pred).flatten(), minlength=100).reshape(10, 10)
            point_record = {}
            # Evaluator-only GT presence determines which positive queries score.
            for method, values in points.items():
                coordinates = values[0].cpu().numpy()
                counts.setdefault(method, {'hits': 0, 'queries': 0})
                small.setdefault(method, {'hits': 0, 'queries': 0})
                point_record[method] = coordinates.tolist()
                for cls in np.unique(target):
                    if not cls:
                        continue
                    x, y = coordinates[int(cls) - 1]
                    hit = int(target[y, x] == cls)
                    counts[method]['hits'] += hit
                    counts[method]['queries'] += 1
                    if (target == cls).mean() <= .01:
                        small[method]['hits'] += hit
                        small[method]['queries'] += 1
            if export:
                filename = hashlib.sha256(str(row['source_id']).encode()).hexdigest()[:24] + '.npz'
                np.savez_compressed(export / 'outputs' / filename,
                                    **{key: value[0].float().cpu().numpy() for key, value in outputs.items() if value is not None})
                append(export / 'predictions.jsonl', {'source_id': row['source_id'], 'split': 'val', 'gt_path': row['mask'],
                       'prediction_path': str((export / 'outputs' / filename).resolve()), 'point_xy': point_record,
                       'native_gt_shape': list(target.shape), 'point_selection_gt_access': False,
                       'query_enumeration': 'all_9_before_evaluator_GT_presence_filter'})
            images += 1
    return dict(metrics(cm, counts, small), images=images)


def rng_state(include_cuda=True):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if include_cuda and torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def configure_threads(threads):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def make_model(config, pretrained=True):
    fields = ('arm', 'width', 'rgb_width', 'point_width', 'offset_radius', 'init_seed')
    joint = JointConfig(**{key: config[key] for key in fields})
    return DinoJoint(joint, weights=config['dino_weights'] if pretrained else None)


def operator_profile(model, sample, device, amp):
    """Measured operator FLOPs are partial coverage, not theoretical total FLOPs."""
    saved = rng_state()
    training = model.training
    model.eval()
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.inference_mode(), torch.profiler.profile(activities=activities, record_shapes=True, with_flops=True) as profile:
            with torch.autocast(device_type=device.type, enabled=amp):
                model(sample[:1].to(device))
        covered = [{'operator': event.key, 'flops': int(event.flops)} for event in profile.key_averages() if event.flops]
        return {'forward_batch': 1, 'input_shape': list(sample[:1].shape),
                'covered_operator_flops': sum(event['flops'] for event in covered), 'operators': covered,
                'coverage': 'PyTorch profiler supported operators only; excludes unsupported elementwise/normalization/attention kernels; not complete theoretical FLOPs'}
    except (RuntimeError, AttributeError) as error:
        return {'available': False, 'reason': str(error), 'coverage': 'Do not substitute parameter count for FLOPs'}
    finally:
        model.train(training)
        restore_rng(saved)


def config_for(args):
    config = vars(args).copy()
    source = sha(__file__)
    component = sha(Path(__file__).with_name('full_joint_models.py'))
    for actual, expected, label in ((source, args.expected_source_sha256, 'trainer'),
                                    (component, args.expected_model_sha256, 'model component')):
        if expected and actual != expected:
            raise ValueError(f'Frozen {label} SHA differs')
    config.update(source_sha256=source, model_source_sha256=component,
                  records_sha256=sha(args.records), pretrained_sha256=sha(args.dino_weights),
                  classes=CLASSES, test_access=False, input_size=args.size,
                  architecture='DINO original SelectiveCL ViT-S/16; last4 blocks and norm trainable; shared real RGB pyramid',
                  selection='validation_native_positive_image_class_point_hit_then_miou_then_earliest',
                  augmentation='same deterministic per-row/per-epoch horizontal flip and ColorJitter(.15,.15,.1,.02)',
                  dense_loss='per-image CE + present-foreground macro Dice plus shared0.2 coarse auxiliary CE after448 upsampling; invariant to physical microbatch split',
                  quality_loss='balanced hard+uniform native448 candidate endpoint quality; own-class interior margin, thin positive retained',
                  offset_loss='native448 nearest class location at output grid; reachable inside/outside balanced SmoothL1',
                  inference='all9 queries model-only; stable float64 class log odds or quality peak plus predicted XY movement',
                  quality_samples_per_group=args.quality_samples, resume_schema=SCHEMA)
    return config


def state_validate(saved):
    required = ('schema', 'model', 'optimizer', 'scaler', 'rng', 'epoch_index', 'next_sample', 'updates',
                'best_key', 'images_exposed', 'config', 'sampler', 'training_update_wall_seconds',
                'training_cuda_event_seconds', 'best_budget_key', 'successful_optimizer_updates', 'amp_skipped_updates',
                'validation_record')
    missing = [key for key in required if key not in saved]
    if missing or saved.get('schema') != SCHEMA:
        raise ValueError(f'Incomplete/incompatible joint resume state: {missing}')
    if saved['sampler']['epoch_index'] != saved['epoch_index'] or saved['sampler']['next_sample'] != saved['next_sample']:
        raise ValueError('Sampler cursor and checkpoint cursor disagree')


def selected_snapshot_matches(saved, expected_key, kind):
    record = saved.get('validation_record')
    key_field = 'best_key' if kind == 'point' else 'best_budget_key'
    return (record is not None and tuple(record['key']) == tuple(expected_key)
            and tuple(saved[key_field]) == tuple(expected_key)
            and saved['updates'] == record['updates']
            and saved['epoch_index'] == record['epoch_index'] and saved['next_sample'] == 0)


def load_resume_consistent(resume, out, config, invariant):
    """Recover an interrupted best/budget/last multi-file commit explicitly.

    Selected files are written before last. If interrupted between renames, the
    newest complete selected snapshot contains optimizer/RNG/sampler too and is
    a valid continuation. A model is never matched to a key from another state.
    Missing selected state is repaired only from a complete matching snapshot;
    otherwise resume fails, retaining the evidence, rather than silently using
    an older model under the newer key.
    """
    out = Path(out)
    paths = list(dict.fromkeys((Path(resume), out / 'best.pt', out / 'best_budget.pt')))
    candidates = []
    for path in paths:
        if not path.exists():
            continue
        saved = torch.load(path, map_location='cpu', weights_only=False)
        state_validate(saved)
        for key in invariant:
            if saved['config'][key] != config[key]:
                raise ValueError(f'Resume frozen input changed in {path.name}: {key}')
        candidates.append((path, saved))
    if not candidates or not Path(resume).exists():
        raise FileNotFoundError(resume)
    origin, saved = max(candidates, key=lambda entry: (entry[1]['updates'], entry[1]['epoch_index'], entry[1]['next_sample']))
    repairs = []
    by_path = {str(path.resolve()): state for path, state in candidates}
    for kind, filename, key_field in (('point', 'best.pt', 'best_key'), ('budget', 'best_budget.pt', 'best_budget_key')):
        expected = saved[key_field]
        if expected[0] < 0:
            continue
        target = out / filename
        current = by_path.get(str(target.resolve()))
        if current is not None and selected_snapshot_matches(current, expected, kind):
            continue
        matching = [state for _, state in candidates if selected_snapshot_matches(state, expected, kind)]
        if not matching:
            raise ValueError(f'Cannot recover {filename}: no complete snapshot for selected key {expected}')
        if target.exists():
            os.replace(target, target.with_name(f'{target.stem}_before_recovery_{time.time_ns()}.pt'))
        save_checkpoint(target, matching[0])
        repairs.append(filename)
    if origin.resolve() != Path(resume).resolve():
        target = out / 'last.pt'
        if target.exists():
            os.replace(target, target.with_name(f'last_before_recovery_{time.time_ns()}.pt'))
        save_checkpoint(target, saved)
        repairs.append('last.pt')
    if repairs:
        write(out / f'resume_recovery_{time.time_ns()}.json',
              {'status': 'RECOVERED_CONSISTENT_COMPLETE_STATE', 'origin': str(origin), 'updates': saved['updates'],
               'epoch_index': saved['epoch_index'], 'repaired_files': repairs, 'old_files_preserved': True})
    return saved


def train(args):
    if args.effective_batch != 16 or args.effective_batch % args.batch:
        raise ValueError('Frozen effective_batch16 must be divisible by physical microbatch')
    if args.size % 16:
        raise ValueError('Input size must divide by patch16')
    out, device = Path(args.output), torch.device(args.device)
    configure_threads(args.cpu_threads)
    if device.type == 'cuda':
        # Reserve headroom for the worker display and avoid WDDM shared-memory
        # spill; all arms use the same allocator cap, not an algorithm change.
        torch.cuda.set_per_process_memory_fraction(.85, device=device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = config_for(args)
    config['cuda_allocator_fraction_of_total'] = .85 if device.type == 'cuda' else None
    rows, val_rows = read_records(args.records)
    cache_index = json.loads((Path(args.cache) / 'index.json').read_text(encoding='utf-8'))
    if cache_index['schema'] != CACHE_SCHEMA or cache_index['records_sha256'] != config['records_sha256'] or cache_index['size'] != args.size:
        raise ValueError('Geometry cache input/schema mismatch')
    if len(cache_index['files']) != len(rows):
        raise ValueError('Geometry cache is incomplete')
    config['cache_index_sha256'] = sha(Path(args.cache) / 'index.json')
    model = make_model(config).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    config.update(train_rows=len(rows), cleanval_rows=len(val_rows), trainable_parameters=sum(p.numel() for p in parameters),
                  total_parameters=sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    epoch_index, next_sample, updates, exposed = 0, 0, 0, 0
    best, best_budget = (-1., -1.), (-1., -1.)
    training_update_wall, training_cuda_event = 0., 0.
    successful_updates, skipped_updates = 0, 0
    validation_record = None
    if args.resume:
        invariant = ('arm', 'size', 'effective_batch', 'seed', 'lr', 'schedule_epochs', 'val_every', 'source_sha256',
                     'model_source_sha256', 'records_sha256', 'pretrained_sha256', 'cache_index_sha256',
                     'width', 'rgb_width', 'point_width', 'offset_radius', 'quality_weight', 'offset_weight',
                     'quality_samples', 'init_seed', 'compute_budget_seconds', 'coarse_weight')
        saved = load_resume_consistent(args.resume, out, config, invariant)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        epoch_index, next_sample, updates = saved['epoch_index'], saved['next_sample'], saved['updates']
        exposed, best = saved['images_exposed'], tuple(saved['best_key'])
        best_budget = tuple(saved['best_budget_key'])
        training_update_wall = saved['training_update_wall_seconds']
        training_cuda_event = saved['training_cuda_event_seconds']
        successful_updates = saved['successful_optimizer_updates']
        skipped_updates = saved['amp_skipped_updates']
        validation_record = saved['validation_record']
        restore_rng(saved['rng'])
    elif (out / 'last.pt').exists():
        raise FileExistsError('Existing run requires explicit --resume')
    write(out / 'config.json', config)
    val_loader = DataLoader(IITJoint(val_rows, args.size, args.seed), batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate, pin_memory=device.type == 'cuda',
                            generator=torch.Generator().manual_seed(args.seed))
    total_updates = math.ceil(len(rows) / args.effective_batch) * args.schedule_epochs
    begin, cpu_begin = time.perf_counter(), time.process_time()
    timed_updates, losses = [], []
    profile_evidence = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    def snapshot(epoch, cursor):
        return {'schema': SCHEMA, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'rng': rng_state(), 'epoch_index': epoch, 'next_sample': cursor, 'updates': updates,
                'images_exposed': exposed, 'best_key': list(best), 'config': config,
                'best_budget_key': list(best_budget), 'training_update_wall_seconds': training_update_wall,
                'training_cuda_event_seconds': training_cuda_event,
                'successful_optimizer_updates': successful_updates, 'amp_skipped_updates': skipped_updates,
                'validation_record': validation_record,
                'sampler': {'kind': 'epoch_seeded_randperm', 'seed': args.seed, 'epoch_index': epoch, 'next_sample': cursor,
                            'augmentation_seed': 'seed+epoch*train_rows+row_index'},
                'cost_this_process': {'wall_seconds': time.perf_counter() - begin, 'cpu_seconds': time.process_time() - cpu_begin}}

    if not args.resume:
        save_checkpoint(out / 'last.pt', snapshot(0, 0))
    for epoch in range(epoch_index, args.epochs):
        cursor = next_sample if epoch == epoch_index else 0
        data = IITJoint(rows, args.size, args.seed, epoch, args.cache)
        loader_kwargs = dict(batch_size=args.batch, sampler=EpochSampler(len(data), args.seed, epoch, cursor),
                             num_workers=args.workers, collate_fn=collate, pin_memory=device.type == 'cuda',
                             generator=torch.Generator().manual_seed(args.seed + epoch))
        if args.workers:
            loader_kwargs.update(prefetch_factor=2)
        loader = DataLoader(data, **loader_kwargs)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_samples = 0
        loss_sum, epoch_images = torch.zeros((), device=device), 0
        tick = time.perf_counter()
        cuda_start = None
        if device.type == 'cuda':
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        for xs, ys, margin, nearest, _ in loader:
            if args.command == 'profile' and profile_evidence is None:
                profile_evidence = operator_profile(model, xs, device, device.type == 'cuda')
                tick = time.perf_counter()
                if cuda_start is not None:
                    cuda_start.record()
            group_start = cursor - accumulated_samples
            group_samples = min(args.effective_batch, len(rows) - group_start)
            masks = torch.stack(ys).to(device, non_blocking=True)
            margin = margin.to(device, non_blocking=True)
            nearest = nearest.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                outputs = model(xs.to(device, non_blocking=True))
                loss, parts = joint_loss(outputs, masks, margin, nearest, args.quality_weight,
                                         args.offset_weight, args.offset_radius, args.quality_samples, args.coarse_weight)
            scaler.scale(loss * len(xs) / group_samples).backward()
            accumulated_samples += len(xs)
            cursor += len(xs)
            exposed += len(xs)
            epoch_images += len(xs)
            loss_sum += loss.detach() * len(xs)
            if accumulated_samples == group_samples:
                lr = args.lr * max(.01, (1 - min(updates, total_updates) / total_updates) ** .9)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < scale_before
                skipped_updates += int(skipped)
                successful_updates += int(not skipped)
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                accumulated_samples = 0
                # One small transfer at the optimizer boundary replaces several
                # float(cuda_tensor) synchronizations per physical microbatch.
                numeric = torch.stack([loss_sum / epoch_images, *parts.values()]).cpu().tolist()
                if not all(math.isfinite(value) for value in numeric):
                    raise FloatingPointError('Nonfinite joint loss; failed record retained, no silent retry')
                numeric_parts = dict(zip(parts, numeric[1:]))
                if device.type == 'cuda':
                    cuda_end = torch.cuda.Event(enable_timing=True)
                    cuda_end.record()
                    torch.cuda.synchronize()
                    training_cuda_event += cuda_start.elapsed_time(cuda_end) / 1000.
                duration = time.perf_counter() - tick
                training_update_wall += duration
                timed_updates.append(duration)
                losses.append(numeric[0])
                if updates % args.log_every == 0 or args.command == 'profile':
                    append(out / 'progress.jsonl', dict(stage='train', arm=args.arm, epoch=epoch + 1, updates=updates,
                           epoch_sample_cursor=cursor, train_rows=len(rows), images_exposed=exposed,
                           loss=numeric[0], last_microbatch_loss_components=numeric_parts, lr=lr,
                           update_seconds=duration, wall_seconds=time.perf_counter() - begin,
                           training_update_wall_seconds=training_update_wall,
                           training_cuda_event_seconds=training_cuda_event,
                           successful_optimizer_updates=successful_updates, amp_skipped_updates=skipped_updates,
                           cpu_seconds=time.process_time() - cpu_begin))
                    write(out / 'progress.json', dict(status='RUNNING', arm=args.arm, epoch=epoch + 1, updates=updates,
                          epoch_sample_cursor=cursor, images_exposed=exposed, update_seconds=duration,
                          wall_seconds=time.perf_counter() - begin))
                if args.command == 'profile' and len(timed_updates) >= args.profile_updates:
                    validation_speed = None
                    if args.profile_val_images:
                        profile_indices = np.linspace(0, len(val_rows) - 1,
                            min(args.profile_val_images, len(val_rows)), dtype=np.int64)
                        profile_rows = [val_rows[int(index)] for index in profile_indices]
                        speed_loader = DataLoader(IITJoint(profile_rows, args.size, args.seed), batch_size=args.batch,
                            shuffle=False, num_workers=args.workers, collate_fn=collate, pin_memory=device.type == 'cuda',
                            generator=torch.Generator().manual_seed(args.seed))
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        startup_begin = time.perf_counter()
                        speed_iterator = iter(speed_loader)
                        first_batch = next(speed_iterator)
                        startup_wall = time.perf_counter() - startup_begin
                        val_begin, val_cpu = time.perf_counter(), time.process_time()
                        speed_check = validate(model, itertools.chain((first_batch,), speed_iterator), device, device.type == 'cuda')
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        val_wall = time.perf_counter() - val_begin
                        validation_speed = dict(scientific_result=False, fixed_manifest_prefix=False,
                            sampling='fixed evenly-spaced manifest indices, no label/outcome selection',
                            source_ids=[row['source_id'] for row in profile_rows],
                            loader_startup_and_first_decode_seconds=startup_wall,
                            images=speed_check['images'], wall_seconds=val_wall, cpu_seconds=time.process_time() - val_cpu,
                            seconds_per_image=val_wall / len(profile_rows),
                            estimated_full_1757_seconds_linear=startup_wall + val_wall / len(profile_rows) * len(val_rows),
                            geometry_check='native point coordinates bounded; all9 generated without GT filtering',
                            limitations='startup measured once separately; remaining native-size work linearly extrapolated, not a full-validation measurement')
                    result = dict(status='PROFILE_COMPLETE', scientific_result=False, arm=args.arm,
                           attempted_optimizer_updates=len(timed_updates), real_optimizer_updates=successful_updates,
                           seconds_per_update=timed_updates,
                           median_seconds_excluding_first=float(np.median(timed_updates[1:] or timed_updates)),
                           wall_seconds=time.perf_counter() - begin, cpu_seconds=time.process_time() - cpu_begin,
                           effective_batch=args.effective_batch, physical_batch=args.batch,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
                           peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0,
                           trainable_parameters=config['trainable_parameters'], total_parameters=config['total_parameters'],
                           successful_optimizer_updates=successful_updates, amp_skipped_updates=skipped_updates,
                           forward_operator_profile=profile_evidence, validation_speed=validation_speed,
                           validation_run=bool(validation_speed),
                           timing_scope='FLOP profiler dry-forward excluded from update ticks; first update warmup excluded from median')
                    result['cpu_seconds_scope'] = 'main training process only; DataLoader worker CPU excluded'
                    write(out / 'terminal.json', result)
                    print(json.dumps(result), flush=True)
                    return
                if updates % args.checkpoint_updates == 0 and cursor < len(rows):
                    save_checkpoint(out / 'last.pt', snapshot(epoch, cursor))
                tick = time.perf_counter()
                if cuda_start is not None:
                    cuda_start.record()
        if accumulated_samples:
            raise AssertionError('Uncommitted gradient group at epoch boundary')
        validate_now = (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        result = validate(model, val_loader, device, device.type == 'cuda') if validate_now else None
        method = 'quality' if result and 'quality' in result['point_hit'] else 'stable'
        key = (result['point_hit'][method], result['miou_foreground']) if result else None
        if result:
            validation_record = {'key': list(key), 'epoch_index': epoch + 1, 'updates': updates,
                                 'selection_method': method, 'training_update_wall_seconds': training_update_wall}
        improved = key is not None and key > best
        if improved:
            best = key
        # Equal-budget supplementary selection uses only scheduled validation
        # boundaries whose accumulated update wall is still within the seal.
        # This does not truncate the equal-exposure run or use UMD to choose.
        budget_eligible = result is not None and args.compute_budget_seconds is not None and training_update_wall <= args.compute_budget_seconds
        budget_improved = budget_eligible and key > best_budget
        if budget_improved:
            best_budget = key
        append(out / 'progress.jsonl', dict(stage='validation' if validate_now else 'epoch', arm=args.arm,
               epoch=epoch + 1, updates=updates, images_exposed=exposed,
               train_loss=float(loss_sum / epoch_images) if epoch_images else None,
               selection_method=method, wall_seconds=time.perf_counter() - begin,
               training_update_wall_seconds=training_update_wall,
               training_cuda_event_seconds=training_cuda_event,
               successful_optimizer_updates=successful_updates, amp_skipped_updates=skipped_updates,
               compute_budget_seconds=args.compute_budget_seconds,
               budget_checkpoint_eligible=budget_eligible,
               cpu_seconds=time.process_time() - cpu_begin, **(result or {})))
        saved_state = snapshot(epoch + 1, 0)
        if improved:
            save_checkpoint(out / 'best.pt', saved_state)
        if budget_improved:
            save_checkpoint(out / 'best_budget.pt', saved_state)
        # Commit continuation last only after the selected snapshots it names.
        save_checkpoint(out / 'last.pt', saved_state)
        if result:
            write(out / 'latest_validation.json', dict(epoch=epoch + 1, selection_method=method, best_key=list(best), **result))
        print(json.dumps({'epoch': epoch + 1, 'updates': updates, 'arm': args.arm, 'best_key': list(best)}), flush=True)
    result = dict(status='COMPLETE', arm=args.arm, epochs=args.epochs, updates=updates, images_exposed=exposed,
                  expected_images_exposed=args.epochs * len(rows), best_key=list(best), test_access=False,
                  training_update_wall_seconds=training_update_wall,
                  training_cuda_event_seconds=training_cuda_event,
                  successful_optimizer_updates=successful_updates, amp_skipped_updates=skipped_updates,
                  compute_budget_seconds=args.compute_budget_seconds, best_budget_key=list(best_budget),
                  compute_budget_granularity='scheduled validation boundaries; last eligible boundary can undershoot the wall budget',
                  updates_scope='updates is attempted AMP step boundaries; successful_optimizer_updates excludes amp_skipped_updates',
                  cuda_event_scope='stream elapsed including stream idle between start/end; not summed kernel execution',
                  cpu_seconds_scope='main training process only; DataLoader worker CPU excluded',
                  wall_seconds=time.perf_counter() - begin, cpu_seconds=time.process_time() - cpu_begin,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0)
    if exposed != args.epochs * len(rows):
        raise AssertionError('Completed run exposure count differs from fixed data exposure')
    write(out / 'terminal.json', result)
    print(json.dumps(result), flush=True)


def export(args):
    configure_threads(args.cpu_threads)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_validate(saved)
    if sha(args.records) != saved['config']['records_sha256']:
        raise ValueError('Manifest differs from checkpoint input')
    if sha(__file__) != saved['config']['source_sha256'] or sha(Path(__file__).with_name('full_joint_models.py')) != saved['config']['model_source_sha256']:
        raise ValueError('Frozen checkpoint source differs')
    _, rows = read_records(args.records)
    model = make_model(saved['config'], pretrained=False)
    model.load_state_dict(saved['model'])
    model.to(args.device)
    loader = DataLoader(IITJoint(rows, saved['config']['size'], saved['config']['seed']), batch_size=args.batch,
                        collate_fn=collate, num_workers=args.workers, pin_memory=args.device.startswith('cuda'))
    result = validate(model, loader, torch.device(args.device), args.device.startswith('cuda'), Path(args.output))
    write(Path(args.output) / 'terminal.json', dict(status='COMPLETE', checkpoint_sha256=sha(args.checkpoint), test_access=False, **result))


def selftest():
    """Synthetic-only meaningful checks, intended for the auxiliary worker CPU."""
    configure_threads(1)
    torch.manual_seed(37)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[3:13, 7] = 5  # thin foreground survives native-mask target generation
    margin_np, nearest_np = geometry(mask, 4)
    assert (margin_np[mask == 5] > 0).all()
    assert (nearest_np[0, 4, 0] == 7).all()
    # Correct flipped x coordinate is8 for a width16 mask, not an interpolated label.
    assert (nearest_np[1, 4, 0] == 8).all()
    masks = torch.from_numpy(mask.astype(np.int64))[None]
    margin = torch.from_numpy(margin_np)[None]
    nearest = torch.from_numpy(nearest_np[0].astype(np.int32))[None]
    outputs = {'seg_logits': torch.randn(1, 10, 4, 4, requires_grad=True),
               'coarse_seg_logits': torch.randn(1, 10, 1, 1, requires_grad=True),
               'quality_logits': torch.randn(1, 9, 4, 4, requires_grad=True),
               'offsets': torch.zeros(1, 9, 2, 4, 4, requires_grad=True)}
    positive = endpoint_targets(masks, margin, None, masks.shape[-2:])
    assert (positive[0, 4][masks[0] == 5] >= .5).all() and positive[0, 4].count_nonzero() == 10
    loss, _ = joint_loss(outputs, masks, margin, nearest, radius=.5)
    loss.backward()
    assert torch.isfinite(loss) and all(value.grad is not None and torch.isfinite(value.grad).all() for value in outputs.values())
    empty = torch.zeros_like(masks)
    empty_nearest = torch.full_like(nearest, 65535)
    empty_loss, _ = joint_loss(outputs, empty, torch.zeros_like(margin), empty_nearest, radius=.5)
    assert torch.isfinite(empty_loss)
    points, _ = infer_points({key: value.detach() for key, value in outputs.items()}, (19, 23))
    assert points['quality'].shape == (1, 9, 2) and (points['quality'][:, :, 0] < 23).all()
    # Per-image segmentation reduction produces the same accumulated gradient.
    z = torch.randn(3, 10, 4, 4, requires_grad=True)
    y = torch.randint(0, 10, (3, 16, 16))
    segmentation_loss(z, y).backward()
    full_gradient = z.grad.clone()
    z.grad = None
    (segmentation_loss(z[:2], y[:2]) * 2 / 3).backward()
    (segmentation_loss(z[2:], y[2:]) / 3).backward()
    assert torch.allclose(full_gradient, z.grad, atol=1e-7, rtol=1e-5)
    order = list(EpochSampler(17, 12, 2))
    assert list(EpochSampler(17, 12, 2, 8)) == order[8:]
    state = dict(schema=SCHEMA, model={}, optimizer={}, scaler={}, rng=rng_state(include_cuda=False), epoch_index=3,
                 next_sample=16, updates=19, best_key=[.5, .6], images_exposed=67, config={},
                 training_update_wall_seconds=2., training_cuda_event_seconds=1., best_budget_key=[.4, .5],
                 successful_optimizer_updates=19, amp_skipped_updates=0,
                 validation_record=None,
                 sampler={'epoch_index': 3, 'next_sample': 16})
    state_validate(state)
    bad = dict(state)
    bad.pop('optimizer')
    try:
        state_validate(bad)
        raise AssertionError('Missing optimizer accepted')
    except ValueError:
        pass
    # Simulate interruption after new best rename and before last rename.
    with tempfile.TemporaryDirectory(prefix='joint_resume_synthetic_', dir=Path(__file__).resolve().parent) as temporary:
        folder = Path(temporary)
        old = dict(state, updates=18, epoch_index=2, next_sample=0, best_key=[.4, .5], best_budget_key=[-1., -1.],
                   sampler={'epoch_index': 2, 'next_sample': 0},
                   validation_record={'key': [.4, .5], 'updates': 18, 'epoch_index': 2})
        newer = dict(old, updates=19, epoch_index=3, best_key=[.5, .6],
                     sampler={'epoch_index': 3, 'next_sample': 0},
                     validation_record={'key': [.5, .6], 'updates': 19, 'epoch_index': 3})
        save_checkpoint(folder / 'last.pt', old)
        save_checkpoint(folder / 'best.pt', newer)
        restored = load_resume_consistent(folder / 'last.pt', folder, {}, ())
        assert restored['updates'] == 19
        assert selected_snapshot_matches(torch.load(folder / 'best.pt', weights_only=False), restored['best_key'], 'point')
    print(json.dumps({'status': 'PASS', 'synthetic_only': True, 'checks': [
        'native_thin_foreground_retained', 'flip_nearest_geometry', 'finite_joint_quality_offset_backward',
        'all_background_no_invented_positive', 'all9_inference_without_GT',
        'physical_microbatch_seg_gradient_normalization', 'resumable_epoch_sampler', 'complete_resume_schema',
        'interrupted_best_last_transaction_recovery', 'shared_coarse_classifier_gradient']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('selftest')
    cache = commands.add_parser('precompute')
    cache.add_argument('--records', required=True)
    cache.add_argument('--output', required=True)
    cache.add_argument('--size', type=int, default=448)
    for command in ('train', 'profile'):
        sub = commands.add_parser(command)
        sub.add_argument('--records', required=True)
        sub.add_argument('--cache', required=True)
        sub.add_argument('--output', required=True)
        sub.add_argument('--dino-weights', required=True)
        sub.add_argument('--arm', choices=ARMS, required=True)
        sub.add_argument('--size', type=int, default=448)
        sub.add_argument('--width', type=int, default=64)
        sub.add_argument('--rgb-width', type=int, default=32)
        sub.add_argument('--point-width', type=int, default=32)
        sub.add_argument('--offset-radius', type=float, default=.125)
        sub.add_argument('--quality-weight', type=float, default=.5)
        sub.add_argument('--coarse-weight', type=float, default=.2)
        sub.add_argument('--offset-weight', type=float, default=2.)
        sub.add_argument('--quality-samples', type=int, default=32)
        sub.add_argument('--batch', type=int, default=2)
        sub.add_argument('--effective-batch', type=int, default=16)
        sub.add_argument('--epochs', type=int, default=80)
        sub.add_argument('--schedule-epochs', type=int, default=80)
        sub.add_argument('--lr', type=float, default=6e-5)
        sub.add_argument('--seed', type=int, default=20261016)
        sub.add_argument('--init-seed', type=int, default=20261016)
        sub.add_argument('--workers', type=int, default=2)
        sub.add_argument('--cpu-threads', type=int, default=2)
        sub.add_argument('--device', default='cuda')
        sub.add_argument('--val-every', type=int, default=2)
        sub.add_argument('--log-every', type=int, default=10)
        sub.add_argument('--checkpoint-updates', type=int, default=100)
        sub.add_argument('--profile-updates', type=int, default=8)
        sub.add_argument('--profile-val-images', type=int, default=16)
        sub.add_argument('--compute-budget-seconds', type=float)
        sub.add_argument('--resume')
        sub.add_argument('--expected-source-sha256')
        sub.add_argument('--expected-model-sha256')
    sub = commands.add_parser('export')
    sub.add_argument('--records', required=True)
    sub.add_argument('--checkpoint', required=True)
    sub.add_argument('--output', required=True)
    sub.add_argument('--batch', type=int, default=2)
    sub.add_argument('--workers', type=int, default=0)
    sub.add_argument('--cpu-threads', type=int, default=2)
    sub.add_argument('--device', default='cuda')
    args = parser.parse_args()
    lock, owns_lock = None, False
    try:
        if hasattr(args, 'output'):
            Path(args.output).mkdir(parents=True, exist_ok=True)
            lock = Path(args.output) / 'owner.lock'
            with lock.open('x', encoding='utf-8') as stream:
                json.dump({'pid': os.getpid(), 'host': socket.gethostname(), 'started_unix': time.time(),
                           'command': args.command, 'arm': getattr(args, 'arm', None)}, stream)
            owns_lock = True
        if args.command == 'selftest':
            selftest()
        elif args.command == 'precompute':
            precompute(args)
        elif args.command == 'export':
            export(args)
        else:
            train(args)
    except BaseException:
        if owns_lock:
            path = Path(args.output) / 'terminal.json'
            if path.exists():
                os.replace(path, path.with_name(f'terminal_previous_{time.time_ns()}.json'))
            write(path, {'status': 'FAILED', 'traceback': traceback.format_exc(), 'test_access': False,
                         'pid': os.getpid(), 'checkpoint_preserved': (Path(args.output) / 'last.pt').exists()})
        raise
    finally:
        if owns_lock and lock and lock.exists():
            try:
                if json.loads(lock.read_text(encoding='utf-8'))['pid'] == os.getpid():
                    lock.unlink()
            except (OSError, ValueError, KeyError):
                pass


if __name__ == '__main__':
    main()

"""Matched V2 learned controls from existing alley first-visible SceneDepth."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from cnh_rgb_alley_v2 import (AzimuthMaskedFrustumFusion, COLLECTION_SHA256,
                               PARTITION_SHA256, QUERY_NAMES, fit_seed, train_pos_weight)
from cnh_rgb_dev_comparison import BATCH, EPOCHS, SEEDS, checked, rates, read_inputs, sha
from cnh_rgb_frustum import zone_rgb_support
from cnh_street_e2e_train import metrics


def native_support(data: dict) -> tuple[np.ndarray, str]:
    first = data['collection']['layouts'][0]
    overlay = json.loads(checked({'path': first['overlay'], 'sha256': first['overlay_sha256']}).read_text(encoding='utf-8-sig'))
    camera_path = checked(overlay['frames'][0]['original_files']['camera.json'])
    camera = json.loads(camera_path.read_text(encoding='utf-8-sig'))
    transform = np.linalg.inv(np.asarray(camera['T_camera_tof'], np.float64))
    support = zone_rgb_support(camera['K'], transform, (640, 360), (90, 160))
    if support.shape != (64, 90, 160) or not np.all(support.sum((1, 2)) > 0):
        raise ValueError('Native depth support invalid')
    return support, sha(camera_path)


def depth_batch(depth: np.ndarray, indices, device: torch.device) -> torch.Tensor:
    source = np.asarray(depth[indices], dtype=np.float32)
    valid = np.isfinite(source) & (source > 0) & (source < 100)
    result = np.empty((len(source), 3, 360, 640), np.float32)
    result[:, 0] = np.log1p(np.where(valid, source, 0)) / math.log(101)
    result[:, 1] = valid
    result[:, 2] = 0
    return torch.from_numpy(result).to(device)


def fit_depth_seed(data: dict, depth: np.ndarray, support_np: np.ndarray,
                   seed: int, output: Path, pos_weight: np.ndarray) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda')
    model = AzimuthMaskedFrustumFusion().to(device)
    histogram = torch.zeros((1, 64, 16), dtype=torch.float32, device=device)
    scalar = torch.zeros((1, 64), dtype=torch.float32, device=device)
    valid = torch.zeros((1, 64), dtype=torch.bool, device=device)
    support = torch.from_numpy(support_np).to(device)
    labels = torch.from_numpy(np.maximum(data['labels'], 0).astype(np.float32)).to(device)
    known = torch.from_numpy(data['labels'] >= 0).to(device)
    weight = torch.from_numpy(pos_weight).to(device)
    train_indices = np.flatnonzero(data['train'])
    encoder = list(model.rgb.parameters())
    encoder_ids = {id(p) for p in encoder}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) not in encoder_ids], 'lr': 3e-4},
        {'params': encoder, 'lr': 3e-5},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    losses = []
    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(train_indices), generator=generator).numpy()
        epoch_loss = 0.0
        for start in range(0, len(train_indices), BATCH):
            indices = train_indices[order[start:start+BATCH]]
            rgb = depth_batch(depth, indices, device)
            size = len(indices)
            optimizer.zero_grad(set_to_none=True)
            logits = model(rgb, histogram.expand(size, -1, -1),
                           scalar.expand(size, -1), scalar.expand(size, -1),
                           valid.expand(size, -1), scalar.expand(size, -1),
                           support)['occupancy_logits']
            loss = F.binary_cross_entropy_with_logits(
                logits, labels[indices], pos_weight=weight, reduction='none')[known[indices]].mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite depth training loss')
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * size
        scheduler.step()
        losses.append(epoch_loss / len(train_indices))
        print(json.dumps({'arm': 'visible_depth_full', 'seed': seed, 'epoch': epoch+1,
                          'train_loss': losses[-1]}), flush=True)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(depth), BATCH):
            indices = np.arange(start, min(start+BATCH, len(depth)))
            size = len(indices)
            rgb = depth_batch(depth, indices, device)
            logits = model(rgb, histogram.expand(size, -1, -1),
                           scalar.expand(size, -1), scalar.expand(size, -1),
                           valid.expand(size, -1), scalar.expand(size, -1),
                           support)['occupancy_logits']
            chunks.append(logits.cpu().numpy())
    predictions = np.concatenate(chunks)
    if predictions.shape != (960, 6) or not np.isfinite(predictions).all():
        raise ValueError('Invalid depth prediction')
    model_path = output/f'visible_depth_full-seed-{seed}.pt'
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_path)
    prediction_path = output/f'visible_depth_full-seed-{seed}-predictions.npz'
    np.savez_compressed(prediction_path, frame_key=np.array([r['frame_key'] for r in data['rows']]),
                        logits=predictions, dev=data['dev'])
    layouts = {}
    for layout in sorted({r['layout_id'] for r in data['rows']}):
        selection = data['dev'] & np.array([r['layout_id'] == layout for r in data['rows']])
        if selection.any():
            layouts[layout] = rates(metrics(data['labels'][selection], predictions[selection]))
    return dict(arm='visible_depth_full', seed=seed, train_losses=losses,
                dev=rates(metrics(data['labels'][data['dev']], predictions[data['dev']])),
                dev_layouts=layouts, model_path=str(model_path), model_sha256=sha(model_path),
                predictions_path=str(prediction_path), predictions_sha256=sha(prediction_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-overlay', type=Path, required=True)
    parser.add_argument('--partition-plan', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Matched learned controls require CUDA')
    if sha(args.collection_overlay) != COLLECTION_SHA256 or sha(args.partition_plan) != PARTITION_SHA256:
        raise ValueError('Frozen Development inputs differ')
    prepared = json.loads((args.prepared/'result.json').read_text(encoding='utf-8'))
    if prepared['status'] != 'COMPLETE_VISIBLE_DEPTH_DEVELOPMENT_AUDIT' or prepared['frame_count'] != 960:
        raise ValueError('Full prepared depth audit required')
    for name, digest in prepared['files'].items():
        if sha(args.prepared/name) != digest:
            raise ValueError('Prepared depth input hash differs: '+name)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError('Fresh or empty output required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    data = read_inputs(args.collection_overlay, args.partition_plan)
    if tuple(data['query_names']) != QUERY_NAMES or len(data['rows']) != 960:
        raise ValueError('Frozen query/row identity differs')
    with np.load(args.prepared/'perfect-tof-h3.npz', allow_pickle=False) as tof:
        if list(tof['frame_key'].astype(str)) != [r['frame_key'] for r in data['rows']]:
            raise ValueError('Prepared ray identity differs')
        perfect = dict(data)
        perfect['histogram'] = tof['histogram'].copy()
        perfect['scalar'] = tof['nearest_m'].copy()
        perfect['valid'] = np.isfinite(perfect['scalar'])
    perfect['ambient'] = np.zeros_like(data['ambient'])
    perfect['rgb'] = np.zeros_like(data['rgb'])
    depth = np.load(args.prepared/'visible-depth-f32.npy', mmap_mode='r', allow_pickle=False)
    if depth.shape != (960, 360, 640) or depth.dtype != np.float32:
        raise ValueError('Native depth dimensions/type differ')
    native, camera_hash = native_support(data)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output/'native-frustum-support.npy', native)
    weight, balance = train_pos_weight(data['labels'], data['train'])
    started = time.monotonic()
    results = []
    for seed in SEEDS:
        result = fit_seed(perfect, 'perfect_tof_h3', seed, args.output, weight)
        results.append(result)
        print(json.dumps({'arm': result['arm'], 'seed': seed, 'dev': result['dev'],
                          'wall_s': time.monotonic()-started}), flush=True)
        result = fit_depth_seed(data, depth, native, seed, args.output, weight)
        results.append(result)
        print(json.dumps({'arm': result['arm'], 'seed': seed, 'dev': result['dev'],
                          'wall_s': time.monotonic()-started}), flush=True)
    report = dict(status='COMPLETE_VISIBLE_DEPTH_LEARNED_CEILINGS_DEVELOPMENT',
                  benchmark_eligible=False, frame_count=960, train_frames=480, dev_frames=480,
                  query_names=QUERY_NAMES, arms=('perfect_tof_h3', 'visible_depth_full'),
                  seeds=SEEDS, epochs=EPOCHS, batch_size=BATCH, threshold_logit=0,
                  train_pos_weight=balance, prepared_path=str(args.prepared),
                  prepared_result_sha256=sha(args.prepared/'result.json'),
                  native_support_sha256=sha(args.output/'native-frustum-support.npy'),
                  camera_sha256=camera_hash, collection_overlay_sha256=sha(args.collection_overlay),
                  partition_sha256=sha(args.partition_plan), code_sha256=sha(__file__),
                  protocol_sha256=sha(Path(__file__).with_name('CNH_RGB_VISIBLE_DEPTH_CEILING_PROTOCOL_20260925.md')),
                  model_code_sha256=sha(Path(__file__).with_name('cnh_rgb_alley_v2.py')),
                  results=results, device=torch.cuda.get_device_name(), wall_s=time.monotonic()-started,
                  claim_limit='Development and first-visible depth only; only 3 train and 3 dev layouts. Learned full-depth arm is an optimization control, not a mathematical ceiling or independent label validation.')
    (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'wall_s': report['wall_s']}), flush=True)


if __name__ == '__main__':
    main()

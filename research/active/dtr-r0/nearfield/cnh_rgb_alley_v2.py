"""Prospective alley Development V2: fixed azimuth queries and train-only balance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cnh_rgb_dev_comparison import BATCH, EPOCHS, SEEDS, rates, read_inputs, sha
from cnh_rgb_fusion import FrustumFusion
from cnh_street_e2e_train import metrics


QUERY_NAMES = ('left_HEAD', 'left_BODY', 'centre_HEAD', 'centre_BODY', 'right_HEAD', 'right_BODY')
AZIMUTH_COLUMNS = ((0, 1, 2), (0, 1, 2), (3, 4), (3, 4), (5, 6, 7), (5, 6, 7))
COLLECTION_SHA256 = 'f72425577e4e06f9a95e39ba2ef0a9b3d6fc087157e74d31bcfce7927ba4840f'
PARTITION_SHA256 = '7ae115183bab26876eedc4d9bd3b11cb0b7613c2b56dddb3055266fe1a586a9b'


def query_azimuth_mask() -> torch.Tensor:
    """[query, zone] hard support for row-major 8x8 ToF zones."""
    mask = torch.zeros((6, 64), dtype=torch.bool)
    for query, columns in enumerate(AZIMUTH_COLUMNS):
        for row in range(8):
            mask[query, [8 * row + column for column in columns]] = True
    return mask


def train_pos_weight(labels: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-query negative/positive ratio, computed on known training labels only."""
    labels = np.asarray(labels)
    train = np.asarray(train)
    if labels.ndim != 2 or labels.shape[1] != 6 or train.shape != (len(labels),):
        raise ValueError('Six-query labels and frame train mask required')
    y = labels[train]
    if not np.isin(y, (-1, 0, 1)).all():
        raise ValueError('Labels must be UNKNOWN, absent, or present')
    positive = (y == 1).sum(axis=0)
    negative = (y == 0).sum(axis=0)
    unknown = (y == -1).sum(axis=0)
    if np.any(positive == 0) or np.any(negative == 0):
        raise ValueError('Each query needs both known train classes')
    weight = (negative / positive).astype(np.float32)
    return weight, dict(positive=positive.tolist(), negative=negative.tolist(),
                        unknown=unknown.tolist(), pos_weight=weight.tolist())


def masked_attention(query: torch.Tensor, zones: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if query.ndim != 2 or zones.ndim != 3 or query.shape[1] != zones.shape[2]:
        raise ValueError('Query/zone width mismatch')
    if mask.shape != (query.shape[0], zones.shape[1]) or mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError('Nonempty Boolean query-zone mask required')
    affinity = torch.einsum('qd,bzd->bqz', query, zones) / zones.shape[-1] ** .5
    return torch.softmax(affinity.masked_fill(~mask[None], torch.finfo(affinity.dtype).min), -1)


class AzimuthMaskedFrustumFusion(FrustumFusion):
    """V1 parameters with a fixed left/centre/right query attention mask."""

    def __init__(self):
        super().__init__(64, 16, width=64, queries=6)
        self.register_buffer('query_zone_mask', query_azimuth_mask())

    def forward(self, rgb, histogram, ambient, scalar_m, scalar_valid, age_s, support, mode='cnh'):
        if mode != 'cnh':
            raise ValueError('V2 comparison fixes CNH mode')
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.shape[2] % 4 or rgb.shape[3] % 4:
            raise ValueError('RGB must be [B,3,H,W] with dimensions divisible by four')
        batch = rgb.shape[0]
        if histogram.shape != (batch, 64, 16):
            raise ValueError('CNH shape mismatch')
        if any(x.shape != (batch, 64) for x in (ambient, scalar_m, scalar_valid, age_s)):
            raise ValueError('Zone metadata shape mismatch')
        if not all(torch.isfinite(x).all() for x in (rgb, histogram, ambient, age_s)):
            raise ValueError('Nonfinite observable input')
        if not torch.isfinite(scalar_m[scalar_valid.bool()]).all() or torch.any(age_s < 0):
            raise ValueError('Invalid scalar or age')
        features = self.rgb(rgb)
        grid = features.shape[-2:]
        if support.ndim == 3:
            support = support.unsqueeze(0).expand(batch, -1, -1, -1)
        if support.shape != (batch, 64, *grid) or not torch.isfinite(support).all():
            raise ValueError('Frustum support/grid mismatch')
        if torch.any((support < 0) | (support > 1)):
            raise ValueError('Frustum support outside [0,1]')
        total = support.sum((-1, -2))
        if torch.any(total <= 0):
            raise ValueError('Each zone needs RGB support')
        rgb_zone = torch.einsum('bchw,bzhw->bzc', features, support) / total[..., None]
        valid = scalar_valid.bool()
        safe_scalar = torch.where(valid, scalar_m, torch.zeros_like(scalar_m))
        public = torch.stack((torch.log1p(torch.clamp_min(ambient, 0)), safe_scalar,
                              valid.to(rgb.dtype), torch.log1p(age_s)), -1)
        token = self.metadata(public)
        token = token + self.cnh(torch.sign(histogram) * torch.log1p(histogram.abs()))
        zones = self.fuse(torch.cat((rgb_zone, token), -1)) + self.zone_position(self.zone_coordinates)[None]
        weights = masked_attention(self.query, zones, self.query_zone_mask)
        decoded = torch.einsum('bqz,bzd->bqd', weights, zones)
        return {'occupancy_logits': self.occupancy(decoded).squeeze(-1),
                'distance_m': F.softplus(self.distance(decoded).squeeze(-1)),
                'query_zone_weights': weights}


def fit_seed(data, arm: str, seed: int, output: Path, pos_weight: np.ndarray) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda')
    model = AzimuthMaskedFrustumFusion().to(device)
    rgb = torch.from_numpy(data['rgb'].astype(np.float32) / 255).to(device)
    if arm == 'tof_only':
        rgb.zero_()
    histogram = torch.from_numpy(data['histogram']).to(device)
    ambient = torch.from_numpy(data['ambient']).to(device)
    scalar = torch.from_numpy(data['scalar']).to(device)
    valid = torch.from_numpy(data['valid']).to(device)
    age = torch.zeros_like(ambient)
    support = torch.from_numpy(data['support']).to(device)
    labels = torch.from_numpy(np.maximum(data['labels'], 0).astype(np.float32)).to(device)
    known = torch.from_numpy(data['labels'] >= 0).to(device)
    weights = torch.from_numpy(pos_weight).to(device)
    train_indices = torch.from_numpy(np.flatnonzero(data['train'])).to(device)
    encoder_parameters = list(model.rgb.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) not in encoder_ids], 'lr': 3e-4},
        {'params': encoder_parameters, 'lr': 3e-5},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    losses = []
    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(train_indices), generator=generator).to(device)
        epoch_loss = 0.0
        for start in range(0, len(train_indices), BATCH):
            indices = train_indices[order[start:start+BATCH]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(rgb[indices], histogram[indices], ambient[indices],
                           scalar[indices], valid[indices], age[indices], support,
                           mode='cnh')['occupancy_logits']
            loss_per_query = F.binary_cross_entropy_with_logits(
                logits, labels[indices], pos_weight=weights, reduction='none')
            loss = loss_per_query[known[indices]].mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(indices)
        scheduler.step()
        losses.append(epoch_loss / len(train_indices))
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(rgb), BATCH):
            indices = slice(start, start+BATCH)
            logits = model(rgb[indices], histogram[indices], ambient[indices],
                           scalar[indices], valid[indices], age[indices], support,
                           mode='cnh')['occupancy_logits']
            chunks.append(logits.cpu().numpy())
    predictions = np.concatenate(chunks)
    if predictions.shape != (960, 6) or not np.isfinite(predictions).all():
        raise ValueError('Invalid prediction shape/values')
    model_path = output / f'{arm}-seed-{seed}.pt'
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_path)
    prediction_path = output / f'{arm}-seed-{seed}-predictions.npz'
    np.savez_compressed(prediction_path, frame_key=np.array([r['frame_key'] for r in data['rows']]),
                        logits=predictions, dev=data['dev'])
    layout_metrics = {}
    for layout in sorted({r['layout_id'] for r in data['rows']}):
        selection = data['dev'] & np.array([r['layout_id'] == layout for r in data['rows']])
        if selection.any():
            layout_metrics[layout] = rates(metrics(data['labels'][selection], predictions[selection]))
    return dict(seed=seed, arm=arm, train_loss_final=losses[-1], train_losses=losses,
                dev=rates(metrics(data['labels'][data['dev']], predictions[data['dev']])),
                dev_layouts=layout_metrics, model_path=str(model_path), model_sha256=sha(model_path),
                predictions_path=str(prediction_path), predictions_sha256=sha(prediction_path))


def decision(results: list[dict]) -> dict:
    paired = {seed: {x['arm']: x['dev'] for x in results if x['seed'] == seed} for seed in SEEDS}
    if any(set(paired[seed]) != {'tof_only', 'cnh_rgb'} for seed in SEEDS):
        raise ValueError('Complete paired seeds required')
    noncollapsed = all(paired[seed][arm]['tp'] > 0 and paired[seed][arm]['tn'] > 0
                       for seed in SEEDS for arm in ('tof_only', 'cnh_rgb'))
    rgb_dominates = all(paired[seed]['cnh_rgb']['tp'] > paired[seed]['tof_only']['tp'] and
                        paired[seed]['cnh_rgb']['fp'] <= paired[seed]['tof_only']['fp']
                        for seed in SEEDS)
    return dict(mechanism='NONCOLLAPSED_DEVELOPMENT' if noncollapsed else 'COLLAPSED_IN_AT_LEAST_ONE_ARM_SEED',
                rgb_increment='CONSISTENT_DEV_PARETO_SIGNAL_ONLY' if noncollapsed and rgb_dominates
                              else 'NO_CONSISTENT_DEV_PARETO_SIGNAL',
                rule='Each arm/seed TP>0 and TN>0 for noncollapse; RGB signal only if noncollapsed and all three seeds RGB TP>ToF TP with RGB FP<=ToF FP at fixed logit threshold0')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-overlay', type=Path, required=True)
    p.add_argument('--partition-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if not torch.cuda.is_available():
        raise RuntimeError('Fixed V2 GPU run requires CUDA')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError('Fresh or empty output required')
    if sha(args.collection_overlay) != COLLECTION_SHA256 or sha(args.partition_plan) != PARTITION_SHA256:
        raise ValueError('Frozen overlay or partition plan hash differs')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data = read_inputs(args.collection_overlay, args.partition_plan)
    if tuple(data['query_names']) != QUERY_NAMES:
        raise ValueError('Query order differs from frozen mask')
    pos_weight, balance = train_pos_weight(data['labels'], data['train'])
    np.save(args.output / 'frustum-support.npy', data['support'])
    results = []
    for seed in SEEDS:
        for arm in ('tof_only', 'cnh_rgb'):
            result = fit_seed(data, arm, seed, args.output, pos_weight)
            results.append(result)
            print(json.dumps(dict(seed=seed, arm=arm, dev=result['dev'], wall_s=time.monotonic()-started)), flush=True)
    report = dict(status='COMPLETE_PROSPECTIVE_ALLEY_DEVELOPMENT_V2', benchmark_eligible=False,
                  protocol='CNH_RGB_ALLEY_DEV_V2_PROTOCOL_20260925.md',
                  frame_count=960, train_frames=480, dev_frames=480, query_names=QUERY_NAMES,
                  seeds=SEEDS, arms=('tof_only', 'cnh_rgb'), epochs=EPOCHS, batch_size=BATCH,
                  input_definition='Same six alley overlays and frozen split as V1; ToF-only black RGB, fused repaired left RGB 128x72; same H3 CNH/ambient/scalar/valid/age',
                  query_mask=dict(columns=AZIMUTH_COLUMNS, zone_order='row-major 8x8',
                                  cardinalities=query_azimuth_mask().sum(-1).tolist()),
                  class_balance=dict(source='known train labels only', formula='per query negative/positive', **balance),
                  optimization='Both arms identical FrustumFusion parameters, fixed azimuth mask, AdamW lr3e-4/RGB encoder3e-5, weight decay1e-4, cosine 24 epochs, batch16, final epoch, fixed logit threshold0',
                  distance_head='UNTRAINED_NOT_EVALUATED', split=data['split'],
                  partition_plan_sha256=sha(args.partition_plan),
                  collection_overlay=dict(path=str(args.collection_overlay), sha256=sha(args.collection_overlay)),
                  overlays=data['overlays'], materialized_sources=data['materialized_sources'],
                  frustum_cache=dict(identity='SHA256 canonical JSON exact K, T_camera_tof, native_size, rgb_size, feature_size',
                                     keys=[data['support_key']], entries=1, frame_reuses=959,
                                     support_path=str(args.output/'frustum-support.npy'),
                                     support_sha256=sha(args.output/'frustum-support.npy')),
                  code_sha256=sha(__file__), v1_loader_code_sha256=sha(Path(__file__).with_name('cnh_rgb_dev_comparison.py')),
                  model_base_code_sha256=sha(Path(__file__).with_name('cnh_rgb_fusion.py')),
                  protocol_sha256=sha(Path(__file__).with_name('CNH_RGB_ALLEY_DEV_V2_PROTOCOL_20260925.md')),
                  results=results, decision=decision(results), device=torch.cuda.get_device_name(),
                  wall_s=time.monotonic()-started,
                  claim_limit='Development only; three train and three dev alley sites with 160 correlated frames/layout and six correlated queries/frame; geometry-derived labels lack independent precision admission; no protected test, hardware, generalization, or algorithm benefit claim')
    (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], decision=report['decision'], wall_s=report['wall_s'])), flush=True)


if __name__ == '__main__':
    main()

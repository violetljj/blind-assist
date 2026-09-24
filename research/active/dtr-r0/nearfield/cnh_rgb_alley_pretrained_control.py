"""Frozen ResNet18 shallow spatial encoder control for alley V2 Development."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from cnh_rgb_alley_v2 import (AZIMUTH_COLUMNS, COLLECTION_SHA256, PARTITION_SHA256,
                              QUERY_NAMES, AzimuthMaskedFrustumFusion, masked_attention,
                              query_azimuth_mask, train_pos_weight)
from cnh_rgb_dev_comparison import BATCH, EPOCHS, SEEDS, rates, read_inputs, sha
from cnh_rgb_v2_perturbation import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, bootstrap, score
from cnh_street_e2e_train import metrics


WEIGHTS_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class FrozenResNet18Stem(nn.Module):
    """ImageNet ResNet18 through layer1, exactly frozen and always in eval mode."""

    def __init__(self, weights: Path):
        super().__init__()
        if sha(weights) != WEIGHTS_SHA256:
            raise ValueError('Frozen pretrained weight hash mismatch')
        full = resnet18(weights=None)
        full.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.stem = nn.Sequential(full.conv1, full.bn1, full.relu, full.maxpool, full.layer1)
        self.register_buffer('mean', torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 72, 128):
            raise ValueError('Expected [B,3,72,128] RGB in [0,1]')
        if not torch.isfinite(rgb).all() or torch.any((rgb < 0) | (rgb > 1)):
            raise ValueError('RGB must be finite in [0,1]')
        with torch.no_grad():
            features = self.stem((rgb - self.mean) / self.std)
        if features.shape != (rgb.shape[0], 64, 18, 32):
            raise ValueError('Pretrained spatial feature grid differs')
        return features


class PretrainedAzimuthFusion(AzimuthMaskedFrustumFusion):
    """V2 query path with a trainable 1x1 adapter on frozen spatial features."""

    def __init__(self):
        super().__init__()
        self.rgb = nn.Sequential(nn.Conv2d(64, 64, 1), nn.GELU())

    def forward(self, rgb_features, histogram, ambient, scalar_m, scalar_valid, age_s,
                support, mode='cnh'):
        if mode != 'cnh':
            raise ValueError('Control fixes CNH mode')
        if rgb_features.ndim != 4 or rgb_features.shape[1:] != (64, 18, 32):
            raise ValueError('Frozen spatial features must be [B,64,18,32]')
        batch = rgb_features.shape[0]
        if histogram.shape != (batch, 64, 16):
            raise ValueError('CNH shape mismatch')
        if any(x.shape != (batch, 64) for x in (ambient, scalar_m, scalar_valid, age_s)):
            raise ValueError('Zone metadata shape mismatch')
        if not all(torch.isfinite(x).all() for x in (rgb_features, histogram, ambient, age_s)):
            raise ValueError('Nonfinite observable input')
        if not torch.isfinite(scalar_m[scalar_valid.bool()]).all() or torch.any(age_s < 0):
            raise ValueError('Invalid scalar or age')
        features = self.rgb(rgb_features)
        if support.ndim == 3:
            support = support.unsqueeze(0).expand(batch, -1, -1, -1)
        if support.shape != (batch, 64, 18, 32) or not torch.isfinite(support).all():
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
                              valid.to(rgb_features.dtype), torch.log1p(age_s)), -1)
        token = self.metadata(public)
        token = token + self.cnh(torch.sign(histogram) * torch.log1p(histogram.abs()))
        zones = self.fuse(torch.cat((rgb_zone, token), -1)) + self.zone_position(self.zone_coordinates)[None]
        weights = masked_attention(self.query, zones, self.query_zone_mask)
        decoded = torch.einsum('bqz,bzd->bqd', weights, zones)
        return {'occupancy_logits': self.occupancy(decoded).squeeze(-1),
                'distance_m': F.softplus(self.distance(decoded).squeeze(-1)),
                'query_zone_weights': weights}


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def precompute_features(data, weights: Path, output: Path, device):
    encoder = FrozenResNet18Stem(weights).to(device)
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('Encoder must be completely frozen in eval')
    rgb = data['rgb']
    features = np.empty((len(rgb), 64, 18, 32), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rgb), BATCH):
            image = torch.from_numpy(rgb[start:start+BATCH].astype(np.float32) / 255).to(device)
            features[start:start+len(image)] = encoder(image).cpu().numpy()
        black = encoder(torch.zeros((1, 3, 72, 128), device=device)).cpu().numpy()
    if not np.isfinite(features).all() or not np.isfinite(black).all():
        raise ValueError('Nonfinite pretrained feature')
    rgb_path, black_path = output / 'resnet18-rgb-features.npy', output / 'resnet18-black-feature.npy'
    np.save(rgb_path, features)
    np.save(black_path, black)
    return features, black, dict(rgb_path=str(rgb_path), rgb_sha256=sha(rgb_path),
                                 black_path=str(black_path), black_sha256=sha(black_path),
                                 feature_shape=list(features.shape), black_shape=list(black.shape),
                                 frozen_parameters=sum(p.numel() for p in encoder.parameters()))


def fit_seed(data, features, black, arm, seed, output, pos_weight, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PretrainedAzimuthFusion().to(device)
    initial_hash = model_hash(model)
    if arm == 'tof_only':
        visual = torch.from_numpy(black).to(device).expand(len(features), -1, -1, -1)
    else:
        visual = torch.from_numpy(features).to(device)
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
    adapter = list(model.rgb.parameters())
    adapter_ids = {id(p) for p in adapter}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) not in adapter_ids], 'lr': 3e-4},
        {'params': adapter, 'lr': 3e-5},
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
            logits = model(visual[indices], histogram[indices], ambient[indices],
                           scalar[indices], valid[indices], age[indices], support)['occupancy_logits']
            per_query = F.binary_cross_entropy_with_logits(
                logits, labels[indices], pos_weight=weights, reduction='none')
            loss = per_query[known[indices]].mean()
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
        for start in range(0, len(features), BATCH):
            indices = slice(start, start+BATCH)
            chunks.append(model(visual[indices], histogram[indices], ambient[indices],
                                scalar[indices], valid[indices], age[indices], support)
                          ['occupancy_logits'].cpu().numpy())
    predictions = np.concatenate(chunks)
    if predictions.shape != (960, 6) or not np.isfinite(predictions).all():
        raise ValueError('Invalid prediction shape')
    model_path = output / f'{arm}-seed-{seed}.pt'
    prediction_path = output / f'{arm}-seed-{seed}-predictions.npz'
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_path)
    np.savez_compressed(prediction_path, frame_key=np.array([r['frame_key'] for r in data['rows']]),
                        logits=predictions, dev=data['dev'])
    return dict(seed=seed, arm=arm, initial_model_hash=initial_hash,
                train_losses=losses, train_loss_final=losses[-1],
                dev_counts=rates(metrics(data['labels'][data['dev']], predictions[data['dev']])),
                model_path=str(model_path), model_sha256=sha(model_path),
                predictions_path=str(prediction_path), predictions_sha256=sha(prediction_path)), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-overlay', type=Path, required=True)
    parser.add_argument('--partition-plan', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError('Fresh or empty output required')
    if sha(args.collection_overlay) != COLLECTION_SHA256 or sha(args.partition_plan) != PARTITION_SHA256:
        raise ValueError('Frozen input identity differs')
    if sha(args.weights) != WEIGHTS_SHA256:
        raise ValueError('Frozen pretrained weight hash differs')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this fixed run')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    data = read_inputs(args.collection_overlay, args.partition_plan)
    if tuple(data['query_names']) != QUERY_NAMES or len(data['rows']) != 960 or int(data['train'].sum()) != 480 or int(data['dev'].sum()) != 480:
        raise ValueError('Frozen alley data identity differs')
    pos_weight, balance = train_pos_weight(data['labels'], data['train'])
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    features, black, feature_receipt = precompute_features(data, args.weights, args.output, device)
    results, predictions = [], {}
    for seed in SEEDS:
        paired = []
        for arm in ('tof_only', 'cnh_rgb'):
            row, pred = fit_seed(data, features, black, arm, seed, args.output, pos_weight, device)
            results.append(row)
            predictions[(seed, arm)] = pred[data['dev']]
            paired.append(row)
            print(json.dumps(dict(seed=seed, arm=arm, dev=row['dev_counts'], elapsed_s=time.monotonic()-started)), flush=True)
        if paired[0]['initial_model_hash'] != paired[1]['initial_model_hash']:
            raise ValueError('Paired arms have different initial trainable parameters')
    del features, black
    torch.cuda.empty_cache()
    print(json.dumps(dict(status='GPU_TRAINING_COMPLETE')), flush=True)
    dev_indices = np.flatnonzero(data['dev'])
    labels = data['labels'][dev_indices]
    groups = np.array([data['rows'][i]['layout_id'] for i in dev_indices])
    scores = {}
    for seed in SEEDS:
        arm_logits = {arm: predictions[(seed, arm)] for arm in ('tof_only', 'cnh_rgb')}
        all_indices = np.arange(len(dev_indices))
        scores[str(seed)] = dict(total=bootstrap(labels, arm_logits, groups, all_indices),
                                 layouts={layout: bootstrap(labels, arm_logits, groups, all_indices[groups == layout])
                                          for layout in sorted(set(groups))})
    report = dict(status='COMPLETE_FROZEN_PRETRAINED_DEVELOPMENT_CONTROL', benchmark_eligible=False,
                  protocol='CNH_RGB_ALLEY_PRETRAINED_CONTROL_PROTOCOL_20260925.md',
                  collection=dict(path=str(args.collection_overlay), sha256=sha(args.collection_overlay)),
                  partition=dict(path=str(args.partition_plan), sha256=sha(args.partition_plan)),
                  pretrained_weights=dict(path=str(args.weights), sha256=sha(args.weights)),
                  feature_cache=feature_receipt, support_key=data['support_key'],
                  query_mask=dict(columns=AZIMUTH_COLUMNS, zone_order='row-major 8x8',
                                  cardinalities=query_azimuth_mask().sum(-1).tolist()),
                  balance=balance, train_frames=480, dev_frames=480, queries_per_frame=6,
                  seeds=SEEDS, epochs=EPOCHS, batch_size=BATCH,
                  optimization='AdamW adapter lr3e-5, remaining trainable lr3e-4, wd1e-4, cosine24; weighted occupancy BCE; final epoch',
                  bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES,
                                 unit='frame within each fixed dev layout, six queries together'),
                  results=results, ranking=scores,
                  code_sha256=sha(__file__), protocol_sha256=sha(Path(__file__).with_name('CNH_RGB_ALLEY_PRETRAINED_CONTROL_PROTOCOL_20260925.md')),
                  device=torch.cuda.get_device_name(), wall_s=time.monotonic()-started,
                  claim_limit='Development only; fixed three dev alley layouts; geometry-derived label precision not independently admitted; no protected test/hardware/generalization/safety claim')
    (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], wall_s=report['wall_s'], result_sha256=sha(args.output / 'result.json'))), flush=True)


if __name__ == '__main__':
    main()

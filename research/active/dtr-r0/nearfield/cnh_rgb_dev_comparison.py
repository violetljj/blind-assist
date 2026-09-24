"""Fixed three-seed, alley-only Development comparison of CNH and CNH+RGB.

Both arms use the same six-query FrustumFusion and the same ToF observations.
The CNH control receives black RGB; the fused arm receives the repaired RGB
bound by the immutable alley overlay. This is a layout holdout diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from cnh_rgb_frustum import zone_rgb_support
from cnh_street_e2e_partitions import planned_split
from cnh_street_e2e_train import load_inputs, metrics


SEEDS = (20260924, 20260925, 20260926)
EPOCHS = 24
BATCH = 16
RGB_SIZE = (128, 72)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(bound):
    path = Path(bound['path'])
    if not path.is_file() or sha(path) != bound['sha256']:
        raise ValueError(f'Overlay-bound input differs: {path}')
    return path


def read_inputs(collection_path, partition_path):
    from PIL import Image

    collection = json.loads(Path(collection_path).read_text(encoding='utf-8-sig'))
    if (collection.get('status') != 'PASS_DEVELOPMENT_RGB_TOF_OVERLAY_ONLY' or
            collection.get('layout_count') != 6 or collection.get('frame_count') != 960 or
            collection.get('test_collection') != 'NOT_RUN'):
        raise ValueError('Six accepted Development overlays required')
    layouts = collection['layouts']
    if len(layouts) != 6 or sorted(row['split'] for row in layouts) != ['dev'] * 3 + ['train'] * 3:
        raise ValueError('Three train and three dev layouts required')
    overlays = []
    roots = []
    for item in layouts:
        overlay_path = checked({'path': item['overlay'], 'sha256': item['overlay_sha256']})
        overlay = json.loads(overlay_path.read_text(encoding='utf-8-sig'))
        if (overlay.get('status') != 'JOINED_DEVELOPMENT_NUMERIC_RGB_VISUAL_REVIEW_REQUIRED' or
                overlay.get('data_role') != 'Development' or overlay.get('frame_count') != 160):
            raise ValueError('Development overlay status/count differs')
        for name in ('materialized_manifest', 'materialized_receipt', 'observations', 'targets'):
            checked(overlay[name])
        root = Path(overlay['materialized_manifest']['path']).parent
        if root / 'receipt.json' != Path(overlay['materialized_receipt']['path']):
            raise ValueError('Materialized receipt root differs')
        if item['layout_id'] != overlay['frames'][0]['layout_id'] or len(overlay['frames']) != 160:
            raise ValueError('Overlay layout/count differs')
        roots.append(root)
        overlays.append(overlay)
    plan = json.loads(Path(partition_path).read_text(encoding='utf-8-sig'))
    _, labels, rows, query_names, materialized_sources = load_inputs(roots, plan)
    train, dev, split = planned_split(rows, plan)
    if len(rows) != 960 or train.sum() != 480 or dev.sum() != 480:
        raise ValueError('Frozen alley-only split must have 480/480 frames')
    if len(query_names) != 6 or labels.shape != (960, 6):
        raise ValueError('Six-query labels required')

    rgb = np.empty((960, 3, RGB_SIZE[1], RGB_SIZE[0]), dtype=np.uint8)
    histogram = np.empty((960, 64, 16), dtype=np.float32)
    ambient = np.empty((960, 64), dtype=np.float32)
    scalar = np.empty((960, 64), dtype=np.float32)
    valid = np.empty((960, 64), dtype=np.bool_)
    support_cache = {}
    support_key_for_frame = []
    cursor = 0
    for item, overlay, root in zip(layouts, overlays, roots):
        with np.load(root / 'observations.npz', allow_pickle=False) as obs:
            n = len(overlay['frames'])
            histogram[cursor:cursor+n] = obs['histogram'].reshape(n, 64, 16)
            ambient[cursor:cursor+n] = obs['ambient'].reshape(n, 64)
            scalar[cursor:cursor+n] = obs['distance_m'].reshape(n, 64)
            valid[cursor:cursor+n] = obs['valid'].reshape(n, 64)
            if list(obs['frame_key'].astype(str)) != [r['frame_key'] for r in overlay['frames']]:
                raise ValueError('Overlay observation keys differ')
        for local, (frame, materialized_row) in enumerate(zip(overlay['frames'], rows[cursor:cursor+n])):
            if (frame['frame_key'] != materialized_row['frame_key'] or
                    frame['layout_id'] != item['layout_id'] or
                    materialized_row['layout_id'] != item['layout_id']):
                raise ValueError('Overlay/materialized frame identity differs')
            if bool(train[cursor + local]) != (item['split'] == 'train'):
                raise ValueError('Frozen partition differs from overlay split')
            camera_path = checked(frame['original_files']['camera.json'])
            camera = json.loads(camera_path.read_text(encoding='utf-8-sig'))
            if (camera['width'], camera['height']) != (640, 360):
                raise ValueError('Unexpected camera image size')
            calibration = dict(K=camera['K'], T_camera_tof=camera['T_camera_tof'],
                               native_size=[camera['width'], camera['height']], rgb_size=RGB_SIZE,
                               feature_size=[RGB_SIZE[1] // 4, RGB_SIZE[0] // 4])
            cache_key = hashlib.sha256(json.dumps(calibration, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if cache_key not in support_cache:
                k = np.asarray(camera['K'], dtype=np.float64).copy()
                sx, sy = RGB_SIZE[0] / camera['width'], RGB_SIZE[1] / camera['height']
                k[0, 0] *= sx
                k[1, 1] *= sy
                k[0, 2] = (k[0, 2] + .5) * sx - .5
                k[1, 2] = (k[1, 2] + .5) * sy - .5
                tof_from_camera = np.linalg.inv(np.asarray(camera['T_camera_tof'], dtype=np.float64))
                support = zone_rgb_support(k, tof_from_camera, RGB_SIZE,
                                           (RGB_SIZE[1] // 4, RGB_SIZE[0] // 4))
                if not np.all(support.sum((1, 2)) > 0):
                    raise ValueError('Zone has no RGB support')
                support_cache[cache_key] = support
            support_key_for_frame.append(cache_key)
            rgb_path = checked(frame['repaired_rgb']['left.png'])
            with Image.open(rgb_path) as image:
                if image.size != (640, 360):
                    raise ValueError('Overlay RGB size differs')
                resized = image.convert('RGB').resize(RGB_SIZE, Image.Resampling.BILINEAR)
                rgb[cursor + local] = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)
        cursor += n
    if cursor != 960 or not np.isfinite(histogram).all() or not np.isfinite(ambient).all():
        raise ValueError('Incomplete or nonfinite observations')
    if not np.isfinite(scalar[valid]).all() or len(set(support_key_for_frame)) != 1:
        raise ValueError('Invalid scalar or inconsistent camera calibration')
    support_key = support_key_for_frame[0]
    return dict(rgb=rgb, histogram=histogram, ambient=ambient, scalar=scalar, valid=valid,
                support=support_cache[support_key], support_key=support_key,
                labels=labels, rows=rows, train=train, dev=dev, split=split, query_names=query_names,
                materialized_sources=materialized_sources, collection=collection,
                overlays=[dict(path=item['overlay'], sha256=item['overlay_sha256']) for item in layouts])


def rates(counts):
    positives = counts['tp'] + counts['fn']
    negatives = counts['fp'] + counts['tn']
    return dict(**counts, positive_query_denominator=positives, negative_query_denominator=negatives,
                recall=counts['tp'] / positives if positives else None,
                false_alert_rate=counts['fp'] / negatives if negatives else None)


def fit_seed(data, arm, seed, output):
    import torch
    from torch.nn import functional as F
    from cnh_rgb_fusion import FrustumFusion

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda')
    model = FrustumFusion(64, 16, width=64, queries=6).to(device)
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
            loss = F.binary_cross_entropy_with_logits(logits[known[indices]], labels[indices][known[indices]])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-overlay', type=Path, required=True)
    parser.add_argument('--partition-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Fixed GPU comparison requires CUDA')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError('Fresh or empty output required')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data = read_inputs(args.collection_overlay, args.partition_plan)
    np.save(args.output / 'frustum-support.npy', data['support'])
    results = []
    for seed in SEEDS:
        for arm in ('tof_only', 'cnh_rgb'):
            result = fit_seed(data, arm, seed, args.output)
            results.append(result)
            print(json.dumps(dict(seed=seed, arm=arm, dev=result['dev'], wall_s=time.monotonic()-started)), flush=True)
    summary = {}
    for arm in ('tof_only', 'cnh_rgb'):
        subset = [result['dev'] for result in results if result['arm'] == arm]
        summary[arm] = {key: dict(mean=float(np.mean([x[key] for x in subset])),
                                  minimum=float(np.min([x[key] for x in subset])),
                                  maximum=float(np.max([x[key] for x in subset])))
                        for key in ('tp', 'fp', 'fn', 'tn', 'recall', 'false_alert_rate')}
    report = dict(status='PASS_ALLEY_ONLY_DEVELOPMENT_COMPARISON', benchmark_eligible=False,
                  street_rgb_limit='Street worker thin copy has only 8 of 960 left RGB frames; full frozen Street+alley comparison not run',
                  frame_count=960, train_frames=480, dev_frames=480, query_names=data['query_names'],
                  seeds=SEEDS, arms=('tof_only', 'cnh_rgb'), epochs=EPOCHS, batch_size=BATCH,
                  optimization='AdamW lr 3e-4 / RGB encoder lr 3e-5, weight decay 1e-4, cosine 24 epochs; masked occupancy BCE; final epoch; logit threshold 0',
                  input_definition='Both arms same FrustumFusion architecture and CNH/ambient/scalar/valid/age; ToF-only RGB exactly zero; fused RGB repaired native left images resized to 128x72',
                  distance_head='UNTRAINED_NOT_EVALUATED',
                  split=data['split'], partition_plan_sha256=sha(args.partition_plan),
                  collection_overlay=dict(path=str(args.collection_overlay), sha256=sha(args.collection_overlay)),
                  overlays=data['overlays'], materialized_sources=data['materialized_sources'],
                  frustum_cache=dict(identity='SHA256 canonical JSON exact K, T_camera_tof, native_size, rgb_size, feature_size',
                                     keys=[data['support_key']], entries=1, frame_reuses=959,
                                     support_path=str(args.output/'frustum-support.npy'),
                                     support_sha256=sha(args.output/'frustum-support.npy')),
                  code_sha256=sha(__file__), fusion_code_sha256=sha(Path(__file__).with_name('cnh_rgb_fusion.py')),
                  frustum_code_sha256=sha(Path(__file__).with_name('cnh_rgb_frustum.py')),
                  device=torch.cuda.get_device_name(), wall_s=time.monotonic()-started,
                  dev_results=results, seed_summary=summary,
                  claim_limit='Development layout holdout only: 3 train and 3 dev alley physical sites; 160 correlated frames per layout, six correlated queries per frame; geometry-derived labels lack independent precision admission; no protected test, hardware, generalization, or algorithm benefit claim')
    (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], seed_summary=summary, wall_s=report['wall_s'])), flush=True)


if __name__ == '__main__':
    main()

"""Frozen features on existing relation/distance TRAIN only; no fitting."""
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from mz5_ensemble_readout import CompactEnsemble, load_npz, read, write, sha
from body_query_context_evidence import ContextEvidence
from body_query_fresh_size_eval import FROZEN
from body_query_range import range_from_counts
from multizone64_observation import native_events
from mz9_contributors import reconstruct
from mz9_source_readout import SourceReadout
from mz11_selective_addition import gate_features


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    work = root/'artifacts.local/work'
    rows, inputs = [], {}
    for dataset, relative, expected in [('relation', 'body-query-10000-20260909/final-dataset-v2', 5000),
                                        ('distance', 'body-query-distance-5000-20260909/final-dataset-v1', 2500)]:
        folder = work/relative
        manifest = read(folder/'manifest.json')
        assert manifest['status'] == 'PASS'
        for name in ['manifest.json', 'index.json', 'summary.json']:
            digest = sha(folder/name)
            inputs[str(folder/name)] = digest
            if name != 'manifest.json':
                assert digest == manifest[name.split('.')[0]+'_sha256']
        records = read(folder/'index.json')['frames' if dataset == 'relation' else 'records']
        selected = []
        for index, r in enumerate(records):
            distance = dataset == 'distance'
            if (r.get('source_partition') != 'train' if distance else r.get('source_role') != 'TRAIN_ONLY'):
                continue
            assert r['accepted'] if distance else r['status'] == 'PASS'
            selected.append(dict(dataset=dataset, index=index, role='TRAIN_ONLY', camera=r['camera'],
                site=r['site_id'], group=r['pair_id' if distance else 'group_id'], family=r['family'],
                condition='HEAD_ONLY' if distance else r['condition'],
                rgb=r['rgb_path' if distance else 'rgb_file'], rgb_sha=r['rgb_sha256'],
                native=r['native_path' if distance else 'native_file'], native_sha=r['native_sha256']))
        assert len(selected) == expected
        rows.extend(selected)
    assert len(rows) == 7500
    write(output/'selected.json', rows)
    base = work/'body-query-10000-b-20260909/run-v1'
    decoder = work/'body-query-context-decoder-20260909/run-v1'
    run9 = work/'mz9-source-supervision-20260910/run-v1'
    checkpoint = work/'mz5-fixed-ensemble-20260910/compact-v1/compact.pt'
    frozen = {}
    for name, digest in FROZEN.items():
        path = (base if name == 'NEW-step2000.pt' else decoder)/name
        assert sha(path) == digest
        frozen[str(path)] = digest
    assert sha(checkpoint) == 'ccbfd6817bfec226f0db74e7ff4b22ce4f153255a9102d5c8684de91197486e1'
    frozen[str(checkpoint)] = sha(checkpoint)
    r9 = read(run9/'receipt.json')
    assert r9['status'] == 'PASS'
    inputs[str(run9/'receipt.json')] = sha(run9/'receipt.json')
    for name in ['SOURCE_RGB.pt', 'normalization.npz', 'result.json']:
        assert sha(run9/name) == r9['outputs'][name]
        frozen[str(run9/name)] = sha(run9/name)
    source_threshold = np.array(read(run9/'result.json')['dev']['SOURCE_RGB']['threshold'])
    n = load_npz(run9/'normalization.npz')
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    context = ContextEvidence(base, decoder, work/'body-query-v1-20260908/model-inputs/pretrained').cuda().eval()
    m = context.base
    source = SourceReadout().cuda().eval()
    source.load_state_dict(torch.load(run9/'SOURCE_RGB.pt', weights_only=True))
    fixed = CompactEnsemble.from_checkpoint(checkpoint).cuda()
    norm_mean, norm_std = torch.from_numpy(n['mean']).cuda(), torch.from_numpy(n['std']).cuda()
    write(output/'start.json', dict(status='STARTED', frames=7500, training_steps=0,
        selection_sha256=sha(output/'selected.json'), source_sha256=sha(Path(__file__)),
        source_thresholds=source_threshold.tolist(), scope='Existing original TRAIN_ONLY; no DEV/EVAL pixels'))
    arrays = {}
    for begin in range(0, len(rows), 16):
        batch = rows[begin:begin+16]
        images, depths = [], []
        for row in batch:
            assert row['role'] == 'TRAIN_ONLY'
            assert sha(row['rgb']) == row['rgb_sha'] and sha(row['native']) == row['native_sha']
            assert abs(row['camera']['pitch']) < 1e-6 and abs(row['camera']['roll']) < 1e-6
            with Image.open(row['rgb']) as im:
                assert im.size == (640, 360)
                images.append(np.array(im.convert('RGB').resize((256, 144), Image.Resampling.BOX)))
            depth = np.load(row['native'], allow_pickle=False)
            assert depth.shape == (360, 640)
            depths.append(depth)
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).cuda().float()/255
        depth = torch.from_numpy(np.stack(depths)).cuda()
        with torch.inference_mode():
            deep, shallow = m.extract((x-m.image_mean)/m.image_std)
            deep = F.interpolate(m.deep_projection(deep), size=(18, 32), mode='bilinear', align_corners=False)
            dense = torch.cat([deep, m.detail(shallow)], 1)
            sampled = (dense.flatten(2)@m.query_projection.T).transpose(1, 2).reshape(len(batch), 12, 27, 64)
            mask = m.query_valid[None, :, :, None]
            pooled = (sampled*mask).sum(2)/mask.sum(2).clamp_min(1)
            normalized = (pooled-context.feature_mean)/context.feature_std
            visual = torch.cat([normalized.flatten(1), range_from_counts(context.decoder(normalized)).sigmoid().flatten(1)], 1)
            packet = reconstruct(depth)
            ranges, valid = torch.nan_to_num(packet['range_m']).float(), packet['valid']
            tof = torch.cat([ranges.flatten(1)/4, valid.flatten(1).float()], 1)
            baseline = fixed(visual, tof).cpu().numpy()
            out = source.inspect((dense-norm_mean)/norm_std, ranges, valid)
            available = out['support'].cpu().numpy()
            margin = np.where(available, out['logits'].cpu().numpy()-source_threshold, -1e6)
            e = out['eligible']
            count = e.sum((1, 2, 3)).cpu().numpy()
            zones = e.any(3).any(2).sum(1).cpu().numpy()
            returns = e.any(3).sum((1, 2)).cpu().numpy()
            features = gate_features(np.where(available, margin, 0), count, zones, returns)
            truth = native_events(depth, crop=False)['events'].cpu().numpy()
        values = dict(features=features, baseline=baseline, source=margin, available=available,
            eligible=(baseline < 0) & (margin >= 0) & available, truth=truth,
            count=count, zones=zones, returns=returns)
        for key, value in values.items():
            arrays.setdefault(key, []).append(value)
        if begin % 320 == 0:
            print('TRAIN_FEATURES', begin+len(batch), '/7500', flush=True)
            write(output/'progress.json', dict(frames=begin+len(batch), total=7500))
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    arrays['dataset'] = np.array([r['dataset'] for r in rows])
    arrays['source_index'] = np.array([r['index'] for r in rows])
    assert arrays['features'].shape == (7500, 4, 4)
    np.savez_compressed(output/'features.npz', **arrays)
    torch.cuda.synchronize()
    write(output/'receipt.json', dict(status='PASS', frames=7500, relation_train=5000, distance_train=2500,
        training_steps=0, threshold_changes=0, backend='CUDA', device=torch.cuda.get_device_name(),
        seconds=time.perf_counter()-start, source_sha256=sha(Path(__file__)), inputs=inputs,
        frozen_hashes=frozen, selected_sha256=sha(output/'selected.json'), source_thresholds=source_threshold.tolist(),
        files={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        scope='Original TRAIN_ONLY different existing configurations; native supplies evaluator labels only; no independent confirmation'))
    print('PASS', 7500, time.perf_counter()-start, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    main(args.root, args.output)

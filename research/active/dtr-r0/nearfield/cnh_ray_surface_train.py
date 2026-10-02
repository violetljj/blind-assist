"""Paired native-CNH ray depth/visibility bottleneck, single-seed prototype.

BCEO never opens depth/valid targets. The alarm head only sees public-query
integrals, visibility/depth summaries, query embedding and frozen M3 logits.
"""
import argparse
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import cnh_surface_distribution_train as ST
import cnh_cvr_pilot as CP
import cnh_margin_confirm as MC
import cnh_near_range as NR
import cnh_margin_labels as ML

OUT = ST.SS.WORK / 'cnh-ray-surface-20261002'
RUN_ID = 'CNH_RAY_SURFACE_20261002'
ARMS = ('BCEO', 'RAY')
SPLITS, FRAMES = ST.SPLITS, ST.FRAMES
EPOCHS, BATCH, LR, WD = 10, 64, .001, .0001
RADIUS, BETA, SOFT = .0375348*128, .01, .01
sha, read, create_json = ST.sha, ST.read, ST.create_json


def plan_sha():
    p = read(OUT / 'PLAN.json')
    fixed = dict(run=RUN_ID, training_units=SPLITS['train'], calibration_units=SPLITS['calib'],
                 evaluation_units=SPLITS['evaluation'], decision_frames=FRAMES.tolist(),
                 arms=['M3', *ARMS], train_arms=list(ARMS), seeds=[0], epochs=EPOCHS,
                 batch_size=BATCH, learning_rate=LR, weight_decay=WD, loss_ray_weight=1.,
                 smooth_l1_beta=BETA, soft_boundary_m=SOFT)
    if any(p.get(k) != v for k, v in fixed.items()):
        raise ValueError('PLAN differs from fixed ray-surface recipe')
    if RUN_ID not in (Path(__file__).parent.parent / 'RUNS.md').read_text(encoding='utf8'):
        raise ValueError('Missing parent RUNS registration')
    return sha(OUT / 'PLAN.json')


def compact_membership(points, low, high):
    """Smoothstep with compact support, based on minimum six-face margin."""
    margin = torch.minimum(points-low, high-points).amin(-1)
    t = (.5+margin/(2*SOFT)).clamp(0, 1)
    return t.square()*(3-2*t)


def micro_integral(values, weights):
    """global128 y/x -> 16x16 microcells; each cell sums 8x8 ray weights."""
    shape = values.shape[:-2]
    return (values*weights).reshape(*shape, 16, 8, 16, 8).sum(-1).sum(-2)


class RaySurface(nn.Module):
    def __init__(self):
        super().__init__()
        from cnh_ray_surface_data import public_rays, QUERY_LOW, QUERY_HIGH
        rays, weights = public_rays()
        if rays.shape != (128, 128, 3) or weights.shape != (128, 128):
            raise ValueError('Public global ray/weight axes mismatch')
        self.register_buffer('rays', torch.as_tensor(np.array(rays), dtype=torch.float32))
        self.register_buffer('weights', torch.as_tensor(np.array(weights), dtype=torch.float32))
        self.register_buffer('query_low', torch.as_tensor(np.array(QUERY_LOW), dtype=torch.float32))
        self.register_buffer('query_high', torch.as_tensor(np.array(QUERY_HIGH), dtype=torch.float32))
        self.conv1 = nn.Conv3d(8, 32, 3, padding=1)
        self.pose = nn.Linear(104, 32)
        self.conv2 = nn.Conv3d(32, 32, 3, padding=1)
        self.ray_head = nn.Sequential(nn.Conv2d(512, 64, 1), nn.GELU(), nn.Conv2d(64, 512, 1))
        self.query_embedding = nn.Embedding(2, 8)
        self.alarm = nn.Sequential(nn.Linear(777, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.alarm[-1].weight)
        nn.init.zeros_(self.alarm[-1].bias)

    def predict_rays(self, native_log, pose104):
        h = F.gelu(self.conv1(native_log) + self.pose(pose104)[:, :, None, None, None])
        h = F.gelu(self.conv2(h))
        # Each input bin channel c/r remains associated with coarse zone y/x.
        h = h.permute(0, 1, 4, 2, 3).reshape(len(h), 512, 8, 8)
        out = F.pixel_shuffle(self.ray_head(h), 16)
        return out[:, 0].sigmoid(), out[:, 1]

    def alarm_features(self, depth_norm, vis_prob, transform, m3_raw):
        points_sensor = self.rays[None]*(depth_norm*RADIUS)[..., None]
        points = torch.einsum('bij,byxj->byxi', transform[:, :3, :3], points_sensor) + transform[:, None, None, :3, 3]
        query_mass = []
        for q in range(2):
            member = compact_membership(points, self.query_low[q], self.query_high[q])
            query_mass.append(micro_integral(member*vis_prob, self.weights).flatten(1))
        query_mass = torch.stack(query_mass, 1)
        visibility = micro_integral(vis_prob, self.weights).flatten(1)
        depth_mass = micro_integral(vis_prob*depth_norm, self.weights).flatten(1)
        features = torch.cat((query_mass, visibility[:, None].expand(-1, 2, -1),
                              depth_mass[:, None].expand(-1, 2, -1),
                              self.query_embedding.weight[None].expand(len(depth_norm), -1, -1),
                              m3_raw.detach()[..., None]), -1)
        return features

    def forward(self, native_log, pose104, sensor_to_travel, m3_raw):
        depth_norm, vis_logits = self.predict_rays(native_log, pose104)
        features = self.alarm_features(depth_norm, vis_logits.sigmoid(), sensor_to_travel, m3_raw)
        alarm = m3_raw.detach()+self.alarm(features).squeeze(-1)
        return alarm, (depth_norm, vis_logits)


class NativeRayInputs:
    """Native z1 + causal noisy relative poses + public current rigid transform.

    No query-weight cube or geometric target is constructed/opened here.
    """
    def __init__(self, split, units=None):
        if split not in SPLITS:
            raise ValueError('Unknown split')
        self.units = list(SPLITS[split] if units is None else units)
        if not self.units or len(set(self.units)) != len(self.units) or set(self.units)-set(SPLITS[split]):
            raise ValueError('Invalid native unit selection')
        self.configs = 22 if split == 'train' else 40
        self.raw, self.poses, self.transforms, self.input_sha256 = {}, {}, {}, {}
        root = NR.OUT if split == 'train' else MC.OUT
        for unit in self.units:
            path = root / 'features' / split / f'unit{unit}.npz'
            self.input_sha256[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                raw = z['z1']
                expected = [(c, f) for c in range(self.configs) for f in range(16)]
                if list(zip(z['scene'].tolist(), z['frame'].tolist())) != expected or raw.shape != (self.configs*16, 8, 8, 16) or not np.isfinite(raw).all():
                    raise ValueError('Native input identity/axes mismatch')
                self.raw[unit] = raw.reshape(self.configs, 16, 8, 8, 16).astype(np.float16)
            for config in range(self.configs):
                sensor, travel, noisy = CP.motion_metadata(unit, config)
                self.poses[unit, config] = noisy
                current = (np.linalg.inv(travel)@sensor)[FRAMES]
                key = unit % 3
                if key not in self.transforms:
                    self.transforms[key] = current
                elif not np.allclose(current, self.transforms[key], rtol=0, atol=1e-12):
                    raise ValueError('Public current transforms differ within mode')

    def batch(self, units, configs, frames):
        if not (len(units) == len(configs) == len(frames)):
            raise ValueError('Batch identity lengths mismatch')
        v = [ST.window_inputs(self.raw[int(u)][int(c)], self.poses[int(u), int(c)], int(f))
             for u, c, f in zip(units, configs, frames)]
        return (np.stack([a[0] for a in v]), np.stack([a[1] for a in v]),
                np.stack([self.transforms[int(u) % 3][int(f)-3] for u, f in zip(units, frames)]).astype(np.float32))

    def close(self):
        self.raw.clear()
        self.poses.clear()
        self.transforms.clear()


class RayLabels:
    def __init__(self, digest, observations):
        self.maps = {}
        aggregate = read(OUT / 'labels_receipt.json')
        if aggregate['status'] != 'COMPLETE' or aggregate['plan_sha256'] != digest:
            raise ValueError('Ray label aggregate is incomplete/incorrect')
        for unit in SPLITS['train']:
            folder = OUT / 'labels/train' / f'unit{unit}'
            receipt = read(folder / 'receipt.json')
            if receipt['status'] != 'COMPLETE' or receipt['plan_sha256'] != digest or receipt['unit'] != unit:
                raise ValueError('Ray label unit identity mismatch')
            if aggregate['unit_receipt_sha256'][f'train/unit{unit}/receipt.json'] != sha(folder / 'receipt.json'):
                raise ValueError('Aggregate/unit receipt binding mismatch')
            obs = NR.OUT / 'features/train' / f'unit{unit}.npz'
            if receipt['observation_sha256'] != observations[str(obs)]:
                raise ValueError('Ray label/native observation mismatch')
            for name in ('depth.npy', 'valid.npy', 'metadata.npz'):
                if receipt['output_sha256'][name] != sha(folder / name):
                    raise ValueError('Ray label file hash mismatch')
            with np.load(folder / 'metadata.npz', allow_pickle=False) as z:
                if not np.array_equal(z['configs'], np.arange(22)) or not np.array_equal(z['frames'], FRAMES):
                    raise ValueError('Ray label row axes mismatch')
            depth = np.load(folder / 'depth.npy', mmap_mode='r')
            valid = np.load(folder / 'valid.npy', mmap_mode='r')
            if depth.shape != (22, 13, 128, 128) or valid.shape != depth.shape or depth.dtype != np.float16 or valid.dtype != np.bool_:
                raise ValueError('Ray label shape/dtype mismatch')
            self.maps[unit] = (depth, valid)

    def batch(self, units, configs, frames):
        depth = np.stack([self.maps[int(u)][0][int(c), int(f)-3] for u, c, f in zip(units, configs, frames)]).astype(np.float32)
        valid = np.stack([self.maps[int(u)][1][int(c), int(f)-3] for u, c, f in zip(units, configs, frames)])
        if not np.isfinite(depth).all() or np.any((depth < 0) | (depth > 1)) or np.any(depth[~valid] != 0):
            raise ValueError('Ray target range/invalid mask mismatch')
        return depth, valid

    def close(self):
        for values in self.maps.values():
            for x in values:
                x._mmap.close()
        self.maps.clear()


def auxiliary_loss(depth_norm, vis_logits, target_depth, target_valid):
    valid = target_valid.to(depth_norm.dtype)
    visible_loss = F.binary_cross_entropy_with_logits(vis_logits, valid)
    element = F.smooth_l1_loss(depth_norm, target_depth, beta=BETA, reduction='none')
    depth_loss = (element*valid).sum()/valid.sum().clamp_min(1)
    return visible_loss+depth_loss, visible_loss, depth_loss


def retained_baseline(rows):
    """Reuse exact prior five-M3 logits; there is no inference implementation."""
    receipt = ST.OUT / 'm3_train_receipt.json'
    r = read(receipt)
    if r['status'] != 'COMPLETE' or r['units'] != SPLITS['train'] or r['frames'] != FRAMES.tolist():
        raise ValueError('Retained M3 baseline identity mismatch')
    path = ST.OUT / 'm3_train_scores.npz'
    if sha(path) != r['output_sha256'][path.name]:
        raise ValueError('Retained M3 logits changed')
    # Recheck small original model/source/metadata identities. The old voxel
    # hashes are provenance only; this task neither reloads nor re-infers voxels.
    for original, expected in r['input_sha256'].items():
        p = Path(original)
        if p.suffix != '.npy' and sha(p) != expected:
            raise ValueError('Retained M3 source/model/metadata changed')
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {str(u) for u in SPLITS['train']}:
            raise ValueError('Retained M3 unit keys mismatch')
        values = {u: z[str(u)] for u in SPLITS['train']}
    if any(v.shape != (22, 13, 2) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Retained M3 raw score axes mismatch')
    baseline = np.stack([values[int(u)][int(c), int(f)-3] for u, c, f in zip(rows['unit'], rows['config'], rows['frame'])])
    return baseline, {str(receipt): sha(receipt), str(path): sha(path)}


def train():
    digest = plan_sha()
    runtime = ST.cuda_runtime()
    data = NativeRayInputs('train')
    ray_labels = net = opt = scheduler = None
    try:
        rows, labels = ST.training_rows()
        baseline, baseline_hashes = retained_baseline(rows)
        import cnh_ray_surface_data as RD
        request = dict(plan_sha256=digest, source_sha256={str(p): sha(p) for p in
                       [Path(__file__), Path(RD.__file__), Path(ST.__file__), Path(CP.__file__)]},
                       native_sha256=data.input_sha256, baseline_sha256=baseline_hashes,
                       labels_receipt_sha256=sha(OUT / 'labels_receipt.json'),
                       original_M3_labels_sha256=sha(ML.OUT / 'train_labels.npz'), runtime=runtime,
                       recipe=dict(epochs=EPOCHS, batch_size=BATCH, learning_rate=LR, weight_decay=WD,
                                   optimizer='AdamW', scheduler='CosineAnnealingLR T_max=10', seed=0,
                                   main='original M3 frame BCE', auxiliary='all-ray valid BCE + valid-only normalized-depth smooth L1',
                                   auxiliary_weight=1, beta=BETA, soft_boundary_m=SOFT, max_radius_m=RADIUS,
                                   BCEO_geometric_labels_read=False, head='777->64 GELU->1; zero final layer'))
        req = OUT / 'training_request.json'
        if req.exists():
            if read(req) != request:
                raise ValueError('Frozen ray training request changed')
        else:
            create_json(req, request)
        rd = sha(req)
        done = OUT / 'training_receipt.json'
        if done.exists():
            result = read(done)
            if result['status'] != 'COMPLETE' or result['request_sha256'] != rd or any(sha(OUT / p) != v for p, v in result['models_sha256'].items()):
                raise ValueError('Completed ray training changed')
            return result
        histories, models, initial_hashes = {}, {}, {}
        for arm in ARMS:
            if arm == 'RAY':
                ray_labels = RayLabels(digest, data.input_sha256)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            rng = np.random.default_rng(0)
            net = RaySurface().cuda()
            initial_hashes[arm] = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            folder = OUT / 'checkpoints' / arm / 'seed0'
            folder.mkdir(parents=True, exist_ok=True)
            last = ST.load_epoch(folder, rd)
            history, start = [], 0
            if last is not None:
                if last['arm'] != arm:
                    raise ValueError('Checkpoint arm mismatch')
                net.load_state_dict(last['model'], strict=True)
                opt.load_state_dict(last['optimizer'])
                scheduler.load_state_dict(last['scheduler'])
                rng.bit_generator.state = last['numpy_rng']
                torch.set_rng_state(last['torch_rng'].cpu())
                torch.cuda.set_rng_state_all([v.cpu() for v in last['cuda_rng']])
                history, start = last['history'], last['epoch']
            for epoch in range(start, EPOCHS):
                began = time.perf_counter()
                order = rng.permutation(len(labels))
                losses = np.zeros(4)
                net.train()
                for offset in range(0, len(order), BATCH):
                    ids = order[offset:offset+BATCH]
                    u, c, f = (rows[k][ids] for k in ('unit', 'config', 'frame'))
                    x, pose, transform = [torch.as_tensor(v, device='cuda') for v in data.batch(u, c, f)]
                    base = torch.as_tensor(baseline[ids], device='cuda')
                    target = torch.as_tensor(labels[ids], device='cuda')
                    opt.zero_grad(set_to_none=True)
                    alarm, (depth, vis) = net(x, pose, transform, base)
                    main = F.binary_cross_entropy_with_logits(alarm, target)
                    if arm == 'RAY':
                        td, tv = [torch.as_tensor(v, device='cuda') for v in ray_labels.batch(u, c, f)]
                        aux, valid_loss, depth_loss = auxiliary_loss(depth, vis, td, tv)
                    else:
                        aux = valid_loss = depth_loss = main.new_zeros(())
                    loss = main+aux
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite ray training loss')
                    loss.backward()
                    opt.step()
                    losses += np.asarray([loss.item(), main.item(), valid_loss.item(), depth_loss.item()])*len(ids)
                scheduler.step()
                entry = dict(epoch=epoch+1, loss=float(losses[0]/len(labels)), main_loss=float(losses[1]/len(labels)),
                             valid_loss=float(losses[2]/len(labels)), depth_loss=float(losses[3]/len(labels)),
                             permutation_sha256=hashlib.sha256(order.tobytes()).hexdigest(), elapsed_s=time.perf_counter()-began)
                history.append(entry)
                payload = dict(epoch=epoch+1, arm=arm, request_sha256=rd, model=net.state_dict(), optimizer=opt.state_dict(),
                               scheduler=scheduler.state_dict(), numpy_rng=rng.bit_generator.state,
                               torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), history=history)
                path = folder / f'epoch{epoch+1}.pt'
                with path.open('xb') as stream:
                    torch.save(payload, stream)
                create_json(path.with_suffix('.json'), dict(status='COMPLETE', request_sha256=rd, epoch=epoch+1, sha256=sha(path)))
                print(arm, entry, flush=True)
            final = OUT / 'models' / arm / 'model_seed0.pt'
            final.parent.mkdir(parents=True, exist_ok=True)
            state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            if not all(bool(torch.isfinite(v).all()) for v in state.values()):
                raise FloatingPointError('Nonfinite deployment parameters')
            if final.exists():
                old = torch.load(final, map_location='cpu', weights_only=True)
                if old.keys() != state.keys() or any(not torch.equal(old[k], v) for k, v in state.items()):
                    raise ValueError('Existing deployment differs from final epoch')
            else:
                with final.open('xb') as stream:
                    torch.save(state, stream)
            models[str(final.relative_to(OUT)).replace('\\', '/')] = sha(final)
            histories[arm] = history
            net = opt = scheduler = last = payload = None
            gc.collect()
            torch.cuda.empty_cache()
        if initial_hashes['BCEO'] != initial_hashes['RAY'] or [v['permutation_sha256'] for v in histories['BCEO']] != [v['permutation_sha256'] for v in histories['RAY']]:
            raise ValueError('Paired initialization or order mismatch')
        if plan_sha() != digest or sha(req) != rd or any(sha(p) != v for p, v in request['source_sha256'].items()):
            raise ValueError('Frozen training sources changed')
        result = dict(status='COMPLETE', plan_sha256=digest, request_sha256=rd, models_sha256=models,
                      source_sha256=request['source_sha256'], labels_receipt_sha256=request['labels_receipt_sha256'],
                      baseline_sha256=baseline_hashes, runtime=runtime, recipe=request['recipe'], history=histories,
                      paired_initialization_exact=True, paired_shuffles_exact=True, seeds=[0],
                      semantics='Visibility labels mean first geometric return in radial domain, not low-SNR/device UNKNOWN. Zero predicted visibility is explicit to the head and is not declared clear.',
                      limit='One new-branch seed; existing consumed Development; no seed stability or hardware evidence')
        create_json(done, result)
        return result
    finally:
        data.close()
        if ray_labels is not None:
            ray_labels.close()
        net = opt = scheduler = None
        gc.collect()
        torch.cuda.empty_cache()


def check(gpu=False):
    """Synthetic geometry/model mechanics only, no cohort files or labels."""
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device, batch = ('cuda', 64) if gpu else ('cpu', 2)
    if gpu:
        ST.cuda_runtime()
        torch.cuda.reset_peak_memory_stats()
    net = RaySurface().to(device)
    # PixelShuffle channel=(output*256+suby*16+subx) is global angular layout.
    x = torch.arange(512*8*8).reshape(1, 512, 8, 8)
    shuffled = F.pixel_shuffle(x, 16)
    for channel, zy, zx, sy, sx in ((0, 0, 0, 0, 0), (1, 3, 5, 7, 11), (1, 7, 7, 15, 15)):
        assert shuffled[0, channel, zy*16+sy, zx*16+sx] == x[0, channel*256+sy*16+sx, zy, zx]
    from cnh_surface_distribution_data import public_rays as old_rays
    rays_micro, weights_micro = old_rays()
    for y, xx in ((0, 0), (53, 85), (127, 127)):
        np.testing.assert_allclose(net.rays[y, xx].cpu(), rays_micro[y//8, xx//8, (y%8)*8+xx%8], atol=3e-8, rtol=0)
        assert abs(float(net.weights[y, xx])-weights_micro[y//8, xx//8, (y%8)*8+xx%8]) < 1e-8
    torch.testing.assert_close(micro_integral(torch.ones(1, 128, 128, device=device), net.weights),
                               torch.ones(1, 16, 16, device=device), rtol=0, atol=2e-7)
    point = torch.tensor([[0., 0., 1.], [.32, 0., 1.], [.3, 0., 1.]], device=device)
    torch.testing.assert_close(compact_membership(point, net.query_low[0], net.query_high[0]),
                               torch.tensor([1., 0., .5], device=device), rtol=0, atol=1e-6)
    raw = np.arange(16*8*8*16, dtype=np.float32).reshape(16, 8, 8, 16)
    poses = np.repeat(np.eye(4)[None], 16, axis=0)
    a, p = ST.window_inputs(raw, poses, 3)
    raw[4:] = -123
    poses[4:, 0, 3] = 999
    b, q = ST.window_inputs(raw, poses, 3)
    assert np.array_equal(a, b) and np.array_equal(p, q) and not a[:4].any()
    assert np.array_equal(p[-8:], [0, 0, 0, 0, 1, 1, 1, 1])
    native = torch.randn(batch, 8, 8, 8, 16, device=device)
    pose = torch.randn(batch, 104, device=device)
    transform = torch.eye(4, device=device)[None].repeat(batch, 1, 1)
    base = torch.randn(batch, 2, device=device, requires_grad=True)
    began = time.perf_counter()
    pred, (depth, vis) = net(native, pose, transform, base)
    assert depth.shape == vis.shape == (batch, 128, 128) and bool(((depth > 0) & (depth < 1)).all())
    torch.testing.assert_close(pred, base, rtol=0, atol=0)
    features = net.alarm_features(depth, vis.sigmoid(), transform, base)
    assert features.shape == (batch, 2, 777)
    # Explicit lack-of-visibility features survive as a distinct state; no
    # clear decision is made by the geometric integral.
    absent = net.alarm_features(depth, torch.zeros_like(vis), transform, base)
    assert not bool(absent[..., :768].any())
    assert bool(features[..., 256:512].mean() > 0)
    # Recover alarm exactly using only public ray bottleneck features.
    torch.testing.assert_close(pred, base.detach()+net.alarm(features).squeeze(-1), rtol=0, atol=0)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred)).backward()
    assert base.grad is None and bool(net.alarm[-1].weight.grad.abs().sum() > 0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    pred, (depth, vis) = net(native, pose, transform, base)
    F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred)).backward()
    assert bool(net.conv1.weight.grad.abs().sum() > 0)
    opt.zero_grad(set_to_none=True)
    depth, vis = net.predict_rays(native, pose)
    target_depth = torch.full_like(depth, .3)
    target_valid = torch.ones_like(vis, dtype=torch.bool)
    aux, vl, dl = auxiliary_loss(depth, vis, target_depth, target_valid)
    aux.backward()
    assert bool(net.conv1.weight.grad.abs().sum() > 0)
    zero_aux, _, zero_depth = auxiliary_loss(depth.detach(), vis.detach(), torch.zeros_like(depth), torch.zeros_like(target_valid))
    assert zero_depth == 0 and torch.isfinite(zero_aux)
    if gpu:
        torch.cuda.synchronize()
    result = dict(status='PASS', synthetic_only=True, source_sha256=sha(__file__), device=device, batch=batch,
                  parameters=sum(v.numel() for v in net.parameters()), elapsed_s=time.perf_counter()-began,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else None,
                  checks=['128x128 pixel shuffle/ray axes', '8x8 weighted microcell integral', 'compact 1cm query boundary',
                          'causal input left padding', 'initial exact M3/no baseline gradient', '777 ray-only head/no latent shortcut',
                          'explicit predicted visibility', 'BCEO second-step gradient', 'RAY auxiliary gradient/empty-valid finite'])
    print(__import__('json').dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['check', 'train'])
    p.add_argument('--gpu', action='store_true')
    a = p.parse_args()
    if a.stage == 'check':
        check(a.gpu)
    else:
        train()

"""Observed-bin design repair with physical, per-episode instrument randomization.

No pilot observations, protected test data, or model alarm outputs are read here.
The original M3/ordinary optimizer recipes are retained. Existing train geometry
is rerendered before pulse integration, rather than blurring already-noisy CNH.
The one DR draw belongs to a physical scene/replica and is shared by its windows.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-no-bin0-dr-retrain-dev-20261011'
OLD = ROOT/'artifacts.local/work/cnh-temporal-readout-20261004'
AUG = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
FRAMES = range(3, 16)
ORDINARY_SEEDS = (2026100955, 2026100956, 2026100957)
CHANNELS = (0, 1, 2, 3, 4, 5, 12, 13, 14)
DR_SEED = 2026101117


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 23), b''): h.update(b)
    return h.hexdigest()


def configure():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Match frozen FP32 accumulation while permitting deterministic convolution
    # algorithm reuse across the many identical training minibatch shapes.
    torch.backends.cudnn.benchmark = True
    dll = Path(torch.__file__).parent.parent/'nvidia/cublas/bin'
    handle = os.add_dll_directory(str(dll)) if os.name == 'nt' and dll.is_dir() else None
    if handle is not None: os.environ['PATH'] = str(dll)+os.pathsep+os.environ.get('PATH', '')
    return handle


class Stage:
    def __init__(self, name):
        self.name = name; self.start = time.monotonic()
        self.previous = sum(json.loads(p.read_text())['seconds'] for p in (OUT/'gpu_receipts').glob('*.json'))
        self.cap = json.loads((OUT/'PLAN.json').read_text(encoding='utf-8-sig')).get('gpu_wall_cap_seconds', 5400)

    def check(self):
        if self.previous+time.monotonic()-self.start >= self.cap:
            raise TimeoutError('Shared GPU command-stage wall budget reached')

    def finish(self, status, **metadata):
        row = dict(stage=self.name, status=status, seconds=time.monotonic()-self.start,
                   previous_seconds=self.previous, **metadata)
        save(OUT/'gpu_receipts'/f'{self.name}_{time.time_ns()}.json', row)
        return row


def draw_parameters(key):
    rng = np.random.default_rng(np.random.SeedSequence([DR_SEED, *map(int, key)]))
    return dict(gain=float(rng.uniform(.42, .73)), sigma_m=float(rng.uniform(.070, .115)),
        tail_fraction=float(rng.uniform(0, .05)), tail_scale_m=.5,
        range_offset_m=float(rng.uniform(-.1, .1)), ambient=float(4*rng.uniform(1, 2.6)),
        noise_a=float(rng.uniform(0, .25357091365)),
        noise_b=float(rng.uniform(.08671592722, .09326653791)),
        noise_c=float(rng.uniform(.5807095993, .68960300094)))


def raw_expected(renderer):
    """Native128 visible-return expectation, before pulse, noise and crosstalk."""
    cp = renderer.cp
    distance = renderer.distance.reshape(renderer.npose, renderer.nray)
    valid = cp.isfinite(distance) & (distance > 0)
    bins = cp.floor(cp.where(valid, distance, 0)/.0375348).astype(cp.int64)
    inside = valid & (bins >= 0) & (bins < 128)
    safe = cp.maximum(cp.where(valid, distance, 1), .05)
    rho = renderer.rho.reshape(renderer.npose, renderer.nray)
    cosine = renderer.cosine.reshape(renderer.npose, renderer.nray)
    # Sum quarters first: exact linear signal expectation; electronics now uses
    # the effective decoded variance bridge specified in PLAN, not Skellam.
    energy = (renderer.params.signal_counts/4)*(rho*4*renderer.fractions[None])*cosine*renderer.weights[None]/safe**2
    zone = cp.arange(renderer.nray, dtype=cp.int64)//256
    index = ((cp.arange(renderer.npose, dtype=cp.int64)[:, None]*64+zone[None])*128+cp.clip(bins, 0, 127))
    hist = cp.zeros((renderer.npose, 8, 8, 128), dtype=cp.float64)
    cp.add.at(hist.reshape(-1), index.reshape(-1), cp.where(inside, energy, 0).reshape(-1))
    return hist


def sample_response(raw, parameters, seed, signal_counts):
    """Integrate a shifted Gaussian/exponential pulse into aggregate bins."""
    import cupy as cp
    from cupyx.scipy.special import ndtr
    p = parameters
    centers = (cp.arange(128, dtype=cp.float64)+.5)*.0375348+p['range_offset_m']
    edges = cp.arange(17, dtype=cp.float64)*(.0375348*8)
    x = edges[None]-centers[:, None]
    gaussian = ndtr(x/p['sigma_m'])
    tail = cp.where(x > 0, -cp.expm1(-cp.maximum(x, 0)/p['tail_scale_m']), 0)
    cdf = (1-p['tail_fraction'])*gaussian+p['tail_fraction']*tail
    kernel = cp.diff(cdf, axis=-1)
    signal = (raw@kernel)*p['gain']
    old = signal.copy(); leak = .02/4
    signal[:, 1:] += leak*(old[:, :-1]-old[:, 1:])
    signal[:, :-1] += leak*(old[:, 1:]-old[:, :-1])
    signal[:, :, 1:] += leak*(old[:, :, :-1]-old[:, :, 1:])
    signal[:, :, :-1] += leak*(old[:, :, 1:]-old[:, :, :-1])
    # Preserve nominal residual crosstalk at 4cm; apply the same pulse/gain.
    xtalk = kernel[int(.04/.0375348)]*(signal_counts*.02*p['gain'])
    mean = signal+xtalk
    variance = p['noise_a']*cp.maximum(signal, 0)+p['noise_b']*p['ambient']+p['noise_c']
    rng = cp.random.RandomState(int(seed))
    hist = mean+rng.standard_normal(mean.shape)*cp.sqrt(cp.maximum(variance, 1e-12))
    # Fixed original training-contract bias normalization, not fitted-B0 removal.
    import cnh_displacement_ceiling as D
    bias = cp.asarray(np.load(D.BIAS).astype(np.float32))
    numerator = hist.astype(cp.float32)-bias
    denominator = cp.sqrt(cp.maximum(16*np.float32(p['ambient'])+cp.maximum(bias, 0), np.float32(1e-9)))
    return cp.asnumpy((numerator/denominator).astype(cp.float16))


def sequences():
    """Existing train physical geometry in exactly the inherited row order."""
    import cnh_near_range as NR
    import cnh_temporal_readout_data as T
    import cnh_cvr_pilot as CP
    plan = json.loads((OLD/'PLAN.json').read_text())
    for unit in plan['natural_train_units']:
        for sc in NR.scenes_for(int(unit)):
            sensor, travel, noisy = CP.motion_metadata(int(unit), int(sc['config']))
            yield dict(source='old', key=(0, int(unit), int(sc['config']), 0), boxes=sc['boxes'],
                       sensor=sensor, travel=travel, noisy=noisy, m3=True)
    for unit in plan['train_units']:
        truth = json.loads((OLD/'truth/train'/f'unit{unit}.json').read_text())
        with np.load(OLD/'observations/train'/f'unit{unit}.npz') as d:
            sensor, travel, noisy = (d[k] for k in ('sensor', 'travel', 'noisy'))
        for variant, boxes in enumerate(truth['boxes']):
            for replica in range(2):
                yield dict(source='old', key=(1, int(unit), variant, replica), boxes=boxes,
                           sensor=sensor, travel=travel, noisy=noisy[replica], m3=False)
    rows = json.loads((AUG/'scene_rows.json').read_text())['ordinary_train']
    import cnh_aligned_boundary_dev as B
    sensor, query = B.poses(-10.)
    # materialize augmentation uses query[f]@inv(sensor[f])@sensor[past], so
    # provide those exact inherited transforms separately in prepare().
    for row in rows:
        for replica in range(2):
            yield dict(source='augmentation', key=(2, int(row['scene_id']), 0, replica), boxes=row['boxes'],
                       sensor=sensor, travel=sensor@np.linalg.inv(query), noisy=sensor,
                       public_query=query, m3=True)


@torch.no_grad()
def project_native(histories, transforms, lengths, projector=None, *, extra_drop_bins=None):
    """Batched version of frozen finite-volume projector, identical arithmetic."""
    from cnh_cvr_projection import Projector, EDGE, WIDTH, SUB, SHAPE
    projector = Projector() if projector is None else projector
    b = len(histories)
    t = torch.as_tensor(transforms, device='cuda', dtype=torch.float64)
    z = torch.as_tensor(histories, device='cuda', dtype=torch.float64)
    le = torch.as_tensor(lengths, device='cuda')
    total = torch.zeros((b, *SHAPE), device='cuda'); count = torch.zeros_like(total)
    second_total = torch.zeros_like(total) if extra_drop_bins is not None else None
    current = None
    for j in range(8):
        p = torch.bmm(projector.points[None]-t[:, j, None, :3, 3], t[:, j, :3, :3])
        radius = torch.linalg.vector_norm(p, dim=2)
        xy = p[:, :, :2]/p[:, :, 2:3].clamp_min(1e-30)
        ij = torch.floor((xy+EDGE)/(2*EDGE)*8).long()
        bins = torch.floor(radius/WIDTH).long()
        valid = (p[:, :, 2] > 0)&(ij >= 0).all(2)&(ij < 8).all(2)&(bins >= 0)&(bins < 16)
        valid &= (j >= 8-le)[:, None]
        index = (ij[:, :, 1].clamp(0, 7)*8+ij[:, :, 0].clamp(0, 7))*16+bins.clamp(0, 15)
        weight = valid*projector.voxel_volume/(SUB**3)/projector.volumes[index]
        mass = torch.gather(z[:, j].reshape(b, -1), 1, index)*weight
        if extra_drop_bins is not None:
            keep = torch.ones_like(index, dtype=torch.bool)
            for drop in extra_drop_bins: keep &= index.remainder(16) != drop
            second_evidence = (mass*keep).reshape(b, -1, SUB**3).sum(2).reshape(b, *SHAPE).float()
            second_total += second_evidence
        evidence = mass.reshape(b, -1, SUB**3).sum(2).reshape(b, *SHAPE).float()
        coverage = valid.reshape(b, -1, SUB**3).double().mean(2).reshape(b, *SHAPE).float()
        total += evidence; count += coverage; current = evidence
    result = torch.stack((total, count, current), 1)
    if extra_drop_bins is not None:
        return result, torch.stack((second_total, count, second_evidence), 1)
    return result


def prep_cvr(voxels, masks):
    x = voxels.float().clone()
    x[:, 0] = x[:, 0].sign()*x[:, 0].abs().log1p()
    x[:, 2] = x[:, 2].sign()*x[:, 2].abs().log1p(); x[:, 1] /= 8
    return torch.cat((x, masks[None].expand(len(x), -1, -1, -1, -1)), 1)


def profile():
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks, Projector
    import cnh_boundary_token_model as M
    import cnh_location_reference_gpu as G
    stage = Stage('profile'); result = {}
    try:
        seq = next(sequences()); p = draw_parameters(seq['key'])
        t0 = time.monotonic(); renderer = G.ExpectedRenderer(seq['sensor'], seq['boxes'])
        try:
            raw = raw_expected(renderer); z = sample_response(raw, p, 42, renderer.params.signal_counts)
        finally: renderer.close()
        torch.cuda.synchronize(); result['render_cold_scene_seconds'] = time.monotonic()-t0
        render_times = []
        for warm_index in range(6):
            t0 = time.monotonic(); renderer = G.ExpectedRenderer(seq['sensor'], seq['boxes'])
            try:
                raw = raw_expected(renderer); z = sample_response(raw, p, 43+warm_index, renderer.params.signal_counts)
            finally: renderer.close()
            torch.cuda.synchronize(); render_times.append(time.monotonic()-t0)
        result['render_scene_seconds'] = float(np.median(render_times))
        result['render_warm_times_seconds'] = render_times
        from cnh_cvr_pilot import relative_transforms
        transforms = np.stack([np.eye(4)]*8)
        transforms[:] = relative_transforms(seq['sensor'], seq['travel'], seq['noisy'], 15)
        zz = np.repeat(z[-8:][None], 4, axis=0); tt = np.repeat(transforms[None], 4, axis=0)
        zz[..., 0] = 0
        projector = Projector()
        v = project_native(zz, tt, [8]*4, projector)
        exact = projector.sequence(zz[0], tt[0])
        result['projector_max_abs'] = float((v[0]-exact).abs().max())
        torch.cuda.synchronize(); t0 = time.monotonic()
        for _ in range(3): project_native(zz, tt, [8]*4, projector)
        torch.cuda.synchronize(); result['project_seconds_per_row'] = (time.monotonic()-t0)/12
        masks = torch.as_tensor(query_masks(), device='cuda')
        xv = v.repeat(64, 1, 1, 1, 1); labels = torch.zeros((256, 2), device='cuda')
        net = CVR().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
        for _ in range(2):
            opt.zero_grad(set_to_none=True); F.binary_cross_entropy_with_logits(net(prep_cvr(xv, masks)), labels).backward(); opt.step()
        torch.cuda.synchronize(); t0 = time.monotonic()
        for _ in range(8):
            opt.zero_grad(set_to_none=True); F.binary_cross_entropy_with_logits(net(prep_cvr(xv, masks)), labels).backward(); opt.step()
        torch.cuda.synchronize(); result['m3_step_seconds'] = (time.monotonic()-t0)/8
        del net, opt, xv
        geometry = M.feature_geometry(torch.as_tensor(tt, device='cuda'), torch.tensor([8]*4, device='cuda'), 'center')
        feat = M.build_features(torch.as_tensor(zz, device='cuda'), geometry, torch.tensor([8]*4, device='cuda'))
        feat[:, :, :, 6:12] = 0
        xx = feat.repeat(64, 1, 1, 1, 1, 1)
        net = M.BoundaryTokenReadout().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
        lens = torch.tensor([8]*256, device='cuda')
        for _ in range(2):
            opt.zero_grad(set_to_none=True); F.binary_cross_entropy_with_logits(net(xx, lens), labels).backward(); opt.step()
        torch.cuda.synchronize(); t0 = time.monotonic()
        for _ in range(8):
            opt.zero_grad(set_to_none=True); F.binary_cross_entropy_with_logits(net(xx, lens), labels).backward(); opt.step()
        torch.cuda.synchronize(); result['ordinary_step_seconds'] = (time.monotonic()-t0)/8
        result['estimate_main_m3_train_seconds'] = result['m3_step_seconds']*math.ceil(47424/256)*20*5
        result['estimate_main_ordinary_train_seconds'] = result['ordinary_step_seconds']*7488*3
        result['estimate_two_arm_projection_seconds'] = result['project_seconds_per_row']*47424*2
        result['estimate_render_scenes_seconds'] = result['render_scene_seconds']*(3072+1536)
        save(OUT/'profile.json', result)
        print(json.dumps(result), flush=True)
        return stage.finish('COMPLETE', **result)
    except Exception as error:
        stage.finish('FAILED', error=repr(error)); raise


def prepare():
    """Rerender inputs; preserve ordinary labels/weights and public transforms."""
    import cnh_location_reference_gpu as G
    import cnh_margin_labels as ML
    stage = Stage('prepare_native')
    maps = {}; params = []; m3indices = []; m3labels = []; cursor = 0
    prior_geometry_key = None; prior_raw = None; signal_counts = None
    try:
        folder = OUT/'inputs/train'; folder.mkdir(parents=True, exist_ok=True)
        if (folder/'histories.npy').exists(): raise FileExistsError('Preserve existing native preparation; use a documented repair output')
        olddir = OLD/'inputs/train'; augdir = AUG/'data/ordinary_train'
        counts = [len(np.load(p/'length.npy', mmap_mode='r')) for p in (olddir, augdir)]
        n = sum(counts)
        for name in ('transforms', 'length', 'labels', 'weights', 'mask'):
            arrays = [np.load(p/(name+'.npy'), mmap_mode='r') for p in (olddir, augdir)]
            np.save(folder/(name+'.npy'), np.concatenate(arrays))
        maps['histories'] = np.lib.format.open_memmap(folder/'histories.npy', mode='w+', dtype=np.float16, shape=(n, 8, 8, 8, 16))
        for index, seq in enumerate(sequences()):
            stage.check(); p = draw_parameters(seq['key'])
            geometry_key = seq['key'][:-1]
            if geometry_key != prior_geometry_key:
                renderer = G.ExpectedRenderer(seq['sensor'], seq['boxes'])
                try:
                    prior_raw = raw_expected(renderer); signal_counts = renderer.params.signal_counts
                finally: renderer.close()
                prior_geometry_key = geometry_key
            seed = int(np.random.SeedSequence([DR_SEED+1, *seq['key']]).generate_state(1)[0])
            z = sample_response(prior_raw, p, seed, signal_counts)
            lab = ML.labels_for(seq['boxes'], seq['travel'], .33) if seq['m3'] else None
            for f in FRAMES:
                begin = max(0, f-7); le = f-begin+1
                maps['histories'][cursor] = 0; maps['histories'][cursor, 8-le:] = z[begin:f+1]
                if seq['m3']: m3indices.append(cursor); m3labels.append(lab[f])
                cursor += 1
            params.append(dict(row_start=cursor-13, key=seq['key'], source=seq['source'], **p))
            if (index+1) % 128 == 0:
                print('PREPARE_NATIVE', index+1, cursor, round(time.monotonic()-stage.start, 1), flush=True)
                save(OUT/'progress/native.json', dict(episodes=index+1, rows=cursor))
        if cursor != n: raise ValueError(f'Inherited row identity differs: {cursor} vs {n}')
        maps['histories'].flush()
        np.save(folder/'m3_indices.npy', np.asarray(m3indices, np.int64))
        np.save(folder/'m3_labels.npy', np.asarray(m3labels, np.float32))
        save(OUT/'domain_randomization_draws.json', params)
        receipt = stage.finish('COMPLETE', ordinary_rows=n, m3_rows=len(m3indices), episodes=len(params),
                              train_sources=[str(olddir), str(augdir)], per_episode_constant=True)
        save(folder/'receipt.json', receipt)
        return receipt
    except Exception as error:
        stage.finish('FAILED', row=cursor, error=repr(error)); raise
    finally:
        maps.clear(); torch.cuda.empty_cache()


def prepare_features(arm, *, resume_ordinary=False):
    import cnh_boundary_token_model as M
    from cnh_cvr_projection import Projector
    if arm == 'both': return prepare_features_paired(resume_ordinary=resume_ordinary)
    if resume_ordinary: raise ValueError('The documented ordinary-cache repair applies to paired features only')
    stage = Stage('prepare_features_'+arm); dropped = [0] if arm == 'no_bin0' else [0, 1]
    data = OUT/'inputs/train'; folder = OUT/'features'/arm; folder.mkdir(parents=True, exist_ok=True)
    z = np.load(data/'histories.npy', mmap_mode='r'); t = np.load(data/'transforms.npy', mmap_mode='r')
    le = np.load(data/'length.npy', mmap_mode='r'); ids = np.load(data/'m3_indices.npy')
    try:
        ordinary = np.lib.format.open_memmap(folder/'ordinary.npy', mode='w+', dtype=np.float16, shape=(len(z), 2, 16, 9, 8, 8))
        with torch.no_grad():
            for begin in range(0, len(z), 64):
                stage.check(); end = min(begin+64, len(z))
                zz = torch.as_tensor(np.array(z[begin:end]), device='cuda').float(); zz[..., dropped] = 0
                tt = torch.as_tensor(np.array(t[begin:end]), device='cuda'); ll = torch.as_tensor(np.array(le[begin:end]), device='cuda')
                geometry = M.feature_geometry(tt, ll, 'center')
                ordinary[begin:end] = M.build_features(zz, geometry, ll)[:, :, :, list(CHANNELS)].half().cpu().numpy()
        ordinary.flush(); del ordinary, zz, tt, ll, geometry
        vox = np.lib.format.open_memmap(folder/'m3.npy', mode='w+', dtype=np.float16, shape=(len(ids), 3, 24, 17, 33))
        projector = Projector()
        for begin in range(0, len(ids), 8):
            stage.check(); ix = ids[begin:begin+8]
            zz = np.array(z[ix]); zz[..., dropped] = 0
            vox[begin:begin+len(ix)] = project_native(zz, np.array(t[ix]), np.array(le[ix]), projector).half().cpu().numpy()
            if begin % 2600 == 0:
                print('PREPARE_FEATURES', arm, begin, '/', len(ids), round(time.monotonic()-stage.start, 1), flush=True)
        vox.flush(); del vox, projector
        receipt = stage.finish('COMPLETE', arm=arm, dropped_bins=dropped, ordinary_rows=len(z), m3_rows=len(ids))
        save(folder/'receipt.json', receipt); return receipt
    except Exception as error:
        stage.finish('FAILED', error=repr(error)); raise
    finally: torch.cuda.empty_cache()


def prepare_features_paired(*, resume_ordinary=False):
    """Two masks share geometric transport and feature geometry, not signals."""
    import cnh_boundary_token_model as M
    from cnh_cvr_projection import Projector
    stage = Stage('prepare_features_both'); arrays = {}
    data = OUT/'inputs/train'
    z = np.load(data/'histories.npy', mmap_mode='r'); t = np.load(data/'transforms.npy', mmap_mode='r')
    le = np.load(data/'length.npy', mmap_mode='r'); ids = np.load(data/'m3_indices.npy')
    try:
        resume_checks = []
        if resume_ordinary:
            failures = [json.loads(p.read_text()) for p in (OUT/'gpu_receipts').glob('prepare_features_both_*.json')]
            if not any(r['status']=='FAILED' and 'UnboundLocalError' in r.get('error', '') and 'paired_errors' in r.get('error', '') for r in failures):
                raise ValueError('No matching documented post-ordinary projection failure')
            for arm in ('no_bin0', 'no_bin01'):
                cached = np.load(OUT/'features'/arm/'ordinary.npy', mmap_mode='r')
                if cached.shape != (len(z), 2, 16, 9, 8, 8) or cached.dtype != np.float16:
                    raise ValueError('Completed ordinary cache shape/dtype mismatch')
                # The preserved traceback occurred after ordinary.flush(), and
                # before the first M3 write. Recompute both endpoint rows to
                # bind the existing complete cache to the repaired extractor.
                for index in (0, len(z)-1):
                    zz = torch.as_tensor(np.array(z[index:index+1]), device='cuda').float()
                    zz[..., [0] if arm=='no_bin0' else [0, 1]] = 0
                    tt = torch.as_tensor(np.array(t[index:index+1]), device='cuda')
                    ll = torch.as_tensor(np.array(le[index:index+1]), device='cuda')
                    geometry = M.feature_geometry(tt, ll, 'center')
                    expected = M.build_features(zz, geometry, ll)[:, :, :, list(CHANNELS)].half().cpu().numpy()[0]
                    if not np.array_equal(cached[index], expected):
                        raise ValueError('Preserved ordinary endpoint differs: '+arm+str(index))
                    resume_checks.append(dict(arm=arm, row=index, exact=True))
                del cached
            stamp = time.time_ns()
            preserved = []
            for arm in ('no_bin0', 'no_bin01'):
                partial = OUT/'features'/arm/'m3.npy'
                if not partial.is_file(): raise FileNotFoundError('Expected documented M3 partial: '+str(partial))
                target = partial.with_name(f'm3_failed_unbound_paired_errors_{stamp}.npy')
                if not partial.resolve().is_relative_to(OUT.resolve()) or not target.parent.resolve().is_relative_to(OUT.resolve()):
                    raise ValueError('Repair path escapes this task output')
                partial.rename(target); preserved.append(str(target))
            save(OUT/'feature_resume_ordinary_receipt.json', dict(reason='paired_errors initializer source repair',
                cached_rows=len(z), endpoint_checks=resume_checks, preserved_partial_paths=preserved,
                native_rerender=False, ordinary_rerender=False))
        else:
            for arm in ('no_bin0', 'no_bin01'):
                folder = OUT/'features'/arm; folder.mkdir(parents=True, exist_ok=True)
                if (folder/'ordinary.npy').exists() or (folder/'m3.npy').exists():
                    raise FileExistsError('Preserve existing feature preparation: '+str(folder))
                arrays[arm] = np.lib.format.open_memmap(folder/'ordinary.npy', mode='w+', dtype=np.float16, shape=(len(z), 2, 16, 9, 8, 8))
            with torch.no_grad():
                for begin in range(0, len(z), 64):
                    stage.check(); end = min(begin+64, len(z))
                    zz = torch.as_tensor(np.array(z[begin:end]), device='cuda').float(); zz[..., 0] = 0
                    tt = torch.as_tensor(np.array(t[begin:end]), device='cuda'); ll = torch.as_tensor(np.array(le[begin:end]), device='cuda')
                    geometry = M.feature_geometry(tt, ll, 'center')
                    arrays['no_bin0'][begin:end] = M.build_features(zz, geometry, ll)[:, :, :, list(CHANNELS)].half().cpu().numpy()
                    zz[..., 1] = 0
                    arrays['no_bin01'][begin:end] = M.build_features(zz, geometry, ll)[:, :, :, list(CHANNELS)].half().cpu().numpy()
            for arr in arrays.values(): arr.flush()
            arrays.clear()
        del zz, tt, ll, geometry
        for arm in ('no_bin0', 'no_bin01'):
            arrays[arm] = np.lib.format.open_memmap(OUT/'features'/arm/'m3.npy', mode='w+', dtype=np.float16, shape=(len(ids), 3, 24, 17, 33))
        projector = Projector()
        paired_errors = None
        for begin in range(0, len(ids), 8):
            stage.check(); ix = ids[begin:begin+8]
            zz = np.array(z[ix]); zz[..., 0] = 0
            a, b = project_native(zz, np.array(t[ix]), np.array(le[ix]), projector, extra_drop_bins=[1])
            if paired_errors is None:
                ll = int(le[ix[0]])
                direct_main = projector.sequence(zz[0, -ll:], np.array(t[ix[0], -ll:]))
                second_z = zz[0, -ll:].copy(); second_z[..., 1] = 0
                direct_sensitive = projector.sequence(second_z, np.array(t[ix[0], -ll:]))
                paired_errors = [float((a[0]-direct_main).abs().max()), float((b[0]-direct_sensitive).abs().max())]
                if max(paired_errors) > 1e-6: raise ValueError('Paired projection differs from frozen separate transport')
            arrays['no_bin0'][begin:begin+len(ix)] = a.half().cpu().numpy()
            arrays['no_bin01'][begin:begin+len(ix)] = b.half().cpu().numpy()
            if begin % 2600 == 0:
                print('PREPARE_FEATURES_BOTH', begin, '/', len(ids), round(time.monotonic()-stage.start, 1), flush=True)
        for arr in arrays.values(): arr.flush()
        receipt = stage.finish('COMPLETE', arms=['no_bin0', 'no_bin01'], ordinary_rows=len(z), m3_rows=len(ids), shared_geometry=True,
                               paired_projection_max_abs=paired_errors, resumed_ordinary=resume_ordinary,
                               ordinary_endpoint_checks=resume_checks)
        for arm in arrays: save(OUT/'features'/arm/'receipt.json', dict(receipt, arm=arm))
        return receipt
    except Exception as error:
        stage.finish('FAILED', error=repr(error)); raise
    finally: arrays.clear(); torch.cuda.empty_cache()


def load_gpu(path):
    arr = np.load(path, mmap_mode='r')
    result = torch.empty(arr.shape, device='cuda', dtype=torch.float16)
    for b in range(0, len(arr), 512): result[b:b+512] = torch.from_numpy(np.array(arr[b:b+512]))
    return result


def train_m3(arm, seeds):
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    stage = Stage('train_m3_'+arm)
    try:
        x = load_gpu(OUT/'features'/arm/'m3.npy')
        y = torch.as_tensor(np.load(OUT/'inputs/train/m3_labels.npy'), device='cuda')
        masks = torch.as_tensor(query_masks(), device='cuda')
        for seed in seeds:
            target = OUT/'models'/arm/'M3'/f'model_seed{seed}.pt'; target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): raise FileExistsError(target)
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed); net = CVR().cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20); history = []
            for epoch in range(20):
                t0 = time.monotonic(); order = rng.permutation(len(x)); total = 0
                for begin in range(0, len(x), 256):
                    stage.check(); ix = torch.as_tensor(np.sort(order[begin:begin+256]), device='cuda')
                    opt.zero_grad(set_to_none=True)
                    loss = F.binary_cross_entropy_with_logits(net(prep_cvr(x[ix], masks)), y[ix])
                    loss.backward(); opt.step(); total += float(loss.detach())*len(ix)
                scheduler.step(); row = dict(epoch=epoch+1, loss=total/len(x), seconds=time.monotonic()-t0)
                history.append(row); save(target.parent/f'seed{seed}_history.json', history)
                print('TRAIN_M3', arm, seed, epoch+1, round(row['loss'], 6), round(row['seconds'], 1), flush=True)
            torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, target)
            save(target.with_suffix('.json'), dict(arm=arm, seed=seed, epochs=20, rows=len(x), steps=20*math.ceil(len(x)/256),
                sha256=sha(target), drop_bins=[0] if arm=='no_bin0' else [0, 1], recipe='Original M3 BCE AdamW.002/wd.0001 batch256 cosine20; train geometry union'))
            del net, opt, scheduler
        return stage.finish('COMPLETE', arm=arm, seeds=list(seeds))
    except Exception as error:
        if 'net' in locals() and 'target' in locals():
            torch.save(dict(state_dict=net.cpu().state_dict(), final=False, epoch=locals().get('epoch'),
                            arm=arm, seed=seed), target.with_name(target.stem+f'_partial_{time.time_ns()}.pt'))
        stage.finish('FAILED', error=repr(error)); raise
    finally: torch.cuda.empty_cache()


def train_ordinary(arm, seeds):
    import cnh_boundary_token_model as M
    from cnh_counterfactual_train_dev import restore_features, slot_schedule
    stage = Stage('train_ordinary_'+arm)
    try:
        x = load_gpu(OUT/'features'/arm/'ordinary.npy')
        y = torch.as_tensor(np.load(OUT/'inputs/train/labels.npy'), device='cuda')
        w = torch.as_tensor(np.load(OUT/'inputs/train/weights.npy'), device='cuda')
        le = torch.as_tensor(np.load(OUT/'inputs/train/length.npy'), device='cuda')
        for seed in seeds:
            target = OUT/'models'/arm/f'ordinary_seed{seed}.pt'; target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): raise FileExistsError(target)
            torch.manual_seed(seed); net = M.BoundaryTokenReadout().cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001); history = []; steps = 0
            for epoch in range(32):
                t0 = time.monotonic(); order, _ = slot_schedule(seed, epoch, 39936, 19968)
                weighted_sum = mass_sum = 0
                for begin in range(0, len(x), 256):
                    stage.check(); ix = torch.as_tensor(order[begin:begin+256], device='cuda')
                    raw = net(restore_features(x[ix]), le[ix]); lossraw = F.binary_cross_entropy_with_logits(raw, y[ix], reduction='none')
                    mass = w[ix].sum(); weighted = (lossraw*w[ix]).sum(); loss = weighted/mass
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); steps += 1
                    weighted_sum += float(weighted.detach()); mass_sum += float(mass)
                row = dict(epoch=epoch+1, loss=weighted_sum/mass_sum, seconds=time.monotonic()-t0, steps=steps)
                history.append(row); save(target.with_suffix('.history.json'), history)
                print('TRAIN_ORDINARY', arm, seed, epoch+1, round(row['loss'], 6), round(row['seconds'], 1), flush=True)
            if steps != 7488: raise ValueError('Original ordinary schedule differs')
            torch.save(dict(state_dict=net.cpu().state_dict(), arm=arm, seed=seed, epochs=32, steps=steps,
                            drop_bins=[0] if arm=='no_bin0' else [0, 1]), target)
            save(target.with_suffix('.json'), dict(arm=arm, seed=seed, epochs=32, rows=len(x), steps=steps,
                sha256=sha(target), recipe='Original ordinary masked weighted BCE AdamW.001/wd.0001 batch256 constantlr'))
            del net, opt
        return stage.finish('COMPLETE', arm=arm, seeds=list(seeds))
    except Exception as error:
        if 'net' in locals() and 'target' in locals():
            torch.save(dict(state_dict=net.cpu().state_dict(), final=False, steps=locals().get('steps', 0),
                            arm=arm, seed=seed), target.with_name(target.stem+f'_partial_{time.time_ns()}.pt'))
        stage.finish('FAILED', error=repr(error)); raise
    finally: torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('profile', 'prepare', 'features', 'train_m3', 'train_ordinary'))
    parser.add_argument('--arm', choices=('no_bin0', 'no_bin01', 'both'), default='no_bin0')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--resume-ordinary', action='store_true')
    a = parser.parse_args(); handle = configure()
    try:
        if a.stage == 'profile': profile()
        elif a.stage == 'prepare': prepare()
        elif a.stage == 'features': prepare_features(a.arm, resume_ordinary=a.resume_ordinary)
        elif a.stage == 'train_m3': train_m3(a.arm, a.seeds or (list(range(5)) if a.arm=='no_bin0' else [0]))
        elif a.stage == 'train_ordinary': train_ordinary(a.arm, a.seeds or (ORDINARY_SEEDS if a.arm=='no_bin0' else [ORDINARY_SEEDS[0]]))
    finally:
        torch.cuda.empty_cache()
        if handle is not None: handle.close()


if __name__ == '__main__': main()

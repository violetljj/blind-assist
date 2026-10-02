"""Frozen BCEO/RAY training-only ray geometry and oracle-head diagnostic."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import cnh_ray_surface_train as T
import cnh_ray_surface_data as D

OUT = T.ST.SS.WORK/'cnh-ray-surface-geometry-probe-20261003'
FACTOR = T.ST.SS.WORK/'cnh-surface-factorization-probe-20261002'
RUN_ID = 'CNH_RAY_SURFACE_GEOMETRY_PROBE_20261003'
UNITS = list(range(93000, 93096))
CATS = ('all', 'contact0-2cm', 'contact2-5cm', 'contact>5cm', 'pass0-10cm', 'clear')
GROUPS = (*CATS, 'contact0-2cm_EXACT_supported')
HYBRIDS = ('PD_PV', 'TD_PV', 'PD_TV', 'TD_TV')
METRICS = ('visible_query_radial_MAE_m', 'visible_query_valid_FN_rate', 'visible_query_within2cm_rate', 'teacher_soft_support_rate') + tuple(
    f'{h}_{m}' for h in HYBRIDS for m in ('mass', 'L1_vs_TD_TV', 'added_mass', 'lost_mass')) + (
    'native_logit', 'teacher_logit', 'teacher_minus_native_logit', 'absolute_logit_change') + tuple(h+'_logit_delta_vs_native' for h in HYBRIDS)
RAY_METRICS = ('teacher_valid_radial_MAE_m', 'valid_FN_rate', 'valid_FP_rate')
sha, read, save = T.sha, T.read, T.create_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity():
    receipt_path = T.OUT/'training_receipt.json'
    r = read(receipt_path)
    if r['status'] != 'COMPLETE' or sha(T.OUT/'training_request.json') != r['request_sha256']:
        raise ValueError('Frozen training receipt/request mismatch')
    for p, expected in {**r['source_sha256'], **r['baseline_sha256']}.items():
        if sha(p) != expected:
            raise ValueError('Frozen source/baseline changed: '+p)
    for p, expected in r['models_sha256'].items():
        if sha(T.OUT/p) != expected:
            raise ValueError('Frozen checkpoint changed: '+p)
    labels = T.OUT/'labels_receipt.json'
    if sha(labels) != r['labels_receipt_sha256'] or read(labels)['status'] != 'COMPLETE':
        raise ValueError('Frozen label receipt changed')
    representative = T.OUT/'labels/train/unit93000/receipt.json'
    if read(labels)['unit_receipt_sha256']['train/unit93000/receipt.json'] != sha(representative):
        raise ValueError('Representative label source receipt mismatch')
    extra_sources = read(representative)['source_sha256']
    for p, expected in extra_sources.items():
        if sha(p) != expected:
            raise ValueError('Frozen public-ray/label dependency changed: '+p)
    return dict(source_sha256={str(Path(__file__)): sha(__file__), **r['source_sha256'], **extra_sources},
                training_receipt_sha256=sha(receipt_path), models_sha256=r['models_sha256'],
                baseline_sha256=r['baseline_sha256'], labels_receipt_sha256=sha(labels),
                factorization_result_sha256=sha(FACTOR/'result.json'), units=UNITS, frames=list(range(3, 16)),
                metrics=METRICS, hybrids=HYBRIDS, visibility_threshold=.5, soft_boundary_m=T.SOFT,
                bootstrap=dict(replicates=1000, seed=2026100301))


def unit_inputs(unit):
    folder = T.OUT/'labels/train'/f'unit{unit}'
    receipt = read(folder/'receipt.json')
    aggregate = read(T.OUT/'labels_receipt.json')
    if receipt['status'] != 'COMPLETE' or receipt['unit'] != unit or sha(folder/'receipt.json') != aggregate['unit_receipt_sha256'][f'train/unit{unit}/receipt.json']:
        raise ValueError('Label unit binding mismatch')
    paths = {name: folder/name for name in ('depth.npy', 'valid.npy', 'metadata.npz')}
    for name, p in paths.items():
        if sha(p) != receipt['output_sha256'][name]:
            raise ValueError('Label file changed')
    with np.load(paths['metadata.npz'], allow_pickle=False) as z:
        if not np.array_equal(z['configs'], np.arange(22)) or not np.array_equal(z['frames'], T.FRAMES):
            raise ValueError('Label axes differ')
    obs = T.NR.OUT/'features/train'/f'unit{unit}.npz'
    obs_sha = sha(obs)
    if obs_sha != receipt['observation_sha256']:
        raise ValueError('Native/label observation differs')
    fp = FACTOR/'units'/f'unit{unit}.npz'
    fr = read(fp.with_suffix('.json'))
    aggregate_factor = read(FACTOR/'result.json')
    if fr['status'] != 'COMPLETE' or sha(fp) != fr['output_sha256'] or sha(fp.with_suffix('.json')) != aggregate_factor['unit_receipt_sha256'][str(unit)]:
        raise ValueError('Factorization category cache changed')
    with np.load(fp, allow_pickle=False) as z:
        category = z['category']
        exact_support = z['support'][..., 0].reshape(286, 2)
        if int(z['unit']) != unit or not np.array_equal(z['frames'], T.FRAMES) or category.shape != (22, 13, 2) or set(np.unique(category))-set(CATS[1:]):
            raise ValueError('Category identity/shape differs')
    inputs = dict(label_receipt_sha256=sha(folder/'receipt.json'), label_sha256=receipt['output_sha256'],
                  native_sha256=obs_sha, factor_sha256=sha(fp), factor_receipt_sha256=sha(fp.with_suffix('.json')))
    return paths, category.reshape(286, 2), exact_support, inputs


def group_mask(category, exact_support, name):
    if name == 'all':
        return np.ones(category.shape, bool)
    if name == GROUPS[-1]:
        return (category == 'contact0-2cm') & exact_support
    return category == name


def teacher_membership(net, td, transform):
    points = net.rays[None]*(td*T.RADIUS)[..., None]
    points = torch.einsum('bij,byxj->byxi', transform[:, :3, :3], points)+transform[:, None, None, :3, 3]
    return torch.stack([T.compact_membership(points, net.query_low[q], net.query_high[q]) for q in range(2)], 1)


def hybrid_features(net, pd, pv, td, tv, transform, base):
    return {name: net.alarm_features(d, v, transform, base) for name, d, v in
            [('PD_PV', pd, pv), ('TD_PV', td, pv), ('PD_TV', pd, tv), ('TD_TV', td, tv)]}


def batch_metrics(net, pd, pv, td, tv, transform, base):
    features = hybrid_features(net, pd, pv, td, tv, transform, base)
    reference = features['TD_TV'][..., :256]
    error = (pd-td).abs()*T.RADIUS
    pred_valid = pv >= .5
    valid = tv.bool()
    global_ray = torch.stack((torch.stack(((error*tv).double().sum(), tv.double().sum())),
                             torch.stack(((~pred_valid & valid).double().sum(), valid.double().sum())),
                             torch.stack(((pred_valid & ~valid).double().sum(), (~valid).double().sum()))))
    local = (teacher_membership(net, td, transform) > 0) & valid[:, None]
    local_n = local.double().sum((-2, -1))
    one = torch.ones_like(local_n)
    values = [(error[:, None]*local).double().sum((-2, -1)), ((~pred_valid[:, None]) & local).double().sum((-2, -1)),
              ((error[:, None] <= .02) & local).double().sum((-2, -1)), (local_n > 0).double()]
    denominators = [local_n, local_n, local_n, one]
    for name in HYBRIDS:
        mass = features[name][..., :256]
        difference = mass-reference
        values.extend([mass.double().mean(-1), difference.abs().double().mean(-1),
                       difference.clamp_min(0).double().mean(-1), (-difference).clamp_min(0).double().mean(-1)])
        denominators.extend([one]*4)
    native = base+net.alarm(features['PD_PV']).squeeze(-1)
    teacher = base+net.alarm(features['TD_TV']).squeeze(-1)
    values.extend([native.double(), teacher.double(), (teacher-native).double(), (teacher-native).abs().double()])
    denominators.extend([one]*4)
    for name in HYBRIDS:
        hybrid_logit = base+net.alarm(features[name]).squeeze(-1)
        values.append((hybrid_logit-native).double())
        denominators.append(one)
    return torch.stack((torch.stack(values, -1), torch.stack(denominators, -1)), -1), global_ray


def run(chunk):
    if (OUT/'result.json').exists():
        raise FileExistsError('Preserve completed result')
    if RUN_ID not in (Path(__file__).parent.parent/'RUNS.md').read_text(encoding='utf8'):
        raise RuntimeError('Parent RUNS preregistration required')
    k, n = map(int, chunk.split('/'))
    if not 0 <= k < n <= 96:
        raise ValueError('Expected zero-based chunk k/n')
    frozen = identity()
    runtime = T.ST.cuda_runtime()
    nets = {}
    try:
        for arm in T.ARMS:
            net = T.RaySurface().cuda().eval()
            net.load_state_dict(torch.load(T.OUT/'models'/arm/'model_seed0.pt', map_location='cuda', weights_only=True), strict=True)
            nets[arm] = net
        with np.load(T.ST.OUT/'m3_train_scores.npz', allow_pickle=False) as z:
            baseline = {u: z[str(u)] for u in UNITS[k::n]}
        for unit in UNITS[k::n]:
            paths, category, exact_support, inputs = unit_inputs(unit)
            binding = dict(identity_sha256=digest(frozen), inputs=inputs)
            path = OUT/'units'/f'unit{unit}.json'
            if path.exists():
                old = read(path)
                if old['status'] != 'COMPLETE' or old['binding'] != binding:
                    raise ValueError('Existing unit cache belongs to changed inputs')
                continue
            began = time.perf_counter()
            depth, valid, data = None, None, None
            try:
                depth, valid = (np.load(paths[name], mmap_mode='r') for name in ('depth.npy', 'valid.npy'))
                if depth.shape != (22, 13, 128, 128) or valid.shape != depth.shape or depth.dtype != np.float16 or valid.dtype != np.bool_:
                    raise ValueError('Dense teacher axes/dtype mismatch')
                data = T.NativeRayInputs('train', [unit])
                stats = np.zeros((2, len(GROUPS), len(METRICS), 2), np.float64)
                rays = np.zeros((2, len(RAY_METRICS), 2), np.float64)
                configs, frames = np.repeat(np.arange(22), 13), np.tile(T.FRAMES, 22)
                if baseline[unit].shape != (22, 13, 2) or not np.isfinite(baseline[unit]).all():
                    raise ValueError('Baseline axes/nonfinite')
                with torch.inference_mode():
                    for start in range(0, 286, 64):
                        c, f = configs[start:start+64], frames[start:start+64]
                        x, pose, transform = [torch.as_tensor(v, device='cuda') for v in data.batch(np.full(len(c), unit), c, f)]
                        td = torch.as_tensor(np.array(depth[c, f-3]), device='cuda').float()
                        tv = torch.as_tensor(np.array(valid[c, f-3]), device='cuda').float()
                        base = torch.as_tensor(baseline[unit][c, f-3], device='cuda')
                        if not bool(torch.isfinite(td).all()) or not bool(((td >= 0) & (td <= 1)).all()) or bool((td[tv == 0] != 0).any()):
                            raise ValueError('Teacher target invalid')
                        for ai, arm in enumerate(T.ARMS):
                            pd, vis = nets[arm].predict_rays(x, pose)
                            metrics, ray = batch_metrics(nets[arm], pd, vis.sigmoid(), td, tv, transform, base)
                            block = metrics.cpu().numpy()
                            if not np.isfinite(block).all():
                                raise FloatingPointError('Nonfinite diagnostic metric')
                            rays[ai] += ray.cpu().numpy()
                            for ci, cat in enumerate(GROUPS):
                                mask = group_mask(category[start:start+len(c)], exact_support[start:start+len(c)], cat)
                                stats[ai, ci] += block[mask].sum(0)
                counts = {cat: int((category == cat).sum()) for cat in CATS[1:]}
                result = dict(status='COMPLETE', unit=unit, binding=binding, category_counts=counts,
                              exact_support_counts={cat: int(((category == cat) & exact_support).sum()) for cat in CATS[1:]},
                              stats=stats.tolist(), ray_stats=rays.tolist(), elapsed_s=time.perf_counter()-began, runtime=runtime)
                save(path, result)
                print('COMPLETE geometry probe unit', unit, round(result['elapsed_s'], 2), flush=True)
            finally:
                if data is not None:
                    data.close()
                for array in (depth, valid):
                    if array is not None:
                        array._mmap.close()
        if identity() != frozen:
            raise ValueError('Frozen identity changed during run')
    finally:
        nets.clear()
        net = None
        gc.collect()
        torch.cuda.empty_cache()


def ratios(sums):
    return np.divide(sums[..., 0], sums[..., 1], out=np.full(sums.shape[:-1], np.nan), where=sums[..., 1] > 0)


def summarize(values, draws):
    total = values.sum(0)
    point = ratios(total)
    boot = ratios(values[draws].sum(1))
    out = {}
    for ai, arm in enumerate(T.ARMS):
        v = boot[:, ai]
        good = np.isfinite(v)
        out[arm] = dict(value=float(point[ai]) if np.isfinite(point[ai]) else None,
                       numerator=float(total[ai, 0]), denominator=float(total[ai, 1]),
                       ci95=np.quantile(v[good], [.025, .975]).tolist() if good.any() else None,
                       valid_bootstrap_draws=int(good.sum()))
    delta = boot[:, 1]-boot[:, 0]
    good = np.isfinite(delta)
    out['RAY_minus_BCEO'] = dict(value=float(point[1]-point[0]) if np.isfinite(point).all() else None,
                               ci95=np.quantile(delta[good], [.025, .975]).tolist() if good.any() else None,
                               valid_bootstrap_draws=int(good.sum()))
    return out


def finalize():
    if (OUT/'result.json').exists() or (OUT/'REPORT.md').exists():
        raise FileExistsError('Preserve completed diagnostic outputs')
    frozen = identity()
    units, hashes = [], {}
    for unit in UNITS:
        p = OUT/'units'/f'unit{unit}.json'
        r = read(p)
        _, _, _, inputs = unit_inputs(unit)
        if r['status'] != 'COMPLETE' or r['unit'] != unit or r['binding'] != dict(identity_sha256=digest(frozen), inputs=inputs):
            raise ValueError('Unit diagnostic is incomplete or stale')
        units.append(r)
        hashes[str(unit)] = sha(p)
    stats = np.asarray([r['stats'] for r in units])
    rays = np.asarray([r['ray_stats'] for r in units])
    if stats.shape != (96, 2, len(GROUPS), len(METRICS), 2) or rays.shape != (96, 2, 3, 2):
        raise ValueError('Unit aggregate shape mismatch')
    draws = np.random.default_rng(2026100301).integers(0, 96, (1000, 96))
    counts = {cat: sum(r['category_counts'][cat] for r in units) for cat in CATS[1:]}
    if sum(counts.values()) != 54912:
        raise ValueError('Missing training query frames')
    result = dict(status='COMPLETE', run=RUN_ID, identity=frozen, unit_sha256=hashes,
                  counts=dict(units=96, scenes=2112, frames=27456, queryframes=54912, physical_rayframes=27456*128*128),
                  category_counts=counts, ray_metrics={m: summarize(rays[:, :, i], draws) for i, m in enumerate(RAY_METRICS)},
                  exact_support={c: dict(n=sum(r['exact_support_counts'][c] for r in units), denominator=counts[c],
                     rate=sum(r['exact_support_counts'][c] for r in units)/counts[c] if counts[c] else None) for c in CATS[1:]},
                  categories={c: {m: summarize(stats[:, :, ci, mi], draws) for mi, m in enumerate(METRICS)} for ci, c in enumerate(GROUPS)},
                  definitions=dict(local_visible='Teacher valid and soft query membership >0, including the fixed +/-1cm edge band; unweighted physical-ray MAE/FN',
                    geometric_mass='Original 64 positive ray weights normalize each8x8 microcell; equal mean of256 microcells, not physical area/probability or false stop rate',
                    substitution='Four combinations replace depth/visibility-derived head features, holding query embedding, M3 and frozen head fixed; oracle intervention and distribution shift, neither performance upper bound nor deployment gain',
                    labels='Training-only stored float16 normalized radial depth and separate bool valid; MAE in meters relative to half labels, not exact unquantized rays',
                    bootstrap='1000 paired whole-unit resamples seed2026100301; pooled numerator/denominator ratio; empty denominators omitted from CI, counted explicitly',
                    limit='Consumed Development, single seed, training fit only. No cal/eval, new observation, retraining or threshold search.'))
    save(OUT/'result.json', result)
    lines = ['TRAINING_ONLY_GEOMETRY_DIAGNOSTIC — 描述性结果，无成功判据。', '',
             '96训练单位、2112场景、27456帧、54912 query-frame。物理ray统计每帧仅计一次；标签为既存half径向深度及独立valid。', '']
    lines += ['旧缓存EXACT可见采样支持（与teacher half深度的soft支持不同）：']
    lines += [f"{c}: {r['n']}/{r['denominator']}" for c, r in result['exact_support'].items()]
    lines += ['', 'within2cm为teacher局部ray径向误差≤2cm占比，不是及时停率。', '']
    for title, metrics in [('physical rays', result['ray_metrics'])]+list(result['categories'].items()):
        lines += [title+('（query-frame分母 '+str(counts[title])+'）' if title in counts else ''), '',
                  '| 指标 | BCEO [95% CI] /分母 | RAY [95% CI] /分母 | RAY−BCEO [95% CI] |', '|---|---|---|---|']
        for name, metric in metrics.items():
            cells = []
            for arm in (*T.ARMS, 'RAY_minus_BCEO'):
                v = metric[arm]
                s = 'NA' if v['value'] is None else f"{v['value']:.7g} [{v['ci95'][0]:.7g}, {v['ci95'][1]:.7g}]"
                cells.append(s+(f" /{v['denominator']:g}" if 'denominator' in v else ''))
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines += [f'{k}: {v}' for k, v in result['definitions'].items()]
    with (OUT/'REPORT.md').open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines)+'\n')
    return result


def check():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    net = T.RaySurface().eval()
    td = torch.rand(2, 128, 128)
    tv = (torch.rand_like(td) > .25).float()
    td *= tv
    transform = torch.eye(4)[None].repeat(2, 1, 1)
    base = torch.randn(2, 2)
    with torch.inference_mode():
        h = hybrid_features(net, td, tv, td, tv, transform, base)
        for name in HYBRIDS:
            torch.testing.assert_close(h[name], h['TD_TV'], rtol=0, atol=0)
        metrics, rays = batch_metrics(net, td, tv, td, tv, transform, base)
        for name in HYBRIDS:
            for suffix in ('L1_vs_TD_TV', 'added_mass', 'lost_mass'):
                assert not bool(metrics[..., METRICS.index(name+'_'+suffix), 0].any())
        assert rays[0, 0] == rays[1, 0] == rays[2, 0] == 0
        torch.testing.assert_close(T.micro_integral(torch.ones_like(td), net.weights), torch.ones(2, 16, 16), atol=2e-7, rtol=0)
        assert not bool(metrics[..., METRICS.index('teacher_minus_native_logit'), 0].any())
    synthetic = np.ones((96, 2, 2))
    synthetic[:, 1, 0] = 2
    r = summarize(synthetic, np.random.default_rng(2026100301).integers(0, 96, (1000, 96)))
    assert r['RAY_minus_BCEO']['ci95'] == [1., 1.]
    print(json.dumps(dict(status='PASS', synthetic_only=True, source_sha256=sha(__file__),
                         checks=['all hybrid identity', 'zero teacher L1/added/lost', 'physical ray zero error',
                                 'micro integration', 'oracle head identity', 'paired whole-unit ratios']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['check', 'run', 'finalize'])
    p.add_argument('--chunk', default='0/1')
    a = p.parse_args()
    if a.stage == 'check':
        check()
    elif a.stage == 'run':
        run(a.chunk)
    else:
        finalize()

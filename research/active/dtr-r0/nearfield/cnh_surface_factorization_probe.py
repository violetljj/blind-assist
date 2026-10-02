"""Training-only oracle audit of radial quantization versus angle/range marginals.

No models, calibration/evaluation units, observation synthesis, or decision gate.
Mass and L1 use equal-microzone means, not full-FOV solid angle or area.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import cnh_surface_distribution_data as D
import cnh_near_range as NR
import cnh_proposal_attribution_scenes as S
import cnh_sequence_observed_geometry as G

OUT = NR.SS.WORK / 'cnh-surface-factorization-probe-20261002'
RUN_ID = 'CNH_SURFACE_FACTORIZATION_PROBE_20261002'
UNITS = list(range(93000, 93096))
FRAMES = np.arange(3, 16)
ARMS = ('EXACT', 'JOINT_BIN', 'MARGINAL')
ERRORS = ('radial_quantization', 'angle_range_marginalization', 'total')
CATS = G.CATEGORIES
TOL, SEED, BOOT = 1e-12, 2026100233, 1000


def identity():
    paths = [Path(__file__), Path(D.__file__), Path(G.__file__), Path(NR.__file__), Path(S.__file__),
             Path(NR.SS.__file__), *[S.SOURCE / n for n in
             ('cnh_route_sensor.py', 'cnh_track_a_geometry.py', 'cnh_track_a_fov.py', 'cnh_track_a_v13_sensor.py')]]
    return dict(source_sha256={str(p): D.sha(p) for p in paths}, units=UNITS,
                frames=FRAMES.tolist(), configs=22, arms=ARMS, errors=ERRORS,
                radial_width=D.RAW_WIDTH, radial_bins=128, query_low=D.QUERY_LOW.tolist(),
                query_high=D.QUERY_HIGH.tolist(), query_eps=D.EPS, support_tolerance=TOL,
                mass_unit='equal-microzone mean mass', arithmetic='float64 histogram and public angular weights',
                bootstrap_seed=SEED, bootstrap_replicates=BOOT)


def identity_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ray_membership(radial, rays, transform):
    """Finite positive true distance, exact same-ray point and strict public box."""
    radial = np.asarray(radial, float)
    valid = np.isfinite(radial) & (radial > 0)
    directions = rays @ transform[:3, :3].T
    points = directions * np.where(valid, radial, 0)[..., None] + transform[:3, 3]
    return np.stack([valid & ((points >= lo+D.EPS) & (points <= hi-D.EPS)).all(-1)
                     for lo, hi in zip(D.QUERY_LOW, D.QUERY_HIGH)])


def public_weights64(transform):
    """Same fixed quadrature/bin-center formula as D; avoid float32 rounding."""
    rays, weights = D.public_rays()
    directions = rays @ transform[:3, :3].T
    radius = (np.arange(128)+.5)*D.RAW_WIDTH
    result = np.zeros((2, 16, 16, 129), np.float64)
    for query in range(2):
        inside = np.ones((16, 16, 64, 128), bool)
        for axis in range(3):
            coordinate = directions[..., axis, None]*radius + transform[axis, 3]
            inside &= (coordinate >= D.QUERY_LOW[query, axis]+D.EPS) & (coordinate <= D.QUERY_HIGH[query, axis]-D.EPS)
        result[query, ..., :128] = np.einsum('yxrb,yxr->yxb', inside, weights)
    return result


def masses(radial, transform, public_weights):
    rays, ray_weights = D.public_rays()
    radial = np.asarray(radial, np.float64)
    exact_membership = ray_membership(radial, rays, transform)
    if np.any(exact_membership & (radial[None] >= D.RAW_WIDTH*128)):
        raise AssertionError('Out-of-domain ray enters query: cannot call this pure radial quantization')
    exact = np.sum(exact_membership*ray_weights[None], -1)
    valid = np.isfinite(radial) & (radial > 0) & (radial < D.RAW_WIDTH*128)
    centers = np.full(radial.shape, np.inf)
    centers[valid] = (np.floor(radial[valid]/D.RAW_WIDTH)+.5)*D.RAW_WIDTH
    joint = np.sum(ray_membership(centers, rays, transform)*ray_weights[None], -1)
    # Float64 histogram is intentional: half-storage quantization is not this audit.
    distribution = D.distribution_from_radial(radial)
    marginal = (distribution[None]*np.asarray(public_weights, np.float64)).sum(-1)
    return np.stack((exact, joint, marginal), -1)  # query,y,x,arm


def frame_metrics(radial, transform, public_weights):
    m = masses(radial, transform, public_weights)
    if np.any((m[..., 1] > TOL) & (m[..., 2] <= TOL)):
        raise AssertionError('Positive JOINT_BIN support must survive positive-weight marginalization')
    errors = np.stack((np.abs(m[..., 1]-m[..., 0]), np.abs(m[..., 2]-m[..., 1]),
                       np.abs(m[..., 2]-m[..., 0])), -1).mean((1, 2))
    return dict(mass=m.mean((1, 2)), errors=errors,
                support=np.any(m > TOL, axis=(1, 2)),
                unsupported_mass=(m*(m[..., :1] <= TOL)).mean((1, 2)))


def run_unit(unit, frozen, weight_cache):
    folder = OUT / 'units'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'unit{unit}.npz'
    receipt = path.with_suffix('.json')
    digest = identity_sha(frozen)
    if receipt.exists():
        old = D.read(receipt)
        if old['status'] != 'COMPLETE' or old['identity_sha256'] != digest or old['output_sha256'] != D.sha(path):
            raise ValueError('Existing unit cache changed or belongs to another source identity')
        return old
    if path.exists():
        raise FileExistsError('Partial unit cache requires inspection')
    began = time.monotonic()
    scenes = NR.scenes_for(unit)
    if len(scenes) != 22 or [s['config'] for s in scenes] != list(range(22)):
        raise ValueError('Training scene identity mismatch')
    shape = (22, 13, 2, 3)
    arrays = {k: np.empty(shape, np.float64) for k in ('mass', 'errors', 'unsupported_mass')}
    arrays['support'] = np.empty(shape, bool)
    arrays['category'] = np.empty((22, 13, 2), '<U16')
    arrays['target_front_range'] = np.empty((22, 13), np.float64)
    rays, _ = D.public_rays()
    for config, scene in enumerate(scenes):
        triangles = np.concatenate([S.box_mesh(b['lo'], b['hi']) for b in scene['boxes']])
        target = G.corners(scene['boxes'][0])
        for fi, frame in enumerate(FRAMES):
            sensor, travel = scene['poses'][frame], scene['travel'][frame]
            transform = np.linalg.inv(travel) @ sensor
            key = (unit % 3, int(frame))
            if key not in weight_cache:
                weight_cache[key] = (transform, public_weights64(transform))
            elif not np.allclose(weight_cache[key][0], transform, rtol=0, atol=1e-12):
                raise ValueError('Public transform is not shared within mode/frame')
            hits = S.raycast_boxes(sensor[:3, 3], rays@sensor[:3, :3].T, scene['boxes'])
            values = frame_metrics(hits['distance'], transform, weight_cache[key][1])
            for name, value in values.items():
                arrays[name][config, fi] = value
            local = G.transform(triangles, travel)
            categories = [G.surface_category(local, q) for q in range(2)]
            arrays['category'][config, fi] = categories
            arrays['target_front_range'][config, fi] = G.target_front(target, travel)
            if any(values['support'][q, 0] and not categories[q].startswith('contact') for q in range(2)):
                raise AssertionError('Exact visible query support must imply all-object surface contact')
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays, unit=np.asarray(unit), configs=np.arange(22), frames=FRAMES,
                            arms=np.asarray(ARMS), error_names=np.asarray(ERRORS))
    result = dict(status='COMPLETE', unit=unit, identity_sha256=digest, output_sha256=D.sha(path),
                  elapsed_s=time.monotonic()-began, queryframes=572, scenes=22)
    D.create_json(receipt, result)
    print('COMPLETE unit', unit, round(result['elapsed_s'], 3), 's', flush=True)
    return result


def run(chunk):
    if (OUT / 'result.json').exists():
        raise FileExistsError('Preserve COMPLETE probe; no rerun/overwrite')
    if RUN_ID not in (Path(__file__).parent.parent / 'RUNS.md').read_text(encoding='utf8'):
        raise RuntimeError('Parent RUNS predeclaration required')
    k, n = map(int, chunk.split('/'))
    if not 0 <= k < n <= 96:
        raise ValueError('Chunk must be zero-based k/n')
    frozen, cache = identity(), {}
    for unit in UNITS[k::n]:
        run_unit(unit, frozen, cache)
    if frozen != identity():
        raise ValueError('Source changed during chunk')
    print('COMPLETE chunk', chunk, flush=True)


def metric_summary(mask, mass, support, unsupported):
    n = int(mask.sum())
    return dict(n=n, arms={arm: dict(support_n=int(support[..., a][mask].sum()),
                support_rate=float(support[..., a][mask].mean()) if n else None,
                mean_mass=float(mass[..., a][mask].mean()) if n else None,
                mean_mass_in_exact_unsupported_microcells=float(unsupported[..., a][mask].mean()) if n else None)
                for a, arm in enumerate(ARMS)})


def summarize(arrays):
    errors = arrays['errors']
    unit_errors = errors.mean(axis=(1, 2, 3))
    if unit_errors.shape != (96, 3):
        raise ValueError('Expected 96 complete unit blocks')
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, 96, size=(BOOT, 96))
    boot = unit_errors[draws].mean(1)
    point = unit_errors.mean(0)
    ci = np.quantile(boot, [.025, .975], axis=0)
    diff = boot[:, 1]-boot[:, 0]
    overall = {name: dict(mean=float(point[i]), ci95=ci[:, i].tolist()) for i, name in enumerate(ERRORS)}
    overall['angle_minus_quantization'] = dict(mean=float(point[1]-point[0]), ci95=np.quantile(diff, [.025, .975]).tolist())
    overall['marginal_total_minus_joint_total'] = dict(mean=float(point[2]-point[0]),
        ci95=np.quantile(boot[:, 2]-boot[:, 0], [.025, .975]).tolist())
    cat = arrays['category']
    frame = {name: metric_summary(cat == name, arrays['mass'], arrays['support'], arrays['unsupported_mass']) for name in CATS}
    # Conditional class means pool the class's frames, while resampling all 96
    # whole units together. Empty class/unit cells contribute zero numerator
    # and zero denominator; no hand-picked unit subset is used.
    conditional = {}
    for name in CATS:
        mask = cat == name
        counts = mask.sum((1, 2, 3))
        measures = dict(marginal_total_minus_joint_total=errors[..., 2]-errors[..., 0])
        for arm, ai, ei in [('JOINT_BIN', 1, 0), ('MARGINAL', 2, 2)]:
            signed = arrays['mass'][..., ai]-arrays['mass'][..., 0]
            measures[arm+'_signed_mass_minus_exact'] = signed
            measures[arm+'_added_mass_vs_exact'] = (errors[..., ei]+signed)/2
            measures[arm+'_removed_mass_vs_exact'] = (errors[..., ei]-signed)/2
        record = dict(n_queryframes=int(counts.sum()), contributing_units=int((counts > 0).sum()), metrics={})
        for key, values in measures.items():
            sums = (values*mask).sum((1, 2, 3))
            denominator = counts[draws].sum(1)
            b = np.divide(sums[draws].sum(1), denominator, out=np.full(BOOT, np.nan), where=denominator > 0)
            valid = np.isfinite(b)
            record['metrics'][key] = dict(mean=float(sums.sum()/counts.sum()) if counts.sum() else None,
                ci95=np.quantile(b[valid], [.025, .975]).tolist() if valid.any() else None,
                valid_bootstrap_draws=int(valid.sum()))
        conditional[name] = record
    # Episode label is the deepest all-object category appearing in 13 frames.
    severity = {name: v for v, name in enumerate(('clear', 'pass0-10cm', 'contact0-2cm', 'contact2-5cm', 'contact>5cm'))}
    code = np.zeros(cat.shape, np.int8)
    for name, value in severity.items():
        code[cat == name] = value
    episode_code = code.max(2)
    episode = {name: metric_summary(episode_code == severity[name], arrays['mass'].mean(2),
               arrays['support'].any(2), arrays['unsupported_mass'].mean(2)) for name in CATS}
    assert np.array_equal(episode_code == 0, np.all(cat == 'clear', axis=2))
    return dict(unit_mean_L1=overall, frame_categories=frame, episode_categories=episode,
                category_error_changes=conditional,
                counts=dict(units=96, scenes=2112, queryepisodes=4224, queryframes=54912),
                episode_definition='Deepest full-surface category across 13 frames; clear only if all 13 frames clear; support is any of 13 sampled frames',
                mass_definition='Each microzone normalizes its 64 solid-angle ray weights; cross-microzone reduction is equal mean over 256 cells, not FOV area/probability',
                unsupported_mass_definition='Mass assigned to microcells with no EXACT sampled visible-query support; contact categories may also contain these cells. For pass/clear, all query mass is spurious relative to current full-surface query contact.',
                boundary='Oracle first-visible geometry only; no model, low-SNR, device UNKNOWN, hardware or product safety inference. Zero visible mass does not establish clear space.',
                support_invariant='Positive JOINT_BIN implies positive MARGINAL with the same positive ray weights. Marginalization alone cannot erase zero-threshold joint support; it can dilute amplitude or add support. This does not predict a learned operating-threshold alarm.',
                out_of_domain_query_hits=0,
                out_of_domain_check='Every frame asserts no exact query member has radius >=128*binwidth; therefore J-E contains no query-support removal by radial truncation.',
                bootstrap='1000 paired whole-unit draws, seed2026100233; all 22 configs x13 frames x2 queries included; no success gate')


def write_report(result):
    lines = ['ORACLE_GEOMETRY_DIAGNOSTIC — 描述性拆分，无成功判据。', '',
             '仅原训练96单位、2112场景、54912 query-frame；未读取校准/评价或模型，未渲染。', '',
             '质量和L1均为256微格等权均值，不是整个视场固体角概率或物理面积。', '',
             '| 误差 | 单位均匀均值 | 单位bootstrap 95%区间 |', '|---|---:|---|']
    for key, value in result['unit_mean_L1'].items():
        lines.append(f"| {key} | {value['mean']:.8f} | [{value['ci95'][0]:.8f}, {value['ci95'][1]:.8f}] |")
    lines += ['', '按当前全物体类别条件汇总误差变化（帧均值，全部96单位共同重采样；正负质量由逐微格L1与有符号质量差拆分）：', '',
              '| 类别 | query-frame /单位 | 指标 | 均值 | 配对单位95%区间 |', '|---|---:|---|---:|---|']
    for category, record in result['category_error_changes'].items():
        for key, value in record['metrics'].items():
            mean = 'NA' if value['mean'] is None else f"{value['mean']:.8f}"
            ci = 'NA' if value['ci95'] is None else f"[{value['ci95'][0]:.8f}, {value['ci95'][1]:.8f}]"
            lines.append(f"| {category} | {record['n_queryframes']} / {record['contributing_units']} | {key} | {mean} | {ci} |")
    for level in ('frame', 'episode'):
        lines += ['', f'{level}分类：episode按13帧最深类别归组；清晰须13帧全清晰，支持取任一帧。', '',
                  '| 类别 | 分母 | 臂 | 支持n/分母 | 平均质量 | EXACT不支持微格内质量 |', '|---|---:|---|---:|---:|---:|']
        for category, record in result[level+'_categories'].items():
            for arm, v in record['arms'].items():
                fmt = lambda x: 'NA' if x is None else f'{x:.8f}'
                lines.append(f"| {category} | {record['n']} | {arm} | {v['support_n']}/{record['n']} | {fmt(v['mean_mass'])} | {fmt(v['mean_mass_in_exact_unsupported_microcells'])} |")
    lines += ['', 'frame质量为当前帧；episode质量为13帧均值而非最大值。EXACT不支持微格内质量仅描述采样几何差异；擦身/清晰类别的全部query质量均为相对全物体contact的虚假质量。', '',
              '相同正权射线集合下，JOINT_BIN正支持必然保留为MARGINAL正支持。边缘化可稀释幅度、增加虚假支持，不能单独删除零阈值下的支持；不能据此把全部误差解释为漏检。L1增量允许为负，反映误差抵消。', '',
              '全物体surface类别包含遮挡表面；可见采样支持与它分开统计。零可见质量不证明清晰，第129类仅几何无命中/域外，不是设备低SNR或产品UNKNOWN。未改动任何既有实验结论。']
    path = OUT / 'REPORT.md'
    with path.open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines)+'\n')


def finalize():
    if (OUT / 'result.json').exists() or (OUT / 'REPORT.md').exists():
        raise FileExistsError('Preserve existing completed probe outputs')
    frozen = identity()
    parts, receipts = [], {}
    for unit in UNITS:
        path = OUT / 'units' / f'unit{unit}.npz'
        receipt = D.read(path.with_suffix('.json'))
        if receipt['status'] != 'COMPLETE' or receipt['identity_sha256'] != identity_sha(frozen) or receipt['output_sha256'] != D.sha(path):
            raise ValueError('Unit cache not complete/current')
        with np.load(path, allow_pickle=False) as z:
            if int(z['unit']) != unit or not np.array_equal(z['frames'], FRAMES) or not np.array_equal(z['configs'], np.arange(22)):
                raise ValueError('Unit axes mismatch')
            parts.append({k: z[k] for k in ('mass', 'errors', 'support', 'unsupported_mass', 'category', 'target_front_range')})
        receipts[str(unit)] = D.sha(path.with_suffix('.json'))
    arrays = {key: np.stack([p[key] for p in parts]) for key in parts[0]}
    result = dict(status='COMPLETE', run=RUN_ID, identity=frozen, unit_receipt_sha256=receipts, **summarize(arrays))
    D.create_json(OUT / 'result.json', result)
    write_report(result)
    print(json.dumps(result['unit_mean_L1'], indent=2), flush=True)
    return result


def check():
    started = time.perf_counter()
    # Two angle/range pairings have the same marginal histogram but different
    # true query masses; all rays remain in the same conceptual microcell.
    rays2 = np.array([[.1, 0, 1.], [.3, 0, 1.]])
    rays2 /= np.linalg.norm(rays2, axis=1, keepdims=True)
    a, b = np.array([1., 2.]), np.array([2., 1.])
    wa = ray_membership(a, rays2, np.eye(4))[0].mean()
    wb = ray_membership(b, rays2, np.eye(4))[0].mean()
    assert np.array_equal(np.sort(a), np.sort(b)) and wa == .5 and wb == 1.
    # Identical radial bin for all rays makes the public angular integral and
    # same-ray JOINT_BIN identical apart from the public float32 output.
    t = time.perf_counter()
    q = public_weights64(np.eye(4))
    query_s = time.perf_counter()-t
    public32 = D.query_weights(np.eye(4))
    assert np.max(np.abs(q-public32)) < 5e-8
    radial = np.full((16, 16, 64), (40+.5)*D.RAW_WIDTH)
    m = masses(radial, np.eye(4), q)
    np.testing.assert_allclose(m[..., 0], m[..., 1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(m[..., 1], q[..., 40], rtol=0, atol=1e-14)
    np.testing.assert_allclose(m[..., 2], m[..., 1], rtol=0, atol=1e-14)
    varied = np.broadcast_to(np.linspace(.35, 3.5, 64), (16, 16, 64)).copy()
    varied_mass = masses(varied, np.eye(4), q)
    assert np.any(varied_mass[..., 1] > TOL)
    assert np.all(varied_mass[..., 2][varied_mass[..., 1] > TOL] > TOL)
    for value in (np.inf, D.RAW_WIDTH*128):
        assert not masses(np.full(radial.shape, value), np.eye(4), q).any()
    # Simple physical front surface: any exact inside point implies a full
    # all-object contact category, with no nominal target-offset shortcut.
    boxes = [dict(lo=[-.1, -.1, 1.], hi=[.1, .1, 1.1], rho=.4)]
    rays, _ = D.public_rays()
    t = time.perf_counter()
    hit = S.raycast_boxes(np.zeros(3), rays, boxes)['distance']
    met = frame_metrics(hit, np.eye(4), q)
    local = np.concatenate([S.box_mesh(x['lo'], x['hi']) for x in boxes])
    cats = [G.surface_category(local, k) for k in range(2)]
    assert met['support'][0, 0] and all(not met['support'][k, 0] or cats[k].startswith('contact') for k in range(2))
    frame_s = time.perf_counter()-t
    fixture = dict(errors=np.broadcast_to([.02, .08, .06], (96, 22, 13, 2, 3)).copy(),
                   mass=np.zeros((96, 22, 13, 2, 3)), support=np.zeros((96, 22, 13, 2, 3), bool),
                   unsupported_mass=np.zeros((96, 22, 13, 2, 3)),
                   category=np.full((96, 22, 13, 2), 'clear', dtype='<U16'))
    fixture['category'][:, 0, 0, 0] = 'contact0-2cm'
    summary = summarize(fixture)
    assert summary['frame_categories']['contact0-2cm']['n'] == 96
    assert summary['episode_categories']['clear']['n'] == 4224-96
    np.testing.assert_allclose(summary['category_error_changes']['contact0-2cm']['metrics']['marginal_total_minus_joint_total']['ci95'], [.04, .04])
    result = dict(status='PASS', synthetic_only=True, source_sha256=D.sha(__file__),
                  checks=['same histogram different pairing mass', 'same-ray bincenter/public integral parity',
                          'unknown zero mass', 'exact support implies surface contact', 'joint positive support retained by marginal',
                          'float64 public integral agrees with frozen float32 within5e-8', 'no out-of-domain exact query members',
                          'whole-unit paired bootstrap and all13frames-clear episode fixture'],
                  timing=dict(query_table_s=query_s, one_box_frame_s=frame_s, elapsed_s=time.perf_counter()-started),
                  estimate=dict(serial_seconds_optimistic=frame_s*96*22*13+query_s*39,
                                caveat='Synthetic one-box timing only; real train scenes have multiple boxes and category clips. Budget several-fold this lower estimate; two chunks can overlap CPU work.'))
    print(json.dumps(result, indent=2), flush=True)
    return result


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

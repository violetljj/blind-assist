"""Paired, deterministic native-return spot checks on consumed Development.

No sampling, training, model inference, GPU or M3 voxel reconstruction. Truth
and authoring geometry select diagnostic cases only; summary() accepts native
observations and public query/sensor transforms, never object geometry. A peak
is an observed peak, with no target attribution or free-space implication.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'artifacts.local/work/cnh-counterfactual-dev-20261009'
OUTPUT = ROOT / 'artifacts.local/work/cnh-graded-evidence-dev-20261010/raw_echo'
BIAS = ROOT / 'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy'
SEEDS = (2026100955, 2026100956, 2026100957)
WIDTH, EDGE = 8 * .0375348, np.tan(np.pi / 8)
BOXES = np.array([[-.30, -.20, .30, .30, .42, 3.],
                  [-.30, .42, .30, .30, .90, 3.]])
HEIGHTS = ('HEAD', 'BODY')
FRAMES = tuple(range(3, 16))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 23), b''):
            digest.update(block)
    return digest.hexdigest()


def smooth(raw):
    result = np.empty_like(raw, dtype=np.float64)
    for j in range(13):
        start = max(0, j - 4)
        weights = 2. ** np.arange(j - start + 1)
        result[..., j, :] = (raw[..., start:j+1, :] * weights[:, None]).sum(-2) / weights.sum()
    return result


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def points(mode):
    offsets = np.array([.5] if mode == 'center' else [1/6, .5, 5/6], np.float32)
    slopes = -EDGE + (np.arange(8, dtype=np.float32)[:, None] + offsets) * (2*EDGE/8)
    y = np.broadcast_to(slopes[:, None, :, None], (8, 8, len(offsets), len(offsets)))
    x = np.broadcast_to(slopes[None, :, None, :], y.shape)
    rays = np.stack((x, y, np.ones_like(x)), -1).reshape(8, 8, -1, 3)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    radius = (np.arange(16, dtype=np.float32) + .5) * WIDTH
    return rays[:, :, None] * radius[None, None, :, None, None]


def membership(transforms, query_box, mode):
    """Small native-bin query proxy, same soft-box formula as token geometry.

    Angular node/bin-center hypotheses, not finite-volume M3 reconstruction.
    No scene/object/target parameters are accepted.
    """
    t = np.asarray(transforms, np.float32)
    xyz = np.einsum('tij,yxrnj->tyxrni', t[:, :3, :3], points(mode))
    xyz += t[:, None, None, None, None, :3, 3]
    box = np.asarray(query_box, np.float32)
    outside = np.maximum(box[:3]-xyz, 0) + np.maximum(xyz-box[3:], 0)
    return np.exp(-outside.sum(-1)/np.float32(.05)).mean(-1), xyz


def local_peaks(z):
    absent = np.full((*z.shape[:-1], 1), -np.inf)
    left, right = np.concatenate((absent, z[..., :-1]), -1), np.concatenate((z[..., 1:], absent), -1)
    return (z >= 3) & (z > left) & (z >= right)


def declare(output, wall_cap, sampling):
    plan = dict(task='CNH_GRADED_RAW_ECHO_DEV_20261010', lane='EXPLORE consumed simulated Development',
        goal='Locate native weak/thin missed support with detected contact, same-background pass and clear controls',
        budget=dict(CPU_IO_engineering_wall_soft_seconds=wall_cap, GPU_seconds=0),
        scope='Existing validation ideal ordinary arm, all three seed outputs; frozen M3/old5 retained; no new draws/train/infer',
        sampling='For each shape_family x HEAD/BODY, choose lexicographically first (scene_id,replica) contact '
                 'that is thin (minimum target dimension<=.04m) or weak (rho<=.25), and not timely for all three '
                 'ordinary seeds under their existing cal-only standalone thresholds. Maximum eight anchors. '
                 'Truth/authoring boxes select diagnostic samples only. This is purposive diagnosis, not prevalence estimation.',
        controls='Per anchor: contact timely for all three seeds; pass at anchor query; jointly-clear. '
                 'Require same background_id and replica; rank by shape/group/side/rho mismatch count, then scene_id. '
                 'Record mismatches or missing controls; no manufactured matched scene. Prefer a present clear over absent '
                 'only if matching rank dictates. Categories include all physical boxes, not only target.',
        time='Report all f3..15, timely f3..13, 8-frame history max(0,f-7)..f; normalization FP32 then inherited FP16',
        observations='One copied hist.npy sample slice reused for raw peaks and native center/extent query membership summaries. '
                     'Same public query[f] @ inv(sensor[f]) @ sensor[history] as inherited pipeline. '
                     'Query H/B boxes fixed; expanded x+/- .40m is a diagnostic existence-support proxy, not trained existence axis.',
        peaks='z>=3 and z>left and z>=right; threshold inherited bin-summary decoder, not fitted here. '
              'Top three local peaks per frame and radial band [0,3] or >3m, ordered descending z then zone/bin; '
              'bin-center distance is coarse radial observation, no strongest-peak target attribution.',
        comparison='For each matched role/frame, compare native near peak and center/extent gated history peak/signedlog support '
                   'to anchor. Contact versus controls is descriptive; photons differ. No obstacle-minus-background signal truth.',
        limitations='No independent object-return truth. Native or gated peaks do not establish target visibility; far returns '
                    'only indicate background-compatible evidence, never free. Sample verdict may be EVIDENCE_INSUFFICIENT. '
                    'No absence-of-peaks conclusion closes grading or peak-preservation work.',
        deliverables='selected_cases.csv, sampled_observations.npz, frames.csv, peaks.csv, matched_differences.csv, '
                     'result.json, REPORT.md, receipt with commands/time/hash; failure lineage preserved',
        source_sha256=sha(__file__))
    if sampling == 'layered':
        plan.update(sampling='For each validation background_family x HEAD/BODY x all-seed-miss/all-seed-success, '
            'choose lexicographically first (scene_id,replica) thin or weak contact. Eight strata, at most eight anchors. '
            'All-seed outcomes select diagnostics only; no runtime multi-model combination. This repairs observed '
            'single-background coverage in v1/v2; not weak-peak winner selection.',
            controls='Per anchor select counterpart contact outcome, pass and jointly-clear. Require same background_id, '
            'same group, side and replica index. Rank shape/rho metadata mismatch count then scene_id; report differences '
            'including target dimensions. No manufacture of a strict match. Photons are separate draws.',
            sampling_variant='layered_background_height_outcome')
    save(output/'PLAN.json', plan)
    return plan


def choose_cases(rows, category, timely):
    candidates, anchors, strata = [], [], []
    for family in sorted({r['shape_family'] for r in rows}):
        for q in (0, 1):
            valid = []
            for i, row in enumerate(rows):
                thickness = np.min(np.asarray(row['target_box']['hi']) - row['target_box']['lo'])
                if row['shape_family'] == family and category[i, q] == 'contact' and (thickness <= .04000001 or row['rho'] <= .25):
                    valid += [(i, k) for k in range(timely.shape[2]) if not timely[:, i, k, q].any()]
            strata.append(dict(family=family, height=HEIGHTS[q], eligible_all_seed_misses=len(valid)))
            if valid:
                i, k = sorted(valid)[0]
                anchors.append((i, k, q))
    for aid, (i, k, q) in enumerate(anchors):
        anchor = rows[i]
        for role in ('miss_contact', 'detected_contact', 'pass', 'clear'):
            if role == 'miss_contact':
                selected, mismatch = i, []
            else:
                pool = []
                for j, row in enumerate(rows):
                    if j == i or row['background_id'] != anchor['background_id']:
                        continue
                    eligible = ((category[j, q] == 'contact' and timely[:, j, k, q].all()) if role == 'detected_contact'
                                else category[j, q] == 'pass' if role == 'pass' else (category[j] == 'clear').all())
                    if eligible:
                        differing = [n for n in ('shape_family', 'group', 'side', 'rho') if row[n] != anchor[n]]
                        pool.append((len(differing), j, differing))
                if not pool:
                    candidates.append(dict(anchor_id=aid, role=role, scene=-1, replica=k, query=q,
                                           match_status='NOT_EVALUABLE_NO_SAME_BACKGROUND_CONTROL', mismatch_fields=''))
                    continue
                _, selected, mismatch = sorted(pool)[0]
            candidates.append(dict(anchor_id=aid, role=role, scene=selected, replica=k, query=q,
                                   match_status='MATCHED_BACKGROUND', mismatch_fields=';'.join(mismatch)))
    return candidates, strata


def choose_layered(rows, category, timely):
    cases, strata = [], []
    for background in sorted({r['background_family'] for r in rows}):
        for q in (0, 1):
            for state in ('miss', 'detected'):
                valid = []
                for i, row in enumerate(rows):
                    dims = np.asarray(row['target_box']['hi']) - row['target_box']['lo']
                    if row['background_family'] != background or category[i, q] != 'contact' or not (dims.min() <= .04000001 or row['rho'] <= .25):
                        continue
                    for k in range(timely.shape[2]):
                        outcome = not timely[:, i, k, q].any() if state == 'miss' else timely[:, i, k, q].all()
                        if outcome:
                            valid.append((i, k))
                strata.append(dict(background_family=background, height=HEIGHTS[q], outcome=state, eligible=len(valid)))
                if not valid:
                    continue
                i, k = sorted(valid)[0]
                anchor, aid = rows[i], len({c['anchor_id'] for c in cases})
                anchor_role = 'miss_contact' if state == 'miss' else 'detected_contact'
                for role in ('miss_contact', 'detected_contact', 'pass', 'clear'):
                    if role == anchor_role:
                        selected, mismatch = i, []
                    else:
                        pool = []
                        for j, row in enumerate(rows):
                            if j == i or any(row[n] != anchor[n] for n in ('background_id', 'group', 'side')):
                                continue
                            eligible = ((category[j, q]=='contact' and timely[:, j, k, q].all()) if role=='detected_contact'
                                else (category[j, q]=='contact' and not timely[:, j, k, q].any()) if role=='miss_contact'
                                else category[j, q]=='pass' if role=='pass' else (category[j]=='clear').all())
                            if eligible:
                                mismatch = [n for n in ('shape_family', 'rho') if row[n] != anchor[n]]
                                dims_a = np.asarray(anchor['target_box']['hi'])-np.asarray(anchor['target_box']['lo'])
                                dims_b = np.asarray(row['target_box']['hi'])-np.asarray(row['target_box']['lo'])
                                extra = ['target_dimensions'] if not np.allclose(dims_a, dims_b, atol=1e-8, rtol=0) else []
                                pool.append((len(mismatch), j, mismatch+extra))
                        if not pool:
                            cases.append(dict(anchor_id=aid, anchor_state=state, anchor_scene=i,
                                role=role, scene=-1, replica=k, query=q,
                                match_status='NOT_EVALUABLE_NO_REQUIRED_BACKGROUND_GROUP_SIDE_CONTROL', mismatch_fields=''))
                            continue
                        _, selected, mismatch = sorted(pool)[0]
                    cases.append(dict(anchor_id=aid, anchor_state=state, anchor_scene=i,
                        role=role, scene=selected, replica=k, query=q,
                        match_status='MATCHED_BACKGROUND_GROUP_SIDE', mismatch_fields=';'.join(mismatch)))
    return cases, strata


def summarize(z, sensor, query, q):
    """Return observation/public-transform-only summaries and peak records."""
    records, peaks, gated = [], [], []
    radius = (np.arange(16) + .5) * WIDTH
    for frame in FRAMES:
        begin = max(0, frame-7)
        transforms = (query[frame] @ np.linalg.inv(sensor[frame]))[None] @ sensor[begin:frame+1]
        history = z[begin:frame+1].astype(np.float32)
        logs = np.sign(history) * np.log1p(np.abs(history))
        center, xyz = membership(transforms, BOXES[q], 'center')
        extent, _ = membership(transforms, BOXES[q], 'extent')
        expanded_box = BOXES[q].copy()
        expanded_box[[0, 3]] = (-.4, .4)
        expanded, _ = membership(transforms, expanded_box, 'center')
        current = history[-1]
        p = local_peaks(current)
        near = np.broadcast_to(radius <= 3, current.shape)
        r = dict(frame=frame, history_begin=begin, history_length=frame-begin+1,
                 native_near_z_max=float(current[near].max()), native_far_z_max=float(current[~near].max()),
                 native_near_peak_count=int((p & near).sum()), native_far_peak_count=int((p & ~near).sum()),
                 native_near_signedlog_total=float(logs[-1][near].sum()))
        for name, weights in (('center', center), ('extent', extent), ('expanded', expanded)):
            current_weights = weights[-1]
            gated_logs = (logs * weights).mean(0)
            r.update({name+'_current_gated_z_max': float((current*current_weights).max()),
                      name+'_history_gated_z_max': float((history*weights).max()),
                      name+'_current_peak_count': int((p & (current_weights >= .5)).sum()),
                      name+'_current_membership_mass': float(current_weights.sum()),
                      name+'_history_signedlog_total': float(gated_logs.sum()),
                      name+'_history_signedlog_max': float(gated_logs.max())})
            if name == 'center':
                gated.append(gated_logs)
        records.append(r)
        for band, mask in (('near', p & near), ('far', p & ~near)):
            indices = np.argwhere(mask)
            ranked = sorted(indices.tolist(), key=lambda ix: (-float(current[tuple(ix)]), *ix))[:3]
            for rank, (y, x, b) in enumerate(ranked, 1):
                loc = xyz[-1, y, x, b, 0]
                peaks.append(dict(frame=frame, band=band, rank=rank, zone_y=y, zone_x=x, bin=b,
                    radial_bin_center_m=float(radius[b]), normalized_z=float(current[y, x, b]),
                    query_x_m=float(loc[0]), query_y_m=float(loc[1]), query_z_m=float(loc[2]),
                    center_membership=float(center[-1, y, x, b]), extent_mean_membership=float(extent[-1, y, x, b]),
                    attribution='OBSERVED_PEAK_TARGET_UNKNOWN'))
    return records, peaks, np.asarray(gated)


def run(output, wall_cap, sampling='original'):
    began = time.monotonic()
    output = output.resolve()
    if not output.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Use canonical artifact tree')
    if (output/'PLAN.json').exists():
        raise FileExistsError('Preserve prior PLAN/result; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    plan = declare(output, wall_cap, sampling)
    phase = 'selection'
    def check():
        if time.monotonic()-began >= wall_cap:
            raise TimeoutError('Raw-echo engineering wall scope reached')
    try:
        rows = read(SOURCE/'scene_rows.json')['validation']
        with np.load(SOURCE/'data/validation/geometry.npz', allow_pickle=False) as archive:
            category = archive['category']
            sensor, query = archive['sensor'], archive['public_query']
        assert category.shape == (len(rows), 2)
        thresholds = read(SOURCE/'evaluation/thresholds.json')['arms']['ordinary']
        raw = []
        for seed in SEEDS:
            with np.load(SOURCE/f'scores/ordinary_seed{seed}_validation.npz', allow_pickle=False) as archive:
                assert archive['seed'].item() == seed and archive['arm'].item() == 'ordinary'
                raw.append(archive['raw'])
        scores = smooth(np.stack(raw))
        theta = np.array([thresholds[str(seed)]['standalone']['value'] for seed in SEEDS])
        flags = scores >= theta[:, None, None, None, None]
        timely = flags[..., :11, :].any(-2)
        cases, strata = (choose_layered(rows, category, timely) if sampling=='layered'
                         else choose_cases(rows, category, timely))
        if not any(case['scene'] >= 0 for case in cases):
            save(output/'result.json', dict(status='NOT_EVALUABLE_NO_ELIGIBLE_ANCHORS', strata=strata))
            return
        save(output/'sampling.json', dict(strata=strata, cases=cases, selection_thresholds=dict(zip(map(str, SEEDS), theta.tolist()))))
        selected = []
        for sample, case in enumerate(cases):
            row = rows[case['scene']] if case['scene'] >= 0 else None
            dims = (np.asarray(row['target_box']['hi'])-np.asarray(row['target_box']['lo'])) if row else None
            selected.append(dict(sample=sample, **case,
                height=HEIGHTS[case['query']], family=row['shape_family'] if row else '',
                background_id=row['background_id'] if row else -1,
                background_family=row['background_family'] if row else '',
                placement=row['placement'] if row else '', rho=row['rho'] if row else '',
                target_dimensions_m=';'.join(f'{v:.5g}' for v in dims) if row else '',
                category=category[case['scene'], case['query']] if row else '',
                verdict='EVIDENCE_INSUFFICIENT_FOR_OBJECT_RETURN_ATTRIBUTION'))
        write_csv(output/'selected_cases.csv', selected)
        check()
        phase = 'same-read native/query summaries'
        hist_map = np.load(SOURCE/'data/validation/hist.npy', mmap_mode='r', allow_pickle=False)
        with np.load(SOURCE/'data/validation/physics.npz', allow_pickle=False) as archive:
            ambient = archive['ambient']
            np.testing.assert_array_equal(archive['sensor'], sensor)
            np.testing.assert_array_equal(archive['public_query'], query)
        bias = np.load(BIAS, allow_pickle=False).astype(np.float32)
        # Load each distinct selected raw observation once, then share it among roles.
        observations = {}
        for case in cases:
            if case['scene'] < 0:
                continue
            key = case['scene'], case['replica']
            if key not in observations:
                check()
                h = np.array(hist_map[key], copy=True)
                den = np.sqrt(np.maximum(16*ambient.astype(np.float32)[..., None] + np.maximum(bias, 0), 1e-9))
                z = ((h.astype(np.float32)-bias)/den).astype(np.float16)
                observations[key] = h, z
        del hist_map
        histories = np.load(SOURCE/'data/validation/histories.npy', mmap_mode='r', allow_pickle=False)
        cache_path = SOURCE/'features/validation.npy'
        features = np.load(cache_path, mmap_mode='r', allow_pickle=False) if cache_path.exists() else None
        frames_out, peaks_out, parities, sample_h, sample_z, sample_ids = [], [], [], [], [], []
        for sample, case in enumerate(cases):
            if case['scene'] < 0:
                continue
            check()
            i, k, q = case['scene'], case['replica'], case['query']
            h, z = observations[i, k]
            sample_h.append(h); sample_z.append(z); sample_ids.append(sample)
            summary, peaks, gated = summarize(z, sensor, query, q)
            for fj, r in enumerate(summary):
                f, length = r['frame'], r['history_length']
                cache_index = (i*4+k)*13+fj
                np.testing.assert_array_equal(histories[cache_index, 8-length:], z[max(0, f-7):f+1])
                frames_out.append(dict(sample=sample, anchor_id=case['anchor_id'], role=case['role'], scene=i,
                    replica=k, height=HEIGHTS[q], category=category[i, q], **r,
                    **{f'ordinary_seed{seed}_smoothed_score': float(scores[si, i, k, fj, q]) for si, seed in enumerate(SEEDS)},
                    **{f'ordinary_seed{seed}_alarm': int(flags[si, i, k, fj, q]) for si, seed in enumerate(SEEDS)}))
                if features is not None:
                    actual = np.asarray(features[cache_index, q, :, 2]).transpose(1, 2, 0).astype(np.float32)
                    reference = gated[fj].astype(np.float16).astype(np.float32)
                    error = float(np.abs(reference-actual).max())
                    if error > .0078125:
                        raise ValueError(('Cached query feature mismatch', sample, f, error))
                    parities.append(error)
            peaks_out += [dict(sample=sample, anchor_id=case['anchor_id'], role=case['role'], scene=i,
                                replica=k, height=HEIGHTS[q], **p) for p in peaks]
        write_csv(output/'frames.csv', frames_out)
        if peaks_out:
            write_csv(output/'peaks.csv', peaks_out)
        np.savez_compressed(output/'sampled_observations.npz', sample=np.array(sample_ids),
            hist=np.stack(sample_h), normalized_z=np.stack(sample_z), ambient=ambient,
            sensor=sensor, public_query=query, bias=bias, seeds=np.array(SEEDS), thresholds=theta)
        phase = 'matched differences'
        by_key = {(r['anchor_id'], r['role'], r['frame']): r for r in frames_out}
        differences = []
        quantities = ['native_near_z_max', 'native_near_peak_count', 'center_history_gated_z_max',
                      'center_history_signedlog_total', 'extent_history_gated_z_max', 'expanded_history_gated_z_max']
        for r in frames_out:
            if r['role'] == 'miss_contact':
                continue
            anchor = by_key[r['anchor_id'], 'miss_contact', r['frame']]
            differences.append(dict(anchor_id=r['anchor_id'], control_role=r['role'], frame=r['frame'],
                anchor_scene=anchor['scene'], control_scene=r['scene'],
                **{name+'_anchor_minus_control': float(anchor[name]-r[name]) for name in quantities}))
        if differences:
            write_csv(output/'matched_differences.csv', differences)
        result = dict(status='COMPLETE', sampling_variant=sampling, anchors=len({c['anchor_id'] for c in cases}),
            selected_cases=len(cases), evaluated_cases=len(sample_ids), unique_raw_reads=len(observations),
            missing_controls=[c for c in cases if c['scene'] < 0], frames=len(frames_out), peaks=len(peaks_out),
            strata=strata, cache_histories_parity='EXACT all valid selected history rows',
            native_center_feature_parity=dict(checked=len(parities), max_abs=max(parities, default=None),
                status='CHECKED' if parities else 'NOT_EVALUABLE_FEATURE_CACHE_NOT_RETAINED',
                comparison='FP16 feature2 membership-gated history mean; numpy versus inherited CUDA FP32 arithmetic, tolerance 0.0078125'),
            verdict='EVIDENCE_INSUFFICIENT_FOR_OBJECT_RETURN_ATTRIBUTION',
            inference=0, training=0, GPU_seconds=0,
            interpretation='Small purposive all-seed-missed/all-seed-detected diagnostic cases. Raw peaks and query '
                           'gating remain measurable on misses and controls, but background peaks cannot be attributed '
                           'to a target. This does not close graded alerts or peak retention; no free-space rule evaluated.')
        save(output/'result.json', result)
        text = ['# 原始回波带对照抽查（模拟 Development）', '',
                f"选择 {result['anchors']} 个 contact 锚点（{sampling}抽样），形成 {len(sample_ids)} 个有效样本、"
                f"{len(frames_out)} 个帧摘要；读取 {len(observations)} 个独特原始直方图。", '',
                ('按背景结构 × height × 全 seed 错/对固定抽样顺序；成功 contact、pass、joint-clear 都要求同背景、同 group/side、同 replica 编号，'
                 if sampling=='layered' else '按 shape × height 固定抽样顺序；成功 contact、pass、joint-clear 都要求同背景、同 replica 编号，')+
                '不同场景的噪声抽样不同。'
                '并非随机总体样本。表中 mismatch_fields 保留目标形态/高度组/侧别/反射率不匹配。', '',
                '同一次原 hist 读取用于 native peaks、径向 bin 中心及 public query 门控摘要。'
                'center/extent 仅为原生 bin/angular-node query 支持代理，没有重建 M3 体素或运行模型。', '',
                (f"原缓存 history 完全一致；center 门控 feature2 最大绝对差 {max(parities):.8g}。" if parities
                 else '原缓存 history 完全一致；原 feature 缓存已清理，center 门控 feature2 缓存对照无法评价。'), '',
                '|锚点|形态/高度|漏检场景|成功 contact|pass|clear|f13 center门控历史峰 z（漏检/成功/pass/clear）|',
                '|---|---|---:|---:|---:|---:|---|']
        for c in cases:
            if c['role'] != 'miss_contact':
                continue
            aid = c['anchor_id']; row = rows[c['scene']]
            roles = ('miss_contact', 'detected_contact', 'pass', 'clear')
            controls = {role: next((v['scene'] for v in cases if v['anchor_id']==aid and v['role']==role), -1) for role in roles}
            values = [by_key[aid, role, 13]['center_history_gated_z_max'] if (aid, role, 13) in by_key else None for role in roles]
            formatted = '/'.join('NA' if value is None else f'{value:.3f}' for value in values)
            text.append(f"|{aid}|{row['shape_family']}/{HEIGHTS[c['query']]}|{controls['miss_contact']}|"
                        f"{controls['detected_contact']}|{controls['pass']}|{controls['clear']}|{formatted}|")
        text += ['', '每例结论均为“目标回波归因证据不足”。z≥3 的原生局部峰会出现在背景/对照里；'
                 '最强峰不能当成目标，远峰只算背景兼容观测，不能判 free。'
                 '对照差异和 FP16 history 一致只帮助定位组合，不证明目标信号在某层丢失。'
                 '少量样本不足以关闭全部分级或峰值保留方向。', '',
                 ('本次按背景结构×高度×全seed错/对分层，每格取首例，覆盖L_sidewall与stacked_side_shelves。'
                  '每类仅一个背景实例，不能推广到全部背景。同形态对照也可能有厚度/长度差异。' if sampling=='layered'
                  else '本次确定性最小 scene_id 抽样全部落在 L_sidewall 的同一个背景实例；'
                       '未抽到 stacked_side_shelves，不能推广到全部背景。同形态对照也可能有厚度/长度差异。'), '',
                 '数据：selected_cases.csv、frames.csv、peaks.csv、matched_differences.csv、sampled_observations.npz。']
        (output/'REPORT.md').write_text('\n'.join(text)+'\n', encoding='utf8')
        check()
    except BaseException as error:
        save(output/f'failure_{time.time_ns()}.json', dict(phase=phase, error=repr(error),
            traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-began))
        raise
    finally:
        save(output/'execution_receipt.json', dict(seconds=time.monotonic()-began, command=[sys.executable, *sys.argv],
            source_sha256=sha(__file__), plan_sha256=sha(output/'PLAN.json'), GPU_seconds=0,
            note='CPU/I/O wall time for this command only; no old Stage budget charged',
            outputs_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'execution_receipt.json'}))


def background_candidate(sample_dir, output):
    """Declared light-only silence probe on small selected cases, no retuning."""
    began = time.monotonic()
    sample_dir, output = sample_dir.resolve(), output.resolve()
    if not output.is_relative_to((ROOT/'artifacts.local').resolve()) or (output/'PLAN.json').exists():
        raise ValueError('Preserve previous result and use canonical output')
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(task='BACKGROUND_VISIBLE_SMALL_SAMPLE_LIGHT_SILENCE', lane='EXPLORE selected simulated Development',
        status='Predeclared candidate, no threshold retuning', sample_source=str(sample_dir),
        rule='At current public-query frame: corridor x/y and z>3m local observed peak zscore>=2.5; '
             'side ring .30<abs(x)<=.40 and z in[.30,3] within query y local peak>=2.5; '
             'far peak public forward z minus ring peak public forward z >=.75m; '
             'maximum corridor near local peak<2.5. Only grade1 is silenced, grade2 unchanged.',
        axes='Peak locations are fixed native zone-center/radial-bin-center hypotheses transformed by '
             'public_query[f] @ inv(sensor[f]) @ sensor[f]. No recovered sub-zone/object geometry or free claim.',
        peak='Leftmost local maximum including endpoints, amplitude threshold2.5 newly proposed Explore value; far/ring choose highest zscore then native index',
        matching='contact/pass same background_id, query, replica, frame, shape_family, rho, group and side; '
                 'same public center/extent membership support (automatic shared poses), observed far and ring forward '
                 f'depths each differ <= actual one native bin WIDTH={WIDTH:.7f}m. Authoring geometry only matching, never rule.',
        calibration_correction='Requested .375m one-bin reference is not this cache calibration; actual WIDTH=8*.0375348=.3002784m. '
            'Use actual one bin, not redefine bins. Query forward depth differs from radial distance.',
        baseline='strong_preserve validation grades [3,384,4,13,2], seeds955/956/957; grade1 only silence',
        scope='Existing v3 small sample only; unique scene/replica/query slots deduplicated. Matched comparison NOT_EVALUABLE if absent; '
              'unmatched candidate counts may describe sample but do not validate silence. Truth used for final costs only.',
        budget='Within raw-echo CPU/I/O600s scope, no GPU/train/inference or full-cohort policy')
    save(output/'PLAN.json', plan)
    with (sample_dir/'selected_cases.csv').open(encoding='utf8') as stream:
        samples = list(csv.DictReader(stream))
    with np.load(sample_dir/'sampled_observations.npz', allow_pickle=False) as archive:
        zs, ids, sensor, query = archive['normalized_z'], archive['sample'], archive['sensor'], archive['public_query']
    zlookup = dict(zip(ids.tolist(), zs))
    rows = read(SOURCE/'scene_rows.json')['validation']
    with np.load(ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/strong_preserve/validation_grades.npz') as archive:
        grades = archive['grades']
    records, identities = [], set()
    for sample in samples:
        i, k, q = int(sample['scene']), int(sample['replica']), int(sample['query'])
        if i < 0 or (i, k, q) in identities:
            continue
        identities.add((i, k, q))
        z = zlookup[int(sample['sample'])].astype(np.float32)
        for frame in FRAMES:
            transform = (query[frame]@np.linalg.inv(sensor[frame])@sensor[frame]).astype(np.float32)
            locs = points('center')[..., 0, :]@transform[:3, :3].T + transform[:3, 3]
            x, y, depth = np.moveaxis(locs, -1, 0)
            vertical = (y >= BOXES[q, 1]) & (y <= BOXES[q, 4])
            corridor = vertical & (np.abs(x) <= .30)
            near = (depth >= .30) & (depth <= 3)
            far_mask = corridor & (depth > 3)
            ring_mask = vertical & (np.abs(x) > .30) & (np.abs(x) <= .40) & near
            current = z[frame]
            missing = np.full((*current.shape[:-1], 1), -np.inf)
            peak = (current > np.concatenate((missing, current[..., :-1]), -1)) & (current >= np.concatenate((current[..., 1:], missing), -1))
            def strongest(mask):
                allowed = np.argwhere(mask & peak)
                if not len(allowed):
                    return None, None
                best = sorted(allowed.tolist(), key=lambda ix: (-float(current[tuple(ix)]), *ix))[0]
                return float(current[tuple(best)]), float(depth[tuple(best)])
            fa, fd = strongest(far_mask)
            ra, rd = strongest(ring_mask)
            na, nd = strongest(corridor & near)
            flag = bool(fa is not None and ra is not None and fa >= 2.5 and ra >= 2.5
                        and fd-rd >= .75 and (na is None or na < 2.5))
            rawgrades = grades[:, i, k, frame-3, q]
            record = dict(scene=i, replica=k, height=HEIGHTS[q], query=q, frame=frame,
                category=sample['category'], background_id=rows[i]['background_id'], shape_family=rows[i]['shape_family'],
                rho=rows[i]['rho'], group=rows[i]['group'], side=rows[i]['side'],
                far_peak_zscore=fa, far_peak_forward_depth_m=fd, ring_peak_zscore=ra, ring_peak_forward_depth_m=rd,
                near_peak_zscore=na, candidate_background_visible=int(flag),
                public_center_near_membership_support=float((corridor & near).sum()))
            for si, seed in enumerate(SEEDS):
                record[f'grade_{seed}'] = int(rawgrades[si])
                record[f'silenced_light_{seed}'] = int(flag and rawgrades[si] == 1)
            records.append(record)
    write_csv(output/'candidate_slots.csv', records)
    matched = []
    for a in records:
        if a['category'] != 'contact' or a['far_peak_forward_depth_m'] is None or a['ring_peak_forward_depth_m'] is None:
            continue
        options = []
        for b in records:
            if b['category'] != 'pass' or any(a[n] != b[n] for n in ('background_id', 'query', 'replica', 'frame', 'shape_family', 'rho', 'group', 'side')):
                continue
            if b['far_peak_forward_depth_m'] is None or b['ring_peak_forward_depth_m'] is None:
                continue
            if max(abs(a[n]-b[n]) for n in ('far_peak_forward_depth_m', 'ring_peak_forward_depth_m')) <= WIDTH+1e-9:
                options.append(b)
        if options:
            b = sorted(options, key=lambda r: r['scene'])[0]
            matched.append(dict(contact_scene=a['scene'], pass_scene=b['scene'], replica=a['replica'],
                query=a['query'], frame=a['frame'],
                **{f'contact_silenced_light_{seed}': a[f'silenced_light_{seed}'] for seed in SEEDS},
                **{f'pass_silenced_light_{seed}': b[f'silenced_light_{seed}'] for seed in SEEDS}))
    if matched:
        write_csv(output/'matched_slots.csv', matched)
    summary = []
    for seed in SEEDS:
        for category in ('contact', 'pass', 'clear'):
            keep = [r for r in records if r['category']==category]
            before = sum(r[f'grade_{seed}']>0 for r in keep)
            reduction = sum(r[f'silenced_light_{seed}'] for r in keep)
            summary.append(dict(seed=seed, category=category, unique_query_slots=len(keep), baseline_ANY_slots=before,
                silenced_light_slots=reduction, remaining_ANY_slots=before-reduction,
                lost_timely_contact_events=sum(all(r[f'silenced_light_{seed}'] for r in keep if r['scene']==i and r['replica']==k and r['query']==q and r['frame']<=13 and r[f'grade_{seed}']>0)
                    and any(r[f'grade_{seed}']>0 for r in keep if r['scene']==i and r['replica']==k and r['query']==q and r['frame']<=13)
                    for i, k, q in sorted({(r['scene'], r['replica'], r['query']) for r in keep})) if category=='contact' else 0))
    write_csv(output/'sample_costs.csv', summary)
    result = dict(status='COMPLETE', matched_status='MATCHED' if matched else 'NOT_EVALUABLE_NO_DISTANCE_AND_METADATA_MATCHES',
        unique_selected_query_events=len(identities), unique_query_slots=len(records), matched_contact_pass_slots=len(matched),
        candidate_background_visible_slots=sum(r['candidate_background_visible'] for r in records),
        matched_per_seed=[dict(seed=seed, contact_silenced_light_slots=sum(r[f'contact_silenced_light_{seed}'] for r in matched),
            pass_silenced_light_slots=sum(r[f'pass_silenced_light_{seed}'] for r in matched)) for seed in SEEDS],
        unique_sample_costs=summary, seconds=time.monotonic()-began, actual_native_bin_width_m=WIDTH,
        conclusion='Candidate only, not free; sample and matching limits preclude full-cohort silence recommendation',
        inference=0, training=0, GPU_seconds=0, command=[sys.executable, *sys.argv])
    save(output/'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--wall-cap', type=float, default=600)
    parser.add_argument('--sampling', choices=('original', 'layered'), default='original')
    parser.add_argument('--background-samples', type=Path)
    args = parser.parse_args()
    if args.background_samples:
        background_candidate(args.background_samples, args.output)
    else:
        run(args.output, args.wall_cap, args.sampling)

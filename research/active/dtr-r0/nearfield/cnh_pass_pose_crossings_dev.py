"""Cached within-model ideal/+3 threshold crossings for all 1cm contacts.

No inference, photons, fitting, calibration changes or threshold selection.
The event unit is scene x replica x height, using fixed timely f3..13.
Sides are signed query/world fixture x coordinates, not body-turn labels.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import cnh_counterfactual_eval_dev as E

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/cnh-pass-boundary-dev-20261009'
OUT = ROOT/'artifacts.local/work/cnh-pass-pose-diagnostic-dev-20261009'
DEST = OUT/'crossings'
ARMS = ('control', 'weak_pass')
SEEDS = (2026100955, 2026100956, 2026100957)
POINTS = ('pass_20pct', 'pass_30pct', 'pass_40pct')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def csvwrite(path, rows):
    with Path(path).open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def state(a, b):
    return 'keep' if a and b else 'rescue' if b else 'loss' if a else 'both_miss'


def stats(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {k: float(f(v)) if len(v) else None for k, f in
            (('mean', np.mean), ('median', np.median), ('min', np.min), ('max', np.max))}


def run():
    started = time.monotonic()
    DEST.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT/'stage_receipts'/f'crossings_{time.time_ns()}.json'
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    plan = read(OUT/'PLAN.json')
    cap = plan['allocations']['crossings']
    source_digest = sha(__file__)
    snapshot = DEST/'source_snapshot.py'
    if snapshot.exists():
        raise FileExistsError('Preserve previous diagnostic source/output')
    snapshot.write_bytes(Path(__file__).read_bytes())
    def check():
        if time.monotonic()-started > cap:
            raise TimeoutError('Crossing diagnostic CPU stage budget reached')
    try:
        assert cap == 70 and plan['arms'] == list(ARMS) and plan['seeds'] == list(SEEDS)
        assert plan['points'] == list(POINTS)
        manifest_path = SOURCE/'eval_manifest_with_continuation.json'
        thresholds_path = SOURCE/'workpoints_continued/thresholds.json'
        manifest, thresholds = read(manifest_path), read(thresholds_path)
        datasets = {d['branch']: d for d in manifest['datasets'] if d['split'] == 'validation'}
        assert set(datasets) == {'ideal', 'yaw_plus3'}
        rows = read(datasets['ideal']['scene_rows'])['validation']
        assert len(rows) == 384 and [r['scene_id'] for r in rows] == list(range(384))
        with np.load(datasets['ideal']['physics'], allow_pickle=False) as archive:
            category, sensor, query = (archive[k] for k in ('category', 'sensor', 'public_query'))
        assert sensor.shape == query.shape == (16, 4, 4)
        a = np.deg2rad(3.)
        ry = np.eye(4); ry[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        metadata_checks = []
        for branch in ('ideal', 'yaw_plus3'):
            with np.load(datasets[branch]['baseline'], allow_pickle=False) as archive:
                estimated, source_sensor = archive['estimated_query'], archive['sensor']
            expected = query if branch == 'ideal' else ry@query
            np.testing.assert_array_equal(source_sensor, sensor)
            np.testing.assert_allclose(estimated, expected, atol=1e-12, rtol=0)
            metadata_checks.append(dict(branch=branch, query_max_abs=float(np.max(np.abs(estimated-expected))),
                convention='stored public_query=current sensor->query; W=public_query@inv(sensor)'))
        # Fixture truth is fixed world axes: ideal query/world rotations cancel.
        expected_w = np.repeat(np.eye(4)[None], 16, axis=0)
        expected_w[:, :3, 3] = -sensor[:, :3, 3]
        np.testing.assert_allclose(query@np.linalg.inv(sensor), expected_w, atol=1e-12, rtol=0)
        front_point = np.array([0., 0., 1., 1.])
        assert (ry@front_point)[0] > 0
        payloads, raw, smooth = {}, {}, {}
        for branch in ('ideal', 'yaw_plus3'):
            jobs = {(j['arm'], int(j['seed'])): j['path'] for j in datasets[branch]['scores']}
            for arm in ARMS:
                for seed in SEEDS:
                    check()
                    path = Path(jobs[arm, seed])
                    with np.load(path, allow_pickle=False) as archive:
                        score = archive['raw']
                    assert score.shape == (384, 4, 13, 2) and np.isfinite(score).all()
                    raw[arm, seed, branch] = score.astype(np.float64)
                    smooth[arm, seed, branch] = E.smooth(score)
                    payloads[str(path)] = sha(path)
        events, by_key = [], {}
        for arm in ARMS:
            for seed in SEEDS:
                for point in POINTS:
                    record = thresholds['arms'][arm][str(seed)]['standalone'][point]
                    assert record['status'] == 'CALIBRATED'
                    theta = E.threshold(record['threshold'])
                    for row in rows:
                        if row['placement'] != 'in1cm':
                            continue
                        sid = row['scene_id']; side = 'negative_x' if row['side'] == -1 else 'positive_x'
                        assert row['side'] in (-1, 1)
                        for q, height in enumerate(('HEAD', 'BODY')):
                            if category[sid, q] != 'contact':
                                continue
                            assert row['group'] == q
                            for rep in range(4):
                                rr = [raw[arm, seed, b][sid, rep, :11, q] for b in ('ideal', 'yaw_plus3')]
                                ss = [smooth[arm, seed, b][sid, rep, :11, q] for b in ('ideal', 'yaw_plus3')]
                                ri, ryaw = map(lambda v: float(v.max()), rr)
                                si, syaw = map(lambda v: float(v.max()), ss)
                                ideal_flag, yaw_flag = si >= theta, syaw >= theta
                                event = dict(arm=arm, seed=seed, point=point, scene_id=sid, replica=rep,
                                    height=height, side=side, side_sign=row['side'], background_id=row['background_id'],
                                    background_family=row['background_family'], shape_family=row['shape_family'],
                                    target_id=row['target_id'], placement='in1cm', threshold=theta,
                                    front_distance_f10_m=float(row['target_box']['lo'][2]-sensor[10, 2, 3]),
                                    ideal_raw_peak=ri, yaw_raw_peak=ryaw, raw_peak_delta=ryaw-ri,
                                    raw_same_frame_delta_mean=float(np.mean(rr[1]-rr[0])),
                                    raw_same_frame_delta_min=float(np.min(rr[1]-rr[0])),
                                    raw_same_frame_delta_max=float(np.max(rr[1]-rr[0])),
                                    ideal_smoothed_peak=si, yaw_smoothed_peak=syaw, smoothed_peak_delta=syaw-si,
                                    ideal_peak_frame=int(np.argmax(ss[0])+3), yaw_peak_frame=int(np.argmax(ss[1])+3),
                                    ideal_margin=si-theta, yaw_margin=syaw-theta,
                                    ideal_detected=int(ideal_flag), yaw_detected=int(yaw_flag),
                                    crossing=state(ideal_flag, yaw_flag),
                                    ideal_smoothed_f10=float(ss[0][7]), yaw_smoothed_f10=float(ss[1][7]))
                                key = arm, seed, point, sid, rep, height
                                assert key not in by_key
                                events.append(event); by_key[key] = event
        assert len(events) == 4608, len(events)
        summaries = []
        for arm in ARMS:
            for seed in SEEDS:
                for point in POINTS:
                    for height in ('HEAD', 'BODY'):
                        for side in ('all', 'negative_x', 'positive_x'):
                            sample = [e for e in events if e['arm'] == arm and e['seed'] == seed
                                      and e['point'] == point and e['height'] == height
                                      and (side == 'all' or e['side'] == side)]
                            assert len(sample) == (128 if side == 'all' else 64)
                            summary = dict(arm=arm, seed=seed, point=point, height=height, side=side,
                                event_denominator=len(sample), physical_scene_denominator=len({e['scene_id'] for e in sample}),
                                threshold=sample[0]['threshold'],
                                ideal_detected=sum(e['ideal_detected'] for e in sample),
                                yaw_detected=sum(e['yaw_detected'] for e in sample))
                            summary.update({name: sum(e['crossing'] == name for e in sample)
                                            for name in ('keep', 'rescue', 'loss', 'both_miss')})
                            for field in ('raw_peak_delta', 'smoothed_peak_delta', 'ideal_margin', 'yaw_margin'):
                                summary.update({field+'_'+k: v for k, v in stats([e[field] for e in sample]).items()})
                            assert summary['ideal_detected'] == summary['keep']+summary['loss']
                            assert summary['yaw_detected'] == summary['keep']+summary['rescue']
                            assert sum(summary[k] for k in ('keep', 'rescue', 'loss', 'both_miss')) == len(sample)
                            summaries.append(summary)
        pairs = []
        for key, weak in by_key.items():
            if key[0] != 'weak_pass':
                continue
            control = by_key[('control', *key[1:])]
            pairs.append(dict(seed=weak['seed'], point=weak['point'], scene_id=weak['scene_id'],
                replica=weak['replica'], height=weak['height'], side=weak['side'],
                control_crossing=control['crossing'], weak_crossing=weak['crossing'],
                control_ideal=control['ideal_detected'], weak_ideal=weak['ideal_detected'],
                control_yaw=control['yaw_detected'], weak_yaw=weak['yaw_detected'],
                ideal_pair=state(control['ideal_detected'], weak['ideal_detected']),
                yaw_pair=state(control['yaw_detected'], weak['yaw_detected']),
                control_yaw_effect=control['yaw_detected']-control['ideal_detected'],
                weak_yaw_effect=weak['yaw_detected']-weak['ideal_detected'],
                binary_yaw_effect_difference=(weak['yaw_detected']-weak['ideal_detected'])
                                           -(control['yaw_detected']-control['ideal_detected'])))
        check()
        csvwrite(DEST/'crossings_events.csv', events)
        csvwrite(DEST/'crossing_summary.csv', summaries)
        csvwrite(DEST/'weak_control_event_identity.csv', pairs)
        result = dict(status='PASS', source_sha256=source_digest, event_rows=len(events),
            unique_event_denominator_per_model_point_height=128, physical_scenes_per_height=32,
            noise_replicas=4, per_side_event_denominator=64, timely_frames=list(range(3, 14)),
            fixed_distance_definition='target front world-z minus sensor world-z at f10; forward distance, not Euclidean/slant range',
            fixed_f10_front_distance_m=sorted({e['front_distance_f10_m'] for e in events}),
            pose_metadata_checks=metadata_checks,
            side_interpretation='Ry(+3) left multiplies sensor-to-query. For forward points x_query increases: negative-x target moves toward query center; positive-x target moves outward. This is a coordinate fact, not proof that score changes follow it.',
            summaries=summaries, raw_score_sha256=payloads,
            inputs_sha256={str(manifest_path): sha(manifest_path), str(thresholds_path): sha(thresholds_path),
                          datasets['ideal']['scene_rows']: sha(datasets['ideal']['scene_rows'])},
            evaluator_source_sha256=sha(E.__file__),
            scope='Consumed validation Development; all 1cm contacts including misses retained; no threshold search',
            limits='Logit scales differ across models; margins only within each model/own threshold. Timely maxima may occur at different frames. Correlated fixtures/replicas/seeds, no -3 sign-reversal test, hardware or causal attribution.')
        save(DEST/'crossing_summary.json', result)
        lines = ['# 1cm 接触事件的既有阈值跨越诊断', '',
            '单位为scene×noise replica×height；每模型/工作点HEAD和BODY各128事件（32物理场景×4噪声），每侧64。全部漏检保留。',
            '及时峰值固定取f3..13；f10目标前沿到传感器的前向z距离为1.45m，不是斜距。每模型使用自己的既有ideal-cal阈值。', '',
            'public_query为当前sensor→query。元数据核对+3分支为Ry(+3)左乘，前方点query x增加；negative_x趋向中心，positive_x趋向外侧。此几何事实不证明模型的响应机制，也不表示实物左/右转。', '',
            '| arm | seed | point | height | side | ideal | +3 | rescue | loss | both miss | denominator |',
            '|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|']
        for s in summaries:
            if s['side'] == 'all':
                lines.append('|'+ '|'.join(str(s[k]) for k in ('arm','seed','point','height','side','ideal_detected','yaw_detected','rescue','loss','both_miss','event_denominator'))+'|')
        lines += ['', '完整side分层、raw/平滑峰值差、逐帧raw差、阈值裕量和峰值帧在CSV/JSON。weak_control_event_identity.csv按同事件记录两模型各自ideal→+3跨越，以及weak相对control在ideal和+3的救/损。',
            '跨模型logit尺度不同，不能把两个模型的平均logit差直接作为因果效应；不同帧的峰值差与同帧差已分开。仅有+3，不能检验反号对称性；没有新训练、推理、光子或硬件证据。']
        (DEST/'findings.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
        check()
        save(receipt_path, dict(stage='crossings', status='PASS', seconds=time.monotonic()-started,
            cpu_only=True, source_sha256=source_digest, plan_sha256=sha(OUT/'PLAN.json'), outputs=str(DEST)))
        print(json.dumps(dict(status='PASS', seconds=time.monotonic()-started, event_rows=len(events)), ensure_ascii=False))
    except BaseException as exc:
        save(receipt_path, dict(stage='crossings', status='FAILED', seconds=time.monotonic()-started,
            cpu_only=True, error=repr(exc), source_sha256=source_digest))
        raise


if __name__ == '__main__':
    run()

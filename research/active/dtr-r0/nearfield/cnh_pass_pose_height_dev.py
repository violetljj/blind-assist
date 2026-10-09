"""Read-only fixed-distance HEAD/BODY fixture diagnostic on saved scores.

Strict matching retains reflectance. A separately named geometry-only
correspondence drops reflectance and cannot remove that confounding. Replicas
share their index only: physical keys and sampled photons differ across height.
No threshold search, model inference, projection or histogram loading occurs.
"""
import argparse
import csv
import hashlib
import itertools
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
TASK = ROOT/'artifacts.local/work/cnh-pass-pose-diagnostic-dev-20261009'
SOURCE = ROOT/'artifacts.local/work/cnh-pass-boundary-dev-20261009'
FRAMES = (3, 8, 10, 13)
BANDS = ('lt1m', '1_to_1p5m', '1p5_to_2m', 'ge2m')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def write_csv(path, rows, fields):
    with Path(path).open('x', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(raw):
    values = np.asarray(raw, np.float64)
    result = np.empty_like(values)
    for f in range(13):
        start = max(0, f-4)
        weights = 2.**np.arange(f-start+1)
        result[:, :, f] = np.sum(values[:, :, start:f+1]*weights[:, None], axis=2)/weights.sum()
    return result


def threshold(record):
    return np.inf if record['positive_infinity'] else float(record['value'])


def band(distance):
    return BANDS[0] if distance < 1 else BANDS[1] if distance < 1.5 else BANDS[2] if distance < 2 else BANDS[3]


def fixed_geometry(row, sensor, public_query):
    box = row['target_box']
    corners = np.array(list(itertools.product(*zip(box['lo'], box['hi']))), np.float64)
    distances = {}
    for f in FRAMES:
        query_from_world = public_query[f]@np.linalg.inv(sensor[f])
        xyz = corners@query_from_world[:3, :3].T+query_from_world[:3, 3]
        distances[f] = float(xyz[:, 2].min())
    return distances


def fixture_key(row, distance, *, retain_rho):
    box = row['target_box']
    lo, hi = np.array(box['lo']), np.array(box['hi'])
    size = tuple(np.round(hi-lo, 12))
    xz = tuple(np.round([lo[0], hi[0], lo[2], hi[2]], 12))
    background = json.dumps(row['background_boxes'], sort_keys=True, separators=(',', ':'))
    key = (row['shape_family'], row['side'], row['background_id'], row['background_family'],
           background, size, xz, row['occurrence_draw_id'], round(distance, 12))
    return key+(row['rho'], box['rho']) if retain_rho else key


def build_mapping(rows, category, sensor, query):
    selected = {q: [r for r in rows if r['placement'] == 'in1cm' and r['presence']
                    and int(r['group']) == q and category[r['scene_id'], q] == 'contact'] for q in (0, 1)}
    distances = {r['scene_id']: fixed_geometry(r, sensor, query) for q in selected for r in selected[q]}
    mapping, pairs = [], {}
    for mode, retain in (('strict_rho', True), ('geometry_only_rho_uncontrolled', False)):
        buckets = {q: defaultdict(list) for q in (0, 1)}
        for q in (0, 1):
            for row in selected[q]:
                buckets[q][fixture_key(row, distances[row['scene_id']][10], retain_rho=retain)].append(row)
        paired, matched = [], {0: set(), 1: set()}
        for key in sorted(set(buckets[0]) & set(buckets[1]), key=str):
            if len(buckets[0][key]) != 1 or len(buckets[1][key]) != 1:
                raise ValueError('Ambiguous many-to-many fixture correspondence; do not pair by outcomes')
            head, body = buckets[0][key][0], buckets[1][key][0]
            matched[0].add(head['scene_id']); matched[1].add(body['scene_id'])
            paired.append((head, body))
            mapping.append(dict(mode=mode, status='PAIRED', head_scene=head['scene_id'], body_scene=body['scene_id'],
                shape_family=head['shape_family'], side=head['side'], background_id=head['background_id'],
                background_family=head['background_family'], head_rho=head['rho'], body_rho=body['rho'],
                head_target_id=head['target_id'], body_target_id=body['target_id'],
                head_physical_key=head['physical_key'], body_physical_key=body['physical_key'],
                head_occurrence_draw_id=head['occurrence_draw_id'], body_occurrence_draw_id=body['occurrence_draw_id'],
                distance_f10_head=distances[head['scene_id']][10], distance_f10_body=distances[body['scene_id']][10],
                distance_band=band(distances[head['scene_id']][10]), shared_photons=False))
        for q in (0, 1):
            for row in selected[q]:
                if row['scene_id'] not in matched[q]:
                    mapping.append(dict(mode=mode, status='UNMATCHED', head_scene=row['scene_id'] if q == 0 else '',
                        body_scene=row['scene_id'] if q == 1 else '', shape_family=row['shape_family'], side=row['side'],
                        background_id=row['background_id'], background_family=row['background_family'],
                        head_rho=row['rho'] if q == 0 else '', body_rho=row['rho'] if q == 1 else '',
                        head_target_id=row['target_id'] if q == 0 else '', body_target_id=row['target_id'] if q == 1 else '',
                        head_physical_key=row['physical_key'] if q == 0 else '', body_physical_key=row['physical_key'] if q == 1 else '',
                        head_occurrence_draw_id=row['occurrence_draw_id'] if q == 0 else '', body_occurrence_draw_id=row['occurrence_draw_id'] if q == 1 else '',
                        distance_f10_head=distances[row['scene_id']][10] if q == 0 else '',
                        distance_f10_body=distances[row['scene_id']][10] if q == 1 else '',
                        distance_band=band(distances[row['scene_id']][10]), shared_photons=False))
        pairs[mode] = paired
    coverage = dict(head_contact_scenes=len(selected[0]), body_contact_scenes=len(selected[1]),
        strict_pairs=len(pairs['strict_rho']), geometry_only_pairs=len(pairs['geometry_only_rho_uncontrolled']),
        strict_unmatched_head=len(selected[0])-len(pairs['strict_rho']),
        strict_unmatched_body=len(selected[1])-len(pairs['strict_rho']),
        geometry_only_unmatched_head=len(selected[0])-len(pairs['geometry_only_rho_uncontrolled']),
        geometry_only_unmatched_body=len(selected[1])-len(pairs['geometry_only_rho_uncontrolled']),
        distance_band_scenes={b: sum(band(distances[r['scene_id']][10]) == b for q in selected for r in selected[q]) for b in BANDS},
        distance_band_head_scenes={b: sum(band(distances[r['scene_id']][10]) == b for r in selected[0]) for b in BANDS},
        distance_band_body_scenes={b: sum(band(distances[r['scene_id']][10]) == b for r in selected[1]) for b in BANDS},
        match_key='shape,side,background instance/boxes,target xyz size/xz bounds,occurrence index,f10 ideal query front distance; strict additionally rho. Absolute target y is the compared height; target IDs and physical keys recorded, not assumed equal',
        key_rounding='12 decimal places for floating geometry; scene background boxes exact serialized match',
        photon_matching='No common physical noise: corresponding K indices are not shared photons')
    return mapping, pairs, distances, coverage


def aggregate(events, identity, stratum, mode, unmatched, strict_pair_count, exposure):
    selected = events if stratum == 'all' else [e for e in events if e['distance_band'] == stratum]
    h, b = np.array([e['head_timely'] for e in selected], bool), np.array([e['body_timely'] for e in selected], bool)
    return dict(**identity, mode=mode, stratum=stratum, status='EVALUABLE_CONFOUNDED' if len(selected) else 'NOT_EVALUABLE',
        paired_indexed_events=len(selected), head_timely=int(h.sum()), body_timely=int(b.sum()),
        both=int((h & b).sum()), head_only=int((h & ~b).sum()), body_only=int((~h & b).sum()), neither=int((~h & ~b).sum()),
        body_minus_head=int(b.sum()-h.sum()), strict_fixture_pairs=strict_pair_count,
        global_unmatched_head_events=unmatched[0], global_unmatched_body_events=unmatched[1],
        unmatched_scope='whole_1cm_cohort', head_contact_exposure_events=exposure[0], body_contact_exposure_events=exposure[1],
        mean_body_minus_head_peak=float(np.mean([e['body_peak']-e['head_peak'] for e in selected])) if selected else None)


def run(plan_path, output, manifest_path, thresholds_path, prior_seconds=0.):
    began = time.monotonic()
    output = Path(output)
    plan = read(plan_path)
    cap = plan['allocations']['matched_height']
    if output.exists():
        raise FileExistsError('Preserve prior matched-height outputs')
    if not output.resolve().is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Output must remain under canonical artifacts.local')
    output.mkdir(parents=True)
    receipt = dict(status='RUNNING', cpu_only=True, prior_inspection_seconds=prior_seconds,
        cap_seconds=cap, plan_sha256=sha(plan_path), source_sha256=sha(__file__))
    def check():
        if prior_seconds+time.monotonic()-began >= cap:
            raise TimeoutError('Matched-height cumulative analysis budget reached')
    try:
        shutil.copyfile(__file__, output/'source_snapshot.py')
        manifest, thresholds = read(manifest_path), read(thresholds_path)['arms']
        specs = [d for d in manifest['datasets'] if d['split'] == 'validation']
        if {s['branch'] for s in specs} != {'ideal', 'yaw_plus3'}:
            raise ValueError('Both frozen validation branches required')
        rows = read(specs[0]['scene_rows'])['validation']
        with np.load(specs[0]['physics'], allow_pickle=False) as physics:
            category, sensor, query, scene_ids = (physics[k] for k in ('category', 'sensor', 'public_query', 'scene_ids'))
        if [r['scene_id'] for r in rows] != scene_ids.tolist():
            raise ValueError('Scene-row/physics ordering differs')
        mapping, pairs, distances, coverage = build_mapping(rows, category, sensor, query)
        coverage.update(replicas=4, head_contact_events=coverage['head_contact_scenes']*4,
            body_contact_events=coverage['body_contact_scenes']*4, strict_unmatched_events=(coverage['strict_unmatched_head']+coverage['strict_unmatched_body'])*4)
        fields = ['mode','status','head_scene','body_scene','shape_family','side','background_id','background_family','head_rho','body_rho','head_target_id','body_target_id','head_physical_key','body_physical_key','head_occurrence_draw_id','body_occurrence_draw_id','distance_f10_head','distance_f10_body','distance_band','shared_photons']
        write_csv(output/'pair_mapping.csv', mapping, fields)
        events, summaries, sources = [], [], []
        for spec in specs:
            with np.load(spec['physics'], allow_pickle=False) as physics:
                np.testing.assert_array_equal(physics['category'], category)
                np.testing.assert_array_equal(physics['scene_ids'], scene_ids)
            for score in spec['scores']:
                if score['arm'] not in plan['arms']:
                    continue
                check()
                arm, seed = score['arm'], score['seed']
                with np.load(score['path'], allow_pickle=False) as z:
                    raw = z['raw']
                    if str(z['arm'].item()) != arm or int(z['seed'].item()) != seed or str(z['split'].item()) != 'validation':
                        raise ValueError('Score identity differs')
                if raw.shape != (384, 4, 13, 2) or not np.isfinite(raw).all():
                    raise ValueError('Full finite saved raw-score cohort required')
                values = smooth(raw)
                sources.append(dict(path=score['path'], sha256=sha(score['path'])))
                for point in plan['points']:
                    theta = threshold(thresholds[arm][str(seed)]['standalone'][point]['threshold'])
                    identity = dict(arm=arm, seed=seed, branch=spec['branch'], point=point,
                        policy='standalone', threshold=theta if np.isfinite(theta) else None)
                    cell = []
                    for mode in ('strict_rho', 'geometry_only_rho_uncontrolled'):
                        mode_events = []
                        for pid, (head, body) in enumerate(pairs[mode]):
                            h, b = head['scene_id'], body['scene_id']
                            for k in range(4):
                                hp, bp = float(values[h,k,:11,0].max()), float(values[b,k,:11,1].max())
                                row = dict(**identity, mode=mode, pair_id=pid, head_scene=h, body_scene=b, replica=k,
                                    shape_family=head['shape_family'], side=head['side'], background_id=head['background_id'],
                                    background_family=head['background_family'], head_rho=head['rho'], body_rho=body['rho'],
                                    head_physical_key=head['physical_key'], body_physical_key=body['physical_key'], shared_photons=False,
                                    distance_f10=distances[h][10], distance_band=band(distances[h][10]),
                                    head_peak=hp, body_peak=bp, head_margin=hp-theta if np.isfinite(theta) else None,
                                    body_margin=bp-theta if np.isfinite(theta) else None, head_timely=int(hp>=theta), body_timely=int(bp>=theta))
                                for f in FRAMES:
                                    row[f'distance_f{f}'] = distances[h][f]
                                    row[f'head_score_f{f}'] = float(values[h,k,f-3,0])
                                    row[f'body_score_f{f}'] = float(values[b,k,f-3,1])
                                    row[f'head_alarm_f{f}'] = int(values[h,k,f-3,0]>=theta)
                                    row[f'body_alarm_f{f}'] = int(values[b,k,f-3,1]>=theta)
                                mode_events.append(row)
                        events.extend(mode_events)
                        unmatched = (coverage['strict_unmatched_head']*4, coverage['strict_unmatched_body']*4) if mode == 'strict_rho' else (coverage['geometry_only_unmatched_head']*4, coverage['geometry_only_unmatched_body']*4)
                        for stratum in ('all',)+BANDS:
                            exposure = (coverage['head_contact_events'], coverage['body_contact_events']) if stratum == 'all' else (
                                coverage['distance_band_head_scenes'][stratum]*4, coverage['distance_band_body_scenes'][stratum]*4)
                            summaries.append(aggregate(mode_events, identity, stratum, mode, unmatched, len(pairs['strict_rho']), exposure))
        if len(summaries) != len(plan['arms'])*len(plan['seeds'])*len(plan['points'])*2*2*5:
            raise ValueError('Summary cells do not preserve declared arms/seeds/points/branches')
        if events:
            write_csv(output/'paired_height_events.csv', events, list(events[0]))
        else:
            write_csv(output/'paired_height_events.csv', [], ['arm','seed','branch','point','mode','head_scene','body_scene','replica'])
        write_csv(output/'height_summary.csv', summaries, list(summaries[0]))
        result = dict(status='COMPLETE', coverage=coverage, summary=summaries,
            strict_conclusion='NOT_EVALUABLE: no same-rho HEAD/BODY fixtures' if not pairs['strict_rho'] else 'Matched-height descriptive evidence only',
            supplement='Geometry-only correspondence does not control rho and uses different physical keys/photons; not a causal height or projection/zone test',
            distance_rule='Ideal physical-query near-z at fixed f10, retained across ideal/+3; additional fixed f3/8/10/13; not first-alarm distance',
            distance_limit='Front-z is not slant range or full geometric registration; one occupied band gives no new independent distance control, and rho remains confounded',
            unmatched_count_scope='global_unmatched_* are whole-1cm-cohort matching counts repeated across strata; head/body_contact_exposure_events are stratum-specific',
            bands=list(BANDS), input_scores=sources, threshold_sha256=sha(thresholds_path), manifest_sha256=sha(manifest_path),
            selected_physics_arrays_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('category',category),('sensor',sensor),('public_query',query),('scene_ids',scene_ids)]},
            input_scope='Only named physics arrays loaded; histograms, photons and evaluator recalibration not read or generated')
        save(output/'height_summary.json', result)
        geometric = [r for r in summaries if r['mode']=='geometry_only_rho_uncontrolled' and r['stratum']=='all']
        negative = sum(r['body_minus_head'] < 0 for r in geometric)
        positive = sum(r['body_minus_head'] > 0 for r in geometric)
        zero = sum(r['body_minus_head'] == 0 for r in geometric)
        lines = ['# 固定距离 HEAD/BODY 对应诊断', '',
            f"严格同反射率匹配：{coverage['strict_pairs']} 对；HEAD/BODY各{coverage['head_contact_scenes']}/{coverage['body_contact_scenes']}场景、各{coverage['head_contact_events']}/{coverage['body_contact_events']}事件；严格未匹配{coverage['strict_unmatched_events']}事件。",
            result['strict_conclusion']+'。不得根据零匹配判断高度效应消失。', '',
            f"另列 geometry-only：{coverage['geometry_only_pairs']} 对、每工作点{coverage['geometry_only_pairs']*4}个K索引对应。rho随height/group交换；不控制反射率，physical_key不同，K编号不代表共同光子。",
            f"固定f10距离band场景计数：{coverage['distance_band_scenes']}；保留全部空band，不后调分箱。",
            '固定f10为ideal物理query的前向z近端距离（均1.45m），不等同斜距或全部几何配准。单一占用band没有提供新的独立距离对照；rho仍混杂。',
            'summary表global_unmatched_*是全1cm cohort的未匹配总数，重复列示，不表示空band有128事件；head/body_contact_exposure_events为该stratum真实暴露数。',
            f"36个model/seed/point/branch条件中BODY命中少于HEAD={negative}，多于={positive}，相等={zero}。这是混杂对应的描述，不构成高度/投影/zone机制证明。", '',
            '全部1cm contacts保留，包括未检出。及时peak限定f3..13；固定f3/8/10/13分数与几何距离均保留。各seed和K/帧相关；不做独立性或显著性主张。']
        (output/'findings.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
        check()
        receipt.update(status='COMPLETE', runtime_seconds=time.monotonic()-began,
            total_analysis_seconds=prior_seconds+time.monotonic()-began, rows=len(events), summary_rows=len(summaries),
            strict_pairs=coverage['strict_pairs'], geometry_only_pairs=coverage['geometry_only_pairs'],
            inputs='Saved scores/thresholds and existing authoring metadata only', GPU=False,
            scope='No training/inference/projection/threshold search/new photons/hardware')
        save(output/'receipt.json', receipt)
        return receipt
    except Exception as error:
        receipt.update(status='FAILED', runtime_seconds=time.monotonic()-began,
            total_analysis_seconds=prior_seconds+time.monotonic()-began, error=repr(error))
        save(output/'receipt.json', receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=TASK/'PLAN.json')
    parser.add_argument('--output', type=Path, default=TASK/'matched_height')
    parser.add_argument('--manifest', type=Path, default=SOURCE/'eval_manifest_with_continuation.json')
    parser.add_argument('--thresholds', type=Path, default=SOURCE/'workpoints_continued/thresholds.json')
    parser.add_argument('--prior-seconds', type=float, default=0.)
    args = parser.parse_args()
    print(json.dumps(run(args.plan,args.output,args.manifest,args.thresholds,args.prior_seconds)), flush=True)

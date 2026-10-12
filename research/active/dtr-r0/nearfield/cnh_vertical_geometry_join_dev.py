"""Join existing geometry/clear/event ledgers and read saved Dwin score arrays.

All geometry is evaluator-only. This script performs no geometry replay,
projection, rendering, model inference, training or separability experiment.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
OUT = WORK / 'cnh-vertical-geometry-join-dev-20261009'
SHAPE = WORK / 'cnh-aligned-shapes-dev-20261008'
QUERY = WORK / 'cnh-bar-query-cost-dev-20261009'
BOUNDARY = WORK / 'cnh-bar-boundary-contrast-dev-20261009'
PRESSURE = (-3, -2, 2, 3)
T0, TM, TL = .8557642486787612, .9404184587540165, 4.625390338985158


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')


def meta(row):
    side = int(row['side'])
    inner = row['lo'][0] if side > 0 else -row['hi'][0] if side < 0 else 0.
    return dict(side=side, width_cm=round(100*(row['hi'][0]-row['lo'][0]), 6),
                rho=row['rho'], span=row['group'], variant=row['variant'], placement=row['placement'],
                nominal_signed_boundary_gap_cm=round(100*(inner-.30), 6))


def main():
    start = time.monotonic()
    inputs = [SHAPE/'PLAN.json', SHAPE/'geometry.npz', SHAPE/'event_ledger.csv',
              QUERY/'clear_slots.csv', QUERY/'review/geometry.csv',
              BOUNDARY/'event_ledger.csv', BOUNDARY/'result_analysis.json',
              *[BOUNDARY/f'scores_{a:+d}.npz' for a in (0, -3, 3)],
              WORK/'cnh-bar-category-priority-dev-20261009/query_clear_by_category.csv']
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(task='CNH_VERTICAL_GEOMETRY_JOIN_DEV_20261009', lane='EXPLORE consumed cache',
                goal='Join pressure clear geometry and Dwin shallow-intrusion loss conditions; narrow next hypothesis',
                authorization='User consensus step2: existing ledger join only; decide separability diagnosis after results',
                budget_CPU_wall_seconds=60, unit='Cumulative cache joining and focused independent recomputation wall seconds',
                adjustable_scope='Join repair and reading saved per-frame scores for missing event alarm distances only',
                stop='Complete fixed ±2/±3 join plus focused check, no new experiment or outcome-driven threshold/group search',
                interpretation='True clear gap is fixed15cm; saved negative SAT overlap margin is a separating-axis proxy, not Euclidean lateral gap',
                information='Geometry and shapes are evaluator-only, never added to algorithm/gate. No statistical stop criterion.',
                inputs_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    write_json(OUT/'PLAN.json', plan)
    shape_plan = json.loads(inputs[0].read_text(encoding='utf-8'))
    rows = shape_plan['scene_rows']
    with np.load(SHAPE/'geometry.npz') as z:
        sensor = z['sensor']
        categories = z['category']
    distances = {f: float(rows[0]['lo'][2]-sensor[f, 2, 3]) for f in range(3, 16)}
    for f, distance in distances.items():
        assert abs(distance-(3.05-.16*f)) < 1e-12
    clear_source = [r for r in read_csv(QUERY/'clear_slots.csv') if r['family']=='vertical' and int(r['angle']) in PRESSURE]
    geometry = {(r['branch'], int(r['scene']), int(r['frame']), r['query']): r
                for r in read_csv(QUERY/'review/geometry.csv')}
    assert len(clear_source) == 4992 and len(geometry) == 13728
    assert len({(r['angle'], r['scene'], r['replica'], r['frame']) for r in clear_source}) == 4992
    joined = []
    for r in clear_source:
        a, scene, f = int(r['angle']), int(r['scene']), int(r['frame'])
        row = rows[scene]
        m = meta(row)
        assert (categories[scene]=='clear').all() and m['nominal_signed_boundary_gap_cm']==15
        g = {h: geometry[f'query_{abs(a)}_{int(np.sign(a)):+d}', scene, f, h] for h in ('HEAD', 'BODY')}
        relevant = ('HEAD', 'BODY') if row['group']=='BOTH' else (row['group'],)
        margin = min(float(g[h]['target_SAT_overlap_margin_m']) for h in relevant)
        # Margin remains the original evaluator SAT result; no new box transform.
        rec = dict(angle=a, scene=scene, replica=int(r['replica']), frame=f, distance_m=distances[f],
                   timely_window=int(f<=13), **m,
                   rotation_relation='toward_boundary' if a*m['side']<0 else 'away_from_boundary',
                   relevant_target_SAT_margin_m=margin,
                   relevant_axis_separation_m=max(0., -margin),
                   relevant_target_contact=int(any(int(g[h]['target_contact']) for h in relevant)),
                   M3=int(r['M3']), raised=int(r['raised']), local=int(r['local']), fusion=int(r['fusion']),
                   ideal=int(r['ideal']), ideal_added=int(r['ideal_added']), ideal_lost=int(r['ideal_lost']))
        for h in ('HEAD', 'BODY'):
            rec['M3_'+h] = int(float(r['M_'+h])>=T0)
            rec['fusion_'+h] = int(float(r['M_'+h])>=TM or float(r['L_'+h])>=TL)
            rec['SAT_margin_'+h+'_m'] = float(g[h]['target_SAT_overlap_margin_m'])
            rec['target_contact_'+h] = int(g[h]['target_contact'])
        assert rec['M3'] == int(rec['M3_HEAD'] or rec['M3_BODY'])
        assert rec['fusion'] == int(rec['fusion_HEAD'] or rec['fusion_BODY'])
        joined.append(rec)

    def costs(ee):
        cc = defaultdict(dict)
        for r in ee:
            cc[r['scene'], r['replica']][r['frame']] = r
        out = dict(denominator=len(ee), clip_denominator=len(cc))
        for method in ('M3', 'fusion'):
            out[method+'_slots'] = sum(r[method] for r in ee)
            out[method+'_segments'] = sum(sum(frames[f][method] and (f-1 not in frames or not frames[f-1][method])
                                                  for f in sorted(frames)) for frames in cc.values())
            out[method+'_clips'] = sum(any(r[method] for r in frames.values()) for frames in cc.values())
        out['added_vs_ideal'] = sum(r['ideal_added'] for r in ee)
        out['removed_vs_ideal'] = sum(r['ideal_lost'] for r in ee)
        return out

    summary = []
    by_frame = []
    by_condition = []
    gap_summary = []
    reference = {(int(r['angle']), r['category']): r for r in read_csv(inputs[-1])}
    for a in PRESSURE:
        ee = [r for r in joined if r['angle']==a]
        rec = dict(angle=a, **costs(ee))
        old = reference[a, 'vertical']
        for method in ('M3', 'fusion'):
            for metric in ('slots', 'segments', 'clips'):
                assert rec[method+'_'+metric] == int(old[method+'_'+metric])
        rec['net_vs_ideal'] = rec['added_vs_ideal']-rec['removed_vs_ideal']
        assert rec['denominator']==1248 and rec['clip_denominator']==96
        summary.append(rec)
        for f in range(3, 16):
            ff = [r for r in ee if r['frame']==f]
            by_frame.append(dict(angle=a, frame=f, distance_m=distances[f], timely_window=int(f<=13), **costs(ff)))
        for width in (4., 10.):
            for side in (-1, 1):
                ss = [r for r in ee if r['width_cm']==width and r['side']==side]
                by_condition.append(dict(angle=a, width_cm=width, side=side,
                                         rotation_relation=ss[0]['rotation_relation'], **costs(ss)))
        for method in ('M3', 'fusion'):
            alarms = [r for r in ee if r[method]]
            gap_summary.append(dict(angle=a, method=method, nominal_gap_cm=15,
                                    alarm_slots=len(alarms), target_contact_slots=sum(r['relevant_target_contact'] for r in alarms),
                                    separation_proxy_min_m=min(r['relevant_axis_separation_m'] for r in alarms),
                                    separation_proxy_max_m=max(r['relevant_axis_separation_m'] for r in alarms),
                                    toward_slots=sum(r['rotation_relation']=='toward_boundary' for r in alarms),
                                    away_slots=sum(r['rotation_relation']=='away_from_boundary' for r in alarms)))

    ledger = [r for r in read_csv(BOUNDARY/'event_ledger.csv') if r['probe']=='Dwin']
    losses = []
    loss_windows = []
    loss_summary = []
    for a in (0, -3, 3):
        with np.load(BOUNDARY/f'scores_{a:+d}.npz') as z:
            ids = z['scene_ids']; score = z['smooth'][:,:,:,2,:]; m3 = z['m3']; local = z['local']
        index = {int(i): n for n, i in enumerate(ids)}
        veto = np.isfinite(score) & (score<0)
        for base in ('M3', 'fusion'):
            flags = m3>=T0 if base=='M3' else (m3>=TM)|(local>=TL)
            kept = flags & ~veto
            for h, hi in (('HEAD', 0), ('BODY', 1)):
                group_ledger = [r for r in ledger if int(r['angle'])==a and r['base']==base and r['height']==h]
                assert len(group_ledger)==224
                selected = []
                for e in group_ledger:
                    scene, k = int(e['scene']), int(e['replica'])
                    ii = index[scene]
                    old = flags[ii,k,:11,hi].any()
                    new = kept[ii,k,:11,hi].any()
                    assert int(old)==int(e['baseline']) and int(new)==int(e['candidate'])
                    assert int(old and not new)==int(e['loss'])
                    if e['loss']!='1':
                        continue
                    ff = np.flatnonzero(flags[ii,k,:11,hi])+3
                    assert len(ff)>0 and not kept[ii,k,:11,hi].any()
                    m = meta(rows[scene])
                    assert rows[scene]['placement'] in ('in_1cm', 'in_4cm')
                    rec = dict(angle=a, base=base, height=h, scene=scene, replica=k, **m,
                               rotation_relation='toward_boundary' if a*m['side']<0 else 'away_from_boundary',
                               first_baseline_alarm_frame=int(ff[0]), first_alarm_distance_m=distances[int(ff[0])],
                               last_baseline_alarm_frame=int(ff[-1]), last_alarm_distance_m=distances[int(ff[-1])],
                               removed_timely_height_windows=len(ff))
                    losses.append(rec); selected.append(rec)
                    for f in ff:
                        loss_windows.append(dict(angle=a, base=base, height=h, scene=scene, replica=k, frame=int(f),
                                                 distance_m=distances[int(f)], **m, removed_original_alarm=1))
                loss_summary.append(dict(angle=a, base=base, height=h, denominator=224, lost_events=len(selected),
                                         in_1cm=sum(r['placement']=='in_1cm' for r in selected),
                                         in_4cm=sum(r['placement']=='in_4cm' for r in selected),
                                         removed_timely_height_windows=sum(r['removed_timely_height_windows'] for r in selected)))

    # Match conditions, not clear/contact event identities; no performance claim.
    matching = []
    for a in (-3, 3):
        for base in ('M3', 'fusion'):
            for relation in ('same_side', 'mirrored_side'):
                selected = [r for r in losses if r['angle']==a and r['base']==base]
                keys = {(r['side'] if relation=='same_side' else -r['side'], r['width_cm'], r['rho'], r['variant']) for r in selected}
                ee = [r for r in joined if r['angle']==a and r['timely_window'] and
                      (r['side'], r['width_cm'], r['rho'], r['variant']) in keys]
                matching.append(dict(angle=a, base=base, condition_relation=relation,
                                     loss_events=len(selected), selected_condition_cells=len(keys),
                                     clear_timely_denominator=len(ee), clear_alarm_slots=sum(r[base] for r in ee)))
    group_windows = defaultdict(list)
    for r in loss_windows:
        group_windows[r['angle'], r['base'], r['height'], r['frame']].append(r)
    loss_by_frame = [dict(angle=k[0], base=k[1], height=k[2], frame=k[3], distance_m=distances[k[3]],
                          lost_event_alarm_windows=len(v)) for k,v in sorted(group_windows.items())]
    report = dict(status='PASS', summary=summary, by_condition=by_condition, by_frame=by_frame,
                  gap_proxy=gap_summary, loss_summary=loss_summary, matching=matching, loss_by_frame=loss_by_frame,
                  joined_rows=len(joined), loss_events=len(losses), loss_windows=len(loss_windows),
                  nominal_clear_gap_values_cm=sorted({r['nominal_signed_boundary_gap_cm'] for r in joined}),
                  nominal_loss_gap_values_cm=sorted({r['nominal_signed_boundary_gap_cm'] for r in losses}),
                  distances_m={str(f): d for f,d in distances.items()},
                  missing='Dwin ±2 unavailable; contact-loss rotated SAT geometry unavailable in existing clear-only geometry ledger',
                  seconds=time.monotonic()-start, inference=0, training=0, new_samples=0, new_geometry=0,
                  statistical_tests=0, separability_diagnosis='NOT_RUN')
    assert report['seconds']<60
    write_csv(OUT/'clear_geometry_join.csv', joined)
    write_csv(OUT/'clear_by_frame.csv', by_frame)
    write_csv(OUT/'clear_by_width_side.csv', by_condition)
    write_csv(OUT/'dwin_loss_events.csv', losses)
    write_csv(OUT/'dwin_loss_alarm_windows.csv', loss_windows)
    write_csv(OUT/'condition_comparison.csv', matching)
    write_json(OUT/'result.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'joined_rows', 'loss_events', 'loss_windows', 'seconds')}, ensure_ascii=False))


if __name__=='__main__':
    main()

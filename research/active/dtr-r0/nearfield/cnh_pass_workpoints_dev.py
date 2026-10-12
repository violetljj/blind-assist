"""Calibrated clear/pass workpoints from frozen saved scores, CPU only.

This consumed-Development diagnostic performs no training or forward pass.
Only ideal calibration chooses a threshold; all other branches transport it.
Threshold levels are distinct finite smoothed cal scores plus +infinity,
with score>=threshold whole ties. The lowest feasible level is selected.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import cnh_counterfactual_eval_dev as E


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'artifacts.local/work/cnh-counterfactual-dev-20261009'
OUT = ROOT / 'artifacts.local/work/cnh-pass-boundary-dev-20261009'
ARMS = ('base', 'ordinary')
POLICIES = ('standalone', 'old_fusion_plus_candidate')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def exposure(category):
    contact = category == 'contact'
    return (category == 'clear').all(1), (category == 'pass').any(1) & ~contact.any(1)


def select_threshold(scores, category, *, clear_cap, pass_cap, old_flags=None):
    """Lowest feasible whole-score-tie level, without optimizing contact recall.

    Clear vectors are max-query per time slot. Pass vectors are max over every
    retained f3..15 and both queries, per scene x replica clip. For OR, subtract
    already occupied old slots/clips from the eligible vectors, not histograms.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 4 or scores.shape[-2:] != (13, 2) or not np.isfinite(scores).all():
        raise ValueError('Finite scores [scene,K,13,2] required')
    clear, passed = exposure(category)
    joint = scores.max(-1)
    clear_values = joint[clear]
    pass_values = joint[passed].max(-1)
    base_clear = base_pass = 0
    if old_flags is not None:
        if old_flags.shape != scores.shape:
            raise ValueError('Fixed OR flags must have identical score shape')
        old_joint = old_flags.any(-1)
        occupied_clear = old_joint[clear]
        occupied_pass = old_joint[passed].any(-1)
        base_clear, base_pass = int(occupied_clear.sum()), int(occupied_pass.sum())
        clear_values = clear_values[~occupied_clear]
        pass_values = pass_values[~occupied_pass]
    meta = dict(clear_cap=int(clear_cap), pass_cap=int(pass_cap),
                clear_floor=base_clear, pass_floor=base_pass,
                pass_denominator=int(passed.sum()*scores.shape[1]),
                clear_denominator=int(clear.sum()*scores.shape[1]*13),
                threshold_grid='all distinct finite smoothed ideal-cal scores plus +inf; score>=theta',
                selection='lowest feasible whole-tie grid level; no contact objective')
    if base_clear > clear_cap or base_pass > pass_cap:
        reason = 'PASS_FLOOR' if base_pass > pass_cap else 'CLEAR_FLOOR'
        return dict(meta, status='NOT_POSSIBLE_FIXED_OR_'+reason, threshold=None)
    levels = np.r_[np.unique(scores), np.inf]
    cc, pp = np.sort(clear_values.reshape(-1)), np.sort(pass_values.reshape(-1))

    def costs(theta):
        return (base_clear + len(cc)-int(np.searchsorted(cc, theta, side='left')),
                base_pass + len(pp)-int(np.searchsorted(pp, theta, side='left')))

    lo, hi = 0, len(levels)-1
    while lo < hi:
        mid = (lo+hi)//2
        c, p = costs(levels[mid])
        if c <= clear_cap and p <= pass_cap:
            hi = mid
        else:
            lo = mid+1
    selected = float(levels[lo])
    c, p = costs(selected)
    if c > clear_cap or p > pass_cap:
        raise AssertionError('All-suppressed candidate should be feasible above fixed floors')
    if lo and all(a <= b for a, b in zip(costs(levels[lo-1]), (clear_cap, pass_cap))):
        raise AssertionError('Selected threshold was not the lowest feasible level')
    return dict(meta, status='CALIBRATED', threshold=E.record_threshold(selected),
                clear_slots=c, pass_clips=p, grid_levels=len(levels), selected_grid_index=lo)


def flags_at(scores, record, policy, old):
    if record['threshold'] is None:
        return None
    flags = scores >= E.threshold(record['threshold'])
    if policy == 'old_fusion_plus_candidate':
        flags |= old
        if not np.all(~old | flags):
            raise AssertionError('Frozen OR must never revoke an old alarm')
    return flags


def describe(data, flags, references):
    metrics, timely = E.summary(flags, data['category'])
    comparisons = {}
    for name, old in references.items():
        if old is not None:
            comparisons[name] = E.paired(E.summary(old, data['category'])[1], timely,
                                         data['category'], data['scene_ids'])
    groups = dict(data['groups'])
    placements = np.array([r['placement'] for r in data['rows']])
    for value in sorted(set(placements.tolist())):
        groups['placement:'+value] = placements == value
    strata = {}
    for name, keep in groups.items():
        gm, gt = E.summary(flags[keep], data['category'][keep])
        strata[name] = dict(metrics=gm, paired={key: E.paired(
            E.summary(old[keep], data['category'][keep])[1], gt,
            data['category'][keep], data['scene_ids'][keep])
            for key, old in references.items() if old is not None})
    return dict(metrics=metrics, paired=comparisons, groups=strata), timely


def scalar(dataset, arm, seed, policy, point, stratum, report):
    m = report['metrics']
    row = dict(dataset=dataset, arm=arm, seed=seed, policy=policy, point=point, stratum=stratum,
               HEAD=m['counts'][0], BODY=m['counts'][1], HEAD_denominator=m['contact_event_denominators'][0],
               BODY_denominator=m['contact_event_denominators'][1], clear_slots=m['clear_slots'],
               clear_denominator=m['clear_denominator'], clear_segments=m['clear_segments'],
               clear_clips=m['clear_clips'], clear_clip_denominator=m['clear_clip_denominator'],
               pass_clips=m['pass_clips'], pass_clip_denominator=m['pass_clip_denominator'],
               physical=m['physical_contact_any_height'], physical_denominator=m['physical_contact_denominator'])
    for name in ('M3', 'old_local_fusion', 'reference_same_arm_same_policy',
                 'base_same_point_same_policy', 'ordinary_same_point_same_policy', 'control_same_point_same_policy'):
        for q, height in enumerate(('HEAD', 'BODY')):
            for field in ('gain', 'loss', 'net'):
                row[height+'_'+field+'_vs_'+name] = report['paired'][name][q][field] if name in report['paired'] else ''
    return row


def mdtable(header, rows):
    return '\n'.join(['| '+' | '.join(map(str, header))+' |', '| '+' | '.join(['---']*len(header))+' |']
                     + ['| '+' | '.join(map(str, row))+' |' for row in rows])


def run(plan_path, output=OUT/'workpoints_initial', source=SOURCE, manifest_path=None):
    began = time.monotonic()
    plan_path, output, source = Path(plan_path).resolve(), Path(output).resolve(), Path(source).resolve()
    if not plan_path.is_file():
        raise FileNotFoundError('Root must freeze this task PLAN before diagnostic execution')
    if not output.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('All outputs must remain under canonical artifacts.local')
    plan = read(plan_path)
    rates = plan.get('workpoint_pass_rates', [.2, .3, .4])
    if rates != [.2, .3, .4]:
        raise ValueError('This fixed diagnostic implements only the three declared pass budgets')
    supplied_manifest = manifest_path is not None
    manifest_path = Path(manifest_path).resolve() if supplied_manifest else source/'eval_manifest.json'
    manifest = read(manifest_path)
    data_root = manifest_path.parent
    arms = manifest['arm_names'] if supplied_manifest else list(ARMS)
    if 'base' not in arms or 'ordinary' not in arms:
        raise ValueError('Retain frozen base and ordinary controls alongside new arms')
    seeds = manifest['seed_ids']
    if plan.get('seeds', seeds) != seeds:
        raise ValueError('Preserve every frozen seed; no subset selection')
    cap = float(plan.get('cpu_analysis_cap_seconds', 600))
    prior = float(plan.get('diagnostic_prior_cpu_seconds', 0))

    def check():
        if prior+time.monotonic()-began >= cap:
            raise TimeoutError('CPU diagnostic budget reached')

    work = output
    if work.exists():
        raise FileExistsError('Preserve workpoint diagnostics; do not overwrite previous results')
    work.mkdir(parents=True)
    receipt = dict(status='RUNNING', plan_sha256=sha(plan_path), source_sha256=sha(__file__),
                   cpu_only=True, training=0, inference=0, threshold_origin='cal/ideal only')
    try:
        previous = read(source/'evaluation/thresholds.json')['arms']
        datasets, sources = {}, {manifest_path, source/'evaluation/thresholds.json',
                                 source/'evaluation/metrics.json', plan_path, Path(__file__)}
        for spec in manifest['datasets']:
            check()
            selected = dict(spec, scores=[s for s in spec['scores'] if s['arm'] in arms])
            data = E.load_dataset(selected, data_root, arms, seeds)
            datasets[data['split']+'/'+data['branch']] = data
            for name in ('physics', 'scene_rows', 'baseline'):
                sources.add((data_root/spec[name]).resolve())
            sources.update((data_root/s['path']).resolve() for s in selected['scores'])
        if set(datasets) != {'cal/ideal', 'cal/yaw_plus3', 'validation/ideal', 'validation/yaw_plus3'}:
            raise ValueError('Exactly four fixed split/branch cohorts required')
        for split in ('cal', 'validation'):
            E.pose_identity(datasets[split+'/ideal'], datasets[split+'/yaw_plus3'])
        cal = datasets['cal/ideal']
        old_cal = E.old_fusion(cal['m3'], cal['local'])
        cm, _ = E.summary(cal['m3'] >= E.M3_THETA, cal['category'])
        om, _ = E.summary(old_cal, cal['category'])
        thresholds = {}
        for ai, arm in enumerate(arms):
            thresholds[arm] = {}
            for si, seed in enumerate(seeds):
                thresholds[arm][str(seed)] = {}
                for policy in POLICIES:
                    inherited = previous.get(arm, {}).get(str(seed))
                    if inherited is None:
                        inherited = E.calibrate(cal['m3'], cal['local'], cal['candidates'][ai,si], cal['category'])
                    result = {'reference': dict(status='FROZEN_REFERENCE' if arm in previous else 'CALIBRATED_CLEAR_ONLY_REFERENCE',
                        threshold=inherited[policy], selection='same clear-only policy; cal only')}
                    for rate in rates:
                        pass_cap = math.floor(rate*cm['pass_clip_denominator'])
                        record = select_threshold(cal['candidates'][ai, si], cal['category'],
                            clear_cap=cm['clear_slots'] if policy == 'standalone' else om['clear_slots'],
                            pass_cap=pass_cap, old_flags=None if policy == 'standalone' else old_cal)
                        result[f'pass_{round(rate*100)}pct'] = dict(record, pass_rate_budget=rate)
                    thresholds[arm][str(seed)][policy] = result
        save(work/'thresholds.json', dict(status='COMPLETE', only_source='cal/ideal', arms=thresholds,
             calibration_M3=cm, calibration_old_local_fusion=om))
        result = dict(status='COMPLETE', arms=arms, seeds=seeds, datasets={},
                      threshold_origin='cal/ideal only; fixed for validation and yaw',
                      interpretation='Consumed Development diagnostic; no fresh confirmation or deployment claim',
                      training=0, inference=0, calibration_floors=dict(M3=cm, old_local_fusion=om))
        scalars, event_rows = [], 0
        event_header = ['dataset', 'arm', 'seed', 'policy', 'point', 'scene', 'replica', 'height',
                        'placement', 'shape_family', 'background_family', 'timely', 'late_only',
                        'M3_timely', 'old_local_fusion_timely', 'reference_timely', 'base_timely', 'ordinary_timely']
        with (work/'event_ledger.csv').open('x', encoding='utf8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=event_header)
            writer.writeheader()
            for dataset_name, data in datasets.items():
                check()
                baseline = dict(M3=data['m3'] >= E.M3_THETA,
                                old_local_fusion=E.old_fusion(data['m3'], data['local']))
                cache = {}
                for ai, arm in enumerate(arms):
                    for si, seed in enumerate(seeds):
                        for policy in POLICIES:
                            for point, threshold in thresholds[arm][str(seed)][policy].items():
                                cache[arm, seed, policy, point] = flags_at(data['candidates'][ai, si],
                                    threshold, policy, baseline['old_local_fusion'])
                dataset_report = dict(baseline={n:E.summary(f, data['category'])[0] for n,f in baseline.items()}, arms={})
                result['datasets'][dataset_name] = dataset_report
                for arm in arms:
                    dataset_report['arms'][arm] = {}
                    for seed in seeds:
                        dataset_report['arms'][arm][str(seed)] = {}
                        for policy in POLICIES:
                            policy_report = {}
                            dataset_report['arms'][arm][str(seed)][policy] = policy_report
                            for point, threshold in thresholds[arm][str(seed)][policy].items():
                                check()
                                flags = cache[arm, seed, policy, point]
                                if flags is None:
                                    policy_report[point] = dict(status=threshold['status'], calibration=threshold,
                                        metrics=None, paired=None, groups={},
                                        frozen_OR_floor_metrics=dataset_report['baseline']['old_local_fusion'])
                                    continue
                                refs = dict(baseline,
                                    reference_same_arm_same_policy=cache[arm, seed, policy, 'reference'],
                                    base_same_point_same_policy=cache['base', seed, policy, point],
                                    ordinary_same_point_same_policy=cache['ordinary', seed, policy, point])
                                if 'control' in arms:
                                    refs['control_same_point_same_policy'] = cache['control', seed, policy, point]
                                report, timely = describe(data, flags, refs)
                                report.update(status=threshold['status'], calibration=threshold,
                                              threshold=threshold['threshold'])
                                policy_report[point] = report
                                if dataset_name == 'cal/ideal' and point != 'reference':
                                    if report['metrics']['clear_slots'] > threshold['clear_cap'] or report['metrics']['pass_clips'] > threshold['pass_cap']:
                                        raise AssertionError('Actual cal flags violate selected workpoint constraints')
                                scalars.append(scalar(dataset_name, arm, seed, policy, point, 'all', report))
                                for stratum, gr in report['groups'].items():
                                    scalars.append(scalar(dataset_name, arm, seed, policy, point, stratum, gr))
                                rt = {key:E.summary(value, data['category'])[1] for key,value in refs.items()}
                                for i, k, q in np.argwhere(np.broadcast_to((data['category']=='contact')[:,None,:], timely.shape)):
                                    row = data['rows'][i]
                                    writer.writerow(dict(dataset=dataset_name, arm=arm, seed=seed, policy=policy,
                                        point=point, scene=int(data['scene_ids'][i]), replica=int(k),
                                        height=('HEAD','BODY')[q], placement=row['placement'],
                                        shape_family=row['shape_family'], background_family=row['background_family'],
                                        timely=int(timely[i,k,q]), late_only=int(not timely[i,k,q] and flags[i,k,11:,q].any()),
                                        M3_timely=int(rt['M3'][i,k,q]), old_local_fusion_timely=int(rt['old_local_fusion'][i,k,q]),
                                        reference_timely=int(rt['reference_same_arm_same_policy'][i,k,q]),
                                        base_timely=int(rt['base_same_point_same_policy'][i,k,q]),
                                        ordinary_timely=int(rt['ordinary_same_point_same_policy'][i,k,q])))
                                    event_rows += 1
        check()
        result['event_ledger_rows'] = event_rows
        save(work/'metrics.json', result)
        with (work/'summary.csv').open('x', encoding='utf8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scalars[0]))
            writer.writeheader(); writer.writerows(scalars)
        lines = ['# 固定模型 clear/pass 工作点（已消费 Development）', '',
                 '只在 ideal cal 选最低可行完整 score-tie 阈值；validation 与 query +3° 固定使用该阈值。'
                 '独立读出 clear cap=M3 实际 cal 成本；OR clear cap=旧融合 cal 成本且保留全部旧报警。'
                 'pass 以 scene×replica 全 f3..15、任 query 报警计，contact 及时仍为 f3..13。'
                 '20/30/40% 用 floor(rate×cal pass denominator) 整数 clip 预算；'
                 '无优化接触命中，无 validation 选点；各工作点和三个 seed 全部保留。', '',
                 f"cal pass denominator={cm['pass_clip_denominator']}，M3 clear/pass={cm['clear_slots']}/{cm['pass_clips']}，"
                 f"旧融合 clear/pass={om['clear_slots']}/{om['pass_clips']}。低于固定 OR floor 的点明确不可行。", '']
        for dataset_name, dataset in result['datasets'].items():
            lines += ['## '+dataset_name, '']
            rows = []
            for name,m in dataset['baseline'].items():
                rows.append([name,'—','fixed','reference','FROZEN',m['counts'][0],m['counts'][1],m['clear_slots'],
                             m['clear_segments'],m['clear_clips'],m['pass_clips'],m['physical_contact_any_height'],'—','—'])
            for arm, ss in dataset['arms'].items():
                for seed, pp in ss.items():
                    for policy, points in pp.items():
                        for point, report in points.items():
                            m = report['metrics']
                            gains = ['—','—'] if m is None else [
                                f"{p['gain']}/{p['loss']}" for p in report['paired']['reference_same_arm_same_policy']]
                            rows.append([arm,seed,policy,point,report['status']] +
                                ([m['counts'][0],m['counts'][1],m['clear_slots'],m['clear_segments'],m['clear_clips'],m['pass_clips'],
                                  m['physical_contact_any_height']] if m else ['—']*7) + gains)
            lines += [mdtable(['方法','seed','策略','工作点','status','HEAD /384','BODY /384','clear /6656',
                              '段','clips /512','pass /256','physical /768','vsref H救/损','vsref B救/损'],rows),'']
        lines += ['完整同点 Base/Ordinary 与 M3/旧融合配对损失、1/4/12cm placement、shape/background strata，'
                  '见 metrics.json / summary.csv；事件身份与及时/晚响见 event_ledger.csv。'
                  '这些工作点只诊断本套模板和已消费 Development，不是新的确认或部署证书。','']
        (work/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')
        check()
        save(work/'input_manifest.json', {str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(sources)})
        receipt.update(status='COMPLETE', event_rows=event_rows, scalar_rows=len(scalars),
                       clear_cal_cap=cm['clear_slots'], OR_clear_cal_cap=om['clear_slots'],
                       pass_cal_denominator=cm['pass_clip_denominator'], fixed_OR_pass_floor=om['pass_clips'])
    except Exception as error:
        receipt.update(status='FAILED',error=repr(error))
        raise
    finally:
        receipt.update(seconds=time.monotonic()-began, prior_cpu_seconds=prior, cpu_cap_seconds=cap)
        save(output/'run_receipt.json', receipt)
    print(json.dumps(receipt))
    return result


def check_fixture():
    """Costs, whole ties, frozen OR infeasibility and fixed-threshold transport."""
    cat = np.array([['clear','clear'],['pass','clear'],['contact','clear']])
    score = np.zeros((3,2,13,2)); score[0,0,:,0]=1; score[0,1,:,0]=2
    score[1,0,:,0]=3; score[1,1,:,0]=4; score[2]=5
    r = select_threshold(score,cat,clear_cap=13,pass_cap=1)
    assert E.threshold(r['threshold'])==4 and r['pass_clips']==1 and r['clear_slots']==0
    old = np.zeros_like(score,dtype=bool); old[1,0,0,0]=True
    impossible=select_threshold(score,cat,clear_cap=13,pass_cap=0,old_flags=old)
    assert impossible['status']=='NOT_POSSIBLE_FIXED_OR_PASS_FLOOR' and impossible['threshold'] is None
    r = select_threshold(score,cat,clear_cap=13,pass_cap=1,old_flags=old)
    assert E.threshold(r['threshold'])==5 and r['pass_clips']==1
    flags=flags_at(score,r,'old_fusion_plus_candidate',old)
    assert flags[1,0,0,0] and not flags[1,1].any()
    tied=np.zeros_like(score);tied[1]=2
    r=select_threshold(tied,cat,clear_cap=26,pass_cap=1)
    assert r['threshold']['positive_infinity'] and r['pass_clips']==0
    transported=score+10
    assert E.summary(flags_at(transported,r,'standalone',old),cat)[0]['pass_clips']==0
    print('PASS whole ties, floor, lowest feasible level, OR retention, fixed transport fixtures')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=OUT/'PLAN.json')
    parser.add_argument('--output',type=Path,default=OUT/'workpoints_initial')
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--manifest',type=Path,help='Optional combined base/ordinary/new-arm score manifest')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    check_fixture() if args.check else run(args.plan,args.output,args.source,args.manifest)

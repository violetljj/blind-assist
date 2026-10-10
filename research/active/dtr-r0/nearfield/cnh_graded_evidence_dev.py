"""Cache-only graded-alert Explore; causal policies never consume evaluator rows.

Frozen M3/local and all three ordinary seeds, ideal cal/validation only. Local is
a query-inside support proxy, not an independent obstacle/exterior classifier.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import cnh_counterfactual_eval_dev as E

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'artifacts.local/work/cnh-counterfactual-dev-20261009'
OUT = ROOT / 'artifacts.local/work/cnh-graded-evidence-dev-20261010'
SEEDS = (2026100955, 2026100956, 2026100957)
POLICIES = ('dual', 'dual_max3', 'dual_k2of3')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temporal(raw, kind):
    """Past-only raw-logit order statistic; 2-of-3 needs two observations."""
    out = np.full_like(raw, -np.inf, dtype=float)
    for t in range(13):
        window = raw[..., max(0, t-2):t+1, :]
        if kind == 'max3':
            out[..., t, :] = window.max(-2)
        elif window.shape[-2] >= 2:
            out[..., t, :] = np.sort(window, axis=-2)[..., -2, :]
    return out


def trend(raw):
    slope = np.zeros_like(raw, dtype=float)
    residual = np.zeros_like(raw, dtype=float)
    for t in range(13):
        y = raw[..., max(0, t-4):t+1, :]
        x = np.arange(y.shape[-2], dtype=float)
        x -= x.mean()
        mean = y.mean(-2)
        if len(x) > 1:
            b = (y*x[:, None]).sum(-2)/(x*x).sum()
            slope[..., t, :] = b
            residual[..., t, :] = np.sqrt(((y-mean[..., None, :]-b[..., None, :]*x[:, None])**2).mean(-2))
    return slope, residual


def grades(m3, local, ordinary, single_theta, or_theta, extra=None, extra_theta=None):
    """0 silent, 1 light, 2 strong. Thresholds are shared HEAD/BODY."""
    exists = local >= E.OLD_LOCAL
    intrusion = ordinary >= single_theta
    strong = (m3 >= E.OLD_RAISED) | (exists & intrusion) | (ordinary >= or_theta)
    any_alert = E.old_fusion(m3, local) | intrusion | strong
    if extra is not None:
        any_alert |= extra >= extra_theta
    return np.where(strong, 2, np.where(any_alert, 1, 0)).astype(np.int8)


def first(flags, end=13):
    seen = flags[..., :end, :].any(-2)
    return np.where(seen, flags[..., :end, :].argmax(-2)+3, -1)


def paired_timing(before, after, category):
    a, b = first(before, 11), first(after, 11)
    reports = []
    for q in range(2):
        mask = np.broadcast_to((category[:, q] == 'contact')[:, None], a[..., q].shape)
        aa, bb = a[..., q][mask], b[..., q][mask]
        both = (aa >= 0) & (bb >= 0)
        delta = aa[both]-bb[both]
        reports.append(dict(height=E.HEIGHTS[q], denominator=int(mask.sum()),
            rescue=int(((aa < 0) & (bb >= 0)).sum()), loss=int(((aa >= 0) & (bb < 0)).sum()),
            both_timely=int(both.sum()), earlier=int((delta > 0).sum()), same=int((delta == 0).sum()),
            later=int((delta < 0).sum()), median_advance_frames=float(np.median(delta)) if len(delta) else None,
            advance_frame_histogram={str(int(d)): int((delta == d).sum()) for d in np.unique(delta)}))
    return reports


def cost(joint_flags, mask):
    flags = joint_flags[mask]
    return dict(slots=int(flags.sum()), slot_denominator=int(flags.size),
        clips=int(flags.any(-1).sum()), clip_denominator=int(np.prod(flags.shape[:-1])),
        segments=int((np.diff(np.pad(flags.astype(np.int8), ((0,0),(0,0),(1,0))), axis=-1) == 1).sum()),
        longest_run_frames=max((max_run(row) for row in flags.reshape(-1, 13)), default=0))


def max_run(row):
    run = longest = 0
    for value in row:
        run = run+1 if value else 0
        longest = max(longest, run)
    return longest


def describe(grade, data, references):
    category = data['category']
    result = {}
    for name, flags in (('any', grade > 0), ('strong', grade == 2), ('light', grade == 1)):
        metrics, _ = E.summary(flags, category)
        result[name] = metrics
    result['paired_any'] = {name: paired_timing(flags, grade > 0, category) for name, flags in references.items()}
    result['paired_strong'] = {name: paired_timing(flags, grade == 2, category) for name, flags in references.items()}
    clear = (category == 'clear').all(1)
    passed = (category == 'pass').any(1) & ~(category == 'contact').any(1)
    joint = grade.max(-1)
    result['joint_costs'] = {kind: {level: cost(joint == number, mask)
        for level, number in (('light',1), ('strong',2))} for kind, mask in (('clear',clear), ('pass',passed))}
    timely_strong = (grade[..., :11, :] == 2).any(-2)
    timely_light = (grade[..., :11, :] == 1).any(-2)
    result['light_only_timely_contact'] = ((timely_light & ~timely_strong)
        & (category == 'contact')[:, None, :]).sum((0,1)).tolist()
    result['old_alert_query_slots_to_light'] = int((references['old_fusion'] & (grade == 1)).sum())
    result['ordinary_OR_query_slots_to_light'] = int((references['ordinary_OR'] & (grade == 1)).sum())
    result['contact_old_timely_to_light_only'] = ((references['old_fusion'][..., :11, :].any(-2)
        & timely_light & ~timely_strong) & (category == 'contact')[:,None,:]).sum((0,1)).tolist()
    groups = {}
    for name, mask in data['groups'].items():
        groups[name] = {'any': E.summary((grade > 0)[mask], category[mask])[0],
            'vs_old_fusion': paired_timing(references['old_fusion'][mask], (grade > 0)[mask], category[mask])}
    result['groups'] = groups
    return result


def write_csv(path, rows):
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    began = time.monotonic()
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Preserve completed or failed run; no implicit overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [SOURCE/'eval_manifest.json', SOURCE/'evaluation/thresholds.json', SOURCE/'scene_rows.json']
    inputs += [SOURCE/f'baselines/{s}_ideal.npz' for s in ('cal','validation')]
    inputs += [SOURCE/f'scores/ordinary_seed{seed}_{s}.npz' for s in ('cal','validation') for seed in SEEDS]
    plan = dict(task='CNH_GRADED_EVIDENCE_DEV_20261010', lane='EXPLORE consumed simulated Development',
        authorization='User 推进 corrected discussion, including diagnosis/pilots/docs/delivery',
        goal='Test useful timely light alerts and explicit strong/light costs; diagnose weak/thin misses',
        budget=dict(score_analysis_CPU_command_seconds=600, raw_echo_CPU_command_seconds=600,
                    GPU_seconds=0, training=0, inference=0, sampling=0),
        stop='Complete declared cache comparisons and sampled echo diagnostic; preserve failures and old frozen runs',
        adjustable_scope='Implementation repair and proportionate numerical verification; no protected480/hardware access',
        seeds=list(SEEDS), seed_use='One ordinary model at runtime; all three seeds reported separately, never ensemble/vote',
        existence_proxy='Old query-inside local patch score >= frozen4.625390338985158; not exterior/independent presence',
        intrusion_proxy='Ordinary smoothed contact-trained readout >= its old single-cal threshold; not known body penetration',
        strong='M3>=oldraised OR (local>=oldlocal AND ordinary>=single) OR ordinary>=oldORaddition',
        light='Not strong AND (oldfusion OR ordinary>=single OR optional temporal>=caltheta)',
        policies=list(POLICIES), temporal='Past raw max3 or second-largest of last<=3 with >=2 observations; only adds light',
        calibration='Keep existing single/OR thresholds. Each temporal theta: whole-score ties nearest existing65 joint-clear cal slots, higher theta on equal residual. Union actualcost reported, not asserted65.',
        grid='Deadline f13 per scene/replica/query: exists/local; intrusion/ordinary; support slope positive; detrended residual>cal-all-f13 median. Background-visible UNKNOWN globally. Grid is descriptive, not policy input.',
        timing='f3..13 timely; f14..15 late. Report advance in frames on both-timely pairs plus full rescue/loss separately.',
        interpretation='Dual grading itself leaves coverage equal oldOR|ordinarySingle; extra detection belongs to additional ordinary/temporal evidence. No static distance gate or validated silence evidence.',
        backend='TASK_NOT_GPU_SUITABLE: cached small scalar arrays and CSV counts; no tensor reconstruction',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs}, source_sha256=sha(Path(__file__)))
    save(OUT/'PLAN.json', plan)
    try:
        manifest = E.read_json(SOURCE/'eval_manifest.json')
        specs = {s['split']: s for s in manifest['datasets'] if s['branch'] == 'ideal'}
        # Load only ordinary seed scores: no other models are selected or rerun.
        data = {}
        for split, spec in specs.items():
            selected = dict(spec, scores=[j for j in spec['scores'] if j['arm'] == 'ordinary'])
            data[split] = E.load_dataset(selected, SOURCE, ['ordinary'], list(SEEDS))
        old_thresholds = E.read_json(SOURCE/'evaluation/thresholds.json')['arms']['ordinary']
        thresholds = {}
        for i, seed in enumerate(SEEDS):
            old = old_thresholds[str(seed)]
            raw = E.read_job(SOURCE/f'scores/ordinary_seed{seed}_cal.npz', 'ordinary', seed, 'cal','ideal')
            slope, residual = trend(raw)
            clear = (data['cal']['category'] == 'clear').all(1)
            record = dict(single=E.threshold(old['standalone']), addition=E.threshold(old['old_fusion_plus_candidate']),
                          residual_cutoff=float(np.median(residual[...,10,:])), temporal={})
            for kind in ('max3','k2of3'):
                temporal_score = temporal(raw, kind)
                theta, actual = E.F.nearest(temporal_score[clear].max(-1), old['standalone_actual_clear_slots'])
                record['temporal'][kind] = dict(theta=theta, target_slots=65, actual_slots=actual, residual_slots=actual-65)
            thresholds[str(seed)] = record
        save(OUT/'thresholds.json', thresholds)
        metrics, event_rows, grid_rows, summary_rows = {}, [], [], []
        for split, ds in data.items():
            category = ds['category']
            metrics[split] = {}
            all_grades, all_slope, all_residual = [], [], []
            for i, seed in enumerate(SEEDS):
                if time.monotonic()-began >= 600:
                    raise TimeoutError('Score-analysis600s command cap reached')
                th = thresholds[str(seed)]
                ordinary = ds['candidates'][0,i]
                raw = E.read_job(SOURCE/f'scores/ordinary_seed{seed}_{split}.npz', 'ordinary',seed,split,'ideal')
                slope, residual = trend(raw)
                references = dict(M3=ds['m3'] >= E.M3_THETA,
                    old_fusion=E.old_fusion(ds['m3'], ds['local']),
                    ordinary_single=ordinary >= th['single'],
                    ordinary_OR=E.old_fusion(ds['m3'],ds['local']) | (ordinary >= th['addition']))
                old_first = first(references['old_fusion'])
                or_first = first(references['ordinary_OR'])
                seed_metrics, seed_grades = {}, []
                for p, policy in enumerate(POLICIES):
                    kind = None if p == 0 else ('max3' if p == 1 else 'k2of3')
                    grade = grades(ds['m3'],ds['local'],ordinary,th['single'],th['addition'],
                        temporal(raw,kind) if kind else None, th['temporal'][kind]['theta'] if kind else None)
                    # Diagnostic invariants: all additions are light; dual == old|single.
                    if p == 0:
                        dual = grade
                        np.testing.assert_array_equal(grade > 0, references['old_fusion'] | references['ordinary_single'])
                    else:
                        np.testing.assert_array_equal(grade == 2, dual == 2)
                        assert np.all((grade > 0) | ~(dual > 0))
                    report = describe(grade, ds, references)
                    report['paired_any_vs_dual'] = paired_timing(dual > 0, grade > 0, category)
                    seed_metrics[policy] = report
                    seed_grades.append(grade)
                    summary_rows.append(dict(split=split,seed=seed,policy=policy,
                        HEAD_any=report['any']['counts'][0], BODY_any=report['any']['counts'][1],
                        HEAD_strong=report['strong']['counts'][0],BODY_strong=report['strong']['counts'][1],
                        HEAD_light_only=report['light_only_timely_contact'][0],BODY_light_only=report['light_only_timely_contact'][1],
                        clear_any_slots=report['any']['clear_slots'],clear_any_clips=report['any']['clear_clips'],
                        clear_strong_slots=report['joint_costs']['clear']['strong']['slots'],
                        clear_light_slots=report['joint_costs']['clear']['light']['slots'],
                        pass_any_clips=report['any']['pass_clips'],pass_strong_clips=report['joint_costs']['pass']['strong']['clips'],
                        pass_light_clips=report['joint_costs']['pass']['light']['clips']))
                    first_any, first_strong = first(grade > 0), first(grade == 2)
                    for n,row in enumerate(ds['rows']):
                        for k in range(grade.shape[1]):
                            for q,height in enumerate(E.HEIGHTS):
                                event_rows.append(dict(split=split,seed=seed,policy=policy,scene=int(ds['scene_ids'][n]),replica=k,height=height,
                                    category=category[n,q], family=row['shape_family'],rho=row['rho'],placement=row['placement'],
                                    background_family=row['background_family'],first_any_frame=int(first_any[n,k,q]),
                                    first_strong_frame=int(first_strong[n,k,q]),
                                    old_first_frame=int(old_first[n,k,q]),
                                    ordinary_OR_first_frame=int(or_first[n,k,q])))
                metrics[split][str(seed)] = seed_metrics
                all_grades.append(seed_grades); all_slope.append(slope); all_residual.append(residual)
                exists = ds['local'][...,10,:] >= E.OLD_LOCAL
                intrusion = ordinary[...,10,:] >= th['single']
                rising = slope[...,10,:] > 0
                volatile = residual[...,10,:] > th['residual_cutoff']
                cells = exists.astype(int)*8+intrusion.astype(int)*4+rising.astype(int)*2+volatile.astype(int)
                for q,height in enumerate(E.HEIGHTS):
                    for cell in range(16):
                        mask = cells[...,q] == cell
                        for cat in ('contact','pass','clear'):
                            selected = mask & (category[:,q] == cat)[:,None]
                            grid_rows.append(dict(split=split,seed=seed,height=height,exists=(cell>>3)&1,
                                intrusion=(cell>>2)&1,support_enhancing=(cell>>1)&1,high_detrended_residual=cell&1,
                                background_visible='UNKNOWN_NOT_FULL_COHORT',category=cat,count=int(selected.sum())))
                assert sum(r['count'] for r in grid_rows if r['split']==split and r['seed']==seed) == category.size*raw.shape[1]
            np.savez_compressed(OUT/f'{split}_grades.npz', grades=np.asarray(all_grades),
                slope=np.asarray(all_slope),residual=np.asarray(all_residual),scene_ids=ds['scene_ids'],category=category)
        write_csv(OUT/'summary.csv',summary_rows)
        write_csv(OUT/'event_ledger.csv',event_rows)
        write_csv(OUT/'deadline_grid.csv',grid_rows)
        save(OUT/'metrics.json',metrics)
        save(OUT/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,
            event_rows=len(event_rows),grid_rows=len(grid_rows),backend=plan['backend'],
            source_sha256=sha(Path(__file__)),no_inference=True,no_sampling=True,
            invariants='dual union exact; causal temporal; temporal strong identical; all seeds; grid denominators exact'))
        print(json.dumps({'status':'COMPLETE','seconds':time.monotonic()-began,'summary':summary_rows}))
    except BaseException as error:
        save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began))
        raise


def preserve_strong():
    """Declared follow-up: identical dual coverage, keep ordinary-OR strong."""
    began = time.monotonic()
    output = OUT/'strong_preserve'
    if (output/'PLAN.json').exists():
        raise FileExistsError('Keep successor attempt evidence')
    save(output/'PLAN.json', dict(task='SAME_COVERAGE_STRONG_PRESERVE',
        rationale='Initial dual downgraded contact events for small strong-cost savings; isolate grading tradeoff',
        strong='Frozen oldfusion OR ordinary>=fixedoldORaddition',
        light='Not strong AND ordinary>=fixedsingletheta',
        selection='Post-initial-outcome Explore successor, no fresh confirmation; no threshold adjustment',
        budget='Uses original score-analysis600s scope; no model/geometry changes',
        source_sha256=sha(Path(__file__)),parent_plan_sha256=sha(OUT/'PLAN.json')))
    manifest = E.read_json(SOURCE/'eval_manifest.json')
    thresholds = E.read_json(OUT/'thresholds.json')
    all_metrics, rows = {}, []
    for spec in manifest['datasets']:
        if spec['branch'] != 'ideal':
            continue
        split = spec['split']
        ds = E.load_dataset(dict(spec,scores=[j for j in spec['scores'] if j['arm']=='ordinary']),
                            SOURCE,['ordinary'],list(SEEDS))
        with np.load(OUT/f'{split}_grades.npz') as archive:
            old_grades = archive['grades'][:,0]
        all_metrics[split] = {}
        saved = []
        for i,seed in enumerate(SEEDS):
            th = thresholds[str(seed)]
            ordinary = ds['candidates'][0,i]
            references = dict(M3=ds['m3'] >= E.M3_THETA,old_fusion=E.old_fusion(ds['m3'],ds['local']),
                ordinary_single=ordinary>=th['single'],
                ordinary_OR=E.old_fusion(ds['m3'],ds['local']) | (ordinary>=th['addition']))
            strong = references['ordinary_OR']
            grade = np.where(strong,2,np.where(references['ordinary_single'],1,0)).astype(np.int8)
            np.testing.assert_array_equal(grade>0,old_grades[i]>0)
            np.testing.assert_array_equal(grade==2,strong)
            report = describe(grade,ds,references)
            report['paired_strong_vs_initial_dual'] = paired_timing(old_grades[i]==2,grade==2,ds['category'])
            all_metrics[split][str(seed)] = report
            saved.append(grade)
            rows.append(dict(split=split,seed=seed,HEAD_any=report['any']['counts'][0],BODY_any=report['any']['counts'][1],
                HEAD_strong=report['strong']['counts'][0],BODY_strong=report['strong']['counts'][1],
                HEAD_light_only=report['light_only_timely_contact'][0],BODY_light_only=report['light_only_timely_contact'][1],
                clear_any_slots=report['any']['clear_slots'],clear_any_clips=report['any']['clear_clips'],
                clear_strong_slots=report['joint_costs']['clear']['strong']['slots'],
                clear_light_slots=report['joint_costs']['clear']['light']['slots'],
                pass_any_clips=report['any']['pass_clips'],pass_strong_clips=report['joint_costs']['pass']['strong']['clips'],
                pass_light_clips=report['joint_costs']['pass']['light']['clips']))
        np.savez_compressed(output/f'{split}_grades.npz',grades=np.asarray(saved))
    write_csv(output/'summary.csv',rows)
    save(output/'metrics.json',all_metrics)
    save(output/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,
         invariant='Same total coverage as originaldual; strong exactly ordinaryOR',source_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-began,summary=rows)))


if __name__ == '__main__':
    if '--preserve-strong' in sys.argv:
        preserve_strong()
    else:
        run()

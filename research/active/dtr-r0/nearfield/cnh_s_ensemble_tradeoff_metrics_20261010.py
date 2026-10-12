"""Negative-pool clip-rate calibration and descriptive fresh-hold tradeoffs.

All score ties are enumerated. Selection reads negative clips only; fresh hold
curves and clustered intervals never select a model or operating point.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from numba import njit

import cnh_counterfactual_common_dev as C
import cnh_graded_evidence_dev as G
import cnh_frozen_e2e_metrics_20261010 as M
import cnh_cost_v2_metrics_20261010 as V
import cnh_task_cost_metrics_20261010 as T

ARM_KEYS = ('E', 'S955', 'S956', 'S957', 'Sensemble')
STAT_NAMES = (*T.STAT_NAMES, 'far_pass_plus_clear_notified_clips',
              'HEAD_dark_thin_net', 'BODY_dark_thin_net',
              'HEAD_sign_edge_net', 'BODY_sign_edge_net')
CLIP_INCREMENT_CAP = .03
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261010


def load_archive(path):
    with np.load(path, allow_pickle=False) as saved:
        result = {k:saved[k].copy() for k in saved.files}
    if tuple(str(k) for k in result['arm_keys']) != ARM_KEYS:
        raise ValueError('Unexpected fixed arm order')
    if result['old5'].shape != result['light_scores'].shape[1:]:
        raise ValueError('Score and original strong arrays do not align')
    if np.isnan(result['light_scores']).any() or np.isposinf(result['light_scores']).any():
        raise ValueError('Nonfinite positive/NaN evidence')
    if not np.isin(result['old5'], (0, 2)).all():
        raise ValueError('Original old5 strong grades must be 0/2')
    return result


@njit(cache=False)
def _clip_stats(grade, negative, near, contact, base_timely, dark, sign):
    answer = np.zeros(13, np.int64)
    answer[:8] = T._clip_stats(grade, negative, near, contact, base_timely)
    if negative and not near:
        answer[8] = int(np.any(grade > 0))
    for q in range(2):
        net = answer[3+3*q]-answer[4+3*q]
        answer[9+q] = net*int(dark)
        answer[11+q] = net*int(sign)
    return answer


@njit(cache=False)
def _sweep(grade, clips, frames, queries, bounds, negative, near, contact,
           base_timely, dark, sign):
    n = grade.shape[0]
    local = np.zeros((n, 13), np.int64)
    total = np.zeros(13, np.int64)
    for c in range(n):
        local[c] = _clip_stats(grade[c], negative[c], near[c], contact[c],
                              base_timely[c], dark[c], sign[c])
        total += local[c]
    answer = np.zeros((len(bounds)+1, 13), np.int64)
    answer[0] = total
    marked = np.full(n, -1, np.int64)
    for tie in range(len(bounds)-1):
        start, stop = bounds[tie], bounds[tie+1]
        for j in range(start, stop):
            grade[clips[j], frames[j], queries[j]] = 0
        for j in range(start, stop):
            c = clips[j]
            if marked[c] == tie:
                continue
            marked[c] = tie
            fresh = _clip_stats(grade[c], negative[c], near[c], contact[c],
                                base_timely[c], dark[c], sign[c])
            total += fresh-local[c]
            local[c] = fresh
        answer[tie+1] = total
    answer[-1] = total
    return answer


def sweep(old5, score, category, rows, check, negatives_only=False, original_both=None):
    """Exact ascending canonical ties, including both infinite endpoints.

    The both-control curve starts at zero and retains its original extra grade
    strengths until removed. Its zero endpoint reproduces original both955.
    S/E curves preserve only original old5 strong grades and append light.
    """
    check()
    masks, near, _, _ = V.negative_masks(category, rows)
    select = masks['all_negative'] if negatives_only else np.ones(len(rows), bool)
    strong = np.asarray(old5[select], np.int8)
    evidence = np.asarray(score[select], np.float64)
    if original_both is None:
        grade = T.apply_gate(strong, evidence, -np.inf)
        eligible = np.isfinite(evidence) & (strong != 2)
        first_tau = -np.inf
    else:
        grade = np.asarray(original_both[select], np.int8).copy()
        if not np.all(grade[strong == 2] == 2):
            raise ValueError('Both control must retain old5 strong slots')
        eligible = (grade > 0) & (strong != 2) & np.isfinite(evidence)
        if (evidence[eligible] < 0).any() or ((grade > 0) & (strong != 2) & ~np.isfinite(evidence)).any():
            raise ValueError('Both extras require finite nonnegative frozen margin')
        first_tau = 0.
    grade = grade.reshape(-1, 13, 2)
    c, f, q = np.where(eligible.reshape(-1, 13, 2))
    values = evidence.reshape(-1, 13, 2)[c,f,q]
    order = np.argsort(values, kind='stable')
    values, c, f, q = [x[order] for x in (values,c,f,q)]
    bounds = np.r_[0,np.flatnonzero(np.diff(values))+1,len(values)] if len(values) else np.array([0])
    taus = np.r_[first_tau,np.nextafter(values[bounds[:-1]],np.inf),np.inf] if len(values) else np.array([first_tau,np.inf])
    k = old5.shape[1]
    repeat = lambda v:np.repeat(np.asarray(v)[select],k,axis=0)
    first = V.first_notice(M.replay_gap1(strong)).reshape(-1,2)
    stats = _sweep(grade,c,f,q,bounds,repeat(masks['all_negative']),repeat(near),
        repeat(category == 'contact'), (first >= 0)&(first < 11),
        repeat([bool(r['dark_thin']) for r in rows]),
        repeat([r['shape_family'] == 'sign_edge' for r in rows]))
    np.testing.assert_array_equal(grade,strong.reshape(-1,13,2))
    assert len(taus) == len(stats)
    check()
    return taus,stats,len(bounds)-1


def _write_curve(writer, arm, taus, stats, denominator, base_clips, cap=None):
    for tau, count in zip(taus,stats):
        row = dict(arm=arm,**T.threshold_record(tau),
            **{key:int(v) for key,v in zip(STAT_NAMES,count)},
            weighted_cost=float(count[0]/4),HEAD_net=int(count[3]-count[4]),
            BODY_net=int(count[6]-count[7]),far_clear_clip_denominator=denominator,
            far_clear_clip_rate=float(count[8]/denominator),
            far_clear_increment_pp=float(100*(count[8]-base_clips)/denominator))
        if cap is not None:
            row['feasible_clip_rate_cap'] = bool(count[8] <= cap)
        writer.writerow(row)


def _curve_writer(handle, cal):
    fields = ['arm','tau','tau_kind',*STAT_NAMES,'weighted_cost','HEAD_net','BODY_net',
              'far_clear_clip_denominator','far_clear_clip_rate','far_clear_increment_pp']
    if cal:
        fields += ['feasible_clip_rate_cap']
    writer = csv.DictWriter(handle,fieldnames=fields)
    writer.writeheader()
    return writer


def family_rates(notice, base_notice, category, rows):
    masks,_,_,_ = V.negative_masks(category,rows)
    family = np.array([r['background_family'] for r in rows])
    notified = (notice > 0).any((2,3))
    base_notified = (base_notice > 0).any((2,3))
    result = []
    for key in sorted(set(family)):
        mask = (family == key)&masks['far_pass_plus_clear']
        den = int(mask.sum()*notice.shape[1])
        hit,base = int(notified[mask].sum()),int(base_notified[mask].sum())
        result.append(dict(family=str(key),physical_scenes=int(mask.sum()),clip_denominator=den,
            notified_clips=hit,old5_notified_clips=base,clip_rate=hit/den,
            old5_clip_rate=base/den,increment_pp=100*(hit-base)/den))
    return result


def calibrate(out, pool_rows, check):
    out = Path(out)
    began = time.monotonic()
    if (out/'data/hold/task_scores.npz').exists():
        raise ValueError('Fresh hold cannot precede pooled threshold sealing')
    scores = load_archive(out/'pool_scores.npz')
    category,old5 = scores['category'],scores['old5']
    masks,_,_,_ = V.negative_masks(category,pool_rows)
    if not masks['all_negative'].all() or (category == 'contact').any():
        raise ValueError('Pooled chooser may contain pass/clear only')
    baseline = M.replay_gap1(old5)
    den = int(masks['far_pass_plus_clear'].sum()*old5.shape[1])
    base_clips = int((baseline[masks['far_pass_plus_clear']] > 0).any((2,3)).sum())
    cap = base_clips+CLIP_INCREMENT_CAP*den
    records,selected,notices = {},[],[]
    with (out/'pool_calibration_curve.csv').open('x',newline='',encoding='utf-8') as handle:
        writer = _curve_writer(handle,True)
        for ai,arm in enumerate(ARM_KEYS):
            taus,stats,ties = sweep(old5,scores['light_scores'][ai],category,pool_rows,check,True)
            chosen = int(np.flatnonzero(stats[:,8] <= cap)[0])
            grade = T.apply_gate(old5,scores['light_scores'][ai],taus[chosen])
            notice = M.replay_gap1(grade)
            rates = family_rates(notice,baseline,category,pool_rows)
            increments = np.array([r['increment_pp'] for r in rates])
            records[arm] = dict(**T.threshold_record(taus[chosen]),candidate_index=chosen,
                candidate_thresholds=len(taus),negative_margin_ties=ties,
                far_clear_clip_denominator=den,old5_notified_clips=base_clips,
                notified_clips=int(stats[chosen,8]),old5_clip_rate=base_clips/den,
                selected_clip_rate=float(stats[chosen,8]/den),
                selected_increment_pp=float(100*(stats[chosen,8]-base_clips)/den),
                cap_increment_pp=3.,maximum_notified_clips=cap,contact_utility_access=False,
                families=rates,family_increment_pp=dict(median=float(np.median(increments)),
                    p90=float(np.quantile(increments,.9)),maximum=float(increments.max())))
            selected.append(grade);notices.append(notice)
            _write_curve(writer,arm,taus,stats,den,base_clips,cap)
    np.savez_compressed(out/'pool_grades_notifications.npz',keys=np.array(ARM_KEYS),
        grades=np.array(selected),notifications=np.array(notices),category=category,
        baseline_notifications=baseline)
    counts = []
    for family in sorted({r['background_family'] for r in pool_rows}):
        for dataset,split in sorted({(r.get('source_dataset',''),r.get('source_split','')) for r in pool_rows if r['background_family']==family}):
            pick = np.array([r['background_family']==family and r.get('source_dataset','')==dataset and r.get('source_split','')==split for r in pool_rows])
            counts.append(dict(family=family,source_dataset=dataset,source_split=split,
                negative_scenes=int(pick.sum()),negative_clips=int(pick.sum()*old5.shape[1]),
                far_clear_scenes=int((pick&masks['far_pass_plus_clear']).sum()),
                far_clear_clips=int((pick&masks['far_pass_plus_clear']).sum()*old5.shape[1])))
    G.write_csv(out/'pool_family_counts.csv',counts)
    files = ['pool_scores.npz','pool_calibration_curve.csv','pool_grades_notifications.npz','pool_family_counts.csv']
    for optional in ('pool_rows.json','pool_provenance.json','pool_input_receipt.json','protocol_snapshot.json','PLAN.json','frozen_spec.json','execution_manifest.json'):
        if (out/optional).exists():
            files.append(optional)
    T.save_new(out/'sealed_calibration.json',dict(calibrations=records,
        sealed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=C.sha(Path(__file__)),
        bindings={p:C.sha(out/p) for p in files},
        selection='Lowest canonical threshold across -inf, nextafter every finite eligible negative tie, +inf; exact gap1 joint clip-any rate <= same-pool old5 +0.03; all ties, no monotonic shortcut, no contact utility',
        pool_negative_scenes=len(pool_rows),pool_negative_clips=len(pool_rows)*old5.shape[1],
        pool_family_counts=counts,primary_arm='Sensemble',hold_selection=False,
        fixed_strong='Old5 grade2 every slot; all S/E additions light',
        hold_access='Seal completed before generating fresh hold geometry or rendering'))
    return dict(duration_seconds=time.monotonic()-began,sealed_calibration_sha256=C.sha(out/'sealed_calibration.json'),calibrations=records)


def clustered_bootstrap(notice, baseline, category, rows, replicates=BOOTSTRAP_REPLICATES):
    """Paired physical-scene, family, and two-level percentile descriptions."""
    first,base_first = V.first_notice(notice),V.first_notice(baseline)
    hit,base_hit = (first>=0)&(first<11),(base_first>=0)&(base_first<11)
    masks,near,_,_ = V.negative_masks(category,rows)
    joint,basejoint = notice.max(-1),baseline.max(-1)
    fields = []
    vectors = []
    for q,height in enumerate(V.HEIGHTS):
        truth = category[:,q] == 'contact'
        vectors += [((hit[:,:,q].astype(int)-base_hit[:,:,q])*truth[:,None]).sum(1),truth.astype(int)*notice.shape[1]]
        fields += [height+'_net',height+'_denominator']
    fc = masks['far_pass_plus_clear']
    vectors += [((joint>0).any(-1).astype(int)-(basejoint>0).any(-1))*fc[:,None],fc.astype(int)*notice.shape[1]]
    vectors[-2] = vectors[-2].sum(1)
    fields += ['far_clear_notified_clip_increment','far_clear_clip_denominator']
    weights = np.where(near[:,None],.25,1.)
    vectors += [(((joint==1).sum(2)*weights+(joint==2).sum(2))*masks['all_negative'][:,None]).sum(1),
                (((basejoint==1).sum(2)*weights+(basejoint==2).sum(2))*masks['all_negative'][:,None]).sum(1)]
    fields += ['weighted_cost','old5_weighted_cost']
    values = np.column_stack(vectors).astype(float)
    families = np.array([r['background_family'] for r in rows])
    family_indices = [np.flatnonzero(families==f) for f in sorted(set(families))]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {key:np.empty((replicates,8)) for key in ('physical_scene','family','hierarchical_family_scene')}
    for b in range(replicates):
        samples['physical_scene'][b] = values[rng.integers(len(rows),size=len(rows))].sum(0)
        chosen = rng.integers(len(family_indices),size=len(family_indices))
        clusters = [family_indices[j] for j in chosen]
        samples['family'][b] = sum((values[idx].sum(0) for idx in clusters),start=np.zeros(8))
        samples['hierarchical_family_scene'][b] = sum((values[idx[rng.integers(len(idx),size=len(idx))]].sum(0) for idx in clusters),start=np.zeros(8))
    result = {}
    for kind,totals in samples.items():
        metric = dict(HEAD_net=totals[:,0],BODY_net=totals[:,2],HEAD_BODY_net=totals[:,0]+totals[:,2],
            HEAD_net_pp=100*totals[:,0]/totals[:,1],BODY_net_pp=100*totals[:,2]/totals[:,3],
            HEAD_BODY_net_pp=100*(totals[:,0]+totals[:,2])/(totals[:,1]+totals[:,3]),
            far_clear_increment_pp=100*totals[:,4]/totals[:,5],weighted_cost=totals[:,6],
            weighted_cost_increment=totals[:,6]-totals[:,7])
        result[kind] = {key:dict(lower=float(np.quantile(v,.025)),upper=float(np.quantile(v,.975))) for key,v in metric.items()}
    return dict(replicates=replicates,seed=BOOTSTRAP_SEED,confidence_level=.95,
        method='Paired percentile; K replicas kept within physical scene; family-only whole clusters, hierarchical families then scenes; descriptive with only four hold families',
        family_count=len(family_indices),intervals=result,scene_vector_fields=fields,
        scene_vectors=values.tolist())


def _plot_curves(out, curves, selected, contacts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for arm,(taus,stats,den,base) in curves.items():
        x = 100*(stats[:,8]-base)/den
        # Plot every distinct descriptive outcome pair; CSV retains every tie.
        for panel,group in enumerate(('all','dark_thin','sign_edge')):
            for q,height in enumerate(V.HEIGHTS):
                net = stats[:,3+3*q]-stats[:,4+3*q] if group=='all' else stats[:,9+q] if group=='dark_thin' else stats[:,11+q]
                denom = contacts[height if group=='all' else height+'_'+group]['denominator']
                pairs = np.unique(np.column_stack((x,100*net/denom)),axis=0)
                axes[panel].plot(pairs[:,0],pairs[:,1],label=arm+' '+height,alpha=.8,linewidth=1.4,linestyle='-' if q==0 else '--')
                if arm in selected:
                    point = selected[arm]
                    cname = height if group=='all' else height+'_'+group
                    axes[panel].scatter(point['far_clear_increment_pp'],100*point['contacts'][cname]['net']/denom,s=22)
    for ax,title in zip(axes,('All contact','Dark-thin contact','Sign-edge contact')):
        ax.scatter([0],[0],s=35,c='black',label='old5')
        ax.axvline(3,color='grey',alpha=.4)
        ax.set(xlabel='Far+clear notified-clip increment (pp)',ylabel='Timely net gain (pp)',title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(out/'tradeoff_curves.png',dpi=180)
    fig.savefig(out/'tradeoff_curves.svg')
    plt.close(fig)
    rates = selected['Sensemble']['families']
    fig,ax = plt.subplots(figsize=(9,4),layout='constrained')
    ax.bar([r['family'] for r in rates],[r['increment_pp'] for r in rates])
    ax.axhline(3,color='grey',linestyle='--',label='Pool cap +3 pp')
    ax.set(ylabel='Far+clear notified-clip increment (pp)',title='Selected S ensemble: fresh hold family variation')
    ax.tick_params(axis='x',labelrotation=15)
    ax.legend()
    fig.savefig(out/'family_variation.png',dpi=180)
    plt.close(fig)


def evaluate(out, hold_rows, check):
    out = Path(out)
    began = time.monotonic()
    binding = C.read(out/'sealed_calibration.json')
    seal = C.sha(out/'sealed_calibration.json')
    for path,digest in binding['bindings'].items():
        if C.sha(out/path) != digest:
            raise ValueError('Pool calibration binding changed: '+path)
    scores = load_archive(out/'data/hold/task_scores.npz')
    category,old5 = scores['category'],scores['old5']
    baseline = M.replay_gap1(old5)
    first_base = V.first_notice(baseline)
    both = scores['both'][0] if scores['both'].ndim==5 else scores['both']
    first_both = V.first_notice(M.replay_gap1(both))
    masks = V.contact_masks(category,hold_rows)
    negative_masks,near,layer,_ = V.negative_masks(category,hold_rows)
    den = int(negative_masks['far_pass_plus_clear'].sum()*old5.shape[1])
    base_clips = int((baseline[negative_masks['far_pass_plus_clear']] > 0).any((2,3)).sum())
    arms = [('fixed/m3',scores['m3']),('fixed/old5',old5),('both955',both)]
    for ai,arm in enumerate(ARM_KEYS):
        arms.append((arm+'/selected',T.apply_gate(old5,scores['light_scores'][ai],T.record_threshold(binding['calibrations'][arm]))))
    reports,event_rows,notification_rows,keys,grades,notices = {},[],[],[],[],[]
    family_rows = []
    selected = {}
    for key,grade in arms:
        check()
        if '/selected' in key:
            np.testing.assert_array_equal(grade==2,old5==2)
        notice = M.replay_gap1(grade)
        first = V.first_notice(notice)
        rates = family_rates(notice,baseline,category,hold_rows)
        report = dict(contacts=V.contact_summary(first,first_base,masks),
            contacts_vs_both955=V.contact_summary(first,first_both,masks),
            costs=V.cost_summary(notice,category,hold_rows),families=rates,
            far_clear_increment_pp=100*(int((notice[negative_masks['far_pass_plus_clear']]>0).any((2,3)).sum())-base_clips)/den)
        reports['hold/'+key] = report
        family_rows.extend([dict(arm=key,**r) for r in rates])
        if key.endswith('/selected'):
            selected[key.split('/')[0]] = report
        keys.append(key);grades.append(grade);notices.append(notice)
        for n,row in enumerate(hold_rows):
            for k in range(grade.shape[1]):
                metadata = V._metadata(row,'hold',n,k)
                for q,height in enumerate(V.HEIGHTS):
                    index,base,oldboth = int(first[n,k,q]),int(first_base[n,k,q]),int(first_both[n,k,q])
                    timely,basetime,bothtime = 0<=index<11,0<=base<11,0<=oldboth<11
                    contact = category[n,q]=='contact'
                    event_rows.append(dict(**metadata,arm=key,height=height,category=str(category[n,q]),first_index=index,
                        first_nominal_frame=index+3 if index>=0 else -1,outcome='timely' if timely else 'late' if index>=11 else 'silent',
                        old5_first_index=base,both955_first_index=oldboth,rescue=int(contact and timely and not basetime),
                        loss=int(contact and basetime and not timely),rescue_vs_both955=int(contact and timely and not bothtime),
                        loss_vs_both955=int(contact and bothtime and not timely),light_notifications=int((notice[n,k,:,q]==1).sum()),
                        strong_notifications=int((notice[n,k,:,q]==2).sum())))
                joint = notice[n,k].max(-1)
                for f in np.flatnonzero(joint):
                    union = int(joint[f]);negative = bool(negative_masks['all_negative'][n])
                    def weight(w):
                        return (w if near[n] and union==1 else 1.) if negative else 0.
                    notification_rows.append(dict(**metadata,arm=key,frame_index=int(f),nominal_frame=int(f)+3,
                        union_grade=union,HEAD_grade=int(notice[n,k,f,0]),BODY_grade=int(notice[n,k,f,1]),
                        HEAD_truth=str(category[n,0]),BODY_truth=str(category[n,1]),negative_cost=int(negative),
                        partition='near_pass' if near[n] else 'far_pass' if negative_masks['far_pass'][n] else 'clear' if negative_masks['clear'][n] else 'contact',
                        evaluated_pass_layer=str(layer[n]),weight_w0=weight(0.),weight_w025=weight(.25),weight_w05=weight(.5)))
    np.savez_compressed(out/'hold_grades_notifications.npz',keys=np.array(keys),grades=np.array(grades),
        notifications=np.array(notices),category=category,scene_ids=np.arange(len(hold_rows)))
    G.write_csv(out/'event_ledger.csv',event_rows)
    G.write_csv(out/'notification_ledger.csv',notification_rows)
    G.write_csv(out/'hold_family_rates.csv',family_rows)
    curves = {}
    with (out/'hold_tradeoff_curve.csv').open('x',newline='',encoding='utf-8') as handle:
        writer = _curve_writer(handle,False)
        for arm in ('Sensemble','E','both955'):
            ai = ARM_KEYS.index(arm) if arm!='both955' else -1
            evidence = scores['light_scores'][ai] if ai>=0 else scores['margin'][0]
            taus,stats,_ = sweep(old5,evidence,category,hold_rows,check,False,both if arm=='both955' else None)
            curves[arm] = (taus,stats,den,base_clips)
            _write_curve(writer,arm,taus,stats,den,base_clips)
    bootstrap = {}
    for key,notice in zip(keys,notices):
        check()
        if key!='fixed/old5':
            bootstrap[key] = clustered_bootstrap(notice,baseline,category,hold_rows)
    T.save_new(out/'bootstrap_intervals.json',bootstrap)
    _plot_curves(out,curves,selected,reports['hold/fixed/old5']['contacts'])
    T.save_new(out/'metrics.json',dict(reports=reports,calibrations=binding['calibrations'],
        contract=dict(main_weight=.25,cap_increment_pp=3.,hold_selection=False,binary_pass_fail_policy=False,
            fixed_strong='Old5 grade2 every slot for S/E; originalboth control retains original grades',
            both_curve='Originalboth955 frozen positive extras removed at each margin tie; zero reproduces originalboth, infinity old5',
            cost='gap1 per query, same-frame union by max; near light .25, far/clear all1',
            contact='f3..13 timely, f14..15 late, silent otherwise',independent_unit='Physical scene and family; K/frame/seed not independent',
            hold_curve='Descriptive only, every finite tie and endpoints; no hold choice',protected_access=0),
        bootstrap_path='bootstrap_intervals.json'))
    if C.sha(out/'sealed_calibration.json') != seal:
        raise AssertionError('Pooled calibration seal mutated')
    outputs = ['metrics.json','event_ledger.csv','notification_ledger.csv','hold_grades_notifications.npz',
               'hold_family_rates.csv','hold_tradeoff_curve.csv','bootstrap_intervals.json','tradeoff_curves.png','tradeoff_curves.svg','family_variation.png']
    return dict(duration_seconds=time.monotonic()-began,cells=len(reports),events=len(event_rows),
        joined_notifications=len(notification_rows),sealed_calibration_sha256=seal,
        source_sha256=C.sha(Path(__file__)),outputs_sha256={p:C.sha(out/p) for p in outputs})

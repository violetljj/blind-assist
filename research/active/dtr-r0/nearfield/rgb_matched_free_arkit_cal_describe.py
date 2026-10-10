"""Describe cached R0 scores at exact attainable sampledFREE counts; never deploy cuts."""
from __future__ import annotations
import argparse
import bisect
import csv
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ARMS = ('dav_raw', 'uni_raw')
FREE = 'FREE_ON_SAMPLED_RAYS'
OLD = ('arkit16', 'arkit_40777060', 'arkit_40777065', 'new_arkit_41069021',
       'new_arkit_41069042', 'new_arkit_41069048')
FRESH = ('confirm_arkit_41125718', 'confirm_arkit_41125756', 'confirm_arkit_41142278',
         'confirm_arkit_41159503', 'confirm_arkit_41159519', 'confirm_arkit_41159529')
BANDS = ('near', 'midfar')
REQUESTED = (0, 7, 5, 10, 15, 20)
DAV_ORIGINAL_CUT = 0.24403834342956543
FROZEN_DEN = {'old6/near':(225,604,35),'old6/midfar':(1033,237,458),
              'fresh6/near':(292,515,57),'fresh6/midfar':(1047,107,574)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csvwrite(path, rows):
    if not rows: return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def band(row):
    return 'near' if row['distance_band']=='0.3-0.8m' else 'midfar'


def finite(values):
    arr=np.asarray(values, dtype=np.float64)
    return np.sort(arr[np.isfinite(arr)])


def supported(arr, cut):
    return len(arr)-int(np.searchsorted(arr,cut,side='left'))


def sweep(rows):
    """Every finite score tie plus a strictly empty endpoint; -inf never supported."""
    qq=finite([r['query_score'] for r in rows])
    pp=finite([r['known_positive_score'] for r in rows if r['reference_state']=='POSITIVE'])
    events=np.unique(np.concatenate((qq,pp)))[::-1]
    free=finite([r['query_score'] for r in rows if r['reference_state']==FREE])
    witness=finite([r['known_positive_score'] for r in rows if r['reference_state']=='POSITIVE'])
    positive=finite([r['query_score'] for r in rows if r['reference_state']=='POSITIVE'])
    unknown=finite([r['query_score'] for r in rows if r['reference_state']=='UNKNOWN'])
    den={s:sum(r['reference_state']==s for r in rows) for s in ('POSITIVE',FREE,'UNKNOWN')}
    # +inf is only an empty-support sentinel, never a supported-score event.
    cuts=[(None,'EMPTY')]+[(float(x),'FULL_FINITE' if i==len(events)-1 else 'TIE') for i,x in enumerate(events)]
    if not len(events):cuts.append((None,'FULL_FINITE'))
    result=[]
    for cut,endpoint in cuts:
        actual=math.inf if cut is None else cut
        result.append(dict(cutoff_m=cut,endpoint=endpoint,FREE=supported(free,actual),W=supported(witness,actual),
            POS_support=supported(positive,actual),UNKNOWN=supported(unknown,actual),POS_den=den['POSITIVE'],
            FREE_den=den[FREE],UNKNOWN_den=den['UNKNOWN'],finite_query_scores=len(qq),finite_positive_scores=len(witness)))
    exact={}
    for row in result:
        f=row['FREE']
        # Last tie at F gives maximum W, with its entire tie supported.
        if f not in exact or row['W']>=exact[f]['W']: exact[f]=dict(row)
    envelope=[dict(exact[f],actual_FREE=f) for f in sorted(exact)]
    budgets=[]
    best=None
    for budget in range(den[FREE]+1):
        if budget in exact and (best is None or exact[budget]['W']>best['W']): best=exact[budget]
        assert best is not None
        budgets.append(dict(budget_FREE=budget,actual_FREE=best['FREE'],W=best['W'],cutoff_m=best['cutoff_m'],
                            status='EXACT_BUDGET' if best['FREE']==budget else 'AT_MOST_NOT_EXACT',FREE_den=den[FREE]))
    return result,envelope,budgets


def compare(envelopes):
    by={a:{r['actual_FREE']:r for r in envelopes[a]} for a in ARMS}
    common=sorted(set(by[ARMS[0]])&set(by[ARMS[1]]))
    comparisons=[dict(actual_FREE=f,DAV_W=by['dav_raw'][f]['W'],Uni_W=by['uni_raw'][f]['W'],
        Uni_minus_DAV_W=by['uni_raw'][f]['W']-by['dav_raw'][f]['W'],
        DAV_cutoff_m=by['dav_raw'][f]['cutoff_m'],Uni_cutoff_m=by['uni_raw'][f]['cutoff_m'],
        FREE_den=by['dav_raw'][f]['FREE_den'],POS_den=by['dav_raw'][f]['POS_den']) for f in common]
    runs=[]
    for row in comparisons:
        sign='UNI_LOWER' if row['Uni_minus_DAV_W']<0 else 'UNI_HIGHER' if row['Uni_minus_DAV_W']>0 else 'EQUAL'
        if not runs or runs[-1]['ordering']!=sign:
            runs.append(dict(ordering=sign,first_attainable_FREE=row['actual_FREE'],last_attainable_FREE=row['actual_FREE'],
                             joint_attainable_FREE_counts=[row['actual_FREE']]))
        else:
            runs[-1]['last_attainable_FREE']=row['actual_FREE'];runs[-1]['joint_attainable_FREE_counts'].append(row['actual_FREE'])
    return comparisons,dict(joint_attainable_points=len(common),Uni_weakly_dominates_all_exact_points=all(r['Uni_minus_DAV_W']>=0 for r in comparisons),
        UNI_LOWER_points=[r for r in comparisons if r['Uni_minus_DAV_W']<0],ordering_intervals=runs,
        interval_semantics='Runs over listed discrete joint attainable counts only; gaps are not interpolated')


def requested(envelopes,budgets,requests):
    by={a:{r['actual_FREE']:r for r in envelopes[a]} for a in ARMS}
    results=[]
    for request in requests:
        exactboth=all(request in by[a] for a in ARMS)
        for a in ARMS:
            own=by[a].get(request)
            le=budgets[a][min(request,len(budgets[a])-1)]
            results.append(dict(requested_FREE=request,model=a,status='EXACT' if own else 'NOT_EXACT',
                exact_W=None if own is None else own['W'],exact_actual_FREE=None if own is None else request,
                joint_exact=exactboth,at_most_W=le['W'],at_most_actual_FREE=le['actual_FREE'],
                FREE_den=le['FREE_den'],over_denominator=request>le['FREE_den']))
    return results


def paired_at_cuts(group,cuts):
    def supports(value,cut):return math.isfinite(value) and cut is not None and value>=cut
    pairs=dict(W_rescue=0,W_loss=0,FREE_added=0,FREE_removed=0,DAV_W=0,Uni_W=0,DAV_FREE=0,Uni_FREE=0)
    for d,u in zip(group['dav_raw'],group['uni_raw']):
        state=d['reference_state'];key='known_positive_score' if state=='POSITIVE' else 'query_score'
        dd=supports(d[key],cuts['dav_raw']);uu=supports(u[key],cuts['uni_raw'])
        if state=='POSITIVE':
            pairs['DAV_W']+=dd;pairs['Uni_W']+=uu;pairs['W_rescue']+=uu and not dd;pairs['W_loss']+=dd and not uu
        if state==FREE:
            pairs['DAV_FREE']+=dd;pairs['Uni_FREE']+=uu;pairs['FREE_added']+=uu and not dd;pairs['FREE_removed']+=dd and not uu
    return pairs


def compare_budgets(budgets,maxbudget):
    comparisons=[];intervals=[]
    for f in range(maxbudget+1):
        d,u=budgets['dav_raw'][f],budgets['uni_raw'][f];delta=u['W']-d['W']
        row=dict(budget_FREE=f,DAV_actual_FREE=d['actual_FREE'],Uni_actual_FREE=u['actual_FREE'],
                 DAV_W=d['W'],Uni_W=u['W'],Uni_minus_DAV_W=delta)
        comparisons.append(row);sign='UNI_LOWER' if delta<0 else 'UNI_HIGHER' if delta>0 else 'EQUAL'
        if not intervals or intervals[-1]['ordering']!=sign:intervals.append(dict(ordering=sign,first_budget_FREE=f,last_budget_FREE=f))
        else:intervals[-1]['last_budget_FREE']=f
    return comparisons,dict(common_finite_max_budget=maxbudget,ordering_intervals=intervals,
                            interpretation='AT_MOST supplemental budget curve; actual FREE counts may differ; not primary gate')


def figures(groups,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def panel(ax,g):
        for arm,color,label in [('dav_raw','#3264a8','DAV raw'),('uni_raw','#d97721','Uni raw')]:
            e=g['envelopes'][arm]
            ax.plot([r['actual_FREE'] for r in e],[r['W'] for r in e],'.',markersize=2.2,color=color,label=label)
        ax.set_title(g['name'],fontsize=9);ax.set_xlabel('Actual sampledFREE supported');ax.set_ylabel('Positive witness W')
        ax.grid(alpha=.2)
    fig,axes=plt.subplots(6,4,figsize=(16,22),layout='constrained')
    for ax,g in zip(axes.flat,[groups[f'{c}/{b}'] for c in OLD+FRESH for b in BANDS]):panel(ax,g)
    axes.flat[0].legend();fig.suptitle('All attainable whole-tie points; cached Development, no interpolation',fontsize=14)
    for suffix in ('png','pdf'):fig.savefig(out/f'capture_24panels.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,key in zip(axes.flat,('old6/near','old6/midfar','fresh6/near','fresh6/midfar')):panel(ax,groups[key])
    axes.flat[0].legend();fig.suptitle('Four prespecified aggregate groups: exact actual-FREE witness envelopes')
    for suffix in ('png','pdf'):fig.savefig(out/f'aggregate_4panels.{suffix}',dpi=180)
    plt.close(fig)


def main(args):
    start=time.perf_counter();out=args.output.resolve();runroot=args.runroot.resolve();source=args.source_run.resolve()
    if out.exists():raise FileExistsError('Preserve earlier results: choose a new output path')
    # Read PLAN and seal before touching any cached score file.
    plan=json.loads((runroot/'PLAN.json').read_text(encoding='utf-8-sig'))
    seal=json.loads((runroot/'plan_seal.json').read_text(encoding='utf-8-sig'))
    plansha=sha(runroot/'PLAN.json')
    assert plansha in json.dumps(seal),'PLAN SHA must be in seal receipt'
    assert 'MATCHED' in plan['run'].upper()
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'executed_describe.py')
    receipt=dict(status='STARTING',started_utc=datetime.now(timezone.utc).isoformat(),PLAN_sha256=plansha,source_sha256=sha(__file__),
                 GPU=0,downloads=0,training=0,cutoffs_deployed=0,source_score_inputs=[])
    write(out/'execution.json',receipt)
    try:
        data={}
        for c in OLD+FRESH:
            path=source/'final'/f'{c}_scores.json'
            receipt['source_score_inputs'].append(dict(path=str(path),sha256=sha(path)))
            arms=json.loads(path.read_text(encoding='utf-8'))['arms']
            assert set(arms)==set(ARMS)
            ids=[]
            for a in ARMS:
                for row in arms[a]:
                    for key in ('query_score','known_positive_score'):row[key]=float(row[key])
                assert all(r['cohort']==c for r in arms[a])
                assert all(r['distance_band'] in ('0.3-0.8m','0.8-1.5m','1.5-3m') for r in arms[a])
                assert all(not math.isfinite(r['known_positive_score']) or math.isfinite(r['query_score']) and
                           r['query_score']>=r['known_positive_score'] for r in arms[a]),'Finite witness implies query support'
                ids.append([(r['frame'],r['query'],r['reference_state'],r['distance_band']) for r in arms[a]])
            assert ids[0]==ids[1],'No model-dependent denominator deletion'
            data[c]=arms
        groups={}
        for c in OLD+FRESH:
            for b in BANDS:groups[f'{c}/{b}']={a:[r for r in data[c][a] if band(r)==b] for a in ARMS}
        for cohortlist,label in ((OLD,'old6'),(FRESH,'fresh6')):
            for b in BANDS:groups[f'{label}/{b}']={a:[r for c in cohortlist for r in data[c][a] if band(r)==b] for a in ARMS}
        ties=[];env=[];budgetrows=[];common=[];req=[];described={};requestpairs=[];scorepairs=[];budgetcomparisons=[];baselinepoints=[]
        for c in OLD+FRESH:
            for d,u in zip(data[c]['dav_raw'],data[c]['uni_raw']):
                scorepairs.append(dict(cohort=c,frame=d['frame'],query=d['query'],distance_band=d['distance_band'],
                    reference_state=d['reference_state'],DAV_query_score=d['query_score'],Uni_query_score=u['query_score'],
                    DAV_known_positive_score=d['known_positive_score'],Uni_known_positive_score=u['known_positive_score']))
        for name,group in groups.items():
            if time.perf_counter()-start>args.budget_s:raise TimeoutError('CPU command-wall cap')
            ee={};bb={}
            if name in FROZEN_DEN:
                for a in ARMS:
                    observed=tuple(sum(r['reference_state']==s for r in group[a]) for s in ('POSITIVE',FREE,'UNKNOWN'))
                    assert observed==FROZEN_DEN[name],(name,a,observed,FROZEN_DEN[name])
            for a in ARMS:
                tt,ee[a],bb[a]=sweep(group[a])
                for rows,target in ((tt,ties),(ee[a],env)):
                    target.extend(dict(group=name,model=a,**row) for row in rows)
            common_max=min(max(r['actual_FREE'] for r in ee[a]) for a in ARMS)
            for a in ARMS:
                budgetrows.extend(dict(group=name,model=a,**row) for row in bb[a][:common_max+1])
            bc,bd=compare_budgets(bb,common_max);budgetcomparisons.extend(dict(group=name,**r) for r in bc)
            cc,decision=compare(ee);common.extend(dict(group=name,**r) for r in cc)
            requests=list(args.requested_free)
            if name in FROZEN_DEN:
                original=paired_at_cuts(group,dict(dav_raw=DAV_ORIGINAL_CUT,uni_raw=None))
                f=original['DAV_FREE'];exact={a:next((r for r in ee[a] if r['actual_FREE']==f),None) for a in ARMS}
                if f not in requests:requests.append(f)
                baselinepoints.append(dict(group=name,DAV_original_cutoff_m=DAV_ORIGINAL_CUT,actual_FREE=f,DAV_original_W=original['DAV_W'],
                    DAV_upper_envelope_W=exact['dav_raw']['W'],Uni_upper_envelope_W=None if exact['uni_raw'] is None else exact['uni_raw']['W'],
                    status='EXACT' if exact['uni_raw'] is not None else 'NOT_EXACT',interpretation='Original DAV point versus descriptive maximum at its actual FREE; no operational selection'))
            req.extend(dict(group=name,**r) for r in requested(ee,bb,requests))
            described[name]=dict(name=name,envelopes=ee,comparison=decision,supplemental_budget_comparison=bd)
            if name.startswith(('old6/','fresh6/')):
                by={a:{r['actual_FREE']:r for r in ee[a]} for a in ARMS}
                for f in requests:
                    exact=all(f in by[a] for a in ARMS)
                    for mode in ('EXACT','AT_MOST_SUPPLEMENT'):
                        selected={a:by[a][f] for a in ARMS} if exact and mode=='EXACT' else {
                            a:bb[a][min(f,len(bb[a])-1)] for a in ARMS}
                        counts=paired_at_cuts(group,{a:r['cutoff_m'] for a,r in selected.items()}) if mode!='EXACT' or exact else {}
                        requestpairs.append(dict(group=name,requested_FREE=f,mode=mode,status='EXACT' if mode=='EXACT' and exact else
                            'NOT_EXACT' if mode=='EXACT' else 'AT_MOST_ACTUAL_FREE_SHOWN',
                            DAV_actual_FREE=selected['dav_raw'].get('actual_FREE') if counts else None,
                            Uni_actual_FREE=selected['uni_raw'].get('actual_FREE') if counts else None,
                            **{k:counts.get(k) for k in ('W_rescue','W_loss','FREE_added','FREE_removed','DAV_W','Uni_W','DAV_FREE','Uni_FREE')}))
        for filename,rows in [('all_ties.csv',ties),('envelope.csv',env),('at_most_budget_envelope.csv',budgetrows),
                              ('common_actual_FREE.csv',common),('requested_FREE.csv',req),('query_score_pairs.csv',scorepairs),
                              ('requested_FREE_pairs.csv',requestpairs),('supplemental_budget_comparison.csv',budgetcomparisons)]:csvwrite(out/filename,rows)
        csvwrite(out/'original_DAV_points.csv',baselinepoints)
        primary={k:described[k]['comparison'] for k in ('old6/near','old6/midfar','fresh6/near','fresh6/midfar')}
        crossing=any(not v['Uni_weakly_dominates_all_exact_points'] for v in primary.values())
        summary=dict(status='COMPLETE',scope='Consumed Development descriptive analysis, not selected deployment/calibration working points',
            primary_groups=primary,any_primary_crossing=crossing,stage2_allowed=not crossing,
            groups={k:v['comparison'] for k,v in described.items()},requested_FREE=args.requested_free,
            supplemental_budget_groups={k:v['supplemental_budget_comparison'] for k,v in described.items()},
            original_DAV_points=baselinepoints,frozen_denominators={k:list(v) for k,v in FROZEN_DEN.items()},
            endpoint='Strictly empty plus all finite query/known-POS score ties; -inf never supported',
            model_invalid='Cached denominators unchanged; invalid score never supported',interpolation=0,random_tie_splitting=0,
            scientific_figures='Markers are exact attainable points; no lines interpolate unattainable costs')
        write(out/'summary.json',summary);write(out/'curves.json',described)
        figures(described,out)
        receipt.update(status='COMPLETE',groups=len(groups),score_rows=sum(len(x[a]) for x in data.values() for a in ARMS),
                       tie_rows=len(ties),exact_envelope_rows=len(env),joint_exact_rows=len(common),stage2_allowed=not crossing)
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL',error=repr(exc));raise
    finally:
        receipt.update(command_wall_s=time.perf_counter()-start,completed_utc=datetime.now(timezone.utc).isoformat())
        write(out/'terminal.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('source-run','runroot','output'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--requested-free',type=int,nargs='+',default=list(REQUESTED))
    p.add_argument('--budget-s',type=float,default=350)
    main(p.parse_args())

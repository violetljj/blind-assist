"""Frozen near-band Uni/DAV hybrid; consumed ARKit FREE-only calibration."""
from __future__ import annotations
import argparse,json,math,shutil,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import rgb_backbone_confirm_evaluate as base
from rgb_body_query_query_calibration_probe import FREE,select_cutoff,write
from rgb_body_query_residual_readout import evaluate,paired,summary
from rgb_body_query_scene_diagnostic import write_csv
from rgb_matched_free_arkit_cal_describe import OLD,FRESH,sweep

START=time.perf_counter()
DAV=.24403834342956543
UNI_POOLED=.09616100788116455
NEAR='0.3-0.8m'
ARMS=('DAV_ONLY','HYBRID_ARK','HYBRID_POOLED')


def decode(rows):
    for r in rows:
        for key in ('query_score','known_positive_score'):
            if key in r:r[key]=float(r[key])
    return rows


def calibrate(args,inputs,out):
    retained={};frames=set();inputsources=[];duplicates=[];counts=Counter();excluded=0
    def consume(rows,source):
        nonlocal excluded
        frameids={(r['scan'],r['frame']) for r in rows if r['scan'].startswith('arkitscenes_')}
        frames.update(frameids)
        for row in rows:
            if not row['scan'].startswith('arkitscenes_'):
                if row['reference_state']==FREE:excluded+=1
                continue
            if row['reference_state']!=FREE:continue
            assert row['distance_band'] in base.BANDS
            key=(row['scan'],row['frame'],row['query']);value=float(row['query_score'])
            if math.isnan(value) or value==math.inf:raise ValueError('Invalid calibration score')
            clean=dict(scan=row['scan'],frame=row['frame'],query=row['query'],distance_band=row['distance_band'],
                reference_state=FREE,query_score=value,source_cohort=source)
            if key in retained:
                assert retained[key]['query_score']==value and retained[key]['distance_band']==clean['distance_band']
                duplicates.append(dict(scan=key[0],frame=key[1],query=key[2],duplicate_source=source));continue
            retained[key]=clean;counts[source]+=1
        inputsources.append(dict(source=source,frames=len(frameids),FREE_before_dedup=sum(r['reference_state']==FREE and
                                  r['scan'].startswith('arkitscenes_') for r in rows)))
    for c in OLD+FRESH:
        data=inputs.json(args.consumed/'final'/f'{c}_scores.json')['arms']
        consume(data['uni_raw'],c)
    pool=inputs.json(args.pooled/'calibration'/'pooled_FREE_scores.json')['arms']['uni_raw/R0']
    assert all(r['reference_state']==FREE for r in pool)
    consume(pool,'original48ARK_cal')
    rows=list(retained.values());choice=select_cutoff([r['query_score'] for r in rows])
    assert choice['negative_queries']==len(rows)
    write(out/'deduplicated_FREE_scores.json',rows);write(out/'duplicate_query_receipts.json',duplicates)
    frozen=dict(status='SEALED_FREE_ONLY_BEFORE_FRESH_MODEL_SCORE',frozen_utc=base.utc(),PLAN_sha256=base.sha(args.runroot/'PLAN.json'),
        evaluator_source_sha256=base.sha(__file__),source_inputs=inputs.rows,source_rows=inputsources,
        calibration_frames_after_dedup=len(frames),FREE_queries_after_dedup=len(rows),FREE_contributions=dict(counts),
        duplicate_queries_removed=len(duplicates),excluded_3RScan_FREE=excluded,Uni_ARK=choice,DAV_cutoff=DAV,Uni_POOLED_cutoff=UNI_POOLED,
        all_bands_used=list(base.BANDS),POS_UNKNOWN_selection=0,fresh_eval_used=0,training=0)
    write(out/'thresholds.json',frozen)
    write(out/'threshold_seal.json',dict(frozen_utc=frozen['frozen_utc'],thresholds_sha256=base.sha(out/'thresholds.json'),
        FREE_scores_sha256=base.sha(out/'deduplicated_FREE_scores.json'),PLAN_sha256=frozen['PLAN_sha256']))
    print('CAL SEALED',frozen['calibration_frames_after_dedup'],len(rows),choice,flush=True)


def final(args,inputs,out):
    frozen=inputs.json(args.calibration/'thresholds.json');seal=inputs.json(args.calibration/'threshold_seal.json')
    assert base.sha(args.calibration/'thresholds.json')==seal['thresholds_sha256']
    assert frozen['PLAN_sha256']==base.sha(args.runroot/'PLAN.json') and frozen['fresh_eval_used']==0
    uc=frozen['Uni_ARK']['cutoff'];assert uc is not None
    roster=inputs.json(args.runroot/'new_public_roster.json');order=roster['accepted_cohorts_order']
    assert len(order)==len(set(order)) and set(order)=={r['cohort'] for r in roster['rows']}
    mapping=inputs.json(args.runroot/'new_prediction_manifests.json')
    sources={a:base.prediction_index(mapping[m],inputs) for a,m in [('dav_raw','dav2'),('uni_raw','unidepth')]}
    cohorts={};decisions={};allpairs=[];alltable=[];raw={};midfarchecks=0
    for c in order:
        manifest=inputs.json(args.runroot/'new-eval-sensor'/c/'dataset_manifest.json')
        public={base.key(r):r for r in roster['rows'] if r['cohort']==c}
        assert len(public)==len(manifest['rows'])==16 and set(public)=={base.key(r) for r in manifest['rows']}
        for ref in manifest['rows']:
            for key in ('rgb_sha256','depth_K','depth_shape'):assert ref[key]==public[base.key(ref)][key]
        data=base.score(c,manifest,sources,inputs,False,True);raw[c]=data
        write(out/f'{c}_scores.json',dict(arms=data))
        dav=evaluate(data['dav_raw'],dict(cutoff=DAV),False);ev={'DAV_ONLY':dav}
        for arm,cut in [('HYBRID_ARK',uc),('HYBRID_POOLED',UNI_POOLED)]:
            uni=evaluate(data['uni_raw'],dict(cutoff=cut),False)
            ev[arm]=[u if d['distance_band']==NEAR else dict(d) for d,u in zip(dav,uni,strict=True)]
            for d,r in zip(dav,ev[arm],strict=True):
                if d['distance_band']!=NEAR:
                    assert d==r,'Mid/far must be every-field identical DAV decision';midfarchecks+=1
            allpairs+=paired(ev[arm],dav,c,arm,'DAV_ONLY','FROZEN_BAND_HYBRID')
        decisions[c]=ev;cohorts[c]={a:summary(rows) for a,rows in ev.items()}
        write(out/f'{c}_evaluation.json',ev)
        for a,groups in cohorts[c].items():
            for g,r in groups.items():alltable.append(dict(cohort=c,arm=a,group=g,**r))
    write_csv(out/'query_pairs.csv',allpairs);write_csv(out/'summary.csv',alltable)
    write(out/'results.json',dict(cohorts=cohorts,thresholds=frozen,capture_order=order,pairs=summary(allpairs,True)))
    aggregates={};nearrows=[]
    for a in ARMS:
        rr=[r for c in order for r in decisions[c][a]];aggregates[a]=summary(rr)
    gains=[]
    for c in order:
        dd=cohorts[c]['DAV_ONLY']['band/'+NEAR];aa=cohorts[c]['HYBRID_ARK']['band/'+NEAR]
        if aa['positive_known_witness']>dd['positive_known_witness']:gains.append(c)
        nearrows.append(dict(cohort=c,arms={a:cohorts[c][a]['band/'+NEAR] for a in ARMS}))
    near=aggregates['HYBRID_ARK']['band/'+NEAR];nf=near['free_total'];fp=near['free_support'];complete=len(order)==6
    gate=dict(complete_six_fresh_captures=complete,FREE_denominator_positive=nf>0,near_strictly_improved_captures=gains,
        improved_count=len(gains),required_improved_count=4,absolute_near_FREE_support=fp,near_FREE_denominator=nf,
        allowed_FREE_floor_2pct=nf//50,absolute_rate_condition_50F_le_N=50*fp<=nf if nf else None,
        definition='Absolute hybridARK near FREE rate, not net increase')
    passed=complete and nf>0 and len(gains)>=4 and 50*fp<=nf
    result=dict(status='COMPLETE' if complete else 'PARTIAL',criterion='PASS' if passed else 'NOT_EVALUABLE' if not complete or not nf else 'FAIL',
        adoption_gate=gate,recommend_hybrid=passed,capture_order=order,near_capture_rows=nearrows,aggregates=aggregates,
        midfar_identical_decision_checks=midfarchecks,paired_summary=summary(allpairs,True),
        frozen_cutoffs=dict(DAV_ONLY=DAV,HYBRID_ARK_near=uc,HYBRID_POOLED_near=UNI_POOLED),
        data_role='Fresh capture Development, near reference coverage enriched; no safety claim',
        midfar_preservation='Structural reuse of DAV rows; no independent mid/far improvement evidence')
    write(out/'summary.json',result)
    curves(raw,decisions,order,out,uc)


def curves(raw,decisions,order,out,uc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];env=[];markers=[];groups={}
    for c in order+['aggregate_fresh']:
        group={a:[r for cap in (order if c=='aggregate_fresh' else [c]) for r in raw[cap][a] if r['distance_band']==NEAR]
               for a in ('dav_raw','uni_raw')};groups[c]={}
        for a in group:
            tt,ee,_=sweep(group[a]);groups[c][a]=ee
            rows.extend(dict(cohort=c,model=a,**r) for r in tt);env.extend(dict(cohort=c,model=a,**r) for r in ee)
        for a,cut,model in [('DAV_ONLY',DAV,'dav_raw'),('HYBRID_ARK',uc,'uni_raw'),('HYBRID_POOLED',UNI_POOLED,'uni_raw')]:
            rr=[r for cap in (order if c=='aggregate_fresh' else [c]) for r in decisions[cap][a] if r['distance_band']==NEAR]
            measured=summary(rr)['all'];markers.append(dict(cohort=c,arm=a,raw_model=model,cutoff_m=cut,
                actual_FREE=measured['free_support'],W=measured['positive_known_witness'],FREE_den=measured['free_total'],POS_den=measured['positive_total'],
                semantics='Actual frozen operational point; never substitute witness upper envelope'))
    write_csv(out/'near_all_ties.csv',rows);write_csv(out/'near_envelope.csv',env);write_csv(out/'near_operational_points.csv',markers)
    write(out/'near_curves.json',dict(groups=groups,operational_points=markers,operational_cutoff_selection=0))
    fig,axes=plt.subplots(4,2,figsize=(12,16),layout='constrained')
    for ax,c in zip(axes.flat,order+['aggregate_fresh']):
        for a,color,label in [('dav_raw','#3264a8','DAV raw envelope'),('uni_raw','#d97721','Uni raw envelope')]:
            e=groups[c][a];ax.plot([r['actual_FREE'] for r in e],[r['W'] for r in e],'.',color=color,markersize=3,label=label)
        for r,shape,color in zip([r for r in markers if r['cohort']==c],['s','*','X'],['#1e3a5f','#9f1239','#6a4792']):
            ax.scatter(r['actual_FREE'],r['W'],marker=shape,s=75,color=color,label=r['arm']+' frozen point',zorder=5)
        ax.set_title(c);ax.set_xlabel('Actual sampledFREE supported');ax.set_ylabel('Positive witness W');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7);axes.flat[-1].set_visible(False)
    fig.suptitle('Near-band descriptive envelopes and separately measured frozen working points')
    for suffix in ('png','pdf'):fig.savefig(out/f'near_capture_and_aggregate.{suffix}',dpi=180)
    plt.close(fig)


def main(args):
    for field in ('repo','runroot','consumed','pooled','output'):setattr(args,field,getattr(args,field).resolve())
    if args.calibration:args.calibration=args.calibration.resolve()
    if args.output.exists():raise FileExistsError('Preserve completed and failed stages')
    args.output.mkdir(parents=True);inputs=base.Inputs(args.budget_s)
    plan=inputs.json(args.runroot/'PLAN.json');assert plan['run']=='RGB_BAND_HYBRID_DEV_20261010'
    seal=inputs.json(args.runroot/'plan_seal.json');assert base.sha(args.runroot/'PLAN.json') in json.dumps(seal)
    shutil.copyfile(__file__,args.output/'executed_evaluate.py')
    receipt=dict(stage=args.stage,status='STARTING',started_utc=base.utc(),source_sha256=base.sha(__file__),GPU=0,downloads=0,training=0)
    try:
        if args.stage=='calibrate':calibrate(args,inputs,args.output)
        else:final(args,inputs,args.output)
        receipt['status']='COMPLETE'
    except Exception as e:receipt.update(status='FAILED_PARTIAL',error=repr(e));raise
    finally:
        receipt.update(command_wall_s=time.perf_counter()-START,completed_utc=base.utc(),checks=dict(inputs.checks))
        write(args.output/'inputs.json',inputs.rows);write(args.output/'terminal.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('repo','runroot','consumed','pooled','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--calibration',type=Path);p.add_argument('--stage',choices=('calibrate','final'),required=True)
    p.add_argument('--budget-s',type=float,default=350)
    main(p.parse_args())

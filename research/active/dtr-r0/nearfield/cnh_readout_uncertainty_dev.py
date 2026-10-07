"""Two distinct unit-bootstrap uncertainties, fixed readouts, no fitting."""
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
import cnh_group_calibration_dev as G
import cnh_model_stability_ranking_dev as K

D,ROOT,HERE,WORK=G.D,G.ROOT,G.HERE,G.WORK
SOURCE=G.OUT
OUT=WORK/'cnh-readout-uncertainty-dev-20261007'
DRAWS=1000
SEEDS=dict(ranking_evaluation=2026100742,policy_calibration=2026100743,policy_evaluation=2026100744)
MODELS={'M3':(0,'original_center'),**{f'fit{f}':(f,'raw_hb_aug') for f in range(3)}}


def stratified_weights(unit,mode,mirror,draws,seed):
    units,inverse=np.unique(unit,return_inverse=True);strata={}
    for j,u in enumerate(units):
        values=set(zip(np.asarray(mode)[unit==u].tolist(),np.asarray(mirror)[unit==u].tolist()))
        if len(values)!=1:raise ValueError('A unit must have one mode/mirror stratum')
        strata[j]=next(iter(values))
    rng=np.random.default_rng(seed);weights=np.zeros((draws,len(units)),np.int32)
    for group in sorted(set(strata.values())):
        ids=np.array([j for j in range(len(units)) if strata[j]==group])
        for b,draw in enumerate(rng.integers(0,len(ids),(draws,len(ids)))):weights[b,ids]=np.bincount(draw,minlength=len(ids))
    return units,inverse,weights


def score_groups(values,unit_index,nunits):
    values=np.asarray(values,float);unit_index=np.asarray(unit_index,int)
    if values.ndim!=1 or unit_index.shape!=values.shape or np.isnan(values).any():raise ValueError('Aligned non-NaN scores and unit indices required')
    unique,index=np.unique(values,return_inverse=True);counts=np.zeros((nunits,len(unique)),float)
    np.add.at(counts,(unit_index,index),1.)
    return dict(values=unique,counts=counts,mass=counts.sum(1))


def ranking_groups(positive,positive_units,negative,negative_units,nunits):
    values,inverse=np.unique(np.concatenate((positive,negative)),return_inverse=True)
    pc=np.zeros((nunits,len(values)),float);nc=pc.copy();n=len(positive)
    np.add.at(pc,(positive_units,inverse[:n]),1.);np.add.at(nc,(negative_units,inverse[n:]),1.)
    return dict(values=values,positive=pc,negative=nc)


def weighted_step_area(groups,weights,cap=.05):
    """Exact whole-tie step envelope, not linear ROC interpolation."""
    weights=np.asarray(weights,float);pc=weights@groups['positive'][:,::-1];nc=weights@groups['negative'][:,::-1]
    pm=pc.sum(1);nm=nc.sum(1);valid=(pm>0)&(nm>0)
    tp=np.divide(np.cumsum(pc,axis=1),pm[:,None],out=np.zeros_like(pc),where=pm[:,None]>0)
    fp=np.divide(np.cumsum(nc,axis=1),nm[:,None],out=np.zeros_like(nc),where=nm[:,None]>0)
    clipped=np.minimum(fp,cap);width=np.diff(np.concatenate((np.zeros((len(weights),1)),clipped),axis=1),axis=1)
    previous=np.concatenate((np.zeros((len(weights),1)),tp[:,:-1]),axis=1)
    area=(width*previous).sum(1)
    if tp.shape[1]:area+=(cap-clipped[:,-1])*tp[:,-1]
    result=area/cap;result[~valid]=np.nan
    return result,pm,nm


def weighted_cutoff(groups,weights,cap):
    """Same kth descending observation and nextafter as expanded threshold_at_cap."""
    counts=np.asarray(weights,float)@groups['counts'][:,::-1];mass=counts.sum(1)
    thresholds=np.full(len(weights),np.nan);valid=mass>0
    if counts.shape[1] and valid.any():
        k=np.floor(cap*mass[valid]);index=np.argmax(np.cumsum(counts[valid],axis=1)>k[:,None],axis=1)
        thresholds[valid]=np.nextafter(groups['values'][::-1][index],np.inf)
    return thresholds,mass


def weighted_above(groups,weights,thresholds):
    """Per-draw unit-weighted >= count without expanding or resorting observations."""
    suffix=np.cumsum(groups['counts'][:,::-1],axis=1)[:,::-1]
    suffix=np.concatenate((suffix,np.zeros((suffix.shape[0],1))),axis=1)
    indices=np.searchsorted(groups['values'],thresholds,side='left')
    counts=(np.asarray(weights,float)*suffix[:,indices].T).sum(1)
    counts[~np.isfinite(thresholds)]=np.nan
    return counts


def event_and_control(scores,contact,control,deadline,mask,inverse,nunits,ranking=False):
    selected=contact&mask;positive_rows=np.flatnonzero(selected&((deadline>=2) if ranking else True))
    positive=np.array([scores[i,2:int(deadline[i])+1].max() if deadline[i]>=2 else -np.inf for i in positive_rows])
    negative=scores[control&mask,2:12].ravel();negative_units=np.repeat(inverse[control&mask],10)
    if not np.isfinite(negative).all() or not np.isfinite(positive[deadline[positive_rows]>=2]).all():raise ValueError('Nonfinite used score')
    if ranking:return ranking_groups(positive,inverse[positive_rows],negative,negative_units,nunits)
    return score_groups(positive,inverse[positive_rows],nunits),score_groups(negative,negative_units,nunits)


def ci(values):
    values=np.asarray(values);good=np.isfinite(values)
    return dict(ci95=np.quantile(values[good],[.025,.975]).tolist() if good.any() else None,
        valid_draws=int(good.sum()),missing_draws=int((~good).sum()))


def distribution(values):
    values=np.asarray(values);good=np.isfinite(values)
    return dict(point=float(values[0]) if np.isfinite(values[0]) else None,
        bootstrap_quantiles=np.quantile(values[1:][good[1:]],[0,.025,.5,.975,1]).tolist() if good[1:].any() else None,
        missing_draws=int((~good[1:]).sum()))


def freeze():
    paths=[Path(__file__),Path(K.__file__),Path(G.__file__),Path(G.C.__file__),
        SOURCE/'PLAN.json',SOURCE/'result.json',SOURCE/'thresholds.json',SOURCE/'ledger.npz',SOURCE/'calibration_ledger.npz',SOURCE/'analysis_terminal.json',
        HERE/'cnh_probability_query_dev.py',HERE/'cnh_direction_information_dev.py']
    assert D.read(SOURCE/'analysis_terminal.json')['status']=='COMPLETE'
    OUT.mkdir(parents=True,exist_ok=True)
    D.save(OUT/'PLAN.json',dict(phase='POSTHOC uncertainty audit; fixed consumed evaluation and fixed models',
        authorization='User requested two distinct confidence intervals, then prioritize direction estimation; no ensemble exploration',
        budget_cpu_wall_seconds=600,no_gpu=True,threads=4,draws=DRAWS,seeds=SEEDS,models=MODELS,
        A='Fixed model score ranking: paired evaluation-unit mode/mirror stratified1000draws, shared across models/tags; difference of0..5%whole-tie reachable step area divided by.05. No calibration or training uncertainty.',
        B='Calibration48units and evaluation48units independently mode/mirror stratified1000draws. One calibration draw synchronizes original/HB and allmodels. Recompute per-model max(original,HB) whole-tie>= thresholds at2.5/5%, then weighted evaluation no-warmup timely and actualFA. No model refit.',
        computation='Sorted fixed score groups with per-unit counts; integer bootstrap multiplicities, exact nextafter order-statistic threshold. Focused tests compare expanded observations and point matches parents.',
        missingness='Zero calibration or evaluation denominators are explicitly counted and excluded only from the affected CI, never replaced with zero risk. Early event deadlines remain failures for B but are unevaluable ranking events for A.',
        intervals='Percentile2.5/97.5 paired cluster intervals. A includes only evaluation sampling. B additionally includes empirical calibration sampling, never training uncertainty. No model selection correction, no fresh confirmation.',
        interpretation='A interval touching/crossing zero gives insufficient confirmation of positive ranking gain. B timely andFA differences must be read jointly; neither CI is a deployment guarantee.',
        hashes={str(p.relative_to(ROOT)):D.sha(p) for p in paths}))
    print('FROZEN two distinct unit-bootstrap CI analyses,CPU600s',flush=True)


def run():
    import cnh_direction_information_dev as I
    from threadpoolctl import threadpool_limits
    start=time.monotonic();plan=D.read(OUT/'PLAN.json')
    if (OUT/'RUN_STARTED.json').exists():raise FileExistsError('Single run; preserve outputs')
    D.save(OUT/'RUN_STARTED.json',dict(started_unix=time.time(),budget_seconds=600))
    def check():
        if time.monotonic()-start>plan['budget_cpu_wall_seconds']:raise TimeoutError('600s CPU budget reached')
    try:
        for path,expected in plan['hashes'].items():assert D.sha(ROOT/path)==expected,path
        wanted=('unit','mode','mirror','tag','valid','evaluable','contact','control','deadline')
        with np.load(SOURCE/'ledger.npz') as z:
            ev={k:z[k] for k in wanted};evscores={name:z[f'score/fold{f}/{m}'] for name,(f,m) in MODELS.items()}
            for fold in (1,2):np.testing.assert_array_equal(z[f'score/fold{fold}/original_center'],evscores['M3'])
        with np.load(SOURCE/'calibration_ledger.npz') as z:
            cal={k:z[k] for k in ('unit','mode','mirror','tag','valid','eligible_control')};calscores={name:z[f'score/fold{f}/{m}'] for name,(f,m) in MODELS.items()}
            for fold in (1,2):np.testing.assert_array_equal(z[f'score/fold{fold}/original_center'],calscores['M3'])
        parent=D.read(SOURCE/'result.json');threshold_parent={(r['fold'],r['cap'],r['method']):r['threshold'] for r in parent['thresholds']}
        eu,ei,aw=stratified_weights(ev['unit'],ev['mode'],ev['mirror'],DRAWS,SEEDS['ranking_evaluation'])
        eu2,ei2,bew=stratified_weights(ev['unit'],ev['mode'],ev['mirror'],DRAWS,SEEDS['policy_evaluation'])
        cu,ci_unit,bcw=stratified_weights(cal['unit'],cal['mode'],cal['mirror'],DRAWS,SEEDS['policy_calibration'])
        np.testing.assert_array_equal(eu,eu2);np.testing.assert_array_equal(ei,ei2);assert len(eu)==len(cu)==48 and not set(eu)&set(cu)
        Aweights=np.vstack((np.ones(len(eu)),aw));Eweights=np.vstack((np.ones(len(eu)),bew));Cweights=np.vstack((np.ones(len(cu)),bcw))
        arrays={'A/evaluation_unit_weights':aw,'B/evaluation_unit_weights':bew,'B/calibration_unit_weights':bcw,'evaluation_units':eu,'calibration_units':cu}
        areas={};A={};B={};threshold_summary={};curves={};evaluation_groups={};thresholds={}
        event=ev['contact']&ev['valid']&ev['evaluable'];control=ev['control']&ev['valid']&ev['evaluable']
        with threadpool_limits(limits=4):
            for tag in ('original','HB'):
                mask=ev['tag']==tag
                for name,score in evscores.items():
                    check();rg=event_and_control(score,event,control,ev['deadline'],mask,ei,len(eu),ranking=True)
                    area,pm,nm=weighted_step_area(rg,Aweights)
                    reference=K.ranking_metrics(score,event,control,ev['deadline'],mask=mask&ev['evaluable'],cap=.05)
                    np.testing.assert_allclose(area[0],reference['partial_step_area_normalized'],rtol=0,atol=1e-12)
                    areas[tag,name]=area;arrays[f'A/area/{tag}/{name}']=area
                    evaluation_groups[tag,name]=event_and_control(score,event,control,ev['deadline'],mask,ei,len(eu))
                for fold in range(3):
                    name=f'fit{fold}';delta=areas[tag,name]-areas[tag,'M3'];interval=ci(delta[1:]);bounds=interval['ci95']
                    crosses=None if bounds is None else bool(bounds[0]<=0<=bounds[1])
                    A[f'{tag}/{name}']=dict(candidate_area=float(areas[tag,name][0]),m3_area=float(areas[tag,'M3'][0]),point_difference=float(delta[0]),
                        **interval,interval_touches_or_crosses_zero=crosses,ranking_events=int(pm[0]),control_intervals=int(nm[0]),
                        zero_evaluation_event_draws=int((pm[1:]==0).sum()),zero_evaluation_control_draws=int((nm[1:]==0).sum()),
                        interpretation='Insufficient confirmation of positive ranking gain' if bounds is None or bounds[0]<=0 else 'Positive conditional evaluation-sampling interval; not fresh confirmation and excludes calibration/training uncertainty')
            for name,score in calscores.items():
                check();per_tag={}
                for tag in ('original','HB'):
                    mask=cal['eligible_control']&cal['valid']&(cal['tag']==tag)
                    values=score[mask,2:12].ravel();assert np.isfinite(values).all()
                    table=score_groups(values,np.repeat(ci_unit[mask],10),len(cu));per_tag[tag]=table
                for cap in (.025,.05):
                    text=f'{cap:.3f}';individual={};mass={}
                    for tag,table in per_tag.items():individual[tag],mass[tag]=weighted_cutoff(table,Cweights,cap)
                    theta=np.maximum(individual['original'],individual['HB']);thresholds[name,text]=theta
                    f,m=MODELS[name];assert theta[0]==threshold_parent[f,text,m]
                    threshold_summary[f'{text}/{name}']=dict(groups={tag:dict(threshold=distribution(individual[tag]),control_intervals=distribution(mass[tag]),zero_control_draws=int((mass[tag][1:]==0).sum())) for tag in per_tag},group_threshold=distribution(theta))
                    for tag in per_tag:arrays[f'B/threshold/{text}/{name}/{tag}']=individual[tag]
                    arrays[f'B/threshold/{text}/{name}/group']=theta
                    for tag in ('original','HB'):
                        pos,neg=evaluation_groups[tag,name];tp=weighted_above(pos,Eweights,theta);fp=weighted_above(neg,Eweights,theta)
                        ed=Eweights@pos['mass'];nd=Eweights@neg['mass']
                        ref=parent['metrics'][f'fold{f}/group_calibrated/{text}/{m}'][tag]['all']
                        assert tp[0]==ref['timely_excluding_warmup'] and fp[0]==ref['false_alarm_intervals']
                        assert ed[0]==ref['events'] and nd[0]==ref['control_intervals']
                        curves[tag,name,text]=(tp,fp,ed,nd)
                        for label,v in [('timely_count',tp),('false_alarm_count',fp),('events',ed),('control_intervals',nd)]:arrays[f'B/{label}/{text}/{tag}/{name}']=v
            for cap in ('0.025','0.050'):
                for tag in ('original','HB'):
                    bt,bf,bed,bnd=curves[tag,'M3',cap]
                    for fold in range(3):
                        name=f'fit{fold}';tp,fp,ed,nd=curves[tag,name,cap]
                        np.testing.assert_array_equal(ed,bed);np.testing.assert_array_equal(nd,bnd)
                        valid_cal=np.isfinite(thresholds[name,cap])&np.isfinite(thresholds['M3',cap])
                        event_diff=np.divide(100*(tp-bt),ed,out=np.full_like(tp,np.nan),where=(ed>0)&valid_cal)
                        fa_diff=np.divide(100*(fp-bf),nd,out=np.full_like(fp,np.nan),where=(nd>0)&valid_cal)
                        arrays[f'B/timely_gain_pp/{cap}/{tag}/{name}']=event_diff;arrays[f'B/false_alarm_delta_pp/{cap}/{tag}/{name}']=fa_diff
                        B[f'{cap}/{tag}/{name}']=dict(events=int(ed[0]),control_intervals=int(nd[0]),
                            candidate_timely=int(tp[0]),m3_timely=int(bt[0]),point_timely_gain=int(tp[0]-bt[0]),point_timely_gain_pp=float(event_diff[0]),timely_gain_pp=ci(event_diff[1:]),
                            candidate_false_alarm=float(fp[0]/nd[0]),m3_false_alarm=float(bf[0]/nd[0]),point_false_alarm_delta_pp=float(fa_diff[0]),false_alarm_delta_pp=ci(fa_diff[1:]),
                            missing_calibration_draws=int((~valid_cal[1:]).sum()),zero_evaluation_event_draws=int((ed[1:]==0).sum()),zero_evaluation_control_draws=int((nd[1:]==0).sum()))
        check();np.savez_compressed(OUT/'bootstrap_arrays.npz',**arrays)
        D.save(OUT/'result.json',dict(status='POSTHOC_UNCERTAINTY_COMPLETE',seconds=time.monotonic()-start,draws=DRAWS,seeds=SEEDS,
            A_fixed_score_ranking=dict(scope='Evaluation-unit sampling only; fixed scores; whole-tie normalized0..5%step area difference; excludes calibration and training uncertainty',results=A),
            B_recalibrated_policy=dict(scope='Independent calibration-unit and evaluation-unit sampling; eachdraw recomputes two-group scalar threshold; fixed models, no training uncertainty',results=B,threshold_summary=threshold_summary),
            point_reproduction='All ranking areas match parent helper; all point thresholds and event/FA counts match group-calibrated parent',
            array_row0='Original unresampled point; rows1..1000 are bootstrap draws',plan_sha256=D.sha(OUT/'PLAN.json'),arrays_sha256=D.sha(OUT/'bootstrap_arrays.npz')))
        D.save(OUT/'TERMINAL.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
        print('COMPLETE',round(time.monotonic()-start,2),'s',flush=True)
    except BaseException as exc:
        D.save(OUT/'TERMINAL.json',dict(status='STOPPED_PARTIAL' if isinstance(exc,TimeoutError) else 'FAILED',seconds=time.monotonic()-start,error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);globals()[parser.parse_args().stage]()

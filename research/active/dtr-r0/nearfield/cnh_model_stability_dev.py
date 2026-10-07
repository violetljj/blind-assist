"""Fixed three-model stability check on new unit observations, no fitting."""
import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
import cnh_double_height_dev as D

ROOT,HERE,WORK=D.ROOT,D.HERE,D.WORK
OUT=WORK/'cnh-model-stability-dev-20261007'
PARENT=WORK/'cnh-raw-training-support-dev-20261007'
SUPPORT=WORK/'cnh-training-support-dev-20261007'
METHODS=('original_center','raw_source_repeat','pair_hb_aug','raw_hb_aug')
TAGS=('original','HB')
BATCH_SIZE=8


def model_paths():
    return {(f,m):p for f in range(3) for m,p in {
        'raw_source_repeat':PARENT/'models'/f'fold{f}_raw_source_repeat.joblib',
        'pair_hb_aug':SUPPORT/'models'/f'fold{f}_hb_aug.joblib',
        'raw_hb_aug':PARENT/'models'/f'fold{f}_raw_hb_aug.joblib'}.items()}


def thresholds(result):
    ts={}
    for record in result['fold_records']:
        mode,cap,method=record['key'].split('/')
        if mode=='calibrated' and method in METHODS:
            key=record['fold'],cap,method
            if key in ts:raise ValueError('Duplicate inherited threshold')
            ts[key]=float(record['threshold'])
    assert set(ts)=={(f,c,m) for f in range(3) for c in ('0.025','0.050') for m in METHODS}
    return ts


def validate_rows(rows):
    units=sorted({r['unit'] for r in rows})
    assert len(units)==48 and len(rows)==3840
    assert [(r['unit'],r['config'],r['tag']) for r in rows]==[(u,c,t) for u in units for c in range(40) for t in TAGS]
    for r in rows:
        assert isinstance(r['valid'],bool) and isinstance(r['evaluable'],bool)
        if r['evaluable']:
            assert r['valid'] and bool(r['contact'])!=bool(r['control'])
            if r['contact']:assert r['deadline_index'] is not None and 0<=r['deadline_index']<13
    return units


def checked_manifest():
    manifest=D.read(OUT/'geometry_manifest.json');assert manifest['status']=='COMPLETE'
    validate_rows(manifest['rows'])
    return manifest


def freeze():
    from cnh_double_height_render_dev import source_paths
    manifest=checked_manifest();rows=manifest['rows'];units=sorted({r['unit'] for r in rows})
    parent=D.read(PARENT/'PLAN.json');old_units={u for f in parent['folds'] for role in f.values() for u in role}
    assert not set(units)&old_units
    hashes=dict(parent['hashes']);hashes.update(D.read(WORK/'cnh-double-height-dev-20261007/PLAN.json')['hashes'])
    paths=[Path(__file__),HERE/'cnh_model_stability_geometry_dev.py',HERE/'cnh_model_stability_ranking_dev.py',
        HERE/'cnh_model_stability_contrasts_dev.py',HERE/'cnh_raw_radial_features_dev.py',
        OUT/'geometry_manifest.json',OUT/'GEOMETRY_PLAN.json',OUT/'head_error_draws.json',PARENT/'PLAN.json',PARENT/'result.json']
    paths.extend(model_paths().values());paths.extend(OUT/'poses'/f'unit{u}.npz' for u in units)
    for path,expected in manifest['source_hashes'].items():
        assert D.sha(ROOT/path)==expected,path
        paths.append(ROOT/path)
    for path,expected in manifest['poses_sha256'].items():assert D.sha(OUT/path)==expected,path
    parity_anchor=D.read(WORK/'cnh-double-height-dev-20261007/geometry_manifest.json')['anchors'][0]
    paths.extend(source_paths(parity_anchor['unit']).values())
    for path in paths:
        path=Path(path);hashes[str(path.relative_to(ROOT))]=D.sha(path)
    ts=thresholds(D.read(PARENT/'result.json'))
    counts={t:dict(rows=sum(r['tag']==t for r in rows),valid=sum(r['tag']==t and r['valid'] for r in rows),
        evaluable=sum(r['tag']==t and r['evaluable'] for r in rows),
        events=sum(r['tag']==t and r['evaluable'] and bool(r['contact']) for r in rows),
        controls=sum(r['tag']==t and r['evaluable'] and bool(r['control']) for r in rows)) for t in TAGS}
    for value in counts.values():
        value['one_event_percentage_points']=100/value['events'] if value['events'] else None
        value['one_control_interval_percentage_points']=10/value['controls'] if value['controls'] else None
    D.save(OUT/'PLAN.json',dict(phase='EXPLORE new-unit fixed-model stability; same simulator/template distribution',
        authorization='Continuous algorithm exploration;48new units x40configs xoriginal/HB fixed batch',
        budget_gpu_wall_seconds=900,budget_analysis_cpu_seconds=300,budget_contrasts_cpu_seconds=60,
        gpu_batch_size=BATCH_SIZE,threads=4,units=units,rows=3840,counts=counts,
        compute='GPU physical rendering and batched five-seed M3 inference; CPU fixed9tree scoring after GPU cleanup',
        methods=list(METHODS),old_model_folds=parent['folds'],
        parity_anchor={k:parity_anchor[k] for k in ('unit','config','anchor_id')},
        parity=dict(hist_exact=True,ambient_atol=1e-6,z1_atol=.002,raw_atol=.005),
        selection='Keep full48x40x2 geometry manifest including pass/censor/overlap records. Render only geometry-defined evaluable event/control rows; no score selection or replacement.',
        input='Renderer uses frozen scene/physical source poses. M3 input is only z1/nn/qq; raw learner pair2+current136 from z1/ambient. Evaluator truth/travel/labels never readout inputs.',
        threshold='All3 existing model folds evaluate every identical new observation with their own original-cal2.5/5% thresholds; no new-unit fold assignment, fitting or recalibration. Identical original_center scores get all3 original thresholds.',
        thresholds=[dict(fold=f,cap=c,method=m,threshold=v) for (f,c,m),v in ts.items()],
        primary='No-warmup timely and actualFA[2:12] separately for each model fold and original/HB tag, plus warmup and any-before counts; none/corner and mode strata retained.',
        ranking='Auxiliary contact peak score[2:deadline+1] versus every strict-control interval score[2:12]; whole-tie reachable step area overFA budget0..5%, normalized by.05, optional pairwiseAUC. Different event/interval statistic units, no interpolated or selected deployment threshold.',
        decision_check='At original-cal2.5%, candidate raw_hb_aug versus each corresponding max threshold in all3model-fold x2tag cells: consistent timely gains without higher actualFA support a stable candidate. Any model/structure regression remains visible and motivates model/calibration diagnosis; no averaging away failures, statistical noninferiority or promotion.',
        limitations=['New unit/noise/placement seeds within the same synthetic configuration distribution, not new real scenes',
                    'Existing real heading-error trajectories are reused intact; no new participant or direction-error evidence',
                    'Three models are correlated fixed fits, not independent training replications','HB structure appeared in prior training; fresh units do not make structure fresh','Geometry missing/pass rows are not negatives'],
        hashes=hashes))
    print('FROZEN',counts,flush=True)


def verify(plan):
    for path,expected in plan['hashes'].items():
        if D.sha(ROOT/path)!=expected:raise ValueError('Frozen input changed: '+path)


def run():
    plan=D.read(OUT/'PLAN.json');verify(plan);manifest=checked_manifest();rows=manifest['rows']
    if (OUT/'GPU_START.json').exists():raise FileExistsError('Single GPU run; retain prior progress and terminal')
    start=time.monotonic();D.save(OUT/'GPU_START.json',dict(pid=os.getpid(),started_unix=time.time(),budget_seconds=900))
    completed=[];runner=None
    def check():
        if time.monotonic()-start>plan['budget_gpu_wall_seconds']:raise TimeoutError('900s GPU wall budget reached')
    try:
        from cnh_double_height_render_dev import DoubleHeightRunner,load_source
        import cnh_model_stability_geometry_dev as G
        runner=DoubleHeightRunner(OUT)
        with runner:
            pa=plan['parity_anchor'];source=load_source(pa['unit'],pa['config'])
            parity=runner.anchor_parity(source,**{k:v for k,v in plan['parity'].items() if k!='hist_exact'})
            D.save(OUT/'parity.json',parity)
            if not parity['passed']:raise ValueError('Old anchor parity failed; no new-unit measurements')
            print('PARITY passed',flush=True)
            for ui,unit in enumerate(plan['units']):
                selected=[r for r in rows if r['unit']==unit and r['evaluable']];source_cache={}
                for bi in range(0,len(selected),plan['gpu_batch_size']):
                    check();batch=selected[bi:bi+plan['gpu_batch_size']];items=[];sources=[]
                    for r in batch:
                        if r['config'] not in source_cache:source_cache[r['config']]=G.load_source(r['unit'],r['config'],OUT)
                        source=source_cache[r['config']];sources.append(source)
                        expected=int(np.random.SeedSequence([2026100606,r['unit'],r['config'],0]).generate_state(1)[0])
                        assert source['photon_seed']==expected
                        items.append(runner.render_observation(source,r['boxes'],r['tag']))
                    runner.infer_observations(items)
                    for r,item,source in zip(batch,items,sources):
                        for k in ('nn','qq'):np.testing.assert_array_equal(item[k],source[k])
                        dest=OUT/'observations'/f"unit{r['unit']}"/f"c{r['config']}_{r['tag']}.npz"
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        if dest.exists():raise FileExistsError(dest)
                        np.savez_compressed(dest,**{k:item[k] for k in ('hist','ambient','z1','nn','qq','raw','smooth')})
                        D.save(dest.with_suffix('.json'),dict(row_id=r['row_id'],unit=r['unit'],config=r['config'],tag=r['tag'],
                            photon_seed=source['photon_seed'],sha256=D.sha(dest),backend=item['backend']))
                        completed.append(r['row_id'])
                    D.save(OUT/'progress.json',dict(stage='render_infer',unit=unit,unit_index=ui+1,units=48,completed=len(completed),
                        total=sum(v['evaluable'] for v in plan['counts'].values()),elapsed_seconds=time.monotonic()-start,last_activity_unix=time.time()),replace=True)
                    print('BATCH',unit,bi//plan['gpu_batch_size']+1,'completed',len(completed),'elapsed',round(time.monotonic()-start,1),flush=True);check()
        assert len(completed)==sum(r['evaluable'] for r in rows);verify(plan);check()
        D.save(OUT/'gpu_terminal.json',dict(status='COMPLETE',completed=completed,seconds=time.monotonic()-start,cleanup=runner.cleanup_receipt,pid=os.getpid()))
    except BaseException as exc:
        D.save(OUT/'gpu_terminal.json',dict(status='STOPPED_PARTIAL' if isinstance(exc,TimeoutError) else 'FAILED',completed=completed,seconds=time.monotonic()-start,
            error=repr(exc),traceback=traceback.format_exc(),cleanup=runner.cleanup_receipt if runner is not None else None,pid=os.getpid()))
        raise


def analyze():
    import cnh_direction_information_dev as I
    import cnh_querywise_calibration_dev as Q
    import cnh_raw_radial_features_dev as F
    from cnh_model_stability_ranking_dev import ranking_metrics
    import joblib
    from threadpoolctl import threadpool_limits
    start=time.monotonic();plan=D.read(OUT/'PLAN.json');verify(plan);manifest=checked_manifest();rows=manifest['rows']
    if D.read(OUT/'gpu_terminal.json')['status']!='COMPLETE':raise ValueError('Incomplete GPU batch; retain partial, no full analysis')
    if (OUT/'CPU_START.json').exists():raise FileExistsError('Single CPU analysis, preserve outputs')
    D.save(OUT/'CPU_START.json',dict(started_unix=time.time(),budget_seconds=300))
    def check():
        if time.monotonic()-start>plan['budget_analysis_cpu_seconds']:raise TimeoutError('300s CPU analysis budget reached')
    try:
        n=len(rows);metadata={k:np.array([r[k] for r in rows]) for k in ('unit','config','tag','mode','mirror','family','valid','evaluable')}
        evaluable=metadata['evaluable'];contact=np.array([bool(r['contact']) for r in rows]);control=np.array([bool(r['control']) for r in rows])
        deadline=np.array([int(r['deadline_index']) if r['evaluable'] and r['deadline_index'] is not None else -1 for r in rows])
        pair=np.full((n,13,2),np.nan);current=np.full((n,13,136),np.nan,np.float32);observations=[]
        for i,r in enumerate(rows):
            check()
            if not r['evaluable']:continue
            path=OUT/'observations'/f"unit{r['unit']}"/f"c{r['config']}_{r['tag']}.npz";receipt=D.read(path.with_suffix('.json'))
            assert receipt['row_id']==r['row_id'] and receipt['sha256']==D.sha(path)
            with np.load(path) as z:
                pair[i]=z['smooth'];current[i]=F.extract_features(z['z1'],z['ambient'],z['nn'])['current']
            observations.append(dict(row_index=i,path=str(path.relative_to(ROOT)),sha256=receipt['sha256']))
        x=np.concatenate((pair,current),axis=-1);assert np.isfinite(x[evaluable]).all()
        ts=thresholds(D.read(PARENT/'result.json'));models={key:joblib.load(p) for key,p in model_paths().items()}
        scores={};alarms={};timely={};post={};metrics={};ranking={};c=contact&evaluable;ct=control&evaluable
        groups={'all':np.ones(n,bool),'family_none':metadata['family']=='none','family_corner':metadata['family']=='corner'}
        groups.update({f'mode{mode}':metadata['mode']==mode for mode in sorted(np.unique(metadata['mode']))})
        with threadpool_limits(limits=4):
            for fold in range(3):
                for method in METHODS:
                    check();score=np.full((n,13),np.nan)
                    if method=='original_center':score[evaluable]=pair[evaluable].max(-1)
                    else:
                        features=x[evaluable] if method.startswith('raw_') else pair[evaluable]
                        score[evaluable]=models[fold,method].predict_proba(features.reshape(-1,features.shape[-1]))[:,1].reshape(evaluable.sum(),13)
                    skey=f'fold{fold}/{method}';scores[skey]=score
                    ranking[skey]={t:ranking_metrics(score,c,ct,deadline,mask=(metadata['tag']==t)&evaluable,cap=.05) for t in TAGS}
                    for cap in ('0.025','0.050'):
                        key=f'fold{fold}/calibrated/{cap}/{method}';al=evaluable[:,None]&(score>=ts[fold,cap,method]);alarms[key]=al
                        timely[key],post[key]=Q.event_states(al,c,deadline)
                        metrics[key]={t:{name:Q.metrics(al,c,ct,deadline,(metadata['tag']==t)&mask&evaluable) for name,mask in groups.items()} for t in TAGS}
                    print('SCORE',fold,method,flush=True)
        check()
        np.savez_compressed(OUT/'ledger.npz',**metadata,contact=contact,control=control,deadline=deadline,
            row_id=np.array([r['row_id'] for r in rows]),anchor_id=np.array([r['anchor_id'] for r in rows]),
            contact_query=np.array([r['contact_query'] for r in rows],bool),pair=pair,current=current,
            **{'score/'+k:v for k,v in scores.items()},**{'alarm/'+k:v for k,v in alarms.items()},
            **{'timely/'+k:v for k,v in timely.items()},**{'timely_excluding_warmup/'+k:v for k,v in post.items()})
        D.save(OUT/'readout_inheritance.json',dict(result_sha256=D.sha(PARENT/'result.json'),
            thresholds=[dict(fold=f,cap=cap,method=m,threshold=v) for (f,cap,m),v in ts.items()],
            model_sha256={f'{f}/{m}':D.sha(p) for (f,m),p in model_paths().items()},no_fit=True,no_new_thresholds=True))
        D.save(OUT/'result.json',dict(status='EXPLORATORY_COMPLETE',seconds=time.monotonic()-start,rows=n,evaluable_rows=int(evaluable.sum()),
            counts=plan['counts'],metrics=metrics,ranking=ranking,observations=observations,plan_sha256=D.sha(OUT/'PLAN.json'),ledger_sha256=D.sha(OUT/'ledger.npz')))
        D.save(OUT/'analysis_terminal.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
        for fold in range(3):
            for method in ('original_center','raw_hb_aug'):
                key=f'fold{fold}/calibrated/0.025/{method}';print('RESULT',key,{t:metrics[key][t]['all'] for t in TAGS},flush=True)
    except BaseException as exc:
        D.save(OUT/'analysis_terminal.json',dict(status='STOPPED_PARTIAL' if isinstance(exc,TimeoutError) else 'FAILED',seconds=time.monotonic()-start,error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','analyze']);globals()[parser.parse_args().stage]()

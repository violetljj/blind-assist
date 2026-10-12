"""Shared original-height control panel, fixed models and consumed evaluation."""
import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
import cnh_model_stability_dev as S

D,ROOT,HERE,WORK=S.D,S.ROOT,S.HERE,S.WORK
OUT=WORK/'cnh-shared-calibration-dev-20261007'
EVALUATION=S.OUT
PARENT=S.PARENT
METHODS=S.METHODS
CAPS=(.025,.05)


def checked_manifest():
    manifest=D.read(OUT/'geometry_manifest.json');assert manifest['status']=='COMPLETE'
    rows=manifest['rows'];units=sorted({r['unit'] for r in rows})
    assert units==list(range(410052,410100)) and len(rows)==1920
    assert [(r['unit'],r['config'],r['tag']) for r in rows]==[(u,c,'original') for u in units for c in range(40)]
    for r in rows:
        expected=bool(r['valid'] and r['control'])
        assert r['evaluable']==r['eligible_control']==expected
        if expected:
            assert not r['contact'] and np.asarray(r['frame_category']).shape==(13,2)
            assert (np.asarray(r['frame_category'])=='clear').all()
    return manifest


def shared_thresholds(scores,eligible,caps=CAPS):
    """Identical calibration rows/time windows, per-model thresholds, intact ties."""
    import cnh_direction_information_dev as I
    eligible=np.asarray(eligible,bool)
    if not eligible.any():raise ValueError('Shared panel needs at least one strict-clear sequence')
    thresholds={};records=[]
    for fold in range(3):
        for method in METHODS:
            x=np.asarray(scores[f'fold{fold}/{method}'])
            if x.shape!=(len(eligible),13) or not np.isfinite(x[eligible]).all():raise ValueError('Finite13-frame calibration score required')
            values=x[eligible,2:12]
            for cap in caps:
                theta=float(I.P.threshold_at_cap(values,cap));alarm=values>=theta
                assert float(alarm.mean())<=cap+1e-12
                key=fold,f'{cap:.3f}',method;thresholds[key]=theta
                records.append(dict(fold=fold,cap=f'{cap:.3f}',method=method,threshold=theta,
                    calibration_controls=int(eligible.sum()),calibration_intervals=int(values.size),
                    calibration_alarm_intervals=int(alarm.sum()),calibration_false_alarm=float(alarm.mean()),
                    tie_intervals_at_threshold=int((values==theta).sum()),
                    excluded_boundary_score=float(np.nextafter(theta,-np.inf)),
                    excluded_boundary_tie_intervals=int((values==np.nextafter(theta,-np.inf)).sum()),
                    unused_allowable_intervals=int(np.floor(cap*values.size))-int(alarm.sum()),comparison='>=; whole ties retained'))
    for fold in (1,2):np.testing.assert_array_equal(scores[f'fold{fold}/original_center'][eligible],scores['fold0/original_center'][eligible])
    for cap in caps:assert len({thresholds[f,f'{cap:.3f}','original_center'] for f in range(3)})==1
    return thresholds,records


def freeze():
    from cnh_double_height_render_dev import source_paths
    manifest=checked_manifest();rows=manifest['rows'];units=sorted({r['unit'] for r in rows})
    parent=D.read(EVALUATION/'PLAN.json');assert not set(units)&set(parent['units'])
    trained={u for fold in parent['old_model_folds'] for role in fold.values() for u in role};assert not set(units)&trained
    hashes=dict(parent['hashes'])
    paths=[Path(__file__),HERE/'cnh_shared_calibration_geometry_dev.py',HERE/'cnh_shared_calibration_contrasts_dev.py',
        OUT/'geometry_manifest.json',OUT/'GEOMETRY_PLAN.json',OUT/'head_error_draws.json',
        EVALUATION/'PLAN.json',EVALUATION/'ledger.npz',EVALUATION/'result.json',PARENT/'result.json']
    paths.extend(S.model_paths().values());paths.extend(OUT/'poses'/f'unit{u}.npz' for u in units)
    for path,expected in manifest['source_hashes'].items():assert D.sha(ROOT/path)==expected,path;paths.append(ROOT/path)
    for path,expected in manifest['poses_sha256'].items():assert D.sha(OUT/path)==expected,path
    pa=parent['parity_anchor'];paths.extend(source_paths(pa['unit']).values())
    for path in paths:path=Path(path);hashes[str(path.relative_to(ROOT))]=D.sha(path)
    n=sum(r['eligible_control'] for r in rows);assert n>0
    D.save(OUT/'PLAN.json',dict(phase='POSTHOC common-calibration diagnostic; evaluation outcomes already consumed',
        authorization='Continuous algorithm exploration; additional48original-only calibration units, fixed evaluation and models',
        budget_gpu_wall_seconds=600,budget_analysis_cpu_seconds=300,budget_contrasts_cpu_seconds=60,
        gpu_batch_size=8,threads=4,units=units,rows=1920,calibration_controls=n,calibration_intervals=10*n,
        parity_anchor=pa,parity=dict(hist_exact=True,ambient_atol=1e-6,z1_atol=.002,raw_atol=.005),methods=list(METHODS),
        compute='GPU batched rendering and M3 only for new strict-clear controls; CPU fixed9trees plus scalar thresholds',
        calibration='Only valid original-height sequences clear in both queries at all13frames. Same selected rows and[2:12]intervals for every3fold x4method. Existing threshold_at_cap with>= semantics and whole ties at2.5/5%. No HB pooled arm.',
        input='Only M3 z1/nn/qq and pair2+current136(z1/ambient) reach fixed readouts; no geometry labels/travel as features.',
        positive_control='Three identical original_center score arrays must yield exactly identical shared thresholds.',
        evaluation='Reuse the immutable completed model-stability ledger predictions and truth/masks for all3models; no evaluation rendering, fitting, rescoring, recalibration or model selection.',
        legacy='Keep every original-cal threshold and resulting evaluation alarm/metric as separate legacy_calibrated reference. New shared_calibrated thresholds are model-specific, not shared numerically across learned models.',
        ranking='Inherited unchanged from completed evaluation result; scalar threshold changes cannot improve score ranking.',
        primary='At shared-cal2.5%, each3model x2tag cell compares candidate raw_hb_aug versus max in no-warmup timely and actualFA. Report alltradeoffs, mode/family breakdown and legacy changes; no same-actualFA claim, statistical noninferiority or promotion.',
        decision_check='Each evaluation tag has140 events(one event0.714percentage points); original371controls/3710intervals, HB357/3570. Legacy max is not saturated. Judge timely together with actualFA, not timely alone; retain every model/tag tradeoff.',
        limitations=['Fixed common calibration data removes the old calibration-panel mixture difference, not fixed-model learning variability',
                    'New calibration units are independent of old model training and evaluation units, but evaluation outcomes are already consumed',
                    'Same simulator and reused heading-error pool, no new real-scene evidence','Only calibration controls are rendered; ignored geometry rows remain missing, not negatives'],
        hashes=hashes))
    print('FROZEN shared original-only calibration',n,'controls',flush=True)


def verify(plan):
    for path,expected in plan['hashes'].items():
        if D.sha(ROOT/path)!=expected:raise ValueError('Frozen input changed: '+path)


def run():
    plan=D.read(OUT/'PLAN.json');verify(plan);rows=checked_manifest()['rows']
    if (OUT/'GPU_START.json').exists():raise FileExistsError('Single GPU run; retain existing progress')
    start=time.monotonic();D.save(OUT/'GPU_START.json',dict(pid=os.getpid(),started_unix=time.time(),budget_seconds=600))
    completed=[];runner=None
    def check():
        if time.monotonic()-start>plan['budget_gpu_wall_seconds']:raise TimeoutError('600s GPU wall budget reached')
    try:
        from cnh_double_height_render_dev import DoubleHeightRunner,load_source
        import cnh_shared_calibration_geometry_dev as G
        runner=DoubleHeightRunner(OUT)
        with runner:
            pa=plan['parity_anchor'];source=load_source(pa['unit'],pa['config'])
            parity=runner.anchor_parity(source,**{k:v for k,v in plan['parity'].items() if k!='hist_exact'})
            D.save(OUT/'parity.json',parity)
            if not parity['passed']:raise ValueError('Old anchor parity failed; stop before new controls')
            for ui,unit in enumerate(plan['units']):
                selected=[r for r in rows if r['unit']==unit and r['eligible_control']]
                for bi in range(0,len(selected),plan['gpu_batch_size']):
                    check();batch=selected[bi:bi+plan['gpu_batch_size']];items=[];sources=[]
                    for r in batch:
                        source=G.load_source(r['unit'],r['config'],OUT);sources.append(source)
                        assert source['photon_seed']==int(np.random.SeedSequence([2026100606,r['unit'],r['config'],0]).generate_state(1)[0])
                        items.append(runner.render_observation(source,r['boxes'],'original'))
                    runner.infer_observations(items)
                    for r,item,source in zip(batch,items,sources):
                        for key in ('nn','qq'):np.testing.assert_array_equal(item[key],source[key])
                        path=OUT/'observations'/f"unit{r['unit']}"/f"c{r['config']}_original.npz";path.parent.mkdir(parents=True,exist_ok=True)
                        if path.exists():raise FileExistsError(path)
                        np.savez_compressed(path,**{k:item[k] for k in ('hist','ambient','z1','nn','qq','raw','smooth')})
                        D.save(path.with_suffix('.json'),dict(row_id=r['row_id'],photon_seed=source['photon_seed'],backend=item['backend'],sha256=D.sha(path)))
                        completed.append(r['row_id'])
                    D.save(OUT/'progress.json',dict(stage='calibration_render',unit=unit,unit_index=ui+1,completed=len(completed),total=plan['calibration_controls'],
                        elapsed_seconds=time.monotonic()-start,last_activity_unix=time.time()),replace=True)
                    print('BATCH',unit,'completed',len(completed),'elapsed',round(time.monotonic()-start,1),flush=True);check()
        assert len(completed)==plan['calibration_controls'];verify(plan);check()
        D.save(OUT/'gpu_terminal.json',dict(status='COMPLETE',completed=completed,seconds=time.monotonic()-start,cleanup=runner.cleanup_receipt,pid=os.getpid()))
    except BaseException as exc:
        D.save(OUT/'gpu_terminal.json',dict(status='STOPPED_PARTIAL' if isinstance(exc,TimeoutError) else 'FAILED',completed=completed,seconds=time.monotonic()-start,
            error=repr(exc),traceback=traceback.format_exc(),cleanup=runner.cleanup_receipt if runner is not None else None,pid=os.getpid()))
        raise


def analyze():
    import cnh_direction_information_dev as I
    import cnh_querywise_calibration_dev as Q
    import cnh_raw_radial_features_dev as F
    import joblib
    from threadpoolctl import threadpool_limits
    start=time.monotonic();plan=D.read(OUT/'PLAN.json');verify(plan);rows=checked_manifest()['rows']
    if D.read(OUT/'gpu_terminal.json')['status']!='COMPLETE':raise ValueError('Calibration rendering incomplete')
    if (OUT/'CPU_START.json').exists():raise FileExistsError('Single CPU analysis; preserve outputs')
    D.save(OUT/'CPU_START.json',dict(started_unix=time.time(),budget_seconds=300))
    def check():
        if time.monotonic()-start>plan['budget_analysis_cpu_seconds']:raise TimeoutError('300s CPU analysis budget reached')
    try:
        n=len(rows);eligible=np.array([r['eligible_control'] for r in rows]);pair=np.full((n,13,2),np.nan);current=np.full((n,13,136),np.nan,np.float32)
        observations=[]
        for i,r in enumerate(rows):
            check()
            if not eligible[i]:continue
            path=OUT/'observations'/f"unit{r['unit']}"/f"c{r['config']}_original.npz";receipt=D.read(path.with_suffix('.json'))
            assert receipt['row_id']==r['row_id'] and receipt['sha256']==D.sha(path)
            with np.load(path) as z:pair[i]=z['smooth'];current[i]=F.extract_features(z['z1'],z['ambient'],z['nn'])['current']
            observations.append(dict(row_index=i,path=str(path.relative_to(ROOT)),sha256=receipt['sha256']))
        x=np.concatenate((pair,current),axis=-1);assert np.isfinite(x[eligible]).all()
        models={key:joblib.load(path) for key,path in S.model_paths().items()};cal_scores={}
        with threadpool_limits(limits=4):
            for fold in range(3):
                for method in METHODS:
                    check();score=np.full((n,13),np.nan)
                    if method=='original_center':score[eligible]=pair[eligible].max(-1)
                    else:
                        features=x[eligible] if method.startswith('raw_') else pair[eligible]
                        score[eligible]=models[fold,method].predict_proba(features.reshape(-1,features.shape[-1]))[:,1].reshape(eligible.sum(),13)
                    cal_scores[f'fold{fold}/{method}']=score
        shared,calrecords=shared_thresholds(cal_scores,eligible);legacy=S.thresholds(D.read(PARENT/'result.json'))
        old_result=D.read(EVALUATION/'result.json')
        with np.load(EVALUATION/'ledger.npz') as z:
            metadata={k:z[k] for k in ('unit','config','tag','mode','mirror','family','valid','evaluable','contact','control','deadline','row_id','anchor_id','contact_query')}
            eval_scores={k:z['score/'+k] for k in cal_scores}
            oldalarms={k:z['alarm/'+k] for k in old_result['metrics']}
        assert not set(metadata['unit'])&{r['unit'] for r in rows}
        ev=metadata['evaluable'];contact=metadata['contact']&ev;control=metadata['control']&ev;deadline=metadata['deadline'];ne=len(ev)
        groups={'all':np.ones(ne,bool),'family_none':metadata['family']=='none','family_corner':metadata['family']=='corner'}
        groups.update({f'mode{m}':metadata['mode']==m for m in sorted(np.unique(metadata['mode']))})
        alarms={};timely={};post={};metrics={}
        for fold in range(3):
            for method in METHODS:
                score=eval_scores[f'fold{fold}/{method}']
                for cap in ('0.025','0.050'):
                    for policy,ts in [('shared_calibrated',shared),('legacy_calibrated',legacy)]:
                        key=f'fold{fold}/{policy}/{cap}/{method}';al=ev[:,None]&(score>=ts[fold,cap,method]);alarms[key]=al
                        timely[key],post[key]=Q.event_states(al,contact,deadline)
                        metrics[key]={tag:{name:Q.metrics(al,contact,control,deadline,(metadata['tag']==tag)&mask&ev) for name,mask in groups.items()} for tag in S.TAGS}
                        if policy=='legacy_calibrated':
                            oldkey=f'fold{fold}/calibrated/{cap}/{method}'
                            np.testing.assert_array_equal(al,oldalarms[oldkey]);assert metrics[key]==old_result['metrics'][oldkey]
        check()
        np.savez_compressed(OUT/'calibration_ledger.npz',unit=np.array([r['unit'] for r in rows]),config=np.array([r['config'] for r in rows]),
            tag=np.array([r['tag'] for r in rows]),mode=np.array([r['mode'] for r in rows]),mirror=np.array([r['mirror'] for r in rows]),
            family=np.array([r['family'] for r in rows]),valid=np.array([r['valid'] for r in rows]),eligible_control=eligible,evaluable=eligible,
            contact=np.array([bool(r['contact']) for r in rows]),control=np.array([bool(r['control']) for r in rows]),pair=pair,current=current,
            **{'score/'+key:value for key,value in cal_scores.items()})
        np.savez_compressed(OUT/'ledger.npz',**metadata,**{'score/'+k:v for k,v in eval_scores.items()},**{'alarm/'+k:v for k,v in alarms.items()},
            **{'timely/'+k:v for k,v in timely.items()},**{'timely_excluding_warmup/'+k:v for k,v in post.items()})
        D.save(OUT/'thresholds.json',dict(shared=calrecords,legacy=[dict(fold=f,cap=cap,method=m,threshold=v) for (f,cap,m),v in legacy.items()],
            original_center_threefold_identical=True,no_hb_calibration=True))
        D.save(OUT/'readout_inheritance.json',dict(evaluation_ledger_sha256=D.sha(EVALUATION/'ledger.npz'),evaluation_result_sha256=D.sha(EVALUATION/'result.json'),
            old_threshold_result_sha256=D.sha(PARENT/'result.json'),model_sha256={f'{f}/{m}':D.sha(p) for (f,m),p in S.model_paths().items()},
            evaluation_scores_exactly_reused=True,legacy_alarms_and_metrics_exact=True,no_refit=True,ranking_unchanged=True))
        D.save(OUT/'result.json',dict(status='POSTHOC_EXPLORATORY_COMPLETE',seconds=time.monotonic()-start,calibration_controls=int(eligible.sum()),calibration_intervals=int(eligible.sum())*10,
            evaluation_rows=ne,evaluation_counts=old_result['counts'],thresholds=calrecords,metrics=metrics,ranking=old_result['ranking'],observations=observations,
            plan_sha256=D.sha(OUT/'PLAN.json'),ledger_sha256=D.sha(OUT/'ledger.npz'),calibration_ledger_sha256=D.sha(OUT/'calibration_ledger.npz')))
        D.save(OUT/'analysis_terminal.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
        for fold in range(3):
            for method in ('original_center','raw_hb_aug'):
                key=f'fold{fold}/shared_calibrated/0.025/{method}';print('RESULT',key,{tag:metrics[key][tag]['all'] for tag in S.TAGS},flush=True)
    except BaseException as exc:
        D.save(OUT/'analysis_terminal.json',dict(status='STOPPED_PARTIAL' if isinstance(exc,TimeoutError) else 'FAILED',seconds=time.monotonic()-start,error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','analyze']);globals()[parser.parse_args().stage]()

"""One scalar threshold constrained by both height-specific clear panels."""
import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
import cnh_shared_calibration_dev as C

D,ROOT,HERE,WORK=C.D,C.ROOT,C.HERE,C.WORK
OUT=WORK/'cnh-group-calibration-dev-20261007'
SHARED=C.OUT
METHODS=C.METHODS
TAGS=('original','HB')


def checked_manifest():
    manifest=D.read(OUT/'geometry_manifest.json');assert manifest['status']=='COMPLETE'
    rows=manifest['rows'];old=C.checked_manifest()['rows'];assert len(rows)==3840 and rows[::2]==old
    units=sorted({r['unit'] for r in rows});assert units==list(range(410052,410100))
    assert [(r['unit'],r['config'],r['tag']) for r in rows]==[(u,c,t) for u in units for c in range(40) for t in TAGS]
    for r in rows:
        eligible=bool(r['valid'] and r['control']);assert r['eligible_control']==r['evaluable']==eligible
        if eligible:assert not r['contact'] and (np.asarray(r['frame_category'])=='clear').all()
    return manifest


def group_thresholds(scores,eligible,tags,caps=C.CAPS):
    """Minimal shared scalar satisfying both empirical group caps with >= ties."""
    eligible=np.asarray(eligible,bool);tags=np.asarray(tags)
    individual={};detail={}
    for tag in TAGS:
        mask=eligible&(tags==tag)
        ts,records=C.shared_thresholds(scores,mask,caps)
        individual[tag]=ts;detail[tag]={(r['fold'],r['cap'],r['method']):r for r in records}
    combined={};records=[]
    for key in individual['original']:
        fold,cap_text,method=key;cap=float(cap_text);theta=max(individual[tag][key] for tag in TAGS)
        combined[key]=theta;groups={};predecessor=float(np.nextafter(theta,-np.inf));previous_violations=[]
        for tag in TAGS:
            values=np.asarray(scores[f'fold{fold}/{method}'])[eligible&(tags==tag),2:12]
            fa=float((values>=theta).mean());previous_fa=float((values>=predecessor).mean())
            assert fa<=cap+1e-12;previous_violations.append(previous_fa>cap+1e-12)
            groups[tag]=dict(**detail[tag][key],group_threshold_alarm_intervals=int((values>=theta).sum()),
                group_threshold_false_alarm=fa,previous_float_false_alarm=previous_fa,
                group_threshold_excluded_boundary_tie_intervals=int((values==predecessor).sum()),
                group_threshold_unused_allowable_intervals=int(np.floor(cap*values.size))-int((values>=theta).sum()))
        assert any(previous_violations),'Previous representable threshold must violate at least one group cap'
        records.append(dict(fold=fold,cap=cap_text,method=method,threshold=theta,predecessor=predecessor,
            original_threshold=individual['original'][key],hb_threshold=individual['HB'][key],groups=groups,
            minimality_verified=True,comparison='>=; one scalar, no height input at inference'))
    for cap in caps:assert len({combined[f,f'{cap:.3f}','original_center'] for f in range(3)})==1
    return combined,records,individual


def freeze():
    manifest=checked_manifest();rows=manifest['rows'];parent=D.read(SHARED/'PLAN.json');hashes=dict(parent['hashes'])
    paths=[Path(__file__),HERE/'cnh_group_calibration_geometry_dev.py',HERE/'cnh_group_calibration_contrasts_dev.py',
        OUT/'geometry_manifest.json',OUT/'GEOMETRY_PLAN.json',SHARED/'PLAN.json',SHARED/'calibration_ledger.npz',
        SHARED/'ledger.npz',SHARED/'result.json',SHARED/'thresholds.json']
    paths.extend(C.S.model_paths().values())
    for path,expected in manifest['source_hashes'].items():assert D.sha(ROOT/path)==expected,path;paths.append(ROOT/path)
    for path,expected in manifest['poses_sha256'].items():
        p=ROOT/manifest['poses_root']/path;assert D.sha(p)==expected,path;paths.append(p)
    for path in paths:path=Path(path);hashes[str(path.relative_to(ROOT))]=D.sha(path)
    counts={tag:sum(r['tag']==tag and r['eligible_control'] for r in rows) for tag in TAGS};assert min(counts.values())>0
    D.save(OUT/'PLAN.json',dict(phase='POSTHOC group calibration on consumed fixed evaluation',
        authorization='Continuous algorithm exploration; per-height empirical false-alarm coverage diagnostic',
        budget_gpu_wall_seconds=600,budget_analysis_cpu_seconds=300,budget_contrasts_cpu_seconds=60,threads=4,gpu_batch_size=8,
        units=parent['units'],rows=3840,calibration_controls=counts,new_hb_renders=counts['HB'],methods=list(METHODS),
        parity_anchor=parent['parity_anchor'],parity=parent['parity'],
        calibration='For each fixed fold/method/cap2.5/5%, compute threshold_at_cap separately for original and HB strict-clear[2:12]; use their maximum as one scalar. Whole ties and>=. No height knowledge at inference, no selected best model.',
        source='Original1920 rows and all original calibration scores exactly reused from shared panel; only independently valid HB clear sequences newly rendered from the same immutable source poses.',
        evaluation='All3840 stability evaluation rows, score arrays and truth/masks reused exactly. Keep legacy_calibrated and shared_calibrated, add group_calibrated; no new evaluation scores/rendering/fitting.',
        known_property='Group threshold>=shared original threshold. Every group alarm must be a subset of its shared-policy alarm, and timely cannot increase for that same model. This is FA coverage versus detection, not added ranking information.',
        metrics='Each3models x2tags at2.5/5%, no-warmup and any-before timely, actualFA[2:12], warmup, family and mode; ranking unchanged.',
        decision='Compare raw_hb_aug with max under the same group-calibration rule, and each model with its shared policy. Judge timely and actualFA jointly, retain all cells; empirical calibration caps do not guarantee evaluation caps.',
        contrasts='1000mode/mirror-stratified complete evaluation-unit bootstrap draws, fixed models/thresholds,60s budget; no calibration-panel resampling.',
        limitations=['Calibration uses known height groups, inference uses a single threshold','Consumed evaluation and reused synthetic family/error pool, not fresh confirmation',
                    'No probability calibration or formal safety guarantee','Calibration-support correction cannot change score ranking'],hashes=hashes))
    print('FROZEN group calibration',counts,flush=True)


def verify(plan):
    for path,expected in plan['hashes'].items():
        if D.sha(ROOT/path)!=expected:raise ValueError('Frozen input changed: '+path)


def run():
    plan=D.read(OUT/'PLAN.json');verify(plan);rows=checked_manifest()['rows']
    if (OUT/'GPU_START.json').exists():raise FileExistsError('Single GPU run; preserve prior payload')
    start=time.monotonic();D.save(OUT/'GPU_START.json',dict(pid=os.getpid(),started_unix=time.time(),budget_seconds=600))
    completed=[];runner=None
    def check():
        if time.monotonic()-start>plan['budget_gpu_wall_seconds']:raise TimeoutError('600s GPU wall budget reached')
    try:
        from cnh_double_height_render_dev import DoubleHeightRunner,load_source
        import cnh_group_calibration_geometry_dev as G
        runner=DoubleHeightRunner(OUT)
        with runner:
            pa=plan['parity_anchor'];source=load_source(pa['unit'],pa['config'])
            parity=runner.anchor_parity(source,**{k:v for k,v in plan['parity'].items() if k!='hist_exact'});D.save(OUT/'parity.json',parity)
            if not parity['passed']:raise ValueError('Old anchor parity failed; no new HB measurements')
            for ui,unit in enumerate(plan['units']):
                selected=[r for r in rows if r['unit']==unit and r['tag']=='HB' and r['eligible_control']]
                for bi in range(0,len(selected),plan['gpu_batch_size']):
                    check();batch=selected[bi:bi+plan['gpu_batch_size']];items=[];sources=[]
                    for r in batch:
                        source=G.load_source(r['unit'],r['config'],OUT);sources.append(source)
                        assert source['photon_seed']==int(np.random.SeedSequence([2026100606,r['unit'],r['config'],0]).generate_state(1)[0])
                        items.append(runner.render_observation(source,r['boxes'],'HB'))
                    runner.infer_observations(items)
                    for r,item,source in zip(batch,items,sources):
                        for k in ('nn','qq'):np.testing.assert_array_equal(item[k],source[k])
                        p=OUT/'observations'/f"unit{r['unit']}"/f"c{r['config']}_HB.npz";p.parent.mkdir(parents=True,exist_ok=True)
                        if p.exists():raise FileExistsError(p)
                        np.savez_compressed(p,**{k:item[k] for k in ('hist','ambient','z1','nn','qq','raw','smooth')})
                        D.save(p.with_suffix('.json'),dict(row_id=r['row_id'],photon_seed=source['photon_seed'],backend=item['backend'],sha256=D.sha(p)))
                        completed.append(r['row_id'])
                    D.save(OUT/'progress.json',dict(stage='hb_calibration_render',unit=unit,unit_index=ui+1,completed=len(completed),total=plan['new_hb_renders'],
                        elapsed_seconds=time.monotonic()-start,last_activity_unix=time.time()),replace=True)
                    print('BATCH',unit,'completed',len(completed),'elapsed',round(time.monotonic()-start,1),flush=True);check()
        assert len(completed)==plan['new_hb_renders'];verify(plan);check()
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
    if D.read(OUT/'gpu_terminal.json')['status']!='COMPLETE':raise ValueError('HB rendering incomplete')
    if (OUT/'CPU_START.json').exists():raise FileExistsError('Single analysis; preserve outputs')
    D.save(OUT/'CPU_START.json',dict(started_unix=time.time(),budget_seconds=300))
    def check():
        if time.monotonic()-start>plan['budget_analysis_cpu_seconds']:raise TimeoutError('300s CPU analysis budget reached')
    try:
        n=len(rows);eligible=np.array([r['eligible_control'] for r in rows]);tags=np.array([r['tag'] for r in rows]);hb=eligible&(tags=='HB')
        pair=np.full((n,13,2),np.nan);current=np.full((n,13,136),np.nan,np.float32);cal_scores={f'fold{f}/{m}':np.full((n,13),np.nan) for f in range(3) for m in METHODS}
        with np.load(SHARED/'calibration_ledger.npz') as z:
            for k in ('unit','config','tag','valid','eligible_control'):np.testing.assert_array_equal(z[k],np.array([r[k] for r in rows[::2]]))
            pair[::2]=z['pair'];current[::2]=z['current']
            for key in cal_scores:cal_scores[key][::2]=z['score/'+key]
        observations=[]
        for i in np.flatnonzero(hb):
            check();r=rows[i];p=OUT/'observations'/f"unit{r['unit']}"/f"c{r['config']}_HB.npz";receipt=D.read(p.with_suffix('.json'))
            assert receipt['row_id']==r['row_id'] and receipt['sha256']==D.sha(p)
            with np.load(p) as z:pair[i]=z['smooth'];current[i]=F.extract_features(z['z1'],z['ambient'],z['nn'])['current']
            observations.append(dict(row_index=int(i),path=str(p.relative_to(ROOT)),sha256=receipt['sha256']))
        x=np.concatenate((pair[hb],current[hb]),axis=-1);assert np.isfinite(x).all();models={k:joblib.load(p) for k,p in C.S.model_paths().items()}
        with threadpool_limits(limits=4):
            for fold in range(3):
                for method in METHODS:
                    check();score=cal_scores[f'fold{fold}/{method}']
                    if method=='original_center':score[hb]=pair[hb].max(-1)
                    else:
                        features=x if method.startswith('raw_') else pair[hb]
                        score[hb]=models[fold,method].predict_proba(features.reshape(-1,features.shape[-1]))[:,1].reshape(hb.sum(),13)
        group,calrecords,individual=group_thresholds(cal_scores,eligible,tags)
        for record in D.read(SHARED/'thresholds.json')['shared']:
            key=record['fold'],record['cap'],record['method'];assert individual['original'][key]==record['threshold'] and group[key]>=record['threshold']
        old_result=D.read(SHARED/'result.json')
        with np.load(SHARED/'ledger.npz') as z:
            metadata={k:z[k] for k in ('unit','config','tag','mode','mirror','family','valid','evaluable','contact','control','deadline','row_id','anchor_id','contact_query')}
            eval_scores={key:z['score/'+key] for key in cal_scores}
            alarms={k[6:]:z[k] for k in z.files if k.startswith('alarm/')}
            timely={k[7:]:z[k] for k in z.files if k.startswith('timely/')}
            post={k[len('timely_excluding_warmup/'):]:z[k] for k in z.files if k.startswith('timely_excluding_warmup/')}
        assert not set(metadata['unit'])&{r['unit'] for r in rows}
        ev=metadata['evaluable'];contact=metadata['contact']&ev;control=metadata['control']&ev;deadline=metadata['deadline'];ne=len(ev)
        groups={'all':np.ones(ne,bool),'family_none':metadata['family']=='none','family_corner':metadata['family']=='corner'}
        groups.update({f'mode{m}':metadata['mode']==m for m in sorted(np.unique(metadata['mode']))});metrics=dict(old_result['metrics'])
        for (fold,cap,method),theta in group.items():
            key=f'fold{fold}/group_calibrated/{cap}/{method}';shared_key=f'fold{fold}/shared_calibrated/{cap}/{method}'
            al=ev[:,None]&(eval_scores[f'fold{fold}/{method}']>=theta);assert not (al&~alarms[shared_key]).any()
            t,p=Q.event_states(al,contact,deadline);assert not (t&~timely[shared_key]).any() and not (p&~post[shared_key]).any()
            alarms[key]=al;timely[key]=t;post[key]=p
            metrics[key]={tag:{name:Q.metrics(al,contact,control,deadline,(metadata['tag']==tag)&mask&ev) for name,mask in groups.items()} for tag in TAGS}
        check()
        np.savez_compressed(OUT/'calibration_ledger.npz',unit=np.array([r['unit'] for r in rows]),config=np.array([r['config'] for r in rows]),tag=tags,
            mode=np.array([r['mode'] for r in rows]),mirror=np.array([r['mirror'] for r in rows]),family=np.array([r['family'] for r in rows]),
            valid=np.array([r['valid'] for r in rows]),evaluable=eligible,eligible_control=eligible,
            contact=np.array([bool(r['contact']) for r in rows]),control=np.array([bool(r['control']) for r in rows]),pair=pair,current=current,
            **{'score/'+k:v for k,v in cal_scores.items()})
        np.savez_compressed(OUT/'ledger.npz',**metadata,**{'score/'+k:v for k,v in eval_scores.items()},**{'alarm/'+k:v for k,v in alarms.items()},
            **{'timely/'+k:v for k,v in timely.items()},**{'timely_excluding_warmup/'+k:v for k,v in post.items()})
        D.save(OUT/'thresholds.json',dict(group=calrecords,original_center_threefold_identical=True,minimality_verified=True))
        D.save(OUT/'readout_inheritance.json',dict(original_calibration_ledger_sha256=D.sha(SHARED/'calibration_ledger.npz'),evaluation_ledger_sha256=D.sha(SHARED/'ledger.npz'),
            evaluation_result_sha256=D.sha(SHARED/'result.json'),model_sha256={f'{f}/{m}':D.sha(p) for (f,m),p in C.S.model_paths().items()},
            original_scores_exact_reuse=True,evaluation_scores_exact_reuse=True,legacy_shared_alarms_exact_reuse=True,ranking_unchanged=True,
            alarm_subset_verified=True,timely_nonincrease_verified=True,no_refit=True))
        D.save(OUT/'result.json',dict(status='POSTHOC_EXPLORATORY_COMPLETE',seconds=time.monotonic()-start,calibration_controls=plan['calibration_controls'],
            evaluation_rows=ne,evaluation_counts=old_result['evaluation_counts'],thresholds=calrecords,metrics=metrics,ranking=old_result['ranking'],observations=observations,
            plan_sha256=D.sha(OUT/'PLAN.json'),ledger_sha256=D.sha(OUT/'ledger.npz'),calibration_ledger_sha256=D.sha(OUT/'calibration_ledger.npz'),alarm_subset_verified=True,timely_nonincrease_verified=True))
        D.save(OUT/'analysis_terminal.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
        for fold in range(3):
            for method in ('original_center','raw_hb_aug'):
                key=f'fold{fold}/group_calibrated/0.025/{method}';print('RESULT',key,{tag:metrics[key][tag]['all'] for tag in TAGS},flush=True)
    except BaseException as exc:
        D.save(OUT/'analysis_terminal.json',dict(status='STOPPED_PARTIAL' if isinstance(exc,TimeoutError) else 'FAILED',seconds=time.monotonic()-start,error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','analyze']);globals()[parser.parse_args().stage]()

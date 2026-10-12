"""Frozen seven-readout check under new physical target heights; no fitting."""
import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
import cnh_double_height_dev as D

ROOT,HERE,WORK=D.ROOT,D.HERE,D.WORK
OUT=WORK/'cnh-height-transfer-dev-20261007'
PARENT=WORK/'cnh-raw-training-support-dev-20261007'
BASE=WORK/'cnh-center-readout-ablation-dev-20261007'
SUPPORT=WORK/'cnh-training-support-dev-20261007'
RAW=WORK/'cnh-raw-radial-information-dev-20261007'
HEIGHTS={'short':[.14,.60],'up':[-.30,.64],'down':[.10,1.04],'tall':[-.30,1.04]}
TAGS=tuple(HEIGHTS)
METHODS=('original_center','pair_only','pair_source_repeat','pair_hb_aug','raw_current','raw_source_repeat','raw_hb_aug')


def model_paths():
    return {(fold,name):path for fold in range(3) for name,path in {
        'pair_only':BASE/'models'/f'in_domain_fold{fold}_pair_only.joblib',
        'pair_source_repeat':SUPPORT/'models'/f'fold{fold}_source_repeat.joblib',
        'pair_hb_aug':SUPPORT/'models'/f'fold{fold}_hb_aug.joblib',
        'raw_current':RAW/'models'/f'fold{fold}_current.joblib',
        'raw_source_repeat':PARENT/'models'/f'fold{fold}_raw_source_repeat.joblib',
        'raw_hb_aug':PARENT/'models'/f'fold{fold}_raw_hb_aug.joblib'}.items()}


def thresholds(result):
    ts={}
    for record in result['fold_records']:
        mode,cap,name=record['key'].split('/')
        if mode=='calibrated' and name in METHODS:
            key=record['fold'],cap,name
            if key in ts:raise ValueError('Duplicate inherited threshold')
            ts[key]=float(record['threshold'])
    assert set(ts)=={(f,c,m) for f in range(3) for c in ('0.025','0.050') for m in METHODS}
    return ts


def checked_manifest():
    manifest=D.read(OUT/'geometry_manifest.json');review=D.read(OUT/'geometry_review.json')
    assert review['passed'] is True and review['manifest_sha256']==D.sha(OUT/'geometry_manifest.json')
    assert manifest['status']=='COMPLETE'
    anchors=manifest['anchors'];assert len(anchors)==72 and len({a['anchor_id'] for a in anchors})==72
    for a in anchors:
        assert set(a['variants'])==set(TAGS)
        for tag,v in a['variants'].items():
            assert isinstance(v['valid'],bool)
            if v['valid']:
                assert [v['boxes'][0]['lo'][1],v['boxes'][0]['hi'][1]]==HEIGHTS[tag]
                assert bool(v['contact'])!=bool(v['control'])
                if v['contact']:assert v['deadline_index'] is not None and 0<=v['deadline_index']<13
    return manifest


def freeze():
    from cnh_double_height_render_dev import source_paths
    manifest=checked_manifest();parent=D.read(PARENT/'PLAN.json');hashes=dict(parent['hashes'])
    hashes.update(D.read(WORK/'cnh-double-height-dev-20261007/PLAN.json')['hashes'])
    paths=[Path(__file__),HERE/'cnh_height_transfer_geometry_dev.py',HERE/'cnh_raw_radial_features_dev.py',
        OUT/'geometry_manifest.json',OUT/'geometry_review.json',PARENT/'PLAN.json',PARENT/'result.json',PARENT/'hb_ledger.npz']
    paths.extend(model_paths().values())
    paths.extend(ROOT/path for path in manifest.get('hashes',{}))
    for a in manifest['anchors']:
        roles=parent['folds'][a['fold']]
        assert a['unit'] in roles['evaluation'] and a['unit'] not in roles['train']+roles['calibration']
        paths.extend(source_paths(a['unit']).values())
    for path in paths:
        path=Path(path);hashes[str(path.relative_to(ROOT))]=D.sha(path)
    thresholds(D.read(PARENT/'result.json'))
    count=sum(v['valid'] for a in manifest['anchors'] for v in a['variants'].values())
    D.save(OUT/'PLAN.json',dict(phase='EXPLORE fixed-model physical height transfer on consumed source units',
        authorization='Continuous algorithm exploration; new bounded physical structures, not replaying old renders',
        budget_gpu_wall_seconds=600,budget_analysis_cpu_seconds=300,threads=4,
        compute='GPU batched raycast and five-seed M3 inference; CPU fixed sklearn scoring after GPU cleanup',
        heights=HEIGHTS,anchors=72,requested_variants=288,valid_variants=count,folds=parent['folds'],methods=list(METHODS),
        parity=dict(hist_exact=True,ambient_atol=1e-6,z1_atol=.002,raw_atol=.005),
        geometry_gate='Complete manifest reviewed before GPU; invalid physical/background/role variants remain missing without replacement. Geometry-review SHA must match.',
        controls='Only target y bounds change. Original x/z/reflectance/background/path/noisy poses/public query errors/photon seed/M3 ensemble retained.',
        input_boundary='Renderer uses scene boxes and source sensor pose to generate observations. M3 receives only z1/nn/qq. Raw learner receives smooth pair2 plus current136 extracted from z1/ambient; no travel, labels or evaluator geometry enter readout features.',
        frozen_readout='All7 existing methods and original-cal2.5/5% fold thresholds inherited. Each anchor only its original OOF model; no fitting, threshold calibration or new-scene matched-FA frontier.',
        primary='raw_hb_aug versus original_center for each new tag, own-valid denominator and all4-tag common-valid subset. Other readouts auxiliary. ActualFA reported, no equalFA claim.',
        decision_check='Each tag has at most36 contacts/36 controls: one event is at least2.78 percentage points and one FA interval at least0.278 percentage points. Primary no-warmup raw_hb_aug timely not lower and actualFA not higher than max on every new tag supports continued candidate work; mixed results restrict its demonstrated scope. This is not statistical noninferiority or promotion.',
        old_hb_reference='Previously trained/exposed HB structure on exactly the selected anchors is a separate reference, including corresponding own-valid/common-valid anchor subsets.',
        metrics='Timely and timely excluding first2 outputs; clear FA[2:12] and warmupFA[:2]. One physical event per valid variant. No three-state outputs without side-query coverage.',
        limitations=['New height geometry but consumed units/simulator, no fresh-scene or safety confirmation',
                    'HB augmentation has exposed a related height structure during training','Invalid variants reduce denominators; no score-driven replacement'],
        hashes=hashes))
    print('FROZEN',count,'valid new renders',flush=True)


def verify(plan):
    for path,expected in plan['hashes'].items():
        if D.sha(ROOT/path)!=expected:raise ValueError('Frozen input changed: '+path)


def run():
    plan=D.read(OUT/'PLAN.json');verify(plan);manifest=checked_manifest()
    if (OUT/'GPU_START.json').exists():raise FileExistsError('Single GPU run; preserve existing progress/terminal')
    start=time.monotonic();D.save(OUT/'GPU_START.json',dict(started_unix=time.time(),pid=os.getpid(),budget_seconds=600))
    completed=[];runner=None
    def check():
        if time.monotonic()-start>plan['budget_gpu_wall_seconds']:raise TimeoutError('600s GPU wall budget reached')
    try:
        from cnh_double_height_render_dev import DoubleHeightRunner,load_source
        if not plan['valid_variants']:raise ValueError('No legal variants; do not allocate GPU')
        runner=DoubleHeightRunner(OUT)
        with runner:
            first=manifest['anchors'][0];source=load_source(first['unit'],first['config'])
            parity=runner.anchor_parity(source,**{k:v for k,v in plan['parity'].items() if k!='hist_exact'})
            D.save(OUT/'parity.json',parity)
            if not parity['passed']:raise ValueError('Unchanged-anchor parity failed; stop before new renders')
            print('PARITY passed',flush=True)
            for ai,a in enumerate(manifest['anchors']):
                check();tick=time.monotonic();valid_tags=[tag for tag in TAGS if a['variants'][tag]['valid']]
                items=[]
                if valid_tags:
                    source=load_source(a['unit'],a['config'])
                    for tag in valid_tags:
                        items.append(runner.render_observation(source,a['variants'][tag]['boxes'],tag))
                    runner.infer_observations(items)
                target=OUT/'anchors'/a['anchor_id']
                for tag,item in zip(valid_tags,items):
                    for key in ('nn','qq'):np.testing.assert_array_equal(item[key],source[key])
                    target.mkdir(parents=True,exist_ok=True);dest=target/f'{tag}.npz'
                    if dest.exists():raise FileExistsError(dest)
                    np.savez_compressed(dest,**{k:item[k] for k in ('hist','ambient','z1','nn','qq','raw','smooth')})
                    D.save(dest.with_suffix('.json'),dict(tag=tag,unit=a['unit'],config=a['config'],photon_seed=source['photon_seed'],
                        backend=item['backend'],sha256=D.sha(dest)))
                    completed.append(a['anchor_id']+'/'+tag)
                D.save(OUT/'progress.json',dict(stage='render_infer',anchor=ai+1,anchors=72,completed=len(completed),total=plan['valid_variants'],
                    elapsed_seconds=time.monotonic()-start,last_activity_unix=time.time()),replace=True)
                print('ANCHOR',ai+1,72,a['anchor_id'],'variants',len(valid_tags),round(time.monotonic()-tick,2),'s',flush=True);check()
        assert len(completed)==plan['valid_variants'];verify(plan);check()
        D.save(OUT/'gpu_terminal.json',dict(status='COMPLETE',completed=completed,seconds=time.monotonic()-start,cleanup=runner.cleanup_receipt,pid=os.getpid()))
    except BaseException as exc:
        D.save(OUT/'gpu_terminal.json',dict(status='STOPPED_PARTIAL' if isinstance(exc,TimeoutError) else 'FAILED',completed=completed,
            seconds=time.monotonic()-start,error=repr(exc),traceback=traceback.format_exc(),cleanup=runner.cleanup_receipt if runner is not None else None,pid=os.getpid()))
        raise


def evaluation_layout(manifest):
    rows=[]
    for ai,a in enumerate(manifest['anchors']):
        common=all(a['variants'][tag]['valid'] for tag in TAGS)
        for tag in TAGS:
            v=a['variants'][tag];valid=v['valid'];deadline=v.get('deadline_index')
            rows.append(dict(anchor_index=ai,anchor_id=a['anchor_id'],unit=a['unit'],config=a['config'],fold=a['fold'],tag=tag,
                valid=valid,common_valid=common,contact=v.get('contact'),control=v.get('control'),
                physical_contact=v.get('physical_contact'),physical_control=v.get('physical_control'),
                deadline=int(deadline) if valid and deadline is not None else -1,contact_query=v.get('contact_query',[False,False]),
                invalid_reasons=v.get('reasons',[])))
    return rows


def analyze():
    # This stage runs separately in the CPU interpreter, after GPU resources close.
    import cnh_direction_information_dev as I
    import cnh_querywise_calibration_dev as Q
    import cnh_raw_radial_features_dev as F
    import joblib
    from threadpoolctl import threadpool_limits
    start=time.monotonic();plan=D.read(OUT/'PLAN.json');verify(plan);manifest=checked_manifest()
    if D.read(OUT/'gpu_terminal.json')['status']!='COMPLETE':raise ValueError('GPU incomplete; retain partial, no full evaluation')
    if (OUT/'CPU_START.json').exists():raise FileExistsError('Single analysis; preserve prior partial outputs')
    D.save(OUT/'CPU_START.json',dict(started_unix=time.time(),budget_seconds=300))
    def check():
        if time.monotonic()-start>plan['budget_analysis_cpu_seconds']:raise TimeoutError('300s CPU analysis budget reached')
    try:
        ts=thresholds(D.read(PARENT/'result.json'));models={k:joblib.load(p) for k,p in model_paths().items()}
        rows=evaluation_layout(manifest);n=len(rows);valid=np.array([r['valid'] for r in rows]);common=np.array([r['common_valid'] for r in rows])
        contact=np.array([bool(r['contact']) for r in rows]);control=np.array([bool(r['control']) for r in rows]);deadline=np.array([r['deadline'] for r in rows]);tag=np.array([r['tag'] for r in rows])
        folds=np.array([r['fold'] for r in rows]);scores={m:np.full((n,13),np.nan) for m in METHODS};pairs=np.full((n,13,2),np.nan);currents=np.full((n,13,136),np.nan,np.float32)
        observations=[]
        with threadpool_limits(limits=4):
            for i,r in enumerate(rows):
                check()
                if not r['valid']:continue
                p=OUT/'anchors'/r['anchor_id']/f"{r['tag']}.npz";receipt=D.read(p.with_suffix('.json'))
                assert D.sha(p)==receipt['sha256'];observations.append(dict(row=i,path=str(p.relative_to(ROOT)),sha256=receipt['sha256']))
                with np.load(p) as z:
                    pair=np.asarray(z['smooth'],np.float64);features=F.extract_features(z['z1'],z['ambient'],z['nn'])
                current=features['current'];x=np.concatenate((pair,current),axis=-1)
                assert pair.shape==(13,2) and x.shape==(13,138) and np.isfinite(x).all()
                pairs[i]=pair;currents[i]=current;scores['original_center'][i]=pair.max(-1)
                for method in METHODS[1:]:scores[method][i]=models[r['fold'],method].predict_proba(x if method.startswith('raw_') else pair)[:,1]
        alarms={};metrics={};timely={};post={};c=contact&valid;ct=control&valid
        for cap in ('0.025','0.050'):
            for method,score in scores.items():
                key=f'calibrated/{cap}/{method}';theta=np.array([ts[f,cap,method] for f in folds])
                al=valid[:,None]&(score>=theta[:,None]);alarms[key]=al;timely[key],post[key]=Q.event_states(al,c,deadline)
                metrics[key]={t:{scope:Q.metrics(al,c,ct,deadline,(tag==t)&mask) for scope,mask in [('own_valid',valid),('common_valid',common)]} for t in TAGS}
        # Old HB is a trained-structure reference, never pooled with new tags.
        old_metrics={};old_path=PARENT/'hb_ledger.npz'
        with np.load(old_path) as z:
            old_hb=np.flatnonzero(z['tag']=='HB');ids=z['anchor_id'][old_hb]
            np.testing.assert_array_equal(ids,np.array([a['anchor_id'] for a in manifest['anchors']]))
            np.testing.assert_array_equal(z['fold'][old_hb],folds[::4])
            oc,on,od=[z[k][old_hb] for k in ('contact','control','deadline')]
            for key in alarms:
                old_al=z['alarm/'+key][old_hb]
                old_metrics[key]={'all_old_anchors':Q.metrics(old_al,oc,on,od,np.ones(len(ids),bool)),
                    'common_valid_new_anchors':Q.metrics(old_al,oc,on,od,common[::4]),
                    'own_valid_new_anchors':{t:Q.metrics(old_al,oc,on,od,valid[j::4]) for j,t in enumerate(TAGS)}}
        check();D.save(OUT/'rows.json',rows)
        np.savez_compressed(OUT/'ledger.npz',unit=np.array([r['unit'] for r in rows]),config=np.array([r['config'] for r in rows]),fold=folds,
            anchor_id=np.array([r['anchor_id'] for r in rows]),tag=tag,valid=valid,common_valid=common,contact=contact,control=control,deadline=deadline,
            contact_query=np.array([r['contact_query'] for r in rows],bool),pair=pairs,current=currents,
            physical_contact=np.array([r['physical_contact'] for r in rows],bool),physical_control=np.array([r['physical_control'] for r in rows],bool),
            **{'score/'+k:v for k,v in scores.items()},**{'alarm/'+k:v for k,v in alarms.items()},
            **{'timely/'+k:v for k,v in timely.items()},**{'timely_excluding_warmup/'+k:v for k,v in post.items()})
        D.save(OUT/'readout_inheritance.json',dict(source_result_sha256=D.sha(PARENT/'result.json'),
            thresholds=[dict(fold=f,cap=cap,method=m,threshold=v) for (f,cap,m),v in ts.items()],
            model_sha256={f'{f}/{m}':D.sha(p) for (f,m),p in model_paths().items()},no_fitting=True))
        D.save(OUT/'result.json',dict(status='EXPLORATORY_COMPLETE',seconds=time.monotonic()-start,requested_variants=n,valid_variants=int(valid.sum()),
            common_valid_anchors=int(common[::4].sum()),missing_variants=int((~valid).sum()),metrics=metrics,old_hb_reference=old_metrics,
            observations=observations,plan_sha256=D.sha(OUT/'PLAN.json'),ledger_sha256=D.sha(OUT/'ledger.npz')))
        D.save(OUT/'analysis_terminal.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
        for method in ('original_center','raw_hb_aug'):
            print('RESULT',method,metrics['calibrated/0.025/'+method],flush=True)
    except BaseException as exc:
        D.save(OUT/'analysis_terminal.json',dict(status='STOPPED_PARTIAL' if isinstance(exc,TimeoutError) else 'FAILED',seconds=time.monotonic()-start,error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','analyze']);globals()[parser.parse_args().stage]()

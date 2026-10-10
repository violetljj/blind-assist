"""Frozen task-cost S: consumed negative pool -> seal -> fresh tradeoff hold."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from types import SimpleNamespace
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
import cnh_counterfactual_common_dev as C
import cnh_counterfactual_eval_dev as E
import cnh_graded_corridor_eval_dev as Corr
import cnh_frozen_e2e_tof_20261010 as F
import cnh_task_cost_retrain_20261010 as R
import cnh_task_cost_train_20261010 as T

OUT=C.ROOT/'artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010'
PRIOR=OUT.parent/'cnh-task-cost-retrain-dev-20261010'
CONFIRM=OUT.parent/'cnh-s-ensemble-confirm-dev-20261010'
POOL=('cnh-cost-v2-holdout-dev-20261010','cnh-task-cost-retrain-dev-20261010','cnh-s-ensemble-confirm-dev-20261010')
ARMS=('E','S955','S956','S957','Sensemble')
def save(name,obj): F.save_new(OUT/name,obj)
def spec():
    OUT.mkdir(parents=True,exist_ok=True)
    old=C.read(CONFIRM/'frozen_spec.json')
    binding={**old['inherited'],**old['feature_sources'],**old['models']}
    for p,h in binding.items(): assert C.sha(p)==h,p
    save('frozen_spec.json',dict(task='CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010',lane='EXPLORE',
        frozen_dependencies=binding,models=old['models'],feature_sources=old['feature_sources'],
        feature_contract='47: smooth ordinary margin/trend/residual (3), current peak22 and missing flags22; corresponding ordinary seed; ensemble mean float32',
        policy='User decision2026-10-10: S ensemble simulation default candidate; old5 strong per slot; new light only. Prior confirm b/c failures retained.',
        selection='Only consumed pool pass/clear; lowest canonical tau with far+clear any-notify clip rate <=old5+.03; all ties; no contact utility; same recipe E/S seeds.',
        budgets=dict(render_forward_gpu_seconds=1200,evaluation_audit_integration_command_wall_seconds=1200),
        hold=dict(physical_scenes=1536,K=2,families=4),bootstrap=dict(replicates=2000,seed=20261010),
        claim_ceiling='Finite AABB synthetic Development; not App, hardware or safety evidence',created_utc=datetime.now(timezone.utc).isoformat()))
    save('protocol_snapshot.json',dict(**C.read(OUT/'frozen_spec.json'),
        decision_check='Existing baseline misses permit positive paired net; event denominator1024, perheight512; FCclipden1536 makes one clip .0651pp; no binary pass/fail. Pool reuse explicitly authorized; hold never selects.',
        both_curve='Original both955 extra grades (including original strong additions) removed by frozen margin; tau0 exact frozen both955. S/E curves add light only.'))
def identity():
    specdata=C.read(OUT/'frozen_spec.json')
    for p,h in specdata['frozen_dependencies'].items():assert C.sha(p)==h,p
    return specdata
def pool():
    began=time.monotonic();identity();blocks=[];allrows=[];bindings={};cohorts=[]
    cuts=C.read(F.THRESHOLDS);jointcuts=C.read(F.JOINT/'calibrations.json')
    for dataset in POOL:
        src=OUT.parent/dataset
        rowsby=C.read(src/'scene_rows.json');bindings[str(src/'scene_rows.json')]=C.sha(src/'scene_rows.json')
        for split in ('cal','hold'):
            folder=src/'data'/split;rows=rowsby[split]
            ids=np.array([i for i,r in enumerate(rows) if r['placement'] in ('pass','clear')])
            with np.load(folder/'geometry.npz') as g:category=g['category'][ids]
            assert not (category=='contact').any()
            with np.load(folder/'scores.npz') as raw:
                ordinary=raw['ordinary_raw'][:,ids];current=raw['current'][ids].astype(float)
                valid=raw['current_valid'][ids]&np.isfinite(current)
                old5=(2*E.old_fusion(E.smooth(raw['m3_raw'][ids]),E.smooth(raw['local_raw'][ids]))).astype(np.int8)
            spatial=np.concatenate((np.where(valid,current,np.nan),(~valid).astype(float)),-1)
            features=np.array([np.concatenate((Corr.build_score_features(ordinary[si],E.smooth(ordinary[si]),
                cuts[str(seed)]['single']),spatial),-1) for si,seed in enumerate(T.SEEDS)],np.float32)
            learned=T.predict_s(PRIOR,features)
            if (folder/'frozen_readout.npz').exists():
                readout=folder/'frozen_readout.npz'
                with np.load(readout) as a:joint=a['joint_scores'][:,ids];np.testing.assert_array_equal(old5,a['old5'][ids])
            elif split=='cal':
                readout=src/'cal_scores_readout.npz'
                with np.load(readout) as a:joint=a['joint_scores'][:,ids];np.testing.assert_array_equal(old5,a['old5'][ids])
            else:
                readout=src/'hold_grades_notifications.npz'
                with np.load(readout) as a:joint=a['joint_scores'][:,ids]
            if (folder/'task_scores.npz').exists():
                with np.load(folder/'task_scores.npz') as a:
                    keys=list(a['arm_keys']);scores=a['light_scores']
                    for si,seed in enumerate(T.SEEDS):np.testing.assert_array_equal(learned[si],scores[keys.index('S'+str(seed)[-3:]),ids])
                    np.testing.assert_array_equal(learned.mean(0),scores[keys.index('Sensemble'),ids])
                bindings[str(folder/'task_scores.npz')]=C.sha(folder/'task_scores.npz')
            om=E.smooth(ordinary)-np.array([cuts[str(s)]['single'] for s in T.SEEDS])[:,None,None,None,None]
            jm=joint-np.array([jointcuts[f'{s}/score_current/c15_p64']['theta'] for s in T.SEEDS])[:,None,None,None,None]
            eraw=np.maximum(om.mean(0),jm.mean(0))
            blocks.append(dict(light_scores=np.array([np.where(eraw>=0,eraw,-np.inf),*learned,learned.mean(0)]),old5=old5,category=category))
            for i in ids:allrows.append(dict(rows[i],source_dataset=dataset,source_split=split,source_scene_index=int(i)))
            for f in (folder/'geometry.npz',folder/'scores.npz',readout):bindings[str(f)]=C.sha(f)
            cohorts.append(dict(dataset=dataset,split=split,negative_scenes=len(ids),negative_clips=len(ids)*2))
            if time.monotonic()-began>170:raise TimeoutError('Pool forward CPU170s sublimit')
    physical=[C.dumps(r['physical_key']) if hasattr(C,'dumps') else __import__('json').dumps(r['physical_key'],sort_keys=True) for r in allrows]
    assert len(set(physical))==len(physical),'Duplicate physical scenes in pool'
    np.savez_compressed(OUT/'pool_scores.npz',arm_keys=np.array(ARMS),
        light_scores=np.concatenate([b['light_scores'] for b in blocks],axis=1),
        old5=np.concatenate([b['old5'] for b in blocks]),category=np.concatenate([b['category'] for b in blocks]))
    save('pool_rows.json',allrows)
    save('pool_input_receipt.json',dict(seconds=time.monotonic()-began,cohorts=cohorts,negative_scenes=len(allrows),
        bindings=bindings,contact_utility_access=False,models_sha256=C.read(OUT/'frozen_spec.json')['models'],
        archive_sha256=C.sha(OUT/'pool_scores.npz'),rows_sha256=C.sha(OUT/'pool_rows.json'),
        feature_recomputation='Same frozen functions and float32 contract, negative rows only; cached retrain/confirm S scores exact bytevalue equality'))
def seal():
    import cnh_s_ensemble_tradeoff_metrics_20261010 as V
    began=time.monotonic();identity()
    def check():
        if time.monotonic()-began>120:raise TimeoutError('Pool calibration CPU120s sublimit')
    result=V.calibrate(OUT,C.read(OUT/'pool_rows.json'),check)
    save('calibration_receipt.json',dict(**result,seconds=time.monotonic()-began,status='SEALED'))
def freeze():
    import cnh_s_ensemble_tradeoff_source_20261010 as S
    import cnh_s_ensemble_tradeoff_metrics_20261010 as V
    identity();assert (OUT/'sealed_calibration.json').exists()
    paths=[Path(__file__),Path(S.__file__),Path(V.__file__),OUT/'frozen_spec.json',OUT/'protocol_snapshot.json',
        OUT/'PLAN.json',OUT/'scene_rows.json',OUT/'inventory.json',OUT/'data/hold/geometry.npz',OUT/'sealed_calibration.json',OUT/'pool_input_receipt.json']
    save('execution_manifest.json',dict(frozen_spec=identity(),new_sources={str(p):C.sha(p) for p in paths},training=0,protected_access=0,hold_before_seal=False,
        frozen_utc=datetime.now(timezone.utc).isoformat()))
    candidate()
def candidate():
    frozen=identity();sealdata=C.read(OUT/'sealed_calibration.json')
    def relative(p):
        path=Path(p)
        if path.is_relative_to(C.ROOT):return path.relative_to(C.ROOT).as_posix()
        # Frozen dependencies sometimes record the F: physical junction target.
        return 'artifacts.local/'+path.resolve().relative_to((C.ROOT/'artifacts.local').resolve()).as_posix()
    save('candidate_manifest.json',dict(schema_version=1,candidate='ToF S ensemble default candidate',
        authority='Explicit user decision2026-10-10; simulation only; old confirm b/c failures retained',
        scope='Frozen synthetic readout candidate for future bench/App integration; no integration performed',
        models=[dict(path=relative(p),sha256=h) for p,h in frozen['models'].items()],
        features=dict(dimensions=47,contract=frozen['feature_contract'],
            implementation=[dict(path=relative(p),sha256=h) for p,h in frozen['feature_sources'].items()],
            function='cnh_task_cost_retrain_20261010.features_and_labels; cnh_task_cost_train_20261010.predict_s',
            dtype='float32 per seed score and float32 arithmetic mean, threshold comparison preserves this rounded mean'),
        ensemble=dict(seeds=list(T.SEEDS),aggregation='Arithmetic mean of corresponding perheight scores before threshold'),
        threshold=sealdata['calibrations']['Sensemble'],
        calibration=dict(path=relative(OUT/'sealed_calibration.json'),sha256=C.sha(OUT/'sealed_calibration.json'),
            pool_sources=list(POOL),only_negative=True,unit='far+clear any-notify clip rate increment',cap_increment_pp=3.0),
        notifier=dict(gap=1,nominal_frames=list(range(3,16)),state='per query grade peak; reset after two quiet frames; emit only onset or grade upgrade',
            joint='Same frame HEAD/BODY maximum emitted grade',implementation=relative(Path(F.__file__)),sha256=C.sha(Path(F.__file__))),
        strong_source=dict(arm='original old5',per_slot='grade2 iff originalold5 grade2; new scores only add grade1',
            frozen_dependencies=[dict(path=relative(p),sha256=h) for p,h in frozen['frozen_dependencies'].items() if p not in frozen['models'] and p not in frozen['feature_sources']]),
        weights=dict(near_pass_max_gap_m=.10,near_light=.25,near_strong=1,far_clear_any=1),
        hold_selection=False,training=0,protected_access=0))
def check_execution():
    identity();m=C.read(OUT/'execution_manifest.json')
    for p,h in m['new_sources'].items():
        if Path(p)==Path(__file__) and C.sha(p)!=h:
            repair=C.read(OUT/'source_serialization_repair_receipt.json')
            assert repair['original_execution_manifest_sha256']==C.sha(OUT/'execution_manifest.json')
            assert C.sha(OUT/repair['original_controller_snapshot'])==h
            assert C.sha(p)==repair['delivery_controller_sha256']
        else:assert C.sha(p)==h,p
def data():
    import cnh_s_ensemble_tradeoff_source_20261010 as S
    began=time.monotonic();check_execution()
    def check():
        if time.monotonic()-began>=1180:raise TimeoutError('Render+forward GPU1200s cap,20s reserve')
    F.S=SimpleNamespace(SPLITS=('hold',),K=S.K,sampling_seed=S.sampling_seed)
    detail=F.scientific(OUT,{'hold':C.read(OUT/'scene_rows.json')['hold']},check)
    (OUT/'render_receipt.json').rename(OUT/'render_hold_receipt.json');check()
    R.OUT=OUT;supervision=R.features_and_labels('hold');folder=OUT/'data/hold'
    with np.load(folder/'frozen_readout.npz') as a:fr={k:a[k].copy() for k in a.files}
    with np.load(folder/'scores.npz') as a:ordinary=E.smooth(a['ordinary_raw'])
    cuts=C.read(F.THRESHOLDS);jointcuts=C.read(F.JOINT/'calibrations.json')
    om=ordinary-np.array([cuts[str(s)]['single'] for s in T.SEEDS])[:,None,None,None,None]
    jm=fr['joint_scores']-np.array([jointcuts[f'{s}/score_current/c15_p64']['theta'] for s in T.SEEDS])[:,None,None,None,None]
    eraw=np.maximum(om.mean(0),jm.mean(0));learned=T.predict_s(PRIOR,np.load(folder/'S_features.npy',mmap_mode='r'))
    np.savez_compressed(folder/'task_scores.npz',arm_keys=np.array(ARMS),
        light_scores=np.array([np.where(eraw>=0,eraw,-np.inf),*learned,learned.mean(0)]),
        **{k:fr[k] for k in ('old5','m3','both','category','margin')})
    save('data_hold_receipt.json',dict(seconds=time.monotonic()-began,status='COMPLETE',detail=detail,supervision=supervision,
        training=0,cal_seal_sha256=C.sha(OUT/'sealed_calibration.json'),outputs_sha256={n:C.sha(folder/n) for n in ('scores.npz','S_features.npy','frozen_readout.npz','task_scores.npz')}))
def evaluate():
    import cnh_s_ensemble_tradeoff_metrics_20261010 as V
    began=time.monotonic();check_execution()
    def check():
        if time.monotonic()-began>160:raise TimeoutError('Evaluation CPU160s sublimit')
    result=V.evaluate(OUT,C.read(OUT/'scene_rows.json')['hold'],check)
    save('evaluation_receipt.json',dict(**result,seconds=time.monotonic()-began,status='COMPLETE'))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('spec','pool','seal','freeze','candidate','data','evaluate'));args=parser.parse_args()
    for key in ('TEMP','TMP','CUPY_CACHE_DIR'):
        path=OUT/'runtime'/key.lower();path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    began=time.monotonic()
    try:
        with threadpool_limits(limits=2):globals()[args.stage]()
        print('TRADEOFF_STAGE',args.stage,round(time.monotonic()-began,3),flush=True)
    except BaseException as err:
        save(f'failure_{args.stage}_{time.time_ns()}.json',dict(seconds=time.monotonic()-began,error=repr(err),traceback=traceback.format_exc()));raise

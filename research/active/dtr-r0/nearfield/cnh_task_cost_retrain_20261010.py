"""Task-cost EXPLORE controller; fresh train -> cal seal -> fresh hold.

Only observation archives and public transforms enter feature/model inference.
Author geometry is isolated in supervision and descriptive evaluation.
"""
import argparse
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
import cnh_counterfactual_common_dev as C
import cnh_counterfactual_data_dev as D
import cnh_counterfactual_eval_dev as E
import cnh_graded_corridor_eval_dev as Corr
import cnh_frozen_e2e_tof_20261010 as F
import cnh_task_cost_source_20261010 as S
import cnh_task_cost_train_20261010 as T
import cnh_task_cost_metrics_20261010 as V

OUT=C.ROOT/'artifacts.local/work/cnh-task-cost-retrain-dev-20261010'


def save(name,data):F.save_new(OUT/name,data)


def freeze():
    began=time.monotonic()
    inherited=C.read(F.DEFAULT_OUT/'input_hashes_frozen.json')
    for p,digest in inherited.items():assert C.sha(p)==digest,p
    paths=[Path(__file__),Path(S.__file__),Path(T.__file__),Path(V.__file__),
        OUT/'PLAN.json',OUT/'training_recipe.json',OUT/'scene_rows.json',OUT/'inventory.json',OUT/'protocol_snapshot.md']
    paths += [OUT/'data'/split/'geometry.npz' for split in S.SPLITS]
    save('execution_manifest.json',dict(inherited=inherited,new_sources={str(p):C.sha(p) for p in paths},
        training_authorized=True,hold_before_seal=False,budgets=dict(render_gpu=1200,train_gpu=3600,B_gpu=2700,evaluation_cpu=1500)))
    save('freeze_receipt.json',dict(status='FROZEN_BEFORE_RENDER',seconds=time.monotonic()-began,
        manifest_sha256=C.sha(OUT/'execution_manifest.json')))


def identity():
    m=C.read(OUT/'execution_manifest.json')
    for p,digest in {**m['inherited'],**m['new_sources']}.items():assert C.sha(p)==digest,p
    return m


def charged(pattern):return sum(C.read(p).get('seconds',0) for p in OUT.glob(pattern))


def data_budget(began):
    # Conservatively charge whole CUDA data/forward/cache commands, CPU included.
    if charged('data_*_receipt.json')+charged('score_*_receipt.json')+charged('failure_data*.json')+time.monotonic()-began>=1190:
        raise TimeoutError('Data/forward GPU allocation cap reached')


def features_and_labels(split):
    folder=OUT/'data'/split
    scored=F.score_split(OUT,split)
    with np.load(folder/'scores.npz',allow_pickle=False) as raw:
        ordinary=raw['ordinary_raw'];current=raw['current'].astype(float)
        valid=raw['current_valid'] & np.isfinite(current)
    spatial=np.concatenate((np.where(valid,current,np.nan),(~valid).astype(float)),-1)
    ths=C.read(F.THRESHOLDS)
    features=np.array([np.concatenate((Corr.build_score_features(ordinary[si],E.smooth(ordinary[si]),
        ths[str(seed)]['single']),spatial),-1) for si,seed in enumerate(T.SEEDS)],np.float32)
    np.save(folder/'S_features.npy',features)
    with np.load(folder/'geometry.npz',allow_pickle=False) as geo:
        category=geo['category'];support=geo['target_support']
    rows=C.read(OUT/'scene_rows.json')[split]
    labels,mask,weights,counts=T.supervision(category,support,scored['old5']>0,[r['lateral_gap_m'] for r in rows])
    near=np.array([r['placement']=='pass' and r['lateral_gap_m']<=.1 for r in rows])
    negative_weight=np.broadcast_to(np.where(near,.25,1.)[:,None,None,None],labels.shape)
    np.savez_compressed(folder/'labels.npz',positive=labels.astype(bool),valid=mask,negative_weight=negative_weight,
        labels=labels,weights=weights)
    np.savez_compressed(folder/'frozen_readout.npz',**scored,category=category)
    return counts


def data(split):
    began=time.monotonic();identity()
    if split=='hold':
        if not (OUT/'sealed_calibration.json').exists():raise ValueError('Seal cal before hold rendering')
        verify_models()
    check=lambda:data_budget(began)
    F.S=SimpleNamespace(SPLITS=(split,),K=S.K,sampling_seed=S.sampling_seed)
    rows={split:C.read(OUT/'scene_rows.json')[split]}
    detail=F.scientific(OUT,rows,check)
    (OUT/'render_receipt.json').rename(OUT/f'render_{split}_receipt.json')
    check();supervision=features_and_labels(split)
    import cnh_aligned_boundary_dev as Boundary
    _,_,_,normal=Boundary.imports()
    folder=OUT/'data'/split
    hist=np.load(folder/'hist.npy',mmap_mode='r');ambient=np.load(folder/'ambient.npy')
    count=len(hist)*S.K
    z=normal(hist.reshape(count,16,8,8,16),np.broadcast_to(ambient,(count,*ambient.shape)))
    with np.load(folder/'geometry.npz') as geom:
        cache=T.prepare_native_cache(OUT,split,z,geom['sensor'],geom['public_query'],check)
    del z,hist
    save(f'data_{split}_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,
        detail=detail,cache=cache,supervision=supervision,training=0,
        cal_seal_sha256=C.sha(OUT/'sealed_calibration.json') if split=='hold' else None,
        outputs_sha256={n:C.sha(folder/n) for n in ('scores.npz','S_features.npy','labels.npz','frozen_readout.npz')}))


def training_inputs():
    with np.load(OUT/'data/train/labels.npz') as a:return a['labels'].copy(),a['weights'].copy()


def checkpoints():return {s:F.OLD/f'models/ordinary_seed{s}.pt' for s in T.SEEDS}


def profile():
    identity();labels,weights=training_inputs()
    T.profile_b(OUT,OUT/'native_cache/train.npy',OUT/'native_cache/train_length.npy',
        checkpoints()[T.SEEDS[0]],labels,weights)


def fit_s():
    began=time.monotonic();identity()
    if not (OUT/'B_profile_receipt.json').exists():raise ValueError('Profile before long fit')
    labels,weights=training_inputs()
    def check():
        if time.monotonic()-began>=180:raise TimeoutError('S fitting CPU sub-budget')
    T.train_s(OUT,np.load(OUT/'data/train/S_features.npy',mmap_mode='r'),labels,weights,check)


def fit_b():
    identity();labels,weights=training_inputs()
    prior=C.read(OUT/'B_profile_receipt.json')['seconds']
    T.train_b(OUT,OUT/'native_cache/train.npy',OUT/'native_cache/train_length.npy',
        checkpoints(),labels,weights,total_seconds=2690,prior_seconds=prior)


def bind_models():
    identity()
    paths=list((OUT/'models').glob('*.pickle'))+list((OUT/'models').glob('*.pt'))
    if not (OUT/'S_training_receipt.json').exists() or not (OUT/'B_training_receipt.json').exists():
        raise ValueError('Both training jobs must finish/stop explicitly')
    save('trained_models_manifest.json',dict(models={str(p):C.sha(p) for p in paths},
        receipts={n:C.sha(OUT/n) for n in ('S_training_receipt.json','B_training_receipt.json','B_profile_receipt.json')},
        plan_sha256=C.sha(OUT/'PLAN.json'),manifest_sha256=C.sha(OUT/'execution_manifest.json')))


def verify_models():
    m=C.read(OUT/'trained_models_manifest.json')
    for p,digest in m['models'].items():assert C.sha(p)==digest,p
    for p,digest in m['receipts'].items():assert C.sha(OUT/p)==digest,p


def score(split):
    began=time.monotonic();identity();verify_models()
    if split=='hold' and not (OUT/'sealed_calibration.json').exists():raise ValueError('Cal seal missing')
    check=lambda:data_budget(began)
    folder=OUT/'data'/split
    with np.load(folder/'frozen_readout.npz') as a:fr={k:a[k].copy() for k in a.files}
    with np.load(folder/'scores.npz') as raw:ordinary=E.smooth(raw['ordinary_raw'])
    cuts=C.read(F.THRESHOLDS);jointcuts=C.read(F.JOINT/'calibrations.json')
    ordinary_margin=ordinary-np.array([cuts[str(s)]['single'] for s in T.SEEDS])[:,None,None,None,None]
    joint_margin=fr['joint_scores']-np.array([jointcuts[f'{s}/score_current/c15_p64']['theta'] for s in T.SEEDS])[:,None,None,None,None]
    raw_ensemble=np.maximum(ordinary_margin.mean(0),joint_margin.mean(0))
    raw_seeds=np.maximum(ordinary_margin,joint_margin)
    keys=['E','E955','E956','E957']
    diagnostic=[raw_ensemble,*raw_seeds]
    light=[np.where(a>=0,a,-np.inf) for a in diagnostic]
    learned=T.predict_s(OUT,np.load(folder/'S_features.npy',mmap_mode='r'))
    keys += ['S955','S956','S957','Sensemble'];light += [*learned,learned.mean(0)]
    diagnostic += [*learned,learned.mean(0)]
    backbone=T.predict_b(OUT,OUT/f'native_cache/{split}.npy',OUT/f'native_cache/{split}_length.npy',fr['old5'].shape,check)
    jobs=C.read(OUT/'B_training_receipt.json')['jobs']
    for si,seed in enumerate(T.SEEDS):
        job=next(j for j in jobs if j['seed']==seed)
        if job['status']=='COMPLETE':
            keys.append(f'B{str(seed)[-3:]}');light.append(backbone[si]);diagnostic.append(backbone[si])
    if all(j['status']=='COMPLETE' for j in jobs):
        keys.append('Bensemble');light.append(backbone.mean(0));diagnostic.append(backbone.mean(0))
    np.savez_compressed(folder/'task_scores.npz',arm_keys=np.array(keys),light_scores=np.array(light),
        diagnostic_scores=np.array(diagnostic),**{k:fr[k] for k in ('old5','m3','both','category')})
    save(f'score_{split}_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,
        arms=keys,models_manifest_sha256=C.sha(OUT/'trained_models_manifest.json'),
        sha256=C.sha(folder/'task_scores.npz'),cal_seal_sha256=C.sha(OUT/'sealed_calibration.json') if split=='hold' else None))


def evaluate(seal):
    began=time.monotonic();identity();verify_models()
    def check():
        used=charged('calibration_receipt.json')+charged('evaluation_receipt.json')+charged('S_training_receipt.json')
        if used+time.monotonic()-began>=780:raise TimeoutError('Evaluation CPU sub-budget; 720s reserved for preparation/audit/integration')
    rows=C.read(OUT/'scene_rows.json')
    detail=V.calibrate(OUT,rows['cal'],check) if seal else V.evaluate(OUT,rows,check)
    save('calibration_receipt.json' if seal else 'evaluation_receipt.json',
        {**detail,'status':'SEALED' if seal else 'COMPLETE','seconds':time.monotonic()-began,
        'models_manifest_sha256':C.sha(OUT/'trained_models_manifest.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','data','profile','fit_s','fit_b','bind','score','seal','evaluate'))
    p.add_argument('--split',choices=S.SPLITS);args=p.parse_args();began=time.monotonic()
    for key in ('TEMP','TMP','CUPY_CACHE_DIR'):
        path=OUT/'runtime'/key.lower();path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    try:
        with threadpool_limits(limits=2):
            if args.stage=='freeze':freeze()
            elif args.stage=='data':data(args.split)
            elif args.stage=='profile':profile()
            elif args.stage=='fit_s':fit_s()
            elif args.stage=='fit_b':fit_b()
            elif args.stage=='bind':bind_models()
            elif args.stage=='score':score(args.split)
            else:evaluate(args.stage=='seal')
        print('TASK_STAGE',args.stage,args.split,round(time.monotonic()-began,3),flush=True)
    except BaseException as err:
        save(f'failure_{args.stage}_{time.time_ns()}.json',dict(seconds=time.monotonic()-began,
            stage=args.stage,split=args.split,error=repr(err),traceback=traceback.format_exc()))
        raise

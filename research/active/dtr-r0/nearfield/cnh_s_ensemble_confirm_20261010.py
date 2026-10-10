"""Frozen S confirmation: committed protocol -> new cal seal -> new hold.

The prior feature function and six HGB models are byte-identical dependencies.
No training/native-backbone fine-tuning occurs in this controller.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import traceback
import subprocess
import numpy as np
from threadpoolctl import threadpool_limits
import cnh_counterfactual_common_dev as C
import cnh_counterfactual_eval_dev as E
import cnh_frozen_e2e_tof_20261010 as F
import cnh_task_cost_retrain_20261010 as R
import cnh_task_cost_train_20261010 as T
import cnh_s_ensemble_confirm_source_20261010 as S
import cnh_s_ensemble_confirm_metrics_20261010 as V

OUT=C.ROOT/'artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010'
PRIOR=C.ROOT/'artifacts.local/work/cnh-task-cost-retrain-dev-20261010'
PROTOCOL=C.ROOT/'research/active/dtr-r0/nearfield/CNH_S_ENSEMBLE_CONFIRM_PROTOCOL_20261010.md'

def save(name,data):F.save_new(OUT/name,data)

def spec():
    OUT.mkdir(parents=True,exist_ok=True)
    inherited=C.read(F.DEFAULT_OUT/'input_hashes_frozen.json')
    prior=C.read(PRIOR/'execution_manifest.json')
    manifest=C.read(PRIOR/'trained_models_manifest.json')
    models={p:h for p,h in manifest['models'].items() if Path(p).name.startswith('S_')}
    assert len(models)==6
    features={p:h for p,h in prior['new_sources'].items() if Path(p).name in
              ('cnh_task_cost_retrain_20261010.py','cnh_task_cost_train_20261010.py','cnh_task_cost_metrics_20261010.py')}
    for p,h in {**inherited,**features,**models}.items():assert C.sha(p)==h,p
    save('frozen_spec.json',dict(task='CNH_S_ENSEMBLE_CONFIRM_DEV_20261010',lane='CONFIRM_DEVELOPMENT',
         inherited=inherited,feature_sources=features,models=models,prior_training_manifest_sha256=C.sha(PRIOR/'trained_models_manifest.json'),
         transfer_calibration_sha256=C.sha(PRIOR/'sealed_calibration.json'),transfer_calibration_path=str(PRIOR/'sealed_calibration.json'),
         metric=dict(contact_events=1024,per_height_events=512,total_net_min=80,per_height_net_min=11,seed_net_min=40,
                     far_clear_ratio=1.10,weighted_hold_ratio=1.25*1.05,cal_cost_ratio=1.25),
         budgets=dict(render_forward_gpu_seconds=1200,evaluation_audit_integration_command_wall_seconds=1200),
         inference='Byte-identical R.features_and_labels feature builder, frozen T.predict_s and six S pickles; mean float32 per prior implementation',
         claim_ceiling='New-family controlled AABB synthetic confirmation; default candidate recommendation only; no App/mainline replacement or hardware/safety claim'))

def freeze():
    began=time.monotonic()
    specdata=C.read(OUT/'frozen_spec.json')
    for p,h in {**specdata['inherited'],**specdata['feature_sources'],**specdata['models']}.items():assert C.sha(p)==h,p
    paths=[Path(__file__),Path(S.__file__),Path(V.__file__),PROTOCOL,OUT/'PLAN.json',OUT/'scene_rows.json',OUT/'inventory.json',OUT/'frozen_spec.json']
    paths += [OUT/'data'/split/'geometry.npz' for split in S.SPLITS]
    # Protocol must be a committed byte before any photon render/model forward.
    rel=PROTOCOL.relative_to(C.ROOT).as_posix()
    committed=subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=C.ROOT)
    assert committed==PROTOCOL.read_bytes(),'Protocol must be committed before rendering'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip()
    save('execution_manifest.json',dict(frozen_spec=specdata,new_sources={str(p):C.sha(p) for p in paths},
        protocol_commit=revision,training=0,protected_access=0,hold_before_seal=False))
    save('freeze_receipt.json',dict(status='FROZEN_COMMITTED_BEFORE_RENDER',seconds=time.monotonic()-began,
        protocol_commit=revision,manifest_sha256=C.sha(OUT/'execution_manifest.json')))

def identity():
    manifest=C.read(OUT/'execution_manifest.json');s=manifest['frozen_spec']
    for p,h in {**s['inherited'],**s['feature_sources'],**s['models'],**manifest['new_sources']}.items():assert C.sha(p)==h,p
    assert C.sha(PRIOR/'sealed_calibration.json')==s['transfer_calibration_sha256']
    return manifest

def charged(pattern):return sum(C.read(p).get('seconds',0) for p in OUT.glob(pattern))

def data_budget(began):
    used=charged('data_*_receipt.json')+charged('failure_data*.json')
    if used+time.monotonic()-began>=1180:raise TimeoutError('Render/forwardGPU cap; startup reserve20s')

def data(split):
    began=time.monotonic();identity()
    if split=='hold' and not (OUT/'sealed_calibration.json').exists():raise ValueError('Seal cal before hold photons/forward')
    check=lambda:data_budget(began)
    F.S=SimpleNamespace(SPLITS=(split,),K=S.K,sampling_seed=S.sampling_seed)
    rows={split:C.read(OUT/'scene_rows.json')[split]}
    detail=F.scientific(OUT,rows,check)
    (OUT/'render_receipt.json').rename(OUT/f'render_{split}_receipt.json')
    check()
    # Execute the byte-identical feature function from6e441f9a against new inputs.
    R.OUT=OUT
    supervision=R.features_and_labels(split)
    folder=OUT/'data'/split
    with np.load(folder/'frozen_readout.npz') as a:fr={k:a[k].copy() for k in a.files}
    with np.load(folder/'scores.npz') as a:ordinary=E.smooth(a['ordinary_raw'])
    cuts=C.read(F.THRESHOLDS);jointcuts=C.read(F.JOINT/'calibrations.json')
    ordinary_margin=ordinary-np.array([cuts[str(s)]['single'] for s in T.SEEDS])[:,None,None,None,None]
    joint_margin=fr['joint_scores']-np.array([jointcuts[f'{s}/score_current/c15_p64']['theta'] for s in T.SEEDS])[:,None,None,None,None]
    raw_ensemble=np.maximum(ordinary_margin.mean(0),joint_margin.mean(0))
    learned=T.predict_s(PRIOR,np.load(folder/'S_features.npy',mmap_mode='r'))
    scores=[np.where(raw_ensemble>=0,raw_ensemble,-np.inf),*learned,learned.mean(0)]
    np.savez_compressed(folder/'task_scores.npz',arm_keys=np.array(['E','S955','S956','S957','Sensemble']),
        light_scores=np.array(scores),diagnostic_scores=np.array([raw_ensemble,*learned,learned.mean(0)]),
        **{k:fr[k] for k in ('old5','m3','both','category')})
    check()
    save(f'data_{split}_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,detail=detail,
        supervision=supervision,training=0,cal_seal_sha256=C.sha(OUT/'sealed_calibration.json') if split=='hold' else None,
        outputs_sha256={n:C.sha(folder/n) for n in ('scores.npz','S_features.npy','labels.npz','frozen_readout.npz','task_scores.npz')}))

def evaluate(seal):
    began=time.monotonic();identity()
    def check():
        if charged('calibration_receipt.json')+charged('evaluation_receipt.json')+time.monotonic()-began>=160:
            raise TimeoutError('Evaluation160s cap; shared1200s includes source/audit/integration')
    rows=C.read(OUT/'scene_rows.json')
    result=V.calibrate(OUT,rows['cal'],check) if seal else V.evaluate(OUT,rows,check)
    save('calibration_receipt.json' if seal else 'evaluation_receipt.json',dict(**result,
        status='SEALED' if seal else 'COMPLETE',seconds=time.monotonic()-began,execution_manifest_sha256=C.sha(OUT/'execution_manifest.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('spec','freeze','data','seal','evaluate'));p.add_argument('--split',choices=S.SPLITS)
    args=p.parse_args();began=time.monotonic()
    for key in ('TEMP','TMP','CUPY_CACHE_DIR'):
        path=OUT/'runtime'/key.lower();path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    try:
        with threadpool_limits(limits=2):
            if args.stage=='spec':spec()
            elif args.stage=='freeze':freeze()
            elif args.stage=='data':data(args.split)
            else:evaluate(args.stage=='seal')
        print('CONFIRM_STAGE',args.stage,args.split,round(time.monotonic()-began,3),flush=True)
    except BaseException as err:
        save(f'failure_{args.stage}_{time.time_ns()}.json',dict(seconds=time.monotonic()-began,stage=args.stage,
            error=repr(err),traceback=traceback.format_exc()))
        raise

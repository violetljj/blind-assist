"""Frozen ToF cost-v2 run controller: cal inference -> seal -> hold inference.

Only this run's fresh metadata/observations enter evaluation. Inherited model,
threshold and source hashes are checked against the frozen E2E manifest.
No fitting, protected split access, hold-driven selection or source substitution.
"""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import cnh_counterfactual_common_dev as C
import cnh_counterfactual_data_dev as D
import cnh_frozen_e2e_tof_20261010 as F
import cnh_cost_v2_source_20261010 as S
import cnh_cost_v2_metrics_20261010 as V
from threadpoolctl import threadpool_limits

ROOT=C.ROOT
OUT=ROOT/'artifacts.local/work/cnh-cost-v2-holdout-dev-20261010'


def save(path,value):
    if path.exists():raise FileExistsError('Retain evidence: '+str(path))
    C.save(path,value)


def charged(pattern):
    return sum(C.read(p).get('seconds',0) for p in OUT.glob(pattern))


def freeze():
    began=time.monotonic()
    plan=C.read(OUT/'PLAN.json')
    inherited=C.read(F.DEFAULT_OUT/'input_hashes_frozen.json')
    for path,digest in inherited.items():
        if C.sha(path)!=digest:raise ValueError('Inherited frozen file changed: '+path)
    D.frozen_imports()
    paths=[Path(__file__),Path(S.__file__),Path(V.__file__),OUT/'PLAN.json',OUT/'scene_rows.json',OUT/'protocol_snapshot.md',
           OUT/'data/cal/geometry.npz',OUT/'data/hold/geometry.npz']
    sources={str(p):C.sha(p) for p in paths}
    save(OUT/'execution_manifest.json',dict(inherited=inherited,new_sources=sources,plan_sha256=C.sha(OUT/'PLAN.json'),
        cal_seal_before_hold_forward=True,scientific_cap_s=1500,gpu_allocation_cap_s=900,
        eval_audit_integration_cap_s=900,model_training=0,hold_selection=0,protected_access=0))
    save(OUT/'freeze_receipt.json',dict(status='FROZEN_BEFORE_RENDER',seconds=time.monotonic()-began,
        manifest_sha256=C.sha(OUT/'execution_manifest.json'),inherited_files=len(inherited),new_files=len(paths)))
    print('COST_V2_FROZEN',len(inherited),flush=True)


def identity():
    m=C.read(OUT/'execution_manifest.json')
    for path,digest in {**m['inherited'],**m['new_sources']}.items():
        if C.sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
    return m


def scientific(split):
    began=time.monotonic();identity()
    if (OUT/f'scientific_{split}_receipt.json').exists():raise FileExistsError('Scientific split already complete')
    seal=OUT/'sealed_calibration.json'
    if split=='hold' and not seal.exists():raise ValueError('Cal must be sealed before hold rendering or model forward')
    prior=charged('scientific_*receipt.json')+charged('scientific_failure_*.json')
    cpu_prior=charged('calibration_receipt.json')+charged('evaluation_receipt.json')
    def check():
        elapsed=time.monotonic()-began
        # Conservatively charge the entire CUDA-stage command as GPU allocation,
        # including its CPU sampler/I/O; reserve startup and cleanup headroom.
        if prior+elapsed>=890 or prior+cpu_prior+elapsed>=1490:
            raise TimeoutError('Shared scientific/GPU allocation cap reached')
    env={k:os.environ.get(k) for k in ('TEMP','TMP','CUPY_CACHE_DIR','PATH')}
    try:
        for key in ('TEMP','TMP','CUPY_CACHE_DIR'):
            p=OUT/'runtime'/key.lower();p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
        F.S=SimpleNamespace(SPLITS=(split,),K=S.K,sampling_seed=S.sampling_seed)
        rows={split:C.read(OUT/'scene_rows.json')[split]}
        check()
        with threadpool_limits(limits=2):detail=F.scientific(OUT,rows,check)
        check()
        generic=OUT/'render_receipt.json';specific=OUT/f'render_{split}_receipt.json'
        if generic.exists():
            if specific.exists():raise FileExistsError('Keep split rendering receipts distinct')
            generic.rename(specific)
        save(OUT/f'scientific_{split}_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,split=split,
            detail=detail,manifest_sha256=C.sha(OUT/'execution_manifest.json'),
            sealed_calibration_sha256=C.sha(seal) if split=='hold' else None,
            outputs_sha256={str(OUT/'data'/split/name):C.sha(OUT/'data'/split/name) for name in ('hist.npy','ambient.npy','scores.npz','scores_receipt.json')},
            training=0,protected_access=0,resource_release=True))
        print('COST_V2_SCIENTIFIC',split,round(time.monotonic()-began,3),flush=True)
    except BaseException as err:
        save(OUT/f'scientific_failure_{time.time_ns()}.json',dict(seconds=time.monotonic()-began,split=split,error=repr(err),traceback=traceback.format_exc()))
        raise
    finally:
        for key,value in env.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def evaluate(calibration_only=False):
    began=time.monotonic();identity()
    prior=charged('calibration_receipt.json')+charged('evaluation_receipt.json')+charged('evaluation_failure_*.json')
    # Metadata/source prep, audit and delivery have their own command-wall
    # receipts/reservations; evaluation stays inside a 300-second sub-budget.
    def check():
        if prior+time.monotonic()-began>=295:raise TimeoutError('Evaluation sub-budget reached')
    try:
        with threadpool_limits(limits=2):
            if calibration_only:
                if not (OUT/'scientific_cal_receipt.json').exists():raise ValueError('Cal inference incomplete')
                rows=C.read(OUT/'scene_rows.json')['cal']
                detail=V.calibrate(OUT,rows,check)
                save(OUT/'calibration_receipt.json',dict(status='SEALED',seconds=time.monotonic()-began,**detail))
            else:
                if not (OUT/'scientific_hold_receipt.json').exists():raise ValueError('Hold inference incomplete')
                rows=C.read(OUT/'scene_rows.json')
                detail=V.evaluate(OUT,rows,check)
                save(OUT/'evaluation_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,**detail))
        print('COST_V2_CAL_SEALED' if calibration_only else 'COST_V2_EVALUATED',round(time.monotonic()-began,3),flush=True)
    except BaseException as err:
        save(OUT/f'evaluation_failure_{time.time_ns()}.json',dict(seconds=time.monotonic()-began,calibration_only=calibration_only,error=repr(err),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','cal','seal','hold','evaluate'));a=p.parse_args()
    if a.stage=='freeze':freeze()
    elif a.stage in ('cal','hold'):scientific(a.stage)
    else:evaluate(a.stage=='seal')

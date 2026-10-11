"""Consumed dorm pilot Development evaluation after simulator model/cut seal."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import real_pilot_calibration as P
import cnh_sim2real_frozen as F
from cnh_no_bin0_dr_replay import DesignReplay
ROOT=P.ROOT
OUT=ROOT/'artifacts.local/work/cnh-no-bin0-dr-retrain-dev-20261011'
JOINT=ROOT/'artifacts.local/work/real-twin-joint-calibration-dev-20261011'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    path=OUT/name
    if path.exists():raise FileExistsError('Preserve replay evidence: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def adapter(d,bias):
    # A numerator-only comparison: denominator identical to v2 in both paths.
    params=P.read('instrument_parameters.json')['noise']
    arrays=np.load(P.OUT/'instrument_arrays.npz')
    a,b,c=params['adapter_coefficients_a_b_c']
    denom=np.sqrt(np.maximum(a*np.maximum(arrays['bias'],0)+b*np.maximum(d['A'],0)[...,None]+c,params['numerical_variance_floor']))
    return dict(v2=P.inputs(d)['v2'],bias_only=(d['H']-bias)/denom)
def describe(result,arm,path,sid):
    mask=result['ready'];n=int(mask.sum());answer=[]
    for qi,height in enumerate(('HEAD','BODY')):
        for model in ('M3','old5','S'):
            if model=='M3':strong=result['m3'][mask,qi];light=np.zeros(n,bool)
            elif model=='old5':strong=result['old5'][mask,qi];light=np.zeros(n,bool)
            else:strong=result['S_grade'][mask,qi]==2;light=result['S_grade'][mask,qi]==1
            answer.append(dict(arm=arm,path=path,condition=sid,height=height,model=model,ready_frames=n,strong=int(strong.sum()),light=int(light.sum()),strong_rate=float(strong.mean()) if n else 0.,light_rate=float(light.mean()) if n else 0.))
    return answer
def run(manifests,seconds=600):
    began=time.monotonic();seal=OUT/'sealed_candidates.json'
    def check_budget():
        if time.monotonic()-began >= seconds:raise TimeoutError('Reserved real replay command-wall budget reached')
    if not seal.exists():raise ValueError('Seal all simulator cuts/models before opening consumed pilot')
    sealdata=json.loads(seal.read_text(encoding='utf8'))
    for section in ('scientific_code_sha256','old_frozen_dependencies_sha256','models_sha256'):
        for path,digest in sealdata[section].items():
            if sha(ROOT/path)!=digest:raise ValueError('Sealed dependency modified: '+path)
    if sha(OUT/'sealed_calibration.json')!=sealdata['calibration_sha256']:
        raise ValueError('Calibration changed after seal')
    for arm,path in manifests.items():
        if sealdata['real_manifest_hashes'][arm]!=sha(path):raise ValueError('Manifest modified after simulator seal')
    bias=np.load(JOINT/'instrument_fit_arrays.npz')['bias'].reshape(64,16)
    all_data=P.data();tables=[];files=[];input_checks=[]
    for arm,manifest in {'old':None,**manifests}.items():
        check_budget()
        engine=F.FrozenReplay() if manifest is None else DesignReplay(manifest)
        try:
            for sid,d in all_data.items():
                for name,z in adapter(d,bias).items():
                    check_budget()
                    result=engine.infer(z.reshape(-1,8,8,16),d['seq'],d['ms'])
                    folder=OUT/'real'/arm/name;folder.mkdir(parents=True,exist_ok=True)
                    path=folder/(sid+'.npz');np.savez_compressed(path,**result)
                    tables.extend(describe(result,arm,name,sid));files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            # The synthetic background used in94d28cef, all other raw bins zero.
            background=np.repeat(bias[None],16,axis=0)
            ambient=np.repeat(all_data['2']['A'][all_data['2']['stable']].mean(0)[None],16,axis=0)
            empty=dict(H=background,A=ambient)
            seq=np.arange(16);ms=np.arange(16)*200
            for name,z in adapter(empty,bias).items():
                check_budget()
                result=engine.infer(z.reshape(-1,8,8,16),seq,ms)
                tables.extend(describe(result,arm,name,'BIN0_BACKGROUND_ONLY'))
                folder=OUT/'real'/arm/name;path=folder/'bin0_only.npz'
                np.savez_compressed(path,**result);files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            if manifest is not None:
                check_budget()
                rng=np.random.default_rng(20261011181)
                clean=rng.normal(0,1,(16,8,8,16)).astype(np.float32)
                poison=clean.copy();poison[...,engine.drop_bins]=rng.uniform(-1e4,1e4,poison[...,engine.drop_bins].shape)
                a=engine.infer(clean,seq,ms);b=engine.infer(poison,seq,ms)
                keys=('m3_raw','local_raw','ordinary_raw','s_seed_score','s_score','m3','old5','S_grade')
                equal={k:bool(np.array_equal(a[k],b[k],equal_nan=True)) for k in keys}
                if not all(equal.values()):raise AssertionError('Excluded native bins still readable')
                input_checks.append(dict(arm=arm,drop_bins=list(engine.drop_bins),poison_score_and_grade_exact=equal))
        finally:engine.close()
    path=OUT/'real_alarm_table.csv'
    with path.open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
    save('real_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,backend='CUDA FP32 engineering / CPU HGB',pilot_role='Consumed Development; never training/calibration',model_seal_sha256=sha(seal),tables=tables,outputs=files,input_checks=input_checks))
    print('REAL_COMPLETE',round(time.monotonic()-began,3),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--sensitivity');p.add_argument('--seconds',type=float,default=600);a=p.parse_args()
    manifests={'main':Path(a.main)}
    if a.sensitivity:manifests['drop01']=Path(a.sensitivity)
    run(manifests,a.seconds)

"""Separate deterministic simulation entry controls; frozen full-chain replay."""
from pathlib import Path
import argparse,json,time
import numpy as np
from local_signed_bias_adapter import sim_adapter,native_evidence,ContractReplay
from local_signed_bias_replay import ROOT,OUT,PREV,sha,table

BIAS=ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy'

def run(seconds):
    began=time.monotonic()
    def check():
        if time.monotonic()-began>=seconds:raise TimeoutError('SIM controls GPU allocation reached')
    bias=np.load(BIAS);assert bias.shape==(8,8,16) and (bias>=0).all()
    zero=np.zeros((16,8,8,16),np.float32);seq=np.arange(16);ms=seq*200
    cases={'SIM_EMPTY_SCENE':(zero,4.),'SIM_PURE_AMBIENT':(zero,10.4),
           'SIM_PURE_CROSSTALK':(np.broadcast_to(bias,zero.shape).copy(),4.)}
    rows=[];outputs=[];native_rows=[]
    for arm,manifest in (('old',None),('main',PREV/'main_real_manifest.json')):
        check();engine=ContractReplay(manifest)
        try:
            for name,(h,ambient_value) in cases.items():
                check();ambient=np.full((16,8,8),ambient_value,np.float32)
                z=sim_adapter(h,ambient,bias);native,positive=native_evidence(z,engine.drop_bins)
                assert not positive.any() and not np.count_nonzero(native)
                result=engine.infer(z,seq,ms,check)
                ready=result['ready'];assert int(ready.sum())==9
                for key in ('m3','old5','S_grade','Sensemble'):
                    assert not result[key][ready].any(),(arm,name,key)
                path=OUT/'sim_controls'/arm/(name+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,**result,seq=seq,ms=ms)
                rows.extend(table(result,arm,name,'sim_deterministic_no_surface'))
                outputs.append(dict(arm=arm,condition=name,path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
                native_rows.append(dict(arm=arm,condition=name,native_nonzero=int(np.count_nonzero(native)),positive_frames=int(positive.sum())))
        finally:engine.close()
    receipt=dict(status='PASS',seconds=time.monotonic()-began,source_sha256=sha(__file__),
                 adapter_sha256=sha(Path(__file__).with_name('local_signed_bias_adapter.py')),
                 bias_source=BIAS.relative_to(ROOT).as_posix(),bias_sha256=sha(BIAS),tables=rows,outputs=outputs,natives=native_rows,
                 interpretation='Deterministic mean/zero-photon controls. Random ambient shot noise can contain positive evidence; not a stochastic false-alarm claim.',
                 unchanged_cuts=True,training=0,protected_access=0)
    with (OUT/'sim_controls_receipt.json').open('x',encoding='utf8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps({key:receipt[key] for key in ('status','seconds','unchanged_cuts')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=30);a=p.parse_args();run(a.seconds)

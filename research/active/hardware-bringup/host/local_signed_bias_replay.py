"""v4 consumed pilot and deterministic full-chain controls, all cuts frozen."""
from pathlib import Path
import argparse,csv,hashlib,json,time
import numpy as np
import real_pilot_calibration as P
import cnh_sim2real_frozen as F
from local_signed_bias_adapter import real_adapter,native_evidence,ContractReplay

ROOT=P.ROOT
OUT=ROOT/'artifacts.local/work/local-signed-bias-contract-dev-20261011'
PREV=ROOT/'artifacts.local/work/cnh-no-bin0-dr-retrain-dev-20261011'
JOINT=ROOT/'artifacts.local/work/real-twin-joint-calibration-dev-20261011'
PLAN=ROOT/'research/active/hardware-bringup/LOCAL_SIGNED_BIAS_CONTRACT_PLAN_20261011.md'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def table(result,arm,condition,role):
    ready=result['ready'];n=int(ready.sum());rows=[]
    for q,height in enumerate(('HEAD','BODY')):
        for model,key in (('M3','m3'),('old5','old5'),('S','S_grade')):
            grade=result[key][ready,q];strong=grade==2 if model=='S' else grade.astype(bool)
            light=grade==1 if model=='S' else np.zeros(n,bool)
            rows.append(dict(arm=arm,path='v4',condition=condition,role=role,height=height,model=model,
                ready_frames=n,strong=int(strong.sum()),light=int(light.sum()),
                strong_rate=float(strong.mean()) if n else None,light_rate=float(light.mean()) if n else None))
    return rows

def run(seconds):
    began=time.monotonic()
    def check():
        if time.monotonic()-began>=seconds:raise TimeoutError('Reserved v4 replay GPU command-wall reached')
    OUT.mkdir(parents=True,exist_ok=True)
    seal=read(PREV/'sealed_candidates.json')
    for section in ('models_sha256','scientific_code_sha256','old_frozen_dependencies_sha256'):
        for path,digest in seal[section].items():assert sha(ROOT/path)==digest,path
    manifest=PREV/'main_real_manifest.json';assert sha(manifest)==seal['real_manifest_hashes']['main']
    arrays=np.load(P.OUT/'instrument_arrays.npz');bias=arrays['bias'].reshape(64,16)
    available=arrays['bias_available'];noise=P.read('instrument_parameters.json')['noise']
    physical=np.maximum(bias,0);statistical=np.minimum(bias,0)
    np.savez_compressed(OUT/'v4_template.npz',physical=physical,statistical=statistical,available=available,coverage=arrays['bias_coverage'])
    inputs=dict(plan_sha256=sha(PLAN),adapter_sha256=sha(Path(__file__).with_name('local_signed_bias_adapter.py')),
        replay_sha256=sha(__file__),prior_seal_sha256=sha(PREV/'sealed_candidates.json'),manifest_sha256=sha(manifest),
        template_source_sha256=sha(P.OUT/'instrument_arrays.npz'),noise_source_sha256=sha(P.OUT/'instrument_parameters.json'),
        template_covered=int(available.sum()),template_total=1024,physical_positive_cells=int((physical>0).sum()),
        statistical_negative_cells=int((statistical<0).sum()),LOCAL_cut_unchanged=True,cuts=read(manifest)['cuts'])
    save('v4_input_seal.json',inputs)
    data=P.data();ambient=data['2']['A'][data['2']['stable']].mean(0)
    zero=np.zeros((16,64,16));seq=np.arange(16);ms=seq*200
    wall=np.load(JOINT/'instrument_fit_arrays.npz')['bias'].reshape(64,16)
    def repeated(b):return np.repeat(b[None],16,axis=0)
    controls={
        'ZERO_RAW':(zero,bias,'deterministic_no_surface'),
        'PHYSICAL_BACKGROUND':(repeated(physical),bias,'deterministic_no_surface'),
        'SIGNED_BACKGROUND':(repeated(bias),bias,'deterministic_no_surface'),
        'PURE_AMBIENT':(zero,bias,'deterministic_no_surface'),
        'PURE_CROSSTALK':(repeated(physical),bias,'deterministic_no_surface'),
        'NEGATIVE_ONLY':(-np.ones_like(zero),bias,'deterministic_no_surface'),
        'NEGATIVE_BIAS_ZERO':(zero,statistical,'deterministic_no_surface'),
        'BIN0_BACKGROUND_ONLY':(repeated(wall),bias,'deterministic_no_surface'),
    }
    target=zero.copy();target[0,:,8]=100
    controls['POSITIVE_THEN_ZERO']=(target,bias,'history_control')
    tables=[];outputs=[];natives=[]
    for arm,modelmanifest in (('old',None),('main',manifest)):
        check();engine=ContractReplay(modelmanifest)
        try:
            observations=[(sid,d,bias,'consumed_pilot') for sid,d in data.items()]
            for name,(h,b,role) in controls.items():
                a=np.repeat(ambient[None],16,axis=0)*(2.6 if name=='PURE_AMBIENT' else 1)
                d=dict(H=h,A=a,seq=seq,ms=ms)
                if arm=='old':
                    np.savez_compressed(OUT/('fixture_'+name+'.npz'),**d,bias=b)
                observations.append((name,d,b,role))
            for sid,d,b,role in observations:
                check();z=real_adapter(d['H'],d['A'],b,noise).reshape(-1,8,8,16)
                native,positive=native_evidence(z,engine.drop_bins)
                result=engine.infer(z,d['seq'],d['ms'],check)
                folder=OUT/'real'/arm;folder.mkdir(parents=True,exist_ok=True)
                path=folder/(sid+'.npz');np.savez_compressed(path,**result,seq=d['seq'],ms=d['ms'])
                tables.extend(table(result,arm,sid,role))
                outputs.append(dict(arm=arm,condition=sid,role=role,path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
                natives.append(dict(arm=arm,condition=sid,role=role,positive_frames=int(positive.sum()),
                    total_frames=len(z),native_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                    native_nonzero=int(np.count_nonzero(native)),evidence_present_frames=int(result['evidence_present'].sum())))
        finally:engine.close()
    with (OUT/'real_alarm_table.csv').open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
    save('real_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,tables=tables,outputs=outputs,
         natives=natives,input_seal_sha256=sha(OUT/'v4_input_seal.json'),backend='CUDA FP32 / CPU HGB',pilot_role='Consumed Development contract/behavior only'))
    print('V4_REAL_COMPLETE',round(time.monotonic()-began,3),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=120);a=p.parse_args();run(a.seconds)

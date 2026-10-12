"""Budgeted frozen real replay; payloads and partial work are retained."""
import json
import time
import traceback
import numpy as np
import cnh_sim2real_input_contract as C
from cnh_sim2real_frozen import FrozenReplay

ARMS=('m3','old5','Sensemble')

def run():
    started=time.monotonic(); engine=None
    # Smoke10.01 + first pass62.6 conservatively charged; this corrected
    # replay has a200s cap and preserves the original pass below pre-trend-repair.
    def check():
        if time.monotonic()-started>=200: raise TimeoutError('Corrected GPU200s cap; smoke/first-pass included in total300s')
    data=C.recordings(); instrument=C.read('instrument_report.json')
    fitted=instrument['noise_parameters']
    noise=np.array([fitted['intercept'],fitted['ambient_coefficient'],fitted['positive_bias_coefficient']])
    with np.load(C.OUT/'instrument_arrays.npz') as p: bias=p['bias']
    report=dict(scope='Existing real replay description; firmware range weak reference; no event truth',
        corrected_run=True,repair='ordinary trend mean must retain float32; first pass retained separately',
        code_sha256={p.name:C.sha(p) for p in [C.Path(__file__),C.Path(__file__).with_name('cnh_sim2real_frozen.py'),C.Path(C.__file__)]},
        plan_sha256=C.sha(C.OUT/'PLAN.json'),instrument_sha256=C.sha(C.OUT/'instrument_report.json'),
        feature_precision='float32-engineering',geometry_precision='float32-frozen',
        input_semantics=dict(old='whole-scene reference subtraction / measured sample SD',new='subtract only far-selected instrument template, use fitted c0+cA*ambient+cB*positive_bias noise floor'),
        noise_parameters=noise.tolist(),segments={})
    try:
        engine=FrozenReplay(feature_precision='float32-engineering',batch_size=32)
        report['models']=engine.used_hashes; report['thresholds']=dict(m3=.8557642486787612,raised=.9404184587540165,local=4.625390338985158,Sensemble=engine.s_tau)
        report['backend']=dict(device=str(engine.device),name=engine.torch.cuda.get_device_name(),cuda=True,batch_size=32)
        for sid,d in data.items():
            check();old,epoch,reference=C.old_input(sid,d,data)
            new=C.adapt(d['H'],d['A'],bias,noise)
            observed,valid_zones=C.support(d)
            base=dict(frames=len(old),reference=reference,reference_epoch_counts={str(i):int((epoch==i).sum()) for i in np.unique(epoch)},paths={})
            report['segments'][sid]=base
            # Reference changes reset both arms for strictly paired histories.
            modified_ms=d['ms'].copy()
            for i in np.flatnonzero(np.r_[False,np.diff(epoch)!=0]):modified_ms[i]=modified_ms[i-1]
            for path,z in [('old',old),('new',new)]:
                check(); result=engine.infer(z.reshape(-1,8,8,16),d['seq'],modified_ms,check)
                result.update(seq=d['seq'],ms=d['ms'],effective_ms=modified_ms,reference_epoch=epoch,
                    weak_support=observed,valid_firmware_zones=valid_zones)
                np.savez_compressed(C.OUT/f'{sid}_{path}_scores.npz',**result)
                ready=result['ready']
                base['paths'][path]=dict(ready=int(ready.sum()),warmup=int((~ready).sum()),resets=int(result['reset'].sum()),
                    seconds=float(result['seconds']),z_quantiles=C.distribution(z),
                    arms={arm:dict(trigger_counts=result[arm].sum(0).tolist(),trigger_rates=(result[arm].sum(0)/max(ready.sum(),1)).tolist(),
                        surface_supported_trigger_counts=(result[arm]&observed).sum(0).tolist(),
                        no_observed_support_trigger_counts=(result[arm]&~observed).sum(0).tolist()) for arm in ARMS})
                C.save('replay_partial.json',report)
                print('REPLAY',sid,path,'ready',int(ready.sum()),'triggers',{arm:result[arm].sum(0).tolist() for arm in ARMS},'elapsed',round(time.monotonic()-started,2),flush=True)
        # Two preidentified cells are tested only after both main paths exist.
        # This probe is diagnostic evidence, never a correction or model policy.
        for sid,d in data.items():
            old,epoch,_=C.old_input(sid,d,data)
            new=C.adapt(d['H'],d['A'],bias,noise)
            modified_ms=d['ms'].copy()
            for i in np.flatnonzero(np.r_[False,np.diff(epoch)!=0]):modified_ms[i]=modified_ms[i-1]
            for path,z in [('old',old),('new',new)]:
                check(); altered=z.copy();altered[:,34,1:3]=0
                cf=engine.infer(altered.reshape(-1,8,8,16),d['seq'],modified_ms,check)
                with np.load(C.OUT/f'{sid}_{path}_scores.npz') as p:orig={k:v.copy() for k,v in p.items()}
                details={}
                for arm in ARMS:
                    flags=orig[arm];fixed=flags&~cf[arm];supported=flags&~fixed&orig['weak_support'];unsupported=flags&~fixed&~orig['weak_support']
                    details[arm]=dict(fixed_pattern_sensitive=fixed.sum(0).tolist(),surface_supported=supported.sum(0).tolist(),
                        no_observed_support=unsupported.sum(0).tolist(),counterfactual_added=(~flags&cf[arm]).sum(0).tolist(),
                        fixed_sensitive_with_surface_support=(fixed&orig['weak_support']).sum(0).tolist())
                    orig[f'{arm}_class']=np.where(fixed,3,np.where(supported,1,np.where(unsupported,2,0))).astype(np.int8)
                np.savez_compressed(C.OUT/f'{sid}_{path}_classes.npz',**{k:orig[k] for k in ('seq','ready','weak_support','m3_class','old5_class','Sensemble_class')},
                    **{f'{arm}_counterfactual':cf[arm] for arm in ARMS})
                report['segments'][sid]['paths'][path]['classes']=details
                C.save('replay_partial.json',report)
                print('ATTRIBUTION',sid,path,'elapsed',round(time.monotonic()-started,2),flush=True)
        report['status']='COMPLETE'
    except BaseException as err:
        report['status']='PARTIAL';report['error']=repr(err);report['traceback']=traceback.format_exc()
        C.save('replay_failure.json',report)
        raise
    finally:
        if engine is not None: engine.close();report['model_resources_released']=True
        report['seconds']=time.monotonic()-started
        C.save('replay_report.json',report)
        print('REPLAY_TERMINAL',report['status'],round(report['seconds'],3),flush=True)

if __name__=='__main__':run()

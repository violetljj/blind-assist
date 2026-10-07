"""Paired new-unit descriptions; fixed models and calibration, no selection."""
import json
from pathlib import Path
import time
import numpy as np
import cnh_training_support_contrasts_dev as C

OUT=C.I.R.WORK/'cnh-model-stability-dev-20261007'
SEED=2026100733


def compare(z,units,inverse,weights,mask,akey,bkey):
    # Reuse the established cluster calculation; this batch has no assigned fold.
    result=C.compare({**z,'fold':np.zeros(len(z['unit']),int)},units,inverse,weights,mask,akey,bkey)
    result.pop('fold_timely_gain')
    delta=(z['timely_excluding_warmup/'+akey].astype(int)-z['timely_excluding_warmup/'+bkey].astype(int))*z['contact']*mask
    result['mode_timely_gain']={str(m):int(delta[z['mode']==m].sum()) for m in range(3)}
    return result


def main():
    start=time.monotonic();source=C.I.R.read(OUT/'result.json')
    if C.I.R.read(OUT/'analysis_terminal.json')['status']!='COMPLETE':raise ValueError('Analysis incomplete')
    path=OUT/'ledger.npz'
    if C.I.R.sha(path)!=source['ledger_sha256']:raise ValueError('Ledger hash mismatch')
    if (OUT/'contrasts.json').exists():raise FileExistsError('Preserve previous contrasts')
    with np.load(path) as data:
        z={k:data[k] for k in data.files if k in ('unit','mode','mirror','contact','control','tag','valid','evaluable')
           or k.startswith(('alarm/','timely_excluding_warmup/'))}
    strata={}
    for u in np.unique(z['unit']):
        values=set(zip(z['mode'][z['unit']==u].tolist(),z['mirror'][z['unit']==u].tolist()))
        if len(values)!=1:raise ValueError('Unit stratum must be constant')
        strata[int(u)]=next(iter(values))
    units,inverse,weights=C.bootstrap_weights(z['unit'],strata,SEED)
    output=dict(scope='New simulated units from the same generator and reused head-error pool. Conditional unit-cluster descriptions, no refit or model-selection correction.',
        seed=SEED,draws=1000,budget_seconds=60,strata='mode and mirror',units=len(units),results={})
    for tag in ('original','HB'):
        mask=(z['tag']==tag)&z['valid']&z['evaluable']
        for cap in ('0.025','0.050'):
            pairs=[]
            for f in range(3):
                a=f'fold{f}/calibrated/{cap}/raw_hb_aug'
                pairs.extend((a,f'fold{f}/calibrated/{cap}/{m}') for m in ('original_center','raw_source_repeat','pair_hb_aug'))
            for method in ('original_center','raw_hb_aug'):
                pairs.extend((f'fold{a}/calibrated/{cap}/{method}',f'fold{b}/calibrated/{cap}/{method}') for a,b in ((1,0),(2,0),(2,1)))
            for akey,bkey in pairs:
                if time.monotonic()-start>60:raise TimeoutError('Frozen contrast budget reached')
                output['results'][f'{tag}/{akey}-minus-{bkey}']=compare(z,units,inverse,weights,mask,akey,bkey)
    output['seconds']=time.monotonic()-start
    output['source_sha256']={str(p.relative_to(C.I.R.ROOT)):C.I.R.sha(p) for p in
        (Path(__file__),Path(C.__file__),OUT/'PLAN.json',OUT/'result.json',path)}
    with (OUT/'contrasts.json').open('x',encoding='utf8') as stream:
        json.dump(output,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print('CONTRASTS',len(output['results']),round(output['seconds'],3),'s')


if __name__=='__main__':main()

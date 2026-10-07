"""Conditional comparisons after shared calibration; no threshold fitting."""
import json
from pathlib import Path
import time
import numpy as np
import cnh_model_stability_contrasts_dev as S

OUT=S.C.I.R.WORK/'cnh-shared-calibration-dev-20261007'


def main():
    start=time.monotonic();result=S.C.I.R.read(OUT/'result.json');path=OUT/'ledger.npz'
    if S.C.I.R.read(OUT/'analysis_terminal.json')['status']!='COMPLETE':raise ValueError('Analysis incomplete')
    if S.C.I.R.sha(path)!=result['ledger_sha256']:raise ValueError('Evaluation ledger hash mismatch')
    if (OUT/'contrasts.json').exists():raise FileExistsError('Preserve prior contrasts')
    with np.load(path) as data:
        z={k:data[k] for k in data.files if k in ('unit','mode','mirror','contact','control','tag','valid','evaluable')
           or k.startswith(('alarm/','timely_excluding_warmup/'))}
    strata={}
    for u in np.unique(z['unit']):
        values=set(zip(z['mode'][z['unit']==u].tolist(),z['mirror'][z['unit']==u].tolist()))
        if len(values)!=1:raise ValueError('Unit stratum must be constant')
        strata[int(u)]=next(iter(values))
    units,inverse,weights=S.C.bootstrap_weights(z['unit'],strata,2026100733)
    output=dict(scope='Posthoc diagnostic on the already consumed stability evaluation units. Common new original-only calibration controls; conditional evaluation-unit bootstrap with fitted thresholds/models fixed, no calibration-resampling or model-selection correction.',
        draws=1000,seed=2026100733,strata='mode and mirror',budget_seconds=60,results={})
    for tag in ('original','HB'):
        mask=(z['tag']==tag)&z['valid']&z['evaluable']
        for cap in ('0.025','0.050'):
            for f in range(3):
                prefix=f'fold{f}/shared_calibrated/{cap}/'
                pairs=[(prefix+'raw_hb_aug',prefix+m) for m in ('original_center','raw_source_repeat','pair_hb_aug')]
                pairs.extend((prefix+m,f'fold{f}/legacy_calibrated/{cap}/{m}') for m in ('original_center','raw_hb_aug'))
                for akey,bkey in pairs:
                    if time.monotonic()-start>60:raise TimeoutError('60s contrast budget reached')
                    output['results'][f'{tag}/{akey}-minus-{bkey}']=S.compare(z,units,inverse,weights,mask,akey,bkey)
    output['seconds']=time.monotonic()-start
    output['source_sha256']={str(p.relative_to(S.C.I.R.ROOT)):S.C.I.R.sha(p) for p in
        (Path(__file__),Path(S.__file__),Path(S.C.__file__),OUT/'PLAN.json',OUT/'result.json',path)}
    with (OUT/'contrasts.json').open('x',encoding='utf8') as stream:
        json.dump(output,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print('CONTRASTS',len(output['results']),round(output['seconds'],3),'s')


if __name__=='__main__':main()

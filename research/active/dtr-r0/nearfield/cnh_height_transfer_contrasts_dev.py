"""Conditional paired unit bootstrap of fixed height-transfer readouts.

No refit, calibration, threshold choice or model/variant selection. Prepare this
script before results; run only after the parent fixed evaluation completes.
"""
import json
from collections import Counter
from pathlib import Path
import time

import numpy as np
import cnh_training_support_contrasts_dev as C

OUT = C.I.R.WORK/'cnh-height-transfer-dev-20261007'
TAGS = ('short','up','down','tall')
CAPS = ('0.025','0.050')
BASELINES = ('original_center','raw_source_repeat','pair_hb_aug')
SEED = 2026100731
BUDGET_SECONDS = 60


def main():
    start = time.monotonic()
    dest = OUT/'contrasts.json'
    if dest.exists():raise FileExistsError(dest)
    source = C.I.R.read(OUT/'result.json')
    terminal = C.I.R.read(OUT/'analysis_terminal.json')
    if source['status']!='EXPLORATORY_COMPLETE' or terminal['status']!='COMPLETE':
        raise ValueError('Wait for complete fixed evaluation before contrasts')
    ledger_path = OUT/'ledger.npz'
    if C.I.R.sha(ledger_path)!=source['ledger_sha256']:
        raise ValueError('Evaluation ledger differs from its result receipt')
    keys = [f'calibrated/{cap}/{method}' for cap in CAPS
            for method in ('raw_hb_aug',*BASELINES)]
    with np.load(ledger_path,allow_pickle=False) as saved:
        z = {k:saved[k] for k in ('unit','fold','contact','control','tag','valid','common_valid')}
        for key in keys:
            for prefix in ('alarm/','timely_excluding_warmup/'):
                z[prefix+key]=saved[prefix+key]
    if len(z['unit'])!=288 or (z['common_valid']&~z['valid']).any():
        raise ValueError('Expected all 288 rows with common valid contained in own valid')
    strata = {}
    for unit in np.unique(z['unit']):
        folds = np.unique(z['fold'][z['unit']==unit])
        if len(folds)!=1:raise ValueError('A unit cannot span multiple OOF folds')
        strata[int(unit)]=int(folds[0])
    units,inverse,weights = C.bootstrap_weights(z['unit'],strata,SEED)
    output = dict(scope='Consumed synthetic Development, conditional descriptive paired contrasts. '
                        'Frozen models and original calibration thresholds; no refit or selection correction. '
                        'Unit clusters resampled within original OOF fold; the same draws serve all contrasts.',
        draws=1000,seed=SEED,budget_seconds=BUDGET_SECONDS,
        candidate='raw_hb_aug',baselines=list(BASELINES),caps=list(CAPS),
        denominator_rule='Each tag own_valid or all-four-tag common_valid. Invalid records remain stored and never become controls.',
        unit_count=len(units),units_per_fold={str(k):v for k,v in sorted(Counter(strata.values()).items())},results={})
    for tag in TAGS:
        for scope in ('own_valid','common_valid'):
            mask=(z['tag']==tag)&z['valid' if scope=='own_valid' else 'common_valid']
            for cap in CAPS:
                akey=f'calibrated/{cap}/raw_hb_aug'
                for baseline in BASELINES:
                    if time.monotonic()-start>BUDGET_SECONDS:
                        raise TimeoutError('60s conditional analysis budget reached')
                    bkey=f'calibrated/{cap}/{baseline}'
                    result=C.compare(z,units,inverse,weights,mask,akey,bkey)
                    result.update(valid_sequences=int(mask.sum()),invalid_excluded=int(((z['tag']==tag)&~mask).sum()),
                        observed_units=int(len(np.unique(z['unit'][mask]))),
                        candidate_timely=int((z['timely_excluding_warmup/'+akey]&z['contact']&mask).sum()),
                        baseline_timely=int((z['timely_excluding_warmup/'+bkey]&z['contact']&mask).sum()))
                    output['results'][f'{tag}/{scope}/calibrated/{cap}/raw_hb_aug-{baseline}']=result
    paths=[Path(__file__),Path(C.__file__),OUT/'PLAN.json',OUT/'geometry_manifest.json',
           OUT/'result.json',ledger_path,OUT/'readout_inheritance.json']
    output['source_sha256']={p.relative_to(C.I.R.ROOT).as_posix():C.I.R.sha(p) for p in paths}
    output['seconds']=time.monotonic()-start
    if output['seconds']>BUDGET_SECONDS:raise TimeoutError('60s conditional analysis budget reached')
    with dest.open('x',encoding='utf8') as stream:
        json.dump(output,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(contrasts=len(output['results']),seconds=output['seconds']),indent=2))
    return output


if __name__=='__main__':main()

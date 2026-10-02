"""Empirical distance anchoring; no physical echo-origin or corridor claim.

Every prediction is formed from sensor summaries and sealed RGB-only depth
before reference is opened. Same-grid rank metrics are deliberately not used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import cnh_zju_rank_localization as grid

ROOT=grid.ROOT
PARENT=grid.OUT
OUT=ROOT/'artifacts.local/work/cnh-zju-distance-anchor-20261002'
RUN_ID='CNH_ZJU_DISTANCE_ANCHOR_20261002'
ARMS=('raw','mono','depthor','global','crossfit','local','shuffled')
SEED=2026100227
read=grid.read
save=grid.save
sha=grid.sha


def observed(boxes,mask,anchor,mono,depthor,key):
    """No reference input. First column is a representative return, not area mean."""
    nodes=[grid.public_nodes(box,mono.shape)[0] for box in boxes]
    m=[mono[n[:,0],n[:,1]].astype(float) for n in nodes]
    med=np.array([np.median(v[np.isfinite(v)&(v>0)]) if np.any(np.isfinite(v)&(v>0)) else np.nan for v in m])
    valid=mask&np.isfinite(anchor)&(anchor>.001)&(anchor<10)&np.isfinite(med)&(med>0)
    ratio=np.full(64,np.nan);ratio[valid]=anchor[valid]/med[valid]
    global_scale=float(np.median(ratio[valid])) if valid.any() else None
    parity=(np.arange(64)//8+np.arange(64)%8)%2
    fit_scales=[float(np.median(ratio[valid&(parity!=fold)])) if np.any(valid&(parity!=fold)) else None for fold in (0,1)]
    indices=np.flatnonzero(valid)
    seed=int.from_bytes(hashlib.sha256((RUN_ID+'|'+key).encode()).digest()[:8],'little')
    permutation=np.random.default_rng(seed).permutation(indices)
    shuffled=anchor.copy();shuffled[indices]=anchor[permutation]
    zones=[]
    for k,(n,value) in enumerate(zip(nodes,m)):
        bad=~np.isfinite(value)|(value<=0);value=value.copy();value[bad]=np.nan
        known_sensor=bool(mask[k] and np.isfinite(anchor[k]) and .001<anchor[k]<10)
        pred=dict(raw=np.full(len(n),anchor[k] if known_sensor else np.nan),mono=value,
            depthor=depthor[n[:,0],n[:,1]].astype(float))
        pred['global']=value*global_scale if global_scale is not None else np.full(len(n),np.nan)
        factor=fit_scales[parity[k]]
        pred['crossfit']=value*factor if factor is not None else np.full(len(n),np.nan)
        pred['local']=value*ratio[k] if valid[k] else np.full(len(n),np.nan)
        pred['shuffled']=value*shuffled[k]/med[k] if valid[k] else np.full(len(n),np.nan)
        representatives={a:float(np.median(v[np.isfinite(v)&(v>0)])) if np.any(np.isfinite(v)&(v>0)) else None for a,v in pred.items()}
        sensor_residual={a:abs(x-float(anchor[k])) if known_sensor and x is not None else None for a,x in representatives.items()}
        iqr=(float(np.percentile(value[~bad],75)-np.percentile(value[~bad],25))/med[k]) if np.any(~bad) else None
        zones.append(dict(zone=k,nodes=n,pred=pred,mask_valid=bool(mask[k]),sensor_valid=known_sensor,
            anchor_eligible=bool(valid[k]),median_mono=float(med[k]) if np.isfinite(med[k]) else None,
            anchor=float(anchor[k]) if np.isfinite(anchor[k]) else None,
            ratio=float(ratio[k]) if valid[k] else None,low_rgb_spread=bool(iqr is not None and iqr<=.1),
            sensor_residual=sensor_residual,shuffle_from=int(permutation[np.flatnonzero(indices==k)[0]]) if valid[k] else None))
    return zones,dict(global_scale=global_scale,crossfit_scales=fit_scales,anchors=int(valid.sum()),
        same_zone_shuffle=int(np.sum(indices==permutation)))


def scored(z,gt):
    nodes=z.pop('nodes');pred=z.pop('pred');d=gt[nodes[:,0],nodes[:,1]]
    known=np.isfinite(d)&(d>.001)&(d<10);near=known&(d<2.1);far=known&(d>=2.1)
    band=known&(d>=.6)&(d<2.1);distant=known&(d>=1.2)&(d<2.1)
    near_calls={a:np.isfinite(v)&(v>0)&(v<2.1) for a,v in pred.items()}
    stats={}
    for a,v in pred.items():
        finite=np.isfinite(v)&(v>0)
        # Common-denominator error penalty explicitly retains missing predictions.
        loss=np.full(len(d),10.);loss[finite]=np.minimum(np.abs(v[finite]-d[finite]),10.)
        call=near_calls[a]
        stats[a]=dict(tp=int((near&call).sum()),fn=int((near&~call).sum()),fp=int((far&call).sum()),tn=int((far&~call).sum()),
            unknown_near=int((near&~finite).sum()),unknown_far=int((far&~finite).sum()),
            capped_known=int((known&finite&(np.abs(v-d)>10)).sum()),
            prediction_missing=int((~finite).sum()),reference_unknown_alerts=int((~known&call).sum()),
            band_tp=int((band&call).sum()),distant_tp=int((distant&call).sum()),
            loss_sum=float(loss[known].sum()),loss_n=int(known.sum()),loss_band_sum=float(loss[band].sum()),loss_band_n=int(band.sum()),
            paired={b:dict(rescued_near=int((near&call&~near_calls[b]).sum()),lost_near=int((near&~call&near_calls[b]).sum()),
                added_far=int((far&call&~near_calls[b]).sum()),removed_far=int((far&~call&near_calls[b]).sum())) for b in ('raw','mono','depthor','global','crossfit','shuffled') if b!=a})
    return dict(**z,node_count=len(d),near=int(near.sum()),far=int(far.sum()),reference_unknown=int((~known).sum()),
        band=int(band.sum()),distant=int(distant.sum()),mixed=bool(near.any() and far.any()),thin=bool(1<=near.sum()<=8 and far.any()),arms=stats)


def frame(row):
    for path_key,hash_key in (('input','input_sha256'),('prediction','prediction_sha256'),('depthor','depthor_sha256')):
        assert sha(row[path_key])==row[hash_key]
    with h5py.File(row['input'],'r') as f:boxes=f['fr'][:];mask=f['mask'][:];anchor=f['hist_data'][:,0]
    with np.load(row['prediction']) as f:mono=f['depth']
    with np.load(row['depthor']) as f:depthor=f['depth']
    zones,scales=observed(boxes,mask,anchor,mono,depthor,row['id'])
    # All candidates fixed above; reference is only opened below.
    with h5py.File(row['input'],'r') as f:gt=f['depth'][:]
    return dict(id=row['id'],index=row['index'],scene=row['scene'],role=row['role'],scales=scales,zones=[scored(z,gt) for z in zones])


def aggregate(zones):
    result={k:sum(z[k] for z in zones) for k in ('node_count','near','far','reference_unknown','band','distant')}
    result.update(zones=len(zones),empty=sum(z['node_count']==0 for z in zones),arms={})
    for a in ARMS:
        records=[z['arms'][a] for z in zones]
        numeric=('tp','fn','fp','tn','unknown_near','unknown_far','capped_known','prediction_missing','reference_unknown_alerts','band_tp','distant_tp','loss_sum','loss_n','loss_band_sum','loss_band_n')
        s={k:sum(x[k] for x in records) for k in numeric}
        def div(x,y):return x/y if y else None
        s.update(recall=div(s['tp'],result['near']),far_fpr=div(s['fp'],result['far']),
            band_recall=div(s['band_tp'],result['band']),distant_recall=div(s['distant_tp'],result['distant']),
            capped_mae=div(s['loss_sum'],s['loss_n']),capped_band_mae=div(s['loss_band_sum'],s['loss_band_n']))
        use=[x['loss_sum']/x['loss_n'] for x in records if x['loss_n']]
        s['zone_mae_sum']=sum(use);s['zone_mae_n']=len(use);s['macro_capped_mae']=div(sum(use),len(use))
        residuals=[z['sensor_residual'][a] for z in zones if z['sensor_residual'][a] is not None]
        s['representative_residual_sum']=sum(residuals);s['representative_residual_n']=len(residuals)
        s['representative_residual']=div(sum(residuals),len(residuals))
        s['paired']={b:{k:sum(x['paired'][b][k] for x in records) for k in ('rescued_near','lost_near','added_far','removed_far')} for b in ('raw','mono','depthor','global','crossfit','shuffled') if b!=a}
        result['arms'][a]=s
    return result


def summarize(frames):
    zones=[z for f in frames for z in f['zones']]
    domain=lambda z:z['mask_valid'] and z['sensor_valid']
    filters=dict(all=lambda z:True,sensor_valid=domain,mixed=lambda z:domain(z) and z['mixed'],
        thin=lambda z:domain(z) and z['thin'],low_rgb_spread=lambda z:domain(z) and z['low_rgb_spread'])
    return {name:aggregate([z for z in zones if fn(z)]) for name,fn in filters.items()}


def prepare():
    assert not (OUT/'PLAN.json').exists()
    previous=read(PARENT/'PLAN.json');seal=read(grid.OLD/'prediction-seal.json')
    rows=[]
    for row in previous['inputs']:
        row=dict(row);f=grid.OLD/'predictions'/f"{row['index']:03d}-depthor.npz"
        assert sha(f)==seal['files'][f.name]
        row.update(depthor=str(f),depthor_sha256=seal['files'][f.name]);rows.append(row)
    OUT.mkdir(parents=True,exist_ok=True)
    line=f'| 2026-10-02 | {RUN_ID} | PRE_RUN; same consumed ZJU160 4cal/4eval and public nodes; sensor representative / mono median scale: global, checkerboard2fold crossfit, local, fixed same-frame shuffled-anchor local; raw/mono/DEPTHOR controls; no GT fit/new inference | NOT_RUN; fixed2.1m full-support TP/FN/FP/TN plus UNKNOWN, all/mixed/thin/.6-2.1/1.2-2.1; capped10m missing-penalized distance error; local paired rescue/loss | Candidate local: mixed error lower than global+shuffle, >=3/4eval scene error wins, full valid near recall>=raw+mono and far FPR<=raw+mono; pass DISTANCE_ANCHORING_COMPONENT_DEV otherwise STOP_MEDIAN_ANCHOR_RECIPE, missing denominators NOT_EVALUABLE. Local self-fit residual/rank unchanged cannot prove benefit. No body/echo/metric-calibration claim | `artifacts.local/work/cnh-zju-distance-anchor-20261002/REPORT.md` |\n'
    runs=grid.RUNS;body=runs.read_text(encoding='utf-8');assert RUN_ID not in body
    runs.write_text(body.rstrip()+'\n'+line,encoding='utf-8');(OUT/'prerun-row.txt').write_text(line,encoding='utf-8')
    save(OUT/'PLAN.json',dict(run_id=RUN_ID,inputs=rows,roles=previous['roles'],source_sha256=sha(__file__),grid_source_sha256=sha(grid.__file__),
        parent_sha256=sha(PARENT/'PLAN.json'),rules=dict(arms=ARMS,threshold=2.1,reference='finite .001<d<10, near<2.1; .6-2.1 and1.2-2.1 subsets diagnostic',
            scale='positive valid mask return / median positive mono public nodes; no clipping scale; no valid anchor => UNKNOWN; global median ratios allvalid; crossfit ratios from opposite public checkerboard parity; local uses own ratio',
            shuffle='Same-frame permutation over anchor-eligible zones; preserve sensor values/missing pattern; fixedpoints reported, not forced derangement',
            truth='All64 zone predictions formed before reference opened; overlapping zones not merged; GT only scores/stratifies',
            error='Per-known-point min(abs(pred-GT),10m); invalid prediction penalty10m, native errors below cap not changed; macro averages zone errors; not ordinary uncapped MAE if missing/capped',
            gate='On eval sensor_valid: local recall>=raw and mono, far FPR<=raw and mono. On mixed: local macro capped error<global and shuffled and>=3/4scene paired error wins both; undefined => NOT_EVALUABLE; numerical tolerance1e-12. DEPTHOR and crossfit retained as stronger contextual controls, no claim to beat them unless measured.',
            scope='Empirical interface distance test; sensor location not area mean/q10, unverified radial/Z; no return attribution or body alarm; consumed Development, no same-recipe retuning'),preregistration=line))
    print('PREPARED',len(rows),flush=True)


def run():
    p=read(OUT/'PLAN.json');assert sha(__file__)==p['source_sha256'] and sha(grid.__file__)==p['grid_source_sha256']
    assert not (OUT/'result.json').exists();start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:frames=list(pool.map(frame,p['inputs']))
    save(OUT/'frame-ledger.json',frames)
    summaries={role:summarize([f for f in frames if f['role']==role]) for role in ('cal','eval')};summaries['all']=summarize(frames)
    scenes={s:summarize([f for f in frames if f['scene']==s]) for s in p['roles']}
    e=summaries['eval'];a=e['sensor_valid']['arms'];m=e['mixed']['arms'];eval_scenes=[s for s,v in p['roles'].items() if v=='eval']
    estimable=all(a[x][k] is not None for x in ('local','raw','mono') for k in ('recall','far_fpr')) and all(m[x]['macro_capped_mae'] is not None for x in ('local','global','shuffled'))
    wins=[]
    for s in eval_scenes:
        q=scenes[s]['mixed']['arms']
        if all(q[x]['macro_capped_mae'] is not None for x in ('local','global','shuffled')) and all(q['local']['macro_capped_mae']<q[x]['macro_capped_mae']-1e-12 for x in ('global','shuffled')):wins.append(s)
    gates=dict(error=estimable and all(m['local']['macro_capped_mae']<m[x]['macro_capped_mae']-1e-12 for x in ('global','shuffled')),
        recall=estimable and all(a['local']['recall']>=a[x]['recall']-1e-12 for x in ('raw','mono')),
        far_fpr=estimable and all(a['local']['far_fpr']<=a[x]['far_fpr']+1e-12 for x in ('raw','mono')),scenes=len(wins)>=3)
    verdict=('DISTANCE_ANCHORING_COMPONENT_DEV' if all(gates.values()) else 'STOP_MEDIAN_ANCHOR_RECIPE') if estimable else 'NOT_EVALUABLE'
    draws=np.random.default_rng(SEED).multinomial(4,np.full(4,.25),1000);bootstrap={}
    for control in ('raw','mono','depthor','global','crossfit','shuffled'):
        intervals={}
        for group,num,den in (('mixed','zone_mae_sum','zone_mae_n'),('sensor_valid','tp','near'),('sensor_valid','fp','far')):
            local=[];other=[];counts=[]
            for s in eval_scenes:
                q=scenes[s][group];local.append(q['arms']['local'][num]);other.append(q['arms'][control][num]);counts.append(q['arms']['local'][den] if den=='zone_mae_n' else q[den])
            d=draws@np.array(counts);ok=d>0;delta=(draws@(np.array(local)-np.array(other)))[ok]/d[ok]
            intervals[num]=dict(ci95=np.percentile(delta,[2.5,97.5]).tolist() if len(delta) else None,valid=int(ok.sum()))
        bootstrap[control]=intervals
    result=dict(verdict=verdict,gates=gates,scene_wins=wins,summaries=summaries,scenes=scenes,bootstrap=bootstrap,seconds=time.perf_counter()-start)
    save(OUT/'result.json',result);print(json.dumps(dict(verdict=verdict,gates=gates,scene_wins=wins)),flush=True)


def selftest():
    boxes=np.tile([0,0,8,8],(64,1));mask=np.ones(64,bool);anchor=np.full(64,2.);mono=np.ones((8,8));other=mono.copy()
    z,scale=observed(boxes,mask,anchor,mono,other,'fixture')
    assert scale['global_scale']==2 and scale['crossfit_scales']==[2.,2.] and np.all(z[0]['pred']['local']==2)
    gt=np.full((8,8),2.);s=scored(z[0],gt);assert s['arms']['local']['tp']==64 and s['arms']['local']['loss_sum']==0
    anchor[:]=np.nan;z,scale=observed(boxes,mask,anchor,mono,other,'missing');s=scored(z[0],gt)
    assert scale['global_scale'] is None and s['arms']['local']['unknown_near']==64 and s['arms']['local']['loss_sum']==640
    anchor[:]=2;anchor[0]=100;z,scale=observed(boxes,mask,anchor,mono,other,'invalid');assert not z[0]['sensor_valid'] and scale['anchors']==63
    # Changing an anchor in fold0 cannot alter its crossfit factor, trained only on fold1.
    anchor[:]=2;before=observed(boxes,mask,anchor,mono,other,'fold')[1];anchor[0]=5;after=observed(boxes,mask,anchor,mono,other,'fold')[1]
    assert before['crossfit_scales'][0]==after['crossfit_scales'][0]
    print('PASS_SCALE_MISSING_INVALID_CROSSFIT')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('selftest','prepare','run'))
    globals()[parser.parse_args().action]()

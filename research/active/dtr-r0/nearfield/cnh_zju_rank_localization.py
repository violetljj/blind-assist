"""Fixed-budget near-surface location from sealed RGB-only ZJU predictions.

No sensor distance distribution assumption, new inference or training. Published
fr rectangles are observation geometry; reference depth is evaluator-only.
Neither rank agreement nor top16 selection identifies a physical return source.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OLD=ROOT/'artifacts.local/work/ba-depth-probe-20260918'
OUT=ROOT/'artifacts.local/work/cnh-zju-rank-localization-20261002'
RUNS=ROOT/'research/active/dtr-r0/RUNS.md'
RUN_ID='CNH_ZJU_RANK_LOCALIZATION_20261002'
ARMS=('mono','shuffled','left','right','up','down')
SPATIAL=ARMS[2:]
SEED=2026100226


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def rng(key):return np.random.default_rng(int.from_bytes(hashlib.sha256((RUN_ID+'|'+key).encode()).digest()[:8],'little'))


def public_nodes(box,shape=(480,640)):
    """Clip the published half-open rectangle, grid sample, de-duplicate pixels."""
    y0,x0,y1,x1=map(int,box);h,w=shape
    clip=np.array([max(0,min(h,y0)),max(0,min(w,x0)),max(0,min(h,y1)),max(0,min(w,x1))])
    a,b,c,d=clip
    if c<=a or d<=b:return np.empty((0,2),np.int32),bool(np.any(clip!=box)),0
    yy=a+np.floor((np.arange(8)+.5)*(c-a)/8).astype(int)
    xx=b+np.floor((np.arange(8)+.5)*(d-b)/8).astype(int)
    y,x=np.meshgrid(yy,xx,indexing='ij')
    nodes=np.stack((y.ravel(),x.ravel()),axis=1)
    nodes=np.unique(nodes,axis=0)
    return nodes,bool(np.any(clip!=box)),64-len(nodes)


def observe_zone(box,mono,key):
    """Only projected footprint and sealed predicted depth; never reference."""
    nodes,clipped,duplicate=public_nodes(box,mono.shape)
    values=mono[nodes[:,0],nodes[:,1]].astype(np.float64)
    values[~np.isfinite(values)|(values<=0)]=np.nan
    n=len(nodes);permutation=rng(key+'|shuffle').permutation(n)
    scores=dict(mono=values,shuffled=values[permutation],left=nodes[:,1].astype(float),
        right=-nodes[:,1].astype(float),up=nodes[:,0].astype(float),down=-nodes[:,0].astype(float))
    tie_rank=np.argsort(rng(key+'|ties').permutation(n))
    selected={}
    for arm,score in scores.items():
        valid=np.flatnonzero(np.isfinite(score))
        selected[arm]=valid[np.lexsort((tie_rank[valid],score[valid]))[:16]]
    assert np.array_equal(np.sort(scores['mono']),np.sort(scores['shuffled']),equal_nan=True)
    return dict(nodes=nodes,clipped=clipped,duplicate=duplicate,scores=scores,selected=selected)


def prepare():
    assert not (OUT/'PLAN.json').exists()
    protocol=read(OLD/'protocol.json');seal=read(OLD/'prediction-seal.json')
    assert protocol['phase']=='EXPLORE_CONSUMED_PUBLIC_DIAGNOSTIC' and seal['protocol_sha256']==sha(OLD/'protocol.json')
    scenes=sorted({r['filename'].split('/')[0] for r in protocol['selected']},key=lambda s:sha_text('scene|'+s))
    assert len(scenes)==8
    roles={s:('cal' if i<4 else 'eval') for i,s in enumerate(scenes)}
    rows=[]
    for i,item in enumerate(protocol['selected']):
        source=OLD/'inputs'/item['filename'];prediction=OLD/'predictions'/f'{i:03d}-mono.npz'
        assert sha(source)==item['sha256'] and sha(prediction)==seal['files'][prediction.name]
        scene=item['filename'].split('/')[0]
        rows.append(dict(index=i,id=item['filename'],scene=scene,role=roles[scene],input=str(source),
            input_sha256=item['sha256'],prediction=str(prediction),prediction_sha256=seal['files'][prediction.name]))
    assert len(rows)==160 and Counter(r['scene'] for r in rows)==Counter({s:20 for s in scenes})
    OUT.mkdir(parents=True,exist_ok=True)
    line=(f'| 2026-10-02 | {RUN_ID} | PRE_RUN; existing consumed ZJU160/8scenes, hash4cal/4eval; original UniDepthV2 cache only, no new inference/fit. Every64 published fr/mask retained; clipped8x8 public nodes de-duplicated; mono vs within-grid fixedshuffle and4spatial directions, max16 observed candidates/zone; GT never selects candidates | NOT_RUN; main valid-mask mixed near<2.1/far>=2.1 zones: macro near recall and selected-known far contamination, selected UNKNOWN/missing budget retained; cal chooses spatial direction; eval must beat shuffle/spatial in macro recall without more far contamination and>=3scenes recall wins both. Pair ordering10/50cm auxiliary only | Pass=RANK_LOCATION_INFORMATION_DEV; otherwise STOP_CACHED_RANK_LOCALIZATION. No return-origin, metric calibration, corridor/alert or fresh confirmation claim; no retuning same cache or deleting failed scenes | `artifacts.local/work/cnh-zju-rank-localization-20261002/REPORT.md` |')
    body=RUNS.read_text(encoding='utf-8');assert RUN_ID not in body
    RUNS.write_text(body.rstrip()+'\n'+line+'\n',encoding='utf-8');(OUT/'prerun-row.txt').write_text(line+'\n',encoding='utf-8')
    save(OUT/'PLAN.json',dict(run_id=RUN_ID,at=datetime.now(timezone.utc).isoformat(),inputs=rows,roles=roles,
        source_sha256=sha(Path(__file__)),upstream={str(p):sha(p) for p in (OLD/'protocol.json',OLD/'prediction-seal.json')},
        rules=dict(arms=ARMS,grid='8x8 centres over clipped fr; nearest containing pixel via floor; unique native pixels, empty rectangles retain zero support',
            budget='At most16 finite prediction locations; no GT filtering; unused slots and selected reference UNKNOWN reported; spatial axes finite for every observed node',
            ties='Shared fixed public permutation rank, independent of shuffle permutation; per-zone ID SHA256 seeded PCG64',
            reference='finite .001<depth<10; near<2.1, far>=2.1; 0.6–2.1 subband separately, <.6 never far',
            primary='Published mask=True and sampled GT has both near and far; all/invalid-mask/thin1..8 near nodes separately; GT strata never influence output',
            selection='Four scene-hash cal scenes only: greatest mixed-zone macro near recall, then lowest pooled selected-known far fraction, then fixed direction order',
            decision='Eval mono macro recall greater than both shuffled and selected spatial, pooled selected-known far fraction no greater than either; >=3eval scenes individually have greater macro recall than both. If undefined main denominators or fewer3supported scenes: NOT_EVALUABLE; otherwise fail=>STOP_CACHED_RANK_LOCALIZATION.',
            pairs='All unique-node unordered pairs, reference-known nearer point<2.1 and separation>=.1/.5; correct1/tie.5/missing0 with abstention reported; not primary task',
            bootstrap='1000 paired whole eval scene resamples, seed2026100226; fixed selection; diagnostic intervals only, no additional gate',
            limitations='8consumed scenes; 4eval scene groups; overlapping zones and repeated frames correlated; sparse grid misses thinner surfaces; RealSense/reference registration uncertain; no body calibration or source-pixel return truth'),
        scope='Actual published fr and validity only; hist_data mean and ambiguous sigma/var never read or treated as q10/distribution',preregistration=line))
    print('PREPARED160',roles,flush=True)


def sha_text(value):return hashlib.sha256((RUN_ID+'|'+value).encode()).hexdigest()


def score_zone(observed,gt):
    nodes=observed['nodes'];d=gt[nodes[:,0],nodes[:,1]]
    known=np.isfinite(d)&(d>.001)&(d<10);near=known&(d<2.1);far=known&(d>=2.1)
    band=near&(d>=.6);ii,jj=np.triu_indices(len(nodes),1)
    pair_known=known[ii]&known[jj]
    separation=np.abs(d[ii]-d[jj]);nearer=np.minimum(d[ii],d[jj])
    truth_delta=d[ii]-d[jj]
    stats={}
    for arm,score in observed['scores'].items():
        chosen=observed['selected'][arm];finite=np.isfinite(score)
        rec=dict(selected_indices=chosen.tolist(),selected=len(chosen),unused_budget=16-len(chosen),
            missing_prediction=int((~finite).sum()),selected_near=int(near[chosen].sum()),selected_far=int(far[chosen].sum()),
            selected_unknown=int((~known[chosen]).sum()),selected_band_06_21=int(band[chosen].sum()),pairs={})
        delta=score[ii]-score[jj];available=finite[ii]&finite[jj]
        for margin in (.1,.5):
            take=pair_known&(nearer<2.1)&(separation>=margin)
            correct=take&available&(delta*truth_delta>0);ties=take&available&(delta==0)
            rec['pairs'][str(margin)]=dict(n=int(take.sum()),correct=int(correct.sum()),ties=int(ties.sum()),missing=int((take&~available).sum()))
        stats[arm]=rec
    return dict(nodes=len(nodes),duplicate_grid_nodes=observed['duplicate'],clipped=observed['clipped'],
        known=int(known.sum()),near=int(near.sum()),far=int(far.sum()),unknown=int((~known).sum()),
        band_06_21=int(band.sum()),mixed=bool(near.any() and far.any()),thin=bool(1<=near.sum()<=8 and far.any()),arms=stats)


def frame(row):
    assert sha(row['input'])==row['input_sha256'] and sha(row['prediction'])==row['prediction_sha256']
    with h5py.File(row['input'],'r') as f:boxes=f['fr'][:];mask=f['mask'][:]
    with np.load(row['prediction']) as f:mono=f['depth']
    assert boxes.shape==(64,4) and mask.shape==(64,) and mono.shape==(480,640)
    observations=[observe_zone(box,mono,row['id']+'|'+str(k)) for k,box in enumerate(boxes)]
    # All outputs above are fixed before reference depth is opened.
    with h5py.File(row['input'],'r') as f:gt=f['depth'][:]
    zones=[dict(zone=k,mask_valid=bool(mask[k]),**score_zone(obs,gt)) for k,obs in enumerate(observations)]
    return dict(id=row['id'],index=row['index'],scene=row['scene'],role=row['role'],zones=zones)


def aggregate(zones):
    result=dict(zones=len(zones),nodes=sum(z['nodes'] for z in zones),near=sum(z['near'] for z in zones),
        far=sum(z['far'] for z in zones),unknown=sum(z['unknown'] for z in zones),
        band_06_21=sum(z['band_06_21'] for z in zones),empty=sum(z['nodes']==0 for z in zones),
        clipped=sum(z['clipped'] for z in zones),duplicate_grid_nodes=sum(z['duplicate_grid_nodes'] for z in zones),arms={})
    for a in ARMS:
        keys=('selected','unused_budget','missing_prediction','selected_near','selected_far','selected_unknown','selected_band_06_21')
        s={k:sum(z['arms'][a][k] for z in zones) for k in keys}
        near_zones=[z for z in zones if z['near']>0]
        s['near_zone_count']=len(near_zones)
        s['macro_recall_sum']=sum(z['arms'][a]['selected_near']/z['near'] for z in near_zones)
        s['macro_near_recall']=s['macro_recall_sum']/len(near_zones) if near_zones else None
        s['pooled_near_recall']=s['selected_near']/result['near'] if result['near'] else None
        s['known_selected']=s['selected_near']+s['selected_far']
        s['far_contamination']=s['selected_far']/s['known_selected'] if s['known_selected'] else None
        s['pairs']={}
        for margin in ('.1','.5'):
            key=str(float(margin));c={k:sum(z['arms'][a]['pairs'][key][k] for z in zones) for k in ('n','correct','ties','missing')}
            c['accuracy']=(c['correct']+.5*c['ties'])/c['n'] if c['n'] else None;s['pairs'][key]=c
        result['arms'][a]=s
    return result


def summarize(frames):
    zones=[z for f in frames for z in f['zones']]
    filters=dict(all=lambda z:True,valid=lambda z:z['mask_valid'],invalid=lambda z:not z['mask_valid'],
        valid_mixed=lambda z:z['mask_valid'] and z['mixed'],valid_thin=lambda z:z['mask_valid'] and z['thin'])
    return {name:aggregate([z for z in zones if fn(z)]) for name,fn in filters.items()}


def run():
    plan=read(OUT/'PLAN.json');assert not (OUT/'result.json').exists()
    assert sha(Path(__file__))==plan['source_sha256']
    for path,digest in plan['upstream'].items():assert sha(path)==digest
    start=time.perf_counter();frames=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(pool.map(frame,plan['inputs'])):
            frames.append(f)
            if (i+1)%20==0:print(f'scored {i+1}/160',flush=True)
    save(OUT/'frame-ledger.json',frames)
    summaries={s:summarize([f for f in frames if f['role']==s]) for s in ('cal','eval')}
    summaries['all']=summarize(frames)
    scenes={s:summarize([f for f in frames if f['scene']==s]) for s in plan['roles']}
    cal=summaries['cal']['valid_mixed']['arms']
    def key(a):
        x=cal[a];return (x['macro_near_recall'] if x['macro_near_recall'] is not None else -1,
            -x['far_contamination'] if x['far_contamination'] is not None else -2,-SPATIAL.index(a))
    eligible=[a for a in SPATIAL if cal[a]['macro_near_recall'] is not None and cal[a]['far_contamination'] is not None]
    chosen=max(eligible,key=key) if eligible else None
    controls=('shuffled',chosen) if chosen is not None else ('shuffled',)
    evaluation=summaries['eval']['valid_mixed']['arms'];mono=evaluation['mono']
    differences={}
    for a in controls:
        q=evaluation[a]
        differences[a]={k:100*(mono[k]-q[k]) if mono[k] is not None and q[k] is not None else None for k in ('macro_near_recall','far_contamination')}
    eval_scenes=[s for s,v in plan['roles'].items() if v=='eval']
    scene_wins=[];supported=[]
    for s in eval_scenes:
        arms=scenes[s]['valid_mixed']['arms']
        if chosen is not None and all(arms[a]['macro_near_recall'] is not None for a in ('mono',*controls)):
            supported.append(s)
            if all(arms['mono']['macro_near_recall']-arms[a]['macro_near_recall']>1e-12 for a in controls):scene_wins.append(s)
    estimable=chosen is not None and len(supported)>=3 and all(v is not None for d in differences.values() for v in d.values())
    passed=estimable and len(scene_wins)>=3 and all(d['macro_near_recall']>1e-10 and d['far_contamination']<=1e-10 for d in differences.values())
    verdict='RANK_LOCATION_INFORMATION_DEV' if passed else ('STOP_CACHED_RANK_LOCALIZATION' if estimable else 'NOT_EVALUABLE')
    draws=np.random.default_rng(SEED).multinomial(4,np.full(4,.25),size=1000)
    intervals={}
    for a in controls:
        arms=[scenes[s]['valid_mixed']['arms'] for s in eval_scenes]
        den=draws@np.array([v['mono']['near_zone_count'] for v in arms]);good=den>0
        rec=100*(draws@np.array([v['mono']['macro_recall_sum']-v[a]['macro_recall_sum'] for v in arms]))[good]/den[good]
        den_m=draws@np.array([v['mono']['known_selected'] for v in arms]);den_c=draws@np.array([v[a]['known_selected'] for v in arms]);valid=(den_m>0)&(den_c>0)
        far=100*((draws@np.array([v['mono']['selected_far'] for v in arms]))[valid]/den_m[valid]-(draws@np.array([v[a]['selected_far'] for v in arms]))[valid]/den_c[valid])
        intervals[a]=dict(macro_recall_ci95_pp=np.percentile(rec,[2.5,97.5]).tolist() if len(rec) else None,recall_valid=int(good.sum()),
            far_contamination_ci95_pp=np.percentile(far,[2.5,97.5]).tolist() if len(far) else None,far_valid=int(valid.sum()))
    result=dict(verdict=verdict,roles=plan['roles'],chosen_spatial=chosen,cal_eligible_spatial=eligible,summaries=summaries,scene_summaries=scenes,
        differences_pp=differences,scene_wins=scene_wins,supported_scenes=supported,bootstrap=intervals,seconds=time.perf_counter()-start,
        scope=plan['scope'],limits=plan['rules']['limitations'])
    save(OUT/'result.json',result);print(json.dumps(dict(verdict=verdict,chosen=chosen,differences=differences,scene_wins=scene_wins)),flush=True)


def selftest():
    nodes,clip,dup=public_nodes([-4,0,4,2],(10,10));assert clip and len(nodes)==8 and dup==56
    nodes,clip,dup=public_nodes([-4,0,-1,2],(10,10));assert len(nodes)==0 and clip
    prediction=np.arange(64,dtype=float).reshape(8,8)+1
    o=observe_zone([0,0,8,8],prediction,'fixture')
    assert len(o['selected']['mono'])==16 and np.array_equal(o['selected']['mono'],np.arange(16))
    gt=np.where(prediction<=16,1.,3.);s=score_zone(o,gt)
    assert s['arms']['mono']['selected_near']==16 and s['arms']['mono']['selected_far']==0 and s['near']==16
    assert s['arms']['mono']['pairs']['0.1']==dict(n=16*48,correct=16*48,ties=0,missing=0)
    gt[:2]=np.nan;s=score_zone(o,gt)
    assert s['arms']['mono']['selected_unknown']==16 and o['selected']['mono'].tolist()==list(range(16))
    p=prediction.copy();p[:]=np.nan;o=observe_zone([0,0,8,8],p,'missing');s=score_zone(o,np.ones((8,8)))
    assert s['arms']['mono']['selected']==0 and s['arms']['mono']['unused_budget']==16
    print('PASS_PUBLIC_GRID_DEDUP_SHUFFLE_GT_INDEPENDENCE_MISSING')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('selftest','prepare','run'))
    globals()[parser.parse_args().action]()

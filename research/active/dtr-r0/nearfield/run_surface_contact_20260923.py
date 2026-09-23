"""Frozen-depth camera-forward contact experiment; predictions sealed before truth."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from collections import Counter
from pathlib import Path
import numpy as np
import mz101_spatial as spatial

ROOT=Path(__file__).resolve().parents[4]
ART=ROOT/'artifacts.local'
BASE=ART/'evidence/ba-surface-contact-20260923'
PRED=ART/'evidence/ba-surface-contact-20260923-predictions-v1'
EVAL=ART/'evidence/ba-surface-contact-20260923-evaluation-v1'
MODELS={'unguided':ART/'evidence/ba-foundation-geometry-20260923-frontend-v1',
        'guided':ART/'evidence/ba-vpp-geometry-20260923-frontend-v1'}
CAPTURES={'mz101':ART/'work/mz101-stereo-tof-spatial-20260912/capture-v1',
          'mz102':ART/'work/mz102-stereo-support-20260912/capture-v1'}
PUBLIC=ART/'evidence/ba-vpp-geometry-20260923-prepared'
BANDS={'BODY':(-1.05,-.30),'HEAD':(-.30,.15)}
WIDTHS={'BODY':(.36,.56,.76),'HEAD':(.24,.36,.48)}
REPS=('raw','fitted','surface')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def fresh(p):
    assert not p.exists() or not any(p.iterdir()),'Preserve previous evidence'
    p.mkdir(parents=True,exist_ok=True)
def guard():
    j=read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])
    assert j['state']=='running' and j['reuse_preflight']['status']=='PASS'
    for p,h in read(BASE/'plan/launch-seal.json')['hashes'].items():assert sha(p)==h,p
def finite(v):return None if v is None or not np.isfinite(v) else float(v)
def fov(p):
    if not len(p):return p
    return p[(p[:,0]>=.5)&(p[:,0]<=4)&(np.abs(p[:,1])<=p[:,0]*np.tan(np.radians(22.5)))&(np.abs(p[:,2])<=p[:,0]*np.tan(np.radians(20)))]
def point_contact(p,lo,hi,axis='x'):
    q=p[((p>=lo)&(p<=hi)).all(axis=1)]
    if not len(q):return None
    return float(q[:,0].min() if axis=='x' else np.abs(q[:,1]).min())
def queries(fn):
    distances={};widths={}
    for part,(zlo,zhi) in BANDS.items():
        distances[part]=[finite(fn([.5,-w/2,zlo],[4,w/2,zhi],'x')) for w in WIDTHS[part]]
        v=fn([.5,-.5,zlo],[3,.5,zhi],'abs_y');widths[part]=finite(None if v is None else 2*v)
    return dict(distance=distances,width=widths)
def minimum(a,b):return b if a is None else a if b is None else min(a,b)

def predict(canary=False):
    import surface_contact_core as core
    guard();dest=BASE/'canary' if canary else PRED;fresh(dest)
    inputs=read(PUBLIC/'rgb-inputs.json');tof=read(PUBLIC/'tof-inputs.json')
    assert len(inputs['frames'])==len(tof['frames'])==576
    assert [(x['panel'],x['id']) for x in inputs['frames']]==[(x['panel'],x['id']) for x in tof['frames']]
    receipts={m:read(p/'receipt.json') for m,p in MODELS.items()}
    rows=[];start=time.perf_counter()
    for i,(entry,t) in enumerate(zip(inputs['frames'],tof['frames'])):
        if canary and i:break
        panel,id=entry['panel'],entry['id'];r=dict(panel=panel,id=id,models={})
        for k in ('ranges','valid'):assert sha(t[k])==t[k+'_sha256']
        points=fov(spatial.tof_points(np.load(t['ranges']),np.load(t['valid'])))
        r['tof']=queries(lambda lo,hi,axis:point_contact(points,lo,hi,axis))
        for model,path in MODELS.items():
            rel=f'{panel}/depth/{id}.npy';assert sha(path/rel)==receipts[model]['hashes'][rel]
            t0=time.perf_counter();g=core.depth_to_geometry(np.load(path/rel),backend='cuda')
            r['models'][model]={rep:queries(lambda lo,hi,axis:core.contact(g,lo,hi,axis=axis,representation=rep)) for rep in REPS}
            r['models'][model]['metadata']=g['metadata']
            r['models'][model]['seconds']=time.perf_counter()-t0
        rows.append(r)
        if (i+1)%24==0:print('CONTACT_PREDICTIONS',i+1,flush=True)
    write(dest/'predictions.json',rows)
    write(dest/'receipt.json',dict(status='PASS',frames=len(rows),elapsed_s=time.perf_counter()-start,
        launch_seal_sha256=sha(BASE/'plan/launch-seal.json'),producer_sha256=sha(__file__),
        core_sha256=sha(Path(core.__file__)),inputs={m:sha(p/'receipt.json') for m,p in MODELS.items()},
        tof_manifest_sha256=sha(PUBLIC/'tof-inputs.json'),hashes={'predictions.json':sha(dest/'predictions.json')}))

def boundary_stats(rows,key='distance'):
    truth=[r for r in rows if r['truth'] is not None]
    both=[r for r in truth if r['pred'] is not None]
    errors=[abs(r['pred']-r['truth']) for r in both]
    negatives=[r for r in rows if r['truth'] is None]
    return dict(queries=len(rows),truth_contact=len(truth),predicted_contact=sum(r['pred'] is not None for r in rows),
        missing_contact=sum(r['pred'] is None for r in truth),unknown=sum(r['pred'] is None for r in rows),
        correct_5cm=sum(e<=.05 for e in errors),hit_5cm=sum(e<=.05 for e in errors)/len(truth) if truth else None,
        conditional_mae_m=float(np.mean(errors)) if errors else None,
        conditional_max_error_m=max(errors) if errors else None,
        negative_queries=len(negatives),false_contact=sum(r['pred'] is not None for r in negatives),
        false_contact_rate=sum(r['pred'] is not None for r in negatives)/len(negatives) if negatives else None)
def counts(pred,truth):
    a=np.asarray(pred,bool);b=np.asarray(truth,bool)
    return dict(TP=int((a&b).sum()),FP=int((a&~b).sum()),FN=int((~a&b).sum()),TN=int((~a&~b).sum()))
def evaluate():
    import surface_contact_oracle as oracle
    guard();fresh(EVAL);receipt=read(PRED/'receipt.json')
    assert receipt['frames']==576 and receipt['hashes']['predictions.json']==sha(PRED/'predictions.json')
    predictions=read(PRED/'predictions.json');truth_rows=[];distance_rows=[];width_rows=[]
    arms=[f'{m}_{r}{s}' for m in MODELS for r in REPS for s in ('','_union')]
    manifests={};specs={};capreceipts={}
    for panel,path in CAPTURES.items():
        rec=read(path/'receipt.json');capreceipts[panel]=rec
        assert sha(path/'manifest.json')==rec['hashes']['manifest.json']
        assert sha(path/'spec.json')==rec['spec_sha256']
        manifests[panel]={x['id']:x for x in read(path/'manifest.json')['frames']}
        specs[panel]={x['id']:x for x in read(path/'spec.json')['frames']}
    old_geometry=read(ART/'evidence/ba-vpp-geometry-20260923-evaluation-v1/geometry-queries.json')
    hard={(r['panel'],r['id'],r['part']) for r in old_geometry if r['model']=='guided' and not r['truth'] and r['origin']['native_far']>0}
    for i,row in enumerate(predictions):
        panel,id=row['panel'],row['id'];frame=manifests[panel][id];spec=specs[panel][id];rec=capreceipts[panel]
        extra=tuple(rec[k] for k in ('background','floor') if rec.get(k))
        npth=CAPTURES[panel]/f'frame/{id}/native-left-depth.npy'
        assert sha(npth)==rec['hashes'][f'frame/{id}/native-left-depth.npy']
        native=np.load(npth);native=np.where(np.isfinite(native)&(native>=.5)&(native<=4),native,np.nan)
        npoints=fov(spatial.depth_points(native));nquery=queries(lambda lo,hi,axis:point_contact(npoints,lo,hi,axis))
        truth=dict(distance={part:[finite(oracle.oracle(frame,part,w,extra_bounds=extra)) for w in WIDTHS[part]] for part in BANDS},
            width={part:finite(oracle.oracle_width(frame,part,extra_bounds=extra)) for part in BANDS})
        common=dict(panel=panel,id=id,family=spec['family'],episode=spec['episode'],time_s=spec['time_s'])
        truth_rows.append(dict(common,truth=truth,native_visible=nquery))
        for model in MODELS:
            for rep in REPS:
                pred=row['models'][model][rep]
                for suffix in ('','_union'):
                    arm=f'{model}_{rep}{suffix}'
                    for part in BANDS:
                        for wi,w in enumerate(WIDTHS[part]):
                            value=pred['distance'][part][wi]
                            if suffix:value=minimum(value,row['tof']['distance'][part][wi])
                            distance_rows.append(dict(common,arm=arm,part=part,width=w,pred=value,truth=truth['distance'][part][wi],
                                native_visible=nquery['distance'][part][wi],old_negative_far=(panel,id,part) in hard))
                        value=pred['width'][part]
                        if suffix:value=minimum(value,row['tof']['width'][part])
                        width_rows.append(dict(common,arm=arm,part=part,pred=value,truth=truth['width'][part],native_visible=nquery['width'][part]))
        if (i+1)%48==0:print('CONTACT_EVALUATION',i+1,flush=True)
    families=sorted({r['family'] for r in distance_rows})
    summaries={}
    for arm in arms:
        d=[r for r in distance_rows if r['arm']==arm];w=[r for r in width_rows if r['arm']==arm]
        slices={'all':lambda r:True,'nonwall':lambda r:r['family']!='wall','BODY':lambda r:r['part']=='BODY','HEAD':lambda r:r['part']=='HEAD'}
        slices.update({f:lambda r,f=f:r['family']==f for f in families})
        summaries[arm]=dict(distance={s:boundary_stats([r for r in d if fn(r)]) for s,fn in slices.items()},
            width={s:boundary_stats([r for r in w if fn(r)]) for s,fn in slices.items()},
            width_positive=boundary_stats([r for r in w if r['truth'] is not None and r['truth']>1e-8]),
            width_zero=boundary_stats([r for r in w if r['truth'] is not None and r['truth']<=1e-8]),
            old_negative_far=boundary_stats([r for r in d if r['old_negative_far']]),
            native_visible_absent=boundary_stats([r for r in d if r['native_visible'] is None]))
    alerts={};alert_rows=[]
    for arm in arms:
        panels={}
        for panel in CAPTURES:
            rows=[r for r in distance_rows if r['arm']==arm and r['panel']==panel and r['width']==WIDTHS[r['part']][1]]
            assert len(rows)==576
            a=np.array([r['pred'] is not None and r['pred']<=3 for r in rows]).reshape(-1,2)
            gt=np.array([r['truth'] is not None and r['truth']<=3 for r in rows]).reshape(-1,2)
            episodes=[r['episode'] for r in rows[::2]];final=spatial.hysteresis(a,episodes)
            panels[panel]=dict(raw=counts(a,gt),final=counts(final,gt))
            for k,r in enumerate(rows):alert_rows.append(dict(panel=panel,id=r['id'],part=r['part'],arm=arm,truth=bool(gt.flat[k]),raw=bool(a.flat[k]),final=bool(final.flat[k])))
        alerts[arm]={stage:{k:sum(p[stage][k] for p in panels.values()) for k in ('TP','FP','FN','TN')} for stage in ('raw','final')}
        alerts[arm]['panels']=panels
    write(EVAL/'truth.json',truth_rows);write(EVAL/'distance-queries.json',distance_rows);write(EVAL/'width-queries.json',width_rows);write(EVAL/'alerts.json',alert_rows)
    write(EVAL/'summary.json',dict(status='PASS',frames=576,evidence='CONSUMED_RENDERED_DEVELOPMENT_CAMERA_ALIGNED_NEW_TASK',
        summaries=summaries,alerts=alerts,definitions=dict(distance='min camera-forward x in0.5..4m, not body travel',width='critical FULL width in1m cap, horizon3m',
        missing='UNKNOWN; fixedtruth positive denominator failure',truth='actualnativecuboid LP incl floor/background, NOT oldbodylabels',
        native='secondary sampled-visible point reference, not complete scene truth'),prediction_receipt_sha256=sha(PRED/'receipt.json')))
    write(EVAL/'receipt.json',dict(status='PASS',evaluator_sha256=sha(__file__),oracle_sha256=sha(Path(oracle.__file__)),
        hashes={p.name:sha(p) for p in EVAL.iterdir() if p.is_file()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['predict','canary','evaluate']);a=p.parse_args()
    if a.stage=='evaluate':evaluate()
    else:predict(a.stage=='canary')

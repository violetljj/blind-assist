"""Independent sealed contact-record accounting and frozen raw-point replay.

No model, local fit or LP oracle is rerun. Oracle correctness is covered by the
separate analytic fixtures, not claimed independently from reused LP code.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4];ART=ROOT/'artifacts.local'
BASE=ART/'evidence/ba-surface-contact-20260923'
PRED=ART/'evidence/ba-surface-contact-20260923-predictions-v1'
EVAL=ART/'evidence/ba-surface-contact-20260923-evaluation-v1'
MODELS={'unguided':ART/'evidence/ba-foundation-geometry-20260923-frontend-v1',
        'guided':ART/'evidence/ba-vpp-geometry-20260923-frontend-v1'}
CAPTURES={'mz101':ART/'work/mz101-stereo-tof-spatial-20260912/capture-v1',
          'mz102':ART/'work/mz102-stereo-support-20260912/capture-v1'}
PUBLIC=ART/'evidence/ba-vpp-geometry-20260923-prepared'
BANDS={'BODY':(-1.05,-.3),'HEAD':(-.3,.15)}
WIDTHS={'BODY':(.36,.56,.76),'HEAD':(.24,.36,.48)}
REPS=('raw','fitted','surface')
ARMS=tuple(f'{m}_{r}{suffix}' for m in MODELS for r in REPS for suffix in ('','_union'))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def statistics(rows):
    positive=negative=predicted=missing=unknown=correct=false=0;errors=[]
    for row in rows:
        a,b=row['pred'],row['truth'];predicted+=a is not None;unknown+=a is None
        if b is None:
            negative+=1;false+=a is not None
        else:
            positive+=1
            if a is None:missing+=1
            else:
                error=abs(a-b);errors.append(error);correct+=error<=.05
    return dict(queries=len(rows),truth_contact=positive,predicted_contact=predicted,
        missing_contact=missing,unknown=unknown,correct_5cm=correct,
        hit_5cm=correct/positive if positive else None,
        conditional_mae_m=sum(errors)/len(errors) if errors else None,
        conditional_max_error_m=max(errors) if errors else None,
        negative_queries=negative,false_contact=false,false_contact_rate=false/negative if negative else None)


def alert_counts(pred,truth):
    out=dict(TP=0,FP=0,FN=0,TN=0)
    for p,g in zip(pred,truth):out['TP' if p and g else 'FP' if p else 'FN' if g else 'TN']+=1
    return out


def debounce(rows):
    """Rows are one part in chronological source order, episode-local state."""
    result=[];last=None;yes=no=0;active=False
    for row in rows:
        if row['episode']!=last:yes=no=0;active=False;last=row['episode']
        p=row['pred'] is not None and row['pred']<=3
        if p:
            yes+=1;no=0
            if yes==2:active=True
        else:
            no+=1;yes=0
            if no==2:active=False
        result.append(active)
    return result


def raw_points(depth):
    y,x=np.indices((360,640));f=320/math.tan(math.radians(35))
    rx=(x-320)/f;rz=-(y-180)/f
    good=np.isfinite(depth)&(depth>=.5)&(depth<=4)&(np.abs(np.degrees(np.arctan(rx)))<=22.5)&(np.abs(np.degrees(np.arctan(rz)))<=20)
    return np.column_stack((depth[good],rx[good]*depth[good],rz[good]*depth[good]))


def point_queries(points):
    result=dict(distance={},width={})
    for part,(bottom,top) in BANDS.items():
        vertical=(points[:,2]>=bottom)&(points[:,2]<=top)
        distances=[]
        for w in WIDTHS[part]:
            p=points[vertical&(np.abs(points[:,1])<=w/2)&(points[:,0]>=.5)&(points[:,0]<=4)]
            distances.append(float(p[:,0].min()) if len(p) else None)
        result['distance'][part]=distances
        p=points[vertical&(np.abs(points[:,1])<=.5)&(points[:,0]>=.5)&(points[:,0]<=3)]
        result['width'][part]=float(2*np.abs(p[:,1]).min()) if len(p) else None
    return result


def tof_points(ranges,valid):
    points=[]
    for i in range(64):
        distance=float(ranges[i])
        if not valid[i] or not math.isfinite(distance) or not .5<=distance<=4:continue
        row,col=divmod(i,8)
        right=math.tan(math.radians((col-3.5)*45/8));up=math.tan(math.radians((3.5-row)*45/8))
        forward=distance/math.sqrt(1+right*right+up*up)
        if .5<=forward<=4:points.append([forward,right*forward,up*forward])
    return np.asarray(points,dtype=np.float64).reshape(-1,3)


def union(a,b):
    values=[v for v in (a,b) if v is not None]
    return min(values) if values else None


def audit(pred,evaluation,output):
    assert read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])['state']=='running'
    assert output.resolve().is_relative_to(ART.resolve()) and not output.exists()
    start=time.perf_counter();checks=0
    def check(v):
        nonlocal checks
        assert v
        checks+=1
    def equal(a,b):
        if isinstance(b,dict):
            check(set(a)==set(b))
            for k in b:equal(a[k],b[k])
        elif isinstance(b,list):
            check(len(a)==len(b))
            for x,y in zip(a,b):equal(x,y)
        elif isinstance(b,float):check(bool(np.isclose(a,b,rtol=1e-9,atol=1e-10)))
        else:check(a==b)
    def verify(folder):
        receipt=read(folder/'receipt.json');check(receipt['status']=='PASS')
        for name,digest in receipt['hashes'].items():
            p=folder/name.replace('\\','/');check(p.resolve().is_relative_to(folder.resolve()));check(sha(p)==digest)
        return receipt
    pr,er=verify(pred),verify(evaluation)
    check(pr['frames']==576)
    check(pr['launch_seal_sha256']==sha(BASE/'plan/launch-seal.json'))
    for p,h in read(BASE/'plan/launch-seal.json')['hashes'].items():check(sha(p)==h)
    for name,digest in [('run_surface_contact_20260923.py',er['evaluator_sha256']),('surface_contact_oracle.py',er['oracle_sha256']),('surface_contact_core.py',pr['core_sha256'])]:
        check(sha(Path(__file__).parent/name)==digest)
    manifest=read(PUBLIC/'rgb-inputs.json');tof=read(PUBLIC/'tof-inputs.json')
    check(sha(PUBLIC/'tof-inputs.json')==pr['tof_manifest_sha256'])
    frames=manifest['frames'];keys=[(r['panel'],r['id']) for r in frames]
    check(len(keys)==len(set(keys))==576)
    check([(r['panel'],r['id']) for r in tof['frames']]==keys)
    predictions=read(pred/'predictions.json');check([(r['panel'],r['id']) for r in predictions]==keys)
    by_frame={(r['panel'],r['id']):r for r in predictions}
    source_receipts={m:read(path/'receipt.json') for m,path in MODELS.items()}
    for m,path in MODELS.items():check(source_receipts[m]['status']=='PASS' and sha(path/'receipt.json')==pr['inputs'][m])
    vertex_count=0
    for r,t in zip(predictions,tof['frames']):
        check(set(r['models'])==set(MODELS))
        for name in ('ranges','valid'):check(sha(t[name])==t[name+'_sha256'])
        equal(r['tof'],point_queries(tof_points(np.load(t['ranges']),np.load(t['valid']))))
        for model,path in MODELS.items():
            record=r['models'][model];meta=record['metadata'];check(set(record)==set(REPS)|{'metadata','seconds'})
            rel=f"{r['panel']}/depth/{r['id']}.npy";check(sha(path/rel)==source_receipts[model]['hashes'][rel])
            points=raw_points(np.load(path/rel));equal(record['raw'],point_queries(points))
            check(meta['valid_vertices']==len(points));vertex_count+=len(points)
            check(meta['vertices_deleted']==meta['missing_pixels_filled']==0)
            check(0<=meta['fitted_vertices']<=meta['valid_vertices'] and 0<=meta['moved_vertices']<=meta['valid_vertices'])
            for part in BANDS:
                for rep in REPS:
                    ds=record[rep]['distance'][part];check(len(ds)==3)
                    for a,b in zip(ds,ds[1:]):
                        if a is not None:check(b is not None and b<=a+1e-9)
                    for v in ds:
                        if v is not None:check(.5-1e-9<=v<=4+1e-9)
                    w=record[rep]['width'][part]
                    if w is not None:check(0<=w<=1+1e-9)
                for a,b in zip(record['fitted']['distance'][part],record['surface']['distance'][part]):
                    if a is not None:check(b is not None and b<=a+1e-9)
                a,b=record['fitted']['width'][part],record['surface']['width'][part]
                if a is not None:check(b is not None and b<=a+1e-9)
    truth=read(evaluation/'truth.json');check([(r['panel'],r['id']) for r in truth]==keys)
    truth_map={(r['panel'],r['id']):r for r in truth}
    # Read metadata only: no new native-depth or scene-oracle cohort computation.
    for panel,cap in CAPTURES.items():
        receipt=read(cap/'receipt.json');check(receipt['status']=='PASS')
        check(sha(cap/'manifest.json')==receipt['hashes']['manifest.json']);check(sha(cap/'spec.json')==receipt['spec_sha256'])
        spec=read(cap/'spec.json')
        for r in spec['frames']:
            tr=truth_map[(panel,r['id'])]
            for k in ('family','episode','time_s'):equal(tr[k],r[k])
    distances=read(evaluation/'distance-queries.json');widths=read(evaluation/'width-queries.json');alerts=read(evaluation/'alerts.json')
    check(len(distances)==576*12*6 and len(widths)==len(alerts)==576*12*2)
    dm={};wm={};am={}
    for group,kind,mapping in [(distances,'distance',dm),(widths,'width',wm)]:
        for row in group:
            key=(row['panel'],row['id']);check(key in by_frame and row['arm'] in ARMS and row['part'] in BANDS)
            arm=row['arm'];model,rep=arm.split('_')[:2];part=row['part'];tr=truth_map[key]
            if kind=='distance':
                wi=WIDTHS[part].index(row['width']);rk=(*key,arm,part,row['width'])
                value=by_frame[key]['models'][model][rep][kind][part][wi]
                other=by_frame[key]['tof'][kind][part][wi]
                equal(row['truth'],tr['truth'][kind][part][wi]);equal(row['native_visible'],tr['native_visible'][kind][part][wi])
            else:
                rk=(*key,arm,part);value=by_frame[key]['models'][model][rep][kind][part];other=by_frame[key]['tof'][kind][part]
                equal(row['truth'],tr['truth'][kind][part]);equal(row['native_visible'],tr['native_visible'][kind][part])
            check(rk not in mapping);mapping[rk]=row
            equal(row['pred'],union(value,other) if arm.endswith('_union') else value)
            for k in ('family','episode','time_s'):equal(row[k],tr[k])
    for row in alerts:
        key=(row['panel'],row['id'],row['arm'],row['part']);check(key not in am);am[key]=row
    report=read(evaluation/'summary.json');check(report['status']=='PASS' and report['prediction_receipt_sha256']==sha(pred/'receipt.json'))
    families=sorted({r['family'] for r in distances})
    for arm in ARMS:
        d=[r for r in distances if r['arm']==arm];w=[r for r in widths if r['arm']==arm]
        slices={'all':lambda r:True,'nonwall':lambda r:r['family']!='wall','BODY':lambda r:r['part']=='BODY','HEAD':lambda r:r['part']=='HEAD'}
        slices.update({family:lambda r,f=family:r['family']==f for family in families})
        expected=dict(distance={name:statistics([r for r in d if fn(r)]) for name,fn in slices.items()},
            width={name:statistics([r for r in w if fn(r)]) for name,fn in slices.items()},
            width_positive=statistics([r for r in w if r['truth'] is not None and r['truth']>1e-8]),
            width_zero=statistics([r for r in w if r['truth'] is not None and r['truth']<=1e-8]),
            old_negative_far=statistics([r for r in d if r['old_negative_far']]),
            native_visible_absent=statistics([r for r in d if r['native_visible'] is None]))
        equal(report['summaries'][arm],expected)
        panel_counts={}
        for panel in CAPTURES:
            scored={stage:dict(pred=[],truth=[]) for stage in ('raw','final')}
            ids=[key for key in keys if key[0]==panel]
            for part in BANDS:
                rr=[dm[(*key,arm,part,WIDTHS[part][1])] for key in ids]
                final=debounce(rr)
                for i,row in enumerate(rr):
                    raw=row['pred'] is not None and row['pred']<=3;gt=row['truth'] is not None and row['truth']<=3
                    saved=am[(panel,row['id'],arm,part)];equal(saved['truth'],gt);equal(saved['raw'],raw);equal(saved['final'],final[i])
                    for stage,p in [('raw',raw),('final',final[i])]:scored[stage]['pred'].append(p);scored[stage]['truth'].append(gt)
            panel_counts[panel]={stage:alert_counts(v['pred'],v['truth']) for stage,v in scored.items()}
        expected={stage:{k:sum(p[stage][k] for p in panel_counts.values()) for k in ('TP','FP','FN','TN')} for stage in ('raw','final')}
        expected['panels']=panel_counts;equal(report['alerts'][arm],expected)
    hard=[r for r in distances if r['family']=='small_head' and r['episode']=='small_head_flat' and r['part']=='HEAD']
    check(len(hard)==12*12*3)
    absence={arm:statistics([r for r in hard if r['arm']==arm]) for arm in ARMS}
    result=dict(status='PASS',assertions=checks,frames=576,model_frame_raw_geometry_replays=1152,
        retained_source_vertex_count=vertex_count,distance_records=len(distances),width_records=len(widths),alert_records=len(alerts),
        small_head_flat_HEAD_complete=absence,prediction_receipt_sha256=sha(pred/'receipt.json'),evaluation_receipt_sha256=sha(evaluation/'receipt.json'),
        script_sha256=sha(__file__),elapsed_s=time.perf_counter()-start,
        backend=dict(device='CPU',reason='TASK_NOT_GPU_SUITABLE',scope='IO_ETL sealed-record arithmetic and rawpoint replay'),
        limits='No model/local-fit/native-depth/LP cohort rerun. All raw query minima and original valid cardinalities independently replayed. '
               'Fitted-point retention is metadata/construction evidence because fitted arrays are not exported. '
               'Independent analytic oracle fixtures validate exactLP; this audit checks sealed truth references and summary arithmetic, not new oracle truth. '
               'All alert hysteresis, fixed denominators, union and width/surface monotonicity independently checked.',fits=0)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('SURFACE_CONTACT_AUDIT_PASS',checks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,default=PRED);p.add_argument('--evaluation',type=Path,default=EVAL);p.add_argument('--output',type=Path,default=ART/'evidence/ba-surface-contact-20260923-audit-v1/result.json')
    a=p.parse_args();audit(a.predictions,a.evaluation,a.output)

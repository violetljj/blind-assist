"""Independent post-seal accounting and axial geometry audit; no model inference."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

REPO = Path(__file__).resolve().parents[4]
WORK = REPO/'artifacts.local/work'
REFERENCE = WORK/'mz103-depth-frontend-20260912/native-v1'
TAO = WORK/'mz104-foundation-stereo-20260912/frontend-v1'
PANELS = {'mz101': ('mz101-stereo-tof-spatial-20260912','comparison-v1/stereo-depth'),
          'mz102': ('mz102-stereo-support-20260912','fresh-v1/depth')}
PARTS = ('BODY','HEAD')
BOXES = (((.18,-.28,.65),(3.18,.28,1.4)), ((.13,-.18,1.4),(3.13,.18,1.85)))
CRITICAL = ('thin_left','thin_right','small_head','occluded_thin')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def independent_geometry(pred, native, pose):
    """Full image-coordinate algebra, no production geometry/readout imports."""
    yy, xx = np.indices((360,640), dtype=np.float64)
    f = 320/math.tan(math.radians(35))
    yaw,pitch,roll = [math.radians(pose.get(k,0)) for k in ('yaw','pitch','roll')]
    cy,sy,cp,sp,cr,sr = math.cos(yaw),math.sin(yaw),math.cos(pitch),math.sin(pitch),math.cos(roll),math.sin(roll)
    rot = ((cp*cy,sr*sp*cy-cr*sy,-cr*sp*cy-sr*sy),
           (cp*sy,sr*sp*sy+cr*cy,cy*sr-cr*sp*sy), (sp,-sr*cp,cr*cp))
    def masks(depth):
        z = depth.astype(np.float64)
        ok = np.isfinite(z) & (z>=.5) & (z<=4)
        right = (xx-320)*z/f; up = -(yy-180)*z/f
        common = (np.abs(np.arctan2(right,z)) <= math.radians(22.5)) & (np.abs(np.arctan2(up,z)) <= math.radians(20))
        body = [rot[k][0]*z+rot[k][1]*right+rot[k][2]*up+pose['camera_in_body_m'][k] for k in range(3)]
        return [ok & common & np.logical_and.reduce([(body[k]>=lo[k]) & (body[k]<=hi[k]) for k in range(3)]) for lo,hi in BOXES]
    pin,nin = masks(pred),masks(native)
    finite = np.isfinite(pred) & (pred>=.5) & (pred<=4)
    near = np.isfinite(native) & (native>=.5) & (native<=4)
    far = np.isfinite(native) & (native>4)
    # Production receives float32 arrays; preserve that subtraction rounding.
    errors = np.abs(pred-native)
    rows=[]
    for p,n in zip(pin,nin):
        origin=dict(pixels=int(p.sum()), native_in_query=int((p&n).sum()), native_far=int((p&far).sum()),
                    native_near_outside=int((p&near&~n).sum()), other=int((p&~near&~far).sum()))
        coverage=dict(native_query_pixels=int(n.sum()), predicted_valid=int((n&finite).sum()),
                      missing=int((n&~finite).sum()), predicted_in_query=int((n&p).sum()))
        for cm in (5,10,20):
            hit=n&finite&(errors<=cm/100)
            coverage['abs_z_'+str(cm)+'cm']=int(hit.sum())
            coverage['in_query_and_abs_z_'+str(cm)+'cm']=int((hit&p).sum())
        rows.append(dict(origin=origin,coverage=coverage,
            conditional_error=dict(count=int((n&finite).sum()),abs_z_sum_m=float(errors[n&finite].sum(dtype=np.float64)))))
    return rows


def summarize(rows):
    result={}
    for model in ('sgbm','tao','candidate'):
        base=[r for r in rows if r['model']==model]
        groups={'all':base}
        groups.update({family:[r for r in base if r['family']==family] for family in CRITICAL})
        groups.update({part:[r for r in base if r['part']==part] for part in PARTS})
        result[model]={}
        for name, selected in groups.items():
            sums={k:defaultdict(int) for k in ('origin','coverage','conditional_error')}
            supported=defaultdict(int)
            for row in selected:
                for category in sums:
                    for key,value in row[category].items():
                        sums[category][key]+=value
                origin=row['origin']
                if origin['pixels']:
                    category='task_tp' if row['truth'] else 'task_fp'
                    supported[category]+=1
                    supported[category+'_without_native_in_query']+=int(origin['native_in_query']==0)
                    supported[category+'_all_far']+=int(origin['native_far']==origin['pixels'])
            cov=dict(sums['coverage']); cond=dict(sums['conditional_error'])
            n=cov.get('native_query_pixels',0)
            cond['mean_abs_z_m']=cond.get('abs_z_sum_m',0)/cond['count'] if cond.get('count',0) else None
            result[model][name]=dict(origin_pixels=dict(sums['origin']),coverage=cov,
                coverage_rates={k:v/n if n else None for k,v in cov.items() if k!='native_query_pixels'},
                supported_queries=dict(supported),conditional_error=cond)
    return result


def run(candidate,evaluation,inputs,output,reference=REFERENCE,tao=TAO):
    assert read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])['state']=='running'
    assert output.resolve().is_relative_to((REPO/'artifacts.local').resolve())
    assert not output.exists(), 'Fresh audit result required'
    started=time.perf_counter(); checks=0
    def check(v):
        nonlocal checks
        assert v
        checks+=1
    def equal(a,b):
        if isinstance(b,dict):
            check(set(a)==set(b))
            for k in b: equal(a[k],b[k])
        elif isinstance(b,float):
            check(bool(np.isclose(a,b,atol=1e-9,rtol=1e-9)))
        else: check(a==b)
    def verify(folder):
        receipt=read(folder/'receipt.json');check(receipt['status']=='PASS')
        for filename,digest in receipt['hashes'].items():
            p=folder/filename.replace('\\','/')
            check(p.resolve().is_relative_to(folder.resolve()))
            check(sha(p)==digest)
        return receipt
    receipts={name:verify(folder) for name,folder in [('candidate',candidate),('evaluation',evaluation),('reference',reference),('tao',tao)]}
    manifest=read(inputs); frames=manifest['frames'];rig=manifest['rig']
    check(len(frames)==576 and receipts['candidate']['frames']==576)
    check(receipts['candidate']['input_manifest_sha256']==sha(inputs))
    check((rig['width'],rig['height'],rig['hfov_deg'],rig['baseline_m'])==(640,360,70,.1))
    fb=rig['width']/(2*math.tan(math.radians(rig['hfov_deg']/2)))*rig['baseline_m']
    check(len({(r['panel'],r['id']) for r in frames})==576)
    for row in frames:
        folder=candidate/row['panel']; id=row['id']
        arrays={key:np.load(folder/key/(id+'.npy'),allow_pickle=False) for key in ('raw_disparity','raw_depth','depth')}
        d,raw,filtered=[arrays[k] for k in ('raw_disparity','raw_depth','depth')]
        for k,a in arrays.items():
            check(a.shape==(360,640) and a.dtype==np.float32)
            relative=f"{row['panel']}/{k}/{id}.npy"
            check(relative in receipts['candidate']['hashes'])
        valid=np.isfinite(d)&(d>0)
        target=np.full(d.shape,np.nan,np.float32)
        target[valid]=(fb/d[valid].astype(np.float64)).astype(np.float32)
        check(np.array_equal(np.isfinite(raw),valid))
        check(bool(np.allclose(raw,target,atol=2e-6,rtol=2e-6,equal_nan=True)))
        visible=np.arange(640)[None,:]-d>=0
        keep=valid&visible&(raw>=.5)&(raw<=4)
        check(np.array_equal(np.isfinite(filtered),keep))
        check(np.array_equal(filtered[keep],raw[keep]))
    queries=read(evaluation/'queries.json');summary=read(evaluation/'summary.json')
    check(len(queries)==576*3*2)
    by_key={(r['panel'],r['id'],r['model'],r['part']):r for r in queries}
    check(len(by_key)==len(queries))
    native_denoms={}
    for row in queries:
        o,c,err=row['origin'],row['coverage'],row['conditional_error']
        check(o['pixels']==o['native_in_query']+o['native_far']+o['native_near_outside']+o['other'])
        check(c['native_query_pixels']==c['predicted_valid']+c['missing'])
        check(c['predicted_in_query']==o['native_in_query'])
        check(err['count']==c['predicted_valid'] and err['abs_z_sum_m']>=0)
        check(0<=c['abs_z_5cm']<=c['abs_z_10cm']<=c['abs_z_20cm']<=c['predicted_valid'])
        check(0<=c['in_query_and_abs_z_5cm']<=c['in_query_and_abs_z_10cm']<=c['in_query_and_abs_z_20cm']<=c['predicted_in_query'])
        for cm in (5,10,20):check(c[f'in_query_and_abs_z_{cm}cm']<=c[f'abs_z_{cm}cm'])
        key=(row['panel'],row['id'],row['part'])
        if key in native_denoms:check(native_denoms[key]==c['native_query_pixels'])
        native_denoms[key]=c['native_query_pixels']
    equal(summary['geometry'],summarize(queries))
    pooled={stage:{model:dict(TP=0,FP=0,FN=0) for model in ('tof','sgbm','sgbm_union','tao','tao_union','candidate','candidate_union')} for stage in ('raw','final')}
    representative=[]
    for panel,(stem,old_depth) in PANELS.items():
        rows=[r for r in frames if r['panel']==panel]
        obs=read(reference/panel/'observations.json');truth=np.load(reference/panel/'truth.npy')
        check([r['id'] for r in rows]==[o['id'] for o in obs])
        check(truth.shape==(len(rows),2))
        panel_summary=read(evaluation/panel/'summary.json')
        seal=read(evaluation/panel/'prediction-seal.json')
        check(seal['predictions_sha256']==sha(evaluation/panel/'predictions.npz'))
        check(seal['candidate_receipt_sha256']==sha(candidate/'receipt.json'))
        check(seal['input_manifest_sha256']==sha(inputs) and seal['baseline_parity'])
        for p,h in seal['depth_hashes'].items():check(sha(Path(p))==h)
        with np.load(evaluation/panel/'predictions.npz') as saved:
            data={k:saved[k] for k in saved.files}
        for stage in ('raw','final'):
            for model in pooled[stage]:
                prediction=data[model+'_support']>0 if stage=='raw' else data[model].astype(bool)
                check(prediction.shape==truth.shape)
                counts=dict(TP=int((prediction&truth).sum()),FP=int((prediction&~truth).sum()),FN=int((~prediction&truth).sum()))
                for k,v in counts.items():
                    equal(panel_summary['scores'][stage][model][k],v)
                    pooled[stage][model][k]+=v
        panel_rows=[r for r in queries if r['panel']==panel]
        equal(panel_summary['geometry'],summarize(panel_rows))
        for i,row in enumerate(rows):
            for model in ('sgbm','tao','candidate'):
                for j,part in enumerate(PARTS):
                    q=by_key[(panel,row['id'],model,part)]
                    check(q['truth']==bool(truth[i,j]))
                    check(q['raw_union']==bool(data[model+'_union_support'][i,j]>0))
                    check(q['final_union']==bool(data[model+'_union'][i,j]))
                    check(q['tof_support']==int(data['tof_support'][i,j]))
        cap=WORK/stem/'capture-v1';capture_receipt=read(cap/'receipt.json')
        check(capture_receipt['status']=='PASS')
        # Frozen positional sample: first, middle, last in each panel, no score access selection.
        for i in (0,len(rows)//2,len(rows)-1):
            id=rows[i]['id'];p=cap/'frame'/id/'native-left-depth.npy'
            check(sha(p)==capture_receipt['hashes'][f'frame/{id}/native-left-depth.npy'])
            native=np.load(p)
            for model,path in [('sgbm',WORK/stem/old_depth/(id+'.npy')),('tao',tao/panel/'depth'/(id+'.npy')),('candidate',candidate/panel/'depth'/(id+'.npy'))]:
                for j,computed in enumerate(independent_geometry(np.load(path),native,obs[i]['pose'])):
                    row=by_key[(panel,id,model,PARTS[j])]
                    for k,v in computed.items():equal(row[k],v)
            representative.append(dict(panel=panel,id=id,index=i))
    for stage in pooled:
        for model in pooled[stage]:
            for key,value in pooled[stage][model].items():equal(summary['pooled_descriptive'][stage][model][key],value)
    output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='PASS',assertions=checks,frames_depth_replayed=576,query_rows_reconciled=len(queries),
        independent_native_frames=representative,models_per_native_frame=3,
        input_receipts={name:sha(folder/'receipt.json') for name,folder in [('candidate',candidate),('evaluation',evaluation),('reference',reference),('tao',tao)]},
        script_sha256=sha(Path(__file__)),input_manifest_sha256=sha(inputs),elapsed_s=time.perf_counter()-started,
        backend=dict(device='CPU',reason='TASK_NOT_GPU_SUITABLE',scope='IO_ETL and independent post-seal scalar/image geometry accounting'),
        limits='No model/checkpoint rerun, threshold selection, independent sensor regeneration or full-cohort native-geometry recomputation. '
               'All 576 disparity/depth masks and all query aggregates/TPFPFN replayed; native transform independently replayed only on fixed first/middle/last per panel. '
               'Event timing/hysteresis outputs hash-verified but not independently regenerated.',fits=0)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('FOUNDATION_GEOMETRY_AUDIT_PASS',checks,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('candidate','evaluation','inputs','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--tao',type=Path,default=TAO)
    a=parser.parse_args();run(a.candidate,a.evaluation,a.inputs,a.output,a.reference,a.tao)

"""Independent public-ray, output-array and post-seal VPP accounting audit."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from numba import njit
import foundation_geometry_audit_20260923 as prior

REPO,WORK=prior.REPO,prior.WORK
read,sha=prior.read,prior.sha
UNGUIDED=REPO/'artifacts.local/evidence/ba-foundation-geometry-20260923-frontend-v1'
ARMS=('tof','sgbm','sgbm_union','unguided','unguided_union','guided','guided_union')
SLICES=('no_direct_hint','nearby_hint','zero_whole_frame')


@njit
def seed_rng(seed):np.random.seed(seed)


def rays(ranges,valid,rig):
    hints=np.zeros((360,640),np.float32);seeds=[]
    f=320/math.tan(math.radians(rig['hfov_deg']/2))
    for i in range(64):
        radial=float(ranges[i])
        if not valid[i] or not math.isfinite(radial) or not .5<=radial<=4:continue
        y,x=divmod(i,8)
        right=math.tan(math.radians((x-3.5)*45/8))
        up=math.tan(math.radians((3.5-y)*45/8))
        z=radial/math.sqrt(1+right*right+up*up)
        xyz=[z,right*z,up*z]
        uf,vf=320+f*right,180-f*up;u,v=round(uf),round(vf)
        d=f*rig['baseline_m']/z
        if not (0<=u<640 and 0<=v<360 and 0<=u-d<=639):continue
        assert hints[v,u]==0
        hints[v,u]=d
        seeds.append(dict(zone=i,u=u,v=v,u_float=uf,v_float=vf,disparity=float(hints[v,u]),
            range_m=radial,axial_z_m=z,xyz_camera=xyz))
    return hints,seeds


def query_hints(seeds,pose):
    a,b,c=[math.radians(pose.get(k,0)) for k in ('yaw','pitch','roll')]
    cy,sy,cp,sp,cr,sr=math.cos(a),math.sin(a),math.cos(b),math.sin(b),math.cos(c),math.sin(c)
    rotation=np.array([[cp*cy,sr*sp*cy-cr*sy,-cr*sp*cy-sr*sy],
        [cp*sy,sr*sp*sy+cr*cy,cy*sr-cr*sp*sy],[sp,-sr*cp,cr*cp]])
    xyz=np.asarray([s['xyz_camera'] for s in seeds],float).reshape(-1,3)
    body=xyz@rotation.T+np.asarray(pose['camera_in_body_m'])
    counts={}
    for part,(lo,hi) in zip(prior.PARTS,prior.BOXES):
        distance=np.sqrt(np.maximum(np.maximum(np.asarray(lo)-body,body-np.asarray(hi)),0).__pow__(2).sum(1))
        counts[part]=dict(direct=int((distance==0).sum()),nearby=int(((distance>0)&(distance<=.25)).sum()))
    return counts


def selection(rows):
    groups={'all':rows}
    groups.update({f:[r for r in rows if r['family']==f] for f in prior.CRITICAL})
    groups.update({p:[r for r in rows if r['part']==p] for p in prior.PARTS})
    groups.update({s:[r for r in rows if r['hint_slices'][s]] for s in SLICES})
    return groups


def geometry_summary(rows):
    out={}
    for model in ('sgbm','unguided','guided'):
        out[model]={}
        for name,group in selection([r for r in rows if r['model']==model]).items():
            out[model][name]=prior.summarize([dict(r,model='candidate') for r in group])['candidate']['all']
    return out


def paired_summary(rows):
    out={}
    for name,group in selection(rows).items():
        sums=defaultdict(int)
        for r in group:
            for k,v in r['paired_geometry'].items():sums[k]+=v
        out[name]=dict(queries=len(group),**sums)
    return out


def point_counts(pred,truth):
    return dict(TP=int(np.count_nonzero(pred&truth)),FP=int(np.count_nonzero(pred&~truth)),
        FN=int(np.count_nonzero(~pred&truth)),TN=int(np.count_nonzero(~pred&~truth)))


def transition(a,b,g):
    patterns={'retained_tp':(True,True,True),'lost_tp':(True,False,True),'new_tp':(False,True,True),
        'removed_fp':(True,False,False),'added_fp':(False,True,False),'retained_fp':(True,True,False)}
    return {k:int(((a==x)&(b==y)&(g==z)).sum()) for k,(x,y,z) in patterns.items()}


def slice_scores(rows):
    out={}
    for name in SLICES:
        rr=[r for r in rows if r['hint_slices'][name]]
        truth=np.array([r['truth'] for r in rr],bool)
        out[name]=dict(queries=len(rr),stages={stage:{arm:point_counts(np.array([r['predictions'][stage][arm] for r in rr],bool),truth)
            for arm in ARMS} for stage in ('raw','final')})
    return out


def run(candidate,evaluation,inputs,tof,output,unguided=UNGUIDED,reference=prior.REFERENCE):
    assert read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])['state']=='running'
    assert output.resolve().is_relative_to((REPO/'artifacts.local').resolve()) and not output.exists()
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
        elif isinstance(b,float):check(bool(np.isclose(a,b,atol=2e-8,rtol=2e-8)))
        else:check(a==b)
    def verify(folder):
        r=read(folder/'receipt.json');check(r['status']=='PASS')
        for f,h in r['hashes'].items():
            p=folder/f.replace('\\','/');check(p.resolve().is_relative_to(folder.resolve()));check(sha(p)==h)
        return r
    receipts={k:verify(p) for k,p in [('candidate',candidate),('evaluation',evaluation),('unguided',unguided),('reference',reference)]}
    manifest,tm=read(inputs),read(tof);frames=manifest['frames'];rig=manifest['rig']
    check(len(frames)==len(tm['frames'])==576 and rig==tm['rig'])
    check(manifest['authority']=='RGB_AND_CALIBRATION_ONLY' and tm['authority']=='TOF_AND_CALIBRATION_ONLY')
    check(receipts['candidate']['input_manifest_sha256']==sha(inputs) and receipts['candidate']['tof_manifest_sha256']==sha(tof))
    hints=read(candidate/'hints.json');check(hints['input_manifest_sha256']==sha(inputs) and hints['tof_manifest_sha256']==sha(tof))
    meta={(r['panel'],r['id']):r for r in hints['frames']};tr={(r['panel'],r['id']):r for r in tm['frames']}
    keys={(r['panel'],r['id']) for r in frames};check(len(keys)==len(meta)==len(tr)==576 and keys==set(meta)==set(tr))
    config=read(candidate/'manifest.json')['configuration']
    vpp_path=Path(config['vpp_source'])/'vpp_standalone.py';check(sha(vpp_path)==config['vpp_sha256'])
    spec=importlib.util.spec_from_file_location('vpp_independent_replay',vpp_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    chosen=set()
    for panel in prior.PANELS:
        rr=[r for r in frames if r['panel']==panel]
        chosen.update((panel,rr[i]['id']) for i in (0,len(rr)//2,len(rr)-1))
    all_seeds={};zero=[];replayed=[];fb=320/math.tan(math.radians(rig['hfov_deg']/2))*rig['baseline_m']
    for row in frames:
        key=(row['panel'],row['id']);t=tr[key];m=meta[key]
        check(set(t)=={'panel','id','ranges','valid','ranges_sha256','valid_sha256'})
        for name in ('ranges','valid'):check(sha(t[name])==t[name+'_sha256'])
        disparity,seeds=rays(np.load(t['ranges']),np.load(t['valid']),rig)
        all_seeds[key]=seeds;equal(m['seeds'],seeds);check(m['total_frame_hints']==len(seeds))
        seed=int.from_bytes(hashlib.sha256((key[0]+'/'+key[1]).encode()).digest()[:4],'little')%(2**31)
        check(m['seed']==seed)
        if key in chosen or not seeds:
            images=[np.array(Image.open(row[v]).convert('RGB')) for v in ('left','right')]
            for view in ('left','right'):check(sha(row[view])==row[view+'_sha256'])
            if key in chosen:
                seed_rng(seed)
                projected=module.vpp(*images,disparity,wsize=3,blending=.4,method='rnd',uniform_color=False,
                    interpolate=True,left2right=True,use_distance_patch=False,use_bilateral_patch=False,g_occ=None)
                replayed.append(dict(panel=key[0],id=key[1]))
            else:projected=images
            for j,view in enumerate(('left','right')):
                check(hashlib.sha256(projected[j].tobytes()).hexdigest()==m[view+'_pixel_sha256'])
                check(int(np.any(projected[j]!=images[j],axis=2).sum())==m['changed_'+view+'_pixels'])
            if not seeds:
                zero.append(key)
                check(m['changed_left_pixels']==m['changed_right_pixels']==0)
                for folder in ('raw_disparity','raw_depth','depth'):
                    check(sha(candidate/key[0]/folder/(key[1]+'.npy'))==sha(unguided/key[0]/folder/(key[1]+'.npy')))
        d,raw,depth=[np.load(candidate/key[0]/folder/(key[1]+'.npy')) for folder in ('raw_disparity','raw_depth','depth')]
        check(d.shape==raw.shape==depth.shape==(360,640))
        positive=np.isfinite(d)&(d>0);target=np.full(d.shape,np.nan,np.float32)
        target[positive]=(fb/d[positive].astype(np.float64)).astype(np.float32)
        check(np.array_equal(np.isfinite(raw),positive));check(bool(np.allclose(raw,target,atol=2e-6,rtol=2e-6,equal_nan=True)))
        mask=positive&(np.arange(640)[None,:]-d>=0)&(raw>=.5)&(raw<=4)
        check(np.array_equal(np.isfinite(depth),mask));check(np.array_equal(depth[mask],raw[mask]))
    geometry=read(evaluation/'geometry-queries.json');paired=read(evaluation/'paired-queries.json');summary=read(evaluation/'summary.json')
    check(len(geometry)==3456 and len(paired)==1152)
    gm={(r['panel'],r['id'],r['model'],r['part']):r for r in geometry}
    pm={(r['panel'],r['id'],r['part']):r for r in paired};check(len(gm)==3456 and len(pm)==1152)
    for row in geometry:
        o,c,e=row['origin'],row['coverage'],row['conditional_error']
        check(o['pixels']==sum(o[k] for k in ('native_in_query','native_far','native_near_outside','other')))
        check(c['native_query_pixels']==c['missing']+c['predicted_valid'] and c['predicted_valid']==e['count'])
        check(c['predicted_in_query']==o['native_in_query'])
    for row in paired:
        p=row['paired_geometry'];k=(row['panel'],row['id']);part=row['part']
        a=gm[(*k,'unguided',part)]['coverage'];b=gm[(*k,'guided',part)]['coverage']
        check(p['native_query_pixels']==a['native_query_pixels']==b['native_query_pixels'])
        check(p['native_query_pixels']==sum(p[x] for x in ('both_correct','corrected','regressed','both_incorrect')))
        check(p['unguided_correct']==p['both_correct']+p['regressed']==a['in_query_and_abs_z_5cm'])
        check(p['guided_correct']==p['both_correct']+p['corrected']==b['in_query_and_abs_z_5cm'])
        check(p['corrected']-p['regressed']==p['guided_correct']-p['unguided_correct'])
        for stat in ('native_query_pixels','corrected','regressed'):
            check(p[stat]==sum(p[band+'_'+stat] for band in ('stamp_0to1px','near_2to16px','beyond16px','zero_whole_frame')))
    equal(summary['geometry'],geometry_summary(geometry));equal(summary['paired_geometry'],paired_summary(paired));equal(summary['hint_slices'],slice_scores(paired))
    totals={stage:{arm:defaultdict(int) for arm in ARMS} for stage in ('raw','final')}
    changes={stage:{arm:defaultdict(int) for arm in ('depth_only','_union')} for stage in ('raw','final')}
    for panel,(stem,old_depth) in prior.PANELS.items():
        obs=read(reference/panel/'observations.json');truth=np.load(reference/panel/'truth.npy').astype(bool)
        seal=read(evaluation/panel/'prediction-seal.json');check(seal['predictions_sha256']==sha(evaluation/panel/'predictions.npz'))
        check(seal['hints_sha256']==sha(candidate/'hints.json') and seal['guided_receipt_sha256']==sha(candidate/'receipt.json'))
        for p,h in seal['depth_hashes'].items():check(sha(p)==h)
        with np.load(evaluation/panel/'predictions.npz') as z:predictions={k:z[k] for k in z.files}
        report=read(evaluation/panel/'summary.json')
        for stage in ('raw','final'):
            pred={arm:(predictions[arm+'_support']>0 if stage=='raw' else predictions[arm].astype(bool)) for arm in ARMS}
            for arm,a in pred.items():
                counts=point_counts(a,truth)
                for k in ('TP','FP','FN'):equal(report['scores'][stage][arm][k],counts[k]);totals[stage][arm][k]+=counts[k]
            for suffix in ('','_union'):
                name=suffix or 'depth_only';t=transition(pred['unguided'+suffix],pred['guided'+suffix],truth)
                equal(report['transitions'][stage][name],t)
                for k,v in t.items():changes[stage][name][k]+=v
        for i,o in enumerate(obs):
            key=(panel,o['id']);qh=query_hints(all_seeds[key],o['pose'])
            for j,part in enumerate(prior.PARTS):
                row=pm[(*key,part)];equal(row['truth'],bool(truth[i,j]));equal(row['query_hints'],qh[part])
                check((qh[part]['direct']>0)==bool(predictions['tof_support'][i,j]>0))
                expected=dict(no_direct_hint=qh[part]['direct']==0,nearby_hint=qh[part]['nearby']>0,zero_whole_frame=not all_seeds[key])
                equal(row['hint_slices'],expected);check(row['total_frame_hints']==len(all_seeds[key]))
                for model in ('sgbm','unguided','guided'):
                    r=gm[(*key,model,part)];equal(r['query_hints'],qh[part]);equal(r['hint_slices'],expected);equal(r['truth'],bool(truth[i,j]))
                for stage in ('raw','final'):
                    for arm in ARMS:
                        actual=predictions[arm+'_support'][i,j]>0 if stage=='raw' else bool(predictions[arm][i,j])
                        equal(row['predictions'][stage][arm],bool(actual))
            if key in chosen:
                cap=WORK/stem/'capture-v1';receipt=read(cap/'receipt.json');npth=cap/'frame'/key[1]/'native-left-depth.npy'
                check(sha(npth)==receipt['hashes'][f'frame/{key[1]}/native-left-depth.npy']);native=np.load(npth)
                for model,path in [('sgbm',WORK/stem/old_depth/(key[1]+'.npy')),('unguided',unguided/panel/'depth'/(key[1]+'.npy')),('guided',candidate/panel/'depth'/(key[1]+'.npy'))]:
                    for j,r in enumerate(prior.independent_geometry(np.load(path),native,o['pose'])):
                        for category,value in r.items():equal(gm[(*key,model,prior.PARTS[j])][category],value)
        pg=[r for r in geometry if r['panel']==panel];pp=[r for r in paired if r['panel']==panel]
        equal(report['geometry'],geometry_summary(pg));equal(report['paired_geometry'],paired_summary(pp));equal(report['hint_slices'],slice_scores(pp))
    for stage in totals:
        for arm in ARMS:
            for k,v in totals[stage][arm].items():equal(summary['pooled_descriptive'][stage][arm][k],v)
    equal(summary['transitions'],changes)
    result=dict(status='PASS',assertions=checks,frames_public_hints_and_depth_replayed=576,seeded_vpp_rgb_replayed=replayed,
        zero_frame_identity_checks=len(zero),zero_frame_ids=[dict(panel=p,id=i) for p,i in zero],
        native_geometry_frames=6,geometry_rows=3456,paired_rows=1152,script_sha256=sha(__file__),
        prior_independent_helper_sha256=sha(Path(prior.__file__)),vpp_source_sha256=sha(vpp_path),elapsed_s=time.perf_counter()-start,
        backend=dict(device='CPU',reason='TASK_NOT_GPU_SUITABLE',scope='IO_ETL and independent post-seal geometry accounting; six fixed official VPP preprocessing replays'),
        limits='No model rerun, thresholds, full native geometry or event/hysteresis regeneration. All576 public ray metadata and raw depth/masks; all zero-frame original RGB and unguided disparity/depth exact hashes; '
               'six fixed first/middle/last frames per two panels replay official seeded VPP image bytes and independent native transforms. All query sums, partitions, hint slices, TPFPFN and transitions checked.',fits=0)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('VPP_GEOMETRY_AUDIT_PASS',checks,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('candidate','evaluation','inputs','tof','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--unguided',type=Path,default=UNGUIDED)
    parser.add_argument('--reference',type=Path,default=prior.REFERENCE)
    a=parser.parse_args();run(a.candidate,a.evaluation,a.inputs,a.tof,a.output,a.unguided,a.reference)

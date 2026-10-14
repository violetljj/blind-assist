"""Generic RGB/Lab cross-sections for autonomous line identity, label-blind.

Finite-strip hints are mechanisms, not pole labels or calibrated probabilities.
No target color, image coordinate, weak annotation or scene class is an input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from atom_camera import AtomCamera, ROOT
from thin_structure_candidates_dev_20261013 import line_parts

OUT = ROOT / 'artifacts.local/work/thin-appearance-dev-20261014/diagnostic'
IN = ROOT / 'artifacts.local/work/thin-continuity-dev-20261014/input'
OFFSETS = np.arange(-16, 17, dtype=float)
CONFIG = dict(along_sections=13, along_range=[.1,.9], normal_offsets=[-16,16],
              outer_scales=[6,10,14], core_halfwidth_px=1,
              minimum_sections=4, finite_cosine=.5, finite_balance=.25,
              minimum_contrast_delta_lab=1., minimum_along_consistent_fraction=.5,
              lab_units='OpenCV float Lab; L0..100, a/b approximately +/-128',
              purpose='appearance compatibility only; not target or distance confirmation')


def prepare_image(bgr):
    if isinstance(bgr, dict):return bgr
    assert bgr.ndim==3 and bgr.shape[2]==3
    floats=bgr.astype(np.float32)/255. if bgr.dtype==np.uint8 else bgr.astype(np.float32)
    return dict(lab=cv2.cvtColor(floats,cv2.COLOR_BGR2Lab),gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32))


def cosine(a,b):
    denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    return None if denom<1e-8 else float(np.clip(np.ravel(a)@np.ravel(b)/denom,-1,1))


def json_array(array):return np.round(np.asarray(array,float),5).tolist()


def strip_stats(values,scale):
    left=np.median(values[(OFFSETS<=-max(3,scale//2))&(OFFSETS>=-scale)],axis=0)
    right=np.median(values[(OFFSETS>=max(3,scale//2))&(OFFSETS<=scale)],axis=0)
    core=np.median(values[np.abs(OFFSETS)<=1],axis=0)
    dl,dr=core-left,core-right
    contrast=core-(left+right)/2
    a,b=float(np.linalg.norm(dl)),float(np.linalg.norm(dr))
    angle=cosine(dl,dr)
    balance=min(a,b)/max(a,b,1e-8)
    peak=float(np.linalg.norm(contrast));step=float(np.linalg.norm(left-right))
    finite=angle is not None and angle>=CONFIG['finite_cosine'] and balance>=CONFIG['finite_balance'] and peak>=1.
    if finite:state='FINITE_BILATERAL_STRIP'
    elif angle is not None and angle<=-.25 and step>=2.:state='SINGLE_STEP_LIKE'
    elif peak<1. and step<1.:state='FLAT_OR_WEAK'
    else:state='MIXED_OR_UNCERTAIN'
    return dict(scale_px=scale,core_lab=json_array(core),left_lab=json_array(left),right_lab=json_array(right),
                contrast_lab=json_array(contrast),bilateral_cosine=angle,bilateral_balance=balance,
                core_background_delta=peak,side_step_delta=step,strip_state=state)


def extract_appearance(image,c):
    image=prepare_image(image)
    lab=image['lab'];gray=image['gray']
    p,m,u,length=line_parts([c['x1'],c['y1'],c['x2'],c['y2']]);n=np.array([-u[1],u[0]])
    centers=p[0]+np.linspace(.1,.9,13)[:,None]*(p[1]-p[0])
    xy=centers[:,None,:]+OFFSETS[None,:,None]*n
    bounded=((xy[...,0]>=1)&(xy[...,0]<lab.shape[1]-2)&(xy[...,1]>=1)&(xy[...,1]<lab.shape[0]-2)).all(1)
    if bounded.sum()<4:return dict(status='UNKNOWN',reason='INSUFFICIENT_FULL_CROSS_SECTIONS',quality=dict(sections=int(bounded.sum()),stripe_reliable=False))
    x=xy[bounded,:,0].astype(np.float32);y=xy[bounded,:,1].astype(np.float32)
    color=cv2.remap(lab,x,y,cv2.INTER_LINEAR)
    luminance=cv2.remap(gray,x,y,cv2.INTER_LINEAR)
    profile=np.median(color,axis=0)
    scales=[strip_stats(profile,s) for s in CONFIG['outer_scales']]
    base=scales[1]
    along=[strip_stats(v,10) for v in color]
    fraction=sum(s['strip_state']=='FINITE_BILATERAL_STRIP' for s in along)/len(along)
    old_profile=np.median(luminance,axis=0)[6:27]  # original +/-10 px, gradient length20
    gradient=np.diff(old_profile);energy=float(np.linalg.norm(gradient))
    reliable=base['strip_state']=='FINITE_BILATERAL_STRIP' and bounded.sum()>=6 and fraction>=.5
    quality=dict(stripe_reliable=bool(reliable),strip_state=base['strip_state'],sections=int(bounded.sum()),
                 bilateral_cosine=base['bilateral_cosine'],bilateral_balance=base['bilateral_balance'],
                 side_step_delta=base['side_step_delta'],core_background_delta=base['core_background_delta'],
                 along_consistent_fraction=fraction,
                 status_scope='finite color-strip diagnostic; not pole purity or confidence probability')
    return dict(status='AVAILABLE',core_lab=base['core_lab'],contrast_lab=base['contrast_lab'],
                lab_profile=json_array(profile),along_core_lab=json_array(np.median(color[:,np.abs(OFFSETS)<=1],axis=1)),
                old_gray_gradient=json_array(gradient/max(energy,1e-8)),old_gray_energy=energy,
                old_gray_profile_valid=energy>=3.,quality=quality,scales=scales,
                normal_xy=json_array(n),length_px=length,
                lab_units=CONFIG['lab_units'],canonical_endpoint_direction='line_parts y-positive/x-positive')


def shifted_cosine(a,b):
    x,y=np.asarray(a,float),np.asarray(b,float)
    values=[]
    for shift in (-1,0,1):
        xx=x[1:] if shift>0 else x[:-1] if shift<0 else x
        yy=y[:-1] if shift>0 else y[1:] if shift<0 else y
        value=cosine(xx,yy)
        if value is not None:values.append(value)
    return max(values) if values else None


def compare_appearance(a,b):
    if a.get('status')!='AVAILABLE' or b.get('status')!='AVAILABLE':return dict(status='UNKNOWN',reason='MISSING_COLOR_CROSS_SECTION')
    x,y=np.asarray(a['lab_profile']),np.asarray(b['lab_profile'])
    shape=[];gradient_shape=[]
    def center(profile):
        background=np.linspace(profile[0],profile[-1],len(profile))
        weights=np.linalg.norm(profile-background,axis=1)*(np.abs(OFFSETS)<=8)
        return float(weights@OFFSETS/max(float(weights.sum()),1e-8))
    center_a,center_b=center(x),center(y)
    for scale_a in (4,6,8,10,12,14):
        for scale_b in (4,6,8,10,12,14):
            grid=np.linspace(-1,1,21)
            pa=np.stack([np.interp(grid*scale_a+center_a,OFFSETS,x[:,k]) for k in range(3)],1)
            pb=np.stack([np.interp(grid*scale_b+center_b,OFFSETS,y[:,k]) for k in range(3)],1)
            # Sparse derivative peaks inherit the raster kernel under scale.
            # Compare the full background-subtracted color strip as the shape;
            # retain the derivative result separately for mechanism diagnosis.
            ga=pa-np.linspace(pa[0],pa[-1],len(pa))
            gb=pb-np.linspace(pb[0],pb[-1],len(pb))
            sim=shifted_cosine(ga,gb)
            if sim is not None:shape.append((sim,scale_a,scale_b))
            derivative=shifted_cosine(np.diff(pa,axis=0),np.diff(pb,axis=0))
            if derivative is not None:gradient_shape.append(derivative)
    best=max(shape,default=(None,None,None),key=lambda q:-2 if q[0] is None else q[0])
    return dict(status='AVAILABLE',lab_core_delta=float(np.linalg.norm(np.asarray(a['core_lab'])-b['core_lab'])),
                lab_contrast_delta=float(np.linalg.norm(np.asarray(a['contrast_lab'])-b['contrast_lab'])),
                normalized_shape_similarity=best[0],best_shape_scale_a=best[1],best_shape_scale_b=best[2],
                gradient_shape_similarity=max(gradient_shape,default=None),
                automatic_profile_center_a=center_a,automatic_profile_center_b=center_b,
                shape_definition='multiscale background-subtracted Lab profile; automatic energy-center alignment',
                old_gray_gradient_similarity=shifted_cosine(a['old_gray_gradient'],b['old_gray_gradient']) if a['old_gray_profile_valid'] and b['old_gray_profile_valid'] else None,
                strip_state_a=a['quality']['strip_state'],strip_state_b=b['quality']['strip_state'],
                stripe_reliable_a=a['quality']['stripe_reliable'],stripe_reliable_b=b['quality']['stripe_reliable'])


def controls():
    def image(color,width=4,step=False):
        a=np.full((160,160,3),200,np.uint8)
        if step:a[:,:80]=color
        else:a[:,80-width//2:80+(width+1)//2]=color
        return a
    c=dict(x1=80,y1=30,x2=80,y2=130)
    d=extract_appearance(image((60,110,60)),c)
    reversed_d=extract_appearance(image((60,110,60)),dict(x1=80,y1=130,x2=80,y2=30))
    assert compare_appearance(d,reversed_d)['lab_core_delta']==0.
    assert d['quality']['strip_state']=='FINITE_BILATERAL_STRIP'
    step=extract_appearance(image((60,110,60),step=True),c)
    assert step['quality']['strip_state']!='FINITE_BILATERAL_STRIP'
    scaled=extract_appearance(image((60,110,60),8),c)
    assert compare_appearance(d,scaled)['normalized_shape_similarity']>.9
    flat=extract_appearance(np.full((160,160,3),100,np.uint8),c)
    assert flat['quality']['stripe_reliable'] is False
    other=extract_appearance(image((140,50,20)),c)
    color_cmp=compare_appearance(d,other)
    assert color_cmp['old_gray_gradient_similarity']>.99 and color_cmp['lab_core_delta']>10
    border=extract_appearance(image((60,110,60)),dict(x1=1,y1=20,x2=1,y2=120))
    assert border['status']=='UNKNOWN'
    return dict(endpoint_order_invariant=True,two_side_strip_vs_single_step=True,scale_shape_similarity=compare_appearance(d,scaled)['normalized_shape_similarity'],
                flat_not_reliable=True,same_normalized_gray_but_different_lab=color_cmp,border_unknown=True)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--controls-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();cv2.setNumThreads(2)
    ctrl=controls();(OUT/'controls.json').write_text(json.dumps(ctrl,indent=2,allow_nan=False),encoding='utf8')
    if args.controls_only:return
    source=IN/'frames.jsonl';rows=[json.loads(s) for s in source.read_text().splitlines()]
    camera=AtomCamera();frames=0;features=0;reliable=0;failures=[]
    try:
        with (OUT/'features.jsonl').open('w',encoding='utf8') as out:
            for r in rows:
                if time.perf_counter()-start>280:raise TimeoutError('diagnostic CPU reserve; do not imply full descriptor cache')
                up=camera.undistort_upright(cv2.imread(r['rgb_file']));prepared=prepare_image(up)
                cs=[]
                for c in r['candidates']:
                    a=extract_appearance(prepared,c);cs.append(dict(rank=c['rank'],feature=a));features+=1;reliable+=a.get('quality',{}).get('stripe_reliable',False)
                out.write(json.dumps(dict(key=r['key'],rgb_index=r['rgb_index'],rgb_file=r['rgb_file'],features=cs,labels_read=False),allow_nan=False)+'\n')
                frames+=1
        assert frames==654 and features==5191
    except Exception as e:
        failures.append(dict(type=type(e).__name__,message=str(e)));raise
    finally:
        receipt=dict(complete=frames==654 and not failures,frames=frames,candidate_features=features,reliable_strip_features=reliable,
                     wall_s=time.perf_counter()-start,gpu_s=0,labels_read=False,config=CONFIG,
                     source_sha256=sha(__file__),input_sha256=sha(source),controls=ctrl,failures=failures)
        (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()

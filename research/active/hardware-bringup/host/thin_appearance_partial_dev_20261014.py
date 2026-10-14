"""Current color core at a physical image boundary; never a bilateral strip.

Standalone helper preserves sealed full-cross-section features and their source.
No annotation, target color, image coordinate or scene class is consumed.
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
from thin_appearance_features_dev_20261014 import prepare_image

OUT=ROOT/'artifacts.local/work/thin-appearance-dev-20261014/diagnostic'
IN=ROOT/'artifacts.local/work/thin-continuity-dev-20261014/input'
OFFSETS=np.arange(-16,17,dtype=float)


def array_json(a):return np.round(np.asarray(a,float),5).tolist()


def extract_partial_core(image,c):
    # Geometric test happens before color extraction; missing flanks stay unknown.
    image=prepare_image(image);lab=image['lab'];h,w=lab.shape[:2]
    p,m,u,length=line_parts([c['x1'],c['y1'],c['x2'],c['y2']])
    if length<1:return dict(status='UNKNOWN',reason='DEGENERATE_SEGMENT')
    normal=np.array([-u[1],u[0]])
    centers=p[0]+np.linspace(.1,.9,13)[:,None]*(p[1]-p[0])
    xy=centers[:,None]+OFFSETS[None,:,None]*normal
    valid=(xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)
    physically_clipped=bool(not valid.all())
    if not physically_clipped:return dict(status='UNKNOWN',reason='NO_PHYSICAL_IMAGE_BOUNDARY_CROP')
    core_idx=np.flatnonzero(np.abs(OFFSETS)<=1)
    core_valid=valid[:,core_idx]
    center_valid=valid[:,16]
    sections=center_valid&(core_valid.sum(axis=1)>=1)
    unique_centers=np.unique(np.rint(centers[sections]).astype(int),axis=0)
    if sections.sum()<4 or len(unique_centers)<4:
        return dict(status='UNKNOWN',reason='INSUFFICIENT_CURRENT_CORE_SECTIONS',core_sections=int(sections.sum()),
                    unique_core_section_pixels=len(unique_centers),physical_image_boundary_crop=True)
    colors=cv2.remap(lab,xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
    # Match the full descriptor's median-over-along, then median-over-normal.
    color_by_offset=[]
    for offset in core_idx:
        keep=sections&valid[:,offset]
        if keep.any():color_by_offset.append(np.median(colors[keep,offset],axis=0))
    core=np.median(color_by_offset,axis=0)
    core_sample_mask=core_valid&sections[:,None]
    core_points=xy[:,core_idx][core_sample_mask]
    flanks={}
    for name,selected in [('negative_normal',(OFFSETS>=-10)&(OFFSETS<=-5)),
                          ('positive_normal',(OFFSETS>=5)&(OFFSETS<=10))]:
        indices=np.flatnonzero(selected);visible=valid[:,indices]&sections[:,None]
        visible_sections=(visible.sum(axis=1)>=1)
        section_n=int(visible_sections.sum());sample_n=int(visible.sum())
        offset_colors=[]
        for offset in indices:
            keep=sections&valid[:,offset]
            if keep.any():offset_colors.append(np.median(colors[keep,offset],axis=0))
        available=section_n>=4 and sample_n>=4 and bool(offset_colors)
        flanks[name]=dict(status='AVAILABLE_VISIBLE_FLANK' if available else 'UNKNOWN_NOT_OBSERVED',
            lab=array_json(np.median(offset_colors,axis=0)) if available else None,
            actual_samples=sample_n,actual_sections=section_n,expected_samples=int(sections.sum())*len(indices),
            missing_samples=int(sections.sum())*len(indices)-sample_n,fully_observed=bool(visible[sections].all()))
    return dict(status='PARTIAL_CORE_AVAILABLE',core_lab=array_json(core),core_sections=int(sections.sum()),
                unique_core_section_pixels=len(unique_centers),actual_core_samples=int(core_sample_mask.sum()),
                unique_actual_core_pixels=len(np.unique(np.rint(core_points).astype(int),axis=0)),
                missing_core_samples=int(sections.sum())*3-int(core_sample_mask.sum()),
                physical_image_boundary_crop=True,crop_status='NORMAL_OR_ALONG_CROSS_SECTION_CLIPPED',
                clipped_sides=dict(negative_normal=bool(not valid[:,OFFSETS<0].all()),
                                   positive_normal=bool(not valid[:,OFFSETS>0].all())),
                missing_flank_sides=[name for name,f in flanks.items() if not f['fully_observed']],
                actual_core_xy_bounds=dict(min=array_json(core_points.min(axis=0)),max=array_json(core_points.max(axis=0))),
                flanks=flanks,quality=dict(stripe_reliable=False,strip_state='UNKNOWN_BOUNDARY_CROPPED',
                                          fully_bilateral=False,shape_support='UNKNOWN_NOT_COMPLETE'),
                lab_units='OpenCV float Lab; L0..100, a/b approximately +/-128',
                evidence_scope='CURRENT_PARTIAL_COLOR_CORE_ONLY; no identity, depth or alarm confirmation')


def flank(d,name):
    if d.get('status')=='PARTIAL_CORE_AVAILABLE':
        f=d['flanks'][name]
        return f['lab'] if f['status']=='AVAILABLE_VISIBLE_FLANK' else None
    if d.get('status')=='AVAILABLE':
        s=next((s for s in d.get('scales',[]) if s['scale_px']==10),None)
        return None if s is None else s['left_lab' if name=='negative_normal' else 'right_lab']
    return None


def compare_partial_core(current,anchor):
    allowed=('PARTIAL_CORE_AVAILABLE','AVAILABLE')
    if current.get('status') not in allowed or anchor.get('status') not in allowed or not current.get('core_lab') or not anchor.get('core_lab'):
        return dict(status='UNKNOWN',reason='NO_CURRENT_OR_ANCHOR_CORE',fully_bilateral=False)
    a,b=np.asarray(current['core_lab']),np.asarray(anchor['core_lab'])
    pairs=[];common=[]
    for name in ('negative_normal','positive_normal'):
        x,y=flank(current,name),flank(anchor,name)
        if x is not None and y is not None:pairs.append((np.asarray(x),np.asarray(y)));common.append(name)
    contrast=None
    if pairs:
        bg_a=np.mean([q[0] for q in pairs],axis=0);bg_b=np.mean([q[1] for q in pairs],axis=0)
        contrast=float(np.linalg.norm((a-bg_a)-(b-bg_b)))
    return dict(status='AVAILABLE',lab_core_delta=float(np.linalg.norm(a-b)),lab_contrast_delta=contrast,
                visible_common_flank_sides=common,common_flank_count=len(common),
                contrast_scope='ONLY_COMMON_OBSERVED_BASE_FLANKS; missing sides were not filled',
                normalized_shape_similarity=None,stripe_reliable_a=False,stripe_reliable_b=False,
                fully_bilateral=False,support_type='PARTIAL_COLOR_ONLY',
                identity_confirmed=False,metric_distance_verified=False)


def controls():
    # A generic finite stripe is physically cut by a canvas edge; color is arbitrary.
    image=np.full((100,100,3),190,np.uint8);image[:,:3]=(80,70,95)
    c=dict(x1=1,y1=10,x2=1,y2=90)
    a=extract_partial_core(image,c)
    assert a['status']=='PARTIAL_CORE_AVAILABLE' and a['core_sections']==13
    assert a['quality']['stripe_reliable'] is False and not a['quality']['fully_bilateral']
    assert a['clipped_sides']['positive_normal'] and 'positive_normal' in a['missing_flank_sides']
    assert a['flanks']['positive_normal']['status']=='UNKNOWN_NOT_OBSERVED'
    reversed_a=extract_partial_core(image,dict(x1=1,y1=90,x2=1,y2=10))
    assert a['core_lab']==reversed_a['core_lab']
    uncropped=extract_partial_core(image,dict(x1=50,y1=20,x2=50,y2=80))
    assert uncropped['status']=='UNKNOWN' and uncropped['reason']=='NO_PHYSICAL_IMAGE_BOUNDARY_CROP'
    outside=extract_partial_core(image,dict(x1=-10,y1=20,x2=-10,y2=80))
    assert outside['status']=='UNKNOWN'
    short=extract_partial_core(image,dict(x1=1,y1=20,x2=1,y2=21))
    assert short['status']=='UNKNOWN'
    comparison=compare_partial_core(a,a)
    assert comparison['lab_core_delta']==0 and comparison['common_flank_count']==1
    assert comparison['fully_bilateral'] is False and comparison['normalized_shape_similarity'] is None
    assert compare_partial_core(a,outside)['status']=='UNKNOWN'
    return dict(edge_core_available=True,missing_flank_never_bilateral=True,endpoint_order_invariant=True,
                uncropped_does_not_override_full=True,unseen_core_unknown=True,duplicate_short_sections_unknown=True,
                common_visible_flank_only=True,comparison_unknown_preserved=True)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--controls-only',action='store_true');args=ap.parse_args()
    start=time.perf_counter();cv2.setNumThreads(2)
    sealed_paths=[OUT/'features.jsonl',OUT/'receipt.json',Path(__file__).parent/'thin_appearance_features_dev_20261014.py']
    sealed={str(p):sha(p) for p in sealed_paths};ctrl=controls()
    if args.controls_only:print(json.dumps(ctrl));return
    rows=[json.loads(s) for s in (IN/'frames.jsonl').read_text().splitlines()]
    full_rows=[json.loads(s) for s in (OUT/'features.jsonl').read_text().splitlines()]
    cam=AtomCamera();completed=0;available=0;unknown=0;failures=[];sessions={}
    output=OUT/'partial_core_features.jsonl'
    if output.exists():raise FileExistsError('Existing partial output retained; no implicit overwrite')
    try:
        with output.open('w',encoding='utf8') as f:
            for r,full in zip(rows,full_rows):
                if time.perf_counter()-start>80:raise TimeoutError('90 second partial-core reserve')
                assert (r['key'],r['rgb_index'])==(full['key'],full['rgb_index'])
                uncertain={q['rank'] for q in full['features'] if q['feature']['status']=='UNKNOWN'}
                entries=[]
                if uncertain:
                    image=prepare_image(cam.undistort_upright(cv2.imread(r['rgb_file'])))
                    for c in r['candidates']:
                        if c['rank'] not in uncertain:continue
                        p=extract_partial_core(image,c)
                        if p['status']=='PARTIAL_CORE_AVAILABLE':
                            entries.append(dict(rank=c['rank'],feature=p));available+=1
                        else:unknown+=1
                f.write(json.dumps(dict(key=r['key'],rgb_index=r['rgb_index'],partial_features=entries,labels_read=False),allow_nan=False)+'\n')
                completed+=1
                if output.stat().st_size>950000:raise RuntimeError('1MB partial payload reserve reached')
        assert completed==654 and all(sha(Path(p))==s for p,s in sealed.items())
    except Exception as error:
        failures.append(dict(type=type(error).__name__,message=str(error)));raise
    finally:
        receipt=dict(complete=completed==654 and not failures,frames=completed,partial_core_available=available,
                     previous_unknown_still_unknown=unknown,wall_s=time.perf_counter()-start,gpu_s=0,
                     labels_read=False,core_min_sections=4,quality='COLOR_ONLY; FULL_STRIP_AND_SHAPE_UNKNOWN',
                     source_sha256=sha(__file__),sealed_input_hashes=sealed,
                     sealed_inputs_unchanged=all(sha(Path(p))==s for p,s in sealed.items()),
                     controls=ctrl,failures=failures)
        (OUT/'partial_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf8')
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()

"""Evaluator-only visible-shaft identity on reused phone Development RGB.

Dense weak source labels are sealed before appearance output is read. Infinite
line alignment and association/composite spans are never finite object support.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from atom_camera import AtomCamera, ROOT
from thin_structure_evaluate_dev_20261013 import matches
from evaluate_thin_continuity_dev_20261014 import geometry, geometry_diagnostic, observations, sample_result, identity_matrix

BASE=ROOT/'artifacts.local/work/thin-appearance-dev-20261014'
OUT=BASE/'evaluation'
CONT=ROOT/'artifacts.local/work/thin-continuity-dev-20261014'
OLD=ROOT/'artifacts.local/work/thin-structure-dev-20261013/annotations.json'
KEYS=('073554','073635','073740','073821')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return [json.loads(v) for v in Path(path).read_text(encoding='utf-8-sig').splitlines() if v.strip()]


def save(name,obj):
    def convert(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n',encoding='utf8')


def prepare():
    start=time.perf_counter();cam=AtomCamera()
    manifest=json.loads((CONT/'input/manifest.json').read_text(encoding='utf8'))
    frames=[r for r in manifest['frames'] if r['key']=='073635' and 10<=r['rgb_index']<=67]
    assert len(frames)==58 and [r['rgb_index'] for r in frames]==list(range(10,68))
    samples=[];images=[]
    for r in frames:
        raw=cv2.imread(r['rgb_file']);im=cam.undistort_upright(raw)
        assert im.shape==(640,480,3)
        assert sha(r['rgb_file'])==r['rgb_sha256']
        samples.append(dict(key='073635',session=r['session'],rgb_index=r['rgb_index'],file=r['rgb_file'],
            raw_sha256=r['rgb_sha256'],device_capture_us=r['device_capture_us'],received_elapsed_ns=r['received_elapsed_ns'],
            boot_id=r['boot_id'],label_status='PENDING_BLIND_SOURCE_REVIEW',visibility='PENDING',
            segment=None,visible_rod_envelope_segment=None,uncertainty_px=None,note=None))
        grid=im.copy()
        for x in range(50,480,50):
            cv2.line(grid,(x,0),(x,639),(150,150,150),1)
            cv2.putText(grid,str(x),(x+1,18),0,.35,(240,240,240),1)
        for y in range(50,640,50):
            cv2.line(grid,(0,y),(479,y),(150,150,150),1)
            cv2.putText(grid,str(y),(2,y-2),0,.35,(240,240,240),1)
        im=cv2.addWeighted(im,.8,grid,.2,0)
        im=cv2.resize(im,(360,480))
        banner=np.zeros((25,360,3),np.uint8)
        cv2.putText(banner,f'H6 rgb={r["rgb_index"]} SOURCE RGB ONLY',(4,18),0,.46,(255,255,255),1)
        images.append(np.vstack((banner,im)))
    for page,lo in enumerate(range(0,58,12)):
        pp=images[lo:lo+12];pp += [np.zeros_like(images[0])]*(12-len(pp))
        cv2.imwrite(str(OUT/f'dense_blind_source_page{page}.jpg'),np.vstack([np.hstack(pp[i:i+4]) for i in range(0,12,4)]))
    save('dense_source_selection.json',dict(schema='thin-appearance-dense-source-selection-v1',
        source='All original H6 RGB10–67, no detector/appearance output read',
        old_annotations_sha256=sha(OLD),old_36_supplement_sha256=sha(CONT/'evaluation/blind_labels.json'),
        manifest_sha256=sha(CONT/'input/manifest.json'),calibration_sha256=sha(ROOT/'artifacts.local/work/atom-camera-calibration-20261012/calibration.json'),
        coordinate='AtomCamera undistorted upright x-right/y-down 480x640',samples=samples))
    save('PLAN.json',dict(schema='thin-appearance-evaluation-plan-v1',date='2026-10-14',lane='EXPLORE',
        budget=dict(cpu_command_wall_s=600,gpu_s=0,new_payload_mb=150),
        source='Existing consumed Development; H6 RGB10–67 all58, existing22 primary and36 supplemental unchanged',
        baseline='dcfb14b7 device-primary current member/pixel support, not family union',
        original_strict='Old matcher unchanged: undirected angle<=15deg, both weak segment endpoints <=10px from infinite candidate line, candidate projected overlap>=50% weak segment',
        line_axis='Same angle and normal rule, ignoring overlap; explicitly not finite visible-object support',
        finite_support='Against finite visible rod envelope: angle<=15deg; both current member endpoints <=10px normal distance to weak rod line; >=50% of current member projection lies inside visible envelope and overlap>=min(10px,0.5 member length). Each actual original/flow member separately, no compound-span filling.',
        uncertainty='Assistant weak centerline/envelope/visibility; not independent human/object truth; unknown/occluded pixels cannot satisfy full identity',
        complete='Same object ID has actual finite shaft support on all58 weak source checks, plus dense per-image identity/background audit; real capture gap keeps separate episode even if same object is reidentified',
        comparison='All-ID matrices/unique images; strict localization, axis alignment, finite support, gap episodes and background error separate. H5/H8 degradation controls use unchanged old labels.',
        causal='Record streaming current-frame versus any offline lookahead source explicitly; no alarm/TTC/safety or live freshness claim',
        labels_sealed_before_new_outputs=True,old_outputs_modified=False))
    save('prepare_receipt.json',dict(status='SOURCE_SELECTION_COMPLETE',raw_rgb_count=58,
        source_hashes_verified=58,new_algorithm_outputs_read=False,internal_wall_s=time.perf_counter()-start,gpu_s=0))
    print('prepared all58 source RGB; weak review pending')


def finite_support(envelope,c,tolerance=10,endpoint_inset=0):
    a,b=np.asarray(envelope,float).reshape(2,2);g=geometry(c)
    p,q=np.array([[g['x1'],g['y1']],[g['x2'],g['y2']]])
    length=float(np.linalg.norm(b-a));size=float(np.linalg.norm(q-p))
    if min(length,size)<1:return dict(status='NOT_EVALUABLE_DEGENERATE',supported=False)
    u=(b-a)/length;v=(q-p)/size;n=np.array([-u[1],u[0]])
    angle=float(np.rad2deg(np.arccos(np.clip(abs(u@v),-1,1))))
    normal=float(max(abs((p-a)@n),abs((q-a)@n)))
    lo,hi=sorted([(p-a)@u,(q-a)@u]);projection=float(hi-lo)
    overlap=float(max(0,min(length-endpoint_inset,hi)-max(endpoint_inset,lo)))
    fraction=overlap/max(projection,1e-9)
    enough=overlap>=min(10.,.5*size) and fraction>=.5
    supported=angle<=15 and normal<=tolerance and enough
    status=('FINITE_ROD_SUPPORT' if supported else 'OFF_OBJECT_NORMAL_OR_DIRECTION' if angle>15 or normal>tolerance
            else 'DISJOINT_VISIBLE_ROD' if overlap==0 else 'INSUFFICIENT_FINITE_OVERLAP')
    return dict(status=status,supported=bool(supported),angle_deg=angle,
        endpoint_normal_distance_px=normal,current_member_length_px=size,finite_overlap_px=overlap,
        current_member_overlap_fraction=fraction,endpoint_inset_px=endpoint_inset,
        projected_member_interval_px=[float(lo),float(hi)],visible_envelope_length_px=length,
        compound_span_used=False,physical_distance_verified=False)


def load_labels():
    seal=json.loads((OUT/'dense_label_receipt.json').read_text(encoding='utf8'))
    assert sha(OUT/'dense_labels.json')==seal['labels_sha256']
    assert seal['new_algorithm_outputs_read'] is False
    assert sha(OLD)==seal['old_annotations_sha256']
    assert sha(CONT/'evaluation/blind_labels.json')==seal['old_36_supplement_sha256']
    dense=json.loads((OUT/'dense_labels.json').read_text(encoding='utf8'))['samples']
    manifest=json.loads((CONT/'input/manifest.json').read_text(encoding='utf8'))
    meta={(r['key'],r['rgb_index']):r for r in manifest['frames']}
    primary=[s for s in json.loads(OLD.read_text(encoding='utf8'))['samples'] if s['session'] in KEYS]
    supp=json.loads((CONT/'evaluation/blind_labels.json').read_text(encoding='utf8'))['samples']
    return dense,meta,primary,supp


def source_episodes(dense,meta):
    episodes=[];current=[]
    for s in dense:
        m=meta[s['key'],s['rgb_index']]
        if current:
            prev=meta['073635',current[-1]]
            if m['boot_id']!=prev['boot_id'] or m['device_capture_us']-prev['device_capture_us']>500000:
                episodes.append(current);current=[]
        current.append(s['rgb_index'])
    if current:episodes.append(current)
    return [dict(source_episode=i,rgb_indices=rr,image_count=len(rr),
        span_s=(meta['073635',rr[-1]]['device_capture_us']-meta['073635',rr[0]]['device_capture_us'])/1e6,
        source_capture_gap_before_s=(meta['073635',rr[0]]['device_capture_us']-meta['073635',episodes[i-1][-1]]['device_capture_us'])/1e6 if i else None)
        for i,rr in enumerate(episodes)]


def matrix(samples,field):
    ids=sorted(set.union(*(set(s[field]) for s in samples))) if samples else []
    rows={tid:[tid in s[field] for s in samples] for tid in ids}
    complete=sorted(set.intersection(*(set(s[field]) for s in samples))) if samples else []
    return dict(image_count=len(samples),rgb_indices=[s['rgb_index'] for s in samples],
        same_id_at_every_check=complete,maximum_same_id_supported_images=max((sum(v) for v in rows.values()),default=0),
        images_with_any_id=sum(bool(s[field]) for s in samples),rows=rows)


def evaluate_arm(tag,path,dense,meta,primary,supp):
    rows=read(path);obs=observations(rows,meta)
    available={(r['key'],r['rgb_index']) for r in rows}
    assert len(available)==len(rows), 'duplicate image rows cannot be counted as extra support'
    assert all((s['key'],s['rgb_index']) in available for s in dense), 'H6 dense source window must be fully processed'
    availability={key:dict(processed=sum(r['key']==key for r in rows),source_total=sum(k==key for k,_ in meta)) for key in KEYS}
    lookup=defaultdict(list)
    for o in obs:lookup[o['key'],o['rgb_index']].append(o)
    modes={}
    for mode in ('candidate_only','candidate_plus_flow'):
        checks=[]
        for s in dense:
            current=[o for o in lookup[s['key'],s['rgb_index']] if mode=='candidate_plus_flow' or o['observation_kind']=='CURRENT_AUTONOMOUS_CANDIDATE']
            detail=[]
            for o in current:
                finite=finite_support(s['visible_rod_envelope_segment'],o['candidate'])
                conservative=finite_support(s['visible_rod_envelope_segment'],o['candidate'],endpoint_inset=s['envelope_endpoint_uncertainty_px'])
                axis=geometry_diagnostic(s['segment'],o['candidate'])
                strict=bool(matches(s['segment'],o['candidate']))
                detail.append(dict(object_id=str(o['track_id']),source_episode_id=o['source_episode_id'],
                    segment=o['segment'],kind=o['observation_kind'],source_observation_kind=o['source_observation_kind'],
                    source_candidate_id=o['source_candidate_id'],strict=strict,line_axis=axis,
                    finite=finite,finite_endpoint_conservative=conservative,
                    finite_boundary_uncertainty=finite['supported']!=conservative['supported']))
            checks.append(dict(key=s['key'],rgb_index=s['rgb_index'],visibility=s['visibility'],
                strict_ids=sorted({o['object_id'] for o in detail if o['strict']}),
                axis_ids=sorted({o['object_id'] for o in detail if o['line_axis']['aligned']}),
                finite_ids=sorted({o['object_id'] for o in detail if o['finite']['supported']}),
                conservative_finite_ids=sorted({o['object_id'] for o in detail if o['finite_endpoint_conservative']['supported']}),
                observations=detail,unique_image_clock_token=[meta[s['key'],s['rgb_index']]['boot_id'],meta[s['key'],s['rgb_index']]['device_capture_us']]))
        ep=[]
        for e in source_episodes(dense,meta):
            cc=[s for s in checks if s['rgb_index'] in e['rgb_indices']]
            mm=matrix(cc,'finite_ids');best=max(mm['rows'],key=lambda tid:sum(mm['rows'][tid]),default=None)
            times=[meta['073635',s['rgb_index']]['device_capture_us'] for s in cc if best and best in s['finite_ids']]
            ep.append(dict(e,finite_matrix=mm,longest_supported_same_id_span_s=(max(times)-min(times))/1e6 if len(times)>1 else 0,
                span_ceiling_is_source=True,missing_current_support_does_not_count_as_observation=True))
        primary_result=sample_result(primary,obs if mode=='candidate_plus_flow' else [o for o in obs if o['observation_kind']=='CURRENT_AUTONOMOUS_CANDIDATE'],'original_22_primary')
        supplemental_result=sample_result(supp,obs if mode=='candidate_plus_flow' else [o for o in obs if o['observation_kind']=='CURRENT_AUTONOMOUS_CANDIDATE'],'existing_36_supplemental')
        for s in primary_result+supplemental_result:
            s['processing_status']='PROCESSED' if (s['key'],s['rgb_index']) in available else 'UNPROCESSED_UNKNOWN'
            if s['processing_status']=='UNPROCESSED_UNKNOWN':
                s['matched']=None;s['strict3px_matched']=None
        controls={}
        for key in ('073554','073821'):
            ss=[s for s in primary_result if s['key']==key and s['matched'] is not None]
            oo=[s for s in supplemental_result if s['key']==key and s['matched'] is not None]
            controls[key]=dict(primary_visible=len(ss),primary_localized=sum(bool(s['matched']) for s in ss),primary_identity=identity_matrix(ss),
                supplemental_visible=len(oo),supplemental_localized=sum(bool(s['matched']) for s in oo),supplemental_identity=identity_matrix(oo),
                primary_labeled_visible=sum(s['session']==key and s.get('label_status')=='visible_segment' for s in primary),
                supplemental_labeled_visible=sum(s['session']==key and s.get('label_status')=='visible_segment' for s in supp),
                processing_complete=availability[key]['processed']==availability[key]['source_total'],
                partial_identity_is_not_full_window_identity=True)
        modes[mode]=dict(dense_checks=checks,matrices={f:matrix(checks,f) for f in ('strict_ids','axis_ids','finite_ids','conservative_finite_ids')},
            source_episodes=ep,old_primary_total=dict(visible=sum(s['matched'] is not None for s in primary_result),localized=sum(bool(s['matched']) for s in primary_result),
                labeled_visible=22,unprocessed_visible=sum(s['label_status']=='visible_segment' and s['processing_status']=='UNPROCESSED_UNKNOWN' for s in primary_result)),
            old_primary_samples=primary_result,old_supplemental_samples=supplemental_result,frontal_degradation_controls=controls,
            complete_status='PENDING_ALL58_DENSE_VISUAL_AUDIT' if matrix(checks,'finite_ids')['same_id_at_every_check'] else 'NO_OBJECT_ID_HAS_FINITE_SUPPORT_ON_ALL58')
    return dict(tag=tag,tracking_sha256=sha(path),tracking_path=str(path),modes=modes,processing_availability=availability,
        evaluation_scope='Actual individual original members/current RGB-supported flow subsegments only; no predicted/retained positions or composite-span support')


def known_background_case(arm,baseline_arm):
    """Find the fixed old bad native observation by source identity, evaluator-only."""
    old16=next(s for s in baseline_arm['modes']['candidate_plus_flow']['dense_checks'] if s['rgb_index']==16)
    bad=next(o for o in old16['observations'] if o['object_id']=='2' and o['kind']=='CURRENT_AUTONOMOUS_CANDIDATE')
    source=bad['source_candidate_id']
    cc=arm['modes']['candidate_plus_flow']['dense_checks']
    now16=next(s for s in cc if s['rgb_index']==16)
    matches_source=[o for o in now16['observations'] if o['kind']=='CURRENT_AUTONOMOUS_CANDIDATE' and o['source_candidate_id']==source]
    result=[]
    for o in matches_source:
        tid=o['object_id'];finite_frames=[s['rgb_index'] for s in cc if tid in s['finite_ids']]
        actual=[s['rgb_index'] for s in cc if any(v['object_id']==tid for v in s['observations'])]
        result.append(dict(object_id=tid,background_segment=o['segment'],finite_status=o['finite']['status'],
            actual_id_frames=actual,finite_rod_support_frames=finite_frames,
            same_id_contaminates_finite_rod=bool(finite_frames),
            interpretation='BACKGROUND_MEMBER_SHARES_ID_WITH_FINITE_ROD' if finite_frames else 'THIS_BACKGROUND_ID_HAS_NO_FINITE_ROD_SUPPORT_IN_DENSE_WINDOW',
            before_after_rod_identity_still_requires_all58=True))
    return dict(rgb_index=16,old_bad_source_candidate_id=source,original_candidate_preserved=bool(matches_source),
        background_members=result,current_frame_any_finite_ids=now16['finite_ids'],
        scope='Source/native identity regression only. Isolating this background member does not establish complete rod identity.')


def partial_h8_controls(meta,primary,supp):
    path=BASE/'tracking/h8_anchor/tracking_anchor_strip.jsonl';receipt=BASE/'tracking/h8_anchor/receipt.json'
    if not path.exists():return None
    rows=read(path);available={(r['key'],r['rgb_index']) for r in rows}
    assert len(available)==len(rows) and all(r['key']=='073821' for r in rows)
    obs=observations(rows,meta);result={}
    for mode in ('candidate_only','candidate_plus_flow'):
        current=obs if mode=='candidate_plus_flow' else [o for o in obs if o['observation_kind']=='CURRENT_AUTONOMOUS_CANDIDATE']
        result[mode]={}
        for name,labels in [('primary',primary),('supplemental',supp)]:
            labels=[s for s in labels if s['session']=='073821']
            rr=sample_result(labels,current,'H8_PARTIAL_SOURCE_OUTPUT_ONLY')
            for s in rr:
                s['processing_status']='PROCESSED' if (s['key'],s['rgb_index']) in available else 'UNPROCESSED_UNKNOWN'
                if s['processing_status']=='UNPROCESSED_UNKNOWN':s['matched']=None;s['strict3px_matched']=None
            processed=[s for s in rr if s['matched'] is not None]
            result[mode][name]=dict(labeled_visible=sum(s['label_status']=='visible_segment' for s in rr),
                processed_visible=len(processed),localized=sum(bool(s['matched']) for s in processed),samples=rr,
                partial_identity=identity_matrix(processed),complete_window_identity='NOT_EVALUABLE_UNPROCESSED_SOURCE_IMAGES')
    return dict(tracking_path=str(path),tracking_sha256=sha(path),receipt_sha256=sha(receipt),
        processed_frames=len(rows),source_total_frames=378,complete=False,modes=result,
        scope='Separate 154-image partial H8 continuation; unavailable rows are UNKNOWN, never false negatives or full-window success')


def baseline():
    start=time.perf_counter();dense,meta,primary,supp=load_labels()
    path=CONT/'tracking/tracking_device_primary.jsonl'
    result=evaluate_arm('dcfb14b7_device_primary',path,dense,meta,primary,supp)
    result['known_rgb16_background_case']=known_background_case(result,result)
    result['labels_sha256']=sha(OUT/'dense_labels.json');result['internal_wall_s']=time.perf_counter()-start
    save('baseline.json',result)
    summary={mode:{field:{k:v for k,v in mm.items() if k!='rows' and k!='rgb_indices'} for field,mm in arm['matrices'].items()} for mode,arm in result['modes'].items()}
    save('baseline_summary.json',summary);print(json.dumps(summary,indent=2))


def evaluate():
    dense,meta,primary,supp=load_labels()
    baseline_result=json.loads((OUT/'baseline.json').read_text(encoding='utf8'))
    result=dict(schema='thin-appearance-paired-evaluation-v1',date='2026-10-14',
        evaluator_source_sha256=sha(Path(__file__)),
        dense_labels_sha256=sha(OUT/'dense_labels.json'),original_labels_sha256=sha(OLD),
        existing36_sha256=sha(CONT/'evaluation/blind_labels.json'),
        plan_sha256=sha(OUT/'PLAN.json'),source_episodes=source_episodes(dense,meta),arms={'dcfb14b7_device_primary':baseline_result},
        source_interval_image_count=58,source_visible_image_count=58,source_not_evaluable=0,
        limits=['Reused consumed Development; source and envelope are assistant weak review, not independent truth',
                'All58 criterion retains UNKNOWN and image-support absence; no denominator shrink',
                'Source longest episode is 4.610374s, an availability ceiling rather than a gain endpoint',
                'Same object ID across capture gaps can be reidentified; no gap observation or continuous episode invented'])
    specs=[(tag,BASE/f'tracking/fast_partial/tracking_{tag}.jsonl',BASE/'tracking/fast_partial/receipt.json') for tag in ('local_lab','local_strip','anchor_strip')]
    specs += [('anchor_finite',BASE/'tracking/tracking_anchor_finite.jsonl',BASE/'tracking/receipt_finite.json'),
              ('local_lab_partial',BASE/'tracking/tracking_local_lab_partial.jsonl',BASE/'tracking/receipt_partial.json')]
    specs += [('offline_family',BASE/'tracking/tracking_offline_family.jsonl',BASE/'tracking/receipt_offline_family.json')]
    for tag,path,receipt in specs:
        rr=json.loads(receipt.read_text(encoding='utf8'))
        assert rr['labels_read'] is False
        lookahead=rr.get('lookahead_frames',rr.get('config',{}).get('lookahead_frames'))
        if tag=='offline_family':assert rr['complete'] is True and rr['frames']==70 and lookahead==69
        else:assert rr['sessions']['073635'][tag]['frames']==70 and lookahead==0
        arm=evaluate_arm(tag,path,dense,meta,primary,supp)
        arm['receipt_sha256']=sha(receipt);arm['receipt_path']=str(receipt);arm['source_code_sha256']=rr.get('source_code_sha256',rr.get('source_sha256'))
        arm['receipt_complete_for_declared_processing_scope']=rr['complete']
        arm['known_rgb16_background_case']=known_background_case(arm,baseline_result)
        arm['lookahead_frames']=lookahead
        arm['temporal_availability']='OFFLINE_FULL70_RGB_OBJECT_GRAPH_LOOKAHEAD69' if lookahead else 'CAUSAL_CURRENT_RGB_NO_LOOKAHEAD'
        result['arms'][tag]=arm
    result['separate_partial_h8_controls']=partial_h8_controls(meta,primary,supp)
    save('evaluation.json',result)
    summary={tag:{mode:{field:{k:v for k,v in mm.items() if k not in ('rows','rgb_indices')} for field,mm in a['matrices'].items()} for mode,a in arm['modes'].items()} for tag,arm in result['arms'].items()}
    save('evaluation_summary.json',summary)
    print(json.dumps(summary,indent=2))


def validate():
    dense,meta,primary,supp=load_labels()
    assert len(dense)==58 and [s['rgb_index'] for s in dense]==list(range(10,68))
    assert len({tuple([meta[s['key'],s['rgb_index']]['boot_id'],meta[s['key'],s['rgb_index']]['device_capture_us']]) for s in dense})==58
    for s in dense:
        m=meta[s['key'],s['rgb_index']]
        assert s['raw_sha256']==m['rgb_sha256'] and s['device_capture_us']==m['device_capture_us']
        assert np.isfinite(s['segment']).all() and np.isfinite(s['visible_rod_envelope_segment']).all()
        assert 0<=min(s['segment'][::2])<=max(s['segment'][::2])<480
    assert len(source_episodes(dense,meta))==5
    envelope=[10,100,10,300]
    above=dict(segment=[10,20,10,90]);inside=dict(segment=[10,160,10,200])
    assert geometry_diagnostic(envelope,above)['aligned']
    assert not finite_support(envelope,above)['supported']
    assert finite_support(envelope,inside)['supported']
    assert finite_support(envelope,dict(segment=[10,200,10,160]))['supported']
    assert not finite_support(envelope,dict(segment=[0,150,50,150]))['supported']
    assert not finite_support(envelope,dict(segment=[10,0,10,150]))['supported']
    save('verification.json',dict(status='PASS',checks=['58 raw hash/device-time/unique capture tokens, finite geometry, existing22/36 unchanged',
        'above-cap infinite-line alias rejected by finite support, endpoint order invariant, crossing/mostly-outside members rejected',
        '5 source capture-gap episodes; span ceiling source-derived'],gpu_s=0))
    print('PASS')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','baseline','evaluate','validate']);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'baseline':baseline,'validate':validate,'evaluate':evaluate}[args.stage]()


if __name__=='__main__':main()

"""Consumed Development continuity diagnostics; weak labels are evaluator-only.

prepare chooses equal-device-time RGBs before reading new algorithm output.
evaluate reports actual observations, never interpolated/retained predictions.
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

BASE = ROOT / 'artifacts.local/work/thin-continuity-dev-20261014'
OUT = BASE / 'evaluation'
OLD = ROOT / 'artifacts.local/work/thin-structure-dev-20261013/annotations.json'
OLD_TRACK = ROOT / 'artifacts.local/work/thin-local-geometry-dev-20261013/camera_chain_frames.jsonl'
RAW = ROOT / 'artifacts.local/work/rgb-real-feasibility-20261012/raw'
KEYS = ('073554', '073635', '073740', '073821')


def read(path):
    return [json.loads(v) for v in Path(path).read_text(encoding='utf-8-sig').splitlines() if v.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(f'Cannot serialize {type(value).__name__}')
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,default=convert) + '\n', encoding='utf8')


def prepare():
    """No inference/tracking output read by this mode."""
    cam = AtomCamera()
    primary = json.loads(OLD.read_text(encoding='utf8'))
    selected = []
    sessions = {}
    for key in ('073554', '073635', '073821'):
        visible = [s for s in primary['samples'] if s['session'] == key and s['segment']]
        rawdir = RAW / visible[0]['session_name']
        index = read(rawdir / 'rgb.jsonl')
        ts = np.array([r['device_capture_timestamp_us'] for r in index], dtype=np.int64)
        first, last = min(s['rgb_index'] for s in visible), max(s['rgb_index'] for s in visible)
        targets = np.linspace(int(ts[first]), int(ts[last]), 12)
        ids = [min(range(first, last+1), key=lambda i: abs(int(ts[i])-target)) for target in targets]
        if len(set(ids)) != 12:
            raise ValueError('equal-time sample maps to repeated images')
        sessions[key] = dict(primary_first_rgb_index=first, primary_last_rgb_index=last,
                             primary_first_us=int(ts[first]), primary_last_us=int(ts[last]),
                             selected_rgb_indices=ids, raw_index_sha256=sha(rawdir/'rgb.jsonl'))
        images = []
        for order, i in enumerate(ids):
            r = index[i]
            path = rawdir / r['file']
            im = cam.undistort_upright(cv2.imread(str(path)))
            row = dict(session=key, session_name=visible[0]['session_name'], rgb_index=i,
                       file=str(path), raw_sha256=sha(path), device_capture_us=int(ts[i]),
                       receipt_t_ns=int(r['received_elapsed_ns']), order=order,
                       label_status='PENDING_BLIND_RGB_REVIEW', segment=None)
            selected.append(row)
            bar = np.zeros((30, 480, 3), np.uint8)
            cv2.putText(bar, f'{key} order={order} rgb_index={i}', (5, 21), 0, .55, (255,255,255), 1)
            images.append(np.vstack((bar, im)))
        for half in (0, 1):
            subset = images[half*6:(half+1)*6]
            cv2.imwrite(str(OUT/f'blind_rgb_{key}_{half}.jpg'),
                        np.vstack((np.hstack(subset[:3]),np.hstack(subset[3:]))))
    save('blind_selection.json', dict(schema='thin-continuity-blind-rgb-selection-v1',
         selection='12 equal device capture time positions within existing first/last visible primary anchors; nearest original RGB; no new algorithm output read',
         original_annotations_sha256=sha(OLD), sessions=sessions, samples=selected))
    save('PLAN.json', dict(schema='thin-continuity-evaluation-plan-v1', date='2026-10-14',
        lane='EXPLORE', budget=dict(cpu_command_wall_s=600,gpu_command_wall_s=0,new_payload_mb=100),
        primary='Existing 24 anchors, 22 visible assistant-weak segments; unchanged per-source hash',
        original_annotations_sha256=sha(OLD), original_tracking_source_sha256=sha(OLD_TRACK),
        supplemental='H5/H6/H8 12 equal-time RGBs each, assistant weak review before reading new output; not new independent truth',
        matching='Existing individual segment rule: undirected <=15deg, both weak endpoints <=10px from candidate infinite line, overlap >=50%; strict3px supplement',
        comparison='old sampled camera-chain; same fullRGB new candidates with legacy tracker vs continuous tracker (paired ablation)',
        continuity='Count only current autonomous candidate observations; longest same-ID chain must hit same weak target at multiple times; report all inspected off-target observations. Missing/retained/predicted positions never count as recovered.',
        coverage='Between same fixed first/last visible primary target anchors, not full unannotated session. Span/maxgap and actual observed unique RGB count separate from weak target support.',
        motion='Same common image y reference for vertical line horizontal bearing; require reference inside both visible segments. Visible crop span only, not physical expansion/TTC. Matched common samples and same duration; no tracking failures as class evidence.',
        limitations=['Body-stationary turn has no independent truth; current scope only frontal vs side descriptive diagnostic', 'Assistant weak labels and sparse audits cannot prove purity of every unlabelled track frame', 'No alarm integration or safety/real-time claim']))
    print(json.dumps(sessions, indent=2))


def geometry(c):
    if all(k in c for k in ('x1','y1','x2','y2')):
        return {k: float(c[k]) for k in ('x1','y1','x2','y2')}
    if 'segment' in c:
        return dict(zip(('x1','y1','x2','y2'), map(float,c['segment'])))
    raise ValueError('missing current observation segment')


def reference(c, K):
    g=geometry(c); a=np.array([g['x1'],g['y1']]); b=np.array([g['x2'],g['y2']]); d=b-a
    if abs(d[1]) < 1e-6 or abs(d[1]) < abs(d[0]):
        return dict(status='NOT_EVALUABLE_NONVERTICAL', length_px=float(np.linalg.norm(d)))
    fraction=(320-a[1])/d[1]; x=float(a[0]+fraction*d[0])
    return dict(status='OBSERVED_IMAGE_SUPPORT' if 0<=fraction<=1 else 'MODEL_EXTRAPOLATED_SUPPORTING_LINE',
        reference_y_px=320., x_at_reference_px=x,
        bearing_deg=float(np.rad2deg(np.arctan((x-K[0,2])/K[0,0]))),
        length_px=float(np.linalg.norm(d)), reference_is_physical_point=False)


def geometry_diagnostic(segment, c):
    a,b=np.asarray(segment,float).reshape(2,2);g=geometry(c)
    p,q=np.array([[g['x1'],g['y1']],[g['x2'],g['y2']]])
    length=float(np.linalg.norm(b-a));size=float(np.linalg.norm(q-p))
    if min(length,size)<1:return dict(status='DEGENERATE',aligned=False)
    u=(b-a)/length;v=(q-p)/size;n=np.array([-v[1],v[0]])
    angle=float(np.rad2deg(np.arccos(np.clip(abs(u@v),-1,1))))
    offset=float(max(abs((a-p)@n),abs((b-p)@n)))
    lo,hi=sorted([(p-a)@u,(q-a)@u]);overlap=float(max(0,min(length,hi)-max(0,lo))/length)
    aligned=angle<=15 and offset<=10
    return dict(status='MATCH' if aligned and overlap>=.5 else 'SHORT_OR_DISJOINT_FRAGMENT' if aligned else 'ANGLE_OR_NORMAL_OFFSET_MISMATCH',
        aligned=bool(aligned),angle_deg=angle,maximum_endpoint_normal_distance_px=offset,weak_segment_overlap_fraction=overlap)


def observations(rows, metadata, old=False):
    obs=[]
    for r in rows:
        key,ri=r['key'],r['rgb_index']; m=metadata[(key,ri)]
        groups=[('CURRENT_AUTONOMOUS_CANDIDATE',r['candidates'])]
        if not old:
            flows=[]
            for flow in r.get('flow_observations',[]):
                if 'supporting_subsegments' in flow:
                    # Family association span is not actual image support.
                    for member in flow['supporting_subsegments']:
                        child=dict(flow)
                        child.update(member if isinstance(member,dict) else dict(segment=member))
                        child.update(geometry(member if isinstance(member,dict) else dict(segment=member)))
                        child['track_id']=flow.get('object_id',flow['track_id'])
                        child['association_composite_segment_not_evaluated']=True
                        flows.append(child)
                else:flows.append(flow)
            groups.append(('FLOW_SUPPORTED',flows))
        for kind, cc in groups:
            for c in cc:
                emitted=c.get('observation_kind',kind)
                if emitted not in ('CURRENT_AUTONOMOUS_CANDIDATE','CURRENT_AUTONOMOUS_FAMILY_MEMBER','FLOW_SUPPORTED'):
                    raise ValueError('prediction/nonobservation emitted in current observation list')
                g=geometry(c)
                identity=c.get('family_track_id',c.get('object_id',c['track_id']))
                obs.append(dict(key=key,rgb_index=ri,t_us=int(m['device_capture_us']),
                    track_id=identity,rank=c.get('rank',9999),observation_kind=kind,
                    source_observation_kind=emitted,
                    segment=list(g.values()),candidate=g,original=c,boot_id=m['boot_id'],
                    independent_hits=c.get('independent_hits',c.get('hits')),
                    source_candidate_id=c.get('source_candidate_id'),member_observation_id=c.get('member_observation_id'),
                    candidate_family_id=c.get('candidate_family_id'),ready_i=r.get('i') if old else None,row=r))
                obs[-1]['source_episode_id']=c.get('episode_id')
    return obs


def sample_result(samples, obs, scope, exact_old_ready=False):
    lookup=defaultdict(list)
    for o in obs: lookup[o['key'],o['rgb_index']].append(o)
    results=[]
    for s in samples:
        allobs=lookup[s['session'],s['rgb_index']]
        if exact_old_ready:allobs=[o for o in allobs if o['ready_i']==s['i']]
        visible=s.get('label_status')=='visible_segment' and s.get('segment') is not None
        hits=[o for o in allobs if visible and matches(s['segment'],o['candidate'])]
        strict=[o for o in hits if matches(s['segment'],o['candidate'],3)]
        aligned=[o for o in allobs if visible and geometry_diagnostic(s['segment'],o['candidate'])['aligned']]
        results.append(dict(key=s['session'],rgb_index=s['rgb_index'],label_status=s['label_status'],
            primary_i=s.get('i'),scope=scope,segment=s['segment'],
            matched=len(hits)>0 if visible else None,strict3px_matched=len(strict)>0 if visible else None,
            matched_ids=sorted(set(str(o['track_id']) for o in hits)),
            line_aligned_ids=sorted(set(str(o['track_id']) for o in aligned)),
            matched_observations=[{k:o[k] for k in ('track_id','rank','observation_kind','source_candidate_id','segment','t_us')} for o in hits],
            observations=len(allobs)))
    return results


def chains(obs, label_results, metadata, intervals):
    grouped=defaultdict(list)
    for o in obs: grouped[o['key'],str(o['track_id'])].append(o)
    labels={(s['key'],s['rgb_index']):s for s in label_results if s['matched'] is not None}
    out=[]
    for (key,tid), rr in grouped.items():
        rr.sort(key=lambda o:(o['t_us'],o['rgb_index'],o['rank']))
        # Candidate/flow repeats of one RGB are one independent observation.
        unique={}
        for o in rr:
            token=(o['boot_id'],o['t_us'])
            if token not in unique:unique[token]=dict(o,supporting_members=[])
            unique[token]['supporting_members'].append(o)
        rr=list(unique.values()); episodes=[];current=[]
        for o in rr:
            if current and (o['boot_id']!=current[-1]['boot_id'] or o['t_us']-current[-1]['t_us']>500000 or
                            o['source_episode_id']!=current[-1]['source_episode_id']):
                episodes.append(current);current=[]
            current.append(o)
        if current:episodes.append(current)
        for episode, ee in enumerate(episodes):
            a,b=intervals[key]; within=[o for o in ee if a<=o['t_us']<=b]
            if not within: continue
            weak=[]
            for o in within:
                s=labels.get((key,o['rgb_index']))
                if s:
                    member_checks=[dict(segment=member['segment'],kind=member['observation_kind'],source_candidate_id=member['source_candidate_id'],geometry_diagnostic=geometry_diagnostic(s['segment'],member['candidate'])) for member in o['supporting_members']]
                    priority={'MATCH':0,'SHORT_OR_DISJOINT_FRAGMENT':1,'ANGLE_OR_NORMAL_OFFSET_MISMATCH':2,'DEGENERATE':3}
                    diagnostic=min(member_checks,key=lambda m:priority[m['geometry_diagnostic']['status']])['geometry_diagnostic']
                    weak.append(dict(rgb_index=o['rgb_index'],matched=any(matches(s['segment'],member['candidate']) for member in o['supporting_members']),
                                     segment=o['segment'],weak_segment=s['segment'],kind=o['observation_kind'],geometry_diagnostic=diagnostic,
                                     member_checks=member_checks,union_span_used_for_match=False))
            good=[v for v in weak if v['matched']];bad=[v for v in weak if not v['matched']]
            times=[o['t_us'] for o in within]
            out.append(dict(key=key,track_id=tid,episode=episode,observed_images=len(within),
                autonomous_candidate_images=sum(any(m['observation_kind']=='CURRENT_AUTONOMOUS_CANDIDATE' for m in o['supporting_members']) for o in within),
                flow_supported_images=sum(any(m['observation_kind']=='FLOW_SUPPORTED' for m in o['supporting_members']) for o in within),
                first_rgb_index=within[0]['rgb_index'],last_rgb_index=within[-1]['rgb_index'],
                duration_s=(times[-1]-times[0])/1e6,coverage_of_fixed_interval=(times[-1]-times[0])/(b-a),
                maximum_observation_gap_s=max(np.diff(times),default=0)/1e6,
                weak_target_matched_images=len(good),weak_target_mismatch_images=len(bad),
                weak_target_line_aligned_images=sum(v['geometry_diagnostic']['aligned'] for v in weak),
                short_fragment_mismatch_images=sum(v['geometry_diagnostic']['status']=='SHORT_OR_DISJOINT_FRAGMENT' for v in weak),
                angle_or_normal_offset_mismatch_images=sum(v['geometry_diagnostic']['status']=='ANGLE_OR_NORMAL_OFFSET_MISMATCH' for v in weak),
                weak_checks=weak,object_identity='WEAK_TARGET_SUPPORTED_AT_CHECKS_ONLY' if good else 'UNVERIFIED_BACKGROUND_OR_TARGET',
                observations=within))
    return out


def compact_chain(c):
    return {k:v for k,v in c.items() if k!='observations'}


def identity_matrix(samples):
    vv=[s for s in samples if s['matched'] is not None]
    result=dict(visible_check_count=len(vv),rgb_indices=[s['rgb_index'] for s in vv],criteria={})
    for field in ('matched_ids','line_aligned_ids'):
        ids=sorted(set.union(*(set(s[field]) for s in vv))) if vv else []
        common=sorted(set.intersection(*(set(s[field]) for s in vv))) if vv else []
        result['criteria'][field]=dict(ids_at_all_checks=common,
            matrix={tid:[tid in s[field] for s in vv] for tid in ids},
            maximum_checks_one_id_covers=max((sum(tid in s[field] for s in vv) for tid in ids),default=0))
    return result


def evaluate():
    start=time.perf_counter();cam=AtomCamera()
    receipt=json.loads((BASE/'tracking/tracking_receipt.json').read_text(encoding='utf8'))
    assert receipt['complete'] is True
    assert json.loads((BASE/'tracking/tracking_receipt_device_primary.json').read_text(encoding='utf8'))['complete'] is True
    if (BASE/'tracking/tracking_reacquire.jsonl').exists():
        assert json.loads((BASE/'tracking/tracking_receipt_reacquire.json').read_text(encoding='utf8'))['complete'] is True
    if (BASE/'tracking/tracking_grouped.jsonl').exists():
        assert json.loads((BASE/'tracking/tracking_receipt_grouped.json').read_text(encoding='utf8'))['complete'] is True
    plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf8'))
    seal=json.loads((OUT/'blind_label_receipt.json').read_text(encoding='utf8'))
    assert sha(OLD)==plan['original_annotations_sha256']
    assert sha(OLD_TRACK)==plan['original_tracking_source_sha256']
    assert sha(OUT/'blind_labels.json')==seal['sha256']
    primary=[s for s in json.loads(OLD.read_text(encoding='utf8'))['samples'] if s['session'] in KEYS]
    supplemental=json.loads((OUT/'blind_labels.json').read_text(encoding='utf8'))['samples']
    manifest=json.loads((BASE/'input/manifest.json').read_text(encoding='utf8'))
    metadata={(r['key'],r['rgb_index']):r for r in manifest['frames']}
    if len(metadata)!=len(manifest['frames']):raise ValueError('duplicate RGB locator in full manifest')
    intervals={}
    for key in KEYS:
        ss=[s for s in primary if s['session']==key and s['segment']]
        intervals[key]=(min(metadata[key,s['rgb_index']]['device_capture_us'] for s in ss),
                        max(metadata[key,s['rgb_index']]['device_capture_us'] for s in ss))
    oldrows=[r for r in read(OLD_TRACK) if r['key'] in KEYS]
    inputs={'old_sampled_camera_chain':(OLD_TRACK,oldrows,True)}
    for arm in ('baseline','continuous','device_primary','reacquire','grouped'):
        if arm in ('reacquire','grouped') and not (BASE/f'tracking/tracking_{arm}.jsonl').exists():continue
        path=BASE/f'tracking/tracking_{arm}.jsonl';inputs[f'fullrgb_{arm}']=(path,read(path),False)
    result=dict(plan_sha256=sha(OUT/'PLAN.json'),original_annotations_sha256=sha(OLD),
        supplemental_annotations_sha256=seal['sha256'],manifest_sha256=sha(BASE/'input/manifest.json'),
        evidence='Repeated consumed Development; fixed original assistant weak labels plus pre-output assistant weak intermediate RGB review',
        intervals_us=intervals,arms={},three_class_classification='NOT_EVALUABLE_NO_BODY_STILL_TURN_TRUTH')
    allchains={}
    for arm,(path,rows,isold) in inputs.items():
        sourceobs=observations(rows,metadata,isold)
        for mode in ('candidate_only','candidate_plus_flow'):
            obs=[o for o in sourceobs if mode=='candidate_plus_flow' or o['observation_kind']=='CURRENT_AUTONOMOUS_CANDIDATE']
            pri=sample_result(primary,obs,'original_primary',exact_old_ready=isold)
            supp=sample_result(supplemental,obs,'pre_output_weak_intermediate')
            combined={(s['key'],s['rgb_index']):s for s in pri if s['matched'] is not None}
            for s in supp:combined.setdefault((s['key'],s['rgb_index']),s)
            cc=chains(obs,list(combined.values()),metadata,intervals)
            tag=f'{arm}:{mode}';allchains[tag]=cc
            sessions={}
            for key in KEYS:
                pp=[s for s in pri if s['key']==key and s['matched'] is not None]
                ss=[s for s in supp if s['key']==key and s['matched'] is not None]
                target=[c for c in cc if c['key']==key and c['weak_target_matched_images']]
                longest=max(target,key=lambda c:(c['duration_s'],c['weak_target_matched_images']),default=None)
                best=max(target,key=lambda c:(c['weak_target_matched_images'],-c['weak_target_mismatch_images'],c['duration_s']),default=None)
                a,b=intervals[key];den=sum(r['key']==key and a<=r['device_capture_us']<=b for r in manifest['frames'])
                sessions[key]=dict(primary_visible=len(pp),primary_localized=sum(s['matched'] for s in pp),
                    primary_strict3px=sum(s['strict3px_matched'] for s in pp),
                    supplemental_visible=len(ss),supplemental_localized=sum(s['matched'] for s in ss),
                    supplemental_strict3px=sum(s['strict3px_matched'] for s in ss),
                    primary_all_candidate_identity_matrix=identity_matrix(pp),
                    supplemental_all_candidate_identity_matrix=identity_matrix(ss),
                    combined_distinct_check_identity_matrix=identity_matrix([s for s in combined.values() if s['key']==key]),
                    longest_weak_target_supported_chain=compact_chain(longest) if longest else None,
                    most_weak_target_checks_chain=compact_chain(best) if best else None,
                    target_supported_chains=len(target),fixed_visible_interval_s=(b-a)/1e6,
                    all_rgb_in_fixed_interval=den,
                    longest_associated_observation_coverage=longest['observed_images']/den if longest else 0,
                    coverage_scope='Actual same-ID emitted observations; target purity only audited at weak check frames. Sparse old sampled coverage has full-RGB denominator.',
                    geometry_mismatches_requiring_audit=[compact_chain(c) for c in target if c['weak_target_mismatch_images']])
            result['arms'][tag]=dict(source_sha256=sha(path),sessions=sessions,primary_samples=pri,supplemental_samples=supp,
                primary_total=dict(visible=sum(s['matched'] is not None for s in pri),localized=sum(bool(s['matched']) for s in pri)),
                emitted_observations=len(obs))
    # Paired motion only on identical weak anchors, same reference, same chosen interval.
    paired={}
    comparison_tags=['old_sampled_camera_chain:candidate_only','fullrgb_baseline:candidate_only','fullrgb_continuous:candidate_plus_flow','fullrgb_device_primary:candidate_plus_flow']
    if 'fullrgb_reacquire' in inputs:comparison_tags.append('fullrgb_reacquire:candidate_plus_flow')
    if 'fullrgb_grouped' in inputs:comparison_tags.append('fullrgb_grouped:candidate_plus_flow')
    for key in KEYS:
        tables=[{s['rgb_index']:s for s in result['arms'][t]['primary_samples'] if s['key']==key and s['matched']} for t in comparison_tags]
        common=sorted(set.intersection(*(set(t) for t in tables)))
        details={}
        for tag,table in zip(comparison_tags,tables):
            vv=[]
            for ri in common:
                chosen=min(table[ri]['matched_observations'],key=lambda o:(o['observation_kind']!='CURRENT_AUTONOMOUS_CANDIDATE',o['rank']))
                ref=reference(dict(segment=chosen['segment']),cam.K_upright)
                vv.append(dict(rgb_index=ri,track_id=chosen['track_id'],time_s=chosen['t_us']/1e6,reference=ref))
            angles=[v['reference'].get('bearing_deg') for v in vv]
            vals=[v for v in angles if v is not None]
            details[tag]=dict(samples=vv,common_sample_count=len(vv),
                same_selected_id= len({str(v['track_id']) for v in vv})==1 if vv else None,
                same_observed_continuous_episode=any(all(any(o['rgb_index']==v['rgb_index'] and str(o['track_id'])==str(v['track_id']) for o in c['observations']) for v in vv) for c in allchains[tag] if c['key']==key) if len(vv)>1 else None,
                fixed_reference_bearing_span_deg=max(vals)-min(vals) if vals else None,
                reference_inside_observations=sum(v['reference']['status']=='OBSERVED_IMAGE_SUPPORT' for v in vv),
                scope='Supporting line at fixed y=320; outside segment is extrapolated model proxy. Track switches or nonidentical physical endpoints forbid approach classification.')
        paired[key]=dict(common_rgb_indices=common,common_duration_s=(metadata[key,common[-1]]['device_capture_us']-metadata[key,common[0]]['device_capture_us'])/1e6 if len(common)>1 else None,arms=details)
    result['paired_primary_fixed_reference_motion']=paired
    result['frontal_vs_side_classification']=dict(status='DESCRIPTIVE_ONLY_PENDING_OBJECT_IDENTITY_AND_COMMON_TIME_SUPPORT',
        reasons=['Lowest-rank selected readout switches track ID; all-candidate strict/line-aligned ID coverage matrices are reported separately and determine object-identity support',
                 'Cross-clip available target observation durations differ; per-clip descriptive spans are not a matched class-separation result',
                 'Fixed y=320 is outside many visible shaft fragments, especially look-down frames; supporting line extrapolation is not observed physical-point motion'],
        direction='Target supporting-line bearing is more informative than changing fragment midpoint, but class discrimination still requires verified target continuity and matched time support')
    result['internal_wall_s']=time.perf_counter()-start
    save('evaluation.json',result)
    # Internal serialized chain audit includes actual pixels, never hidden coast predictions.
    audit=[]
    for tag,cc in allchains.items():
        if tag.startswith('old_') or tag.endswith('candidate_only'):continue
        for key in ('073554','073635','073821'):
            target=[c for c in cc if c['key']==key and c['weak_target_matched_images']]
            top=sorted(target,key=lambda c:(-c['duration_s'],-c['weak_target_matched_images']))[:2]
            if tag.startswith('fullrgb_device_primary') or tag.startswith('fullrgb_reacquire') or tag.startswith('fullrgb_grouped'):
                background=[c for c in cc if c['key']==key and c['weak_target_line_aligned_images']==0 and len(c['weak_checks'])>=2]
                if background:top.append(max(background,key=lambda c:c['duration_s']))
            for c in top:
                ee=c['observations'];ids=sorted(set(round(v*(len(ee)-1)) for v in (0,.25,.5,.75,1)))
                weak_by_ri={v['rgb_index']:v for v in c['weak_checks']}
                ids=sorted(set(ids+ [i for i,o in enumerate(ee) if o['rgb_index'] in weak_by_ri]))
                audit.append(dict(arm=tag,role='WEAK_TARGET_ASSOCIATED' if c['weak_target_matched_images'] else 'NON_TARGET_LONG_CHAIN_CONTROL',chain=compact_chain(c),frames=[dict(rgb_index=ee[i]['rgb_index'],segment=ee[i]['segment'],supporting_members=[dict(segment=m['segment'],kind=m['observation_kind']) for m in ee[i]['supporting_members']],kind=ee[i]['observation_kind'],time_us=ee[i]['t_us'],weak_check=weak_by_ri.get(ee[i]['rgb_index'])) for i in ids]))
    save('audit_selection.json',audit)
    print(json.dumps({tag:{k:{q:v[q] for q in ('primary_visible','primary_localized','supplemental_visible','supplemental_localized')} for k,v in a['sessions'].items()} for tag,a in result['arms'].items()},indent=2))


def audit():
    cam=AtomCamera();manifest=json.loads((BASE/'input/manifest.json').read_text(encoding='utf8'))
    meta={(r['key'],r['rgb_index']):r for r in manifest['frames']}
    items=json.loads((OUT/'audit_selection.json').read_text(encoding='utf8'))
    for j,item in enumerate(items):
        key=item['chain']['key'];ims=[]
        for f in item['frames']:
            im=cam.undistort_upright(cv2.imread(meta[key,f['rgb_index']]['rgb_file']))
            for member in f.get('supporting_members',[f]):
                a,b=np.asarray(member['segment']).round().astype(int).reshape(2,2)
                cv2.line(im,tuple(a),tuple(b),(0,255,0),2)
            if f.get('weak_check'):
                a,b=np.asarray(f['weak_check']['weak_segment']).round().astype(int).reshape(2,2)
                cv2.line(im,tuple(a),tuple(b),(255,0,255),1)
            banner=np.zeros((42,480,3),np.uint8)
            text=f'{key} tid={item["chain"]["track_id"]} rgb={f["rgb_index"]} {f["kind"]}'
            cv2.putText(banner,text,(3,21),0,.4,(255,255,255),1)
            if f.get('weak_check'):
                cv2.putText(banner,f['weak_check']['geometry_diagnostic']['status'],(3,38),0,.38,(255,255,255),1)
            ims.append(np.vstack((banner,im)))
        for part,lo in enumerate(range(0,len(ims),5)):
            cv2.imwrite(str(OUT/f'track_audit_{j:02d}_part{part}.jpg'),np.hstack(ims[lo:lo+5]))
    print(len(items),'track audit sheets')


def dense_side():
    """Every real RGB in the fixed visible side interval; draw actual members only."""
    result=json.loads((OUT/'evaluation.json').read_text(encoding='utf8'))
    if 'fullrgb_grouped:candidate_plus_flow' not in result['arms']:
        raise ValueError('grouped evaluation must finish before dense side audit')
    session=result['arms']['fullrgb_grouped:candidate_plus_flow']['sessions']['073635']
    mat=session['combined_distinct_check_identity_matrix']['criteria']
    candidates=mat['line_aligned_ids']['ids_at_all_checks']
    if not candidates:
        save('dense_side_receipt.json',dict(status='NO_FAMILY_OBJECT_ID_AT_ALL_WEAK_CHECKS',
            reason='Complete interval identity claim is already unsupported; representative and weak intermediate audits remain available',
            checked_original_primary=6,checked_distinct_weak=session['combined_distinct_check_identity_matrix']['visible_check_count']))
        return
    ids=sorted(candidates,key=lambda tid:(-sum(mat['matched_ids']['matrix'].get(tid,[])),tid))[:2]
    manifest=json.loads((BASE/'input/manifest.json').read_text(encoding='utf8'))
    meta={(r['key'],r['rgb_index']):r for r in manifest['frames']}
    obs=observations(read(BASE/'tracking/tracking_grouped.jsonl'),meta)
    lookup=defaultdict(list)
    for o in obs:lookup[o['key'],o['rgb_index'],str(o['track_id'])].append(o)
    a,b=result['intervals_us']['073635']
    frames=[r for r in manifest['frames'] if r['key']=='073635' and a<=r['device_capture_us']<=b]
    assert len(frames)==58
    cam=AtomCamera();gallery=[]
    for tid in ids:
        images=[];observed=[]
        for m in frames:
            im=cam.undistort_upright(cv2.imread(m['rgb_file']))
            current=lookup['073635',m['rgb_index'],tid]
            for o in current:
                p,q=np.asarray(o['segment']).round().astype(int).reshape(2,2)
                cv2.line(im,tuple(p),tuple(q),(0,255,0),2)
            im=cv2.resize(im,(240,320));bar=np.zeros((35,240,3),np.uint8)
            title=f'object={tid} rgb={m["rgb_index"]} obs={len(current)}'
            cv2.putText(bar,title,(2,13),0,.35,(255,255,255),1)
            cv2.putText(bar,'REAL MEMBER SUPPORT' if current else 'NO CURRENT OBSERVATION',(2,29),0,.32,(255,255,255),1)
            images.append(np.vstack((bar,im)))
            observed.append(dict(rgb_index=m['rgb_index'],device_us=m['device_capture_us'],actual_member_observations=len(current),
                source_episode_ids=sorted({o['source_episode_id'] for o in current if o['source_episode_id'] is not None})))
        files=[]
        for part,lo in enumerate((0,29)):
            items=images[lo:lo+29];items+= [np.zeros_like(images[0])]*(30-len(items))
            sheet=np.vstack([np.hstack(items[i:i+6]) for i in range(0,30,6)])
            path=OUT/f'dense_side_object_{tid}_part{part}.jpg';cv2.imwrite(str(path),sheet);files.append(str(path))
        gallery.append(dict(object_id=tid,files=files,frames=observed,full_interval_rgb_count=58,
            currently_observed_frames=sum(bool(r['actual_member_observations']) for r in observed),
            purity='PENDING_DENSE_ASSISTANT_VISUAL_AUDIT; never inferred from weak endpoints alone'))
    save('dense_side_receipt.json',dict(status='READY_FOR_VISUAL_AUDIT',gallery=gallery,
        predictions_drawn=False,composite_span_drawn=False,source_sha256=sha(BASE/'tracking/tracking_grouped.jsonl')))


def validate():
    # Meaningful endpoints/clock/dedup contracts, not an implementation mirror.
    assert matches([10,0,10,100],dict(x1=10,y1=100,x2=10,y2=0))
    assert not matches([10,0,10,100],dict(x1=0,y1=50,x2=100,y2=50))
    assert not matches([10,0,10,100],dict(x1=10,y1=95,x2=10,y2=125))
    K=np.array([[400,0,240],[0,400,320],[0,0,1.]])
    assert reference(dict(segment=[10,400,10,600]),K)['status']=='MODEL_EXTRAPOLATED_SUPPORTING_LINE'
    assert reference(dict(segment=[10,100,10,600]),K)['status']=='OBSERVED_IMAGE_SUPPORT'
    assert geometry_diagnostic([10,0,10,100],dict(segment=[10,95,10,125]))['status']=='SHORT_OR_DISJOINT_FRAGMENT'
    assert geometry_diagnostic([10,0,10,100],dict(segment=[70,0,70,100]))['status']=='ANGLE_OR_NORMAL_OFFSET_MISMATCH'
    seal=json.loads((OUT/'blind_label_receipt.json').read_text(encoding='utf8'))
    assert seal['sha256']==sha(OUT/'blind_labels.json')
    assert seal['new_algorithm_outputs_read_before_seal'] is False
    samples=json.loads((OUT/'blind_labels.json').read_text(encoding='utf8'))['samples']
    assert len({(s['session'],s['rgb_index']) for s in samples})==36
    manifest=json.loads((BASE/'input/manifest.json').read_text(encoding='utf8'))
    meta={(r['key'],r['rgb_index']):r for r in manifest['frames']}
    for s in samples:
        m=meta[s['session'],s['rgb_index']]
        assert m['device_capture_us']==s['device_capture_us']
        assert m['rgb_sha256']==s['raw_sha256']
    primary=[s for s in json.loads(OLD.read_text(encoding='utf8'))['samples'] if s['session'] in KEYS]
    oldrows=[r for r in read(OLD_TRACK) if r['key'] in KEYS]
    oo=observations(oldrows,meta,old=True)
    exact=sample_result(primary,oo,'original_primary',exact_old_ready=True)
    assert sum(bool(s['matched']) for s in exact)==17
    assert next(s for s in exact if s['key']=='073635' and s['primary_i']==69)['matched'] is False
    pooled=sample_result(primary,oo,'same_rgb_pooled_diagnostic',exact_old_ready=False)
    assert next(s for s in pooled if s['key']=='073635' and s['primary_i']==69)['matched'] is True
    first=oo[0];interval={first['key']:(first['t_us']-1,first['t_us']+700000)}
    dedup=chains([first,first],[],meta,interval)
    assert sum(c['observed_images'] for c in dedup)==1
    resumed=dict(first,t_us=first['t_us']+100000,rgb_index=first['rgb_index']+1,source_episode_id=1)
    discontinuous=chains([dict(first,source_episode_id=0),resumed],[],meta,interval)
    assert len(discontinuous)==2 and all(c['observed_images']==1 for c in discontinuous)
    gap=dict(resumed,t_us=first['t_us']+600000,source_episode_id=0)
    assert len(chains([dict(first,source_episode_id=0),gap],[],meta,interval))==2
    fakeflow=dict(x1=10,y1=0,x2=10,y2=100,track_id=9,object_id=9,
        supporting_subsegments=[[10,0,10,10],[10,90,10,100]])
    fakeobs=observations([dict(key=first['key'],rgb_index=first['rgb_index'],candidates=[],flow_observations=[fakeflow])],meta)
    assert len(fakeobs)==2 and all(o['track_id']==9 for o in fakeobs)
    assert not any(matches([10,20,10,80],o['candidate']) for o in fakeobs)
    assert matches([10,20,10,80],fakeflow)
    save('verification.json',dict(status='PASS',checks=['endpoint order, crossing/tiny-overlap rejection','short fragment vs line-offset failure separation','common reference extrapolation explicit','sealed weak labels unchanged','36 distinct RGB locators and exact device-time/hash match','historical primary exact ready_i restores 17/22; i69/rgb67 alternate ready-row pooling cannot rewrite anchor','same capture duplicate never increments observed count; reacquired source episode and >0.5s capture gap split real-observation continuity','family composite span cannot count unsupported pixel gap; actual subsegments tested individually'],gpu_s=0))
    print('PASS')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'evaluate', 'validate', 'audit','dense-side'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'prepare':prepare,'evaluate':evaluate,'audit':audit,'validate':validate,'dense-side':dense_side}[args.stage]()


if __name__ == '__main__':
    main()

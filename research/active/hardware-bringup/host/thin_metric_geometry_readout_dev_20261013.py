"""Readout only: weak-label observability audit, never metric accuracy.

The generating scripts do not read annotations. This evaluator applies the old
fixed image-line match rule and records a separate assistant visual contact
audit. A conservative visible segment is deliberately not a true foot label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from atom_camera import AtomCamera, ROOT
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_query_feasibility_dev_20261012 as F
from thin_metric_geometry_dev_20261013 import CACHE, OUT, SOURCE, geometry
from thin_structure_evaluate_dev_20261013 import matches

ANNOTATIONS = ROOT / 'artifacts.local/work/thin-structure-dev-20261013/annotations.json'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT/'readout.json')
    args=parser.parse_args()
    start = time.perf_counter(); cpu = time.process_time()
    output = args.output
    if output.exists():
        raise FileExistsError('Preserve readout evidence; choose a new run directory in source')
    annotations = json.loads(ANNOTATIONS.read_text())
    manifest = json.loads((CACHE / 'cache_manifest.json').read_text())
    source = {(r['key'],r['i']):r for r in map(json.loads,SOURCE.read_text().splitlines())}
    generated = {(r['key'],r['i']):r for r in map(json.loads,(OUT/'frames.jsonl').read_text().splitlines())}
    meta = {(k,r['i']):r for k,s in manifest['sessions'].items() for r in s['frames']}
    K = AtomCamera().K_upright
    Rct = np.asarray(json.loads((X.OUT/'camera_tof_extrinsic.json').read_text())['rotation'])
    counts = Counter(); rows = []; errors = []
    # Inspection covered the four original six-panel RGB sheets in this run.
    contact_audit = dict(provenance='Assistant visual reinspection of the four original raw RGB sheets; not human metric truth',
                         vertical_target_frames_reviewed=18, visible_verified_rod_floor_contact_points=0,
                         vertical_reason='Rod lower portions disappear behind poufs/seats or the image boundary; line endpoints are visible-segment truncations.',
                         horizontal_target_frames_reviewed=6, horizontal_reason='The crossbar is held by a person; no rod-ground contact.',
                         coordinates_added=0, original_annotation_file_modified=False)
    for (key,i), old in source.items():
        new=generated[key,i]
        assert len(old['candidates'])==len(new['candidates'])
        for j,(a,b) in enumerate(zip(old['candidates'],new['candidates'],strict=True)):
            assert b['candidate_index']==j and a['visual_candidate']==b['visual_candidate']
            assert b['distance_verified'] is False and b['safety_decision'] is None
        counts['frames_identity_verified']+=1; counts['candidates_identity_verified']+=len(old['candidates'])
    for s in annotations['samples']:
        key,i=s['session'],s['i']; frame=generated[key,i]
        row=dict(key=key,i=i,weak_label_status=s['label_status'],matches=[],oracle_visible_segment=None)
        if s['segment']:
            m=meta[key,i]; B=F.rot(m['pitch'],m['roll'])@Rct.T
            c=dict(zip(('x1','y1','x2','y2'),s['segment']))
            row['oracle_visible_segment']=geometry(c,K,B,manifest['height'])
            row['oracle_role']='Evaluator-only visible-segment endpoints, not an automatic foot or metric reference'
            found=[(j,c) for j,c in enumerate(frame['candidates']) if matches(s['segment'],c['visual_candidate'])]
            for j,c in found:
                old=source[key,i]['candidates'][j]
                row['matches'].append(dict(candidate_index=j,gravity_error_deg=c.get('gravity_alignment_error_deg'),
                                           upper_z_over_height=c.get('conditional_ground_upper_z_over_height'),
                                           provisional_upper_z_m=c.get('provisional_upper_z_m'),
                                           old_unvalidated_camera_z_m=old['unvalidated_camera_z_estimate_m'],
                                           old_distance_state=old['distance_state'],
                                           possible_HEAD_rays=c.get('box_ray_geometry',{}).get('HEAD',{}).get('possible_ray_points'),
                                           possible_BODY_rays=c.get('box_ray_geometry',{}).get('BODY',{}).get('possible_ray_points')))
            counts['evaluable_weak_line_frames']+=1
            counts['located_weak_line_frames']+=bool(found)
            if key != '073740':
                errors.append(row['oracle_visible_segment']['gravity_alignment_error_deg'])
                counts['vertical_weak_line_frames']+=1
                counts['vertical_line_over_15deg_gravity_error']+=row['oracle_visible_segment']['gravity_alignment_error_deg']>15
        rows.append(row)
    multiview={}
    for name,folder in [('sift','multiview'),('lk','multiview-lk')]:
        r=json.loads((OUT/folder/'results.json').read_text())
        eligible=[p for p in r['pairs'] if p.get('homography_support_pass')]
        normals=[min(x['normal_error_deg'] for x in p['solutions'] if x['normal_error_deg'] is not None) for p in eligible]
        repro=[p['reprojection_median_px'] for p in eligible]
        multiview[name]=dict(counts=r['counts'],support_pass_pairs=len(eligible),
                             best_normal_error_deg_range=[min(normals),max(normals)] if normals else None,
                             reprojection_median_px_range=[min(repro),max(repro)] if repro else None,
                             pairs_with_20_or_more_inliers=sum(p['inliers']>=20 for p in r['pairs']),
                             inlier_count_range=[min(p['inliers'] for p in r['pairs']),max(p['inliers'] for p in r['pairs'])],
                             wall_s=r['wall_s'],process_cpu_s=r['process_cpu_s'],checks=r['checks'])
    allpairs=[json.loads((OUT/f/'results.json').read_text())['plan']['pairs'] for f in ('multiview','multiview-lk')]
    assert allpairs[0]==allpairs[1]
    boundary=json.loads((OUT/'summary.json').read_text())
    result=dict(conclusion='Ground-contact substitution is not identified in these clips, and neither fixed multiview correspondence arm yields an admissible floor-to-motion solution.',
                next_decision='Acquire a measured 20 cm lateral camera displacement, optionally at positions 0/10/20 cm, with a verified vertical rod and ruler-measured installation height; add matte floor texture if estimating motion from a floor homography. This changes the next collection proposal, not either existing trial.',
                no_claim='No new metric-distance accuracy, detection gain, safe clearance, or alarm promotion.',
                counts=counts,contact_audit=contact_audit,vertical_weak_line_gravity_error_deg_range=[min(errors),max(errors)],
                source_annotation_sha256=hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest(),
                original_annotation_role=annotations['provenance'], rows=rows,multiview=multiview,
                multiview_visual_audit=dict(inspected_sift_pair_indices=[0,6,9],inspected_lk_pair_indices=[3,4,7,11],
                                            finding='ROI contains specular floor highlights and some seat boundaries. This is not verified same-plane floor support; low reprojection residual does not establish the physical plane.',
                                            uninspected_panels_remain_unadjudicated=True),
                checks=dict(candidate_identity=True,source_candidate_count=18483,frame_count=1677,
                            matched_old_weak_line_count_17_of_22=counts['located_weak_line_frames']==17,
                            same_pairs_sift_and_lk=True,ground_and_ray_checks=boundary['checks']),
                compute=dict(boundary_cpu_s=boundary['process_cpu_s'],
                             multiview_cpu_s=sum(x['process_cpu_s'] for x in multiview.values()),
                             readout_process_cpu_s=time.process_time()-cpu,
                             successful_first_readout_process_cpu_s=.328125,
                             total_accounted_cpu_s_conservative=30,
                             accounting_note='Successful main-loop CPU is measured; conservative 30 s also covers the initial failed export, imports, inspection readouts and final regeneration.',
                             total_branch_cpu_limit_s=900,gpu_s=0,new_model_inference=0,downloads=0),
                failure_history=['Initial boundary export stopped on NaN pose JSON serialization; partial output retained; fixed to explicit null without changing geometry.',
                                 'SIFT descriptor matching failed to provide spatial ground support on all 12 pairs.',
                                 'KLT increased support; none of the 12 pairs established an admissible ground-motion solution. No geometric threshold or mask retuning followed.'],
                evidence_limit='Consumed Development; camera height inherited from ToF floor fit rather than ruler measurement; pose/extrinsics and floor/object assumptions remain unverified.',
                wall_s=time.perf_counter()-start)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:result[k] for k in ('counts','contact_audit','vertical_weak_line_gravity_error_deg_range','multiview','compute')}))


if __name__=='__main__':
    main()

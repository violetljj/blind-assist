"""Fixed-identity height-transfer geometry; CPU only, no scores or rendering."""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
# Existing project-local SciPy for the retained geometric clipping helper.
# This adds no model/score module and performs no installation.
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/
    'artifacts.local/work/cnh-direction-information-dev-20261007/runtime'))
import cnh_double_height_geometry_dev as D

G = D.G
ROOT, WORK = D.ROOT, D.WORK
OUT = WORK/'cnh-height-transfer-dev-20261007'
SOURCE = D.OUT/'geometry_manifest.json'
PARENT_PLAN = WORK/'cnh-training-support-dev-20261007/PLAN.json'
HEIGHTS = {'short':(.14,.60),'up':(-.30,.64),'down':(.10,1.04),'tall':(-.30,1.04)}
CONTACT_CATS = D.CONTACT_CATS


def save(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(D._finite_json(value),stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def variant_boxes(scene,tag):
    """Change only target y limits; preserve x/z, reflectance and backgrounds."""
    low,high = HEIGHTS[tag]
    boxes = copy.deepcopy(scene['boxes'])
    boxes[0]['lo'][1],boxes[0]['hi'][1] = low,high
    return boxes


def reference_consistent(reference,original):
    if bool(reference['covered']) != bool(original['covered']):
        return False
    if not reference['covered']:
        return reference['censor_reason']==original['censor_reason']
    return (abs(reference['reference_fraction']-original['reference_fraction'])<=1e-10
            and np.allclose(reference['reference_pose'],original['reference_pose'],atol=1e-10,rtol=0)
            and reference['crossing_left_frame']==original['crossing_left_frame']
            and reference['crossing_right_frame']==original['crossing_right_frame'])


def validate_variant(scene,travel,role,tag,original_reference):
    if role not in ('contact','clear'):
        raise ValueError('Original role must be contact or clear')
    travel = np.asarray(travel,float)
    if travel.shape!=(16,4,4):
        raise ValueError('Original 16 travel poses required')
    boxes = variant_boxes(scene,tag)
    target = boxes[0]
    overlaps = [j for j,b in enumerate(boxes[1:],1)
                if np.all(np.minimum(target['hi'],b['hi'])-np.maximum(target['lo'],b['lo'])>G.EPS)]
    poses = travel[G.FRAMES]
    meshes = [G.S.box_mesh(b['lo'],b['hi']) for b in boxes]
    mesh = np.concatenate(meshes)
    ranges,reference = G.deadline_reference(G.corners(target),poses)
    frames = [[G.surface_category(G.transform(mesh,pose),q) for q in (0,1)] for pose in poses]
    target_frames = [[G.surface_category(G.transform(meshes[0],pose),q) for q in (0,1)] for pose in poses]
    if reference['covered']:
        ref = [G.surface_category(G.transform(mesh,reference['reference_pose']),q) for q in (0,1)]
        target_ref = [G.surface_category(G.transform(meshes[0],reference['reference_pose']),q) for q in (0,1)]
        idx = int(np.searchsorted(G.FRAMES,reference['reference_fraction']+1e-10,side='right')-1)
    else:
        ref = target_ref = ['censored']*2
        idx = None
    cq = [c in CONTACT_CATS for c in ref]
    tcq = [c in CONTACT_CATS for c in target_ref]
    clear = (np.asarray(frames)=='clear').all(0).tolist()
    target_clear = (np.asarray(target_frames)=='clear').all(0).tolist()
    same_reference = reference_consistent(reference,original_reference)
    reasons = []
    if overlaps:reasons.append('target_background_overlap')
    if np.any(np.diff(ranges)>1e-10):reasons.append('nonmonotonic_range')
    # A contact deadline must be observed. Existing censored clear sequences
    # remain valid controls when their censoring/reference and all-frame clear
    # geometry are unchanged; missing a crossing is not an obstacle label.
    if role=='contact' and not reference['covered']:reasons.append('deadline_censored')
    if not same_reference:reasons.append('original_reference_changed')
    if 'frame_ranges' in original_reference and not np.allclose(
            ranges,original_reference['frame_ranges'],atol=1e-10,rtol=0):
        reasons.append('original_target_front_ranges_changed')
    if reference['covered'] and idx!=original_reference['deadline_index']:
        reasons.append('original_causal_deadline_changed')
    if role=='contact':
        if cq!=[True,True]:reasons.append('all_box_contact_pattern')
        if tcq!=[True,True]:reasons.append('target_contact_pattern')
    else:
        if not all(clear):reasons.append('not_both_query_clear_all')
        if not all(target_clear):reasons.append('target_not_both_query_clear_all')
    valid = not reasons
    degree = [CONTACT_CATS.index(c) if c in CONTACT_CATS else None for c in ref]
    return D._finite_json(dict(tag=tag,boxes=boxes,valid=valid,reasons=reasons,
        contact=bool(any(cq)) if valid else None,control=bool(all(clear)) if valid else None,
        contact_query=cq,target_contact_query=tcq,physical_contact=bool(any(cq)),
        physical_control=bool(all(clear)),clear_all=clear,target_clear_all=target_clear,
        contact_degree=degree,contact_degree_definition='0=0-2cm,1=2-5cm,2=>5cm; null=no contact',
        frame_category=frames,target_frame_category=target_frames,
        ref_category=ref,target_ref_category=target_ref,frame_ranges=ranges,deadline_index=idx,
        changed_target_background_overlaps=overlaps,original_reference_consistent=same_reference,
        reusable_original_observation=bool(boxes==scene['boxes']),
        reusable_source_hb_observation=bool(tuple(HEIGHTS[tag])==tuple(D.HEIGHTS['HB'])),
        **reference))


def checked_anchors():
    source = D.read(SOURCE)
    anchors = source['anchors']
    parent = D.read(PARENT_PLAN)
    fold = {}
    for fi,roles in enumerate(parent['folds']):
        for unit in roles['evaluation']:
            if unit in fold:raise ValueError('Repeated evaluation unit in parent folds')
            fold[unit]=fi
    if len(anchors)!=72 or len({a['anchor_id'] for a in anchors})!=72:
        raise ValueError('Exactly the original 72 fixed anchor identities required')
    if Counter(a['role'] for a in anchors)!=Counter(contact=36,clear=36):
        raise ValueError('Original role denominator must remain 36/36')
    for anchor in anchors:
        if fold[anchor['unit']]!=anchor['fold']:
            raise ValueError('Anchor fold differs from frozen candidate parent fold')
        if anchor['source_variant'] not in ('H','B'):
            raise ValueError('Unknown original physical source variant')
    return anchors


def freeze_geometry_plan(outdir=OUT):
    outdir = Path(outdir)
    anchors = checked_anchors()
    paths = [Path(__file__),Path(D.__file__),Path(G.__file__),Path(G.S.__file__),
             Path(sys.modules['cnh_track_a_geometry'].__file__),SOURCE,PARENT_PLAN,
             *[D.source_paths(u)['truth'] for u in sorted({a['unit'] for a in anchors})]]
    plan = dict(status='frozen',geometry_budget_seconds=300,no_score_access=True,
        no_render_or_training=True,variants={k:list(v) for k,v in HEIGHTS.items()},
        identities=[{k:a[k] for k in ('anchor_id','unit','config','fold','role','source_variant')} for a in anchors],
        expected_anchors=72,expected_variant_records=288,
        selection='Reuse all original 72 identities and order, no reselection, replacement or changed bounds',
        geometry='Only target y bounds change; all-box and target-only geometry; original 0.9m target-front crossing and causal deadline',
        expected_roles='contact: covered deadline and both HEAD/BODY contact; clear: both queries clear at all 13 outputs, unchanged original censoring permitted',
        invalid='Preserve every record and reasons; contact/control null for invalid; never infer negative from invalid',
        observation_contract='Reuse original travel, noisy poses, head errors, photon seed, reflectance and backgrounds; new rendering must use actual changed boxes',
        hashes={p.relative_to(ROOT).as_posix():D.sha(p) for p in paths})
    outdir.mkdir(parents=True,exist_ok=True)
    save(outdir/'GEOMETRY_PLAN.json',plan)
    return plan


def build_manifest(outdir=OUT,budget_seconds=300):
    outdir = Path(outdir)
    start = time.monotonic()
    plan_path = outdir/'GEOMETRY_PLAN.json'
    plan = D.read(plan_path)
    budget_seconds = min(float(budget_seconds),float(plan['geometry_budget_seconds']))
    if not np.isfinite(budget_seconds) or budget_seconds<=0:raise ValueError('Positive finite budget required')
    for name in ('geometry_manifest.json','common_valid_anchor_ids.json','GEOMETRY_STARTED.json'):
        if (outdir/name).exists():raise FileExistsError(outdir/name)
    save(outdir/'GEOMETRY_STARTED.json',dict(unix=time.time(),budget_seconds=budget_seconds))
    def check():
        if time.monotonic()-start>budget_seconds:raise TimeoutError('Frozen CPU geometry budget reached')
    try:
        for path,expected in plan['hashes'].items():
            check()
            if D.sha(ROOT/path)!=expected:raise ValueError(f'Frozen source changed: {path}')
        if plan['variants']!={k:list(v) for k,v in HEIGHTS.items()}:raise ValueError('Frozen height bounds changed')
        anchors = checked_anchors()
        identities = [{k:a[k] for k in ('anchor_id','unit','config','fold','role','source_variant')} for a in anchors]
        if identities!=plan['identities']:raise ValueError('Frozen identities changed')
        cache,result = {},[]
        for ai,anchor in enumerate(anchors):
            check()
            u,c = anchor['unit'],anchor['config']
            if u not in cache:cache[u]=D.read(D.source_paths(u)['truth'])
            truth = cache[u]
            scenes = [s for s in truth['scenes'] if s['config']==c]
            if len(scenes)!=1:raise ValueError('Source scene identity absent or repeated')
            scene = scenes[0]
            if scene['boxes']!=anchor['variants'][anchor['source_variant']]['boxes']:
                raise ValueError('Original anchor source physical scene changed')
            item = {k:copy.deepcopy(v) for k,v in anchor.items() if k!='variants'}
            item['variants']={}
            for tag in HEIGHTS:
                check()
                item['variants'][tag]=validate_variant(scene,truth['travel'],anchor['role'],tag,anchor['variants']['HB'])
            item['all_variants_valid']=all(v['valid'] for v in item['variants'].values())
            result.append(item)
            if (ai+1)%12==0:print(f'Geometry {ai+1}/72 anchors; {time.monotonic()-start:.1f}s',flush=True)
        common = [a['anchor_id'] for a in result if a['all_variants_valid']]
        counts = {}
        for tag in HEIGHTS:
            counts[tag]=dict(total=72,valid=sum(a['variants'][tag]['valid'] for a in result),
                invalid=sum(not a['variants'][tag]['valid'] for a in result),
                valid_by_role=dict(Counter(a['role'] for a in result if a['variants'][tag]['valid'])),
                invalid_by_reason=dict(Counter(r for a in result for r in a['variants'][tag]['reasons'])))
        check()
        manifest = dict(status='COMPLETE',anchors=result,variants=plan['variants'],
            query_order=['HEAD','BODY'],output_frames=G.FRAMES.tolist(),original_sequence_frames=list(range(16)),
            counts=dict(anchors=72,variant_records=288,by_variant=counts,common_valid_anchors=len(common),
                        common_valid_by_role=dict(Counter(a['role'] for a in result if a['all_variants_valid']))),
            common_valid_anchor_ids=common,hashes=plan['hashes'],geometry_plan_sha256=D.sha(plan_path),
            no_model_scores_read=True,no_render_or_training=True,budget_seconds=budget_seconds,seconds=time.monotonic()-start,
            limitations=['Consumed synthetic fixed source anchors; new height structures are not independent real scenes',
                        'Invalid geometry is retained with null contact/control, not a negative example'])
        save(outdir/'geometry_manifest.json',manifest)
        save(outdir/'common_valid_anchor_ids.json',dict(anchor_ids=common,count=len(common),criterion='All four fixed variants valid'))
        save(outdir/'GEOMETRY_TERMINAL.json',dict(status='complete',seconds=time.monotonic()-start))
        return manifest
    except BaseException as exc:
        save(outdir/'GEOMETRY_TERMINAL.json',dict(status='failed_or_budget_stop',seconds=time.monotonic()-start,error=repr(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run'])
    parser.add_argument('--outdir',type=Path,default=OUT)
    args=parser.parse_args()
    value=freeze_geometry_plan(args.outdir) if args.stage=='freeze' else build_manifest(args.outdir)
    print(json.dumps({k:value[k] for k in ('status','counts') if k in value},indent=2))

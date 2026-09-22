"""Governed read-only source audit of frozen contact-boundary predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

from boundary_error_math import (label_bracket, curve_crossing, disjoint_crossing,
    point_boundary, support_span, stats)
from contact_boundary_data import queries, camera_boxes, contact_labels, critical_values, boundary_metrics
from query_occupancy_data import read, write, sha, new_stage_directory, observation_tokens, FOCAL
from ba_camera_corridor import sample_indices, sample_native
from tof_fov45_core import boxes45, simulate

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
ART=REPO/'artifacts.local'
OLD=ART/'evidence/ba-contact-boundary-20260923'
SOURCE=ART/'evidence/ba-query-occupancy-20260922'
PREPARED=SOURCE.with_name(SOURCE.name+'-prepared')
CAPTURE=SOURCE.with_name(SOURCE.name+'-capture')
DEST=ART/'evidence/ba-boundary-error-20260923-v3'
ARMS=('direct','geometry')
KINDS=('width','horizon')
SPLITS=('train','dev','evaluation')


def spec():
    entries=[('saved',OLD,'evaluator'),('observations',PREPARED/'observations','observation'),
             ('geometry',CAPTURE/'evaluator','evaluator'),('plan',SOURCE/'plan','configuration')]
    s=dict(schema='blindassist-asset-run-v1',id='boundary-error-20260923-v3',route='ue-query-occupancy',
        question='Do metric boundary errors exceed coarse supervision ambiguity, with native support retained?',
        evaluator='research/active/dtr-r0/nearfield/run_boundary_error.py',
        evidence_boundary='Consumed simulation saved-output diagnostic; evaluator support never a predictor',
        reuse=dict(mode='diagnostic',query='contact boundary saved public features monotonic width horizon labels native support'),
        inputs=[dict(alias=a,path=str(p),role=r,purpose='consumed-boundary-error-diagnostic') for a,p,r in entries],
        outputs=[dict(alias='result',path=str(DEST/'result.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,'-B',str(Path(__file__).resolve()),'run'])
    p=DEST.with_name(DEST.name+'-run.json');write(p,s);print(p)


def nullable(v):
    if isinstance(v,np.generic):return nullable(v.item())
    if isinstance(v,dict):return {k:nullable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [nullable(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def axis_for(kind):
    return np.linspace(.2,1.2,51) if kind=='width' else np.linspace(.3,3.,55)


def native_diagnostic(geo,case,tof,truth):
    path=CAPTURE/'evaluator'/geo['native_path']
    assert sha(path)==geo['native_sha256'],str(path)
    depth=np.load(path,allow_pickle=False)
    values,traces=simulate(sample_native(depth),'query-occupancy/'+case['sensor_noise_key'],boxes45())
    assert np.array_equal(observation_tokens(values,boxes45()),tof),'Public ToF mismatch'
    yy,xx=np.indices(depth.shape)
    valid=np.isfinite(depth)&(depth>0)
    z=np.where(valid,depth,0).astype(float)
    full=np.stack(((xx+.5-320)/FOCAL*z,(yy+.5-180)/FOCAL*z,z),-1).reshape(-1,3)
    ny,nx=sample_indices();native_ids=(ny[:,None]*640+nx[None,:]).ravel()
    sampled=full[native_ids];sampled_valid=valid.ravel()[native_ids]
    zone_ids=np.full((192,256),-1,int)
    for zi,(y0,x0,y1,x1) in enumerate(boxes45()):zone_ids[y0:y1,x0:x1]=zi
    zone_ids=zone_ids.ravel()
    observed=np.zeros(len(sampled),bool)
    for t in traces:
        if t['observed']:observed[t['pixel_indices']]=True
    result={}
    for kind in KINDS:
        for layer in range(2):
            native,_=point_boundary(full,valid.ravel(),kind,layer)
            lattice,_=point_boundary(sampled,sampled_valid&(zone_ids>=0),kind,layer)
            returned,point=point_boundary(sampled,sampled_valid&observed,kind,layer)
            target=float(truth[kind][layer])
            if not np.isfinite(target):target=None
            near_native,_=point_boundary(full,valid.ravel(),kind,layer,target)
            near_lattice,_=point_boundary(sampled,sampled_valid&(zone_ids>=0),kind,layer,target)
            near_returned,point=point_boundary(sampled,sampled_valid&observed,kind,layer,target)
            record=dict(native=native,lattice=lattice,returned=returned,zone=None,
                        near_native=near_native,near_lattice=near_lattice,near_returned=near_returned,
                        sampled_point=point,native_point=None,zone_span=None,ray_span=None,
                        actual_point_in_public_interval=None)
            if target is not None:
                assert abs(near_native-target)<=abs(near_lattice-target)+1e-9
                assert abs(near_lattice-target)<=abs(near_returned-target)+1e-9
            if point is not None:
                zi=int(zone_ids[point]);px=sampled[point];ax,ay=px[:2]/px[2]
                d=float(values[zi]);r=.1+3*(.01+.02*d);interval=(max(.1,d-r),d+r)
                y0,x0,y1,x1=boxes45()[zi]
                # Entire normalized image-edge footprint, same as public token geometry.
                axs=((x0*640/256-320)/FOCAL,(x1*640/256-320)/FOCAL)
                ays=((y0*360/192-180)/FOCAL,(y1*360/192-180)/FOCAL)
                zone=support_span(axs,ays,interval,kind,layer)
                ray=support_span((ax,ax),(ay,ay),interval,kind,layer)
                record.update(zone=zi,zone_span=None if zone is None else zone[1]-zone[0],
                    native_point=int(native_ids[point]),
                    ray_span=None if ray is None else ray[1]-ray[0],
                    actual_point_in_public_interval=bool(interval[0]<=px[2]<=interval[1]),
                    public_interval=list(interval),zone_extent=zone,ray_extent=ray)
            result[(kind,layer)]=record
    return result


def summarize(rows):
    finite=[r for r in rows if r['true_class']=='interior']
    errors=[r for r in finite if not r['within_5cm']]
    obs=[r for r in finite if r.get('support_class')=='returned_near']
    return dict(rows=len(rows),true_censoring=dict(Counter(r['true_class'] for r in rows)),
        predicted_censoring=dict(Counter(r['predicted_class'] for r in rows)),
        interior_true=len(finite),within_5cm=sum(r['within_5cm'] for r in finite),
        errors=len(errors),errors_bracket_incompatible=sum(r['bracket_incompatible'] for r in errors),
        errors_bracket_compatible=sum(not r['bracket_incompatible'] for r in errors),
        bracket_width_m=stats([r['label_upper']-r['label_lower'] for r in finite]),
        bracket_minimax_half_width_m=stats([.5*(r['label_upper']-r['label_lower']) for r in finite]),
        bracket_unbounded=sum(not np.isfinite(r['label_upper']) for r in finite),
        bracket_at_most_10cm=sum(r['label_upper']-r['label_lower']<=.100001 for r in finite),
        prediction_outside_bracket_all=sum(r['bracket_incompatible'] for r in rows),
        error_floor_to_bracket_m=stats([r['bracket_distance'] for r in finite]),
        signed_error_m=stats([r['predicted']-r['truth'] for r in finite]),
        support_classes=dict(Counter(r.get('support_class','not_audited') for r in finite)),
        error_support_classes=dict(Counter(r.get('support_class','not_audited') for r in errors)),
        returned_near_zone_span_m=stats([r['native_info']['zone_span'] for r in obs if r['native_info']['zone_span'] is not None]),
        returned_near_ray_span_m=stats([r['native_info']['ray_span'] for r in obs if r['native_info']['ray_span'] is not None]),
        returned_near_ray_span_at_most_10cm=sum(r['native_info']['ray_span'] is not None and r['native_info']['ray_span']<=.100001 for r in obs),
        first_boundary_near={k:sum(np.isfinite(r['native_info'][k]) and abs(r['native_info'][k]-r['truth'])<=.050001
                              for r in finite if 'native_info' in r) for k in ('native','lattice','returned')},
        returned_near_horizon_tail_over_3m=sum(r['kind']=='horizon' and r['native_info']['ray_extent'] is not None
                                              and r['native_info']['ray_extent'][1]>3 for r in obs),
        returned_near_point_interval_exclusions=sum(r['native_info']['actual_point_in_public_interval'] is False for r in obs))


def feature_audit(features,tof,rgb,identities,boundaries):
    expected=tof.reshape(len(tof),-1)
    assert np.array_equal(features[:,-384:],expected),'Canonical ToF suffix differs'
    out={}
    for name,values in [('rgb',rgb),('tof',tof),('features',features)]:
        groups=defaultdict(list)
        for i,row in enumerate(values):groups[hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()].append(i)
        collisions=[v for v in groups.values() if len(v)>1]
        discrepant=0
        for group in collisions:
            b=boundaries[group]
            for a in range(1,len(b)):
                finite=np.isfinite(b[0])&np.isfinite(b[a])
                if np.any(np.isfinite(b[0])!=np.isfinite(b[a])) or np.any(np.abs(b[0][finite]-b[a][finite])>.1):
                    discrepant+=1
        out[name]=dict(duplicate_groups=len(collisions),duplicate_rows=sum(map(len,collisions)),
                       first_vs_other_boundary_disagreements=discrepant)
    normal=read_npz(OLD/'normalization.npz');std=normal['std']
    pairs=defaultdict(dict)
    for m in identities:pairs[(m['base_group_id'],m['frame_in_clip'])][m['layout_relation']]=m['index']
    distances={name:[] for name in ('visual','sensor','combined')};exact=0;count=0
    for group in pairs.values():
        for rel in [('INSIDE','BOUNDARY'),('BOUNDARY','OUTSIDE'),('INSIDE','OUTSIDE')]:
            a,b=(group[r] for r in rel);delta=(features[a]-features[b])/std;count+=1
            exact+=int(np.array_equal(features[a],features[b]))
            for name,part in [('visual',delta[:-384]),('sensor',delta[-384:]),('combined',delta)]:
                distances[name].append(float(np.sqrt(np.mean(part.astype(float)**2))))
    out['matched_lateral_pairs']=dict(n=count,exact_feature_collisions=exact,
        train_standardized_rms={k:stats(v) for k,v in distances.items()})
    return out


def read_npz(path):
    with np.load(path,allow_pickle=False) as p:return {k:p[k] for k in p.files}


def run():
    journal=os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal and read(journal)['state']=='running','Use tools/ba.ps1 run research-ue'
    new_stage_directory(DEST);started=time.perf_counter()
    files=[OLD/name for name in ('source-seal.json','prediction-seal.json','features.npy','normalization.npz',
        'cohort.json','queries.json','result.json','training-targets.npz','evaluation-targets.npz')]
    files += [OLD/f'{arm}-{split}-predictions.npz' for arm in ARMS for split in SPLITS]
    files += [OLD/f'{arm}-selection.json' for arm in ARMS]
    files += [PREPARED/'observations'/x for x in ('identities.json','rgb.npy','tof.npy')]
    files += [CAPTURE/'evaluator/geometry.json',SOURCE/'plan/spec.json']
    seal={str(p):sha(p) for p in files}
    previous=read(OLD/'source-seal.json')
    for path,digest in previous['input_hashes'].items():
        if path in seal:assert seal[path]==digest
    predseal=read(OLD/'prediction-seal.json')
    for name,digest in predseal['files'].items():assert sha(OLD/name)==digest
    for arm,digest in predseal['selection_hashes'].items():assert sha(OLD/f'{arm}-selection.json')==digest
    assert sha(OLD/'features.npy')==read(OLD/'feature-receipt.json')['features_sha256']
    code={name:sha(HERE/name) for name in ('run_boundary_error.py','boundary_error_math.py',
          'BOUNDARY_ERROR_PROTOCOL_20260923.md','contact_boundary_data.py','tof_fov45_core.py','ba_camera_corridor.py')}
    write(DEST/'input-seal.json',dict(files=seal,code=code))
    identities=read(PREPARED/'observations/identities.json');geometry=read(CAPTURE/'evaluator/geometry.json')
    cases=read(SOURCE/'plan/spec.json')['cases'];q=queries();cohort=read(OLD/'cohort.json')['indices']
    tof=np.load(PREPARED/'observations/tof.npy',mmap_mode='r')
    rgb=np.load(PREPARED/'observations/rgb.npy',mmap_mode='r')
    features=np.load(OLD/'features.npy',mmap_mode='r')
    assert len(identities)==len(geometry)==len(cases)==len(features)==1728
    truth={kind:[] for kind in KINDS};seen=[]
    for i,(m,g,c) in enumerate(zip(identities,geometry,cases)):
        assert m['id']==g['id']==c['name'] and m['index']==g['sample_index']==i
        boxes=camera_boxes(g);seen.append(contact_labels(boxes,q['seen'])[0])
        for kind in KINDS:truth[kind].append(critical_values(boxes,kind))
    truth={k:np.asarray(v) for k,v in truth.items()};seen=np.asarray(seen)
    support={}
    for j,i in enumerate(cohort['evaluation']):
        support[i]=native_diagnostic(geometry[i],cases[i],tof[i],{k:truth[k][i] for k in KINDS})
        if (j+1)%96==0:print('NATIVE',j+1,'/576',flush=True)
    rows=[];summaries={};checks=Counter();signature_summary={}
    old_result=read(OLD/'result.json')
    for split,indices in cohort.items():
        if split in ('train','evaluation'):
            target=read_npz(OLD/('training-targets.npz' if split=='train' else 'evaluation-targets.npz'))
            assert np.array_equal(seen[indices],target['seen']);checks['saved_label_arrays']+=1
            for kind in KINDS:np.testing.assert_array_equal(truth[kind][indices],target[kind+'_boundary'])
        signatures=defaultdict(list)
        for i in indices:signatures[np.packbits(seen[i]).tobytes()].append(i)
        signature_summary[split]={}
        for kind in KINDS:
            wide=0;affected=set();witnesses=[]
            for group in signatures.values():
                for layer in range(2):
                    finite=[i for i in group if np.isfinite(truth[kind][i,layer])]
                    if len(finite)<2:continue
                    lo=min(finite,key=lambda i:truth[kind][i,layer]);hi=max(finite,key=lambda i:truth[kind][i,layer])
                    if truth[kind][hi,layer]-truth[kind][lo,layer]>.100001:
                        wide+=1;affected.update((i,layer) for i in finite)
                        witnesses.append(dict(first=lo,second=hi,layer=layer,diameter_m=float(truth[kind][hi,layer]-truth[kind][lo,layer])))
            signature_summary[split][kind]=dict(full_grid_signature_layer_groups_over_10cm=wide,
                affected_finite_rows=len(affected),affected_interior_rows=sum(axis_for(kind)[0]+1e-6<truth[kind][i,l]<=axis_for(kind)[-1]+1e-6 for i,l in affected),witnesses=witnesses)
        for arm in ARMS:
            pred=read_npz(OLD/f'{arm}-{split}-predictions.npz');threshold=read(OLD/f'{arm}-selection.json')['threshold']
            for kind in KINDS:
                axis=axis_for(kind);curves=pred[kind+'_curve'].reshape(-1,2,len(axis));selected=[]
                assert not (np.diff(curves,axis=-1)<-1e-6).any(),'Sampled probability reversal'
                checks['monotone_curves']+=int(len(curves)*2)
                if split!='dev':
                    expected=old_result['arms'][arm]
                    if split=='train':expected=expected['training_layouts']
                    assert boundary_metrics(pred[kind+'_curve'],threshold,truth[kind][indices],kind)==expected['boundaries'][kind]
                    checks['original_boundary_metrics']+=1
                for local,i in enumerate(indices):
                    for layer in range(2):
                        t=float(truth[kind][i,layer]);lo,hi=label_bracket(q['seen'],seen[i],kind,layer)
                        assert (t>=lo-1e-6 and t<=hi+1e-6),'True boundary outside label bracket'
                        # Independent slice bracket from sorted same-axis truth labels.
                        mask=(q['seen'][:,2]==layer)&np.isclose(q['seen'][:,1 if kind=='width' else 0],3 if kind=='width' else .6)
                        coord=np.round(q['seen'][mask,0 if kind=='width' else 1].astype(float),6);ys=seen[i,mask]
                        low2=max(coord[~ys],default=0. if kind=='width' else .3);high2=min(coord[ys],default=float('inf'))
                        assert abs(low2-lo)<1e-6 and (high2==hi or abs(high2-hi)<1e-6)
                        assert not np.any(ys[:-1]&~ys[1:]);checks['brackets']+=1
                        crossing=curve_crossing(axis,curves[local,layer],threshold);pc=crossing[2]
                        trueclass='left' if t<=axis[0]+1e-6 else 'right' if t>axis[-1]+1e-6 else 'interior'
                        good=trueclass=='interior' and np.isfinite(pc) and abs(pc-t)<=.05+1e-6
                        if trueclass=='interior':
                            oracle=axis[np.flatnonzero(axis>=t-1e-7)[0]]
                            assert abs(oracle-t)<=.05+1e-6;checks['oracle_resolution']+=1
                        distance=max(lo-pc,pc-hi,0.) if np.isfinite(pc) else float('inf') if np.isfinite(hi) else 0.
                        r=dict(index=i,id=identities[i]['id'],split=split,arm=arm,kind=kind,layer=layer,
                            type_id=identities[i]['type_id'],truth=t,true_class=trueclass,predicted=pc,
                            predicted_class=crossing[3],within_5cm=bool(good),label_lower=lo,label_upper=hi,
                            bracket_incompatible=disjoint_crossing((lo,hi),crossing),bracket_distance=distance)
                        if i in support:
                            s=support[i][kind,layer];r['native_info']=s
                            state='none_near'
                            for key,name in [('near_native','native_only_near'),('near_lattice','lattice_only_near'),('near_returned','returned_near')]:
                                if np.isfinite(t) and np.isfinite(s[key]) and abs(s[key]-t)<=.05+1e-6:state=name
                            r['support_class']=state
                        rows.append(r);selected.append(r)
                summaries[f'{split}/{arm}/{kind}']=summarize(selected)
    feature=feature_audit(features,tof,rgb,identities,np.stack([truth[k] for k in KINDS],1))
    for path,digest in seal.items():assert sha(path)==digest,path
    for name,digest in code.items():assert sha(HERE/name)==digest,name
    write(DEST/'rows.json',nullable(rows))
    result=dict(status='PASS',decision='DIAGNOSTIC_ONLY_NO_MODEL_PROMOTION',summaries=summaries,
        signatures=signature_summary,features=feature,checks=dict(checks),
        native_parity_frames=len(support),backend='TASK_NOT_GPU_SUITABLE',elapsed_s=time.perf_counter()-started,
        rows_sha256=sha(DEST/'rows.json'),inputs_unchanged=True,
        limits='Consumed simulation; label brackets do not include image information; native spans are privileged per-return proxies; no causal feature or information ceiling claim')
    write(DEST/'result.json',nullable(result))
    print('COMPLETE',result['decision'],dict(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['spec','run']);args=p.parse_args()
    spec() if args.command=='spec' else run()

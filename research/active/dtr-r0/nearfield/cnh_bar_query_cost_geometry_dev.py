"""Evaluator-only exact OBB/SAT geometry association, consumed cache only."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path('E:/linnan/linnan');WORK=ROOT/'artifacts.local/work'
OUT=WORK/'cnh-bar-query-cost-dev-20261009/review';POSE=WORK/'cnh-bar-transfer-dev-20261009/pose'
SHAPE=WORK/'cnh-aligned-shapes-dev-20261008';LOW=np.array([-.3,-.2,.3]);HIGH=np.array([.3,.42,3.])
BRANCHES=[f'query_{degree}_{sign:+d}' for degree in (1,2,3) for sign in (-1,1)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def obb(box,h):
    lo=np.array(box['lo']);hi=np.array(box['hi']);return h[:3,:3]@((lo+hi)/2)+h[:3,3],(hi-lo)/2,h[:3,:3]

def sat(data,lo,hi):
    center,radius,rotation=data;center2=(lo+hi)/2;radius2=(hi-lo)/2
    axes=[*np.eye(3),*rotation.T]
    for a in np.eye(3):
        for b in rotation.T:
            cross=np.cross(a,b);norm=np.linalg.norm(cross)
            if norm>1e-10:axes.append(cross/norm)
    axes=np.array(axes);margins=np.abs(axes@rotation)@radius+np.abs(axes)@radius2-np.abs(axes@(center-center2))
    margin=float(margins.min());return margin>1e-12,margin

def clipped_polygon(data,lo,hi):
    """Independent yaw-only polygon clipping intersection area plus open y."""
    c,r,R=data
    if min(c[1]+r[1],hi[1])-max(c[1]-r[1],lo[1])<=1e-12:return False
    p=np.array([c+R@np.array([x*r[0],0,z*r[2]]) for x,z in ((-1,-1),(1,-1),(1,1),(-1,1))])[:,[0,2]]
    for axis,bound,sign in ((0,lo[0],1),(0,hi[0],-1),(1,lo[2],1),(1,hi[2],-1)):
        result=[]
        for i,a in enumerate(p):
            b=p[(i+1)%len(p)];ina=sign*(a[axis]-bound)>=0;inb=sign*(b[axis]-bound)>=0
            if ina:result.append(a)
            if ina!=inb:result.append(a+(b-a)*(bound-a[axis])/(b[axis]-a[axis]))
        p=np.array(result)
        if len(p)<3:return False
    area=abs(float(np.dot(p[:,0],np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1))))/2
    return area>1e-12

def bounds(q,expanded=False):
    lo=np.array([-.3,-.2 if q==0 else .42,.3])+1e-8;hi=np.array([.3,.42 if q==0 else .9,3.])-1e-8
    if expanded:lo[0]=-.4;hi[0]=.4
    return lo,hi

def prepare():
    start=time.monotonic();inputs=[SHAPE/'PLAN.json',SHAPE/'geometry.npz',POSE/'PLAN.json',*[POSE/(n+'.npz') for n in BRANCHES]]
    plan=dict(task='CNH_BAR_QUERY_COST_GEOMETRY_REVIEW_DEV_20261009',budget_CPU_wall_seconds=120,
        scope='Read cached six query error branches, strict-clear original88 scenes, all13 f3..15 and both queries. No inference/new draws/training. Geometry is evaluator-only and never feeds score, gate, threshold or candidate.',
        method='World target/background AABB become OBB via estimated_query[f] @ inv(true_sensor[f]); full separating-axis test versus fixed public query volume. Contact original inward1e-8 convention; expanded pass xclosed±.40 and y/z inward1e-8. Independent yaw polygon clipping verifies overlap booleans.',
        argmax='Local saved raw patch centers in query coordinates. Fixed .30×.10×.30m patch clipped by public query y bounds; geometric overlap only, not readout cause or localization proof. Raw maximum differs from smoothed score history.',
        decision='Test whether originalclear targets/sidefaces enter estimatedquery rather than assume it from rising alarms. Costs remain conditioned on unchanged original clear labels. Main agent alone computes score/slot attribution.',
        stop='Finish declared cached diagnosis or cumulative120s; preserve failures, no angle/scene/threshold selection.',
        inputs_sha256={str(p.relative_to(ROOT)).replace(chr(92),'/'):sha(p) for p in inputs},source_sha256=sha(__file__))
    (OUT/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf8');(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'prepare_result.json').write_text(json.dumps(dict(seconds=time.monotonic()-start)),encoding='utf8')

def run():
    start=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text());prior=json.loads((OUT/'prepare_result.json').read_text())['seconds']
    rows=json.loads((SHAPE/'PLAN.json').read_text())['scene_rows']
    with np.load(SHAPE/'geometry.npz') as d:sensor=d['sensor'];query=d['public_query'];category=d['category']
    ids=np.flatnonzero((category=='clear').all(1));assert len(ids)==88
    geometric=[];argmax_rows=[];summaries=[];max_y_error=0.;checks=0
    # Solid/surface parity was independently established for this finite cohort;
    # do not generalize solid intersections to enclosing-object surface labels.
    for name in BRANCHES:
        with np.load(POSE/(name+'.npz')) as d:estimate=d['estimated_query'];centers=d['argmax_xyz']
        for f in range(3,16):
            h=estimate[f]@np.linalg.inv(sensor[f]);assert np.allclose(h[:3,1],[0,1,0],atol=1e-12,rtol=0)
            max_y_error=max(max_y_error,float(np.max(np.abs(h[1]-np.array([0,1,0,0])))))
            for i in ids:
                boxes=[rows[i],*rows[i]['background']];obbs=[obb(b,h) for b in boxes]
                for q in (0,1):
                    lo,hi=bounds(q);elo,ehi=bounds(q,True);target,tm=sat(obbs[0],lo,hi);expanded,em=sat(obbs[0],elo,ehi)
                    background=[sat(b,lo,hi) for b in obbs[1:]];ebg=[sat(b,elo,ehi) for b in obbs[1:]]
                    for b in obbs:
                        assert sat(b,lo,hi)[0]==clipped_polygon(b,lo,hi);assert sat(b,elo,ehi)[0]==clipped_polygon(b,elo,ehi);checks+=2
                    geometric.append(dict(branch=name,scene=int(i),frame=f,query=('HEAD','BODY')[q],family=rows[i]['family'],placement=rows[i]['placement'],side=rows[i]['side'],context=rows[i]['context'],
                        target_contact=int(target),target_expanded=int(expanded),background_contact=int(any(b[0] for b in background)),background_expanded=int(any(b[0] for b in ebg)),
                        target_SAT_overlap_margin_m=tm,target_expanded_SAT_overlap_margin_m=em,background_contact_box_indices=';'.join(str(j+1) for j,b in enumerate(background) if b[0]),
                        background_expanded_box_indices=';'.join(str(j+1) for j,b in enumerate(ebg) if b[0])))
                    for k in range(4):
                        center=centers[i,k,f-3,q];plo=center-np.array([.15,.05,.15]);phi=center+np.array([.15,.05,.15]);plo=np.maximum(plo,lo);phi=np.minimum(phi,hi)
                        assert (phi>plo).all()
                        hits=[sat(b,plo,phi)[0] for b in obbs]
                        argmax_rows.append(dict(branch=name,scene=int(i),replica=k,frame=f,query=('HEAD','BODY')[q],patch_x=float(center[0]),patch_y=float(center[1]),patch_z=float(center[2]),
                            raw_argmax_patch_target_overlap=int(hits[0]),raw_argmax_patch_background_overlap=int(any(hits[1:])),raw_argmax_patch_background_box_indices=';'.join(str(j) for j in range(1,len(hits)) if hits[j])))
                if prior+time.monotonic()-start>120:raise TimeoutError('cumulative120s')
        gr=[r for r in geometric if r['branch']==name]
        summaries.append(dict(branch=name,scene_query_frame_rows=len(gr),target_contact_rows=sum(r['target_contact'] for r in gr),target_expanded_rows=sum(r['target_expanded'] for r in gr),
            target_contact_scenes=sorted({r['scene'] for r in gr if r['target_contact']}),background_contact_rows=sum(r['background_contact'] for r in gr),background_expanded_rows=sum(r['background_expanded'] for r in gr)))
    for filename,data in (('geometry.csv',geometric),('argmax_geometry.csv',argmax_rows)):
        with (OUT/filename).open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    result=dict(status='PASS',seconds=time.monotonic()-start,prior_prepare_seconds=prior,cumulative_seconds=prior+time.monotonic()-start,strictclear_scene_ids=ids.tolist(),geometry_rows=len(geometric),argmax_rows=len(argmax_rows),polygon_crosschecks=checks,max_y_invariance_error=max_y_error,summary=summaries,
        limitations='Changing Q rotates both spatial evidence and query semantics/coverage. OBB overlap can show semantic inclusion but cannot isolate causal signal from coverage, normalization, voxel quadrature/FP16 or nonlinear M3 response. No photon/object attribution replay. Raw local argmax patch overlap is association; smoothing mixes earlier maxima and target vsbackground share bins.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result['summary']),flush=True);print('CPU seconds',result['cumulative_seconds'],flush=True)

if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()

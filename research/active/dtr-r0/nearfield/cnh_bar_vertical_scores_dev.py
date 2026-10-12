"""Replay fixed horizontal patch scores; evaluator-only vertical support split."""
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S
import cnh_bar_local_readout_dev as L

OUT=B.ROOT/'artifacts.local/work/cnh-bar-fusion-vertical-dev-20261009/vertical'


def prepare():
    # Scene family selects this descriptive evaluation subset; does not enter scoring.
    old=json.loads((S.OUT/'PLAN.json').read_text());ids=[i for i,r in enumerate(old['scene_rows']) if r['family']=='vertical']
    A,_,_,_=B.imports()
    sources=[Path(__file__),Path(L.__file__),Path(S.__file__),Path(B.__file__),
             *[Path(m.__file__).resolve() for n,m in list(sys.modules.items())
               if n.startswith('cnh_') and getattr(m,'__file__',None)]]
    inputs=[S.OUT/'PLAN.json',S.OUT/'geometry.npz',S.OUT/'physical.npz',S.OUT/'evaluated.npz',
            L.OUT/'cached_scores.npz',L.OUT/'result_analysis.json']
    B.save(OUT/'PLAN.json',dict(task='CNH_BAR_FUSION_VERTICAL_DEV_20261009_VERTICAL',lane='EXPLORE consumed Development',
        authorization='User 推进 scoped fusion and vertical three-score diagnostics',base_commit='daed724c',
        goal='Provide vertical support/competition/threshold clues without shape-gating observations or causal attribution',
        budgets_wall_seconds=dict(projection_and_scores=600,analysis_and_audit=180),scene_ids=ids,replicas=4,
        adjustable_scope='Implementation repairs only; original6x1x3 patch shape, original photons/public poses and local theta frozen; no M3 inference/new samples/training',
        scores='After producing ALL original horizontal patch scores for each selected scene/window/query, evaluator computes target-overlap max, global max, non-target-overlap max. Missing support is NOT_EVALUABLE, not negative.',
        support='world_to_query=public_query[f]@inv(sensor[f]); transform box corners. Current retained aligned case rotation is identity. Target support: patch contains at least one voxel whose query-clipped volume has positive overlap with target AABB. All3 overlap lengths >1e-10m excludes numerical touching. Every patch remains same public query weighting.',
        decision_check='Vertical original corresponding-height timely HEAD217/BODY177 out224each; local207/168. Existing losses15/22. Inspect original event ledger loss/retention/gain, margin vs common local theta, restricted score vs competing score; no argmax causal gate or information loss claim. Individual observation/window and K4 are related.',
        stop='Complete fixed vertical subset within cumulative caps, preserve partial/failures; no patch sweep, alternate support tuning or new shapes',
        deliverables='All patch arrays, raw and separately causal-smoothed three-score tables, original-global parity, loss-group summaries and geometry/partition fixtures',
        inputs_sha256={B.logical_path(p):B.sha(p) for p in inputs},
        source_sha256={B.logical_path(p):B.sha(p) for p in sorted(set(sources),key=str)}))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('PREPARED vertical scenes',len(ids),flush=True)


def run():
    began=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text());projector=None
    previous=sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('failure_*.json'))
    def check():
        if previous+time.monotonic()-began>600:raise TimeoutError('cumulative600s')
    try:
        for name,digest in {**plan['inputs_sha256'],**plan['source_sha256']}.items():assert B.sha(B.ROOT/name)==digest, name
        A,_,_,normal=B.imports();A.OUT=OUT/'runtime';A.setup_gpu()
        import torch
        from cnh_cvr_v2_materialize import BatchedProjector
        torch.set_num_threads(2);projector=BatchedProjector();engine=SimpleNamespace(torch=torch,projector=projector)
        qs,locations=L.patches();ids=plan['scene_ids']
        with np.load(S.OUT/'physical.npz') as d:hist=d['hist'][ids];ambient=d['ambient']
        with np.load(S.OUT/'geometry.npz') as d:sensor=d['sensor'];query=d['public_query']
        z=normal(hist.reshape(-1,16,8,8,16),np.broadcast_to(ambient,(len(ids)*4,*ambient.shape)))
        patch_values=[np.empty((len(ids)*4,13,q.shape[0]),np.float64) for q in qs]
        norms=[];checks=[]
        for j,f in enumerate(B.FRAMES):
            check();start=max(0,int(f)-7);m=(query[f]@np.linalg.inv(sensor[f]))[None]@sensor[start:f+1]
            cp=L.Projection(engine,m);nn=[]
            for q in qs:
                qp=q@cp.p;v=np.asarray(qp.multiply(qp).sum(1)).ravel();nn.append(np.where(v>0,np.sqrt(v),np.inf))
            norms.append(nn)
            for b in range(0,len(z),32):
                check();feature=cp(z[b:b+32,start:f+1]).cpu().numpy().astype(np.float16)
                if b==0 and f in (3,13):
                    ref=A.Engine.project(engine,z[b:b+1,start:f+1],m[None]).cpu().numpy().astype(np.float16)
                    np.testing.assert_array_equal(feature[:1],ref);checks.append(dict(frame=int(f),FP16_features_max_abs=0))
                total=feature[:,0].astype(np.float64).reshape(len(feature),-1)
                for q,(qq,scale) in enumerate(zip(qs,nn)):
                    val=np.asarray(qq@total.T).T/scale[None];val[:,~np.isfinite(scale)]=-np.inf
                    patch_values[q][b:b+32,j]=val
            print('VERTICAL FRAME',int(f),'/',15,'seconds',round(time.monotonic()-began,3),flush=True)
        raw=np.stack([v.max(-1) for v in patch_values],-1).reshape(len(ids),4,13,2)
        with np.load(L.OUT/'cached_scores.npz') as d:ref=d['raw'][ids]
        np.testing.assert_array_equal(raw,ref)
        for q,v in enumerate(patch_values):np.save(OUT/f'patch_scores_q{q}.npy',v.reshape(len(ids),4,13,-1))
        np.savez_compressed(OUT/'patch_geometry.npz',scene_ids=ids,locations_q0=locations[0],locations_q1=locations[1],norms_q0=np.array([n[0] for n in norms]),norms_q1=np.array([n[1] for n in norms]))
        B.save(OUT/'result_run.json',dict(status='COMPLETE',seconds=time.monotonic()-began,backend='Original CUDA FP32 projection only; CPU patch sums',device=torch.cuda.get_device_name(),projection_parity=checks,global_original_parity_max_abs=0,original_hist_samples=len(ids)*4,model_inference=0,new_samples=0,training=0))
    except BaseException as exc:
        B.save(OUT/f'failure_{time.time_ns()}.json',dict(seconds=time.monotonic()-began,error=repr(exc)));raise
    finally:
        if projector is not None:projector=None;torch.cuda.empty_cache()


def support_masks(row,sensor,query,qs):
    from cnh_cvr_projection import grid,STEP
    centers,_=grid();vlo=(centers-STEP/2).reshape(-1,3);vhi=(centers+STEP/2).reshape(-1,3)
    masks=[];boxes=[]
    corners=np.array(np.meshgrid(*zip(row['lo'],row['hi']),indexing='ij')).reshape(3,-1).T
    for f in B.FRAMES:
        h=query[f]@np.linalg.inv(sensor[f]);np.testing.assert_allclose(h[:3,:3],np.eye(3),atol=1e-14)
        box=corners@h[:3,:3].T+h[:3,3];lo=box.min(0);hi=box.max(0)
        np.testing.assert_allclose(lo,np.asarray(row['lo'])-[0,0,sensor[f,2,3]],atol=1e-14)
        boxes.append([lo,hi]);mm=[]
        for q,(yl,yh) in enumerate(((-.2,.42),(.42,.9))):
            lower=np.maximum(np.maximum(vlo,lo),[-.3,yl,.3]);upper=np.minimum(np.minimum(vhi,hi),[.3,yh,3.])
            overlap=((upper-lower)>1e-10).all(1)
            mm.append(np.asarray(qs[q]@overlap.astype(float)).ravel()>0)
        masks.append(mm)
    return masks,boxes


def analyze():
    began=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text());ids=plan['scene_ids']
    rows=json.loads((S.OUT/'PLAN.json').read_text())['scene_rows'];qs,_=L.patches()
    with np.load(S.OUT/'geometry.npz') as d:sensor=d['sensor'];query=d['public_query'];cat=d['category']
    with np.load(S.OUT/'evaluated.npz') as d:baseline=d['scores'][ids]
    with np.load(L.OUT/'cached_scores.npz') as d:global_raw=d['raw'][ids]
    theta=json.loads((L.OUT/'result_analysis.json').read_text())['candidate']['threshold']
    global_smoothed=B.smooth(global_raw);table=[];events=[];fixture=[]
    vals=[np.load(OUT/f'patch_scores_q{q}.npy',mmap_mode='r') for q in range(2)]
    for n,i in enumerate(ids):
        support,boxes=support_masks(rows[i],sensor,query,qs)
        for q in range(2):
            event_by_k=[[] for _ in range(4)]
            for j,f in enumerate(B.FRAMES):
                ss=support[j][q];observed=np.isfinite(vals[q][n,:,j]).all(0)
                assert observed.any();assert np.array_equal(ss|~ss,np.ones_like(ss))
                for k in range(4):
                    v=vals[q][n,k,j];valid_support=ss&np.isfinite(v);comp=~ss&np.isfinite(v)
                    target=float(v[valid_support].max()) if valid_support.any() else None
                    other=float(v[comp].max()) if comp.any() else None
                    total=float(v.max());partition=max([x for x in (target,other) if x is not None])
                    assert total==partition
                    record=dict(scene=i,replica=k,height=['HEAD','BODY'][q],category=str(cat[i,q]),frame=int(f),
                        target_support_patches=int(valid_support.sum()),competitor_patches=int(comp.sum()),
                        target_raw=target,global_raw=total,competitor_raw=other,
                        actual_local_smoothed=float(global_smoothed[n,k,j,q]),local_theta=theta,M3_smoothed=float(baseline[n,k,j,q]),
                        target_state='EVALUABLE' if target is not None else 'NO_TARGET_QUERY_SUPPORT',
                        target_front=float(boxes[j][0][2]))
                    event_by_k[k].append(record)
            for rr in event_by_k:
                series=np.array([[r['target_raw'] if r['target_raw'] is not None else np.nan,r['competitor_raw'] if r['competitor_raw'] is not None else np.nan] for r in rr])
                smoothed=B.smooth(series)
                for j,r in enumerate(rr):
                    r['target_smoothed']=float(smoothed[j,0]) if np.isfinite(smoothed[j,0]) else None
                    r['competitor_smoothed']=float(smoothed[j,1]) if np.isfinite(smoothed[j,1]) else None
                    r['target_smoothing_complete']=int(np.isfinite(smoothed[j,0]))
                    for v in smoothed[j]:
                        if np.isfinite(v):assert r['actual_local_smoothed']>=v-1e-12
                    table.append(r)
            if cat[i,q]=='contact':
                for k,rr in enumerate(event_by_k):
                    rr=rr[:11];a=any(r['M3_smoothed']>=B.THETA for r in rr);b=any(r['actual_local_smoothed']>=theta for r in rr)
                    target=[r['target_smoothed'] for r in rr if r['target_smoothed'] is not None]
                    comp=[r['competitor_smoothed'] for r in rr if r['competitor_smoothed'] is not None]
                    events.append(dict(scene=i,replica=k,height=['HEAD','BODY'][q],M3_timely=int(a),local_timely=int(b),
                        group='lost' if a and not b else 'retained' if a and b else 'gained' if b else 'missed_both',
                        target_support_windows=sum(r['target_smoothed'] is not None for r in rr),
                        target_peak=max(target) if target else None,competitor_peak=max(comp) if comp else None,
                        local_policy_peak=max(r['actual_local_smoothed'] for r in rr),M3_peak=max(r['M3_smoothed'] for r in rr),theta=theta))
        if n in (0,len(ids)-1):fixture.append(dict(scene=i,frame13_target_front=float(boxes[10][0][2])))
        if time.monotonic()-began>180:raise TimeoutError('analysis180s')
    def write_csv(name,records):
        with (OUT/name).open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write_csv('three_scores.csv',table);write_csv('events.csv',events)
    summary=[]
    for height in ('HEAD','BODY'):
        for group in ('lost','retained','gained','missed_both'):
            rs=[r for r in events if r['height']==height and r['group']==group]
            def stats(field):
                v=[r[field] for r in rs if r[field] is not None]
                return dict(n=len(v),median=float(np.median(v))) if v else dict(n=0,median=None)
            summary.append(dict(height=height,group=group,n=len(rs),target_peak=stats('target_peak'),competitor_peak=stats('competitor_peak'),local_policy_peak=stats('local_policy_peak'),M3_peak=stats('M3_peak'),target_peak_above_theta=sum(r['target_peak'] is not None and r['target_peak']>=theta for r in rs),competitor_peak_above_theta=sum(r['competitor_peak'] is not None and r['competitor_peak']>=theta for r in rs)))
    B.save(OUT/'result_analysis.json',dict(status='COMPLETE',seconds=time.monotonic()-began,windows=len(table),contact_events=len(events),summary=summary,geometry_fixture=fixture,partition_checks=len(table),
        limitations='Oracle geometric overlap does not establish signal provenance/causality. smooth(global max) need not equal max(smooth restricted maxima); no reweighting incomplete windows. Windows share photons; restricted max selection still correlated. No score-driven support tuning.'))
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()

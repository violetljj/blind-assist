"""Fixed yaw-pose error probes, same photons/evaluator and shared M3/local features."""
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S
import cnh_bar_local_readout_dev as L
import cnh_bar_fusion_probe_dev as F

BASE=B.ROOT/'artifacts.local/work/cnh-bar-transfer-dev-20261009'
OUT=BASE/'pose'
M3_THETA=.8557642486787612
FUSION_M3=.9404184587540165
FUSION_LOCAL=4.625390338985158


def rotation(degrees):
    a=np.deg2rad(degrees);c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def jitter():
    x=np.random.default_rng(2026100951).normal(size=16);x-=x.mean()
    return x/np.sqrt(np.mean(x*x))


def variants():
    result=[dict(name='ideal',kind='ideal',degrees=0,sign=0)]
    for kind in ('query','extrinsic','jitter'):
        for a in (1,2,3):
            for sign in (-1,1):result.append(dict(name=f'{kind}_{a}_{sign:+d}',kind=kind,degrees=a,sign=sign))
    return result


def injected(sensor,query,v):
    estimated=sensor.copy();goal=query.copy();a=v['degrees']*v['sign']
    if v['kind']=='query':
        rr=np.eye(4);rr[:3,:3]=rotation(a);goal=rr[None]@goal
    elif v['kind']=='extrinsic':estimated[:,:3,:3]=estimated[:,:3,:3]@rotation(a)
    elif v['kind']=='jitter':
        estimated[:,:3,:3]=np.stack([sensor[t,:3,:3]@rotation(a*jitter()[t]) for t in range(16)])
    return estimated,goal


def injection_check(sensor,query):
    records=[];points=np.array([[-.25,.10,.65],[.18,.70,.65],[.10,.30,1.2]])
    jp=jitter();summaries=[]
    for v in variants()[1:]:
        est,q=injected(sensor,query,v);a=v['degrees']*v['sign'];errors=[];moves=[];voxel_changes=0;earlier=[];last=[]
        for f in B.FRAMES:
            for t in range(max(0,int(f)-7),int(f)+1):
                m=q[f]@np.linalg.inv(est[f])@est[t]
                native=(points-sensor[t,:3,3])@sensor[t,:3,:3]
                actual=native@m[:3,:3].T+m[:3,3]
                h=query[f]@np.linalg.inv(sensor[f]);truth=points@h[:3,:3].T+h[:3,3]
                b=sensor[f,:3,:3].T@(sensor[t,:3,3]-sensor[f,:3,3])
                if v['kind']=='query':expected=truth@rotation(a).T
                elif v['kind']=='extrinsic':expected=truth+(query[f,:3,:3]@(rotation(-a)@b-b))[None]
                else:
                    delta=native@(rotation(a*(jp[t]-jp[f]))-np.eye(3)).T+(rotation(-a*jp[f])@b-b)[None]
                    expected=truth+delta@query[f,:3,:3].T
                error=np.max(np.abs(actual-expected));errors.append(error)
                displacement=np.linalg.norm(actual-truth,axis=1);moves.extend(displacement)
                (last if t==f else earlier).extend(displacement)
                oldidx=np.floor((truth-[-.6,-.5,0])/[.05,.1,.1]).astype(int)
                newidx=np.floor((actual-[-.6,-.5,0])/[.05,.1,.1]).astype(int)
                changed=(oldidx!=newidx).any(1);voxel_changes+=int(changed.sum())
                for k in range(3):records.append(dict(branch=v['name'],frame=int(f),history_frame=t,point=k,
                    expected_x=float(expected[k,0]),expected_y=float(expected[k,1]),expected_z=float(expected[k,2]),
                    actual_x=float(actual[k,0]),actual_y=float(actual[k,1]),actual_z=float(actual[k,2]),
                    displacement_m=float(displacement[k]),voxel_index_changed=int(changed[k]),analytic_error=float(error)))
        assert max(errors)<1e-12 and max(earlier)>1e-4,(v,max(errors),max(earlier))
        # Sensor-pose errors cancel at the last exposure by construction; history must change.
        if v['kind']!='query':assert max(last)<1e-12
        summaries.append(dict(branch=v['name'],PASS=True,max_analytic_error=float(max(errors)),
            min_displacement=float(min(moves)),max_displacement=float(max(moves)),last_max=float(max(last)),
            history_max=float(max(earlier)),changed_voxel_points=voxel_changes,points=len(moves)))
    with (OUT/'injection_points.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    B.save(OUT/'injection_check.json',dict(status='PASS',checks=summaries,jitter_normalized_sequence=jp.tolist(),
        convention='Yaw Y only. Query: Qprime=Ry(a) Q. Sensor errors: Sprime_t=S_t Ry(a_t), Q held true. Local hist stays in true sensor coordinates.',
        analytic='Query rotates current-query point; extrinsic residual=Qrot(Ry(-a)b-b), b=Rf.T(pt-pf). Jitter residual=Qrot[(Ry(a_t-a_f)-I)v_t+(Ry(-a_f)-I)b]. Independent vector formulas checked against homogeneous chain.',
        interpretation='Last-frame sensor cancellation is expected; past-frame translation/relative rotations verified nonzero. Same voxel index need not mean zero metric perturbation. One prescribed jitter trajectory, not a real pose-error distribution.'))


def prepare():
    began=time.monotonic();A,_,_,_=B.imports()
    sources=[Path(__file__),Path(L.__file__),Path(F.__file__),
             *[Path(m.__file__).resolve() for n,m in list(sys.modules.items())
               if n.startswith('cnh_') and getattr(m,'__file__',None)]]
    inputs=[S.OUT/'PLAN.json',S.OUT/'physical.npz',S.OUT/'geometry.npz',S.OUT/'evaluated.npz',L.OUT/'cached_scores.npz',
            *A.M3_MODELS]
    B.save(OUT/'PLAN.json',dict(task='CNH_BAR_POSE_TRANSFER_DEV_20261009',lane='EXPLORE consumed photons, unchanged physical truth',
        authorization='User 推进 crosscal/pose/background checks; same input fairness and analytic injection check required',base_commit='839177bd',
        goal='Compare fixed k5 fusion with M3 under identical estimated-pose input errors and quantify absolute retention/cost',
        budgets_wall_seconds=dict(prepare_and_injection=180,run=1800,analysis_and_audit=180),
        branches=variants(),degrees=[1,2,3],signs=[-1,1],axis='yaw Y only for all3 classes',
        jitter_seed=2026100951,jitter='One zero-mean16-frame normal draw normalized RMS1, scaled to1/2/3deg RMS and mirrored sign, shared all scenes/replicas. Peak degrees recorded. Not measured pose noise or a universal2deg gate.',
        injection='Query left multiplyRy. Constant/variable sensor rotation right multiplyRy, translations fixed; physical sensor/photons/evaluator unchanged. Every branch uses the SAME shared projected FP16 feature for M3 and local; local noise norm rebuilt from that projected map.',
        primary='All492 scenes K4, full13f3..15, timelyf3..13. 19branches x25584=486096 5-model ensemble examples including ideal replay.',
        thresholds=dict(M3=M3_THETA,fusion_M3=FUSION_M3,fusion_local=FUSION_LOCAL),
        calibration='None in this pose stage; thresholds all fixed. Report actual costs, not automatically matched cost.',
        adjustable_scope='Implementation repairs/focused checks only; no error/axis/trajectory/threshold/patch/model tuning or new photons/training',
        decision_check='Baseline H/B688 events, dark56; paired same-error fusion−M3 primary plus each system−its ideal scores for absolute sensitivity. Any increased costs or ideal event losses retained. No causality or safety/pose distribution robustness conclusion.',
        stop='Injection analytic check must pass every nonzero branch before projection; cap or fixed queue completion. Preserve partial/failures, no extra jitter trajectories or axes.',
        inputs_sha256={B.logical_path(p):B.sha(p) for p in inputs},source_sha256={B.logical_path(p):B.sha(p) for p in sorted(set(sources),key=str)}))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    with np.load(S.OUT/'geometry.npz') as d:injection_check(d['sensor'],d['public_query'])
    assert time.monotonic()-began<180
    B.save(OUT/'prepare_result.json',dict(status='PASS',seconds=time.monotonic()-began))
    print('PREPARED 18nonzero + ideal; analytic injection PASS',flush=True)


def run():
    began=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text());engine=None
    previous=sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('failure_*.json'))
    def check():
        if previous+time.monotonic()-began>1800:raise TimeoutError('cumulative1800s')
    try:
        assert json.loads((OUT/'injection_check.json').read_text())['status']=='PASS'
        for name,digest in {**plan['inputs_sha256'],**plan['source_sha256']}.items():assert B.sha(B.ROOT/name)==digest,name
        A,_,_,normal=B.imports();A.OUT=OUT/'runtime';engine=A.Engine();qs,locations=L.patches()
        with np.load(S.OUT/'physical.npz') as d:hist=d['hist'];ambient=d['ambient'];original=d['raw']
        with np.load(S.OUT/'geometry.npz') as d:sensor=d['sensor'];query=d['public_query']
        z=normal(hist.reshape(-1,16,8,8,16),np.broadcast_to(ambient,(1968,*ambient.shape)));records=[]
        for v in plan['branches']:
            check();est,q=injected(sensor,query,v);m3=np.empty((1968,13,2),np.float32);local=np.empty_like(m3,dtype=np.float64)
            argmax=np.empty((1968,13,2,3));coverage=[];transforms=[]
            for j,f in enumerate(B.FRAMES):
                start=max(0,int(f)-7);m=(q[f]@np.linalg.inv(est[f]))[None]@est[start:f+1];transforms.append(m.tolist())
                cp=L.Projection(engine,m);norms=[]
                for qq in qs:
                    qp=qq@cp.p;var=np.asarray(qp.multiply(qp).sum(1)).ravel();norms.append(np.where(var>0,np.sqrt(var),np.inf))
                coverage.append([int(np.isfinite(n).sum()) for n in norms])
                for b in range(0,len(z),32):
                    check();feat=cp(z[b:b+32,start:f+1]).cpu().numpy().astype(np.float16)
                    m3[b:b+32,j]=engine.predict(feat);local[b:b+32,j],argmax[b:b+32,j]=L.scan(feat,qs,norms,locations)
            m3=m3.reshape(492,4,13,2);local=local.reshape(492,4,13,2)
            if v['kind']=='ideal':
                np.testing.assert_array_equal(m3,original)
                with np.load(L.OUT/'cached_scores.npz') as d:np.testing.assert_array_equal(local,d['raw'])
            payload=OUT/(v['name']+'.npz');np.savez_compressed(payload,m3_raw=m3,local_raw=local,argmax_xyz=argmax.reshape(492,4,13,2,3),estimated_sensor=est,estimated_query=q)
            record=dict(**v,path=payload.name,sha256=B.sha(payload),valid_patches_by_frame=coverage,
                seconds_cumulative=time.monotonic()-began,matrices=transforms)
            B.save(OUT/(v['name']+'.json'),record);records.append(record)
            print('POSE',v['name'],len(records),'/',19,'seconds',round(time.monotonic()-began,2),flush=True)
        B.save(OUT/'result_run.json',dict(status='COMPLETE',seconds=time.monotonic()-began,branches=records,
            backend='CUDA original FP32 projection and frozen5-model ensemble; CPU sparse local scan',device=engine.torch.cuda.get_device_name(),
            ensemble_examples=486096,training=0,new_photons=0,ideal_parity='BITWISE both M3 and local'))
    except BaseException as exc:
        B.save(OUT/f'failure_{time.time_ns()}.json',dict(seconds=time.monotonic()-began,error=repr(exc)));raise
    finally:
        if engine is not None:engine.nets=[];engine.projector=None;engine.torch.cuda.empty_cache()


def analyze():
    began=time.monotonic();rr=json.loads((OUT/'result_run.json').read_text())
    with np.load(S.OUT/'geometry.npz') as d:cat=d['category']
    dark=np.zeros(492,bool);old=json.loads((B.ROOT/'artifacts.local/work/cnh-bar-representation-dev-20261008/fp16/PLAN.json').read_text());dark[old['contact_scene_ids']]=True
    data=[];ideala=None;idealf=None;summary=[];ledger=[]
    for v in rr['branches']:
        with np.load(OUT/v['path']) as d:m3=B.smooth(d['m3_raw']);local=B.smooth(d['local_raw'])
        ma,at=F.metrics(m3>=M3_THETA,cat);flags=(m3>=FUSION_M3)|(local>=FUSION_LOCAL);mf,ft=F.metrics(flags,cat)
        if v['kind']=='ideal':ideala=at;idealf=ft
        paired=F.compare(at,ft,cat);darkpair=F.compare(at[dark],ft[dark],cat[dark])
        summary.append(dict(branch=v['name'],kind=v['kind'],degrees=v['degrees'],sign=v['sign'],M3=ma,fusion=mf,paired=paired,dark=darkpair,
            M3_vs_ideal=F.compare(ideala,at,cat),fusion_vs_ideal=F.compare(idealf,ft,cat)))
        for i,k,q in np.argwhere(np.broadcast_to((cat=='contact')[:,None,:],at.shape)):
            ledger.append(dict(branch=v['name'],scene=int(i),replica=int(k),height=['HEAD','BODY'][q],dark4cm=int(dark[i]),
                M3=int(at[i,k,q]),fusion=int(ft[i,k,q]),gain=int(not at[i,k,q] and ft[i,k,q]),loss=int(at[i,k,q] and not ft[i,k,q]),
                M3_ideal=int(ideala[i,k,q]),fusion_ideal=int(idealf[i,k,q])))
        if time.monotonic()-began>180:raise TimeoutError('analysis180s')
    with (OUT/'event_ledger.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    B.save(OUT/'result_analysis.json',dict(status='COMPLETE',seconds=time.monotonic()-began,summary=summary,event_rows=len(ledger),
        limitation='Yaw-only fixed error probes and one mirrored jitter trace; known photons/geometry. Fixed thresholds mean observed costs can differ. Paired relative improvement is not absolute pose robustness.'))
    for r in summary:print(r['branch'],'M3',r['M3']['counts'],'fusion',r['fusion']['counts'],'clear',r['M3']['clear_slots'],r['fusion']['clear_slots'],flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()

"""Frozen M3 angular envelope; unique physical axes share identical observations."""
from pathlib import Path
import gc
import os
import time
import numpy as np
import cnh_dual_sensor_alarm as A

ROOT=A.ROOT
OUT=ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005'
WORK=OUT/'angular'
OLD=A.OLD
PARENT=A.OUT
FRAMES=A.FRAMES
read,save,sha=A.read,A.save,A.sha

def deadline():
    if time.time()>read(OUT/'PLAN.json')['deadline_unix']-180:
        raise TimeoutError('Preserve delivery reserve within original 90-minute budget')

def setup():
    WORK.mkdir(exist_ok=True)
    for key,name in [('TEMP','tmp'),('TMP','tmp'),('CUPY_CACHE_DIR','cupy-cache')]:
        p=OUT/name;p.mkdir(exist_ok=True);os.environ[key]=str(p)

def angle_sets():
    new=read(OUT/'PLAN.json')['A']['new_psi_deg']
    angles=sorted({float(p+d) for p in new for d in (0,-15,15,-22.5,22.5)}|{-7.5,37.5})
    known={0.,15.,30.,-7.5,37.5}
    return np.array(angles),np.array([a for a in angles if a not in known])

def contract():
    angles,new=angle_sets()
    p=OUT/'ENGINEERING_INPUT_CONTRACT.json'
    if not p.exists():
        save(p,dict(parent_plan_sha256=sha(OUT/'PLAN.json'),frozen_unix=time.time(),
            unique_absolute_yaw_deg=angles.tolist(),new_absolute_yaw_deg=new.tolist(),
            reuse={'0':'parent dual15 LEFT','15':'original single','30':'parent dual15 RIGHT',
                   '-7.5':'parent dual22p5 LEFT','37.5':'parent dual22p5 RIGHT'},
            rationale='All mode0 scenes have original yaw15. Same travel path, physical axis, shared head error, query frame, and photon realization are exactly the same observation. Reuse improves pairing and avoids redundant render; fixed arms, thresholds and units unchanged.',
            seeds='SeedSequence([2026100520,unit,round(absolute_yaw*10)+1000,variant,K]); one realization per unique physical axis',
            schema='angles_deg; raw[Naxis,7,4,13,2]; no truth supplied to frozen inference'))
    return angles,new

def render_one(unit,new):
    import cnh_displacement_ceiling_render as R
    import cnh_location_reference_gpu as G
    from cnh_coverage_policy_render import target_ray_counts
    started=time.monotonic();deadline()
    truth=read(OLD/'truth/evaluation'/f'unit{unit}.json')
    with np.load(OLD/'observations/evaluation'/f'unit{unit}.npz') as d:
        sensor,travel,base,amb=(d[x].copy() for x in ('sensor','travel','noisy','ambient'))
    ex=np.stack([A.extrinsic(float(a)-15) for a in new])
    physical=sensor[None]@ex[:,None]
    noisy=base[None]@ex[:,None,None]
    query=np.linalg.inv(travel)[None]@physical
    np.testing.assert_allclose(noisy@np.linalg.inv(ex)[:,None,None],base[None]+np.zeros_like(noisy),atol=1e-12,rtol=0)
    targets=[b[0] for b in truth['boxes']];background=truth['boxes'][0][1:]
    assert all(b[1:]==background for b in truth['boxes'])
    engine=G.ExpectedRenderer(physical.reshape(-1,4,4),background)
    try:
        endpoints=engine.render(targets,candidate_batch=4,pose_batch=32,deadline_check=deadline)
        rho=np.array([b['rho'] for b in targets])
        means=endpoints[:,0]+rho[:,None,None,None,None]/G.ENDPOINT_RHO*(endpoints[:,1]-endpoints[:,0])
        means=means.reshape(7,len(new),16,8,8,16).transpose(1,0,2,3,4,5)
        visible=target_ray_counts(engine,targets).reshape(7,len(new),16).transpose(1,0,2)
        ambient=engine.ambient.reshape(len(new),16,8,8)
        np.testing.assert_array_equal(ambient,np.repeat(amb[None],len(new),0))
        meta=engine.metadata
    finally:engine.close()
    hist=np.empty((len(new),7,4,16,8,8,16),np.int32)
    for s,angle in enumerate(new):
        for v in range(7):
            for k in range(4):
                seed=int(np.random.SeedSequence([2026100520,unit,int(round(angle*10))+1000,v,k]).generate_state(1)[0])
                hist[s,v,k]=R.sample(means[s,v],ambient[s],seed)[0]
    p=WORK/'observations'/f'unit{unit}.npz';p.parent.mkdir(exist_ok=True)
    np.savez_compressed(p,hist=hist,ambient=ambient,noisy=noisy,query=query,angles_deg=new)
    t=WORK/'templates'/f'unit{unit}.npz';t.parent.mkdir(exist_ok=True)
    np.savez_compressed(t,expected=means,target_ray_counts=visible,physical=physical,angles_deg=new)
    save(WORK/'render_receipts'/f'unit{unit}.json',dict(unit=unit,seconds=time.monotonic()-started,
        plan_sha256=sha(OUT/'PLAN.json'),observation_sha256=sha(p),template_sha256=sha(t),renderer=meta))

def run():
    setup();angles,new=contract();plan=read(OUT/'PLAN.json');units=plan['A']['units']
    import torch
    import cnh_displacement_ceiling as D
    import cnh_margin_confirm as MC
    import cnh_cvr_pilot as CP
    import cnh_temporal_readout_evaluate as T
    from cnh_temporal_readout_data import normalized_z
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import query_masks
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    nets=[];projector=None;started=time.monotonic()
    rows,_=T.rows_for(OLD,'fresh_evaluation');original,_=T.prediction(OLD,'fresh_evaluation','M3')
    def score(z,q,n):
        raw=np.empty((7,4,13,2),np.float32)
        for k in range(4):
            vox=[]
            for f in FRAMES:
                deadline();idx=np.arange(max(0,f-7),f+1)
                matrices=q[f]@np.linalg.inv(n[k,f])@n[k,idx]
                vox.append(D.project_many(projector,z[:,k,idx],matrices).half())
            x=torch.stack(vox,1).reshape(91,3,24,17,33)
            logits=[]
            with torch.inference_mode():
                for begin in range(0,len(x),64):
                    xx=x[begin:begin+64].float().clone()
                    xx[:,0]=xx[:,0].sign()*xx[:,0].abs().log1p()
                    xx[:,2]=xx[:,2].sign()*xx[:,2].abs().log1p();xx[:,1]/=8
                    xx=torch.cat((xx,masks[None].expand(len(xx),-1,-1,-1,-1)),1)
                    logits.append(torch.stack([net(xx) for net in nets]).mean(0).cpu().numpy())
            raw[:,k]=np.concatenate(logits).reshape(7,13,2)
        return raw
    try:
        for path in MC.model_paths('M3'):
            assert sha(path)==plan['old_model_sha256'][str(path)]
            net=CP.CVR().cuda().eval();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));nets.append(net)
        projector=BatchedProjector();masks=torch.as_tensor(query_masks(),device='cuda')
        u=units[0]
        with np.load(OLD/'observations/evaluation'/f'unit{u}.npz') as d:
            control=score(normalized_z(d['hist'],d['ambient'][None,None]),np.linalg.inv(d['travel'])@d['sensor'],d['noisy'])
        error=float(np.abs(control-original[rows['unit']==u].reshape(7,4,13,2)).max())
        assert error<1e-5
        save(WORK/'engineering/inference_identity.json',dict(status='PASS',max_abs_error=error,unit=u))
        # CPU/GPU parity at a genuinely new physical axis.
        import cnh_displacement_ceiling_render as R
        import cnh_location_reference_gpu as G
        truth=read(OLD/'truth/evaluation'/f'unit{u}.json');boxes=truth['boxes'][0]
        with np.load(OLD/'observations/evaluation'/f'unit{u}.npz') as d:poses=d['sensor'][[0,13]]@A.extrinsic(-30)
        cpu=R.expected(dict(poses=poses,boxes=boxes));engine=G.ExpectedRenderer(poses,boxes[1:])
        try:
            endpoints=engine.render([boxes[0]],deadline_check=deadline)
            expectation=endpoints[0,0]+boxes[0]['rho']/G.ENDPOINT_RHO*(endpoints[0,1]-endpoints[0,0])
            err=float(np.abs(cpu['expectation']-expectation).max());assert err<1e-9
        finally:engine.close()
        save(WORK/'engineering/render_identity.json',dict(status='PASS',max_abs_error=err,absolute_yaw_deg=-15))
        for i,u in enumerate(units):
            deadline();p=WORK/'scores'/f'unit{u}.npz';p.parent.mkdir(exist_ok=True)
            if p.exists():continue
            if not (WORK/'render_receipts'/f'unit{u}.json').exists():render_one(u,new)
            with np.load(WORK/'observations'/f'unit{u}.npz') as d:
                z=normalized_z(d['hist'],d['ambient'][:,None,None]);q=d['query'].copy();n=d['noisy'].copy()
            raw_new=np.stack([score(z[s],q[s],n[s]) for s in range(len(new))])
            with np.load(PARENT/'scores'/f'unit{u}.npz') as d:l,r=d['raw_full'].copy()
            with np.load(PARENT/'secondary22p5/scores'/f'unit{u}.npz') as d:sl,sr=d['raw_full'].copy()
            known={0.:l,15.:original[rows['unit']==u].reshape(7,4,13,2),30.:r,-7.5:sl,37.5:sr}
            values={**known,**{float(a):raw_new[j] for j,a in enumerate(new)}}
            raw=np.stack([values[float(a)] for a in angles]);assert np.isfinite(raw).all()
            np.savez_compressed(p,raw=raw,angles_deg=angles,unit=u,frames=FRAMES)
            save(p.with_suffix('.json'),dict(status='COMPLETE',score_sha256=sha(p),plan_sha256=sha(OUT/'PLAN.json')))
            save(WORK/'progress.json',dict(completed=i+1,total=len(units),unit=u,seconds=time.monotonic()-started))
            print('angular',i+1,'/',len(units),u,round(time.monotonic()-started,1),flush=True)
            del z,raw_new,raw
        save(WORK/'inference_result.json',dict(status='COMPLETE',seconds=time.monotonic()-started,angles_deg=angles.tolist(),
            new_angles_deg=new.tolist(),units=units,source_sha256=sha(__file__),plan_sha256=sha(OUT/'PLAN.json'),
            runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),tf32=False)))
    finally:
        nets.clear();net=projector=masks=None;gc.collect();torch.cuda.empty_cache()
        save(WORK/'release.json',dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()))

if __name__=='__main__':run()

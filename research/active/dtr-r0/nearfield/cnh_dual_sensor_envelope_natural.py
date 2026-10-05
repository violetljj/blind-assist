"""Natural96000 frozen-M3 dual-view EXPLORE; no training or calibration.

Rendering consumes physical boxes and public poses only. Evaluation uses the
pre-existing all-object truth after predictions are sealed. Photon draws have
the original signed aggregate law with a declared new RNG stream.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PARENT = ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005'
OUT = PARENT/'natural'
SOURCE = ROOT/'artifacts.local/work/cnh-margin-confirm-20261002'
PRECHECK = ROOT/'artifacts.local/work/cnh-dual-sensor-alarm-20261005/natural-precheck/result.json'
FRAMES = np.arange(3,16)
SINGLE = .8557642486787612
DUAL = .6549558985617854


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(), 'Preserve sealed evidence: '+str(p)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def deadline():
    if time.time()>=read(PARENT/'PLAN.json')['deadline_unix']:
        raise TimeoutError('Inherited90min deadline reached')


def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    for key,name in [('TEMP','tmp'),('TMP','tmp'),('CUPY_CACHE_DIR','cupy-cache')]:
        p=OUT/name; p.mkdir(exist_ok=True);os.environ[key]=str(p)


def extrinsic(angle):
    from cnh_cvr_pilot import rotation
    e=np.eye(4); e[:3,:3]=rotation(10,'x')@rotation(angle,'y')@rotation(-10,'x')
    return e


def freeze():
    setup(); deadline()
    import cnh_margin_confirm as MC
    import cnh_displacement_ceiling_render as R
    p=read(PARENT/'PLAN.json'); b=p['B']; old=read(PRECHECK)
    assert old['status']=='FEASIBLE_Z1_REPLAY'
    assert sum(r['compared_elements'] for r in old['rows'])==49152
    assert sum(r['bitwise_differences'] for r in old['rows'])==0
    paths=[PRECHECK,SOURCE/'scene_manifest.json',SOURCE/'PLAN.json',SOURCE/'frame_scores_M3_early.npz',SOURCE/'frame_scores_M3.npz']
    paths += [SOURCE/'features/evaluation'/f'unit{u}.npz' for u in b['units']]
    from cnh_dual_sensor_natural_precheck import BIAS
    paths.append(BIAS)
    save(OUT/'PLAN.json',dict(parent_plan_sha256=sha(PARENT/'PLAN.json'),deadline_unix=p['deadline_unix'],
        units=b['units'],candidate_config_sets=b['budget_fallback_config_sets'],thresholds=b['thresholds'],
        admission=b['admission'],precheck_sha256=sha(PRECHECK),source_sha256={str(x):sha(x) for x in paths},
        models_sha256={str(x):sha(x) for x in MC.model_paths('M3')},renderer_sources=R.source_sha256(),
        photon_prefix=b['new_photon_seed_prefix'],photon_rule='SeedSequence[prefix,unit,config,sensor]; fresh independent coarse32 Skellam; equal in distribution to original quarter/radial sum, not bitwise old photon noise',
        pose_rule='Original noisy_poses(seed2026092900+unit*1000+config*10+7) shared head trajectory@known sensor extrinsic',
        query_rule='Original public mount/head schedule transformed by fixed extrinsic; observation construction never reads evaluation labels',
        inference='Float32 normalized signed hist then float16 z1; original past8 projection/13frames/five-seedM3/causal five-score weights; max after per-sensor smoothing',
        evaluator='Reuse original all-object geometry; query-clear and both-query joint-clear reported; no labels reclassified by sensor',
        script_sha256=sha(__file__)))
    # CPU-only pose identity and no truth use in public query.
    import cnh_cvr_pilot as CP
    for u in (96000,96001,96002):
        sensor,travel,noisy=CP.motion_metadata(u,0)
        ex=np.stack([extrinsic(-15),extrinsic(15)])
        np.testing.assert_allclose((noisy@ex[0])@np.linalg.inv(ex[0]),(noisy@ex[1])@np.linalg.inv(ex[1]),atol=1e-12,rtol=0)
    save(OUT/'engineering/cpu.json',dict(status='PASS',admitted_z1_elements=49152,bitwise_differences=0,shared_pose_units=[96000,96001,96002],backend='CPU metadata; TASK_NOT_GPU_SUITABLE'))


class Predictor:
    def __init__(self):
        import torch
        import cnh_cvr_pilot as CP
        import cnh_margin_confirm as MC
        from cnh_cvr_v2_materialize import BatchedProjector
        from cnh_cvr_projection import query_masks
        self.torch=torch; torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.projector=BatchedProjector();self.masks=torch.as_tensor(query_masks(),device='cuda');self.nets=[]
        for path in MC.model_paths('M3'):
            assert sha(path)==read(OUT/'PLAN.json')['models_sha256'][str(path)]
            n=CP.CVR().cuda().eval();n.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));self.nets.append(n)

    def paired_project(self,z,matrices):
        """Original projection algebra, paired config axes batched in groups4."""
        from cnh_cvr_projection import SUB,SHAPE,EDGE,WIDTH
        torch=self.torch;p=self.projector;c,l=matrices.shape[:2]
        t=torch.as_tensor(matrices.reshape(c*l,4,4),dtype=torch.float64,device='cuda')
        points=torch.bmm(p.points[None]-t[:,None,:3,3],t[:,:3,:3])
        radius=torch.linalg.vector_norm(points,dim=2)
        xy=points[:,:,:2]/points[:,:,2:3].clamp_min(1e-30)
        ij=torch.floor((xy+EDGE)/(2*EDGE)*8).long();bins=torch.floor(radius/WIDTH).long()
        valid=(points[:,:,2]>0)&(ij>=0).all(2)&(ij<8).all(2)&(bins>=0)&(bins<16)
        index=(ij[:,:,1].clamp(0,7)*8+ij[:,:,0].clamp(0,7))*16+bins.clamp(0,15)
        weight=valid*p.voxel_volume/(SUB**3)/p.volumes[index]
        values=torch.as_tensor(z,dtype=torch.float64,device='cuda').reshape(c,l,1024)
        mass=torch.gather(values,2,index.reshape(c,l,-1))*weight.reshape(c,l,-1)
        evidence=mass.reshape(c,l,-1,SUB**3).sum(3).reshape(c,l,*SHAPE).float()
        coverage=valid.reshape(c,l,-1,SUB**3).double().mean(3).reshape(c,l,*SHAPE).float()
        total=torch.zeros_like(evidence[:,0]);count=torch.zeros_like(coverage[:,0])
        for past in range(l):total+=evidence[:,past];count+=coverage[:,past]
        return torch.stack((total,count,evidence[:,-1]),1)

    def predict(self,unit,configs,z,angles):
        import cnh_cvr_pilot as CP
        import cnh_displacement_ceiling as D
        import cnh_m3_pose_ensemble as PE
        torch=self.torch; result=[]
        for s,angle in enumerate(angles):
            vox=[];ex=extrinsic(angle)
            noisy=np.stack([CP.motion_metadata(unit,int(c))[2]@ex for c in configs])
            for ci in range(0,len(configs),4):
                frames=[];nn=noisy[ci:ci+4]
                for f in FRAMES:
                    deadline();q=PE.public_query(unit,int(f))@ex
                    ix=np.arange(max(0,int(f)-7),int(f)+1)
                    matrices=(q@np.linalg.inv(nn[:,f]))[:,None]@nn[:,ix]
                    frames.append(self.paired_project(z[s,ci:ci+4][:,ix],matrices).half())
                vox.append(torch.stack(frames,1).reshape(-1,3,24,17,33))
            x=torch.cat(vox);pred=[]
            with torch.inference_mode():
                for begin in range(0,len(x),64):
                    a=x[begin:begin+64].float().clone();a[:,0]=a[:,0].sign()*a[:,0].abs().log1p();a[:,2]=a[:,2].sign()*a[:,2].abs().log1p();a[:,1]/=8
                    a=torch.cat((a,self.masks[None].expand(len(a),-1,-1,-1,-1)),1)
                    pred.append(torch.stack([n(a) for n in self.nets]).mean(0).cpu().numpy())
            result.append(np.concatenate(pred).reshape(len(configs),13,2))
            del x,vox
        return np.stack(result)

    def close(self):
        self.nets.clear();self.projector=self.masks=None;gc.collect();self.torch.cuda.empty_cache()


def batch_check():
    """Engineering only: batched config projection must preserve original voxels."""
    import cnh_cvr_pilot as CP
    import cnh_displacement_ceiling as D
    import cnh_m3_pose_ensemble as PE
    predictor=Predictor();configs=[0,1,2,3];unit=96001;rows=[]
    try:
        with np.load(SOURCE/'features/evaluation'/f'unit{unit}.npz') as d:
            z=np.stack([d['z1'][d['scene']==c] for c in configs])
        ex=extrinsic(-15);noisy=np.stack([CP.motion_metadata(unit,c)[2]@ex for c in configs])
        for f in (3,8,15):
            ix=np.arange(max(0,f-7),f+1);q=PE.public_query(unit,f)@ex
            matrices=(q@np.linalg.inv(noisy[:,f]))[:,None]@noisy[:,ix]
            actual=predictor.paired_project(z[:,ix],matrices)
            expected=predictor.torch.cat([D.project_many(predictor.projector,z[c,ix][None],matrices[c]) for c in range(4)])
            assert predictor.torch.equal(actual,expected)
            rows.append(dict(frame=f,configurations=4,float32_voxel_bitwise_equal=True))
        # Same retained original sensor three-sequence controls, optimized path.
        with np.load(SOURCE/'frame_scores_M3_early.npz') as early,np.load(SOURCE/'frame_scores_M3.npz') as late:
            for u in (96000,96001,96002):
                with np.load(SOURCE/'features/evaluation'/f'unit{u}.npz') as d: zz=d['z1'][d['scene']==0][None,None]
                actual=predictor.predict(u,[0],zz,[0])[0,0];expected=np.concatenate((early[str(u)][0],late[str(u)][0]))
                err=float(np.max(abs(actual-expected)));assert err<1e-5
                rows.append(dict(unit=u,single_raw_logit_max_abs_error=err))
    finally:predictor.close()
    save(OUT/'engineering/batched_projection.json',dict(status='PASS',rows=rows,source_sha256=sha(__file__),batch_size=4))
    snapshot=OUT/'source'/Path(__file__).name;snapshot.parent.mkdir(exist_ok=True)
    assert not snapshot.exists();snapshot.write_bytes(Path(__file__).read_bytes())
    save(OUT/'ENGINEERING_BATCH_CONTRACT.json',dict(original_plan_script_sha256=read(OUT/'PLAN.json')['script_sha256'],
        current_script_sha256=sha(__file__),snapshot_sha256=sha(snapshot),
        dependency_sha256={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in (
            'cnh_location_reference_gpu.py','cnh_displacement_ceiling.py','cnh_cvr_pilot.py','cnh_cvr_v2_materialize.py',
            'cnh_cvr_projection.py','cnh_m3_pose_ensemble.py','cnh_temporal_readout_data.py','cnh_three_level_sequence.py','cnh_sequence_observed_evaluate.py')},
        amendment='Only config-axis execution batching; same FP64 geometry/FP32 sequential history/float16 voxel and M3/threshold/metric/seed/denominator criteria',
        voxel_bitwise_receipt_sha256=sha(OUT/'engineering/batched_projection.json'),
        original_engineering_receipt_sha256=sha(OUT/'engineering/gpu.json'),
        preserved_original_pilot='unit96000 configs0..39 retained and reusable; new batching algebra passes original D.project_many voxel parity'))
    print('BATCHED CONFIG PROJECTION PASS',flush=True)


def engineering():
    setup();deadline()
    import cnh_displacement_ceiling_render as R
    import cnh_location_reference_gpu as G
    import cnh_cvr_pilot as CP
    rows=read(SOURCE/'scene_manifest.json');r=next(r for r in rows if r['unit']==96000 and r['config']==0)
    sensor,_,_=CP.motion_metadata(96000,0);poses=sensor[[0,13]]@extrinsic(-15)
    cpu=R.expected(dict(poses=poses,boxes=r['boxes']))
    engine=G.ExpectedRenderer(poses,r['boxes'])
    try:
        gpu=engine.background_expected(return_device=False)
        err=float(np.max(abs(cpu['expectation']-gpu)))
        np.testing.assert_allclose(cpu['expectation'],gpu,atol=1e-9,rtol=1e-12)
        np.testing.assert_array_equal(cpu['ambient'],engine.ambient)
        backend=engine.metadata
    finally: engine.close()
    predictor=Predictor();parity=[]
    try:
        with np.load(SOURCE/'frame_scores_M3_early.npz') as early,np.load(SOURCE/'frame_scores_M3.npz') as late:
            for u in (96000,96001,96002):
                with np.load(SOURCE/'features/evaluation'/f'unit{u}.npz') as d:z=d['z1'][d['scene']==0][None,None]
                actual=predictor.predict(u,[0],z,[0])[0,0]
                expected=np.concatenate((early[str(u)][0],late[str(u)][0]))
                error=float(np.max(abs(actual-expected)));assert error<1e-5,error
                parity.append(dict(unit=u,raw_logit_max_abs_error=error))
    finally:predictor.close()
    save(OUT/'engineering/gpu.json',dict(status='PASS',expected_max_abs_error=err,single_inference=parity,backend=backend))
    print('engineering PASS',err,parity,flush=True)


def unit_run(unit,configs,predictor,manifest):
    import cnh_displacement_ceiling_render as R
    import cnh_location_reference_gpu as G
    import cnh_cvr_pilot as CP
    from cnh_temporal_readout_data import normalized_z
    deadline();tick=time.monotonic()
    path=OUT/'scores'/f'unit{unit}.npz';receipt=path.with_suffix('.json')
    if receipt.exists():
        old=read(receipt);assert old['score_sha256']==sha(path)
        assert set(configs).issubset(old['configs']);return old
    ex=np.stack([extrinsic(-15),extrinsic(15)]);hist=[];ambient=[]
    sensor,_,_=CP.motion_metadata(unit,0);poses=(sensor[None]@ex[:,None]).reshape(32,4,4)
    for c in configs:
        deadline();row=manifest[(unit,c)]
        engine=G.ExpectedRenderer(poses,row['boxes'])
        try:
            means=engine.background_expected(return_device=False).reshape(2,16,8,8,16)
            amb=engine.ambient.reshape(2,16,8,8).copy()
        finally:engine.close()
        samples=[]
        for s in range(2):
            seed=int(np.random.SeedSequence([2026100521,unit,c,s]).generate_state(1)[0])
            samples.append(R.sample(means[s],amb[s],seed)[0])
        hist.append(np.stack(samples));ambient.append(amb)
    hist=np.stack(hist,axis=1);ambient=np.stack(ambient,axis=1)
    z=normalized_z(hist,ambient);render_seconds=time.monotonic()-tick
    obs=OUT/'observations'/f'unit{unit}.npz';obs.parent.mkdir(exist_ok=True)
    assert not obs.exists();np.savez_compressed(obs,hist=hist,ambient=ambient,z1=z,configs=configs,unit=unit)
    deadline();it=time.monotonic();raw=predictor.predict(unit,configs,z,[-15,15]);infer_seconds=time.monotonic()-it
    path.parent.mkdir(exist_ok=True);assert not path.exists();np.savez_compressed(path,raw=raw,configs=configs,unit=unit,frames=FRAMES,sensors=['L','R'])
    record=dict(status='COMPLETE',unit=unit,configs=configs,score_sha256=sha(path),observation_sha256=sha(obs),render_seconds=render_seconds,infer_seconds=infer_seconds,total_seconds=time.monotonic()-tick,parent_plan_sha256=sha(PARENT/'PLAN.json'),script_sha256=sha(__file__))
    save(receipt,record);return record


def run(config_step,pilot=False):
    setup();deadline();assert read(OUT/'engineering/gpu.json')['status']=='PASS'
    if not pilot:
        assert read(OUT/'ENGINEERING_BATCH_CONTRACT.json')['current_script_sha256']==sha(__file__)
    configs=list(range(0,40,config_step));assert config_step in (1,2,4)
    if not pilot:
        save(OUT/'selected_configs.json',dict(configs=configs,config_step=config_step,reason='Runtime-only selection before full scientific queue; uniformly spaced config IDs; no score-dependent selection'))
    rows=read(SOURCE/'scene_manifest.json');manifest={(r['unit'],r['config']):r for r in rows if r['split']=='evaluation'}
    predictor=Predictor();receipts=[]
    try:
        for u in ([96000] if pilot else read(OUT/'PLAN.json')['units']):
            r=unit_run(u,configs,predictor,manifest);receipts.append(r)
            progress=dict(stage='natural render/infer',completed=len(receipts),total=1 if pilot else 96,last_unit=u,last_activity_unix=time.time(),configs=configs)
            tmp=OUT/'progress.tmp.json';tmp.write_text(json.dumps(progress,indent=2)+'\n',encoding='utf8');tmp.replace(OUT/'progress.json')
            print('natural',u,'configs',len(configs),'seconds',round(r['total_seconds'],2),flush=True)
    finally:
        predictor.close()
    save(OUT/('pilot.json' if pilot else 'run.json'),dict(status='COMPLETE',units=[r['unit'] for r in receipts],configs=configs,total_seconds=sum(r['total_seconds'] for r in receipts),script_sha256=sha(__file__)))


def evaluate():
    import cnh_sequence_observed_evaluate as OE
    import cnh_three_level_sequence as SE
    from cnh_dual_sensor_evaluate import smooth_full
    deadline();configs=read(OUT/'selected_configs.json')['configs']; units=read(OUT/'PLAN.json')['units']
    for path,digest in read(OUT/'PLAN.json')['source_sha256'].items():
        assert sha(path)==digest, 'Frozen source changed: '+path
    geometry,baseline,provenance=OE.load_inputs()
    keep=(geometry['split']=='evaluation')&np.isin(geometry['config'],configs)
    g={k:v[keep] for k,v in geometry.items() if v.shape[0]==len(keep)};single=baseline['M3'][keep]
    raw=[];score_hashes={}
    for u in units:
        p=OUT/'scores'/f'unit{u}.npz';receipt=read(p.with_suffix('.json'));assert receipt['score_sha256']==sha(p)
        with np.load(p) as d:
            ids=[d['configs'].tolist().index(c) for c in configs];raw.append(d['raw'][:,ids])
        score_hashes[str(p)]=sha(p)
    raw=np.stack(raw);sensor=smooth_full(raw);dual=sensor.max(1).transpose(0,1,3,2).reshape(-1,13)
    sensor_query=sensor.transpose(0,2,4,1,3).reshape(-1,2,13)
    assert single.shape==dual.shape==g['frame_ranges'].shape
    ui=np.repeat(np.arange(len(units)),len(configs)*2)
    rng=np.random.default_rng(2026100522);boot=np.asarray([np.bincount(rng.integers(len(units),size=len(units)),minlength=len(units)) for _ in range(1000)])
    flags={};cells={};draws={}
    for arm,scores,thr in [('single',single,SINGLE),('dual15',dual,DUAL),('dual15_original_threshold',dual,SINGLE)]:
        stopped,timely,lead=SE.first_stops(scores,thr,g['frame_ranges']);flags[arm]=(stopped,timely)
        cells[arm]={};draws[arm]={}
        for group in ('all','mode0','mode1','mode2'):
            mask=np.ones(len(single),bool) if group=='all' else g['unit']%3==int(group[-1]);cell={}
            for category in ('contact0-2cm','contact2-5cm','contact>5cm','clear'):
                den=mask&(g['clear_all'] if category=='clear' else g['covered']&(g['ref_category']==category))
                flag=stopped if category=='clear' else timely
                n=int(den.sum());num=int((den&flag).sum());minutes=n*2.6/60
                count=SE.unit_totals(den*flag,ui,len(units));total=SE.unit_totals(den,ui,len(units))
                with np.errstate(divide='ignore',invalid='ignore'):bd=(boot@count)/(boot@total)
                if category=='clear':bd/=2.6/60
                cell[category]=dict(stops=num,n=n,rate=num/n if n else None,proxy_minutes=minutes if category=='clear' else None,
                    stops_per_proxy_minute=num/minutes if category=='clear' and minutes else None,ci95=SE.interval(bd))
                draws[arm][group+'/'+category]=bd
            cell['censored']=dict(n=int((mask&~g['covered']).sum()),already_alarm=int((mask&~g['covered']&stopped).sum()))
            first=(scores>=thr).argmax(1);pick=np.take_along_axis(sensor_query,first[:,None,None],2)[:,:,0]
            chosen=mask&stopped
            cell['first_report_source']=dict(single=int(chosen.sum())) if arm=='single' else dict(L=int((chosen&(pick[:,0]>pick[:,1])).sum()),R=int((chosen&(pick[:,1]>pick[:,0])).sum()),tie=int((chosen&(pick[:,0]==pick[:,1])).sum()))
            cell['first_report_source_unit']='query episode; HEAD/BODY counted separately'
            cells[arm][group]=cell
    comparisons={}
    for arm in ('dual15','dual15_original_threshold'):
        comparisons[arm]={}
        for category in ('contact0-2cm','contact>5cm','clear'):
            den=g['clear_all'] if category=='clear' else g['covered']&(g['ref_category']==category)
            f=0 if category=='clear' else 1;a=flags[arm][f];b=flags['single'][f]
            comparisons[arm][category]=dict(added=int((den&a&~b).sum()),removed=int((den&b&~a).sum()),paired_unit_delta_ci95=SE.interval(draws[arm]['all/'+category]-draws['single']['all/'+category]))
    joint_den=g['clear_all'].reshape(-1,2).all(1);joint={}
    physical_sensor=sensor.transpose(0,2,1,3,4).reshape(-1,2,13,2).max(-1)
    for arm,(stopped,_) in flags.items():
        n=int(joint_den.sum());stops=int((joint_den&stopped.reshape(-1,2).any(1)).sum())
        joint[arm]=dict(stops=stops,n=n,proxy_minutes=n*2.6/60,stops_per_proxy_minute=stops/(n*2.6/60) if n else None)
        if arm!='single':
            threshold=DUAL if arm=='dual15' else SINGLE
            first=(physical_sensor.max(1)>=threshold).argmax(1)
            values=np.take_along_axis(physical_sensor,first[:,None,None],2)[:,:,0]
            chosen=joint_den&stopped.reshape(-1,2).any(1)
            joint[arm]['first_report_source']=dict(L=int((chosen&(values[:,0]>values[:,1])).sum()),R=int((chosen&(values[:,1]>values[:,0])).sum()),tie=int((chosen&(values[:,0]==values[:,1])).sum()))
    if len(configs)==40:
        assert cells['single']['all']['contact0-2cm']['stops']==26 and cells['single']['all']['contact0-2cm']['n']==31
        assert cells['single']['all']['contact>5cm']['stops']==162 and cells['single']['all']['contact>5cm']['n']==164
        assert cells['single']['all']['clear']['stops']==205 and cells['single']['all']['clear']['n']==4485
    ledger=OUT/'ledger.npz';np.savez_compressed(ledger,single=single,dual=dual,sensor_scores=sensor_query,unit=g['unit'],config=g['config'],query=g['query'],clear=g['clear_all'],covered=g['covered'],ref_category=g['ref_category'],frame_ranges=g['frame_ranges'])
    result=dict(status='COMPLETE',units=units,configs=configs,scenes=len(units)*len(configs),query_episodes=len(single),cells=cells,joint_clear=joint,comparisons=comparisons,thresholds=dict(single=SINGLE,dual15=DUAL,dual15_original_threshold=SINGLE),
        provenance=dict(parent_plan_sha256=sha(PARENT/'PLAN.json'),plan_sha256=sha(OUT/'PLAN.json'),scores_sha256=score_hashes,geometry=provenance,ledger_sha256=sha(ledger),script_sha256=sha(__file__)),
        limits=['Consumed synthetic Development, not real walking prevalence or hardware safety','Clear truth is fixed all-object query-clear; sidebar objects can contribute but added stops cannot establish causal source','Same scene boxes/public pose, new independent photon streams; aggregate photon law equals original quarter/radial distribution','Natural and controlled pilot2 head mode1 scan is +/-20deg; +/-15deg side extrinsics yield +/-35deg outer optical axes','Censored episodes are reported separately; no unobserved deadline extrapolation','Existing controlled matching threshold transfers unchanged; natural cost is not recalibrated'])
    save(OUT/'result.json',result)
    lines=['Natural96000双传感器模拟补完','',f'{len(units)} units × {len(configs)} configs；{len(single)} query episodes。所有阈值冻结。','', '| Arm | clear first stops / minutes | shallow timely | deep timely |','|---|---|---|---|']
    for arm in cells:
        a=cells[arm]['all'];c=a['clear'];sh=a['contact0-2cm'];de=a['contact>5cm']
        lines.append(f"| {arm} | {c['stops']}/{c['proxy_minutes']:.2f} ({c['stops_per_proxy_minute']:.3f}/min) | {sh['stops']}/{sh['n']} | {de['stops']}/{de['n']} |")
    lines+=['','Joint both-query clear physical episodes: '+json.dumps(joint),'',*result['limits']]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({a:cells[a]['all'] for a in cells}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','engineering','batch-check','pilot','run','evaluate']);ap.add_argument('--config-step',type=int,default=1);args=ap.parse_args()
    setup()
    try:
        if args.stage=='freeze':freeze()
        elif args.stage=='engineering':engineering()
        elif args.stage=='batch-check':batch_check()
        elif args.stage in ('pilot','run'):run(args.config_step,args.stage=='pilot')
        else:evaluate()
    except BaseException as e:
        save(OUT/'failures'/f'{args.stage}-{time.time_ns()}.json',dict(stage=args.stage,error=repr(e)))
        raise

"""Two fixed new simulated backgrounds; frozen M3/local fusion transfer."""
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S
import cnh_bar_local_readout_dev as L
import cnh_bar_fusion_probe_dev as F

OUT=B.ROOT/'artifacts.local/work/cnh-bar-transfer-dev-20261009/background'
SPLIT=OUT.parent/'crosscal/background_split.json'
NOISE_PREFIX=2026100967
FIXED_M=.9404184587540165
FIXED_L=4.625390338985158
BACKGROUND_SPECS=[dict(id=0,name='low_floor_wall',floor_rho=.15,wall_rho=.20),
                  dict(id=1,name='strong_floor_wall',floor_rho=.55,wall_rho=.65)]


def save(name,value):B.save(OUT/name,value)


def backgrounds(row,spec):
    assert row['background'][:2]==S.BASE_BG
    bg=[dict(lo=list(b['lo']),hi=list(b['hi']),rho=b['rho']) for b in row['background']]
    bg[0]['rho']=spec['floor_rho'];bg[1]['rho']=spec['wall_rho']
    bg[1]['lo'][2]=3.4;bg[1]['hi'][2]=3.6
    return bg


def split_ids(split):
    assert split['seed']==2026100901
    cal=split['cal_scene_ids'];evaluation=split['eval_scene_ids']
    assert len(set(cal)|set(evaluation))==492 and not(set(cal)&set(evaluation))
    assert sorted(cal+evaluation)==list(range(492))
    return cal,evaluation


def prepare():
    began=time.monotonic();A,G,R,normal,T=S.imports()
    import cnh_displacement_ceiling as D
    p0=json.loads((S.OUT/'PLAN.json').read_text());rows=p0['scene_rows']
    split=json.loads(SPLIT.read_text());cal,evaluation=split_ids(split)
    with np.load(S.OUT/'geometry.npz') as d:category=d['category'];sensor=d['sensor'];query=d['public_query']
    inputs=[S.OUT/'PLAN.json',S.OUT/'geometry.npz',S.OUT/'physical.npz',SPLIT,D.BIAS,*A.M3_MODELS]
    sources={Path(__file__),Path(L.__file__),Path(F.__file__),*[Path(m.__file__).resolve() for n,m in list(sys.modules.items()) if n.startswith('cnh_') and getattr(m,'__file__',None)]}
    plan=dict(task='CNH_BAR_BACKGROUND_TRANSFER_DEV_20261009',lane='EXPLORE new simulated photons; consumed target grid',
        authorization='User advanced cache cross-calibration, pose and new-background transfer',
        goal='Fixed original poses/targets with two predeclared floor/wall backgrounds, fixed transfer followed by separate scene-holdout calibration diagnostic',
        budgets_wall_seconds=dict(prepare_evaluate_focused_CPU=300,render_infer_GPU_wall=1200,independent_audit_CPU=180),
        budget_unit='Cumulative stage wall seconds including failures; programming and waiting excluded. GPU-wall stage includes CPU sampling/parity/I/O surrounding actual CUDA renderer/model kernels.',
        scope='2 backgrounds x492 original target scenes xK4 x16 raw frames; 13 output framesf3..15; timelyf3..13. Frozen -10 poses/M3/6x1x3 local patches/original EMA. No training, background tuning, new scenes, extra MC, hardware or confirmation claim.',
        background_specs=BACKGROUND_SPECS,background_wall_world_z=[3.4,3.6],original_target_scene_count=492,
        replicas=4,raw_frames=16,output_frames=list(range(3,16)),model_ensemble_examples=51168,
        background_only_truth='New floor remains y1.65..1.8, new wall query front>=3.4m for all16frames. Original large_face additions stay unchanged. Every original category must match independent volume and surface classification at all13outputs.',
        fixed=dict(M3_original=B.THETA,fusion_raised_M3=FIXED_M,fusion_local=FIXED_L,gate='unbounded_below',
                   formula='(smooth_M3>=fixed_M) OR ((smooth_M3<fixed_M) AND (smooth_local>=fixed_L)); no calibration of fixed branch'),
        adapt=dict(split_file=B.logical_path(SPLIT),seed=2026100901,fold=0,cal_scene_ids=cal,eval_scene_ids=evaluation,
            original_M3_threshold=B.THETA,k_formula='k=min(C0,5*cal_joint_clear_denominator/4576)',
            C0='Actual background-specific cal clear-slot cost of original M3 threshold',
            raised='Choose M3 threshold>=original nearest whole-tie cal clear cost C0-k; higher threshold at equal residual',
            local='On cal clear slots not already alarmed by raised M3, eligible local max across queries with M3<raised; choose nearest whole ties to C0-raised_actual_cost',
            eval='Apply cal thresholds to every eval scene; report baseline/fixed/adapt costs, residuals and paired events. Eval is never used to choose thresholds. Saturated C0=0 does not require strict improvement.'),
        noise=dict(prefix=NOISE_PREFIX,seed_formula='SeedSequence[prefix,background_id,scene_id,replica].generate_state(1)[0]',
                   sampler='Original R.sample exact aggregate independent Poisson counts/background; signed/count/backgroundint32 archived'),
        physical_parity='GPU ExpectedRenderer against CPU R.expected on first/middle/last representative scene f0/13/15 plus extra-background representatives; no score selection',
        decision_check='Retain original physical/query categories and denominators; compare fusion vs M3 on the same new-background photons. Fixed and adapt branches are distinct. New background calibration is not a blind confirmation. Preserve dark lengths, family and all paired gains/losses and clear segments/clips/pass.',
        stop='Finish two declared backgrounds within cumulative budgets or preserve partial output; no GPU start before parent scheduling message; no retry by background/target/threshold retuning',
        inputs_sha256={B.logical_path(p):B.sha(p) for p in inputs},source_sha256={B.logical_path(p):B.sha(p) for p in sorted(sources,key=str)},
        deliverables='Full16 histogram/expectations/ambient/seeds, M3/local raw scores, fixed/adapt event and cost ledgers, source hashes, cleanup receipts and focused checks')
    save('PLAN.json',plan);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    classifications=0
    for spec in BACKGROUND_SPECS:
        for i,row in enumerate(rows):
            bg=backgrounds(row,spec)
            for f in B.FRAMES:
                shift=sensor[f,2,3]
                boxes=[S.translated([row],shift)[0],*S.translated(bg,shift)]
                surface=[T.classify_boxes(boxes,q,np.eye(4))['all_category'] for q in (0,1)]
                solid=S.solid_categories(boxes)
                assert surface==solid==category[i].tolist(),(spec['id'],i,int(f),surface,solid,category[i])
                assert S.solid_categories(S.translated(bg,shift))==['clear','clear']
                classifications+=1
            assert np.asarray(bg[1]['lo'])[2]-sensor[:,2,3].max()>3.
            if time.monotonic()-began>300:raise TimeoutError('CPU prepare budget300s')
    np.savez_compressed(OUT/'geometry.npz',category=category,sensor=sensor,query=query)
    cc=(category=='clear').all(1)
    assert cc.sum()==88 and ((category=='pass').any(1)&~(category=='contact').any(1)).sum()==88
    save('prepare_result.json',dict(status='COMPLETE',seconds=time.monotonic()-began,
        category_surface_volume_checks=classifications,backgrounds=2,scenes_each=492,
        category_query_contact_scenes=(category=='contact').sum(0).tolist(),clear_scenes_each=88,pass_scenes_each=88,
        cal_clear_scenes=int(cc[cal].sum()),eval_clear_scenes=int(cc[evaluation].sum()),
        baseline_query_denominator_each=(4*(category=='contact').sum(0)).tolist(),
        physical_truth_retained=True,split_sha256=B.sha(SPLIT)))
    print('BACKGROUND_PREPARED',round(time.monotonic()-began,3),classifications,flush=True)


def run():
    began=time.monotonic();p=json.loads((OUT/'PLAN.json').read_text())
    previous=sum(json.loads(f.read_text())['seconds'] for f in OUT.glob('run_failure_*.json'))
    def check():
        if previous+time.monotonic()-began>1200:raise TimeoutError('Cumulative background GPU-wall1200s')
    engine=None;maps=[];renderer=None
    try:
        for name,digest in {**p['inputs_sha256'],**p['source_sha256']}.items():assert B.sha(B.ROOT/name)==digest,name
        A,G,R,normal,T=S.imports();A.OUT=OUT/'runtime';A.setup_gpu();engine=A.Engine()
        rows=json.loads((S.OUT/'PLAN.json').read_text())['scene_rows']
        with np.load(OUT/'geometry.npz') as d:sensor=d['sensor'];query=d['query']
        qs,locations=L.patches();norms=[];projection_parity=[]
        for f in B.FRAMES:
            check();start=max(0,int(f)-7);m=(query[f]@np.linalg.inv(sensor[f]))[None]@sensor[start:f+1]
            cp=L.Projection(engine,m);maps.append(cp);nn=[]
            for q in qs:
                coeff=q@cp.p;var=np.asarray(coeff.multiply(coeff).sum(1)).ravel()
                nn.append(np.where(var>0,np.sqrt(var),np.inf))
            norms.append(nn)
        results=[];physical_parity=[];backend=[]
        for spec in p['background_specs']:
            bgid=spec['id'];expectation=np.empty((492,16,8,8,16),np.float64)
            target_signal=np.empty((492,16),np.float64);ambient=None
            groups={}
            for i,row in enumerate(rows):
                bg=backgrounds(row,spec);key=json.dumps(bg,sort_keys=True);groups.setdefault(key,[]).append(i)
            for key,ids in groups.items():
                check();bg=json.loads(key);renderer=G.ExpectedRenderer(sensor,bg)
                try:
                    if ambient is None:ambient=renderer.ambient.copy()
                    else:np.testing.assert_array_equal(ambient,renderer.ambient)
                    backend.append(dict(background=bgid,context_scene_count=len(ids),renderer=renderer.metadata))
                    for first,ends in renderer.iter_render([rows[i] for i in ids],candidate_batch=4,pose_batch=16,deadline_check=check):
                        for j in range(len(ends)):
                            i=ids[first+j];alpha=rows[i]['rho']/G.ENDPOINT_RHO
                            expectation[i]=ends[j,0]+alpha*(ends[j,1]-ends[j,0])
                            target_signal[i]=(alpha*(ends[j,1]-ends[j,0])).sum((1,2,3))
                    probes=set([ids[0],ids[len(ids)//2],ids[-1]])
                    for i in sorted(probes):
                        check();ref=R.expected(dict(poses=sensor[[0,13,15]],boxes=[rows[i],*bg]))
                        diff=float(np.abs(ref['expectation']-expectation[i,[0,13,15]]).max())
                        np.testing.assert_allclose(ref['expectation'],expectation[i,[0,13,15]],rtol=1e-11,atol=1e-8)
                        np.testing.assert_array_equal(ref['ambient'],ambient[[0,13,15]])
                        physical_parity.append(dict(background=bgid,scene=i,frames=[0,13,15],max_abs=diff))
                finally:
                    renderer.close();renderer=None
            seeds=np.empty((492,4),np.uint32);hist=np.empty((492,4,16,8,8,16),np.int32)
            counts=np.empty_like(hist);background_counts=np.empty_like(hist)
            for i in range(492):
                check()
                for k in range(4):
                    seed=int(np.random.SeedSequence([NOISE_PREFIX,bgid,i,k]).generate_state(1)[0]);seeds[i,k]=seed
                    hist[i,k],counts[i,k],background_counts[i,k]=R.sample(expectation[i],ambient,seed)
            np.testing.assert_array_equal(hist,counts-background_counts)
            z=normal(hist.reshape(-1,16,8,8,16),np.broadcast_to(ambient,(1968,*ambient.shape)))
            m3=np.empty((1968,13,2),np.float32);local=np.empty((1968,13,2),np.float64)
            for j,f in enumerate(B.FRAMES):
                start=max(0,int(f)-7)
                for first in range(0,1968,32):
                    check();feature=maps[j](z[first:first+32,start:f+1]).cpu().numpy().astype(np.float16)
                    if first==0 and f in (3,12,13,15):
                        m=(query[f]@np.linalg.inv(sensor[f]))[None]@sensor[start:f+1]
                        ref=engine.project(z[first:first+1,start:f+1],m[None]).cpu().numpy().astype(np.float16)
                        np.testing.assert_array_equal(feature[:1],ref)
                        projection_parity.append(dict(background=bgid,frame=int(f),max_abs=0))
                    m3[first:first+32,j]=engine.predict(feature)
                    local[first:first+32,j],_=L.scan(feature,qs,norms[j],locations)
                print('BACKGROUND',bgid,'FRAME',int(f),'wall',round(time.monotonic()-began,2),flush=True)
            payload=OUT/f'background_{bgid}_physical.npz';assert not payload.exists()
            np.savez_compressed(payload,hist=hist,counts=counts,background_counts=background_counts,ambient=ambient,
                expectation=expectation,target_signal=target_signal,seeds=seeds,
                m3_raw=m3.reshape(492,4,13,2),local_raw=local.reshape(492,4,13,2))
            record=dict(background=bgid,name=spec['name'],payload=payload.name,payload_sha256=B.sha(payload))
            save(f'background_{bgid}_receipt.json',record);results.append(record)
            del hist,counts,background_counts,z,m3,local,expectation,target_signal
            gc.collect();check()
        save('result_run.json',dict(status='COMPLETE',seconds=time.monotonic()-began,previous_failure_seconds=previous,
            actual_device=engine.torch.cuda.get_device_name(),model_ensemble_examples=51168,
            backend=backend,physical_renderer_parity=physical_parity,projection_parity=projection_parity,
            backgrounds=results,new_simulated_signed_samples=3936,training=0))
    except BaseException as exc:
        save('run_failure_'+str(time.time_ns())+'.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-began));raise
    finally:
        if renderer is not None:renderer.close()
        maps.clear()
        if engine is not None:
            torch=engine.torch;engine.projector=None;engine.nets=[]
            for name in ('_pts32','_vol32'):
                if hasattr(engine,name):delattr(engine,name)
            engine=None;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
            save('release_'+str(time.time_ns())+'.json',dict(pid=os.getpid(),torch_allocated_bytes=torch.cuda.memory_allocated(),
                torch_reserved_bytes=torch.cuda.memory_reserved(),semantics='Before process exit; final PID release checked by parent'))


def adapt_thresholds(m3,local,cat,cal):
    cc=(cat[cal]=='clear').all(1);scores=m3[cal][cc];ls=local[cal][cc]
    flags=scores>=B.THETA;C0=int(flags.any(-1).sum());den=int(flags.shape[0]*4*13)
    k_uncapped=5*den/4576;k=min(C0,k_uncapped)
    raised,cost=F.nearest(scores.max(-1),C0-k,floor=B.THETA)
    base=scores>=raised;remaining=~base.any(-1)
    eligible=np.where(scores<raised,ls,-np.inf).max(-1)
    local_theta,added=F.nearest(eligible[remaining],C0-cost)
    return dict(raised_M3=raised if np.isfinite(raised) else None,local_theta=local_theta if np.isfinite(local_theta) else None,
        raised_is_infinity=bool(np.isinf(raised)),local_is_infinity=bool(np.isinf(local_theta)),
        cal_C0=C0,cal_clear_denominator=den,k_uncapped=k_uncapped,k_capped=k,k_clipped=bool(k<k_uncapped),
        raised_target=C0-k,raised_actual_cost=cost,raised_tie_residual=cost-(C0-k),
        local_added_target=C0-cost,local_added_cost=added,total_cal_cost=cost+added,total_cal_residual=cost+added-C0,
        saturation=bool(C0==0))


def thresholds(a):return (np.inf if a['raised_is_infinity'] else a['raised_M3'],np.inf if a['local_is_infinity'] else a['local_theta'])


def evaluate(m3,local,cat,ids,raised,local_theta,darkids,rows):
    old=F.metrics(m3[ids]>=B.THETA,cat[ids]);flags=F.alarm(m3[ids],local[ids],raised,local_theta,'unbounded_below')
    new=F.metrics(flags,cat[ids]);paired=F.compare(old[1],new[1],cat[ids],ids)
    groups=[]
    for family in S.FAMILIES:
        chosen=np.array([j for j,i in enumerate(ids) if rows[i]['family']==family])
        groups.append(dict(family=family,paired=F.compare(old[1][chosen],new[1][chosen],cat[ids][chosen],np.asarray(ids)[chosen])))
    dark=np.array([j for j,i in enumerate(ids) if i in darkids])
    return dict(baseline=old[0],candidate=new[0],paired=paired,groups=groups,
        clear_cost_residual=new[0]['clear_slots']-old[0]['clear_slots'],
        dark4cm=F.compare(old[1][dark],new[1][dark],cat[ids][dark],np.asarray(ids)[dark]))


def analyze():
    began=time.monotonic();p=json.loads((OUT/'PLAN.json').read_text());prep=json.loads((OUT/'prepare_result.json').read_text())
    def check():
        if prep['seconds']+time.monotonic()-began>300:raise TimeoutError('Shared prepare/eval/focused300s')
    rr=json.loads((OUT/'result_run.json').read_text());rows=json.loads((S.OUT/'PLAN.json').read_text())['scene_rows']
    cal=p['adapt']['cal_scene_ids'];evaluation=p['adapt']['eval_scene_ids']
    darkids=json.loads((B.ROOT/'artifacts.local/work/cnh-bar-representation-dev-20261008/fp16/PLAN.json').read_text())['contact_scene_ids']
    with np.load(OUT/'geometry.npz') as d:cat=d['category']
    ledger=[];results=[];score_arrays=[]
    for receipt in rr['backgrounds']:
        check();bgid=receipt['background'];payload=OUT/receipt['payload'];assert B.sha(payload)==receipt['payload_sha256']
        with np.load(payload) as d:m3=B.smooth(d['m3_raw']);local=B.smooth(d['local_raw'])
        fixed_all=evaluate(m3,local,cat,np.arange(492),FIXED_M,FIXED_L,darkids,rows)
        fixed_eval=evaluate(m3,local,cat,np.array(evaluation),FIXED_M,FIXED_L,darkids,rows)
        adaptive=adapt_thresholds(m3,local,cat,cal);am,al=thresholds(adaptive)
        adapt_eval=evaluate(m3,local,cat,np.array(evaluation),am,al,darkids,rows)
        adapt_cal=evaluate(m3,local,cat,np.array(cal),am,al,darkids,rows)
        assert adapt_cal['candidate']['clear_slots']==adaptive['total_cal_cost']
        results.append(dict(background=bgid,fixed_all=fixed_all,fixed_eval=fixed_eval,
                            adapt_calibration=adaptive,adapt_cal=adapt_cal,adapt_eval=adapt_eval))
        allflags=F.alarm(m3,local,FIXED_M,FIXED_L,'unbounded_below');adaptflags=F.alarm(m3,local,am,al,'unbounded_below')
        original=m3>=B.THETA
        score_arrays.append((m3,local,allflags,adaptflags))
        for scope,ids,candidate in (('fixed_all',np.arange(492),allflags),('adapt_eval',np.array(evaluation),adaptflags)):
            for i,k,q in np.argwhere(np.broadcast_to((cat=='contact')[:,None,:],(492,4,2))):
                if i not in ids:continue
                a=np.flatnonzero(original[i,k,:11,q]);b=np.flatnonzero(candidate[i,k,:11,q])
                ledger.append(dict(background=bgid,scope=scope,scene=int(i),replica=int(k),height=('HEAD','BODY')[q],
                    source_key=f'background{bgid}:scene{i}:replica{k}',family=rows[i]['family'],variant=rows[i]['variant'],
                    placement=rows[i]['placement'],rho=rows[i]['rho'],dark4cm=int(i in darkids),
                    baseline=int(a.size>0),candidate=int(b.size>0),gain=int(a.size==0 and b.size>0),loss=int(a.size>0 and b.size==0),
                    baseline_first_frame=int(B.FRAMES[a[0]]) if a.size else '',candidate_first_frame=int(B.FRAMES[b[0]]) if b.size else ''))
    with (OUT/'event_ledger.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    np.savez_compressed(OUT/'evaluated.npz',m3_smooth=np.stack([v[0] for v in score_arrays]),
        local_smooth=np.stack([v[1] for v in score_arrays]),fixed_alarm=np.stack([v[2] for v in score_arrays]),
        adapted_alarm=np.stack([v[3] for v in score_arrays]),category=cat,cal_scene_ids=cal,eval_scene_ids=evaluation)
    save('result_analysis.json',dict(status='COMPLETE',seconds=time.monotonic()-began,prepare_seconds=prep['seconds'],
        backgrounds=results,ledger_rows=len(ledger),fixed_thresholds=p['fixed'],
        limitations='Both branches use simulated new photons on consumed original shapes, not blind confirmation. Fixed transfer uses old thresholds; adapted thresholds use only fixed scene cal half, and holdout cost equality is not promised. Segments/clips/pass are distinct costs. K4 and related shape cells are not independent deployment trials. No model/policy promotion.'))
    check();print('BACKGROUND_ANALYSIS_COMPLETE',round(time.monotonic()-began,3),flush=True)


def focused_check():
    began=time.monotonic();p=json.loads((OUT/'PLAN.json').read_text());ar=json.loads((OUT/'result_analysis.json').read_text())
    def check():
        if ar['prepare_seconds']+ar['seconds']+time.monotonic()-began>300:raise TimeoutError('Shared CPU300s')
    with np.load(OUT/'evaluated.npz') as d:
        m3=d['m3_smooth'];local=d['local_smooth'];fixed=d['fixed_alarm'];adapted=d['adapted_alarm'];cat=d['category']
        cal=d['cal_scene_ids'];evaluation=d['eval_scene_ids']
    rows=json.loads((S.OUT/'PLAN.json').read_text())['scene_rows']
    darkids=json.loads((B.ROOT/'artifacts.local/work/cnh-bar-representation-dev-20261008/fp16/PLAN.json').read_text())['contact_scene_ids']
    for b in range(2):
        check();wanted=ar['backgrounds'][b];a=adapt_thresholds(m3[b],local[b],cat,cal.tolist())
        assert a==wanted['adapt_calibration'];am,al=thresholds(a)
        np.testing.assert_array_equal(fixed[b],F.alarm(m3[b],local[b],FIXED_M,FIXED_L,'unbounded_below'))
        np.testing.assert_array_equal(adapted[b],F.alarm(m3[b],local[b],am,al,'unbounded_below'))
        for name,ids,tm,tl in (('fixed_all',np.arange(492),FIXED_M,FIXED_L),('fixed_eval',evaluation,FIXED_M,FIXED_L),
                              ('adapt_cal',cal,am,al),('adapt_eval',evaluation,am,al)):
            assert evaluate(m3[b],local[b],cat,ids,tm,tl,darkids,rows)==wanted[name]
        with np.load(OUT/f'background_{b}_physical.npz') as d:
            np.testing.assert_array_equal(d['hist'],d['counts']-d['background_counts'])
            for i,k in ((0,0),(246,2),(491,3)):
                seed=int(np.random.SeedSequence([NOISE_PREFIX,b,i,k]).generate_state(1)[0]);assert d['seeds'][i,k]==seed
    with (OUT/'event_ledger.csv').open(encoding='utf8') as f:ledger=list(csv.DictReader(f))
    for row in ledger:
        b=int(row['background']);i=int(row['scene']);k=int(row['replica']);q=('HEAD','BODY').index(row['height'])
        a=bool((m3[b,i,k,:11,q]>=B.THETA).any());flags=fixed if row['scope']=='fixed_all' else adapted
        v=bool(flags[b,i,k,:11,q].any())
        assert (int(row['baseline']),int(row['candidate']),int(row['gain']),int(row['loss']))==(int(a),int(v),int(not a and v),int(a and not v))
        assert row['source_key']==f'background{b}:scene{i}:replica{k}'
    assert len(ledger)==ar['ledger_rows'];check()
    save('focused_check.json',dict(status='PASS',seconds=time.monotonic()-began,
        CPU_cumulative_seconds=ar['prepare_seconds']+ar['seconds']+time.monotonic()-began,
        backgrounds=2,scheme_reports_recomputed=8,event_ledger_rows=len(ledger),
        counts_signed_parity=True,adapt_eval_not_read_by_calibrator=True,new_inference=0,new_sampling=0))
    print('BACKGROUND_CHECK_PASS',round(time.monotonic()-began,3),flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()

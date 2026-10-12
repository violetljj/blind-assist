"""Aligned straight finite-grid physical pitch probe with frozen M3.

EXPLORE simulator diagnostics, exact relative poses, no training/device claims.
Observation production/inference is independent of the evaluator's categories.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT/'artifacts.local/work/cnh-aligned-boundary-dev-20261008'
THETA = .8557642486787612
K = 4
FRAMES = np.arange(3,16)
XS = (-.50,-.36,-.29,0.,.29,.36,.50)
YS = (-.15,.10,.35,.50,.65,.85)
SIDES = (.04,.10)
RHOS = (.25,.65)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logical_path(path):
    """Keep physical junction targets routed through canonical artifacts.local."""
    path=Path(path)
    if path.is_relative_to(ROOT):return path.relative_to(ROOT).as_posix()
    return (Path('artifacts.local')/path.relative_to((ROOT/'artifacts.local').resolve())).as_posix()


def scenes():
    rows=[]
    for i,(y,x,side,rho) in enumerate(itertools.product(YS,XS,SIDES,RHOS)):
        rows.append(dict(id=i,x=x,y=y,side=side,rho=rho,
          lo=[x-side/2,y-side/2,.65],hi=[x+side/2,y+side/2,.65+side]))
    return rows


def poses(pitch):
    import cnh_cvr_pilot as CP
    sensor=np.repeat(np.eye(4)[None],16,axis=0)
    sensor[:,:3,:3]=CP.rotation(pitch,'x')
    sensor[:,2,3]=np.arange(16)*.16-2.4
    query=np.repeat(np.eye(4)[None],16,axis=0)
    query[:,:3,:3]=sensor[:,:3,:3]
    return sensor,query


def imports():
    import cnh_extrinsic_aug_data as D
    import cnh_active_scan_dev as A
    import cnh_location_reference_gpu as G
    import cnh_displacement_ceiling_render as R
    from cnh_temporal_readout_data import normalized_z
    A.OUT=OUT/'runtime'
    geometry=Path(sys.modules['cnh_track_a_geometry'].__file__).resolve()
    if geometry.parent.name!='source':
        raise ValueError('Do not import shared geometry WIP')
    return A,G,R,normalized_z


def prepare():
    began=time.monotonic()
    A,G,R,_=imports()
    geo=json.loads((OUT/'geometry/result.json').read_text(encoding='utf-8'))
    # Only finite sphere geometry has selected pitch, no M3 outcomes were used.
    pitch=float(geo['selected']['pitch_deg'])
    rows=scenes()
    background=[dict(lo=[-8.,1.65,-8.],hi=[8.,1.80,9.],rho=.30),
                dict(lo=[-8.,-3.,4.2],hi=[8.,1.65,4.4],rho=.35)]
    paths=[HERE/'cnh_aligned_boundary_dev.py',HERE/'cnh_aligned_geometry_map_dev.py',
           OUT/'geometry/PLAN.json',OUT/'geometry/result.json',*A.M3_MODELS]
    paths += [Path(m.__file__).resolve() for n,m in list(sys.modules.items())
              if n.startswith('cnh_') and getattr(m,'__file__',None)]
    paths=sorted(set(paths),key=str)
    plan=dict(task='CNH_ALIGNED_BOUNDARY_DEV_20261008',lane='EXPLORE finite simulated grid',
      authorization='User 推进 after proposed aligned-straight boundary and one targeted change',
      scope='Temporarily defer heading estimation/device work; fixed straight travel/yaw alignment; one single-sensor physical pitch comparison, no training',
      budgets_wall_seconds=dict(geometry_CPU=180,prepare_CPU=180,render_infer=1800,analysis_checks_CPU=300),
      baseline_pitch_deg=-10.,candidate_pitch_deg=pitch,frames=FRAMES.tolist(),replicas=K,
      scene_rows=rows,background=background,
      model='Frozen mean logits of M3 seed0..4, existing exponential last5 smoothing, original theta',
      input='Single sensor; exact relative poses (oracle diagnostic), true install pitch in photons and projection; fixed HEAD/BODY masks, no pose noise, no gate or direction labels',
      time='16 samples f0..15, dt.2s span3.0s; target front3.05-.16*f, timely last f13(.97m)>=.9m; f14(.81m) late. All13 output slots costed, .2s proxy/slot, not real reminders/minute.',
      random='SeedSequence[2026100819,scene_id,replica] identical across pitch; common RNG seed, not identical physical photons. K4 are noise replicates, not participants.',
      truth='Evaluator-only all-object surface contact; pass expands x from .30 to .40 at same y/z. Physical clear only if both height queries clear throughout; contact target-band timely primary, any-height timely secondary.',
      workpoints='Primary fixed theta=.8557642486787612 with actual costs; secondary candidate whole-score-tie threshold nearest to baseline actual all13 jointly-clear alarm-slot cost; report residual and do not call nonzero residual equal cost. Secondary Development evaluation selection, not deployment calibration.',
      decision_check='Existing aligned cohort has only1/2 timely misses; new balanced finite cube grid has72contact/48pass/48clear physical scenes beforeK4 (HEAD/BODY36each). One timely event per-height1/144=.694pp. Missing/unfinished cells kept in168-scene denominator; no partial ranking. A new pitch merits further development only at exactly matched integer clear-slot cost, both height net timely differences>=0 and at least one>0, paired losses<=2 in each height. This intentional investment rule is not statistical noninferiority or user/safety cost.',
      stop='Complete two-pitch queue or stage cap; no M3-result angle retuning, extra models or training; preserve original stopped perturb and coverage optimization recipes',
      deliverables='Geometry and model maps/CSV; actual photons and scores; paired gains/losses and clear-slot/segment/clip costs; focused validation; report and scoped commit/push',
      limitations='Small cubes, fixed floor/wall/reflectance, same simulator and reused frozen model; ideal alignment/exact poses. No population/device/geometry coverage certificate. Downward pitch may sacrifice upper HEAD; no causal label for visible-but-missed.',
      source_sha256={logical_path(p):sha(p) for p in paths})
    save(OUT/'PLAN.json',plan)
    with (OUT/'source_snapshot_pre_run.py').open('xb') as f:
        f.write(Path(__file__).read_bytes())
    save(OUT/'prepare_result.json',dict(status='COMPLETE',seconds=time.monotonic()-began,scenes=len(rows)))
    print('PREPARED',len(rows),'scenes; pitches -10/',pitch,flush=True)


def run():
    began=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'))
    spent=sum(json.loads(p.read_text(encoding='utf-8'))['seconds'] for p in OUT.glob('run_failure_*.json'))
    def check():
        if spent+time.monotonic()-began>plan['budgets_wall_seconds']['render_infer']:
            raise TimeoutError('Render/infer wall cap reached')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            if (ROOT/name).resolve()!=Path(__file__).resolve():
                raise ValueError('Frozen dependency changed: '+name)
    save(OUT/('execution_source_'+str(time.time_ns())+'.json'),dict(source_sha256=sha(__file__),previous_spent_seconds=spent))
    A,G,R,normalized_z=imports();A.setup_gpu()
    rows=plan['scene_rows'];parameters=[];engine=None
    try:
        engine=A.Engine()
        for pitch in (plan['baseline_pitch_deg'],plan['candidate_pitch_deg']):
            sensor,query=poses(pitch)
            renderer=G.ExpectedRenderer(sensor,plan['background'])
            try:
                ambient=renderer.ambient.copy();backend=renderer.metadata
                expectation=np.empty((len(rows),16,8,8,16),dtype=np.float64)
                target_photons=np.empty((len(rows),16),dtype=np.float64)
                for begin,ends in renderer.iter_render(rows,candidate_batch=4,pose_batch=16,deadline_check=check):
                    for j in range(len(ends)):
                        rho=rows[begin+j]['rho']
                        expectation[begin+j]=ends[j,0]+(rho/G.ENDPOINT_RHO)*(ends[j,1]-ends[j,0])
                        target_photons[begin+j]=((rho/G.ENDPOINT_RHO)*(ends[j,1]-ends[j,0])).sum((1,2,3))
                # Independent CPU frozen expectation at actual floor/wall/cube poses.
                references=[]
                for i in (0,len(rows)//2,len(rows)-1):
                    target=dict(lo=rows[i]['lo'],hi=rows[i]['hi'],rho=rows[i]['rho'])
                    ref=R.expected(dict(poses=sensor[[0,11,13,15]],boxes=[target,*plan['background']]))
                    diff=float(np.max(np.abs(ref['expectation']-expectation[i,[0,11,13,15]])))
                    np.testing.assert_allclose(ref['expectation'],expectation[i,[0,11,13,15]],atol=1e-8,rtol=1e-11)
                    references.append(dict(scene=i,max_abs=diff))
            finally:
                renderer.close()
            raw=np.empty((len(rows),K,13,2),np.float32)
            hist=np.empty((len(rows),K,16,8,8,16),np.int32)
            tick=time.monotonic()
            for begin in range(0,len(rows),4):
                check();z=[]
                for c in range(begin,min(begin+4,len(rows))):
                    for k in range(K):
                        seed=int(np.random.SeedSequence([2026100819,c,k]).generate_state(1)[0])
                        hist[c,k]=R.sample(expectation[c],ambient,seed)[0]
                        z.append(normalized_z(hist[c,k][None],ambient[None])[0])
                z=np.stack(z);N=len(z)
                pn=np.broadcast_to(sensor,(N,16,4,4));pq=np.broadcast_to(query,(N,16,4,4))
                fv=[engine.features(z,pn,pq,int(f)) for f in FRAMES]
                preds=engine.predict(np.stack(fv,1).reshape(N*13,3,24,17,33)).reshape(N,13,2)
                raw[begin:begin+N//K]=preds.reshape(N//K,K,13,2)
                if begin%24==0:
                    print('PITCH',pitch,'SCENES',min(begin+4,len(rows)),'/',len(rows),'wall',round(time.monotonic()-began,1),flush=True)
            path=OUT/'physical'/('pitch_'+str(pitch)+'.npz');path.parent.mkdir(exist_ok=True)
            if path.exists():raise FileExistsError('Preserve existing photons/scores')
            np.savez_compressed(path,raw=raw,hist=hist,ambient=ambient,expectation=expectation,
                                target_photons=target_photons,sensor=sensor,public_query=query)
            parameters.append(dict(pitch=pitch,backend=backend,renderer_parity=references,
                 infer_sample_seconds=time.monotonic()-tick,payload_sha256=sha(path),file=str(path.relative_to(OUT))))
            del hist,expectation
        check()
        save(OUT/'run_result.json',dict(status='COMPLETE',seconds=time.monotonic()-began,layouts=parameters))
    except BaseException as exc:
        save(OUT/('run_failure_'+str(time.time_ns())+'.json'),dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-began))
        raise
    finally:
        if engine is not None:
            engine.projector=None;engine.nets=[];engine.torch.cuda.empty_cache()


def smooth(raw):
    raw=np.asarray(raw,dtype=np.float64)
    values=np.empty_like(raw)
    for i in range(13):
        j=max(0,i-4);w=2.**np.arange(i-j+1)
        values[...,i,:]=(raw[...,j:i+1,:]*w[:,None]).sum(-2)/w.sum()
    return values


def truth(rows,background):
    # Same frozen all-object geometry, evaluator-only after photons/scores.
    from cnh_all_object_geometry import classify_boxes
    categories=[]
    for r in rows:
        target=dict(lo=r['lo'],hi=r['hi'],rho=r['rho'])
        categories.append([classify_boxes([target,*background],q,np.eye(4))['all_category'] for q in (0,1)])
    return np.array(categories)


def metrics(scores,category,threshold):
    alarms=scores>=threshold;joint=alarms.any(-1)
    contact=category=='contact';clear=(category=='clear').all(1)
    timely=alarms[:,:,:11].any(2) & contact[:,None,:]
    late=alarms[:,:,11:].any(2) & contact[:,None,:] & ~timely
    cc=joint[clear]
    seg=cc[:, :, 0].sum()+((~cc[:,:,:-1])&cc[:,:,1:]).sum()
    return dict(threshold=float(threshold),timely=timely,
      counts=[int(timely[...,q].sum()) for q in (0,1)],
      late_counts=[int(late[...,q].sum()) for q in (0,1)],
      clear_slots=int(cc.sum()),clear_denominator=int(cc.size),clear_segments=int(seg),
      clear_clips=int(cc.any(-1).sum()),
      physical_contact_any_height=int(joint[:,:,:11].any(2)[contact.any(1)].sum()),
      pass_clips=int(joint[(category=='pass').any(1)&~contact.any(1)].any(-1).sum()))


def serial_metrics(m):
    return {k:v for k,v in m.items() if k!='timely'}


def compare(a,b,category):
    result=[]
    for q in (0,1):
        mask=category[:,q]=='contact'
        aa=a['timely'][mask,:,q];bb=b['timely'][mask,:,q]
        gains=(~aa)&bb;losses=aa&~bb
        result.append(dict(height=('HEAD','BODY')[q],denominator=int(aa.size),
          baseline=int(aa.sum()),candidate=int(bb.sum()),gain=int(gains.sum()),loss=int(losses.sum()),
          net=int(bb.sum()-aa.sum()),
          gain_keys=[[int(np.flatnonzero(mask)[i]),int(k)] for i,k in np.argwhere(gains)],
          loss_keys=[[int(np.flatnonzero(mask)[i]),int(k)] for i,k in np.argwhere(losses)],
          contact_scene_ids=np.flatnonzero(mask).tolist()))
    return result


def analyze():
    began=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'))
    # Supplement the immutable PLAN before joining any evaluator labels/outcomes.
    imports()
    import cnh_all_object_geometry
    import cnh_corridor_labels
    evaluation_paths=[Path(m.__file__).resolve() for n,m in list(sys.modules.items())
                      if n.startswith('cnh_') and getattr(m,'__file__',None)]
    save(OUT/'evaluation_manifest.json',dict(stage='pre-truth-join supplement; PLAN unchanged',
         analysis_source_sha256=sha(__file__),
         source_sha256={logical_path(p):sha(p) for p in sorted(set(evaluation_paths),key=str)}))
    rr=json.loads((OUT/'run_result.json').read_text(encoding='utf-8'))
    arrays=[];scores=[]
    for l in rr['layouts']:
        p=OUT/l['file'];assert sha(p)==l['payload_sha256']
        with np.load(p) as z:
            arrays.append({k:z[k] for k in ('raw','target_photons')})
            scores.append(smooth(z['raw']))
    visibility=finite_visibility(plan)
    cat=truth(plan['scene_rows'],plan['background'])
    assert (cat=='contact').sum()==72
    assert ((cat=='clear').all(1)).sum()==48
    baseline=metrics(scores[0],cat,THETA);fixed=metrics(scores[1],cat,THETA)
    cc=(cat=='clear').all(1)
    v=scores[1][cc].max(-1).reshape(-1)
    thresholds=np.r_[np.inf,np.unique(v)]
    # Whole-tie counts without a huge [threshold,slot] allocation.
    ordered=np.sort(v);cost=len(v)-np.searchsorted(ordered,thresholds,side='left')
    ix=np.lexsort((-thresholds,np.abs(cost-baseline['clear_slots'])))[0]
    threshold=thresholds[ix]
    if not np.isfinite(threshold):threshold=np.nextafter(float(v.max()),np.inf)
    matched=metrics(scores[1],cat,float(threshold))
    residual=matched['clear_slots']-baseline['clear_slots']
    paired=compare(baseline,matched,cat)
    decision=residual==0 and all(p['net']>=0 and p['loss']<=2 for p in paired) and any(p['net']>0 for p in paired)
    result=dict(status='COMPLETE',lane=plan['lane'],scenes=len(cat),replicas=K,contact_events=72*K,
      fixed_theta=dict(baseline=serial_metrics(baseline),candidate=serial_metrics(fixed),paired=compare(baseline,fixed,cat)),
      matched_cost=dict(baseline=serial_metrics(baseline),candidate=serial_metrics(matched),residual=residual,paired=paired),
      visibility=visibility_summary(visibility,cat,baseline,matched),
      decision='CONTINUE_CANDIDATE' if decision else 'DO_NOT_ADVANCE_CURRENT_PITCH',
      workpoint_scope='Development evaluation clear-slot selection; not deployment calibration or confirmation',seconds=time.monotonic()-began)
    save(OUT/'result.json',result)
    np.savez_compressed(OUT/'evaluated.npz',category=cat,scores=np.stack(scores),first_target_rays=visibility,
                        baseline_timely=baseline['timely'],candidate_fixed_timely=fixed['timely'],candidate_matched_timely=matched['timely'])
    with (OUT/'model_map.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pitch_deg','scene','x_m','y_down_m','height_above_floor_m','cube_side_m','rho','frame','front_m','target_first_hit_rays','target_expected_counts','HEAD_category','BODY_category','HEAD_score','BODY_score','alarm_at_original_theta_replicas'])
        for arm,pitch in enumerate((plan['baseline_pitch_deg'],plan['candidate_pitch_deg'])):
            for i,r in enumerate(plan['scene_rows']):
                for j,fr in enumerate(FRAMES):
                    writer.writerow([pitch,i,r['x'],r['y'],1.65-r['y'],r['side'],r['rho'],fr,round(3.05-.16*fr,3),int(visibility[arm,i,fr]),arrays[arm]['target_photons'][i,fr],*cat[i],*scores[arm][i,:,j].mean(0),int((scores[arm][i,:,j].max(-1)>=THETA).sum())])
    plot(plan,result,scores,cat,visibility)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def finite_visibility(plan):
    import cnh_displacement_ceiling_render as R
    directions,_=R.S.angular_rays(16)
    directions=directions.reshape(-1,3)
    counts=np.zeros((2,len(plan['scene_rows']),16),np.int32)
    for arm,pitch in enumerate((plan['baseline_pitch_deg'],plan['candidate_pitch_deg'])):
        sensor,_=poses(pitch);world=directions@sensor[0,:3,:3].T
        for i,r in enumerate(plan['scene_rows']):
            boxes=[dict(lo=r['lo'],hi=r['hi'],rho=r['rho']),*plan['background']]
            for f in range(16):
                hit=R.S.raycast_boxes(sensor[f,:3,3],world,boxes)
                counts[arm,i,f]=np.count_nonzero(hit['object_id']==0)
    return counts


def visibility_summary(visibility,cat,baseline,candidate):
    out=[]
    for arm,m in enumerate((baseline,candidate)):
        seen=visibility[arm,:,:14].any(1)
        for q in (0,1):
            contact=cat[:,q]=='contact';t=m['timely'][contact,:,q];s=seen[contact]
            out.append(dict(arm=arm,height=('HEAD','BODY')[q],contact_scenes=int(contact.sum()),
              unseen_scenes=int((~s).sum()),seen_scenes=int(s.sum()),
              unseen_timely_events=int(t[~s].sum()),seen_timely_events=int(t[s].sum()),
              seen_missed_events=int((~t[s]).sum())))
    return out


def plot(plan,result,scores,cat,visibility):
    import os
    os.environ['MPLCONFIGDIR']=str(OUT/'runtime/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=plan['scene_rows']
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    theta=[THETA,result['matched_cost']['candidate']['threshold']]
    for size_index,side in enumerate(SIDES):
        for arm in range(2):
            ax=axs[size_index,arm];image=np.full((len(YS),len(XS)),np.nan)
            for i,r in enumerate(rows):
                if r['side']!=side or r['rho']!=.25:continue
                if (cat[i]=='contact').any():
                    q=int(np.flatnonzero(cat[i]=='contact')[0])
                    image[YS.index(r['y']),XS.index(r['x'])]=(scores[arm][i,:,:11,q]>=theta[arm]).any(1).mean()
            im=ax.imshow(image,vmin=0,vmax=1,cmap='RdYlGn',origin='upper',aspect='auto')
            ax.set_xticks(range(len(XS)),[str(x) for x in XS]);ax.set_yticks(range(len(YS)),[f'{1.65-y:.2f}' for y in YS])
            ax.set_xlabel('Lateral x (m); blank = pass/clear, see CSV');ax.set_ylabel('Center height above assumed floor (m)')
            ax.set_title(f"Pitch {(-10.,plan['candidate_pitch_deg'])[arm]:g} deg; cube {side*100:g} cm; rho .25")
            for yi in range(len(YS)):
                for xi in range(len(XS)):
                    if np.isfinite(image[yi,xi]):
                        i=next(i for i,r in enumerate(rows) if r['side']==side and r['rho']==.25 and r['y']==YS[yi] and r['x']==XS[xi])
                        unseen=not visibility[arm,i,:14].any()
                        ax.text(xi,yi,f'{round(image[yi,xi]*4)}/4'+('\nunseen' if unseen else ''),ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=axs,label='Correct target-height timely fraction, K4 photon replicates')
    fig.suptitle('Aligned straight / exact poses / frozen M3 / matched clear alarm-slot cost\nFinite synthetic cubes; no device or coverage guarantee')
    fig.savefig(OUT/'model_boundary_map.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run','analyze'))
    args=parser.parse_args();{'prepare':prepare,'run':run,'analyze':analyze}[args.stage]()

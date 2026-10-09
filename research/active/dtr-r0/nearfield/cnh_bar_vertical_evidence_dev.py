"""Vertical query-error evidence and public-support feature ablations; frozen M3."""
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
import cnh_bar_pose_transfer_dev as T

OUT=B.ROOT/'artifacts.local/work/cnh-bar-vertical-evidence-dev-20261009'
POSE=B.ROOT/'artifacts.local/work/cnh-bar-transfer-dev-20261009/pose'
ANGLES=(0,-3,3)
MODES=('full','inside','outside','zero_echo')
def save(name,x):B.save(OUT/name,x)
def name(a):return 'ideal' if a==0 else f'query_3_{int(np.sign(a)):+d}'
def transformed(full,weight):
    x=full.copy()
    x[:,0]=(full[:,0].astype(np.float32)*weight).astype(np.float16)
    x[:,2]=(full[:,2].astype(np.float32)*weight).astype(np.float16)
    return x
def prepare():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve vertical evidence PLAN')
    A,_,_,_=B.imports();rows=json.loads((S.OUT/'PLAN.json').read_text())['scene_rows']
    ids=np.array([i for i,r in enumerate(rows) if r['family']=='vertical']);assert len(ids)==132
    inputs=[S.OUT/'PLAN.json',S.OUT/'geometry.npz',S.OUT/'physical.npz',POSE/'PLAN.json',OUT/'observation/PLAN.json',OUT/'observation/mean_delta.npz',*[POSE/(name(a)+'.npz') for a in ANGLES],*A.M3_MODELS]
    sources=[Path(__file__),*[Path(m.__file__).resolve() for n,m in list(sys.modules.items()) if n.startswith('cnh_') and getattr(m,'__file__',None)]]
    save('PLAN.json',dict(task='CNH_BAR_VERTICAL_EVIDENCE_DEV_20261009',lane='EXPLORE consumed photons and feature intervention',base_commit='3590176b',
        authorization='User 继续 after recommendation to pair verticalclear hist angular/range support, projected query evidence and frozen M3 response',
        budgets_stage_wall_seconds=dict(prepare=180,shared_projection_M3_IO=900,analysis_projection_summary_checks=300,independent_audit=120),
        scope='All132 vertical scenes K4 full13output f3..15, three prescribed anglesideal/-3/+3; original truth/photons, network/thresholds/smoothing fixed.24strictclear,112heightcontacts acrossbothheights,24pass physical.',
        scene_ids=ids.tolist(),angles=list(ANGLES),modes=list(MODES),
        intervention='After original zFP16 -> CUDA FP32 projection -> featureFP16, keep cnt unchanged and multiply ONLY tot/lst inFP32 by unionmask/complement/0 thenFP16. Unionmask=clip(sum(two originalpublicquery_masks),0,1); signedlog/cnt8/publicmask processing and M3 untouched.',
        evidence='Oracle target mean delta from original present expectation minus matched absent cached background; normalize with original FP32den, project via sameactual FP32index/weight map. Save per-bin coefficients for HEAD/BODY and allpast8 frames, positive/negative mass separately. Not inferred physical/noise object removal and not an information-losschain.',
        storage='Per angle fulltot/lst[528,13,2,24,17,33] FP16 + sharedcnt[13,24,17,33]FP16, publicmasks andunion; all4 intervention feature recipes exactly reconstructible, raw logits[132,4,13,4,2], localoriginalscores retained. No need store3 duplicate feature tensors.',
        decision_check='Eachheight verticalcontact224events,24clear*K4*13=1248jointslots; all132 includescontact/pass retention. Fullraw bitwise oldcache required. Report fixedtheta costs and paired losses; ablation distribution shift cannot establish physicaltarget causality/deployment benefit. True labels onlyfor evaluation; no shape-dependent livegate.',
        adjustable_scope='Implementation repairs/focused checks and signal summaries, no mask/angle/model/threshold selection, newnoise/training/hardware or allshape policy promotion.',
        stop='Complete fixed3angle4mode queue or cumulativecap including failures; stop on actualfullparity failure until source repaired, preserve evidence. No extra ablation recipe or angle sweep.',
        inputs_sha256={B.logical_path(p):B.sha(p) for p in inputs},source_sha256={B.logical_path(p):B.sha(p) for p in sorted(set(sources),key=str)},
        ensemble_examples=132*4*13*3*4,prepare_seconds=time.monotonic()-began))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('PREPARED132 vertical/3angles/4modes',flush=True)
def run():
    began=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text());engine=None
    previous=sum(json.loads(x.read_text())['seconds'] for x in OUT.glob('failure_*.json'))
    def check():
        if previous+time.monotonic()-began>900:raise TimeoutError('cumulative run900s')
    try:
        for p,h in {**plan['inputs_sha256'],**plan['source_sha256']}.items():assert B.sha(B.ROOT/p)==h,p
        A,_,_,normal=B.imports();A.OUT=OUT/'runtime';engine=A.Engine()
        from cnh_cvr_projection import query_masks,SHAPE
        from cnh_temporal_readout_data import D
        ids=np.array(plan['scene_ids']);count=len(ids)*4
        masks=query_masks();union=np.clip(masks.sum(0),0,1).astype(np.float32)
        np.savez_compressed(OUT/'masks.npz',query_masks=masks,union=union,complement=1-union)
        with np.load(S.OUT/'geometry.npz') as d:sensor=d['sensor'];q=d['public_query']
        with np.load(S.OUT/'physical.npz') as d:hist=d['hist'][ids];ambient=d['ambient']
        z=normal(hist.reshape(count,16,8,8,16),np.broadcast_to(ambient,(count,*ambient.shape)))
        with np.load(OUT/'observation/mean_delta.npz') as d:delta=d['delta'];clearids=d['scene_ids'];obsambient=d['ambient']
        np.testing.assert_array_equal(ambient,obsambient)
        bias=np.load(D.BIAS).astype(np.float32);den=np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9))
        delta_z=delta/den[None]
        oracle=[];coefficients=[];runrows=[]
        for a in plan['angles']:
            angle_t=time.monotonic();v=dict(kind='ideal' if a==0 else 'query',degrees=abs(a),sign=int(np.sign(a)))
            estimated,goal=T.injected(sensor,q,v)
            with np.load(POSE/(name(a)+'.npz')) as d:oldraw=d['m3_raw'][ids];localraw=d['local_raw'][ids]
            echo=np.empty((count,13,2,*SHAPE),np.float16);cnt=np.empty((13,*SHAPE),np.float16)
            logits=np.empty((count,13,4,2),np.float32)
            for j,f in enumerate(B.FRAMES):
                check();start=max(0,int(f)-7);m=(goal[f]@np.linalg.inv(estimated[f]))[None]@estimated[start:f+1]
                cp=L.Projection(engine,m);cnt[j]=cp.count.cpu().numpy().astype(np.float16)
                # Conditional linear mean contrasts pre-feature-rounding; exact map weight, not signedlog.
                coeff=np.stack([np.asarray(mask.reshape(1,-1)@cp.p).ravel().reshape(len(m),8,8,16) for mask in masks])
                coefficients.append(dict(angle=a,frame=int(f),start=start,coefficient=coeff))
                dz=delta_z[:,start:f+1]
                for qi in (0,1):
                    pos=np.einsum('tlrb,stlrb->s',coeff[qi],np.maximum(dz,0));neg=np.einsum('tlrb,stlrb->s',coeff[qi],np.minimum(dz,0))
                    for si in range(len(clearids)):oracle.append(dict(angle=a,frame=int(f),scene=int(clearids[si]),height=('HEAD','BODY')[qi],positive_tot_mean=float(pos[si]),negative_tot_mean=float(neg[si]),net_tot_mean=float(pos[si]+neg[si])))
                for b in range(0,count,32):
                    check();full=cp(z[b:b+32,start:f+1]).cpu().numpy().astype(np.float16)
                    echo[b:b+len(full),j]=full[:,[0,2]]
                    np.testing.assert_array_equal(full[:,1],np.broadcast_to(cnt[j],full[:,1].shape))
                    for h,mode in enumerate(MODES):
                        xx=full if mode=='full' else transformed(full,union if mode=='inside' else 1-union if mode=='outside' else np.zeros_like(union))
                        np.testing.assert_array_equal(xx[:,1],full[:,1]);logits[b:b+len(full),j,h]=engine.predict(xx)
                np.testing.assert_array_equal(logits[:,j,0].reshape(132,4,2),oldraw[:,:,j])
            payload=OUT/f'angle_{a:+d}.npz'
            if payload.exists():raise FileExistsError('Preserve angle payload')
            np.savez_compressed(payload,scene_ids=ids,echo=echo,count=cnt,logits=logits.reshape(132,4,13,4,2),local_raw=localraw)
            runrows.append(dict(angle=a,path=payload.name,sha256=B.sha(payload),seconds=time.monotonic()-angle_t,cumulative_seconds=time.monotonic()-began))
            print('ANGLE',a,'complete',round(time.monotonic()-began,2),flush=True)
        with (OUT/'projected_mean.csv').open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(oracle[0]));w.writeheader();w.writerows(oracle)
        # Pad earlywindows to fixed8 only in archive, explicit actual length/start retained.
        coeff_array=np.zeros((3,13,2,8,8,8,16));length=np.empty((3,13),int);starts=np.empty_like(length)
        for ix,r in enumerate(coefficients):ia=ix//13;j=ix%13;c=r['coefficient'];coeff_array[ia,j,:,:len(c[0])]=c;length[ia,j]=len(c[0]);starts[ia,j]=r['start']
        np.savez_compressed(OUT/'mean_projection.npz',angles=plan['angles'],frames=B.FRAMES,coefficients=coeff_array,length=length,start=starts,delta_z=delta_z,clear_scene_ids=clearids,denominator=den)
        check();save('result_run.json',dict(status='COMPLETE',seconds=time.monotonic()-began,angles=runrows,full_cached_parity='BITWISE all132xK4x13x3x2 logits',
            backend='CUDA original FP32 mapped projection/5 frozenmodels; interventionFP32multiply->FP16 before originals signedlog',device=engine.torch.cuda.get_device_name(),ensemble_examples=82368,training=0,new_photons=0))
    except BaseException as e:save('failure_'+str(time.time_ns())+'.json',dict(seconds=time.monotonic()-began,error=repr(e)));raise
    finally:
        if engine is not None:engine.nets=[];engine.projector=None;engine.torch.cuda.empty_cache()
def analyze():
    t=time.monotonic();run=json.loads((OUT/'result_run.json').read_text());ids=np.array(json.loads((OUT/'PLAN.json').read_text())['scene_ids'])
    with np.load(S.OUT/'geometry.npz') as d:cat=d['category'][ids]
    reports=[];ledger=[]
    for row in run['angles']:
        with np.load(OUT/row['path']) as d:raw=d['logits'];local=F.smooth(d['local_raw'])
        full=F.smooth(raw[:,:,:,0]);fullm,fullt=F.metrics(full>=T.M3_THETA,cat)
        for mi,mode in enumerate(MODES):
            score=F.smooth(raw[:,:,:,mi]);mm,mt=F.metrics(score>=T.M3_THETA,cat);fusion=(score>=T.FUSION_M3)|(local>=T.FUSION_LOCAL);fm,ft=F.metrics(fusion,cat)
            reports.append(dict(angle=row['angle'],mode=mode,M3=mm,fusion=fm,M3_vs_full=F.compare(fullt,mt,cat,ids)))
            for i,k,q in np.argwhere(np.broadcast_to((cat=='contact')[:,None,:],mt.shape)):
                ledger.append(dict(angle=row['angle'],mode=mode,scene=int(ids[i]),replica=int(k),height=('HEAD','BODY')[q],M3=int(mt[i,k,q]),fusion=int(ft[i,k,q]),full_M3=int(fullt[i,k,q]),gain=int(not fullt[i,k,q] and mt[i,k,q]),loss=int(fullt[i,k,q] and not mt[i,k,q])))
    with (OUT/'event_ledger.csv').open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    save('result_analysis.json',dict(status='COMPLETE',seconds=time.monotonic()-t,summary=reports,event_rows=len(ledger),denominators=dict(height_contact=[int((cat[:,q]=='contact').sum()*4) for q in (0,1)],jointclear=int((cat=='clear').all(1).sum()*4*13)),
        limitation='Public-support feature ablations preserve cnt andallnet/contracts but shift inputdistribution; no physical removal, localvariance diagnosis, fullshape policy gain or matchedcost conclusion. Conditional projectedmean usesprivileged knownbackground delta, not onlineobservable target attribution.'))
    for r in reports:print(r['angle'],r['mode'],'M3',r['M3']['counts'],'clear',r['M3']['clear_slots'],'fusion',r['fusion']['counts'],r['fusion']['clear_slots'],flush=True)
    assert time.monotonic()-t<300
if __name__=='__main__':globals()[sys.argv[1]]()

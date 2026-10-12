"""Fixed public-query horizontal patch scan on consumed Development photons."""
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S
import cnh_bar_projection_diagnostic_dev as P
import cnh_bar_readout_diagnostic_dev as R
from cnh_bar_cached_diagnostic_dev import matched_threshold

OUT=B.ROOT/'artifacts.local/work/cnh-bar-local-readout-dev-20261009'


def patches():
    from cnh_cvr_projection import query_masks, SHAPE, grid
    masks=query_masks();centers,_=grid();matrices=[];locations=[]
    for q in range(2):
        rows=[];cols=[];values=[];xyz=[]
        for x in range(SHAPE[0]-5):
            for y in range(SHAPE[1]):
                for z in range(SHAPE[2]-2):
                    ix=np.arange(np.prod(SHAPE)).reshape(SHAPE)[x:x+6,y:y+1,z:z+3].ravel()
                    v=masks[q].ravel()[ix]
                    # Each voxel intersects query; fractional boundary cells are clipped by mask.
                    if np.count_nonzero(v)!=18:continue
                    rows.extend([len(xyz)]*18);cols.extend(ix);values.extend(v)
                    xyz.append(centers[x:x+6,y:y+1,z:z+3].mean((0,1,2)).tolist())
        matrices.append(sparse.csr_matrix((values,(rows,cols)),shape=(len(xyz),np.prod(SHAPE))))
        locations.append(np.array(xyz))
    return matrices,locations


def prepare():
    A,_,_,_=B.imports()
    sources=[Path(__file__),Path(B.__file__),Path(S.__file__),Path(P.__file__),Path(R.__file__),
             *[Path(m.__file__).resolve() for n,m in list(sys.modules.items())
               if n.startswith('cnh_') and getattr(m,'__file__',None)]]
    inputs=[S.OUT/'PLAN.json',S.OUT/'physical.npz',S.OUT/'geometry.npz',S.OUT/'evaluated.npz',
            R.OUT/'PLAN.json',R.OUT/'draws.npy',R.OUT/'predictions.npz',*A.M3_MODELS]
    B.save(OUT/'PLAN.json',dict(task='CNH_BAR_LOCAL_READOUT_DEV_20261009',lane='EXPLORE consumed simulation Development',
        authorization='User continued after frozen M3 readout diagnosis',base_commit='9e7682b0',
        goal='Test one fixed local angular-range readout at the original clear alarm-slot cost and retain every paired loss',
        budgets_wall_seconds=dict(run=900,analysis=180,audit=180),
        mechanism='Fixed6x1x3 voxel patches (.30x.10x.30m); all18 voxel overlaps inside public query. Weighted sum of FP16 tot, scaled by norm(Ptot.T w). Maximum across all patches, no side or target geometry selection. Reference input-z unit variance scaling, not conditional d/noise estimate. Original z FP16, original CUDA FP32 projection then FP16 voxel, no signedlog/network. Original last5 exponential smoothing.',
        adjustable_scope='Implementation repairs and checks; one patch shape, no training/pitch/layout/M3 changes or patch/threshold outcome tuning',
        primary='All492 original scenes K4, f3..15, timely f3..13; candidate whole-score ties nearest to baseline46/4576 jointly-clear slots. Shared threshold for both queries; residual reported. Calibration uses consumed Development clear only; not deployment calibration.',
        supplementary='All37 saved conditional endpoints x64 draws, complete f3..15; no new noise. Original MC M3 and candidate both evaluated at fixed primary thresholds; MC clear cost reported, never asserted equal.',
        decision_check='HEAD/BODY contact denominators688each, dark4cm bars56each, original dark HEAD8/BODY1. One event=1/688 main,1/56 dark. Prefer further investigation only if dark BODY rescue is repeatable without broad net loss; record both query gains/losses and pass burden. No formal promotion or loss guardrail imported from other recipes.',
        stop='Finish fixed candidate comparison or cumulative budget; no patch sweep/extra draws. Failures and partial outputs retained.',
        inputs_sha256={B.logical_path(p):B.sha(p) for p in inputs},
        source_sha256={B.logical_path(p):B.sha(p) for p in sorted(set(sources),key=str)},
        deliverables='Original K4 and saved MC score arrays, paired event ledger and grouped counts/costs, focused validation, report/current/RUNS/commit/push'))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('PREPARED',flush=True)


class Projection:
    """Same original FP32 gather and reduction order, any past<=8 length."""
    def __init__(self,engine,m):
        from cnh_cvr_projection import SUB,SHAPE
        self.torch=engine.torch;self.shape=SHAPE;self.sub=SUB;self.length=len(m)
        self.p,_,index,weights=P.map_window(m,engine.projector,engine.torch)
        self.index=engine.torch.as_tensor(index,device='cuda')
        self.weight=engine.torch.as_tensor(weights,device='cuda')
        # Count equals the fraction of valid subpoints, independent of observations.
        self.count=(self.weight!=0).reshape(len(m),-1,SUB**3).float().mean(2).sum(0).reshape(SHAPE)

    def __call__(self,z):
        t=self.torch;c=len(z)
        val=t.as_tensor(np.ascontiguousarray(z),device='cuda',dtype=t.float32).reshape(c,self.length,1024)
        mass=t.gather(val,2,self.index[None].expand(c,-1,-1))*self.weight[None]
        ev=mass.reshape(c,self.length,-1,self.sub**3).sum(3).reshape(c,self.length,*self.shape)
        return t.stack((ev.sum(1),self.count[None].expand(c,-1,-1,-1),ev[:,-1]),1)


def scan(feature,qs,norms,locations):
    total=feature[:,0].astype(np.float64).reshape(len(feature),-1)
    scores=[];argmax=[]
    for q,scale,xyz in zip(qs,norms,locations):
        value=np.asarray(q@total.T).T/scale[None]
        value[:,~np.isfinite(scale)]=-np.inf
        ix=value.argmax(1);scores.append(value[np.arange(len(value)),ix]);argmax.append(xyz[ix])
    return np.stack(scores,-1),np.stack(argmax,1)


def run():
    began=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text())
    previous=sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('failure_*.json'))
    def check():
        if previous+time.monotonic()-began>plan['budgets_wall_seconds']['run']:raise TimeoutError('cumulative900s')
    engine=None
    try:
        for name,digest in {**plan['inputs_sha256'],**plan['source_sha256']}.items():
            assert B.sha(B.ROOT/name)==digest,('changed frozen input',name)
        A,_,_,normal=B.imports();A.OUT=OUT/'runtime';engine=A.Engine()
        qs,locations=patches();maps=[];norms=[];parities=[]
        with np.load(S.OUT/'geometry.npz') as d:sensor=d['sensor'];query=d['public_query']
        with np.load(S.OUT/'physical.npz') as d:hist=d['hist'];ambient=d['ambient']
        z=normal(hist.reshape(-1,16,8,8,16),np.broadcast_to(ambient,(hist.size//(16*1024),*ambient.shape)))
        for f in B.FRAMES:
            check();start=max(0,int(f)-7)
            m=(query[f]@np.linalg.inv(sensor[f]))[None]@sensor[start:f+1]
            cp=Projection(engine,m);maps.append(cp)
            nn=[]
            for q in qs:
                qp=q@cp.p;var=np.asarray(qp.multiply(qp).sum(1)).ravel()
                # Unobserved patches cannot alarm. Keep fixed patch set, mark norm inf.
                nn.append(np.where(var>0,np.sqrt(var),np.inf))
            norms.append(nn);sparse.save_npz(OUT/f'map_f{f}.npz',cp.p)
            if f in (3,12,13,15):
                feature=cp(z[:1,start:f+1]);ref=engine.project(z[:1,start:f+1],m[None])
                np.testing.assert_array_equal(feature.cpu().numpy(),ref.cpu().numpy())
                parities.append(dict(frame=int(f),max_abs=float((feature-ref).abs().max())))
        def infer(values,baseline):
            raw=np.empty((len(values),13,2));xyz=np.empty((len(values),13,2,3))
            m3=np.empty((len(values),13,2),np.float32) if baseline else None
            for j,f in enumerate(B.FRAMES):
                start=max(0,int(f)-7)
                for b in range(0,len(values),32):
                    check();feat=maps[j](values[b:b+32,start:f+1]).cpu().numpy().astype(np.float16)
                    raw[b:b+32,j],xyz[b:b+32,j]=scan(feat,qs,norms[j],locations)
                    if baseline:m3[b:b+32,j]=engine.predict(feat)
                print('FRAME',int(f),'samples',len(values),'M3',baseline,'seconds',round(time.monotonic()-began,2),flush=True)
            return raw,xyz,m3
        raw,xyz,_=infer(z,False)
        np.savez_compressed(OUT/'cached_scores.npz',raw=raw.reshape(492,4,13,2),argmax_xyz=xyz.reshape(492,4,13,2,3))
        del z,hist
        draws=np.load(R.OUT/'draws.npy',mmap_mode='r')
        with np.load(R.OUT/'predictions.npz') as d:ids=d['endpoint_ids'];ref_scores=d['mc_scores'][:,:,1]
        vals=normal(draws.reshape(-1,16,8,8,16),np.broadcast_to(ambient,(37*64,*ambient.shape)))
        raw,xyz,m3=infer(vals,True)
        m3=m3.reshape(37,64,13,2)
        reference=ref_scores.transpose(0,2,1,3)
        np.testing.assert_array_equal(m3[:,:,[9,10]],reference)
        np.savez_compressed(OUT/'mc_scores.npz',endpoint_ids=ids,raw=raw.reshape(37,64,13,2),m3_raw=m3,argmax_xyz=xyz.reshape(37,64,13,2,3))
        B.save(OUT/'result_run.json',dict(status='COMPLETE',seconds=time.monotonic()-began,backend='Original CUDA FP32 projection/M3; CPU sparse patch sums/statistics',device=engine.torch.cuda.get_device_name(),patch_counts=[len(x) for x in locations],projection_parity=parities,MC_prior_parity_max_abs=0,MC_ensemble_examples=37*64*13,new_draws=0,training=0))
    except BaseException as exc:
        B.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(exc),seconds=time.monotonic()-began));raise
    finally:
        if engine is not None:engine.nets=[];engine.projector=None;engine.torch.cuda.empty_cache()


def analyze():
    began=time.monotonic();plan=json.loads((S.OUT/'PLAN.json').read_text());rows=plan['scene_rows']
    with np.load(S.OUT/'evaluated.npz') as z:cat=z['category'];baseline=z['scores']
    with np.load(OUT/'cached_scores.npz') as z:raw=z['raw']
    scores=B.smooth(raw);old=B.metrics(baseline,cat,B.THETA)
    theta=matched_threshold(scores,cat,old['clear_slots']);new=B.metrics(scores,cat,theta)
    # Derive dark4cm IDs from the frozen diagnostic plan; scene naming is not a model input.
    dark=np.zeros(len(rows),bool)
    fp=json.loads((R.REP/'fp16/PLAN.json').read_text());dark[fp['contact_scene_ids']]=True
    darkpair=B.compare(B.metrics(baseline[dark],cat[dark],B.THETA),B.metrics(scores[dark],cat[dark],theta),cat[dark])
    grouped=[]
    for family in S.FAMILIES:
        mask=np.array([r['family']==family for r in rows]);a=B.metrics(baseline[mask],cat[mask],B.THETA);b=B.metrics(scores[mask],cat[mask],theta)
        grouped.append(dict(family=family,comparison=B.compare(a,b,cat[mask]),candidate_cost=B.serial_metrics(b)))
    ledger=[]
    for i,k,q in np.argwhere(np.broadcast_to((cat=='contact')[:,None,:],old['timely'].shape)):
        aa=np.flatnonzero(baseline[i,k,:11,q]>=B.THETA);bb=np.flatnonzero(scores[i,k,:11,q]>=theta)
        ledger.append(dict(scene=int(i),replica=int(k),height=['HEAD','BODY'][q],family=rows[i]['family'],dark4cm=int(dark[i]),baseline=int(len(aa)>0),candidate=int(len(bb)>0),gain=int(len(aa)==0 and len(bb)>0),loss=int(len(aa)>0 and len(bb)==0),baseline_first_frame=int(B.FRAMES[aa[0]]) if len(aa) else '',candidate_first_frame=int(B.FRAMES[bb[0]]) if len(bb) else ''))
    with (OUT/'event_ledger.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    with np.load(OUT/'mc_scores.npz') as z:ids=z['endpoint_ids'];ms=B.smooth(z['raw']);mb=B.smooth(z['m3_raw'])
    mc_cat=np.concatenate((cat[ids[:-1]],np.array([['clear','clear']])))
    ma=B.metrics(mb,mc_cat,B.THETA);mn=B.metrics(ms,mc_cat,theta)
    mc_pairs=B.compare(ma,mn,mc_cat)
    for p in mc_pairs:
        # Keys above index endpoints, not original scene IDs; preserve both explicitly.
        p['gain_scene_keys']=[[int(ids[i]),k] for i,k in p['gain_keys']]
        p['loss_scene_keys']=[[int(ids[i]),k] for i,k in p['loss_keys']]
    result=dict(status='COMPLETE',seconds=time.monotonic()-began,baseline=B.serial_metrics(old),candidate=B.serial_metrics(new),cost_residual=new['clear_slots']-old['clear_slots'],paired=B.compare(old,new,cat),dark4cm=darkpair,groups=grouped,MC_baseline=B.serial_metrics(ma),MC_candidate=B.serial_metrics(mn),MC_paired=mc_pairs,decision='RETAIN_M3; local patch candidate evaluated as consumed Development, no model/policy promotion')
    assert time.monotonic()-began<180
    B.save(OUT/'result_analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','paired','MC_paired','dark4cm')},ensure_ascii=False),flush=True)
    print('PAIRED',[(p['height'],p['baseline'],p['candidate'],p['gain'],p['loss']) for p in result['paired']],flush=True)
    print('DARK',[(p['height'],p['baseline'],p['candidate'],p['gain'],p['loss']) for p in darkpair],flush=True)
    print('MC',[(p['height'],p['baseline'],p['candidate'],p['gain'],p['loss']) for p in mc_pairs],flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()

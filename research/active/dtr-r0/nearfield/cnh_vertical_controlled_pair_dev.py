"""Fixed-background Poisson coupling on previously selected shallow bar pairs."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'artifacts.local/work'
OUT=WORK/'cnh-vertical-controlled-pair-dev-20261009'
PREVIOUS=WORK/'cnh-vertical-visible-support-dev-20261009'
SHAPES=WORK/'cnh-aligned-shapes-dev-20261008'
FEATURES=('history_abs_share','current_inner_positive_share','total_density','current_scaled_mass')
PREFIX=2026100991

def save(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False,indent=2)

def csvread(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def csvsave(name,rows):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def seed(group,k,role,arm=0):
    return int(np.random.SeedSequence([PREFIX,group,k,role,arm]).generate_state(1)[0])

def prepare():
    t=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    rr=csvread(PREVIOUS/'near_all_pairs.csv')
    pairs=sorted({(int(r['angle']),r['relation'],r['depth'],int(r['clear_scene']),int(r['contact_scene']),r['height']) for r in rr})
    rows=json.loads((SHAPES/'PLAN.json').read_text(encoding='utf-8'))['scene_rows']
    ids=sorted({s for p in pairs for s in p[3:5]});groups=defaultdict(list)
    for s in ids:
        r=rows[s];assert r['family']=='vertical' and r['context']=='plain'
        assert r['background']==rows[ids[0]]['background']
        groups[(round((r['hi'][0]-r['lo'][0])*100),r['rho'],r['group'],r['variant'])].append(s)
    inputs=[SHAPES/'PLAN.json',SHAPES/'physical.npz',SHAPES/'geometry.npz',PREVIOUS/'PLAN.json',PREVIOUS/'near_all_pairs.csv']
    models=[WORK/f'cnh-margin-labels-20261002/models/M3/model_seed{s}.pt' for s in range(5)]
    save('PLAN.json',dict(task='CNH_VERTICAL_CONTROLLED_PAIR_DEV_20261009',lane='EXPLORE selected consumed Development geometry, new coupled draws',
        base_commit='1a8b3e8a',authorization='User 继续 accepts preceding controlled pairing proposal; routine bounded caps stated before execution',
        budgets_stage_wall_seconds=dict(cpu_prepare_sample_analyze_audit=180,gpu_projection_frozen_M3_IO=240),preliminary_cpu_command_seconds=6.9748011,
        adjustable_scope='Implementation repair and focused audit only; no extra condition, seed, model, threshold, angle or readout selection',
        goal='Check selected near current scaled mass higher and positive inner fraction lower under same world background and explicit paired nuisance coupling; not solve far clear cost',
        pairs=[dict(angle=a,relation=r,depth=d,clear_scene=c,contact_scene=s,height=h) for a,r,d,c,s,h in pairs],
        groups=[dict(key=list(key),scene_ids=value) for key,value in sorted(groups.items())],scene_ids=ids,K=4,angles=[-3,3],output_frames=list(range(3,14)),evaluation_frames=list(range(10,14)),
        coupling='Each group/K: B~Pois(8A), C~Pois(min_i(E_i+8A)), R_i~Pois(E_i+8A-minlambda), independent role/arm seed streams; hist_i=C+R_i-B. B,C shared across arms. Each arm exactly Pois(E_i+8A)-Pois(8A) marginal, count independent of subtract B. Artificial statistical coupling, not common physical photons.',
        background='Reuse whole original per-scene E including target occlusion. Fixed background boxes/rho and true sensor trajectory; visible background can change with target. No expectation rendering or physical no-bar subtraction.',
        contract='Identical normalization/FP16 projection/public inner-outer masks and frozen M3; evaluator shape/side/labels only define retained pairs, never features. raw includes past8; last5 unchanged. Current signed sum / cumulative count is not single-frame density.',
        prediction='Contact-clear current_scaled_mass positive; current_inner_positive_share negative. Means/medians and condition/frame/K distributions all reported. Expected-current linear mass difference also reported as deterministic conditional reference, not expected nonlinear ratio.',
        decision_check='8 selected height conditions x4 near frames xK4=128 Kmatched height-window pairs per angle/relation/depth, 1024 total; clear joint denominator differs when BOTH heights share a scene. No numeric success threshold, dprime bound or costmatched detector trial. Mixed or missing groups reported; mirror consistency neither necessary nor sufficient. Outcome guides mechanism selection only, no policy promotion.',
        stop='Complete fixed queue and focused check or stage cumulative cap including failures; no extra draws/seed search. Preserve failed/partial outputs.',
        inputs_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs+models},source_sha256=sha(Path(__file__)),prepare_seconds=time.monotonic()-t))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('PREPARED',len(ids),'scenes',len(pairs),'height pairs',len(groups),'groups',flush=True)

def cpu_previous():
    plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'))
    seconds=plan['preliminary_cpu_command_seconds']+plan['prepare_seconds']
    for name in ('sample_result.json','analysis_result.json'):
        if (OUT/name).exists():seconds+=json.loads((OUT/name).read_text())['seconds']
    seconds+=sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('cpu_failure_*.json'))
    return seconds

def sample():
    t=time.monotonic();p=json.loads((OUT/'PLAN.json').read_text());previous=cpu_previous()
    try:
        for path,h in p['inputs_sha256'].items():assert sha(ROOT/path)==h,path
        with np.load(SHAPES/'physical.npz') as z:E=z['expectation'][p['scene_ids']];A=z['ambient']
        ids=p['scene_ids'];ix={s:i for i,s in enumerate(ids)};hist=np.empty((len(ids),4,16,8,8,16),np.int32)
        count=np.empty_like(hist);residual=np.empty_like(hist)
        bg=np.empty((len(p['groups']),4,16,8,8,16),np.int32);common=np.empty_like(bg);rates=[]
        background=np.broadcast_to(8*A[...,None],E.shape[1:])
        for g,group in enumerate(p['groups']):
            gi=[ix[s] for s in group['scene_ids']];lam=E[gi]+background;mu=lam.min(0);rates.append(mu)
            for k in range(4):
                bg[g,k]=np.random.default_rng(seed(g,k,0)).poisson(background)
                common[g,k]=np.random.default_rng(seed(g,k,1)).poisson(mu)
                for arm,i in enumerate(gi):
                    residual[i,k]=np.random.default_rng(seed(g,k,2,arm)).poisson(lam[arm]-mu)
                    count[i,k]=common[g,k]+residual[i,k];hist[i,k]=count[i,k]-bg[g,k]
            if previous+time.monotonic()-t>180:raise TimeoutError('CPU cumulative180s')
        np.savez_compressed(OUT/'coupled_physical.npz',scene_ids=ids,expectation=E,ambient=A,hist=hist,count=count,residual=residual,subtract_background=bg,common_count=common,common_rate=rates)
        save('sample_result.json',dict(status='COMPLETE',seconds=time.monotonic()-t,scenes=len(ids),replicas=4,coarse_histogram_bins=int(hist.size),new_sequence_draws=len(ids)*4,new_expectation_render=0,seed_prefix=PREFIX,payload_sha256=sha(OUT/'coupled_physical.npz')))
        print('SAMPLED',len(ids)*4,'new sequences',round(time.monotonic()-t,3),flush=True)
    except BaseException as e:save('cpu_failure_'+str(time.time_ns())+'.json',dict(seconds=time.monotonic()-t,error=repr(e)));raise

def run():
    t=time.monotonic();p=json.loads((OUT/'PLAN.json').read_text());engine=None
    previous=sum(json.loads(x.read_text())['seconds'] for x in OUT.glob('gpu_failure_*.json'))
    def check():
        if previous+time.monotonic()-t>240:raise TimeoutError('GPU run cumulative240s')
    try:
        import cnh_aligned_boundary_dev as B
        import cnh_bar_local_readout_dev as L
        import cnh_bar_pose_transfer_dev as T
        A,_,_,normal=B.imports();A.OUT=OUT/'runtime';engine=A.Engine()
        from cnh_cvr_projection import SHAPE,query_masks
        with np.load(OUT/'coupled_physical.npz') as z:ids=z['scene_ids'];hist=z['hist'];ambient=z['ambient'];E=z['expectation']
        with np.load(SHAPES/'geometry.npz') as z:sensor=z['sensor'];q=z['public_query']
        z=normal(hist.reshape(-1,16,8,8,16),np.broadcast_to(ambient,(len(ids)*4,*ambient.shape)))
        from cnh_temporal_readout_data import D
        bias=np.load(D.BIAS).astype(np.float32);den=np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9))
        ez=(E-bias[None])/den[None]
        inner=query_masks().astype(float);outer=np.zeros_like(inner);yz=inner.sum(1)/12;outer[:,:6]=yz[:,None];outer[:,18:]=yz[:,None]
        masks=np.stack([inner[0],outer[0],inner[1],outer[1]]).reshape(4,-1)
        values=np.empty((2,len(ids),4,11,2,4));raw=np.empty((2,len(ids),4,11,2),np.float32)
        mean_current=np.empty((2,len(ids),11,2));parity=[]
        for ai,a in enumerate(p['angles']):
            estimated,goal=T.injected(sensor,q,dict(kind='query',degrees=3,sign=int(np.sign(a))))
            for j,f in enumerate(p['output_frames']):
                check();start=max(0,f-7);m=(goal[f]@np.linalg.inv(estimated[f]))[None]@estimated[start:f+1]
                cp=L.Projection(engine,m);feature=cp(z[:,start:f+1]).cpu().numpy().astype(np.float16)
                if j==0:
                    ref=engine.project(z[:1,start:f+1],m[None]).cpu().numpy().astype(np.float16)
                    np.testing.assert_array_equal(feature[:1],ref);parity.append(dict(angle=a,FP16_first_record_bitwise=True))
                raw[ai,:,:,j]=engine.predict(feature).reshape(len(ids),4,2)
                total=feature[:,0].astype(float).reshape(-1,np.prod(SHAPE));current=feature[:,2].astype(float).reshape(-1,np.prod(SHAPE))
                coverage=masks@feature[0,1].astype(float).ravel();ts=total@masks.T;cs=current@masks.T
                ha=np.abs(total-current)@masks.T;ca=np.abs(current)@masks.T;cpv=np.maximum(current,0)@masks.T
                for qi in (0,1):
                    inside,outside=2*qi,2*qi+1
                    with np.errstate(divide='ignore',invalid='ignore'):
                        vv=np.stack([ha[:,inside]/(ha[:,inside]+ca[:,inside]),cpv[:,inside]/(cpv[:,inside]+cpv[:,outside]),ts[:,inside]/coverage[inside],cs[:,inside]/coverage[inside]],1)
                    values[ai,:,:,j,qi]=vv.reshape(len(ids),4,4)
                    # Last exposure block only, FP32 map coefficients with FP64 arithmetic.
                    coefficient=np.asarray(masks[inside:inside+1]@cp.p[:,-1024:]).ravel()
                    mean_current[ai,:,j,qi]=(ez[:,f].reshape(len(ids),1024)@coefficient)/coverage[inside]
                del feature,total,current,cp
            print('PROJECTED',a,round(time.monotonic()-t,2),flush=True)
        np.savez_compressed(OUT/'responses.npz',scene_ids=ids,values=values,raw=raw,mean_current=mean_current)
        check();save('run_result.json',dict(status='COMPLETE',seconds=time.monotonic()-t,device=engine.torch.cuda.get_device_name(),examples=len(ids)*4*11*2,projection_parity=parity,new_inference=True,training=0))
    except BaseException as e:save('gpu_failure_'+str(time.time_ns())+'.json',dict(seconds=time.monotonic()-t,error=repr(e)));raise
    finally:
        if engine is not None:engine.nets=[];engine.projector=None;engine.torch.cuda.empty_cache()

def analyze():
    t=time.monotonic();previous=cpu_previous();p=json.loads((OUT/'PLAN.json').read_text());rr=[]
    with np.load(OUT/'responses.npz') as z:ids=z['scene_ids'];values=z['values'];raw=z['raw'];mean=z['mean_current']
    ix={int(s):i for i,s in enumerate(ids)}
    smooth=np.empty_like(raw,dtype=float)
    for j in range(raw.shape[3]):
        start=max(0,j-4);weights=2.**np.arange(j-start+1)
        smooth[:,:,:,j,:]=(raw[:,:,:,start:j+1,:].astype(float)*weights[:,None]).sum(-2)/weights.sum()
    for pair in p['pairs']:
        ai=p['angles'].index(pair['angle']);ci=ix[pair['clear_scene']];pi=ix[pair['contact_scene']];qi=0 if pair['height']=='HEAD' else 1
        for f in p['evaluation_frames']:
            j=f-3
            for k in range(4):
                r=dict(**pair,frame=f,replica=k,clear_M3_alarm=int(smooth[ai,ci,k,j,qi]>=.8557642486787612),contact_M3_alarm=int(smooth[ai,pi,k,j,qi]>=.8557642486787612))
                for n,name in enumerate(FEATURES):
                    cv,pv=values[ai,ci,k,j,qi,n],values[ai,pi,k,j,qi,n]
                    r['clear_'+name]=float(cv) if np.isfinite(cv) else None;r['contact_'+name]=float(pv) if np.isfinite(pv) else None
                    r['delta_'+name]=float(pv-cv) if np.isfinite(pv-cv) else None
                r['delta_raw_M3']=float(raw[ai,pi,k,j,qi]-raw[ai,ci,k,j,qi]);r['delta_smooth_M3']=float(smooth[ai,pi,k,j,qi]-smooth[ai,ci,k,j,qi])
                r['delta_expected_current_scaled_mass']=float(mean[ai,pi,j,qi]-mean[ai,ci,j,qi]);rr.append(r)
    summary=[]
    for a in p['angles']:
        for relation in ('same_side','mirrored_side'):
            for depth in ('in_1cm','in_4cm'):
                rows=[r for r in rr if r['angle']==a and r['relation']==relation and r['depth']==depth]
                metrics={}
                for name in (*FEATURES,'raw_M3','smooth_M3','expected_current_scaled_mass'):
                    v=np.array([r['delta_'+name] for r in rows if r['delta_'+name] is not None])
                    metrics[name]=dict(finite=len(v),mean=float(v.mean()) if len(v) else None,median=float(np.median(v)) if len(v) else None,positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()))
                summary.append(dict(angle=a,relation=relation,depth=depth,paired_height_windows=len(rows),clear_M3_alarm_windows=sum(r['clear_M3_alarm'] for r in rows),contact_M3_alarm_windows=sum(r['contact_M3_alarm'] for r in rows),metrics=metrics))
    assert previous+time.monotonic()-t<180
    csvsave('paired_windows.csv',rr);save('analysis_result.json',dict(status='COMPLETE',seconds=time.monotonic()-t,paired_windows=len(rr),summary=summary,limits='Kmatched shared coupling and selected geometries correlated; nonlinear fraction is noisy; deterministic signed mass reference is not expected fraction; no new readout or equal-cost benefit'))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()

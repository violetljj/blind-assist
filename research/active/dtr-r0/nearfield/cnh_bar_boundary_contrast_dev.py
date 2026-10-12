"""Observable fixed lateral paired strips, cached features and frozen M3 only."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S
import cnh_bar_fusion_probe_dev as F

OUT=B.ROOT/'artifacts.local/work/cnh-bar-boundary-contrast-dev-20261009'
CACHE=B.ROOT/'artifacts.local/work/cnh-bar-vertical-evidence-dev-20261009'
ANGLES=(0,-3,3)
STATS=('I_pair','Dmax','Dwin','I_only')
SHAPE=(24,17,33)
LOW=np.array([-.6,-.5,0.]);STEP=np.array([.05,.1,.1])
THETA=.8557642486787612;TM=.9404184587540165;TL=4.625390338985158

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):B.save(OUT/name,value)
def overlap(lo,hi):
    ix=np.stack(np.meshgrid(*[np.arange(n) for n in SHAPE],indexing='ij'),-1)
    lower=LOW+ix*STEP;upper=lower+STEP
    w=np.prod(np.maximum(0,np.minimum(upper,hi)-np.maximum(lower,lo))/STEP,-1)
    w[w<1e-8]=0
    return w

def strips():
    matrices=[];keys=[]
    for yl,yh in ((-.2,.42),(.42,.9)):
        ins=[];outs=[];kk=[]
        for side in (-1,1):
            for y in range(SHAPE[1]):
                a=max(yl,float(LOW[1]+y*STEP[1]));b=min(yh,float(LOW[1]+(y+1)*STEP[1]))
                if b-a<1e-8:continue
                for z in range(25):
                    zlo=.3+z*.1;zhi=zlo+.3
                    il,ih=(-.3,0.) if side<0 else (0.,.3)
                    ol,oh=(-.6,-.3) if side<0 else (.3,.6)
                    ins.append(sparse.csr_matrix(overlap([il,a,zlo],[ih,b,zhi]).reshape(1,-1)))
                    outs.append(sparse.csr_matrix(overlap([ol,a,zlo],[oh,b,zhi]).reshape(1,-1)))
                    kk.append([side,y,zlo,zhi])
        matrices.append((sparse.vstack(ins),sparse.vstack(outs)));keys.append(np.asarray(kk))
    return matrices,keys

def prepare():
    t=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve boundary contrast PLAN')
    inputs=[CACHE/'PLAN.json',CACHE/'result_run.json',S.OUT/'PLAN.json',S.OUT/'geometry.npz',*[CACHE/f'angle_{a:+d}.npz' for a in ANGLES]]
    source=[Path(__file__),Path(F.__file__),Path(B.__file__),Path(S.__file__),Path(__file__).with_name('cnh_cvr_projection.py')]
    plan=dict(task='CNH_BAR_BOUNDARY_CONTRAST_DEV_20261009',base_commit='615141bf',lane='EXPLORE consumed Development cache',
        authorization='User 继续 after spatial mixing and failed simple crop; observable contrast diagnostics before new readout candidate',
        goal='Does fixed query lateral contrast rank contact above strict-clear within frozen M3 alarms, and what does an uncalibrated zero-contrast veto retain?',
        budgets_CPU_stage_wall_seconds=dict(prepare=180,extraction=300,analysis_checks=180,independent_audit=120),
        scope='132vertical*K4*13outputs, ideal/-3/+3 cache;224contact events perheight,1248jointclear slots,96clear/passclips. No new photons/projection/M3/training.',
        observable='Original full FP16 tot/cnt, fixed public geometry. Left inner[-.3,0] outer[-.6,-.3];right inner[0,.3] outer[.3,.6];same public-height clipped one .10m cell and .30m depth sliding .10m from .3..3m. Both densities=sum(mask*tot)/sum(mask*cnt), not noise standardization.',
        statistics='I_pair maxinside on paired-positivecoverage domain;Dmax maximum paired inside-outside;Dwin paired difference at I_pair firstargmax. I_only inside-only max descriptive support control. No labels/physicalbox/background absence/truequeryerror/side truth in scores.',
        unknown='No valid pairs => NaN. Every last5 raw statistic must be finite, otherwise smoothed unknown. Unknown keeps original M3/fusion alarm; never fillmissing with0/-inf for smoothing. I_pair and Dmax share candidate domain.',
        ordering='Eachframe select raw statistic first, then original last5 expweights; never smooth eachstrip before winner selection. Tot only, cnt unchanged, complete M3/local scores reused.',
        probes='Two predetermined zero-threshold vetoes Dmax and Dwin: baseflag AND NOT(finite smoothed D AND D<0). Apply perheight then joint. Separate frozen originalM3 and fixedfusion. No threshold sweep, k-budget reallocation or policy upgrade.',
        ranking='AUC largerstat favors contact, compare I_pair/Dmax/Dwin/M3 among original M3-positive timelyf3..13 windows. Fixed M3 strata [theta,.940418,1.5,3,inf]; pair within strata. Same jointly-finite domain. Correlated window ranks descriptive, not independent event validation or causal increment.',
        decision_check='One timely event1/224, jointclear cost1/1248; ideal baseline217/177 and12clear, +3 baseline191/143 and117clear. Veto subset gain must0; reduction paired with every lost event. Zero loss plus fewerclear in allangles supports further allshape check only; any loss remains tradeoff, not replacement. Do not close every boundarycontrast method from fixed probes.',
        adjustable_scope='Implementation repairs, complete extraction/ranking and two fixed diagnostic vetoes, focused independent checks/report/current/RUNS/direct push. No new statistical candidate/grid/noise/inference/shape gate.',
        stop='Complete fixed3anglequeue and2zerovetoes or cumulative stagecaps including failures; preserve source/failures/data, no result-based retuning.',
        inputs_sha256={B.logical_path(p):sha(p) for p in inputs},source_sha256={B.logical_path(p):sha(p) for p in source})
    plan['prepare_seconds']=time.monotonic()-t;assert plan['prepare_seconds']<180
    save('PLAN.json',plan);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('PREPARED',plan['prepare_seconds'],flush=True)

def extract():
    t=time.monotonic();previous=sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('failure_extract*.json'))
    try:
        if (OUT/'result_extract.json').exists():raise FileExistsError('Preserve extracted scores')
        plan=json.loads((OUT/'PLAN.json').read_text())
        for p,h in {**plan['inputs_sha256'],**plan['source_sha256']}.items():assert sha(B.ROOT/p)==h,p
        mm,keys=strips()
        for q,(i,o) in enumerate(mm):sparse.save_npz(OUT/f'inside_{q}.npz',i);sparse.save_npz(OUT/f'outside_{q}.npz',o)
        records=[]
        for a in ANGLES:
            path=OUT/f'scores_{a:+d}.npz'
            if path.exists():raise FileExistsError('Preserve score payload')
            with np.load(CACHE/f'angle_{a:+d}.npz') as d:
                echo=d['echo'];cnt=d['count'];ids=d['scene_ids'];m3=d['logits'][:,:,:,0];local=d['local_raw']
            raw=np.full((len(ids)*4,13,4,2),np.nan);wins=np.full((len(ids)*4,13,2),-1,dtype=int)
            counts=np.zeros((13,2,2),dtype=int)
            for f in range(13):
                if previous+time.monotonic()-t>300:raise TimeoutError('extraction cumulative300s')
                total=echo[:,f,0].astype(np.float64).reshape(len(echo),-1);count=cnt[f].astype(float).ravel()
                for q,(im,om) in enumerate(mm):
                    ci=np.asarray(im@count);co=np.asarray(om@count);iv=ci>1e-9;ov=co>1e-9;pair=iv&ov
                    counts[f,q]=[int(iv.sum()),int(pair.sum())]
                    ii=np.full((len(echo),len(ci)),np.nan);oo=np.full_like(ii,np.nan)
                    ii[:,iv]=np.asarray(im[iv]@total.T).T/ci[iv];oo[:,ov]=np.asarray(om[ov]@total.T).T/co[ov]
                    if iv.any():raw[:,f,3,q]=np.max(ii[:,iv],axis=1)
                    if pair.any():
                        pidx=np.flatnonzero(pair);inside=ii[:,pair];diff=inside-oo[:,pair];win=inside.argmax(1)
                        raw[:,f,0,q]=inside[np.arange(len(echo)),win]
                        raw[:,f,1,q]=diff.max(1);raw[:,f,2,q]=diff[np.arange(len(echo)),win]
                        wins[:,f,q]=pidx[win]
            raw=raw.reshape(len(ids),4,13,4,2);smooth=np.stack([F.smooth(raw[:,:,:,s]) for s in range(4)],axis=3)
            np.savez_compressed(path,scene_ids=ids,raw=raw,smooth=smooth,winner=wins.reshape(len(ids),4,13,2),coverage=counts,m3=F.smooth(m3),local=F.smooth(local),keys_head=keys[0],keys_body=keys[1])
            records.append(dict(angle=a,payload=path.name,sha256=sha(path),unknown_smooth=np.isnan(smooth).sum((0,1,2)).tolist(),valid_candidate_counts=counts.tolist()))
            del echo,total
            print('EXTRACTED',a,round(time.monotonic()-t,3),flush=True)
        save('result_extract.json',dict(status='COMPLETE',seconds=time.monotonic()-t,cumulative_seconds=previous+time.monotonic()-t,angles=records,training=0,new_photons=0,new_inference=0,new_projection=0))
    except BaseException as e:save(f'failure_extract_{time.time_ns()}.json',dict(seconds=time.monotonic()-t,error=repr(e)));raise

def auc(pos,neg):
    n=len(pos);m=len(neg)
    if not n or not m:return None
    ranks=rankdata(np.r_[pos,neg],method='average')
    return float((ranks[:n].sum()-n*(n+1)/2)/(n*m))

def csvwrite(name,rows):
    with (OUT/name).open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze():
    t=time.monotonic()
    if (OUT/'result_analysis.json').exists():raise FileExistsError('Preserve analysis')
    with np.load(S.OUT/'geometry.npz') as d:categories=d['category']
    ranks=[];work=[];ledger=[]
    for a in ANGLES:
        with np.load(OUT/f'scores_{a:+d}.npz') as d:ids=d['scene_ids'];stats=d['smooth'];m3=d['m3'];local=d['local']
        cat=categories[ids];clear=(cat=='clear').all(1)
        for q in (0,1):
            ms=m3[:,:,:11,q];ss=stats[:,:,:11,:3,q];finite=np.isfinite(ss).all(-1)
            positive=np.broadcast_to((cat[:,q]=='contact')[:,None,None],ms.shape)
            negative=np.broadcast_to(clear[:,None,None],ms.shape)
            for lower,upper,label in [(THETA,np.inf,'ALL_M3_POSITIVE'),(THETA,TM,'weak'),(TM,1.5,'mid'),(1.5,3.,'strong'),(3.,np.inf,'very_strong')]:
                eligible=finite&(ms>=lower)&(ms<upper);pp=eligible&positive;nn=eligible&negative
                for s,values in [('M3',ms),*[(STATS[k],ss[:,:,:,k]) for k in range(3)]]:
                    ranks.append(dict(angle=a,height=('HEAD','BODY')[q],stratum=label,statistic=s,contact_windows=int(pp.sum()),clear_windows=int(nn.sum()),auc=auc(values[pp],values[nn]),contact_median=float(np.median(values[pp])) if pp.any() else None,clear_median=float(np.median(values[nn])) if nn.any() else None))
        for base,flags in [('M3',m3>=THETA),('fusion',(m3>=TM)|(local>=TL))]:
            bm,bt=F.metrics(flags,cat);work.append(dict(angle=a,base=base,probe='baseline',metrics=bm))
            for k in (1,2):
                score=stats[:,:,:,k];veto=np.isfinite(score)&(score<0);candidate=flags&~veto
                assert not (candidate&~flags).any()
                cm,ct=F.metrics(candidate,cat);paired=F.compare(bt,ct,cat,ids)
                assert all(r['gain']==0 for r in paired)
                work.append(dict(angle=a,base=base,probe=STATS[k]+'_zero_veto',metrics=cm,paired=paired,unknown_windows=int((~np.isfinite(score)).sum()),removed_height_windows=int((flags&veto).sum())))
                for i,r,q in np.argwhere(np.broadcast_to((cat=='contact')[:,None,:],bt.shape)):
                    ledger.append(dict(angle=a,base=base,probe=STATS[k],scene=int(ids[i]),replica=int(r),height=('HEAD','BODY')[q],baseline=int(bt[i,r,q]),candidate=int(ct[i,r,q]),loss=int(bt[i,r,q] and not ct[i,r,q])))
    csvwrite('ranking.csv',ranks);csvwrite('event_ledger.csv',ledger)
    save('result_analysis.json',dict(status='COMPLETE',seconds=time.monotonic()-t,ranking=ranks,workpoints=work,event_rows=len(ledger),limitation='Correlated consumedDevelopment vertical-only ranking and uncalibrated subset-veto sensitivity; no noise-standardized inference/unique causality or fullshape policy improvement. Unknown keeps base alarm.'))
    assert time.monotonic()-t<180
    for r in work:print(r['angle'],r['base'],r['probe'],r['metrics']['counts'],r['metrics']['clear_slots'],flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()

"""Posthoc fit diagnosis: matched scratch CVR, fixed checkpoints and tiny fits.

No candidate, alert threshold or old result is changed. All new payloads live
in a separate ignored directory. Metadata/labels never enter model forward.
"""
import argparse,gc,hashlib,json,os,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn.functional as F
import cnh_readout_pilot2 as P
import cnh_readout_pilot2_geometry as G
import cnh_readout_pilot2_model as TM
import cnh_temporal_readout_train as TR
import cnh_temporal_readout_model as VM

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-t2-fit-diagnostic-20261004'
SOURCE=P.OUT
read=P.read;save=P.save;sha=P.sha

def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    for key,name in [('TEMP','tmp'),('TMP','tmp'),('CUPY_CACHE_DIR','cupy-cache'),('MPLCONFIGDIR','mpl-cache')]:
        path=OUT/name;path.mkdir(exist_ok=True);os.environ[key]=str(path)

def plan():
    setup();path=OUT/'PLAN.json'
    if path.exists():raise FileExistsError(path)
    dependencies=[Path(x.__file__) for x in (P,G,TM,TR,VM)]
    checkpoints=[model_path(a,s) for a,s in [('M3',0),('M3',1),('M3',2),('M3',3),('M3',4),('V',0),('VD',0),('T2',0)]]
    p=dict(run='CNH_T2_FIT_DIAGNOSTIC_20261004',role='EXPLORE posthoc consumed Development; no candidate',source=str(SOURCE),
        authorization='User supplied A/B/C/D diagnostic; no hard time limit, optimize efficiency',
        frozen_utc=datetime.now(timezone.utc).isoformat(),old_plan_sha256=sha(SOURCE/'PLAN.json'),
        train_rows=39936,holdout_rows=17472,holdout='Consumed231001..231072 scene-disjoint from train; generator family shared',
        models=['M3_seed0','V_seed0','VD_seed0','T2_seed0','S_seed0'],
        S=dict(architecture='Original CVR; random seed0, no checkpoint',epochs=20,lr=.002,batch=64,wd=.0001,scheduler='CosineAnnealingLR20',permutation='numpy.default_rng(0), original T2 seed0 hashes all20 epochs',final='fixed epoch20'),
        tiny=dict(sizes=[256,2048],seed=2026100602,scene_selection='Nested 32 scene groups,16natural/16displacement;64 rows/group;256 selects32 rows in first4 groups/domain',
            models=['T2','S'],initialization='random seed0 independently per fit; never resume old T2',lr=.002,wd=.0001,batch=64,scheduler='constant',
            evaluate_every_steps=100,minimum_steps=400,near_zero_hard_loss=.01,near_zero_auc=.995,
            plateau='After>=400updates: last300updates relative best fullsubset loss improvement<.005',
            maximum_updates=10000,maximum_role='numerical nonconvergence guard, not wallclock deadline; report UPDATE_LIMIT not plateau',
            success='Both sizes hardweightedBCE perrow<=.01 AND pooled AUC>=.995; single-size success reported separately',
            diagnosis='Failure can mean optimizer/scales or structural collisions; not automatic numerical bug proof'),
        bootstrap=dict(n=1000,seed=2026100601,unit='whole fresh scenes'),
        support=dict(quantity='sum valid-history query SUB3 zone/bin mass',zero_upper=1e-8,low_high='training positive support median per query; label/score blind, frozen before metrics',holdout='same boundaries'),
        strata=['domain','range<1.2/1.2-2.1/2.1-2.6/>=2.6','label','support','FOV_IN/OUT/NA','HEAD/BODY'],
        metric='hardweightedBCE=sum(mask*weight*BCE)/N, pooled/query raw AUC; normalized weighted BCE plus train displacement-only comparison',
        tolerance=dict(auc_close=.02,loss_close_absolute=.03,loss_close_relative=.20,S_better_auc=.03,S_better_loss=.03),
        branches=dict(S_MATCHES_T2='Both train and holdout abs pooled AUC diff<=.02 AND loss diff<=max(.03,.2*T2loss): consistent with initialization/data/recipe limitation, not unique causal attribution',
            S_BETTER_T2='Heldout S-T2 pooled AUC>=.03 and paired scene CIlo>0 AND hardweighted loss reduction>=.03; tiny T2 succeeds: representation/aggregation recipe suspect, not local context specifically proven',
            T2_TINY_UNFIT='T2 tiny joint criterion fails: diagnose optimization, scale, capacity or feature collisions before architecture verdict',
            T2_HOLDOUT_MATCHES_V='Heldout abs pooled AUC diff<=.02 AND loss diff<=max(.03,.2*Vloss): old training-only gate unsuitable for interpreting heldout performance, original gate decision retained',
            OTHER='MIXED/INCONCLUSIVE; report nonexclusive flags, no posthoc gate rescue'),
        tolerance_basis='AUC .02 is practical near-equality, .03 is oldpilot meaningful increment; loss .03/20% allows scale variation without treating visibly different fits as equal. Descriptive, not equivalence proof.',
        boundaries='No thresholds/natural candidate run/new network/M3replacement. Same optimizer does not establish equal convergence or fair architecture-optimal tuning. M3 train natural rows may overlap original pretraining; only new displacement domain is sample-out reference.',
        source_sha256={str(x):sha(x) for x in dependencies},checkpoint_sha256={str(x):sha(x) for x in checkpoints})
    p['frozen_training_inputs_sha256']=read(SOURCE/'models/T2/seed0/training_receipt.json')['input_sha256']
    p['analysis_source_sha256']={str(x):sha(x) for x in Path(__file__).parent.glob('cnh_t2_fit*.py')}
    save(path,p);save(OUT/'progress.json',dict(status='PLAN_FROZEN',plan_sha256=sha(path)))
    save(OUT/'timing.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),started_unix=time.time(),hard_deadline=None))
    print('PLAN_FROZEN',sha(path),flush=True)

def frozen():
    p=read(OUT/'PLAN.json')
    if p['old_plan_sha256']!=sha(SOURCE/'PLAN.json'):raise ValueError('Old plan changed')
    amendment=OUT/'SUBSET_SCHEMA_AMENDMENT.json'
    if amendment.exists():
        fix=read(amendment)
        if fix['plan_sha256']!=sha(OUT/'PLAN.json'):raise ValueError('Subset amendment belongs to another plan')
        p['tiny']['scene_selection']=fix['scene_selection']
    return p

def folder(split):return P.OLD/'inputs/train' if split=='train' else SOURCE/'inputs/fresh_evaluation'

def model_path(arm,seed=0):
    if arm=='M3':return ROOT/f'artifacts.local/work/cnh-margin-labels-20261002/models/M3/model_seed{seed}.pt'
    if arm=='V':return P.OLD/f'models/V/seed{seed}/model.pt'
    if arm in ('VD','T2'):return SOURCE/f'models/{arm}/seed{seed}/model.pt'
    return OUT/'models/S/seed0/model.pt'

class Data(TR.Inputs):
    def __init__(self,split,arm,training=False):
        self.folder=OUT/'input_hashes'/split;self.folder.mkdir(parents=True,exist_ok=True)
        names=['histories','transforms','length','ambient'] if arm=='T2' else ['voxels']
        self.paths={k:folder(split)/f'{k}.npy' for k in names}
        if training:self.paths.update({k:folder('train')/f'{k}.npy' for k in ('labels','mask','weights')})
        self.maps={k:np.load(v,mmap_mode='r') for k,v in self.paths.items()}
        self.n=len(self.maps[names[0]]);self.gpu_maps={};self.cache_receipt=dict(used=False);self.arm=arm
        self.table=self.index=self.gpu_geometry=None
        if arm=='T2':
            geo=SOURCE/'geometry/train' if split=='train' else OUT/'geometry/fresh_evaluation'
            if read(geo/'receipt.json')['status']!='COMPLETE':raise ValueError('Geometry unavailable')
            self.table=np.load(geo/'table.npy',mmap_mode='r');self.index=np.load(geo/'index.npy',mmap_mode='r')
            self.paths.update(geometry_table=geo/'table.npy',geometry_index=geo/'index.npy')
    def enable_gpu_cache(self,device):
        receipt=super().enable_gpu_cache(device)
        if self.table is not None:
            free,_=torch.cuda.mem_get_info()
            if self.table.nbytes+self.index.nbytes<free-2*1024**3:
                self.gpu_geometry=(torch.from_numpy(self.table).to(device,copy=True),torch.from_numpy(self.index).to(device,copy=True))
                receipt['geometry_cached']=True
        return receipt
    def batch(self,ids,device):
        value=super().batch(ids,device)
        if self.table is not None:
            if self.gpu_geometry is not None:
                table,index=self.gpu_geometry;value['query_weights']=table[index[torch.as_tensor(ids,device=device)].long()]
            else:value['query_weights']=torch.from_numpy(np.array(self.table[self.index[ids]],copy=True)).to(device)
        return value
    def close(self):super().close();self.table=self.index=self.gpu_geometry=None

def forward(net,arm,b):
    if arm=='T2':return net(*(b[k] for k in ('histories','transforms','length','ambient','query_weights')))
    return net(VM.prepare_voxels(b['voxels']))

def network(arm,seed=0,load=True):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    net=TM.T2() if arm=='T2' else VM.CVR()
    if load:net.load_state_dict(torch.load(model_path(arm,seed),map_location='cpu',weights_only=True))
    return net.cuda()

def geometry():
    frozen();TR.configure('cuda');start=time.monotonic();dest=OUT/'geometry/fresh_evaluation';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'receipt.json').exists():return
    ts=np.load(folder('fresh_evaluation')/'transforms.npy',mmap_mode='r');length=np.load(folder('fresh_evaluation')/'length.npy',mmap_mode='r')
    valid=np.arange(8)[None]>=8-length[:,None]
    unique,inverse=np.unique(np.asarray(ts)[valid].reshape(-1,16),axis=0,return_inverse=True)
    table=np.lib.format.open_memmap(dest/'table.npy',mode='w+',dtype=np.float32,shape=(len(unique)+1,2,8,8,16));table[0]=0
    index=np.zeros((len(length),8),np.int32);index[valid]=inverse.astype(np.int32)+1;np.save(dest/'index.npy',index)
    op=G.QueryOverlap('cuda',16)
    for i in range(0,len(unique),16):
        table[i+1:i+17]=op.weights(unique[i:i+16].reshape(-1,4,4)).cpu().numpy()
        if i%3200==0:print('heldout_geometry',i,len(unique),round(time.monotonic()-start,1),flush=True)
    table.flush();del table,op
    save(dest/'receipt.json',dict(status='COMPLETE',rows=len(length),unique_poses=len(unique),seconds=time.monotonic()-start,plan_sha256=sha(OUT/'PLAN.json'),input_sha256={str(folder('fresh_evaluation')/f'{x}.npy'):sha(folder('fresh_evaluation')/f'{x}.npy') for x in ('transforms','length')},output_sha256={str(dest/x):sha(dest/x) for x in ('table.npy','index.npy')},operator_source_sha256=sha(G.__file__)))
    gc.collect();torch.cuda.empty_cache()

def infer():
    frozen();TR.configure('cuda');start=time.monotonic()
    for split in ('train','fresh_evaluation'):
        for arm in ('M3','V','VD','T2','S'):
            if not model_path(arm).exists():continue
            seeds=range(5) if arm=='M3' and split=='fresh_evaluation' else [0]
            data=Data(split,arm);data.enable_gpu_cache('cuda')
            try:
                for seed in seeds:
                    dest=OUT/f'predictions/{split}/{arm}_seed{seed}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists():
                        if not dest.with_suffix('.json').exists():raise ValueError('Incomplete score receipt')
                        continue
                    began=time.monotonic();net=network(arm,seed);net.eval();raw=np.empty((data.n,2),np.float32)
                    with torch.inference_mode():
                        for i in range(0,data.n,128):
                            ids=np.arange(i,min(i+128,data.n));raw[ids]=forward(net,arm,data.batch(ids,'cuda')).float().cpu().numpy()
                    if not np.isfinite(raw).all():raise ValueError('Nonfinite diagnosis logits')
                    np.savez_compressed(dest,raw=raw)
                    save(dest.with_suffix('.json'),dict(status='COMPLETE',role='POSTHOC_DIAGNOSTIC_NOT_CANDIDATE',split=split,arm=arm,seed=seed,rows=data.n,seconds=time.monotonic()-began,model_sha256=sha(model_path(arm,seed)),plan_sha256=sha(OUT/'PLAN.json'),raw_sha256=sha(dest),row_sha256=sha(folder(split)/'rows.npz')))
                    del net;print('diagnostic_infer',split,arm,seed,round(time.monotonic()-began,2),flush=True)
            finally:data.close();gc.collect();torch.cuda.empty_cache()
    save(OUT/'inference_timing.json',dict(seconds=time.monotonic()-start,status='COMPLETE'))

def train_s():
    p=frozen();TR.configure('cuda');cfg=p['S'];dest=OUT/'models/S/seed0';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'model.pt').exists():return
    data=Data('train','S',True);data.enable_gpu_cache('cuda');net=network('S',load=False)
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd']);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'])
    rng=np.random.default_rng(0);history=[];old=read(SOURCE/'models/T2/seed0/training_receipt.json');start=time.monotonic()
    try:
        for epoch in range(20):
            t0=time.monotonic();order=rng.permutation(data.n);digest=hashlib.sha256(order.tobytes()).hexdigest()
            if digest!=old['history'][epoch]['order_sha256']:raise ValueError('S/T2 shuffle mismatch')
            net.train();total=0.
            for i in range(0,data.n,64):
                ids=np.sort(order[i:i+64]);b=data.batch(ids,'cuda');opt.zero_grad(set_to_none=True);loss=TR.loss_for(forward(net,'S',b),b)
                if loss is None:continue
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite S loss')
                loss.backward();opt.step();total+=float(loss.detach())*len(ids)
            schedule.step();history.append(dict(epoch=epoch+1,loss=total/data.n,seconds=time.monotonic()-t0,order_sha256=digest))
            save(dest/'progress.json',dict(status='TRAINING',history=history));print('S',epoch+1,history[-1]['loss'],flush=True)
        torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},dest/'model.pt')
        save(dest/'training_receipt.json',dict(status='COMPLETE',history=history,seconds=time.monotonic()-start,model_sha256=sha(dest/'model.pt'),plan_sha256=sha(OUT/'PLAN.json'),source_sha256=sha(__file__)))
    finally:data.close();del net,opt,schedule;gc.collect();torch.cuda.empty_cache()

def m3_parity():
    """Use original64-row GPU stack/mean path; preserve the128-row inference."""
    frozen();TR.configure('cuda');data=Data('fresh_evaluation','M3');data.enable_gpu_cache('cuda')
    dest=OUT/'predictions/m3_original_batch';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'parity.json').exists():return
    nets=[network('M3',s).eval() for s in range(5)];raw=np.empty((5,data.n,2),np.float32);mean=np.empty((data.n,2),np.float32);start=time.monotonic()
    try:
        with torch.inference_mode():
            for i in range(0,data.n,64):
                ids=np.arange(i,min(i+64,data.n));x=VM.prepare_voxels(data.batch(ids,'cuda')['voxels']);stack=torch.stack([net(x) for net in nets])
                raw[:,ids]=stack.cpu().numpy();mean[ids]=stack.mean(0).cpu().numpy()
        with np.load(SOURCE/'predictions/M3_fresh_evaluation.npz') as z:old=z['raw']
        error=float(np.max(np.abs(mean-old)))
        if error>1e-6:raise ValueError(('Original M3 inference path not reproduced',error))
        for seed in range(5):
            path=dest/f'M3_seed{seed}.npz';np.savez_compressed(path,raw=raw[seed]);save(path.with_suffix('.json'),dict(status='COMPLETE',model_sha256=sha(model_path('M3',seed)),plan_sha256=sha(OUT/'PLAN.json'),raw_sha256=sha(path),row_sha256=sha(folder('fresh_evaluation')/'rows.npz'),batch=64,role='Original batch/kernel path,128-row cache retained separately'))
        np.savez_compressed(dest/'original_GPU_mean5.npz',raw=mean)
        save(dest/'parity.json',dict(status='PASS',maximum_absolute_error=error,bitwise_equal=bool(np.array_equal(mean,old)),seconds=time.monotonic()-start,batch=64,old_mean_sha256=sha(SOURCE/'predictions/M3_fresh_evaluation.npz'),plan_sha256=sha(OUT/'PLAN.json'),notes='Two-ULP128row vs64row inference difference is preserved, no relaxed parity tolerance'))
    finally:data.close();nets.clear();gc.collect();torch.cuda.empty_cache()

def subset_ids():
    path=OUT/'tiny/subsets.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    with np.load(folder('train')/'rows.npz') as z:r={k:z[k] for k in z.files}
    rng=np.random.default_rng(frozen()['tiny']['seed']);large=[];small=[];groups=[]
    for domain in (0,1):
        units=np.unique(r['unit'][r['domain']==domain])
        candidates=[]
        for u in units:
            ids=np.flatnonzero((r['domain']==domain)&(r['unit']==u))
            if len(ids)>=16:candidates.append((int(u),ids))
        if len(candidates)<64:raise ValueError('Insufficient scene units')
        for gi in rng.permutation(len(candidates))[:64]:
            u,ids=candidates[gi];chosen=rng.choice(ids,16,replace=False);large.extend(chosen);groups.append([domain,u])
            if len([g for g in groups if g[0]==domain])<=8:small.extend(chosen)
    result={'n256':np.sort(small),'n2048':np.sort(large)}
    assert len(result['n256'])==256 and len(result['n2048'])==2048 and set(small)<=set(large)
    np.savez_compressed(path,**result);save(path.with_suffix('.json'),dict(groups=groups,sha256=sha(path),scene_grouping='unit;16 rows/scene, nested first8/64 scenes each domain; no evaluation or score-based selection',plan_sha256=sha(OUT/'PLAN.json'),amendment_sha256=sha(OUT/'SUBSET_SCHEMA_AMENDMENT.json')))
    return result

def tiny_fit():
    p=frozen();TR.configure('cuda');cfg=p['tiny'];subsets=subset_ids()
    for arm in ('T2','S'):
        data=Data('train',arm,True);data.enable_gpu_cache('cuda')
        try:
            for key,ids in subsets.items():
                dest=OUT/f'tiny/{arm}_{key}';dest.mkdir(parents=True,exist_ok=True)
                if (dest/'receipt.json').exists():continue
                net=network(arm,load=False);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
                rng=np.random.default_rng(0);order=np.array([],int);position=0;curve=[];start=time.monotonic();stop='UPDATE_LIMIT'
                def metric(step):
                    net.eval();values=[];ys=[];loss_sum=0.;count=0;mass=0.
                    with torch.inference_mode():
                        for i in range(0,len(ids),128):
                            b=data.batch(ids[i:i+128],'cuda');s=forward(net,arm,b).float();e=b['mask']*b['weights']
                            loss_sum+=float((F.binary_cross_entropy_with_logits(s,b['labels'],reduction='none')*e).sum());mass+=float(e.sum())
                            valid=b['mask']>0;values.extend(s[valid].cpu().numpy());ys.extend(b['labels'][valid].cpu().numpy());count+=int(valid.sum())
                    ys=np.asarray(ys)==1;values=np.asarray(values);n1=int(ys.sum());n0=len(ys)-n1
                    auc=None if not n1 or not n0 else float((rankdata(values)[ys].sum()-n1*(n1+1)/2)/(n1*n0))
                    return dict(step=step,hard_weighted_loss=loss_sum/len(ids),normalized_BCE=loss_sum/mass,pooled_auc=auc,valid_queries=count,seconds=time.monotonic()-start)
                curve.append(metric(0))
                for step in range(1,cfg['maximum_updates']+1):
                    if position>=len(order):order=rng.permutation(ids);position=0
                    batch_ids=np.sort(order[position:position+64]);position+=64;b=data.batch(batch_ids,'cuda');net.train();opt.zero_grad(set_to_none=True)
                    loss=TR.loss_for(forward(net,arm,b),b)
                    if loss is None:continue
                    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite tiny loss')
                    loss.backward();opt.step()
                    if step%100==0:
                        cell=metric(step);curve.append(cell);save(dest/'progress.json',dict(status='TRAINING',curve=curve));print('tiny',arm,key,step,cell['hard_weighted_loss'],cell['pooled_auc'],flush=True)
                        if step>=cfg['minimum_steps']:
                            if cell['hard_weighted_loss']<=.01 and cell['pooled_auc'] is not None and cell['pooled_auc']>=.995:stop='NEAR_ZERO';break
                            before=min(x['hard_weighted_loss'] for x in curve[:-3]);after=min(x['hard_weighted_loss'] for x in curve[-3:])
                            if (before-after)/max(before,1e-12)<.005:stop='PLATEAU';break
                torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},dest/'model.pt')
                save(dest/'receipt.json',dict(status='COMPLETE',stop=stop,arm=arm,rows=len(ids),curve=curve,steps=step,seconds=time.monotonic()-start,model_sha256=sha(dest/'model.pt'),plan_sha256=sha(OUT/'PLAN.json'),source_sha256=sha(__file__),subset_sha256=sha(OUT/'tiny/subsets.npz'),diagnosis='tiny memorization only, no heldout/candidate result'))
                del net,opt;gc.collect();torch.cuda.empty_cache()
        finally:data.close();gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=('plan','geometry','infer','S','tiny','M3-parity'));a=parser.parse_args();setup()
    {'plan':plan,'geometry':geometry,'infer':infer,'S':train_s,'tiny':tiny_fit,'M3-parity':m3_parity}[a.stage]()

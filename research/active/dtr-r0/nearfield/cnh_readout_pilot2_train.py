"""Pilot2 fixed training, final-checkpoint fit gate and label-blind inference."""
import argparse
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
import torch
import torch.nn.functional as F

import cnh_readout_pilot2 as C
import cnh_temporal_readout_train as TR
import cnh_temporal_readout_model as OLD_MODEL


class Inputs(TR.Inputs):
    def __init__(self, split, arm, training=False):
        self.folder=C.OUT/'input_hashes'/split; self.folder.mkdir(parents=True,exist_ok=True)
        source=C.input_folder(split)
        names=['voxels'] if arm in ('V_retest','VD','M3') else ['histories','transforms','length','ambient']
        self.paths={k:source/f'{k}.npy' for k in names}
        if training:
            self.paths.update({k:source/f'{k}.npy' for k in ('labels','mask','weights')})
            if arm=='VD' and split=='train': self.paths['labels']=C.OUT/'teacher/soft_labels.npy'
        self.maps={k:np.load(path,mmap_mode='r',allow_pickle=False) for k,path in self.paths.items()}
        self.n=len(self.maps[names[0]])
        if any(len(v)!=self.n for v in self.maps.values()): raise ValueError('Row count mismatch')
        self.gpu_maps={}; self.cache_receipt=dict(used=False,reason='not_requested'); self.arm=arm
        self.geometry_table=None; self.geometry_index=None; self.gpu_geometry=None
        if arm=='T2':
            directory=C.OUT/'geometry'/split
            receipt=C.read(directory/'receipt.json')
            if receipt['status']!='COMPLETE': raise ValueError('Geometry cache incomplete')
            self.geometry_table=np.load(directory/'table.npy',mmap_mode='r')
            self.geometry_index=np.load(directory/'index.npy',mmap_mode='r')
            if self.geometry_index.shape!=(self.n,8): raise ValueError('Geometry index shape')
            self.extra_paths=[directory/'table.npy',directory/'index.npy']
        else: self.extra_paths=[]

    def enable_gpu_cache(self,device):
        result=super().enable_gpu_cache(device)
        if self.geometry_table is not None and device.startswith('cuda'):
            size=self.geometry_table.nbytes+self.geometry_index.nbytes
            free,total=torch.cuda.mem_get_info(device)
            if size<free-2*1024**3:
                self.gpu_geometry=(torch.from_numpy(self.geometry_table).to(device,copy=True),
                                   torch.from_numpy(self.geometry_index).to(device,copy=True))
                result['geometry_gpu_cached']=True;result['geometry_bytes']=size
        return result

    def batch(self,indices,device):
        value=super().batch(indices,device)
        if self.geometry_table is not None:
            if self.gpu_geometry is not None:
                table,index=self.gpu_geometry
                ids=torch.as_tensor(indices,device=device,dtype=torch.long)
                value['query_weights']=table[index[ids].long()]
            else:
                index=np.asarray(self.geometry_index[indices])
                value['query_weights']=torch.from_numpy(np.array(self.geometry_table[index],copy=True)).to(device)
        return value

    def hashes(self):
        result=super().hashes()
        if self.extra_paths:
            path=self.folder/'geometry_hashes.json'
            if path.exists():
                prior=C.read(path)
                for name,item in prior.items():
                    stat=Path(name).stat()
                    if stat.st_size!=item['size'] or stat.st_mtime_ns!=item['mtime_ns']:
                        raise ValueError('Sealed geometry bytes changed')
            else:
                prior={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=C.sha(p)) for p in self.extra_paths}
                C.save(path,prior)
            result.update({k:v['sha256'] for k,v in prior.items()})
        return result

    def close(self):
        super().close(); self.geometry_table=None; self.geometry_index=None; self.gpu_geometry=None


def network(arm,seed,initialize=True):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if arm=='T2':
        import cnh_readout_pilot2_model as M
        net=M.T2(); initial=None
    else:
        net=OLD_MODEL.CVR()
        initial=C.OLD/f'models/V/seed{seed}/model.pt' if arm=='V_retest' else C.ROOT/f'artifacts.local/work/cnh-margin-labels-20261002/models/M3/model_seed{seed}.pt'
        if initialize: net.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True))
    return net.cuda(),initial


def forward(net,arm,batch):
    if arm in ('V_retest','VD','M3'): return net(OLD_MODEL.prepare_voxels(batch['voxels']))
    return net(batch['histories'],batch['transforms'],batch['length'].long(),batch['ambient'],batch['query_weights'])


def build_soft_labels():
    C.check_budget();destination=C.OUT/'teacher/soft_labels.npy'
    if destination.exists(): raise FileExistsError('Teacher labels already exist')
    teacher_path=C.OUT/'teacher/train_scores.npz'
    with np.load(teacher_path) as z:
        llr=z['llr'];units=z['units'];frames=z['frames']
    with np.load(C.input_folder('train')/'rows.npz') as z: rows={k:z[k] for k in z.files}
    source=C.input_folder('train');labels=np.load(source/'labels.npy');mask=np.load(source/'mask.npy')
    result=labels.copy();unitmap={int(u):i for i,u in enumerate(units)};framemap={int(f):i for i,f in enumerate(frames)}
    expected=(len(units),2,len(frames))
    if llr.shape!=expected or not np.isfinite(llr).all(): raise ValueError('Teacher shape/finiteness')
    supervised=[];prob=[];hard=[]
    for row in np.flatnonzero(rows['domain']==1):
        group=int(rows['group'][row])
        if not mask[row,group]: continue
        score=llr[unitmap[int(rows['unit'][row])],int(rows['replica'][row]),framemap[int(rows['frame'][row])]]
        q=float(expit(score));result[row,group]=.5*(labels[row,group]+q)
        supervised.append(int(row));prob.append(q);hard.append(float(labels[row,group]))
    if len(supervised)!=9022: raise ValueError(f'Unexpected teacher effective rows {len(supervised)}')
    modified=result!=labels
    allowed=np.zeros_like(modified)
    for row in supervised: allowed[row,int(rows['group'][row])]=True
    if np.any(modified&~allowed): raise ValueError('Teacher changed forbidden label')
    np.save(destination,result)
    q=np.asarray(prob);y=np.asarray(hard)
    receipt=dict(status='COMPLETE',effective_target_queries=len(supervised),rows=len(labels),
        nontrivial_q_01_to99=float(np.mean((q>.01)&(q<.99))),teacher_hard_agreement=float(np.mean((q>=.5)==y)),
        normalized_loss='BCE((hard+q)/2), original weights unchanged',mask_preserved=True,
        hard_label_sha256=C.sha(source/'labels.npy'),teacher_sha256=C.sha(teacher_path),soft_label_sha256=C.sha(destination),plan_sha256=C.plan_sha())
    C.save(destination.with_suffix('.json'),receipt);return receipt


def fit_metrics(net,arm,data,batch_size=128):
    # Always hard labels, final weights, eval-mode model; no teacher objective.
    source=C.input_folder('train');labels=np.load(source/'labels.npy',mmap_mode='r')
    mask=np.load(source/'mask.npy',mmap_mode='r');weights=np.load(source/'weights.npy',mmap_mode='r')
    values=np.empty((data.n,2),np.float32);loss=0.;net.eval()
    with torch.inference_mode():
        for offset in range(0,data.n,batch_size):
            ids=np.arange(offset,min(offset+batch_size,data.n));batch=data.batch(ids,'cuda');logits=forward(net,arm,batch).float()
            target=torch.from_numpy(np.array(labels[ids],copy=True)).cuda()
            effective=torch.from_numpy(np.array(mask[ids]*weights[ids],copy=True)).cuda()
            loss+=float((F.binary_cross_entropy_with_logits(logits,target,reduction='none')*effective).sum())
            values[ids]=logits.cpu().numpy()
    def auc(y,s):
        y=np.asarray(y)==1;n1=int(y.sum());n0=len(y)-n1
        if not n1 or not n0: raise ValueError('Fit AUC missing a class')
        return float((rankdata(s)[y].sum()-n1*(n1+1)/2)/(n1*n0))
    valid=np.asarray(mask)>0
    return dict(hard_weighted_loss=loss/data.n,pooled_auc=auc(labels[valid],values[valid]),
        query_auc=[auc(labels[:,q][valid[:,q]],values[:,q][valid[:,q]]) for q in range(2)],valid_queries=int(valid.sum()),rows=data.n)


def baseline_fit():
    C.check_budget();TR.configure('cuda');path=C.OUT/'V_final_fit.json'
    if path.exists(): return C.read(path)
    data=Inputs('train','V_retest',training=True);before=data.hashes();cache=data.enable_gpu_cache('cuda');cells=[]
    try:
        for seed in C.SEEDS:
            net,initial=network('V_retest',seed);metrics=fit_metrics(net,'V_retest',data)
            cells.append(dict(seed=seed,model_sha256=C.sha(initial),**metrics));del net
        result=dict(status='COMPLETE',cells=cells,mean_hard_weighted_loss=float(np.mean([x['hard_weighted_loss'] for x in cells])),input_sha256=before,gpu_cache=cache)
        C.save(path,result);return result
    finally: data.close();gc.collect();torch.cuda.empty_cache()


def smoke(arm):
    C.check_budget();TR.configure('cuda');data=Inputs('smoke',arm,training=True);net,_=network(arm,0)
    try:
        batch=data.batch(np.arange(min(64,data.n)),'cuda');before=data.hashes();net.train()
        logits=forward(net,arm,batch);loss=TR.loss_for(logits,batch)
        if logits.shape!=(len(batch['labels']),2) or not bool(torch.isfinite(logits).all()): raise ValueError('Smoke logits')
        padding_delta=0.
        if arm=='T2':
            net.eval()
            with torch.inference_mode():
                reference=forward(net,arm,batch)
                altered=dict(batch);altered['histories']=batch['histories'].clone()
                padding=torch.arange(8,device='cuda')[None]<8-batch['length'][:,None]
                altered['histories'][padding]=123
                padding_delta=float((reference-forward(net,arm,altered)).abs().max())
            if padding_delta>1e-6:raise ValueError('Padding affected T2')
        optimizer=torch.optim.AdamW(net.parameters(),lr=.002 if arm=='T2' else .0003)
        updates=0;total=0.;net.train();start=time.monotonic()
        for offset in range(0,data.n,64):
            ids=np.arange(offset,min(offset+64,data.n));batch=data.batch(ids,'cuda')
            optimizer.zero_grad(set_to_none=True);loss=TR.loss_for(forward(net,arm,batch),batch)
            if loss is None:continue
            loss.backward()
            if not all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None):raise ValueError('Smoke gradients')
            optimizer.step();updates+=1;total+=float(loss.detach())*len(ids)
        if data.hashes()!=before:raise ValueError('Smoke input changed')
        parameters=sum(p.numel() for p in net.parameters())
        if parameters>C.load_plan()['parameter_limit']:raise ValueError('Model parameter cap')
        if arm=='T2' and parameters!=C.load_plan()['T2_model_config']['parameters']:raise ValueError('T2 configuration drift')
        result=dict(status='PASS',arm=arm,loss=total/data.n,epochs=1,updates=updates,seconds=time.monotonic()-start,padding_max_abs=padding_delta,parameters=parameters,finite_gradients=True,input_sha256=before)
        C.save(C.OUT/f'smoke_{arm}.json',result);return result
    finally:data.close();del net;gc.collect();torch.cuda.empty_cache()


def train(arm,seed):
    C.check_budget();TR.configure('cuda');p=C.load_plan();cfg=p['T2_recipe'] if arm=='T2' else p
    if C.read(C.OUT/f'smoke_{arm}.json')['status']!='PASS': raise ValueError('Smoke not passed')
    if arm=='T2' and C.read(C.OUT/'geometry_pulse_check_v2.json')['status']!='PASS':raise ValueError('Geometry pulse acceptance missing')
    if arm=='T2' and seed>0 and C.read(C.OUT/'t2_fit_acceptance.json')['status']!='T2_FIT': raise ValueError('T2 seed0 did not fit')
    folder=C.OUT/f'models/{arm}/seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model.pt').exists(): raise FileExistsError('Completed model exists')
    data=Inputs('train',arm,training=True);before=data.hashes();cache=data.enable_gpu_cache('cuda')
    net,initial=network(arm,seed);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs']);rng=np.random.default_rng(seed);history=[];start=time.monotonic();plan=C.plan_sha()
    source_sha={str(Path(__file__).resolve()):C.sha(__file__)}
    if arm=='T2':
        import cnh_readout_pilot2_model as M
        source_sha[str(Path(M.__file__).resolve())]=C.sha(M.__file__)
    try:
        for epoch in range(cfg['epochs']):
            t0=time.monotonic();order=rng.permutation(data.n);net.train();total=0.;seen=0
            for offset in range(0,data.n,cfg['batch']):
                ids=np.sort(order[offset:offset+cfg['batch']]);batch=data.batch(ids,'cuda');opt.zero_grad(set_to_none=True)
                loss=TR.loss_for(forward(net,arm,batch),batch)
                if loss is None:continue
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
                loss.backward();opt.step();total+=float(loss.detach())*len(ids);seen+=len(ids)
            scheduler.step();history.append(dict(epoch=epoch+1,loss=total/max(1,seen),seconds=time.monotonic()-t0,order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            C.save(folder/'progress.json',dict(status='TRAINING',arm=arm,seed=seed,history=history,seconds=time.monotonic()-start))
            print(arm,seed,epoch+1,history[-1]['loss'],history[-1]['seconds'],flush=True)
        if data.hashes()!=before or C.plan_sha()!=plan or any(C.sha(path)!=v for path,v in source_sha.items()):raise ValueError('Frozen inputs/plan/code changed')
        fit=fit_metrics(net,arm,data) if arm=='T2' and seed==0 else None
        checkpoint=folder/'model.pt';torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},checkpoint)
        receipt=dict(status='COMPLETE',arm=arm,seed=seed,rows=data.n,epochs=cfg['epochs'],batch_size=cfg['batch'],learning_rate=cfg['lr'],history=history,seconds=time.monotonic()-start,model_sha256=C.sha(checkpoint),plan_sha256=plan,input_sha256=before,source_sha256=source_sha,gpu_input_cache=cache,fit=fit,baseline_sha256=None if initial is None else C.sha(initial))
        C.save(folder/'training_receipt.json',receipt)
        if fit is not None:
            baseline=C.read(C.OUT/'V_final_fit.json')['mean_hard_weighted_loss'];passed=fit['hard_weighted_loss']<=2*baseline and fit['pooled_auc']>=.85
            C.save(C.OUT/'t2_fit_acceptance.json',dict(status='T2_FIT' if passed else 'T2_UNFIT',seed=0,baseline_mean_loss=baseline,loss_upper=2*baseline,auc_lower=.85,**fit))
        return receipt
    finally:data.close();del net,opt,scheduler;gc.collect();torch.cuda.empty_cache()


def infer(arm,seed,split):
    C.check_budget();TR.configure('cuda');began=time.monotonic();target=C.OUT/f'scores/{split}/{arm}_seed{seed}.npy';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): raise FileExistsError('Frozen scores exist')
    data=Inputs(split,arm);before=data.hashes();cache=data.enable_gpu_cache('cuda')
    net,initial=network(arm,seed,initialize=arm=='V_retest')
    if arm!='V_retest':
        model=C.OUT/f'models/{arm}/seed{seed}/model.pt';receipt=C.read(model.parent/'training_receipt.json')
        if receipt['status']!='COMPLETE' or C.sha(model)!=receipt['model_sha256']:raise ValueError('Model integrity')
        net.load_state_dict(torch.load(model,map_location='cuda',weights_only=True));initial=model
    net.eval();result=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(data.n,2))
    try:
        with torch.inference_mode():
            for offset in range(0,data.n,64):
                ids=np.arange(offset,min(offset+64,data.n));result[ids]=forward(net,arm,data.batch(ids,'cuda')).float().cpu().numpy()
        result.flush()
        if not np.isfinite(result).all() or data.hashes()!=before:raise ValueError('Scores invalid or inputs changed')
        receipt=dict(status='COMPLETE',arm=arm,seed=seed,split=split,rows=data.n,seconds=time.monotonic()-began,raw_sha256=C.sha(target),model_sha256=C.sha(initial),input_sha256=before,label_access=False,gpu_input_cache=cache)
        C.save(target.with_suffix('.json'),receipt);return receipt
    finally:del result,net;data.close();gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['soft-labels','baseline-fit','smoke','train','infer'])
    parser.add_argument('--arm',choices=['VD','T2','V_retest'],default='VD');parser.add_argument('--seed',type=int,choices=C.SEEDS,default=0);parser.add_argument('--split',default='fresh_evaluation');args=parser.parse_args();C.setup()
    if args.command=='soft-labels':result=build_soft_labels()
    elif args.command=='baseline-fit':result=baseline_fit()
    elif args.command=='smoke':result=smoke(args.arm)
    elif args.command=='train':result=train(args.arm,args.seed)
    else:result=infer(args.arm,args.seed,args.split)
    print(result,flush=True)

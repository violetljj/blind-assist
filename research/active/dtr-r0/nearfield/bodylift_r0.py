"""Frozen RGB appearance + matched depth-lift/query heads; no native runtime input."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from body_query_model import BodyQueryModel,near_from_counts
from body_query_data import QueryRGB,truth,read,write,sha,fresh_output
from body_query_train import metrics
from city_dev_selection import select_threshold


class BodyLiftReadout(nn.Module):
    def __init__(self,base,lift):
        super().__init__();self.lift=lift
        self.point=copy.deepcopy(base.query_point).requires_grad_(True)
        self.count=copy.deepcopy(base.query_readout).requires_grad_(True)
        self.register_buffer('projection',base.query_projection.clone())
        self.register_buffer('valid',base.query_valid.clone());self.register_buffer('xyz',base.query_xyz.clone())
        self.register_buffer('bin',((base.query_xyz[...,0]*3.18)/.3).long().clamp(0,15).flatten())
        self.depth=nn.Conv2d(64,16,1).to(base.query_xyz.device)
        nn.init.zeros_(self.depth.weight);nn.init.zeros_(self.depth.bias)
        self.depth.requires_grad_(lift)

    def forward(self,context):
        flat=context.flatten(2);depth=self.depth(context)
        if self.lift:
            alpha=depth.softmax(1).flatten(2)*16
            sampled=torch.zeros((len(context),64,324),device=context.device,dtype=context.dtype)
            for k in torch.unique(self.bin).tolist():
                ids=torch.where(self.bin==k)[0]
                sampled[:,:,ids]=torch.matmul(flat*alpha[:,k,None],self.projection[ids].T)
        else:sampled=torch.matmul(flat,self.projection.T)
        sampled=sampled.transpose(1,2).reshape(-1,12,27,64)
        feature=self.point(torch.cat((sampled,self.xyz[None].expand(len(context),-1,-1,-1)),-1))
        pooled=(feature*self.valid[None,:,:,None]).sum(2)/self.valid.sum(1)[None,:,None]
        counts=self.count(pooled);return near_from_counts(counts),counts,depth


def context_rgb(base,rgb):
    deep,shallow=base.extract((rgb-base.image_mean)/base.image_std)
    deep=F.interpolate(base.deep_projection(deep),size=(18,32),mode='bilinear',align_corners=False)
    return torch.cat((deep,base.detail(shallow)),1)


def run(a):
    out=fresh_output(a.output);started=time.perf_counter()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    checkpoint=a.baseline/'B-step2000.pt';assert sha(checkpoint)==read(a.baseline/'selection.json')['B']['checkpoint_sha256']
    base=BodyQueryModel(a.pretrained,'B').cuda().eval();base.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));base.requires_grad_(False)
    contexts={};supports={};targets={};records={};histograms={};weights={};checks={};native_stats={}
    world=read(a.world);assert sha(a.world)==read(a.cache/'manifest.json')['world_verification_sha256']
    # No evaluator labels are exposed to the trainable forward path.
    for role in ('train','dev','eval'):
        data=QueryRGB(a.cache,role);chunks=[]
        with torch.inference_mode():
            for begin in range(0,len(data.ids),32):chunks.append(context_rgb(base,data.tensor(np.arange(begin,min(begin+32,len(data.ids))),'cuda')).cpu())
        contexts[role]=torch.cat(chunks).cuda();saved=np.load(a.baseline/f'B-{role}.npz');supports[role]=saved['support']
        rec,gt=truth(a.cache,role);records[role]=rec;targets[role]=gt
        hists=[];ws=[];stats=dict(zero=0,nonfinite=0,negative=0,ge100=0,valid=0)
        for r in rec['records']:
            index=r['sample_index'];path=a.native/f'{index:04d}.npy';assert sha(path)==world['rows'][index]['native_sha256'];x=torch.from_numpy(np.load(path)).cuda()
            valid=torch.isfinite(x)&(x>0)&(x<100)
            stats['zero']+=int((x==0).sum());stats['nonfinite']+=int((~torch.isfinite(x)).sum());stats['negative']+=int((x<0).sum());stats['ge100']+=int((x>=100).sum());stats['valid']+=int(valid.sum())
            bins=(x.clamp(0,100)/.3).long().clamp(0,15)
            h=torch.stack([F.avg_pool2d(((bins==k)&valid).float()[None,None],20)[0,0] for k in range(16)])
            weight=h.sum(0);hists.append(h/weight.clamp_min(1e-8));ws.append(weight)
        histograms[role]=torch.stack(hists);weights[role]=torch.stack(ws);native_stats[role]=stats
        np.savez_compressed(out/f'{role}-depth-labels.npz',histogram=histograms[role].cpu().numpy(),weight=weights[role].cpu().numpy())
        # Exact scale-matched initialization check for both paths.
        checks[role]={}
        for lift in (False,True):
            test=BodyLiftReadout(base,lift).cuda().eval()
            with torch.no_grad():n,_,_=test(contexts[role])
            error=float(abs(n.sigmoid().cpu().numpy()-saved['near']).max());assert error<2e-5;checks[role][str(lift)]=error
            del test
    np.savez_compressed(out/'frozen-contexts.npz',**{k:v.cpu().numpy() for k,v in contexts.items()})
    schedule=np.random.default_rng(17).integers(0,240,(2000,32));assert np.array_equal(schedule,np.load(a.baseline/'schedule.npy'))
    train_near=torch.from_numpy(targets['train']['near']).cuda().float();train_counts=torch.from_numpy(targets['train']['counts']).cuda().long()
    fits={};models={}
    for arm,lift in [('CONTROL',False),('LIFT',True)]:
        torch.manual_seed(17);model=BodyLiftReadout(base,lift).cuda();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);history=[];t=time.perf_counter()
        for step,indices in enumerate(schedule,1):
            ids=torch.from_numpy(indices).cuda();n,c,d=model(contexts['train'][ids])
            ln=F.binary_cross_entropy_with_logits(n,train_near[ids]);lc=-c.log_softmax(-1).gather(-1,train_counts[ids,:,None]).mean()
            ld=(-(d.log_softmax(1)*histograms['train'][ids]).sum(1)*weights['train'][ids]).sum()/weights['train'][ids].sum()
            loss=ln+.25*lc+(.25*ld if lift else 0)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%200==0:
                row=dict(arm=arm,step=step,near=float(ln.detach()),query=float(lc.detach()),depth=float(ld.detach()));history.append(row);write(out/'progress.json',row);print(row,flush=True)
        torch.cuda.synchronize();torch.save(model.state_dict(),out/f'{arm}-step2000.pt');fits[arm]=dict(seconds=time.perf_counter()-t,history=history,checkpoint_sha256=sha(out/f'{arm}-step2000.pt'));models[arm]=model.eval();del opt
    selection={};predictions={};depth_metrics={}
    for arm,model in models.items():
        depth_metrics[arm]={}
        with torch.no_grad():
            for role in ('train','dev','eval'):
                ns=[];cs=[];ds=[]
                for begin in range(0,len(contexts[role]),32):
                    n,c,d=model(contexts[role][begin:begin+32]);ns.append(n.sigmoid().cpu().numpy());cs.append(c.softmax(-1).cpu().numpy());ds.append(d.cpu())
                predictions[arm,role]=dict(near=np.concatenate(ns),counts=np.concatenate(cs),support=supports[role]);depth=torch.cat(ds).cuda()
                h=histograms[role];w=weights[role];log=depth.log_softmax(1)
                depth_metrics[arm][role]={key:float((-(log*h).sum(1)*w*mask).sum()/(w*mask).sum().clamp_min(1e-8)) for key,mask in [('all',torch.ones_like(w)),('near_surface_tiles',h[:,:15].sum(1)>.5)]}
        selection[arm]=[select_threshold(predictions[arm,'dev']['near'][:,h],targets['dev']['near'][:,h],min_count=8)['threshold'] for h in range(2)]
    write(out/'selection.json',selection);selected=sha(out/'selection.json');result=dict(fits=fits,arms={},depth_CE=depth_metrics,initial_checks=checks,native_stats=native_stats)
    for arm in models:
        result['arms'][arm]={}
        for role in ('train','dev','eval'):
            pred=predictions[arm,role];np.savez_compressed(out/f'{arm}-{role}.npz',**pred);result['arms'][arm][role]=metrics(pred,targets[role],records[role],selection[arm])
    assert selected==sha(out/'selection.json')
    # End-to-end RGB-derived context equivalence, no native runtime argument.
    with torch.no_grad():
        sample=QueryRGB(a.cache,'eval').tensor(np.array([0]),'cuda');n,_,_=models['LIFT'](context_rgb(base,sample));delta=float(abs(n.sigmoid().cpu().numpy()-predictions['LIFT','eval']['near'][:1]).max());assert delta<2e-5
    result['rgb_cache_equivalence']=delta;result['seconds']=time.perf_counter()-started;write(out/'result.json',result)
    assert sha(checkpoint)==read(a.baseline/'selection.json')['B']['checkpoint_sha256']
    write(out/'receipt.json',dict(status='PASS',fits=2,steps_each=2000,frozen_B_sha256=sha(checkpoint),code_sha256=sha(Path(__file__)),result_sha256=sha(out/'result.json'),selection_sha256=selected,backend='CUDA',device=torch.cuda.get_device_name()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('cache','baseline','pretrained','world','native','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

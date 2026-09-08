"""Frozen representation probe with geometry-only control and explicit unknowns."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,time
import numpy as np
import torch
import torch.nn.functional as F
from body_query_model import BodyQueryModel,fixed_projection,projection_weights
from body_query_data import QueryRGB,truth,read,write,sha,fresh_output


def auc(x,y):
    pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:return None
    order=np.argsort(x);xs=x[order];starts=np.r_[0,np.flatnonzero(np.diff(xs))+1];ends=np.r_[starts[1:],len(x)]
    ranks=np.empty(len(x));ranks[order]=np.repeat((starts+1+ends)/2,ends-starts)
    return float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg))


def cutoff(x,y):
    order=np.argsort(-x);xx=x[order];yy=y[order];ends=np.r_[np.flatnonzero(np.diff(xx))+1,len(x)]-1
    tp=np.cumsum(yy)[ends];fp=np.cumsum(~yy)[ends];ok=fp<=.05*(~y).sum()
    if not ok.any():return float(x.max()+1)
    ids=np.flatnonzero(ok);best=sorted(ids,key=lambda j:(-tp[j],fp[j],-xx[ends[j]]))[0]
    return float(xx[ends[best]])


def score(x,y,mask,threshold,records):
    def row(m):
        xx=x[m];yy=y[m];p=xx>=threshold;half=xx>=0
        return dict(positive=int(yy.sum()),negative=int((~yy).sum()),AUC=auc(xx,yy),
            TP=int((p&yy).sum()),FP=int((p&~yy).sum()),TP_at05=int((half&yy).sum()),FP_at05=int((half&~yy).sum()))
    result={'all':row(mask),'heads':{},'conditions':{}}
    for h,name in enumerate(('BODY','HEAD')):
        result['heads'][name]={}
        for label,start,end in [('all',0,6),('near',0,3),('far',3,6)]:
            m=np.zeros_like(mask);m[:,h*6+start:h*6+end]=mask[:,h*6+start:h*6+end];result['heads'][name][label]=row(m)
    conditions=np.array([r['condition'] for r in records])
    for c in dict.fromkeys(conditions):
        m=mask&(conditions==c)[:,None,None]
        result['conditions'][c]={'all':row(m),'HEAD':row(m&(np.arange(12)>=6)[None,:,None])}
    return result


def run(a):
    out=fresh_output(a.output);started=time.perf_counter()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    checkpoint=a.r1/'R1-step2000.pt';assert sha(checkpoint)==read(a.r1/'receipt.json')['checkpoint_sha256']
    model=BodyQueryModel(a.pretrained,'B').cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));model.requires_grad_(False)
    captured={}
    def hook(module,inputs,output):captured['features']=output
    handle=model.query_point.register_forward_hook(hook)
    features={'B':{},'R1':{},'XYZ':{}};checks={};inputs={str(checkpoint):sha(checkpoint)}
    grid,valid,xyz=fixed_projection();projection=projection_weights(grid).cuda().reshape(12,27,576)
    try:
        for role in ('train','dev','eval'):
            path=a.bpoints/f'{role}-points.npz';assert sha(path)==read(a.bpoints/'receipt.json')['arrays'][path.name]
            inputs[str(path)]=sha(path);features['B'][role]=np.load(path)['point_features']
            data=QueryRGB(a.cache,role);chunks=[];near=[]
            with torch.inference_mode():
                for i in range(0,len(data.ids),32):
                    n,_,_=model(data.tensor(np.arange(i,min(i+32,len(data.ids))),'cuda'))
                    chunks.append(captured['features'].cpu().numpy());near.append(n.sigmoid().cpu().numpy())
            features['R1'][role]=np.concatenate(chunks);frozen=np.load(a.r1/f'R1-{role}.npz')['near']
            error=float(abs(np.concatenate(near)-frozen).max());assert error<=1e-5;checks[role]=error
            np.savez_compressed(out/f'R1-{role}-features.npz',features=features['R1'][role])
            features['XYZ'][role]=np.broadcast_to(xyz.numpy(),(*features['R1'][role].shape[:3],3)).copy()
    finally:handle.remove()
    del model;captured.clear();torch.cuda.empty_cache()
    labels={};records={};world=read(a.world);q3receipt=read(a.ownership/'receipt.json')
    for role in ('train','dev','eval'):
        rec,_=truth(a.cache,role);records[role]=rec['records'];path=a.ownership/f'{role}-ownership.npz'
        assert sha(path)==q3receipt['arrays'][path.name];inputs[str(path)]=sha(path);d=np.load(path)
        masks=[]
        for r in records[role]:
            idx=r['sample_index'];path=a.native/f'{idx:04d}.npy';assert sha(path)==world['rows'][idx]['native_sha256']
            native=torch.from_numpy(np.load(path)).cuda();unknown=~(torch.isfinite(native)&(native>0)&(native<100))
            u=F.max_pool2d(unknown.float()[None,None],20).flatten()
            masks.append(((torch.einsum('cpk,k->cp',projection,u)==0)&valid.cuda()).cpu().numpy())
        local_mask=np.stack(masks);labels[role]={'ray':(d['owned'].astype(bool),d['known'].astype(bool)),
            'local':(d['local']>=.5,local_mask)}
        np.savez_compressed(out/f'{role}-labels.npz',ray=d['owned'],ray_mask=d['known'],local=d['local']>=.5,local_mask=local_mask)
    results={};selection={};all_logits={};history={}
    # No DEV/EVAL fitting: gradients use only explicitly indexed TRAIN points.
    for target in ('ray','local'):
        results[target]={};selection[target]={}
        y,m=labels['train'][target]
        for name in ('B','R1','XYZ'):
            x=torch.from_numpy(features[name]['train'][m]).cuda();yy=torch.from_numpy(y[m].astype(np.float32)).cuda()
            mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(1e-6);x=(x-mean)/std
            layer=torch.nn.Linear(x.shape[-1],1).cuda();torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
            opt=torch.optim.Adam(layer.parameters(),lr=.03);hist=[]
            for step in range(1,1001):
                loss=F.binary_cross_entropy_with_logits(layer(x).squeeze(-1),yy)+1e-4*layer.weight.square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite probe loss')
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if step in (1,100,500,1000):hist.append(dict(step=step,loss=float(loss.detach())))
            torch.save(dict(state=layer.state_dict(),mean=mean,std=std),out/f'{target}-{name}-probe.pt')
            logits={}
            with torch.no_grad():
                for role in ('train','dev','eval'):
                    z=torch.from_numpy(features[name][role]).cuda();logits[role]=layer((z-mean)/std).squeeze(-1).cpu().numpy()
            dy,dm=labels['dev'][target];threshold=cutoff(logits['dev'][dm],dy[dm]);selection[target][name]=threshold
            all_logits[target,name]=logits;history[target+'-'+name]=hist
            print(target,name,hist[-1],flush=True)
    write(out/'selection.json',selection);selection_hash=sha(out/'selection.json')
    for target in ('ray','local'):
        for name in ('B','R1','XYZ'):
            results[target][name]={};np.savez_compressed(out/f'{target}-{name}-logits.npz',**all_logits[target,name])
            for role in ('train','dev','eval'):
                y,m=labels[role][target];results[target][name][role]=score(all_logits[target,name][role],y,m,selection[target][name],records[role])
    assert selection_hash==sha(out/'selection.json') and sha(checkpoint)==inputs[str(checkpoint)]
    write(out/'result.json',dict(results=results,history=history,seconds=time.perf_counter()-started,near_reproduction=checks))
    write(out/'receipt.json',dict(status='PASS',main_training_steps=0,probe_fits=6,steps_per_probe=1000,R1_inference_frames=320,
        backend='CUDA',device=torch.cuda.get_device_name(),inputs=inputs,code_sha256=sha(Path(__file__)),
        result_sha256=sha(out/'result.json'),selection_sha256=selection_hash,
        arrays={p.name:sha(p) for p in out.glob('*.npz')}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('cache','bpoints','r1','ownership','native','world','pretrained','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

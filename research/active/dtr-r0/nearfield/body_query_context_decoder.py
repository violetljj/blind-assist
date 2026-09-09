"""Frozen RGB-feature native-count decoders: local versus joint spatial context."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from body_query_model import BodyQueryModel, near_from_counts
from body_query_data import QueryRGB, truth, read, write, sha, fresh_output
from body_query_train import digest, metrics
from city_dev_selection import select_threshold

BASE_SHA = 'db39ccfcb9f3d1c8f3722711bab314ff0cf2da708e00d91f7fb263e6fec0ece0'


class CountDecoder(nn.Module):
    def __init__(self, kind, xyz):
        super().__init__()
        self.kind = kind
        self.register_buffer('xyz', xyz.detach().clone())
        self.net = nn.Sequential(nn.Linear(768 if kind == 'JOINT' else 67, 128),
                                 nn.GELU(), nn.Linear(128, 48 if kind == 'JOINT' else 4))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([.94,.02,.02,.02]).log().repeat(12 if kind == 'JOINT' else 1))

    def forward(self, x):
        if x.shape[1:] != (12,64):
            raise ValueError('Expected Bx12x64 frozen visual features')
        if self.kind == 'JOINT':
            return self.net(x.flatten(1)).reshape(-1,12,4)
        if self.kind == 'PRIOR':
            x = torch.zeros_like(x)
        return self.net(torch.cat((x, self.xyz[None].expand(len(x),-1,-1)), -1))


@torch.inference_mode()
def extract(model, cache, role, baseline, output):
    data = QueryRGB(cache, role)
    captured = {}
    hook = model.query_point.register_forward_pre_hook(lambda m, args: captured.update(raw=args[0]))
    features=[]
    saved=np.load(baseline/f'NEW-{role}.npz',allow_pickle=False)
    parity=0.
    try:
        for start in range(0,len(data.ids),32):
            ids=np.arange(start,min(start+32,len(data.ids)))
            near,support,counts=model(data.tensor(ids,'cuda'))
            for key,val in [('near',near.sigmoid()),('support',support.sigmoid()),('counts',counts.softmax(-1))]:
                parity=max(parity,float((val-torch.as_tensor(saved[key][ids],device='cuda')).abs().max()))
            mask=model.query_valid[None,:,:,None]
            raw=(captured['raw'][...,:64]*mask).sum(2)/mask.sum(2).clamp_min(1)
            features.append(raw.cpu().numpy())
        assert parity<2e-6,parity
        x=np.concatenate(features)
        np.save(output/f'features-{role}.npy',x,allow_pickle=False)
        write(output/f'features-{role}.json',dict(rows=len(x),ids=data.ids,parity=parity,
              rgb_manifest_sha256=sha(cache/'manifest.json'),feature_sha256=sha(output/f'features-{role}.npy')))
        return x
    finally:
        hook.remove()


def score(pred,target,record,thresholds):
    # Import only the established independent three-cell probability algebra.
    from body_query_10000_readout_analysis import range_event_from_counts,native_range_events,confusion
    result=metrics(pred,target,record,thresholds)
    events=range_event_from_counts(pred['counts']); native=native_range_events(target['counts'])
    near_only=native[:,1,0]&~native[:,1,1]
    result['wrong_far_on_HEAD_near_only']=dict(fired=int((events[near_only,1,1]>=.5).sum()),total=int(near_only.sum()))
    result['range_events']={};result['query_strata']={}
    for h,head in enumerate(('BODY','HEAD')):
        for d,dist in enumerate(('near','far')):
            key=f'{head}_{dist}';sl=slice(h*6+d*3,h*6+d*3+3)
            result['range_events'][key]=confusion(events[:,h,d],native[:,h,d],.5)
            result['query_strata'][key]=confusion(1-pred['counts'][:,sl,0],target['counts'][:,sl]>0,.5)
    return result


@torch.inference_mode()
def infer(decoder,x,support):
    counts=[];near=[]
    for begin in range(0,len(x),32):
        c=decoder(x[begin:begin+32]);counts.append(c.softmax(-1).cpu().numpy());near.append(near_from_counts(c).sigmoid().cpu().numpy())
    return dict(counts=np.concatenate(counts),near=np.concatenate(near),support=support)


def run(a):
    out=fresh_output(a.output);started=time.perf_counter();stage='preflight';completed={}
    try:
        assert torch.cuda.is_available() and sha(a.baseline/'NEW-step2000.pt')==BASE_SHA
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        manifest_sha=sha(a.cache/'manifest.json')
        protocol=dict(scope='Consumed controlled Development; raw RGB-derived features only',
            baseline_sha256=BASE_SHA,cache_sha256=manifest_sha,seed=17,steps=2000,batch=32,
            lr=.001,weight_decay=.0001,loss='unweighted native count CE only',
            arms=['LOCAL','JOINT','PRIOR'],feature='masked mean 64 channels BEFORE query_point; TRAIN channel normalization',
            source_sha256=sha(Path(__file__)),dependencies={n:sha(Path(__file__).with_name(n)) for n in
              ['body_query_model.py','body_query_data.py','body_query_train.py','city_dev_selection.py','body_query_10000_readout_analysis.py']})
        write(out/'protocol.json',protocol)
        model=BodyQueryModel(a.pretrained,'B').cuda().eval()
        model.load_state_dict(torch.load(a.baseline/'NEW-step2000.pt',map_location='cuda',weights_only=True))
        for parameter in model.parameters():parameter.requires_grad_(False)
        before=digest(model.state_dict()); stage='extract TRAIN'
        raw=extract(model,a.cache,'train',a.baseline,out)
        mean=raw.mean((0,1),keepdims=True);std=raw.std((0,1),keepdims=True).clip(.01)
        np.savez(out/'normalization.npz',mean=mean,std=std)
        xyz=(model.query_xyz*model.query_valid[:,:,None]).sum(1)/model.query_valid.sum(1)[:,None].clamp_min(1)
        x=torch.as_tensor((raw-mean)/std,device='cuda');record,target=truth(a.cache,'train',training=True)
        assert record['sample_indices']==read(out/'features-train.json')['ids']
        y=torch.as_tensor(target['counts'],device='cuda')
        schedule=np.random.default_rng(17).integers(0,len(x),(2000,32))
        assert np.array_equal(schedule,np.load(a.baseline/'schedule.npy',allow_pickle=False))
        np.save(out/'schedule.npy',schedule,allow_pickle=False)
        fits={};decoders={}
        for arm in protocol['arms']:
            stage='fit '+arm;torch.manual_seed(17);torch.cuda.manual_seed_all(17)
            decoder=CountDecoder(arm,xyz).cuda();optimizer=torch.optim.AdamW(decoder.parameters(),lr=.001,weight_decay=.0001)
            timer=time.perf_counter();history=[]
            for step,indices in enumerate(schedule,1):
                ids=torch.as_tensor(indices,device='cuda');c=decoder(x[ids]);loss=-c.log_softmax(-1).gather(-1,y[ids,:,None].long()).mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();completed[arm]=step
                if step%200==0:
                    row=dict(arm=arm,step=step,loss=float(loss.detach()));history.append(row);write(out/'progress.json',row);print(row,flush=True)
            torch.cuda.synchronize();decoder.eval();torch.save(decoder.state_dict(),out/f'{arm}.pt')
            fits[arm]=dict(steps=2000,seconds=time.perf_counter()-timer,history=history,parameters=sum(p.numel() for p in decoder.parameters()),sha256=sha(out/f'{arm}.pt'))
            decoders[arm]=decoder
        write(out/'fits.json',fits)
        # All fits and final checkpoints precede DEV/EVAL feature or target access.
        xs={'train':x}
        stage='extract DEV'
        xs['dev']=torch.as_tensor((extract(model,a.cache,'dev',a.baseline,out)-mean)/std,device='cuda')
        selection={};result={'arms':{arm:{} for arm in ['BASE',*protocol['arms']]}}
        def get_pred(arm,role):
            base=dict(np.load(a.baseline/f'NEW-{role}.npz',allow_pickle=False))
            return base if arm=='BASE' else infer(decoders[arm],xs[role],base['support'])
        rec,gt=truth(a.cache,'dev')
        for arm in result['arms']:
            pred=get_pred(arm,'dev');np.savez_compressed(out/f'{arm}-dev.npz',**pred)
            heads=[select_threshold(pred['near'][:,h],gt['near'][:,h],min_count=48) for h in range(2)]
            selection[arm]=dict(thresholds=[v['threshold'] for v in heads],heads=heads)
            result['arms'][arm]['dev']=score(pred,gt,rec,selection[arm]['thresholds'])
        write(out/'selection.json',selection);selection_sha=sha(out/'selection.json')
        stage='extract EVAL'
        xs['eval']=torch.as_tensor((extract(model,a.cache,'eval',a.baseline,out)-mean)/std,device='cuda')
        for role in ('train','eval'):
            rec,gt=truth(a.cache,role,training=role=='train')
            for arm in result['arms']:
                pred=get_pred(arm,role);np.savez_compressed(out/f'{arm}-{role}.npz',**pred)
                result['arms'][arm][role]=score(pred,gt,rec,selection[arm]['thresholds'])
        assert digest(model.state_dict())==before and sha(a.cache/'manifest.json')==manifest_sha
        assert sha(out/'selection.json')==selection_sha
        write(out/'result.json',result)
        write(out/'receipt.json',dict(status='PASS',fits=3,steps_per_fit=2000,seconds=time.perf_counter()-started,
              baseline_state_unchanged=True,backend='CUDA',device=torch.cuda.get_device_name(),
              result_sha256=sha(out/'result.json'),selection_sha256=selection_sha,schedule_sha256=sha(out/'schedule.npy')))
        print('PASS',out,flush=True)
    except Exception as exc:
        write(out/'failure.json',dict(stage=stage,completed=completed,error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('cache','baseline','pretrained','output'):parser.add_argument('--'+key,type=Path,required=True)
    run(parser.parse_args())

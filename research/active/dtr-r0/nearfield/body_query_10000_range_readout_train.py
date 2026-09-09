"""One paired count-only fit with near/far-specific copied linear readouts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, time
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from body_query_model import BodyQueryModel
from body_query_range_readout import RangeReadout
from body_query_data import QueryRGB, truth, sha, write, read, fresh_output
from body_query_train import digest, metrics, predict, benchmark, INITIAL_SHA
from city_dev_selection import select_threshold
from city_data import pixel_support_bce
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]/'tools'))
from research_backend import torch_observation

OLD_SHA='db39ccfcb9f3d1c8f3722711bab314ff0cf2da708e00d91f7fb263e6fec0ece0'

def strata(pred,target):
    p=1-pred['counts'][:,:,0]; y=target['counts']>0; out={}
    for h,head in enumerate(('BODY','HEAD')):
        for d,name in enumerate(('near','far')):
            sl=slice(h*6+d*3,h*6+d*3+3); pos,fired=y[:,sl],p[:,sl]>=.5
            tp=int((pos&fired).sum()); fn=int((pos&~fired).sum())
            out[f'{head}_{name}']=dict(TP=tp,FN=fn,FP=int((~pos&fired).sum()),TN=int((~pos&~fired).sum()),recall=tp/(tp+fn) if tp+fn else None)
    return out

def evaluate(a,out):
    result={'arms':{}}; selection={}; checkpoints={'OLD':a.old,'NEW':out/'NEW-step2000.pt'}
    for arm,ckpt in checkpoints.items():
        model=BodyQueryModel(a.pretrained,'B').cuda()
        if arm=='NEW': model.query_readout=RangeReadout(model.query_readout)
        model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=True))
        data=QueryRGB(a.cache,'dev'); rec,gt=truth(a.cache,'dev'); pred=predict(model,data); np.savez_compressed(out/f'{arm}-dev.npz',**pred)
        hs=[select_threshold(pred['near'][:,h],gt['near'][:,h],min_count=48) for h in range(2)]
        selection[arm]=dict(thresholds=[x['threshold'] for x in hs],heads=hs,checkpoint_sha256=sha(ckpt))
        result['arms'][arm]={'dev':metrics(pred,gt,rec,selection[arm]['thresholds'])}; result['arms'][arm]['dev']['query_strata']=strata(pred,gt); del model
    write(out/'selection.json',selection); frozen=sha(out/'selection.json')
    for arm,ckpt in checkpoints.items():
        model=BodyQueryModel(a.pretrained,'B').cuda()
        if arm=='NEW': model.query_readout=RangeReadout(model.query_readout)
        model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=True))
        for role in ('train','eval'):
            data=QueryRGB(a.cache,role); rec,gt=truth(a.cache,role); pred=predict(model,data); np.savez_compressed(out/f'{arm}-{role}.npz',**pred)
            result['arms'][arm][role]=metrics(pred,gt,rec,selection[arm]['thresholds']); result['arms'][arm][role]['query_strata']=strata(pred,gt)
        result['arms'][arm]['runtime']=benchmark(model,data); del model
    assert sha(out/'selection.json')==frozen and sha(a.cache/'manifest.json')==read(out/'protocol.json')['cache_sha256']
    write(out/'result.json',result)

def run(a):
    assert torch.cuda.is_available(); assert sha(a.initial)==OLD_SHA and sha(a.old)==OLD_SHA
    cache=read(a.cache/'manifest.json'); assert cache['status']=='PASS' and [cache['partitions'][x]['frames'] for x in ('train','dev','eval')]==[5000,2000,3000]
    historical=read(a.old.parent/'protocol.json')['dependency_sha256']
    for name in ('body_query_model.py','city_data.py','decoupled_model.py','representation_model.py'): assert sha(Path(__file__).with_name(name))==historical[name]
    out=fresh_output(a.output); start=time.perf_counter(); torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    train=QueryRGB(a.cache,'train'); rec,target=truth(a.cache,'train',training=True); assert train.ids==rec['sample_indices']; schedule=np.random.default_rng(17).integers(0,5000,(2000,32)); np.save(out/'schedule.npy',schedule,allow_pickle=False)
    torch.manual_seed(17); torch.cuda.manual_seed_all(17); model=BodyQueryModel(a.pretrained,'B').cuda(); model.load_state_dict(torch.load(a.initial,map_location='cpu',weights_only=True)); initial=digest(model.state_dict()); model.query_readout=RangeReadout(model.query_readout)
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name.startswith('query_readout.'))
    model.eval()
    with torch.inference_mode():
        n0,s0,c0=model(train.tensor(np.arange(32),'cuda'))
        cached=np.load(a.old.parent/'NEW-train.npz',allow_pickle=False)
        parity=max(float((value-torch.tensor(cached[key][:32],device='cuda')).abs().max()) for key,value in [('near',n0.sigmoid()),('support',s0.sigmoid()),('counts',c0.softmax(-1))])
        assert parity<2e-6, parity
    assert np.array_equal(schedule,np.load(a.old.parent/'schedule.npy',allow_pickle=False))
    write(out/'preflight.json',dict(status='PASS',initial_sha256=sha(a.initial),baseline_sha256=sha(a.old),train_prediction_parity=parity,schedule_identical=True,trainable=[name for name,p in model.named_parameters() if p.requires_grad]))
    protocol=dict(seed=17,steps=2000,batch=32,lr=1e-5,weight_decay=1e-4,trainable='two range-specific query_readout maps only',frozen='all other parameters and BN buffers',loss='count CE only; near/far weights untied; body and lateral weights shared',cache_sha256=sha(a.cache/'manifest.json'),initial_sha256=sha(a.initial),old_sha256=sha(a.old),initial_digest=initial,schedule_sha256=sha(out/'schedule.npy'),source_sha256=sha(Path(__file__)),dependency_sha256={n:sha(Path(__file__).with_name(n)) for n in ('body_query_model.py','body_query_data.py','body_query_train.py','city_data.py','city_dev_selection.py','decoupled_model.py','representation_model.py','body_query_range_readout.py')},torch=torch.__version__,device=torch.cuda.get_device_name())
    write(out/'protocol.json',protocol); buffer_digest=digest(dict(model.named_buffers())); frozen_params={k:digest({k:v}) for k,v in model.named_parameters() if k!='query_readout.weight' and k!='query_readout.bias'}
    optimizer=torch.optim.AdamW(model.query_readout.parameters(),lr=1e-5,weight_decay=1e-4); targets={k:torch.from_numpy(v).cuda() for k,v in target.items()}; rgb=torch.from_numpy(train.rgb.copy()).cuda(); model.train()
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm): layer.eval()
    history=[]; completed=0; fit_start=time.perf_counter(); gradient_history=[]
    try:
        for step,indices in enumerate(schedule,1):
            ids=torch.from_numpy(indices).cuda(); n,s,c=model(rgb[ids].permute(0,3,1,2).float()/255.); ln=F.binary_cross_entropy_with_logits(n,targets['near'][ids].float()); ls=pixel_support_bce(s,targets['support'][ids]); lc=-c.log_softmax(-1).gather(-1,targets['counts'][ids].long().unsqueeze(-1)).mean(); loss=lc
            assert torch.isfinite(loss)
            if step==1 or step%100==0:
                pars=list(model.query_readout.parameters())
                ga=torch.cat([g.flatten() for g in torch.autograd.grad(ln,pars,retain_graph=True)])
                gc=torch.cat([g.flatten() for g in torch.autograd.grad(.25*lc,pars,retain_graph=True)])
                gradient_history.append(dict(step=step,alert_norm=float(ga.norm()),weighted_count_norm=float(gc.norm()),cosine=float(F.cosine_similarity(ga,gc,dim=0))))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('query_readout.'))
            if step==1: write(out/'actual-backend.json',asdict(torch_observation(model=model,output=(n,s,c)))); write(out/'first-gradients.json',{k:float(p.grad.norm()) for k,p in model.named_parameters() if p.grad is not None})
            optimizer.step(); completed=step
            if step%100==0: row=dict(step=step,count=float(lc.detach()),loss=float(loss.detach())); history.append(row); write(out/'progress.json',row); print(row,flush=True)
        torch.cuda.synchronize(); assert digest(dict(model.named_buffers()))==buffer_digest
        for k,v in model.named_parameters():
            if k in frozen_params: assert digest({k:v})==frozen_params[k]
        torch.save(model.state_dict(),out/'NEW-step2000.pt'); write(out/'fit-complete.json',dict(steps=2000,seconds=time.perf_counter()-fit_start,history=history,gradient_history=gradient_history,initial_digest=initial,checkpoint_sha256=sha(out/'NEW-step2000.pt'),trainable='two range-specific query_readout maps only',frozen_parameters_verified=True,buffers_verified=True))
        del model,optimizer,rgb,targets,n,s,c,loss,ln,ls,lc; torch.cuda.empty_cache(); evaluate(a,out); write(out/'receipt.json',dict(status='PASS',fits=1,steps=2000,seconds=time.perf_counter()-start,result_sha256=sha(out/'result.json'),selection_sha256=sha(out/'selection.json'),backend='CUDA',device=torch.cuda.get_device_name()))
    except Exception as exc: write(out/'failure.json',dict(error=repr(exc),completed_steps=completed)); raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('cache','pretrained','initial','old','output'): p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())




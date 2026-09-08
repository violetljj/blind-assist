"""One matched attribution-auxiliary challenger; runtime architecture unchanged."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from body_query_model import BodyQueryModel,projection_weights,fixed_projection
from body_query_data import QueryRGB,truth,read,write,sha,fresh_output
from body_query_train import INITIAL_SHA,SEED,STEPS,BATCH,digest,predict,metrics
from body_query_ownership_audit import membership
from city_data import pixel_support_bce
from city_dev_selection import select_threshold


def run(a):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    out=fresh_output(a.output);start=time.perf_counter()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    assert sha(a.initial)==INITIAL_SHA
    train=QueryRGB(a.cache,'train');rec,gt=truth(a.cache,'train',training=True)
    world=read(a.world);assert sha(a.world)==read(a.cache/'manifest.json')['world_verification_sha256']
    grid,valid,_=fixed_projection();projection=projection_weights(grid).cuda().reshape(12,27,576)
    local=[];masks=[];native_hashes={}
    for i,r in enumerate(rec['records']):
        index=r['sample_index'];path=a.native/f'{index:04d}.npy';h=sha(path)
        assert h==world['rows'][index]['native_sha256'];native_hashes[str(path)]=h
        cells,known,_=membership(torch.from_numpy(np.load(path)).cuda())
        assert np.array_equal(cells.sum((-2,-1)).clamp_max(3).cpu().numpy(),gt['counts'][i])
        target=torch.einsum('cpk,ck->cp',projection,F.max_pool2d(cells.float()[:,None],20).flatten(1))
        unknown=F.max_pool2d((~known).float()[None,None],20).flatten()
        mask=(torch.einsum('cpk,k->cp',projection,unknown)==0)&valid.cuda()
        local.append(target);masks.append(mask)
    local=torch.stack(local);masks=torch.stack(masks)
    np.savez_compressed(out/'train-attribution.npz',target=local.cpu().numpy(),mask=masks.cpu().numpy())
    schedule=np.random.default_rng(SEED).integers(0,len(train.ids),(STEPS,BATCH))
    assert np.array_equal(schedule,np.load(a.baseline/'schedule.npy'))
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=BodyQueryModel(a.pretrained,'B').cuda();model.initialize_g13(torch.load(a.initial,map_location='cpu',weights_only=True))
    initial=digest(model.state_dict());baseline=read(a.baseline/'B-fit-complete.json')
    assert initial==baseline['initial_digest']
    aux=torch.nn.Linear(32,1).cuda();captured={}
    def hook(module,inputs,output):captured['feature']=output
    handle=model.query_point.register_forward_hook(hook)
    optimizer=torch.optim.AdamW(list(model.parameters())+list(aux.parameters()),lr=1e-5,weight_decay=1e-4)
    rgb=torch.from_numpy(train.rgb.copy()).cuda().permute(0,3,1,2).float()/255.
    targets={k:torch.from_numpy(v).cuda() for k,v in gt.items()}
    before=digest(dict(model.named_buffers()));model.train()
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
    write(out/'protocol.json',dict(initial_digest=initial,steps=STEPS,seed=SEED,aux_weight=.25,
        attribution_known=int(masks.sum()),attribution_positive_known=int(((local>0)&masks).sum()),
        source_sha256={p:sha(Path(__file__).with_name(p)) for p in ('body_query_attribution_train.py','body_query_ownership_audit.py','body_query_model.py','body_query_train.py')},native_hashes=native_hashes))
    history=[]
    try:
        for step,indices in enumerate(schedule,1):
            ids=torch.from_numpy(indices).cuda();n,s,c=model(rgb[ids])
            ln=F.binary_cross_entropy_with_logits(n,targets['near'][ids].float())
            ls=pixel_support_bce(s,targets['support'][ids])
            lc=-c.log_softmax(-1).gather(-1,targets['counts'][ids].long().unsqueeze(-1)).mean()
            logits=aux(captured['feature']).squeeze(-1)
            loss_map=F.binary_cross_entropy_with_logits(logits,local[ids],reduction='none')
            la=(loss_map*masks[ids]).sum()/masks[ids].sum().clamp_min(1)
            loss=ln+.25*(ls+lc+la)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step%100==0:
                row=dict(step=step,near=float(ln.detach()),support=float(ls.detach()),query=float(lc.detach()),attribution=float(la.detach()))
                history.append(row);write(out/'progress.json',row);print(row,flush=True)
        torch.cuda.synchronize();fit_seconds=time.perf_counter()-start
        assert before==digest(dict(model.named_buffers()))
        torch.save(model.state_dict(),out/'R1-step2000.pt');torch.save(aux.state_dict(),out/'aux-step2000.pt')
        handle.remove();captured.clear()
        del rgb,targets,local,masks,optimizer,n,s,c,logits,loss_map,loss,ln,ls,lc,la
        torch.cuda.empty_cache()
        data=QueryRGB(a.cache,'dev');devrec,devgt=truth(a.cache,'dev');pred=predict(model,data)
        heads=[select_threshold(pred['near'][:,h],devgt['near'][:,h],min_count=8) for h in range(2)]
        thresholds=[h['threshold'] for h in heads]
        write(out/'selection.json',dict(thresholds=thresholds,heads=heads,checkpoint_sha256=sha(out/'R1-step2000.pt')))
        selection_sha=sha(out/'selection.json');np.savez_compressed(out/'R1-dev.npz',**pred)
        result=dict(dev=metrics(pred,devgt,devrec,thresholds),fit_seconds=fit_seconds,history=history)
        for role in ('train','eval'):
            data=QueryRGB(a.cache,role);r,g=truth(a.cache,role);pred=predict(model,data)
            np.savez_compressed(out/f'R1-{role}.npz',**pred);result[role]=metrics(pred,g,r,thresholds)
        assert selection_sha==sha(out/'selection.json')
        write(out/'result.json',result)
        write(out/'receipt.json',dict(status='PASS',steps=STEPS,backend='CUDA',device=torch.cuda.get_device_name(),
            result_sha256=sha(out/'result.json'),checkpoint_sha256=sha(out/'R1-step2000.pt'),selection_sha256=selection_sha))
    except Exception as exc:
        write(out/'failure.json',dict(error=repr(exc)));raise
    finally:handle.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('cache','native','world','baseline','initial','pretrained','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())

"""Repeat one fixed CUDA backward without optimizer steps or parameter updates."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import read,write,sha,load_npz
from mz23_availability import AngularAvailability
from mz24_decisive_loss import availability_objective


def main(root,output):
    output.mkdir(parents=True,exist_ok=False);work=root/'artifacts.local/work';run=work/'mz24-decisive-availability-20260910/run-v1'
    receipt=read(run/'receipt.json');assert receipt['status']=='PASS';inputs={}
    for name in ['initial.pt','loss_samples.npz']:
        assert sha(run/name)==receipt['outputs'][name];inputs[str(run/name)]=sha(run/name)
    samples=load_npz(run/'loss_samples.npz');ii=samples['step1/indices']
    cache=work/'mz16-visual-detail-20260910/cache-v2';ids=np.load(cache/'ids.npy')
    norm=load_npz(work/'mz9-source-supervision-20260910/run-v1/normalization.npz')
    old=load_npz(work/'mz8-attribution-20260910/cache-v5/observations.npz');new=load_npz(work/'mz15-shared-support-20260910/cache-v1/observations.npz')
    for path in [cache/'ids.npy',cache/'dense_HIGH_DETAIL.npy',work/'mz9-source-supervision-20260910/run-v1/normalization.npz',work/'mz8-attribution-20260910/cache-v5/observations.npz',work/'mz15-shared-support-20260910/cache-v1/observations.npz']:
        inputs[str(path)]=sha(path)
    lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids));assert (lookup[ii]>=0).all()
    maps=np.load(cache/'dense_HIGH_DETAIL.npy',mmap_mode='r')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    x=torch.from_numpy((np.array(maps[lookup[ii]])-norm['mean'])/norm['std']).cuda()
    r=torch.from_numpy(np.concatenate([old['ranges'],new['ranges']])[ii].astype(np.float32)).cuda()
    v=torch.from_numpy(np.concatenate([old['valid'],new['valid']])[ii]).cuda()
    args=[torch.from_numpy(samples['step1/'+key]).cuda() for key in ['known','eligible','query','teacher']]
    state=torch.load(run/'initial.pt',map_location='cpu',weights_only=True);arrays={};losses=[]
    for attempt in range(4):
        head=AngularAvailability(state['grid']);head.load_state_dict(state);head.cuda()
        a=head(x,r,v);np.testing.assert_allclose(a.detach().cpu().numpy(),samples['step1/availability'],atol=2e-5,rtol=1e-5)
        loss,parts=availability_objective(a,*args);loss.backward();losses.append(float(loss.detach()))
        arrays[f'{attempt}/availability']=a.detach().cpu().numpy()
        for name,p in head.named_parameters():arrays[f'{attempt}/gradient/{name}']=p.grad.detach().cpu().numpy()
        for name,value in head.state_dict().items():torch.testing.assert_close(value.cpu(),state[name],rtol=0,atol=0)
    diffs={}
    for attempt in range(1,4):
        diffs[str(attempt)]={k.removeprefix('0/'):dict(max_abs=float(np.abs(value-arrays[str(attempt)+k[1:]]).max()),changed=int(np.count_nonzero(value!=arrays[str(attempt)+k[1:]]))) for k,value in arrays.items() if k.startswith('0/')}
    deterministic_error=None
    try:
        torch.use_deterministic_algorithms(True)
        head=AngularAvailability(state['grid']);head.load_state_dict(state);head.cuda()
        loss,_=availability_objective(head(x,r,v),*args);loss.backward()
    except RuntimeError as error:deterministic_error=str(error)
    finally:torch.use_deterministic_algorithms(False)
    result=dict(losses=losses,differences=diffs,deterministic_error=deterministic_error,training_steps=0,optimizer_created=False,
        interpretation='Identical weights and inputs, repeated backward only. Small gradient differences do not quantify their long-trajectory effect.',
        grid_sample_runtime_doc=torch.nn.functional.grid_sample.__doc__,inputs=inputs,code_sha256=sha(Path(__file__)))
    write(output/'result.json',result);np.savez_compressed(output/'repeats.npz',**arrays)
    write(output/'receipt.json',dict(status='PASS',backend='CUDA fixed backward diagnostic',training_steps=0,outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('LOSSES',losses,'DIFFERENCES',diffs,'DETERMINISTIC_ERROR',deterministic_error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)

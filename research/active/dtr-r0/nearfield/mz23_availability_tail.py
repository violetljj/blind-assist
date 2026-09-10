"""Saved-model diagnostic of max-pooling tails; no fit or operating-point change."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import read,write,sha,load_npz
from mz16_detail_readout import make_model
from mz23_availability import AngularAvailability


def main(root,run,output):
    receipt=read(run/'receipt.json');assert receipt['status']=='PASS'
    for name,h in receipt['outputs'].items():assert sha(run/name)==h
    output.mkdir(parents=True,exist_ok=False)
    work=root/'artifacts.local/work';rank=work/'mz20-rank-objective-20260910/run-v1'
    cache=work/'mz16-visual-detail-20260910/cache-v2'
    old=load_npz(work/'mz8-attribution-20260910/cache-v5/observations.npz')
    new=load_npz(work/'mz15-shared-support-20260910/cache-v1/observations.npz')
    norm=load_npz(work/'mz9-source-supervision-20260910/run-v1/normalization.npz')
    ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']])
    ids=np.load(cache/'ids.npy');lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids))
    maps=np.load(cache/'dense_HIGH_DETAIL.npy',mmap_mode='r');saved=load_npz(run/'predictions.npz');cut=np.load(run/'cutoff.npy')
    cohorts=dict(DEV=np.flatnonzero(old['role']=='DEV_ONLY'),clean=np.arange(3500,3700),stress=np.arange(3500,3700),relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    model=make_model(torch.load(work/'mz15-shared-support-20260910/run-v1/QUERY-initial.pt',weights_only=True)).cuda().eval()
    model.load_state_dict(torch.load(rank/'BODY_RANK.pt',weights_only=True))
    head=AngularAvailability(model.grid).cuda().eval();head.load_state_dict(torch.load(run/'availability.pt',weights_only=True))
    arrays={};errors=[]
    with torch.inference_mode():
        for name,indices in cohorts.items():
            tails=[];wins=[];counts=[]
            for start in range(0,len(indices),16):
                ii=indices[start:start+16];x=torch.from_numpy((np.array(maps[lookup[ii]])-norm['mean'])/norm['std']).cuda()
                rr=old['stress_ranges'][ii-3500] if name=='stress' else ranges[ii]
                vv=old['stress_valid'][ii-3500] if name=='stress' else valid[ii]
                r=torch.from_numpy(rr.astype(np.float32)).cuda();v=torch.from_numpy(vv).cuda()
                out=model.inspect(x,r,v);a=head(x,r,v).cpu().numpy()
                np.testing.assert_allclose(a,saved[name+'/availability'][start:start+len(ii)],atol=2e-5,rtol=1e-5)
                eligible=out['eligible'].cpu().numpy();scores=out['candidate_logits'].cpu().numpy()
                decisive=eligible&(scores.astype(np.float64)>=cut)
                values=np.where(decisive,a[:,:,None,:,None],-1e6).reshape(len(ii),-1,4)
                tails.append(values.max(1));wins.append(values.argmax(1));counts.append(decisive.sum((1,2,3)))
            tail=np.concatenate(tails);winner=np.concatenate(wins);count=np.concatenate(counts)
            base=saved[name+'/baseline'];truth=saved[name+'/truth'];prior=saved[name+'/mz20']
            added=(base<0)&(tail>=0)
            np.testing.assert_array_equal(added,saved[name+'/accepted'])
            arrays[name+'/tail']=tail;arrays[name+'/winner']=winner;arrays[name+'/decisive_count']=count
            for i,q in zip(*np.where((base<0)&(prior>=0)&~truth)):
                z,e,c=np.unravel_index(winner[i,q],(64,2,49))
                errors.append(dict(cohort=name,frame=int(i),query=int(q),survives=bool(tail[i,q]>=0),availability=float(tail[i,q]),
                    zone=int(z),echo=int(e),cell=int(c),known=bool(saved[name+'/known'][i,z,c]),decisive_candidates=int(count[i,q])))
    # Descriptive separability ceiling only: strictest common availability cutoff
    # necessary to remove current false additions. It is NOT adopted for inference.
    boundary=max(e['availability'] for e in errors)
    diagnostic={}
    for name in cohorts:
        base=saved[name+'/baseline'];truth=saved[name+'/truth'];tail=arrays[name+'/tail']
        accepted=(base<0)&(tail>boundary);prediction=(base>=0)|accepted
        diagnostic[name]=dict(retained_added_tp=(accepted&truth).sum(0).tolist(),added_fp=(accepted&~truth).sum(0).tolist())
        if name in ['clean','stress']:diagnostic[name]['pole_tp']=(prediction[100:125]&truth[100:125]).sum(0).tolist()
    result=dict(errors=errors,necessary_strict_common_boundary=float(boundary),diagnostic_only=diagnostic,
        interpretation='Posthoc score separability, not a selected cutoff or candidate; frozen MZ23 remains at logit0.',
        inputs={str(run/'receipt.json'):sha(run/'receipt.json')},code_sha256=sha(Path(__file__)))
    np.savez_compressed(output/'tails.npz',**arrays);write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS',training_steps=0,backend='CUDA frozen inference and CPU reductions',outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.run,a.output)

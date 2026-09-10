"""Fixed-sampler parity with final MZ24 weights and58 saved real feature frames."""
import argparse
from pathlib import Path

import numpy as np
import torch

from mz5_ensemble_readout import read, sha, write, load_npz
from mz16_detail_readout import make_model
from mz23_availability import AngularAvailability
from mz25_fixed_sampler import StableAngularAvailability


def restrict_numpy(out,availability,cut,base):
    eligible=out['eligible'].cpu().numpy()&(availability[:,:,None,:,None]>=0)
    support=eligible.any((1,2,3))
    field=out['candidate_logits'].cpu().numpy()
    raw=np.where(support,np.where(eligible,field,-1e6).max((1,2,3)),-20.)
    margin=np.where(support,raw.astype(float)-cut,-1e6)
    accepted=(base<0)&support&(margin>=0)
    return raw,support,np.where(accepted,margin,base)


def main(root,output):
    output.mkdir(parents=True,exist_ok=False)
    work=root/'artifacts.local/work';run=work/'mz24-decisive-availability-20260910/run-v1'
    rank=work/'mz20-rank-objective-20260910/run-v1';cache=work/'mz16-visual-detail-20260910/cache-v2'
    oldpath=work/'mz8-attribution-20260910/cache-v5';newpath=work/'mz15-shared-support-20260910/cache-v1'
    normpath=work/'mz9-source-supervision-20260910/run-v1/normalization.npz'
    prior=work/'mz15-shared-support-20260910/run-v1';inputs={}
    def bind(path,expected=None):
        h=sha(path);assert expected is None or h==expected,str(path);inputs[str(path)]=h
    for folder,names in [(run,['availability.pt','predictions.npz','cutoff.npy']),
                          (rank,['BODY_RANK.pt','predictions.npz']),(prior,['QUERY-initial.pt'])]:
        receipt=read(folder/'receipt.json');assert receipt['status']=='PASS'
        for name in names:bind(folder/name,receipt['outputs'][name])
    for folder,names in [(cache,['dense_HIGH_DETAIL.npy','ids.npy']),(oldpath,['observations.npz']),(newpath,['observations.npz'])]:
        receipt=read(folder/'receipt.json')
        for name in names:bind(folder/name,receipt['files'][name])
    bind(normpath)
    check=work/'mz25-availability-convergence-20260910/sampler-v1';cr=read(check/'receipt.json');assert cr['status']=='PASS'
    bind(check/'receipt.json')
    for name,digest in cr['outputs'].items():bind(check/name,digest)
    for name,digest in cr['code_sha256'].items():bind(Path(__file__).with_name(name),digest)
    bind(Path(__file__).with_name('mz23_availability.py'))
    saved=load_npz(run/'predictions.npz');previous=load_npz(rank/'predictions.npz');cut=np.load(run/'cutoff.npy')
    old,new=load_npz(oldpath/'observations.npz'),load_npz(newpath/'observations.npz')
    ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']])
    cohorts=dict(clean=np.arange(3500,3700),stress=np.arange(3500,3700),relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    selections={name:np.unique(np.argwhere(previous['BODY_RANK/'+name+'/accepted']&~previous['BODY_RANK/'+name+'/truth'])[:,0]) for name in ['relation10000','distance5000']}
    assert sum(len(v) for v in selections.values())==8
    selections.update(clean=np.arange(100,125),stress=np.arange(100,125))
    ids=np.load(cache/'ids.npy');lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids))
    norm=load_npz(normpath);maps=np.load(cache/'dense_HIGH_DETAIL.npy',mmap_mode='r')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.is_available()
    frozen=make_model(torch.load(prior/'QUERY-initial.pt',map_location='cpu',weights_only=True))
    frozen.load_state_dict(torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True));frozen.cuda().eval()
    state=torch.load(run/'availability.pt',map_location='cpu',weights_only=True)
    original=AngularAvailability(state['grid']);original.load_state_dict(state);original.cuda().eval()
    stable=StableAngularAvailability(state['grid']);stable.load_state_dict(state);stable.cuda().eval()
    assert set(stable.state_dict())==set(state)
    assert sum(p.numel() for p in stable.parameters())==10577
    arrays={};summaries={};frames=0
    try:
        for name,chosen in selections.items():
            pieces=[]
            for begin in range(0,len(chosen),16):
                ff=chosen[begin:begin+16];ii=cohorts[name][ff]
                assert (lookup[ii]>=0).all()
                x=torch.from_numpy((np.array(maps[lookup[ii]])-norm['mean'])/norm['std']).cuda()
                rr,vv=(old['stress_ranges'][ii-3500],old['stress_valid'][ii-3500]) if name=='stress' else (ranges[ii],valid[ii])
                r=torch.from_numpy(rr.astype(np.float32)).cuda();v=torch.from_numpy(vv).cuda()
                with torch.inference_mode():out=frozen.inspect(x,r,v);a=original(x,r,v).cpu().numpy();b=stable(x,r,v).cpu().numpy()
                np.testing.assert_allclose(a,saved[name+'/availability'][ff],atol=2e-5,rtol=1e-5)
                np.testing.assert_allclose(b,a,atol=2e-5,rtol=1e-5)
                base=saved[name+'/baseline'][ff]
                ar,asup,ac=restrict_numpy(out,a,cut,base);br,bsup,bc=restrict_numpy(out,b,cut,base)
                np.testing.assert_array_equal(asup,saved[name+'/support'][ff])
                np.testing.assert_allclose(ar,saved[name+'/raw'][ff],atol=2e-5,rtol=1e-5)
                np.testing.assert_array_equal(ac>=0,saved[name+'/candidate'][ff]>=0)
                np.testing.assert_array_equal(bsup,asup);np.testing.assert_array_equal(br,ar);np.testing.assert_array_equal(bc,ac)
                pieces.append(dict(frames=ff,global_ids=ii,availability_original=a,availability_stable=b,
                    raw_original=ar,raw_stable=br,support_original=asup,support_stable=bsup,
                    candidate_original=ac,candidate_stable=bc))
            values={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
            for key,value in values.items():arrays[name+'/'+key]=value
            summaries[name]=dict(frames=len(chosen),availability_max_abs_error=float(np.abs(values['availability_stable']-values['availability_original']).max()),
                availability_sign_changes=int(((values['availability_stable']>=0)!=(values['availability_original']>=0)).sum()),
                restricted_raw_max_abs_error=float(np.abs(values['raw_stable']-values['raw_original']).max()),
                task_bit_changes=int(((values['candidate_stable']>=0)!=(values['candidate_original']>=0)).sum()),support_bit_changes=0)
            frames+=len(chosen)
        for name,value in stable.state_dict().items():torch.testing.assert_close(value.cpu(),state[name],atol=0,rtol=0)
    finally:
        frozen.cpu();original.cpu();stable.cpu();del maps;torch.cuda.empty_cache()
    assert frames==58
    result=dict(status='PASS',frames=frames,task_bits=frames*4,cohorts=summaries,training_steps=0,optimizer_created=False,
        checkpoint='Unchanged final MZ24 availability.pt',checkpoint_sha256=inputs[str(run/'availability.pt')],
        parameters=10577,state_dict_unchanged=True,sampling_matrix_bytes=stable.sampler.matrix.numel()*stable.sampler.matrix.element_size(),
        inputs=inputs,code_sha256=sha(Path(__file__)),
        scope='Real cached features for8original MZ20 placement false-addition frames plus25pole clean/stress; no fitting or fresh source.',
        limits='Selected inference equivalence only; no new trained candidate, full training trajectory equivalence, convergence or task improvement is established. MZ24 and failed MZ25 decisions remain unchanged.')
    write(output/'result.json',result);np.savez_compressed(output/'predictions.npz',**arrays)
    write(output/'receipt.json',dict(status='PASS',training_steps=0,backend='CUDA frozen checkpoint inference, independent NumPy task composition',inputs=inputs,
        code_sha256=sha(Path(__file__)),outputs={name:sha(output/name) for name in ['result.json','predictions.npz']}))
    print('PASS',frames,'frames',summaries)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.output)

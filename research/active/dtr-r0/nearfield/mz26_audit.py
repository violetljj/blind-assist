"""Independent two-endpoint audit of one deterministic MZ26 trajectory."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path

import numpy as np
import torch

from mz5_ensemble_readout import read, write, sha, load_npz
from mz11_audit import scalar_metrics
from mz15_evaluator import align_stress_labels
from mz16_detail_readout import make_model
from mz24_audit import geometry, scalar_objective
from mz25_fixed_sampler import StableAngularAvailability


def audit_arm(arm,a,result,reference,ids,known,source,query,ranges,valid,old,rays,rows,cut):
    bits=geometry_bits=0;contributors={}
    for name,m in result['metrics'].items():
        normal=name.removesuffix('_wrong');ii=ids[normal];pref='BODY_RANK/'+name+'/'
        get=lambda key:a[name+'/'+key]
        raw,sup,base,target,previous=(get(k) for k in ['raw','support','baseline','truth','mz20'])
        for key in ['baseline','truth']:np.testing.assert_array_equal(get(key),reference[pref+key])
        np.testing.assert_array_equal(previous,reference[pref+'candidate'])
        np.testing.assert_allclose(get('original_raw'),reference[pref+'raw'],atol=2e-5,rtol=1e-5)
        np.testing.assert_array_equal(get('original_support'),reference[pref+'support'])
        np.testing.assert_array_equal(get('known'),known[ii])
        assert get('availability').shape==(len(ii),64,49) and np.isfinite(get('availability')).all()
        margin=np.where(sup,raw.astype(float)-cut,-1e6);accepted=(base<0)&sup&(margin>=0)
        candidate=np.where(accepted,margin,base)
        for key,value in [('margin',margin),('accepted',accepted),('candidate',candidate)]:np.testing.assert_array_equal(get(key),value)
        assert not ((candidate>=0)&(previous<0)).any()
        np.testing.assert_array_equal(candidate[base>=0],base[base>=0])
        np.testing.assert_array_equal(candidate[~sup],base[~sup])
        assert not (sup&~get('original_support')).any()
        assert (raw[sup]<=get('original_raw')[sup]+2e-5).all()
        for key,score in [('baseline',base),('mz20',previous),('candidate',candidate)]:assert scalar_metrics(score,target)==m[key]
        changes=dict(added_tp=(accepted&target).sum(0).tolist(),added_fp=(accepted&~target).sum(0).tolist(),
            mz20_tp_lost=((previous>=0)&(candidate<0)&target).sum(0).tolist(),mz20_fp_removed=((previous>=0)&(candidate<0)&~target).sum(0).tolist())
        assert all(m[key]==value for key,value in changes.items());bits+=candidate.size
        if name in result['thin']:assert scalar_metrics(candidate[100:125],target[100:125])==result['thin'][name]
        predicted=get('availability')>=0;k=known[ii]
        tp=int((predicted&k).sum());fp=int((predicted&~k).sum());fn=int((~predicted&k).sum())
        if name in result['availability']:
            assert result['availability'][name]==dict(tp=tp,fp=fp,fn=fn,positive=int(k.sum()),negative=int((~k).sum()),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
        rr,vv=(old['stress_ranges'][ii-3500],old['stress_valid'][ii-3500]) if normal=='stress' else (ranges[ii],valid[ii])
        ss,qq=source[ii],query[ii]
        if normal=='stress':ss,qq,_=align_stress_labels(ss,qq,ranges[ii],valid[ii],rr,vv)
        counts=dict(source_cells=int(ss.sum()),source_cells_rejected=int((ss&~predicted[:,:,None,:]).sum()),
            query_cells=qq.sum((0,1,2,3)).tolist(),query_cells_rejected=(qq&~predicted[:,:,None,:,None]).sum((0,1,2,3)).tolist(),
            eligible_query_cells=[0]*4,eligible_query_cells_rejected=[0]*4)
        for begin in range(0,len(ii),64):
            end=min(begin+64,len(ii));e=geometry(rr[begin:end],vv[begin:end],rays);restricted=e&predicted[begin:end,:,None,:,None]
            np.testing.assert_array_equal(e.any((1,2,3)),get('original_support')[begin:end])
            np.testing.assert_array_equal(restricted.any((1,2,3)),sup[begin:end]);geometry_bits+=e.size
            z=qq[begin:end]&e
            counts['eligible_query_cells']=(np.array(counts['eligible_query_cells'])+z.sum((0,1,2,3))).tolist()
            counts['eligible_query_cells_rejected']=(np.array(counts['eligible_query_cells_rejected'])+(z&~restricted).sum((0,1,2,3))).tolist()
        contributors[name]=counts
    for name in ['relation10000','distance5000']:
        rr=[r for r in rows if r['dataset']==name]
        for field,units in result['groups'][name].items():
            assert set(units)=={r[field] for r in rr}
            for unit,m in units.items():
                ix=[i for i,r in enumerate(rr) if r[field]==unit]
                assert m==dict(frames=len(ix),baseline=scalar_metrics(a[name+'/baseline'][ix],a[name+'/truth'][ix]),candidate=scalar_metrics(a[name+'/candidate'][ix],a[name+'/truth'][ix]))
    m=result['metrics'];retained=sum(sum(m[n]['added_tp'][1::2]) for n in ['relation10000','distance5000'])
    gates=dict(no_added_fp=all(sum(m[n]['added_fp'])==0 for n in ids),far_retention=retained>=68,
        placement_far_gain=all(sum(m[n]['added_tp'][1::2])>0 for n in ['relation10000','distance5000']),
        pole_clean=sum(result['thin']['clean']['tp'][1::2])>=48,pole_stress=sum(result['thin']['stress']['tp'][1::2])>=48,
        baseline_positives_retained=all(not ((a[n+'/baseline']>=0)&(a[n+'/candidate']<0)).any() for n in ids))
    gates['useful_effect']=all(gates.values());assert gates==result['gates'] and retained==result['retained_far_additions']
    assert bits==31200
    return dict(scalar_bits=bits,geometric_candidate_bits=geometry_bits,contributor_rejection=contributors,gates=gates,retained_far_additions=retained)


def main(root,run):
    assert not (run/'audit.json').exists()
    receipt,start,result=(read(run/name) for name in ['receipt.json','start.json','result.json'])
    assert receipt['status']=='PASS'
    for name,digest in receipt['outputs'].items():assert sha(run/name)==digest,name
    for name,digest in start['inputs'].items():assert sha(name)==digest,name
    for name,digest in start['code_sha256'].items():assert sha(Path(__file__).with_name(name))==digest,name
    assert sha(Path(__file__).with_name('MZ26_DETERMINISTIC_CONVERGENCE_PROTOCOL_20260910.md'))==start['protocol_sha256']
    work=root/'artifacts.local/work';rank=work/'mz20-rank-objective-20260910/run-v1'
    cut=np.load(run/'cutoff.npy');assert sha(run/'cutoff.npy')==sha(rank/'BODY_RANK-cutoff.npy')
    batches=np.load(run/'batches.npy');original_batches=np.load(rank/'batches.npy')
    assert batches.shape==(4800,16) and original_batches.shape==(1200,16)
    np.testing.assert_array_equal(batches,np.tile(original_batches,(4,1)))
    assert start['steps']==4800 and start['seed']==123 and start['unique_train']==len(np.unique(batches))==7562
    assert start['endpoints']==[1200,4800] and start['batch_cycles']==4 and start['draws']==batches.size==76800
    runtime=start['runtime']
    for name,value in dict(deterministic=True,cudnn_deterministic=True,cudnn_benchmark=False,matmul_tf32=False,cublas_workspace=':4096:8').items():
        assert runtime[name]==value,name
    assert all(runtime[name] for name in ['torch','cuda','cudnn','device']) and isinstance(runtime['cudnn_tf32'],bool)
    assert start['objective_weights']==dict(dense=1.,negative=.25,positive=.25)
    assert set(result['arms'])=={'step1200','step4800'}
    assert result['fit']['steps']==4800
    assert [r['step'] for r in result['fit']['history']]==sorted(set([*range(1,4801,400),1200,4800]))
    for endpoint in [1200,4800]:assert result['arms'][f'step{endpoint}']['fit']==dict(steps=endpoint,parameters=10577)
    old=load_npz(work/'mz8-attribution-20260910/cache-v5/observations.npz')
    newpath=work/'mz15-shared-support-20260910/cache-v1';new=load_npz(newpath/'observations.npz');rows=read(newpath/'selected.json')
    ol=load_npz(work/'mz9-source-supervision-20260910/labels-v1/evaluator.npz');nl=load_npz(newpath/'evaluator.npz')
    known=np.concatenate([ol['cell_known_counts']>0,nl['known']]);source=np.concatenate([ol['source_counts']>0,nl['source_presence']])
    query=np.concatenate([ol['query_counts']>0,nl['query_presence']]);del ol,nl
    ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']])
    allowed=np.r_[np.flatnonzero(old['role']=='TRAIN_ONLY'),np.arange(3500,11200)]
    assert np.isin(batches,allowed).all() and not np.isin(batches,np.flatnonzero(old['role']=='DEV_ONLY')).any()
    assert all(r['role']=='TRAIN_ONLY' for r in rows[:7500]) and all(r['role']=='DEV_ONLY' for r in rows[7500:])
    ids=dict(DEV=np.flatnonzero(old['role']=='DEV_ONLY'),clean=np.arange(3500,3700),stress=np.arange(3500,3700),relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    state=torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True);rays=state['rays']
    initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=True)
    mz24_initial_path=work/'mz24-decisive-availability-20260910/run-v1/initial.pt'
    mz24_initial=torch.load(mz24_initial_path,map_location='cpu',weights_only=True)
    torch.set_num_threads(1);torch.manual_seed(123);expected=StableAngularAvailability(state['grid'])
    assert set(expected.state_dict())==set(initial)==set(mz24_initial)
    for name,value in expected.state_dict().items():
        torch.testing.assert_close(initial[name],value,rtol=0,atol=0)
        torch.testing.assert_close(initial[name],mz24_initial[name],rtol=0,atol=0)
    assert sum(p.numel() for p in expected.parameters())==result['fit']['parameters']==10577
    losses=load_npz(run/'loss_samples.npz');loss_checks={}
    for step in [1,1200,4800]:
        prefix=f'step{step}/';ii=losses[prefix+'indices'];np.testing.assert_array_equal(ii,batches[step-1])
        v={name:losses[prefix+name] for name in ['availability','known','eligible','query','teacher','gradient']}
        np.testing.assert_array_equal(v['known'],known[ii]);np.testing.assert_array_equal(v['query'],query[ii])
        np.testing.assert_array_equal(v['eligible'],geometry(ranges[ii],valid[ii],rays))
        total,parts,gradient=scalar_objective(v['availability'],v['known'],v['eligible'],v['query'],v['teacher'])
        np.testing.assert_allclose(total,float(losses[prefix+'total']),atol=2e-6,rtol=1e-6)
        for name,value in parts.items():np.testing.assert_allclose(value,float(losses[prefix+name]),atol=2e-6,rtol=1e-6)
        history=next(row for row in result['fit']['history'] if row['step']==step)
        assert history['loss']==float(losses[prefix+'total'])
        for name in parts:assert history[name]==float(losses[prefix+name])
        np.testing.assert_allclose(gradient,v['gradient'],atol=2e-8,rtol=2e-5)
        loss_checks[str(step)]=dict(loss_abs_error=abs(total-float(losses[prefix+'total'])),gradient_max_abs_error=float(np.abs(gradient-v['gradient']).max()),gradient_elements=gradient.size,parts=parts)
    reference=load_npz(rank/'predictions.npz');arms={}
    cache=work/'mz16-visual-detail-20260910/cache-v2';cacheids=np.load(cache/'ids.npy');lookup=np.full(14200,-1,int);lookup[cacheids]=np.arange(len(cacheids))
    selections={name:np.unique(np.argwhere(reference['BODY_RANK/'+name+'/accepted']&~reference['BODY_RANK/'+name+'/truth'])[:,0]) for name in ['relation10000','distance5000']}
    selections.update(clean=np.arange(100,125),stress=np.arange(100,125))
    assert sum(len(v) for v in selections.values())==58
    norm=load_npz(work/'mz9-source-supervision-20260910/run-v1/normalization.npz')
    maps=np.load(cache/'dense_HIGH_DETAIL.npy',mmap_mode='r')
    assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=runtime['cudnn_tf32']
    assert runtime['torch']==torch.__version__ and runtime['cuda']==torch.version.cuda and runtime['cudnn']==torch.backends.cudnn.version()
    assert runtime['device']==torch.cuda.get_device_name()
    frozen=make_model(torch.load(work/'mz15-shared-support-20260910/run-v1/QUERY-initial.pt',map_location='cpu',weights_only=True));frozen.load_state_dict(state);frozen.cuda().eval()
    head=None;checkpoint_hashes={}
    try:
        for arm in ['step1200','step4800']:
            path=run/f'availability-{arm}.pt';assert path.name in receipt['outputs'];checkpoint_hashes[arm]=sha(path)
            a=load_npz(run/f'{arm}-predictions.npz')
            arms[arm]=audit_arm(arm,a,result['arms'][arm],reference,ids,known,source,query,ranges,valid,old,rays,rows,cut)
            head=StableAngularAvailability(state['grid']);head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));head.cuda().eval()
            replay={}
            for name,chosen in selections.items():
                availability_error=raw_error=0.
                for begin in range(0,len(chosen),16):
                    ff=chosen[begin:begin+16];ii=ids[name][ff]
                    x=torch.from_numpy((np.array(maps[lookup[ii]])-norm['mean'])/norm['std']).cuda()
                    rr,vv=(old['stress_ranges'][ii-3500],old['stress_valid'][ii-3500]) if name=='stress' else (ranges[ii],valid[ii])
                    r=torch.from_numpy(rr.astype(np.float32)).cuda();v=torch.from_numpy(vv).cuda()
                    with torch.inference_mode():out=frozen.inspect(x,r,v);availability=head(x,r,v).cpu().numpy()
                    eligible=out['eligible'].cpu().numpy()&(availability[:,:,None,:,None]>=0)
                    support=eligible.any((1,2,3));raw=np.where(support,np.where(eligible,out['candidate_logits'].cpu().numpy(),-1e6).max((1,2,3)),-20.)
                    np.testing.assert_allclose(availability,a[name+'/availability'][ff],atol=2e-5,rtol=1e-5)
                    np.testing.assert_array_equal(availability>=0,a[name+'/availability'][ff]>=0)
                    np.testing.assert_allclose(raw,a[name+'/raw'][ff],atol=2e-5,rtol=1e-5);np.testing.assert_array_equal(support,a[name+'/support'][ff])
                    base=a[name+'/baseline'][ff];margin=np.where(support,raw.astype(float)-cut,-1e6)
                    candidate=np.where((base<0)&support&(margin>=0),margin,base)
                    np.testing.assert_array_equal(candidate>=0,a[name+'/candidate'][ff]>=0)
                    availability_error=max(availability_error,float(np.abs(availability-a[name+'/availability'][ff]).max()));raw_error=max(raw_error,float(np.abs(raw-a[name+'/raw'][ff]).max()))
                replay[name]=dict(frames=len(chosen),availability_max_abs_error=availability_error,raw_max_abs_error=raw_error)
            arms[arm]['selected_inference']=dict(frames=58,cohorts=replay,backend='strict deterministic CUDA plus independent NumPy restriction')
            head.cpu();head=None
    finally:
        frozen.cpu()
        if head is not None:head.cpu()
        del maps;torch.use_deterministic_algorithms(False);torch.cuda.empty_cache()
    audit=dict(status='PASS',code_sha256=sha(Path(__file__)),scalar_bits=sum(v['scalar_bits'] for v in arms.values()),
        initialization='Exact seed123StableAngularAvailability and MZ24initialstate',batch_cycles=4,unique_train=7562,
        draws=76800,runtime=runtime,inputs_verified=len(start['inputs']),receipt_sha256=sha(run/'receipt.json'),
        audit_dependency_sha256={name:sha(Path(__file__).with_name(name)) for name in ['mz24_audit.py','mz11_audit.py','mz15_evaluator.py','mz16_detail_readout.py','mz25_fixed_sampler.py']},
        checkpoints=checkpoint_hashes,shared_trajectory='Both predeclared endpoints belong to the same run; no external MZ24trajectory equality gate and no endpoint selection.',
        loss_scalar_and_gradient=loss_checks,arms=arms,
        limits=['Saved scalar outputs are fully audited; frozen checkpoint inference reproduction covers58frames per endpoint.',
                'Contributor-cell counts are correlated evidence units, not independent obstacle events.',
                'Strict flags and same-run ownership do not prove a mathematical convergence or capacity limit.'])
    assert audit['scalar_bits']==62400
    write(run/'audit.json',audit);print('PASS',audit['scalar_bits'],'taskbits','116inferenceframes',{k:v['gates'] for k,v in arms.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.run)

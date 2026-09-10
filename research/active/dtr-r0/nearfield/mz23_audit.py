"""Independent MZ23 decisions, contributor retention and selected inference audit."""
import argparse
from pathlib import Path

import numpy as np
import torch

from mz5_ensemble_readout import read, write, sha, load_npz
from mz11_audit import scalar_metrics
from mz15_evaluator import align_stress_labels
from mz16_detail_readout import make_model
from mz23_availability import AngularAvailability
from mz8_attribution import query_membership


def geometry(ranges, valid, rays):
    r=torch.from_numpy(ranges.astype(np.float32));v=torch.from_numpy(valid)
    good=v & torch.isfinite(r) & (r>0) & (r<=4)
    points=torch.where(good,r,0)[...,None,None]*rays[None,:,None]+torch.tensor([0.,0.,1.7])
    return (query_membership(points)&good[...,None,None]).numpy()


def main(root,run):
    receipt,start,result=(read(run/name) for name in ['receipt.json','start.json','result.json'])
    assert receipt['status']=='PASS' and not (run/'audit.json').exists()
    for name,digest in receipt['outputs'].items():assert sha(run/name)==digest,name
    for name,digest in start['inputs'].items():assert sha(name)==digest,name
    for name,digest in start['code_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
    assert sha(Path(__file__).with_name('MZ23_AVAILABILITY_PROTOCOL_20260910.md'))==start['protocol_sha256']
    work=root/'artifacts.local/work';rank=work/'mz20-rank-objective-20260910/run-v1'
    oracle=work/'mz23-availability-20260910/oracle-v1'
    ore,orv=read(oracle/'receipt.json'),read(oracle/'result.json')
    assert ore['status']=='PASS'
    for name,digest in ore['outputs'].items():assert sha(oracle/name)==digest
    for name,digest in ore['inputs'].items():assert sha(name)==digest
    assert orv['placement_far_retained']>=68 and orv['placement_added_fp_after']==0
    assert orv['placement_added_fp_before']==8 and all(v>=48 for v in orv['pole_retained'].values())
    assert sha(run/'cutoff.npy')==sha(rank/'BODY_RANK-cutoff.npy')
    assert sha(run/'batches.npy')==sha(rank/'batches.npy')
    cut=np.load(run/'cutoff.npy');batches=np.load(run/'batches.npy')
    assert batches.shape==(1200,16) and start['steps']==1200 and start['seed']==123
    assert start['draws']==batches.size and start['unique_train']==len(np.unique(batches))==7562
    old=load_npz(work/'mz8-attribution-20260910/cache-v5/observations.npz')
    newcache=work/'mz15-shared-support-20260910/cache-v1'
    new=load_npz(newcache/'observations.npz');rows=read(newcache/'selected.json')
    ol=load_npz(work/'mz9-source-supervision-20260910/labels-v1/evaluator.npz')
    nl=load_npz(newcache/'evaluator.npz')
    known=np.concatenate([ol['cell_known_counts']>0,nl['known']])
    source=np.concatenate([ol['source_counts']>0,nl['source_presence']])
    query=np.concatenate([ol['query_counts']>0,nl['query_presence']]);del ol,nl
    ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']])
    allowed=np.r_[np.flatnonzero(old['role']=='TRAIN_ONLY'),np.arange(3500,11200)]
    assert np.isin(batches,allowed).all() and (batches<11200).all()
    assert not np.isin(batches,np.flatnonzero(old['role']=='DEV_ONLY')).any()
    assert all(r['role']=='TRAIN_ONLY' for r in rows[:7500]) and all(r['role']=='DEV_ONLY' for r in rows[7500:])
    ids=dict(DEV=np.flatnonzero(old['role']=='DEV_ONLY'),clean=np.arange(3500,3700),stress=np.arange(3500,3700),
        relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    state=torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True)
    initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=True)
    torch.set_num_threads(1);torch.manual_seed(123)
    expected=AngularAvailability(state['grid'])
    for name,value in expected.state_dict().items():torch.testing.assert_close(initial[name],value,rtol=0,atol=0)
    assert sum(p.numel() for p in expected.parameters())==result['fit']['parameters']
    rays=state['rays'];a=load_npz(run/'predictions.npz');ref=load_npz(rank/'predictions.npz')
    bits=0;contributor={};support_bits=0
    for name,m in result['metrics'].items():
        normal=name.removesuffix('_wrong');ii=ids[normal];prefix='BODY_RANK/'+name+'/'
        v={key:a[name+'/'+key] for key in ['raw','support','original_raw','original_support','availability','baseline','truth','mz20','margin','accepted','candidate','known']}
        for key in ['baseline','truth']:
            np.testing.assert_array_equal(v[key],ref[prefix+key])
        np.testing.assert_array_equal(v['mz20'],ref[prefix+'candidate'])
        np.testing.assert_allclose(v['original_raw'],ref[prefix+'raw'],atol=2e-5,rtol=1e-5)
        np.testing.assert_array_equal(v['original_support'],ref[prefix+'support'])
        np.testing.assert_array_equal(v['known'],known[ii])
        assert v['availability'].shape==(len(ii),64,49) and np.isfinite(v['availability']).all()
        margin=np.where(v['support'],v['raw'].astype(float)-cut,-1e6)
        accept=(v['baseline']<0)&v['support']&(margin>=0)
        candidate=np.where(accept,margin,v['baseline'])
        for value,key in [(margin,'margin'),(accept,'accepted'),(candidate,'candidate')]:np.testing.assert_array_equal(value,v[key])
        assert not ((candidate>=0)&(v['mz20']<0)).any()
        np.testing.assert_array_equal(candidate[v['baseline']>=0],v['baseline'][v['baseline']>=0])
        np.testing.assert_array_equal(candidate[~v['support']],v['baseline'][~v['support']])
        assert not (v['support']&~v['original_support']).any()
        assert (v['raw'][v['support']]<=v['original_raw'][v['support']]+2e-5).all()
        for key,score in [('baseline',v['baseline']),('mz20',v['mz20']),('candidate',candidate)]:assert scalar_metrics(score,v['truth'])==m[key]
        changes=dict(added_tp=(accept&v['truth']).sum(0).tolist(),added_fp=(accept&~v['truth']).sum(0).tolist(),
            mz20_tp_lost=((v['mz20']>=0)&(candidate<0)&v['truth']).sum(0).tolist(),
            mz20_fp_removed=((v['mz20']>=0)&(candidate<0)&~v['truth']).sum(0).tolist())
        assert all(m[k]==val for k,val in changes.items());bits+=candidate.size
        if name in result['thin']:assert scalar_metrics(candidate[100:125],v['truth'][100:125])==result['thin'][name]
        predicted=v['availability']>=0;k=known[ii]
        tp=int((predicted&k).sum());fp=int((predicted&~k).sum());fn=int((~predicted&k).sum())
        if name in result['availability']:
            assert result['availability'][name]==dict(tp=tp,fp=fp,fn=fn,positive=int(k.sum()),negative=int((~k).sum()),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
        rr,vv=(old['stress_ranges'][ii-3500],old['stress_valid'][ii-3500]) if normal=='stress' else (ranges[ii],valid[ii])
        ss,qq=source[ii],query[ii]
        if normal=='stress':ss,qq,_=align_stress_labels(ss,qq,ranges[ii],valid[ii],rr,vv)
        totals=dict(source_cells=int(ss.sum()),source_cells_rejected=int((ss&~predicted[:,:,None,:]).sum()),
            query_cells=qq.sum((0,1,2,3)).tolist(),query_cells_rejected=(qq&~predicted[:,:,None,:,None]).sum((0,1,2,3)).tolist(),
            eligible_query_cells=[0]*4,eligible_query_cells_rejected=[0]*4)
        for begin in range(0,len(ii),64):
            end=min(begin+64,len(ii));eligible=geometry(rr[begin:end],vv[begin:end],rays)
            restricted=eligible&predicted[begin:end,:,None,:,None]
            np.testing.assert_array_equal(eligible.any((1,2,3)),v['original_support'][begin:end])
            np.testing.assert_array_equal(restricted.any((1,2,3)),v['support'][begin:end]);support_bits+=eligible.size
            z=qq[begin:end]&eligible
            totals['eligible_query_cells']=(np.array(totals['eligible_query_cells'])+z.sum((0,1,2,3))).tolist()
            totals['eligible_query_cells_rejected']=(np.array(totals['eligible_query_cells_rejected'])+(z&~restricted).sum((0,1,2,3))).tolist()
        contributor[name]=totals
    for name in ['relation10000','distance5000']:
        rr=[r for r in rows if r['dataset']==name]
        for field,units in result['groups'][name].items():
            assert set(units)=={r[field] for r in rr}
            for unit,m in units.items():
                ii=[i for i,r in enumerate(rr) if r[field]==unit]
                assert m==dict(frames=len(ii),baseline=scalar_metrics(a[name+'/baseline'][ii],a[name+'/truth'][ii]),candidate=scalar_metrics(a[name+'/candidate'][ii],a[name+'/truth'][ii]))
    m=result['metrics'];retained=sum(sum(m[n]['added_tp'][1::2]) for n in ['relation10000','distance5000'])
    gates=dict(no_added_fp=all(sum(m[n]['added_fp'])==0 for n in ids),far_retention=retained>=68,
        placement_far_gain=all(sum(m[n]['added_tp'][1::2])>0 for n in ['relation10000','distance5000']),
        pole_clean=sum(result['thin']['clean']['tp'][1::2])>=48,pole_stress=sum(result['thin']['stress']['tp'][1::2])>=48,
        baseline_positives_retained=all(not ((a[n+'/baseline']>=0)&(a[n+'/candidate']<0)).any() for n in ids))
    gates['useful_effect']=all(gates.values());assert gates==result['gates'] and retained==result['retained_far_additions']
    # Selected fresh inference verifies exported networks and masked maxima;
    # the candidate reduction below is independent NumPy, not restrict_candidates.
    selections={name:np.unique(np.argwhere(ref['BODY_RANK/'+name+'/accepted']&~ref['BODY_RANK/'+name+'/truth'])[:,0]) for name in ['relation10000','distance5000']}
    selections.update(clean=np.arange(100,125),stress=np.arange(100,125))
    cache=work/'mz16-visual-detail-20260910/cache-v2';cacheids=np.load(cache/'ids.npy')
    lookup=np.full(14200,-1,int);lookup[cacheids]=np.arange(len(cacheids))
    maps=np.load(cache/'dense_HIGH_DETAIL.npy',mmap_mode='r')
    norm=load_npz(work/'mz9-source-supervision-20260910/run-v1/normalization.npz')
    assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False
    prior=work/'mz15-shared-support-20260910/run-v1/QUERY-initial.pt'
    frozen=make_model(torch.load(prior,map_location='cpu',weights_only=True));frozen.load_state_dict(state);frozen.cuda().eval()
    head=AngularAvailability(state['grid']);head.load_state_dict(torch.load(run/'availability.pt',map_location='cpu',weights_only=True));head.cuda().eval()
    replay={};frames=0
    try:
        for name,chosen in selections.items():
            error_a=error_raw=0.
            for begin in range(0,len(chosen),16):
                ff=chosen[begin:begin+16];ii=ids[name][ff]
                x=torch.from_numpy((np.array(maps[lookup[ii]])-norm['mean'])/norm['std']).cuda()
                rr,vv=(old['stress_ranges'][ii-3500],old['stress_valid'][ii-3500]) if name=='stress' else (ranges[ii],valid[ii])
                r=torch.from_numpy(rr.astype(np.float32)).cuda();v=torch.from_numpy(vv).cuda()
                with torch.inference_mode():out=frozen.inspect(x,r,v);availability=head(x,r,v).cpu().numpy()
                e=out['eligible'].cpu().numpy()&(availability[:,:,None,:,None]>=0)
                support=e.any((1,2,3));logits=out['candidate_logits'].cpu().numpy()
                raw=np.where(support,np.where(e,logits,-1e6).max((1,2,3)),-20.)
                np.testing.assert_allclose(availability,a[name+'/availability'][ff],atol=2e-5,rtol=1e-5)
                np.testing.assert_allclose(raw,a[name+'/raw'][ff],atol=2e-5,rtol=1e-5)
                np.testing.assert_array_equal(support,a[name+'/support'][ff])
                np.testing.assert_array_equal(availability>=0,a[name+'/availability'][ff]>=0)
                error_a=max(error_a,float(np.abs(availability-a[name+'/availability'][ff]).max()))
                error_raw=max(error_raw,float(np.abs(raw-a[name+'/raw'][ff]).max()));frames+=len(ff)
            replay[name]=dict(frames=len(chosen),availability_max_abs_error=error_a,restricted_raw_max_abs_error=error_raw)
    finally:
        frozen.cpu();head.cpu();del maps
        torch.cuda.empty_cache()
    audit=dict(status='PASS',code_sha256=sha(Path(__file__)),scalar_bits=bits,geometric_candidate_bits=support_bits,
        frozen_cutoff_and_batches='byte-identical to MZ20',initialization='exact seed123 CPU parameter reconstruction',
        baseline_positives='all retained exactly',no_positives_beyond_mz20=True,oracle_bound_and_admitted=True,
        train_calibration_placement_disjoint=True,groups='family/group/site exact',gates=gates,
        contributor_rejection=contributor,selected_inference=dict(backend='CUDA',frames=frames,cohorts=replay,
            scope='All8 prior placement false-addition frames plus25pole frames clean and stress; independent NumPy restricted maximum'),
        limits=['Candidate maximum inference replay is selected58frames; all31200task bits and full geometry/support masks are independently checked.',
                'Wrong-zone shifts both local chains; this does not isolate availability-only correspondence contribution.',
                'Knownness describes native<=4m angular evidence availability, not physical occupancy or sensor reliability.'])
    write(run/'audit.json',audit);print('PASS',bits,'task bits',frames,'inference frames',gates)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.run)

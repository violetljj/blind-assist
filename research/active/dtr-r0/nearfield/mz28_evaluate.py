"""One full-cohort packet-only availability comparator; no parameter updates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from mz5_ensemble_readout import read,write,sha,load_npz
from mz16_detail_readout import make_model
from mz23_availability import restrict_candidates
from mz28_packet_availability import PacketAvailability,packet4
from mz8_train import metrics


def main(root,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();work=root/'artifacts.local/work';inputs={}
    def bind(path,expected=None):
        h=sha(path);assert expected is None or h==expected,str(path);inputs[str(path)]=h
    old=work/'mz8-attribution-20260910/cache-v5';new=work/'mz15-shared-support-20260910/cache-v1'
    cache=work/'mz16-visual-detail-20260910/cache-v2';rank=work/'mz20-rank-objective-20260910/run-v1'
    prior=work/'mz15-shared-support-20260910/run-v1';labels=work/'mz9-source-supervision-20260910/labels-v1'
    run26=work/'mz26-deterministic-convergence-20260910/run-v1';run27=work/'mz27-feature-separation-20260910/run-v1'
    normpath=work/'mz9-source-supervision-20260910/run-v1/normalization.npz'
    indexpath=work/'body-query-5000-20260909/dataset-v1/index.json'
    for folder,names in [(old,['observations.npz']),(new,['observations.npz','evaluator.npz','selected.json']),
                         (cache,['dense_HIGH_DETAIL.npy','ids.npy','selected.json'])]:
        receipt=read(folder/'receipt.json');assert receipt['status']=='PASS';bind(folder/'receipt.json')
        for name in names:bind(folder/name,receipt['files'][name])
    for folder,names in [(rank,['BODY_RANK.pt','BODY_RANK-cutoff.npy','predictions.npz','batches.npy']),
                         (prior,['QUERY-initial.pt']),(run26,['step4800-predictions.npz','result.json']),
                         (run27,['result.json'])]:
        receipt=read(folder/'receipt.json');assert receipt['status']=='PASS';bind(folder/'receipt.json')
        for name in names:bind(folder/name,receipt['outputs'][name])
    bind(labels/'receipt.json');bind(labels/'evaluator.npz',read(labels/'receipt.json')['labels_sha256'])
    bind(normpath,read(normpath.parent/'receipt.json')['outputs']['normalization.npz'])
    identity_receipt=work/'mz1-tiny-fusion-20260910/cache-v1/features-receipt.json';bind(identity_receipt)
    bind(indexpath,read(identity_receipt)['source_index_sha256'])
    d,nd=load_npz(old/'observations.npz'),load_npz(new/'observations.npz')
    ol,nl=load_npz(labels/'evaluator.npz'),load_npz(new/'evaluator.npz')
    known=np.concatenate([ol['cell_known_counts']>0,nl['known']]);query=np.concatenate([ol['query_counts']>0,nl['query_presence']]);del ol,nl
    ranges=np.concatenate([d['ranges'],nd['ranges']]);valid=np.concatenate([d['valid'],nd['valid']])
    train=np.unique(np.load(rank/'batches.npy'));bankids=train[(train<3500)|(train>=3700)]
    assert len(train)==7562 and len(bankids)==7362
    cohorts=dict(DEV=np.flatnonzero(d['role']=='DEV_ONLY'),clean=np.arange(3500,3700),stress=np.arange(3500,3700),relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    oldindex=read(indexpath)['frames'];rows=read(new/'selected.json');meta={r['global_id']:r for r in read(cache/'selected.json')}
    def identity(i):
        if i<3500:
            r=oldindex[int(d['old_index'][i])];assert r['rgb_sha256']==meta[i]['rgb_sha']
            return dict(site=r['site_id'],group=r['group_id'])
        r=rows[i-3700];assert r['cache_index']==i-3700 and r['rgb_sha']==meta[i]['rgb_sha']
        return dict(site=r['site'],group=r['group'])
    bankident=[identity(int(i)) for i in bankids];sites={r['site'] for r in bankident};assert len(sites)==375
    overlap={}
    for name in ['DEV','relation10000','distance5000']:
        assert not np.isin(bankids,cohorts[name]).any()
        overlap[name]=len(sites&{identity(int(i))['site'] for i in cohorts[name]});assert overlap[name]==0
    bankpacket=packet4(torch.from_numpy(ranges[bankids]),torch.from_numpy(valid[bankids])).numpy()
    np.savez_compressed(output/'bank.npz',ids=bankids,packet=bankpacket,known=known[bankids])
    write(output/'bank-identities.json',bankident)
    ref=load_npz(rank/'predictions.npz');control=load_npz(run26/'step4800-predictions.npz');cut=np.load(rank/'BODY_RANK-cutoff.npy');np.save(output/'cutoff.npy',cut)
    ids=np.load(cache/'ids.npy');lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids));n=load_npz(normpath)
    maps=np.load(cache/'dense_HIGH_DETAIL.npy',mmap_mode='r')
    torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    frozen=make_model(torch.load(prior/'QUERY-initial.pt',map_location='cpu',weights_only=True))
    frozen.load_state_dict(torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True));frozen.cuda().eval().requires_grad_(False)
    head=PacketAvailability(torch.from_numpy(bankids),torch.from_numpy(bankpacket),torch.from_numpy(known[bankids])).cuda().eval()
    code={name:sha(Path(__file__).with_name(name)) for name in [Path(__file__).name,'mz28_packet_availability.py','mz23_availability.py','mz16_detail_readout.py']}
    write(output/'start.json',dict(status='STARTED',training_steps=0,inputs=inputs,code_sha256=code,
        protocol_sha256=sha(Path(__file__).with_name('MZ28_PACKET_AVAILABILITY_PROTOCOL_20260910.md')),
        reference_frames=len(bankids),excluded_sequence_frames=200,bank_sites=len(sites),dev_site_overlap=overlap,
        bank_source='Exact MZ20 TRAIN IDs excluding200 MZ6 frames; no current labels or site IDs in forward',
        bank_array_bytes=int(bankids.nbytes+bankpacket.nbytes+known[bankids].nbytes),runtime_buffer_bytes=sum(b.numel()*b.element_size() for b in head.buffers()),
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),deterministic=torch.are_deterministic_algorithms_enabled(),distance_dtype='float64',packet_dtype='float32'),
        inference_interface='PacketAvailability(ranges,valid); no current native/query/site/known data',fixed=dict(neighbors=5,minimum_votes=3)))
    result=dict(training_steps=0,metrics={},availability={},thin={},groups={},witnesses={});arrays={};retrieval_seconds=0.;frozen_seconds=0.
    try:
        for name,ii in cohorts.items():
            parts=[]
            with torch.inference_mode():
                for begin in range(0,len(ii),16):
                    batch=ii[begin:begin+16];rr=d['stress_ranges'][batch-3500] if name=='stress' else ranges[batch]
                    vv=d['stress_valid'][batch-3500] if name=='stress' else valid[batch]
                    r=torch.from_numpy(rr).cuda();v=torch.from_numpy(vv).cuda();torch.cuda.synchronize();tick=time.perf_counter()
                    h=head(r,v);torch.cuda.synchronize();retrieval_seconds+=time.perf_counter()-tick
                    parts.append({k:value.cpu().numpy() for k,value in h.items()})
            avail={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
            for wrong in ([False] if name=='DEV' else [False,True]):
                key=name+('_wrong' if wrong else '');parts=[]
                with torch.inference_mode():
                    for begin in range(0,len(ii),16):
                        batch=ii[begin:begin+16];x=torch.from_numpy((np.array(maps[lookup[batch]])-n['mean'])/n['std']).cuda()
                        rr=d['stress_ranges'][batch-3500] if name=='stress' else ranges[batch];vv=d['stress_valid'][batch-3500] if name=='stress' else valid[batch]
                        r=torch.from_numpy(rr.astype(np.float32)).cuda();v=torch.from_numpy(vv).cuda();torch.cuda.synchronize();tick=time.perf_counter()
                        o=frozen.inspect(x,r,v,wrong);z=restrict_candidates(o,torch.from_numpy(avail['availability'][begin:begin+len(batch)]).cuda())
                        torch.cuda.synchronize();frozen_seconds+=time.perf_counter()-tick
                        parts.append(dict(raw=z['logits'].cpu().numpy(),support=z['support'].cpu().numpy(),original_raw=o['logits'].cpu().numpy(),original_support=o['support'].cpu().numpy()))
                        if not wrong:
                            eligible=o['eligible'].cpu().numpy();score=o['candidate_logits'].cpu().numpy()
                            positive=eligible&query[batch];mask=positive.reshape(len(batch),-1,4);present=mask.any(1)
                            pos=np.where(positive,score,-np.inf).reshape(len(batch),-1,4).argmax(1)
                            votes=np.broadcast_to(avail['votes'][begin:begin+len(batch),:,None,:,None],eligible.shape).reshape(len(batch),-1,4)
                            selected=np.take_along_axis(votes,pos[:,None,:],1)[:,0].astype(np.int16);selected[~present]=-1
                            parts[-1].update(teacher_witness_index=pos,teacher_witness_valid=present,teacher_witness_votes=selected,
                                actual_contributors=positive.sum((1,2,3)),rejected_contributors=(positive&(avail['votes'][begin:begin+len(batch),:,None,:,None]<3)).sum((1,2,3)))
                a={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};a.update(avail);prefix='BODY_RANK/'+key+'/'
                np.testing.assert_allclose(a['original_raw'],ref[prefix+'raw'],atol=2e-5,rtol=1e-5);np.testing.assert_array_equal(a['original_support'],ref[prefix+'support'])
                b,t,previous=(ref[prefix+k] for k in ['baseline','truth','candidate']);mz26=control[key+'/candidate']
                margin=np.where(a['support'],a['raw'].astype(float)-cut,-1e6);accept=(b<0)&a['support']&(margin>=0);candidate=np.where(accept,margin,b)
                assert not ((candidate>=0)&(previous<0)).any()
                a.update(baseline=b,truth=t,mz20=previous,mz26=mz26,margin=margin,accepted=accept,candidate=candidate,known=known[ii],global_ids=ii)
                result['metrics'][key]=dict(baseline=metrics(b,t),mz20=metrics(previous,t),mz26=metrics(mz26,t),candidate=metrics(candidate,t),
                    added_tp=(accept&t).sum(0).tolist(),added_fp=(accept&~t).sum(0).tolist(),mz20_tp_lost=((previous>=0)&(candidate<0)&t).sum(0).tolist(),
                    mz20_fp_removed=((previous>=0)&(candidate<0)&~t).sum(0).tolist(),mz26_tp_lost=((mz26>=0)&(candidate<0)&t).sum(0).tolist(),
                    mz26_tp_recovered=((mz26<0)&(candidate>=0)&t).sum(0).tolist(),mz26_fp_removed=((mz26>=0)&(candidate<0)&~t).sum(0).tolist(),mz26_fp_added=((mz26<0)&(candidate>=0)&~t).sum(0).tolist())
                if not wrong:
                    p=a['votes']>=3;k=known[ii];tp=int((p&k).sum());fp=int((p&~k).sum());fn=int((~p&k).sum());w=a['teacher_witness_valid']
                    result['availability'][key]=dict(tp=tp,fp=fp,fn=fn,positive=int(k.sum()),negative=int((~k).sum()),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),vote_histogram=np.bincount(a['votes'].flatten(),minlength=6).tolist(),nearest_distance_quantiles=np.quantile(a['neighbor_distances'][:,:,0],[0,.5,.9,.99,1]).tolist())
                    result['witnesses'][key]=dict(valid=w.sum(0).tolist(),retained=((a['teacher_witness_votes']>=3)&w).sum(0).tolist(),actual_contributors=a['actual_contributors'].sum(0).tolist(),rejected_contributors=a['rejected_contributors'].sum(0).tolist())
                if name in ['clean','stress']:result['thin'][key]=metrics(candidate[100:125],t[100:125])
                for k,v in a.items():arrays[key+'/'+k]=v
                print(key,'exact',result['metrics'][key]['candidate']['exact'],'addedTP',result['metrics'][key]['added_tp'],'FP',result['metrics'][key]['added_fp'],flush=True)
                write(output/'progress.json',dict(completed_cohort=key,training_steps=0))
        for name in ['relation10000','distance5000']:
            rr=[r for r in rows if r['dataset']==name];result['groups'][name]={}
            for field in ['family','group','site']:
                units={}
                for i,r in enumerate(rr):units.setdefault(r[field],[]).append(i)
                result['groups'][name][field]={u:dict(frames=len(ix),baseline=metrics(arrays[name+'/baseline'][ix],arrays[name+'/truth'][ix]),candidate=metrics(arrays[name+'/candidate'][ix],arrays[name+'/truth'][ix])) for u,ix in units.items()}
        m=result['metrics'];far=sum(sum(m[n]['added_tp'][1::2]) for n in ['relation10000','distance5000'])
        gates=dict(no_added_fp=all(sum(m[n]['added_fp'])==0 for n in cohorts),far_retention=far>=68,placement_far_gain=all(sum(m[n]['added_tp'][1::2])>0 for n in ['relation10000','distance5000']),
            pole_clean=sum(result['thin']['clean']['tp'][1::2])>=48,pole_stress=sum(result['thin']['stress']['tp'][1::2])>=48,
            baseline_positives_retained=all(not ((arrays[n+'/baseline']>=0)&(arrays[n+'/candidate']<0)).any() for n in cohorts))
        gates['useful_effect']=all(gates.values());result.update(gates=gates,retained_far_additions=far,retrieval_seconds=retrieval_seconds,frozen_scoring_seconds=frozen_seconds,retrieval_frames=sum(map(len,cohorts.values())))
        np.savez_compressed(output/'predictions.npz',**arrays);write(output/'result.json',result)
        write(output/'receipt.json',dict(status='PASS',training_steps=0,backend='CUDA fixed TRAIN-bank retrieval and frozen MZ20 inference',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started,
            inputs=inputs,code_sha256=code,outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
        print('PASS',gates,'far',far,flush=True)
    finally:
        frozen.cpu();head.cpu();del maps;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();existed=a.output.exists()
    try:main(a.root,a.output)
    except Exception as e:
        if not existed and a.output.exists():write(a.output/'failure.json',dict(status='FAIL',error_type=type(e).__name__,error=str(e),training_steps=0))
        raise

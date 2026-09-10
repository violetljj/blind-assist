"""Matched detail-only comparison; no new training/cutoff source exposure."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mz5_ensemble_readout import read,write,sha,load_npz
from mz15_train import balanced_local,cutoff_zero_added
from mz15_evaluator import align_stress_labels
from mz16_detail_readout import make_model
from mz16_local_metrics import summarize_local,threshold_at_fpr
from mz8_train import metrics


def main(root,cache,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();work=root/'artifacts.local/work'
    old=work/'mz8-attribution-20260910/cache-v5';new=work/'mz15-shared-support-20260910/cache-v1'
    prior=work/'mz15-shared-support-20260910/run-v1'
    cr=read(cache/'receipt.json');assert cr['status']=='PASS'
    inputs={}
    def bind(p,expected=None):
        h=sha(p);assert expected is None or h==expected,str(p);inputs[str(p)]=h
    for name,h in cr['files'].items():bind(cache/name,h)
    for folder,files,hashes in [(old,['observations.npz','evaluator.npz'],read(old/'receipt.json')['files']),
        (new,['observations.npz','evaluator.npz','selected.json'],read(new/'receipt.json')['files']),
        (prior,['batches.npy','QUERY-initial.pt'],read(prior/'receipt.json')['outputs'])]:
        for name in files:bind(folder/name,hashes[name])
    labels=work/'mz9-source-supervision-20260910/labels-v1'
    bind(labels/'evaluator.npz',read(labels/'receipt.json')['labels_sha256'])
    norm=work/'mz9-source-supervision-20260910/run-v1/normalization.npz';bind(norm)
    baseline=work/'mz13-training-coverage-20260910/run-v1/predictions.npz'
    bind(baseline,read(baseline.parent/'receipt.json')['outputs']['predictions.npz'])
    d,nd=load_npz(old/'observations.npz'),load_npz(new/'observations.npz')
    ol,nl=load_npz(labels/'evaluator.npz'),load_npz(new/'evaluator.npz')
    truth=np.concatenate([load_npz(old/'evaluator.npz')['truth'],nl['truth']])
    ranges,valid=(np.concatenate([d[k],nd[k]]) for k in ['ranges','valid'])
    query=np.concatenate([ol['query_counts']>0,nl['query_presence']])
    source=np.concatenate([ol['source_counts']>0,nl['source_presence']])
    known=np.concatenate([ol['cell_known_counts']>0,nl['known']]);del ol,nl
    b=load_npz(baseline);n=load_npz(norm)
    cohorts=dict(DEV=np.flatnonzero(d['role']=='DEV_ONLY'),clean=np.arange(3500,3700),
        stress=np.arange(3500,3700),relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    for name,ids in cohorts.items():np.testing.assert_array_equal(truth[ids],b[name+'/truth'])
    batches=np.load(prior/'batches.npy');np.save(output/'batches.npy',batches)
    cache_ids=np.load(cache/'ids.npy');lookup=np.full(14200,-1,dtype=int);lookup[cache_ids]=np.arange(len(cache_ids))
    expected=np.unique(np.r_[batches.flatten(),*cohorts.values()]);np.testing.assert_array_equal(cache_ids,expected)
    initial=torch.load(prior/'QUERY-initial.pt',weights_only=True)
    torch.set_num_threads(1);torch.manual_seed(115);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    write(output/'start.json',dict(status='STARTED',steps_per_arm=1200,inputs=inputs,
        train_batch_sha256=sha(output/'batches.npy'),initial_checkpoint_sha256=sha(prior/'QUERY-initial.pt'),
        code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mz16_detail_readout.py'),Path(__file__).with_name('mz16_local_metrics.py')]},
        protocol_sha256=sha(Path(__file__).with_name('MZ16_VISUAL_DETAIL_PROTOCOL_20260910.md'))))
    result=dict(metrics={},local={},fits={},cutoffs={},groups={});arrays={};local_arrays={}
    for arm in ['LOW_DETAIL','HIGH_DETAIL']:
        maps=np.load(cache/('dense_'+arm+'.npy'),mmap_mode='r')
        model=make_model(initial).cuda();torch.save(model.cpu().state_dict(),output/(arm+'-initial.pt'));model.cuda()
        def get(ids,stress=False):
            indices=lookup[ids];assert (indices>=0).all()
            dense=(np.array(maps[indices])-n['mean'])/n['std']
            r=d['stress_ranges'][ids-3500].astype(np.float32) if stress else ranges[ids]
            v=d['stress_valid'][ids-3500] if stress else valid[ids]
            return torch.from_numpy(dense).cuda(),torch.from_numpy(r).cuda(),torch.from_numpy(v).cuda()
        begin_fit=time.perf_counter();history=[];model.train();opt=torch.optim.Adam(model.parameters(),lr=.001)
        for step,ids in enumerate(batches):
            x,r,v=get(ids);out=model.inspect(x,r,v);q=torch.from_numpy(query[ids]).cuda()
            k=torch.from_numpy(known[ids]).cuda()[:,:,None,:,None]&v[:,:,:,None,None]
            local=balanced_local(out['field'],q,k);gt=torch.from_numpy(truth[ids]).cuda()
            positive=(q&out['eligible']).flatten(1,3).any(1);mask=out['support']&(~gt|positive)
            qloss=(F.binary_cross_entropy_with_logits(out['logits'],gt.float(),reduction='none')*mask).sum()/mask.sum().clamp_min(1)
            loss=local+.25*qloss;assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step()
            if step%200==0 or step==1199:
                row=dict(step=step+1,loss=float(loss.detach()),local=float(local.detach()),query=float(qloss.detach()))
                history.append(row);print(arm,row,flush=True);write(output/'progress.json',dict(arm=arm,**row))
        model.eval();torch.save(model.cpu().state_dict(),output/(arm+'.pt'));model.cuda()
        result['fits'][arm]=dict(steps=1200,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-begin_fit,history=history)
        def infer(name,wrong=False):
            ids=cohorts[name];parts=[];scores=[[] for _ in range(4)];labels_q=[[] for _ in range(4)];frames=[[] for _ in range(4)]
            with torch.inference_mode():
                for begin in range(0,len(ids),16):
                    ii=ids[begin:begin+16];x,r,v=get(ii,name=='stress');out=model.inspect(x,r,v,wrong)
                    sn,qn=source[ii],query[ii]
                    if name=='stress':sn,qn,_=align_stress_labels(sn,qn,ranges[ii],valid[ii],d['stress_ranges'][ii-3500],d['stress_valid'][ii-3500])
                    s=torch.from_numpy(sn).cuda();q=torch.from_numpy(qn).cuda()
                    eligible=out['eligible'];field=out['candidate_logits'];masked=field.masked_fill(~eligible,-1e6)
                    win=masked.flatten(1,3).argmax(1)
                    parts.append(dict(raw=out['logits'].cpu().numpy(),support=out['support'].cpu().numpy(),
                        winning_source=s[...,None].expand_as(field).flatten(1,3).gather(1,win[:,None]).squeeze(1).cpu().numpy(),
                        winning_query=q.flatten(1,3).gather(1,win[:,None]).squeeze(1).cpu().numpy()))
                    if not wrong:
                        localmask=eligible & torch.from_numpy(known[ii]).cuda()[:,:,None,:,None] & v[:,:,:,None,None]
                        for j in range(4):
                            mask=localmask[...,j];coordinates=mask.nonzero()
                            scores[j].append(field[...,j][mask].cpu().numpy());labels_q[j].append(q[...,j][mask].cpu().numpy())
                            frames[j].append(coordinates[:,0].cpu().numpy()+begin)
            return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]},[np.concatenate(v) if v else np.array([]) for v in scores],[np.concatenate(v) if v else np.array([],bool) for v in labels_q],[np.concatenate(v) if v else np.array([],int) for v in frames]
        dev,ds,dl,df=infer('DEV');cut=cutoff_zero_added(dev['raw'],dev['support'],b['DEV/baseline'],b['DEV/truth'])
        localcut=np.array([threshold_at_fpr(s,y) for s,y in zip(ds,dl)])
        np.save(output/(arm+'-cutoff.npy'),cut);np.save(output/(arm+'-local-cutoff.npy'),localcut)
        result['cutoffs'][arm]=dict(alert=cut.tolist(),local=localcut.tolist());result['metrics'][arm]={};result['local'][arm]={};result['groups'][arm]={}
        for name in cohorts:
            for wrong in ([False] if name=='DEV' else [False,True]):
                key=name+('_wrong' if wrong else '')
                a,ls,ll,lf=(dev,ds,dl,df) if name=='DEV' else infer(name,wrong)
                base,gt=b[name+'/baseline'],b[name+'/truth'];margin=np.where(a['support'],a['raw'].astype(np.float64)-cut,-1e6)
                accept=(base<0)&a['support']&(margin>=0);candidate=np.where(accept,margin,base)
                a.update(margin=margin,baseline=base,truth=gt,accepted=accept,candidate=candidate)
                result['metrics'][arm][key]=dict(baseline=metrics(base,gt),candidate=metrics(candidate,gt),
                    added_tp=(accept&gt).sum(0).tolist(),added_fp=(accept&~gt).sum(0).tolist(),
                    false_winner_no_source=(accept&~gt&~a['winning_source']).sum(0).tolist(),no_candidate=(~a['support']).sum(0).tolist())
                for k,val in a.items():arrays[arm+'/'+key+'/'+k]=val
                if not wrong:
                    result['local'][arm][key]=summarize_local(ls,ll,localcut)
                    counts=np.zeros((len(gt),4,5),dtype=np.int64)
                    for q,(s,y,f) in enumerate(zip(ls,ll,lf)):
                        local_arrays[arm+'/'+key+f'/{q}/score']=s;local_arrays[arm+'/'+key+f'/{q}/label']=y;local_arrays[arm+'/'+key+f'/{q}/frame']=f
                        p=s.astype(np.float64)>=localcut[q]
                        for j,m in enumerate([p&y,p&~y,~p&y,y,~y]):counts[:,q,j]=np.bincount(f[m],minlength=len(gt))
                    arrays[arm+'/'+key+'/local_counts']=counts
                print('EVAL',arm,key,result['metrics'][arm][key],flush=True)
        model.cpu()
        records=read(new/'selected.json')
        for name in ['relation10000','distance5000']:
            rows=[r for r in records if r['dataset']==name];result['groups'][arm][name]={}
            for field in ['family','group','site']:
                units={}
                for i,r in enumerate(rows):units.setdefault(r[field],[]).append(i)
                result['groups'][arm][name][field]={u:dict(frames=len(ii),baseline=metrics(arrays[arm+'/'+name+'/baseline'][ii],arrays[arm+'/'+name+'/truth'][ii]),candidate=metrics(arrays[arm+'/'+name+'/candidate'][ii],arrays[arm+'/'+name+'/truth'][ii])) for u,ii in units.items()}
    result['thin']={};result['gates']={}
    for arm in ['LOW_DETAIL','HIGH_DETAIL']:
        m=result['metrics'][arm];thin={}
        for name in ['clean','stress','clean_wrong','stress_wrong']:
            thin[name]=metrics(arrays[arm+'/'+name+'/candidate'][100:125],arrays[arm+'/'+name+'/truth'][100:125])
        result['thin'][arm]=thin
        gates=dict(no_added_fp=all(sum(m[n]['added_fp'])==0 for n in cohorts),
            placement_far_gain=all(sum(m[n]['added_tp'][1::2])>0 for n in ['relation10000','distance5000']),
            thin_clean=sum(thin['clean']['tp'][1::2])>=48,thin_stress=sum(thin['stress']['tp'][1::2])>=48,
            near_retained=all(all(m[n]['candidate']['tp'][q]>=m[n]['baseline']['tp'][q] for q in [0,2]) for n in cohorts))
        gates['useful_effect']=all(gates.values());result['gates'][arm]=gates
    result['higher_detail_local_ap_gain']=all(result['local']['HIGH_DETAIL'][n]['macro_ap']>result['local']['LOW_DETAIL'][n]['macro_ap'] for n in ['relation10000','distance5000'])
    write(output/'result.json',result);np.savez_compressed(output/'predictions.npz',**arrays);np.savez_compressed(output/'local_samples.npz',**local_arrays)
    torch.cuda.synchronize();write(output/'receipt.json',dict(status='PASS',backend='CUDA',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started,cache_receipt_sha256=sha(cache/'receipt.json'),outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('PASS',result['gates'],'LOCAL_AP_GAIN',result['higher_detail_local_ap_gain'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','cache','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();main(a.root,a.cache,a.output)

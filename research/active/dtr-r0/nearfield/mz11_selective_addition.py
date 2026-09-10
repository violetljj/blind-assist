"""One bounded correction gate; existing positive judgments are immutable."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mz5_ensemble_readout import CompactEnsemble, load_npz, read, sha, write
from mz9_source_readout import SourceReadout
from mz8_train import metrics


def gate_features(source, count, zones, returns):
    return np.stack([source/4, np.log1p(count)/6, zones/8, returns/8], -1).astype(np.float32)


def compose(baseline, source, available, score, threshold):
    eligible = (baseline < 0) & (source >= 0) & available
    accepted = eligible & (score >= threshold)
    return np.where(accepted, source, baseline), accepted


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    start=time.perf_counter()
    cache=root/'artifacts.local/work/mz8-attribution-20260910/cache-v5'
    mz9=root/'artifacts.local/work/mz9-source-supervision-20260910/run-v1'
    mz10=root/'artifacts.local/work/mz10-availability-20260910/run-v2'
    cr=read(cache/'receipt.json');r9=read(mz9/'receipt.json');r10=read(mz10/'receipt.json')
    for name in ['observations.npz','evaluator.npz','dense.npy']:
        assert sha(cache/name)==cr['files'][name]
    for name in ['SOURCE_RGB.pt','normalization.npz','result.json','predictions.npz','raw-support.npz']:
        assert sha(mz9/name)==r9['outputs'][name]
    assert sha(mz10/'predictions.npz')==r10['outputs']['predictions.npz']
    assert sha(cache/'receipt.json')==r9['cache_receipt_sha256']==r10['cache_receipt_sha256']
    obs=load_npz(cache/'observations.npz');truth=load_npz(cache/'evaluator.npz')['truth']
    n=load_npz(mz9/'normalization.npz');maps=np.load(cache/'dense.npy',mmap_mode='r')
    p9=load_npz(mz9/'predictions.npz');raw=load_npz(mz9/'raw-support.npz');p10=load_npz(mz10/'predictions.npz')
    source_threshold=np.array(read(mz9/'result.json')['dev']['SOURCE_RGB']['threshold'])
    torch.set_num_threads(1);torch.manual_seed(111);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    model=SourceReadout().cuda().eval();model.load_state_dict(torch.load(mz9/'SOURCE_RGB.pt',weights_only=True))
    checkpoint=root/'artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/compact.pt'
    assert sha(checkpoint)=='ccbfd6817bfec226f0db74e7ff4b22ce4f153255a9102d5c8684de91197486e1'
    fixed=CompactEnsemble.from_checkpoint(checkpoint).cuda()
    cohorts={};arrays={}
    groups={'TRAIN':np.flatnonzero(obs['role']=='TRAIN_ONLY'),'DEV':np.flatnonzero(obs['role']=='DEV_ONLY'),
            'clean':np.arange(3500,3700),'stress':np.arange(3500,3700),
            'clean_wrong':np.arange(3500,3700),'stress_wrong':np.arange(3500,3700)}
    with torch.inference_mode():
        for cohort,ids in groups.items():
            parts={k:[] for k in ['baseline','source','available','count','zones','returns']}
            for begin in range(0,len(ids),32):
                batch=ids[begin:begin+32]
                ranges=obs['stress_ranges'][batch-3500] if cohort.startswith('stress') else obs['ranges'][batch]
                valid=obs['stress_valid'][batch-3500] if cohort.startswith('stress') else obs['valid'][batch]
                r=torch.from_numpy(ranges.astype(np.float32)).cuda();v=torch.from_numpy(valid).cuda()
                x=torch.from_numpy((np.array(maps[batch])-n['mean'])/n['std']).cuda()
                tokens=model.sample_visual(x)
                out=model.forward_tokens(tokens,r,v,wrong_zone=cohort.endswith('wrong'),return_details=True)
                e=out['eligible']
                packed=torch.cat([r.flatten(1)/4,v.flatten(1).float()],1)
                parts['baseline'].append(fixed(torch.from_numpy(obs['visual'][batch]).cuda(),packed).cpu().numpy())
                parts['source'].append(out['logits'].cpu().numpy()-source_threshold)
                parts['available'].append(out['support'].cpu().numpy())
                parts['count'].append(e.sum((1,2,3)).cpu().numpy())
                parts['zones'].append(e.any(3).any(2).sum(1).cpu().numpy())
                parts['returns'].append(e.any(3).sum((1,2)).cpu().numpy())
            a={k:np.concatenate(v) for k,v in parts.items()}
            a['source']=np.where(a['available'],a['source'],-1e6)
            if cohort=='DEV':
                np.testing.assert_array_equal(a['baseline']>=0,p10['DEV/baseline']>=0)
                np.testing.assert_array_equal(a['source']>=0,p10['DEV/source']>=0)
                np.testing.assert_array_equal(a['available'],p10['DEV/available'])
                a['source']=p10['DEV/source'];a['baseline']=p10['DEV/baseline']
            elif cohort not in ['TRAIN','DEV']:
                condition=cohort.split('_')[0];key='SOURCE_RGB/'+condition+('/wrong' if cohort.endswith('wrong') else '')
                np.testing.assert_array_equal(a['source']>=0,p9[key]>=0)
                np.testing.assert_array_equal(a['baseline']>=0,p10[condition+'/baseline']>=0)
                np.testing.assert_array_equal(a['available'],raw[key+'/support'])
                a['source']=p9[key];a['baseline']=p10[condition+'/baseline']
            a['features']=gate_features(np.where(a['available'],a['source'],0),a['count'],a['zones'],a['returns'])
            a['eligible']=(a['baseline']<0)&(a['source']>=0)&a['available']
            a['truth']=truth[ids];a['cache_index']=ids
            cohorts[cohort]=a
    write(output/'start-receipt.json',dict(seed=111,steps=600,parameters=20,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).with_name('MZ11_SELECTIVE_ADDITION_PROTOCOL_20260910.md')),
        training_group_counts={k:{'positive':(cohorts[k]['eligible']&cohorts[k]['truth']).sum(0).tolist(),
            'negative':(cohorts[k]['eligible']&~cohorts[k]['truth']).sum(0).tolist()} for k in ['TRAIN','clean']}))
    # Separate per-query linear gates; no cross-query label information.
    weight=torch.nn.Parameter(torch.zeros(4,4,device='cuda'));bias=torch.nn.Parameter(torch.zeros(4,device='cuda'))
    opt=torch.optim.AdamW([weight,bias],lr=.03,weight_decay=.001)
    fit={k:{v:torch.from_numpy(cohorts[k][v]).cuda() for v in ['features','eligible','truth']} for k in ['TRAIN','clean']}
    history=[]
    for step in range(600):
        losses=[]
        for a in fit.values():
            z=(a['features']*weight).sum(-1)+bias
            element=F.binary_cross_entropy_with_logits(z,a['truth'].float(),reduction='none')
            for q in range(4):
                for positive in [False,True]:
                    mask=a['eligible'][:,q]&(a['truth'][:,q]==positive)
                    if mask.any():losses.append(element[:,q][mask].mean())
        loss=torch.stack(losses).mean();assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();opt.step()
        if step in [0,199,399,599]:
            history.append(dict(step=step+1,loss=float(loss.detach())))
            print(history[-1],flush=True)
    torch.save(dict(weight=weight.detach().cpu(),bias=bias.detach().cpu()),output/'gate.pt')
    for a in cohorts.values():
        with torch.inference_mode():a['score']=((torch.from_numpy(a['features']).cuda()*weight).sum(-1)+bias).cpu().numpy()
    dev=cohorts['DEV'];threshold=[]
    for q in range(4):
        eligible=dev['eligible'][:,q];negative=eligible&~dev['truth'][:,q]
        threshold.append(np.nextafter(dev['score'][negative,q].max(),np.float32(np.inf)) if negative.any()
                         else (-np.inf if eligible.any() else np.inf))
    threshold=np.array(threshold,dtype=np.float32)
    np.save(output/'threshold.npy',threshold)
    result={'metrics':{},'clips':{},'fit_history':history}
    for cohort,a in cohorts.items():
        z,accepted=compose(a['baseline'],a['source'],a['available'],a['score'],threshold)
        a['candidate']=z;a['accepted']=accepted
        result['metrics'][cohort]={'candidate':metrics(z,a['truth']),'baseline':metrics(a['baseline'],a['truth']),
            'added_tp':(accepted&a['truth']).sum(0).tolist(),'added_fp':(accepted&~a['truth']).sum(0).tolist(),
            'baseline_positive_lost':int(((a['baseline']>=0)&(z<0)).sum())}
        if cohort not in ['TRAIN','DEV']:
            result['clips'][cohort]={clip:metrics(z[obs['clip']==clip],a['truth'][obs['clip']==clip]) for clip in dict.fromkeys(obs['clip'].tolist())}
        arrays.update({cohort+'/'+k:v for k,v in a.items()})
    m=result['metrics'];clips=result['clips']
    gate={'no_fp_increase':all(not any(m[c]['added_fp']) for c in ['DEV','clean','stress']),
          'all_baseline_positives_preserved':all(m[c]['baseline_positive_lost']==0 for c in m),
          'dev_far_gain':sum(m['DEV']['added_tp'][1::2])>0,
          'thin_clean_retained':sum(clips['clean']['thin_pole_background_target']['tp'][1::2])>=48,
          'thin_stress_retained':sum(clips['stress']['thin_pole_background_target']['tp'][1::2])>=48}
    result['gate']=gate;result['confirmation_candidate']=all(gate.values())
    result['wrong_correspondence_reduces_thin']=sum(clips['clean_wrong']['thin_pole_background_target']['tp'][1::2])<sum(clips['clean']['thin_pole_background_target']['tp'][1::2])
    result['comparators']={'mz9_dev':read(mz9/'result.json')['dev'],'mz9_sequence':read(mz9/'result.json')['metrics'],
                          'mz10':read(mz10/'result.json')}
    np.savez_compressed(output/'predictions.npz',**arrays,clip=obs['clip'])
    write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS',training_steps=600,parameters=20,backend='CUDA',device=torch.cuda.get_device_name(),
        seconds=time.perf_counter()-start,cache_receipt_sha256=sha(cache/'receipt.json'),mz9_receipt_sha256=sha(mz9/'receipt.json'),
        mz10_receipt_sha256=sha(mz10/'receipt.json'),outputs={p.name:sha(p) for p in output.iterdir()}))
    print('RESULT', {k:v for k,v in result.items() if k not in ['comparators','clips']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)

"""Frozen no-fit composition, with support from observable geometry only."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import CompactEnsemble,read,write,sha,load_npz
from mz9_source_readout import SourceReadout
from mz8_train import metrics


def compose(baseline,source,available):
    if baseline.shape!=source.shape or source.shape!=available.shape or available.dtype!=np.bool_:
        raise ValueError('Matching logit/support arrays and boolean support required')
    return np.where(available,source,baseline)


def transitions(candidate,baseline,truth):
    c,b=candidate>=0,baseline>=0
    return dict(tp_gained=(c&~b&truth).sum(0).tolist(),tp_lost=(~c&b&truth).sum(0).tolist(),
        fp_added=(c&~b&~truth).sum(0).tolist(),fp_removed=(~c&b&~truth).sum(0).tolist())


def main(root,output):
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    cache=root/'artifacts.local/work/mz8-attribution-20260910/cache-v5'
    run=root/'artifacts.local/work/mz9-source-supervision-20260910/run-v1'
    receipt=read(run/'receipt.json');cr=read(cache/'receipt.json')
    assert receipt['status']=='PASS' and sha(cache/'receipt.json')==receipt['cache_receipt_sha256']
    for name in ['SOURCE_RGB.pt','MZ5_ADAPT.pt','normalization.npz','result.json','predictions.npz','raw-support.npz']:
        assert sha(run/name)==receipt['outputs'][name]
    for name in ['dense.npy','observations.npz','evaluator.npz']:assert sha(cache/name)==cr['files'][name]
    old=load_npz(cache/'observations.npz');ev=load_npz(cache/'evaluator.npz');dev=np.flatnonzero(old['role']=='DEV_ONLY')
    n=load_npz(run/'normalization.npz');maps=np.load(cache/'dense.npy',mmap_mode='r');prior=read(run/'result.json')
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    model=SourceReadout().cuda().eval();model.load_state_dict(torch.load(run/'SOURCE_RGB.pt',weights_only=True))
    checkpoint=root/'artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/compact.pt'
    assert sha(checkpoint)=='ccbfd6817bfec226f0db74e7ff4b22ce4f153255a9102d5c8684de91197486e1'
    fixed=CompactEnsemble.from_checkpoint(checkpoint).cuda()
    adapted=CompactEnsemble.from_checkpoint(run/'MZ5_ADAPT.pt').cuda()
    source=[];support=[];adapt=[]
    with torch.inference_mode():
        visual=torch.from_numpy(old['visual'][dev]).cuda()
        tof=torch.from_numpy(np.concatenate([old['ranges'][dev].reshape(-1,128)/4,old['valid'][dev].reshape(-1,128).astype(np.float32)],1)).cuda()
        baseline=fixed(visual,tof).cpu().numpy()
        for begin in range(0,len(dev),32):
            ids=dev[begin:begin+32]
            x=torch.from_numpy((np.array(maps[ids])-n['mean'])/n['std']).cuda()
            out=model.inspect(x,torch.from_numpy(old['ranges'][ids]).cuda(),torch.from_numpy(old['valid'][ids]).cuda())
            source.append(out['logits'].cpu().numpy());support.append(out['support'].cpu().numpy())
            adapt.append(adapted(visual[begin:begin+32],tof[begin:begin+32]).cpu().numpy())
    support=np.concatenate(support)
    source=np.where(support,np.concatenate(source)-np.array(prior['dev']['SOURCE_RGB']['threshold']),-1e6)
    adapt=np.concatenate(adapt)-np.array(prior['dev']['MZ5_ADAPT']['threshold'])
    assert metrics(source,ev['truth'][dev])==prior['dev']['SOURCE_RGB']['dev']
    assert metrics(adapt,ev['truth'][dev])==prior['dev']['MZ5_ADAPT']['dev']
    assert metrics(baseline,ev['truth'][dev])==prior['baseline_dev']
    arrays={'DEV/baseline':baseline,'DEV/source':source,'DEV/adapted':adapt,
        'DEV/available':support,'DEV/truth':ev['truth'][dev],
        'DEV/cache_index':dev,'DEV/original_frame_id':old['old_index'][dev],
        'DEV/composite':compose(baseline,source,support)}
    seq=load_npz(run/'predictions.npz');raw=load_npz(run/'raw-support.npz')
    frozen_seq_path=root/'artifacts.local/work/mz6-short-sequence-20260910/evaluation-v1/predictions.npz'
    frozen_seq=load_npz(frozen_seq_path)
    for condition in ['clean','stress']:
        arrays[condition+'/baseline']=frozen_seq[condition+'/CURRENT']
        np.testing.assert_array_equal(arrays[condition+'/baseline']>=0,seq['FROZEN']>=0)
        arrays[condition+'/source']=seq['SOURCE_RGB/'+condition]
        arrays[condition+'/adapted']=seq['MZ5_ADAPT/'+condition]
        arrays[condition+'/available']=raw['SOURCE_RGB/'+condition+'/support']
        arrays[condition+'/truth']=seq['truth']
        arrays[condition+'/composite']=compose(arrays[condition+'/baseline'],arrays[condition+'/source'],arrays[condition+'/available'])
    result={};records=[]
    for cohort in ['DEV','clean','stress']:
        truth=arrays[cohort+'/truth'];s=arrays[cohort+'/available'];b=arrays[cohort+'/baseline'];c=arrays[cohort+'/composite']
        result[cohort]=dict(metrics={a:metrics(arrays[cohort+'/'+a],truth) for a in ['baseline','adapted','source','composite']},
            selected_source=s.sum(0).tolist(),selected_baseline=(~s).sum(0).tolist(),
            changes=transitions(c,b,truth),
            composite_fp_from_source=((c>=0)&~truth&s).sum(0).tolist(),
            composite_fp_from_fallback=((c>=0)&~truth&~s).sum(0).tolist())
        for i in range(len(truth)):
            for q in range(4):
                if (c[i,q]>=0)!=(b[i,q]>=0):records.append(dict(cohort=cohort,index=i,query=q,
                    truth=bool(truth[i,q]),source_selected=bool(s[i,q]),baseline=float(b[i,q]),composite=float(c[i,q])))
    result['clips']={}
    for clip in dict.fromkeys(seq['clip'].tolist()):
        mask=seq['clip']==clip
        result['clips'][clip]={a:metrics(arrays['clean/'+a][mask],seq['truth'][mask]) for a in ['baseline','adapted','source','composite']}
    db=result['DEV']['metrics']['baseline'];dc=result['DEV']['metrics']['composite']
    sb=result['clean']['metrics']['baseline'];sc=result['clean']['metrics']['composite']
    thin=result['clips']['thin_pole_background_target']['composite']
    gate=dict(dev_fp_preserved=all(a<=b for a,b in zip(dc['fp'],db['fp'])),
        dev_near_preserved=all(dc['tp'][q]>=db['tp'][q] for q in [0,2]),dev_far_preserved=sum(dc['tp'][1::2])>=sum(db['tp'][1::2]),
        sequence_fp_preserved=all(a<=b for a,b in zip(sc['fp'],sb['fp'])),sequence_near_preserved=all(sc['tp'][q]>=sb['tp'][q] for q in [0,2]),
        sequence_far_improved=sum(sc['tp'][1::2])>sum(sb['tp'][1::2]),thin_retained=sum(thin['tp'][1::2])>=48)
    result['gate']=gate;result['confirmation_admitted']=all(gate.values())
    np.savez_compressed(output/'predictions.npz',**arrays,clip=seq['clip'])
    write(output/'result.json',result);write(output/'changed-bits.json',records)
    write(output/'receipt.json',dict(status='PASS',training_steps=0,threshold_changes=0,backend='CUDA',device=torch.cuda.get_device_name(),
        seconds=time.perf_counter()-start,cache_receipt_sha256=sha(cache/'receipt.json'),mz9_receipt_sha256=sha(run/'receipt.json'),
        frozen_sequence_predictions_sha256=sha(frozen_seq_path),
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).with_name('MZ10_AVAILABILITY_PROTOCOL_20260910.md')),
        outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print({k:v for k,v in result.items() if k!='clips'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)

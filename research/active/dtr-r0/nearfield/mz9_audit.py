"""Scalar scoring, contributor contradictions, and old-DEV coverage audit."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import CompactEnsemble,load_npz,read,write,sha
from mz9_source_readout import SourceReadout
from mz8_train import metrics


def main(root,cache,supervision,run):
    p=load_npz(run/'predictions.npz');r=read(run/'result.json');groups=0
    for arm,clips in r['metrics'].items():
        for clip,expected in clips.items():
            ids=[i for i in range(200) if clip=='ALL' or p['clip'][i]==clip]
            tp=[0]*4;fp=[0]*4;pos=[0]*4;exact=0
            for i in ids:
                correct=0
                for q in range(4):
                    v=bool(p[arm][i,q]>=0);t=bool(p['truth'][i,q]);tp[q]+=v and t;fp[q]+=v and not t;pos[q]+=t;correct+=v==t
                exact+=correct==4
            assert expected==dict(exact=exact,tp=tp,fp=fp,positives=pos);groups+=1
    torch.set_num_threads(1)
    d=load_npz(cache/'observations.npz');e=load_npz(cache/'evaluator.npz');lab=load_npz(supervision/'evaluator.npz')
    dev=np.flatnonzero(d['role']=='DEV_ONLY');truth=e['truth'][dev]
    model=SourceReadout().cuda().eval();model.load_state_dict(torch.load(run/'SOURCE_RGB.pt',weights_only=True))
    n=load_npz(run/'normalization.npz');maps=np.load(cache/'dense.npy',mmap_mode='r')
    outputs=[];supports=[];source_support=[]
    with torch.inference_mode():
        for begin in range(0,len(dev),32):
            ids=dev[begin:begin+32]
            x=torch.from_numpy((np.array(maps[ids])-n['mean'])/n['std']).cuda()
            out=model.inspect(x,torch.from_numpy(d['ranges'][ids]).cuda(),torch.from_numpy(d['valid'][ids]).cuda())
            outputs.append(out['logits'].cpu().numpy());supports.append(out['support'].cpu().numpy())
            local=torch.from_numpy(lab['query_counts'][ids]>0).cuda()
            source_support.append((out['eligible']&local).flatten(1,3).any(1).cpu().numpy())
        source_z=np.concatenate(outputs);support=np.concatenate(supports);source_support=np.concatenate(source_support)
        threshold=np.array(r['dev']['SOURCE_RGB']['threshold']);source_z=np.where(support,source_z-threshold,-1e6)
        assert metrics(source_z,truth)==r['dev']['SOURCE_RGB']['dev']
        visual=torch.from_numpy(d['visual'][dev]).cuda()
        tof=torch.from_numpy(np.concatenate([d['ranges'][dev].reshape(-1,128)/4,d['valid'][dev].reshape(-1,128).astype(np.float32)],1)).cuda()
        old=CompactEnsemble.from_checkpoint(root/'artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/compact.pt').cuda()
        baseline=old(visual,tof).cpu().numpy()
        assert metrics(baseline,truth)==r['baseline_dev']
        adapted=CompactEnsemble.from_checkpoint(run/'MZ5_ADAPT.pt').cuda()
        adapted_z=np.concatenate([adapted(visual[i:i+32],tof[i:i+32]).cpu().numpy() for i in range(0,len(dev),32)])-np.array(r['dev']['MZ5_ADAPT']['threshold'])
        assert metrics(adapted_z,truth)==r['dev']['MZ5_ADAPT']['dev']
    loss=(baseline>=0)&(source_z<0)&truth;gain=(baseline<0)&(source_z>=0)&truth
    old_winners=read(root/'artifacts.local/work/mz8-attribution-20260910/run-v1/audit.json')['rows']
    empty=[v for v in old_winners if v['index']>=125]
    assert len(empty)==22
    corrected=sum(lab['query_counts'][3500+v['index'],v['zone'],v['return_index'],v['sample'],v['query']]==0 for v in empty)
    out=dict(status='PASS',scalar_groups=groups,dev_replay_bits=8000,
        old_false_positive_winner_labels_corrected=int(corrected),
        dev_positive_geometry_ceiling=(support&truth).sum(0).tolist(),
        dev_positive_exact_source_ceiling=(source_support&truth).sum(0).tolist(),
        source_lost_baseline_tp=loss.sum(0).tolist(),source_gained_tp=gain.sum(0).tolist(),
        lost_tp_without_any_geometry=(loss&~support).sum(0).tolist(),
        lost_tp_without_exact_source=(loss&~source_support).sum(0).tolist(),
        adapted_lost_baseline_tp=((baseline>=0)&(adapted_z<0)&truth).sum(0).tolist(),
        adapted_gained_tp=((baseline<0)&(adapted_z>=0)&truth).sum(0).tolist(),
        script_sha256=sha(Path(__file__)))
    write(run/'audit.json',out);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','cache','supervision','run']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();main(a.root,a.cache,a.supervision,a.run)

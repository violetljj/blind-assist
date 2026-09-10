"""Scalar scoring and winning-return diagnosis; no fitting or model changes."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import read,write,load_npz,sha
from mz8_attribution import AngularReturnReadout,query_membership


def main(cache,run):
    p=load_npz(run/'predictions.npz');result=read(run/'result.json')['metrics']
    groups=0
    for arm,clips in result.items():
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
    model=AngularReturnReadout().cuda().eval();model.load_state_dict(torch.load(run/'model.pt',weights_only=True))
    n=load_npz(run/'normalization.npz');d=load_npz(cache/'observations.npz');e=load_npz(cache/'evaluator.npz')
    maps=np.load(cache/'dense.npy',mmap_mode='r');threshold=np.array(read(run/'thresholds.json')['thresholds'])
    rows=[]
    with torch.inference_mode():
        for begin in range(3600,3650,10):
            x=torch.from_numpy((np.array(maps[begin:begin+10])-n['mean'])/n['std']).cuda()
            r=torch.from_numpy(d['ranges'][begin:begin+10]).cuda();v=torch.from_numpy(d['valid'][begin:begin+10]).cuda()
            out=model.inspect(x,r,v)
            z=np.where(out['support'].cpu().numpy(),out['logits'].cpu().numpy()-threshold,-1e6)
            np.testing.assert_array_equal(z>=0,p['clean/correct'][begin-3500:begin-3490]>=0)
            candidates=out['candidate_logits'].cpu().numpy();eligible=out['eligible'].cpu().numpy()
            for b in range(len(x)):
                i=begin+b
                for q in [1,3]:
                    if z[b,q]<0:continue
                    score=np.where(eligible[b,...,q],candidates[b],-np.inf)
                    zone,k,s=np.unravel_index(score.argmax(),score.shape)
                    echo=float(d['ranges'][i,zone,k]);native=float(e['sampled_radial'][i,zone,s])
                    paired=i+25 if i<3625 else i-25
                    other=d['ranges'][paired,zone][d['valid'][paired,zone]]
                    close=bool(len(other) and np.min(np.abs(other-echo))<=.10)
                    grid=model.grid[zone,s].cpu().numpy()
                    px=np.rint((grid[0]+1)*320-.5);py=np.rint((grid[1]+1)*180-.5)
                    focal=320/math.tan(math.radians(50));ray=np.array([1,(px-319.5)/focal,-(py-179.5)/focal]);ray/=np.linalg.norm(ray)
                    native_point=ray*native+np.array([0,0,1.7])
                    hypothetical=model.rays[zone,s].cpu().numpy()*echo+np.array([0,0,1.7])
                    rows.append(dict(index=i-3500,query=q,truth=bool(e['truth'][i,q]),zone=int(zone),return_index=int(k),sample=int(s),
                        echo_m=echo,sampled_native_m=native if np.isfinite(native) else None,
                        native_compatible=bool(np.isfinite(native) and abs(native-echo)<=.10),
                        hypothetical_xyz=hypothetical.tolist(),sampled_native_xyz=native_point.tolist() if np.isfinite(native) else None,
                        sampled_native_inside_query=bool(query_membership(torch.tensor(native_point))[q]) if np.isfinite(native) else False,
                        paired_packet_has_similar_echo=close))
    target=[x for x in rows if x['index']<125];empty=[x for x in rows if x['index']>=125]
    out=dict(status='PASS',scalar_metric_groups=groups,winner_replay_bits=200,
        target_winners=len(target),target_winners_similar_echo_in_empty=sum(x['paired_packet_has_similar_echo'] for x in target),
        target_winners_native_compatible=sum(x['native_compatible'] for x in target),
        empty_false_positive_winners=len(empty),empty_winners_native_compatible=sum(x['native_compatible'] for x in empty),
        empty_winners_native_inside_query=sum(x['sampled_native_inside_query'] for x in empty),
        interpretation='Nearest-pixel0.10m compatibility and paired similar echo are descriptive diagnostics, not return identity or exhaustive geometry',
        rows=rows,script_sha256=sha(Path(__file__)))
    write(run/'audit.json',out);print({k:v for k,v in out.items() if k!='rows'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.cache,a.run)

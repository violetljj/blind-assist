"""Controlled angular inverse problem from motion-coded single-zone returns.

Finite scene-family feasibility, not calibrated sensing or neural reconstruction.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from body_query_collection_labels import read,write,sha
from contact_retina_spec import BODY_BOXES
from tof_jitter_geometry import trajectory,footprint


def catalog(device):
    centers=np.array([(y,z) for y in range(-9,10) for z in range(-9,10)])
    low=2.5*np.tan(np.deg2rad(centers-1));high=2.5*np.tan(np.deg2rad(centers+1))
    low[:,1]+=1.7;high[:,1]+=1.7
    bl,bh=BODY_BOXES[1]
    head=(low[:,0]<=bh[1])&(high[:,0]>=bl[1])&(low[:,1]<=bh[2])&(high[:,1]>=bl[2])
    pairs=np.array([(i,j) for i in range(len(centers)) for j in range(i+1,len(centers)) if (np.abs(centers[i]-centers[j])>=2).any()])
    scene_head=np.concatenate(([False],head,head[pairs].any(1)))
    family=np.concatenate(([0],np.ones(len(centers),int),np.full(len(pairs),2,int)))
    poses=[dict(pitch_deg=0.,yaw_deg=0.)]+trajectory()
    yy,xx=torch.meshgrid(torch.arange(360,device=device),torch.arange(640,device=device),indexing='ij')
    f=320/np.tan(np.deg2rad(50));rx=(xx-319.5)/f;rz=-(yy-179.5)/f
    angles=torch.stack((torch.rad2deg(torch.atan(rx)),torch.rad2deg(torch.atan(rz))),-1)
    crop=(angles.abs()<=10).all(-1)
    angle=angles[crop];c=torch.as_tensor(centers,device=device)
    masks=((angle[None]>=c[:,None]-1)&(angle[None]<c[:,None]+1)).all(-1).float()
    norm=(1+rx.square()+rz.square()).sqrt();weights=norm.pow(-3)
    assert float((2.5*norm[crop]).max())<2.6
    pose_weights=[]
    for pose in poses:
        mask,_,_=footprint(device,pose['pitch_deg'],pose['yaw_deg'])
        assert not (mask&~crop).any()
        assert float((3.5*norm[mask]).min())>=3.5 and float((3.5*norm[mask]).max())<3.6
        pose_weights.append((weights*mask)[crop]/weights[mask].sum())
    coverage=torch.stack(pose_weights)@masks.T
    pair_tensor=torch.as_tensor(pairs,device=device)
    coverage=torch.cat((torch.zeros((26,1),device=device),coverage,coverage[:,pair_tensor].sum(-1)),1)
    assert float(coverage.max())<.5 and float((1-coverage).min())>=.02
    # Return code is near-bin versus background-bin, after hypothetical support.
    code=coverage>=.02
    return centers,pairs,head,scene_head,family,code,coverage


def infer(predicted,observed,valid,scene_head):
    # Crucially, compatibility does not read HEAD membership. It retains every
    # scene matching the range sequence, including multiple-return-source scenes.
    compatible=((predicted==observed[:,None])|~valid[:,None]).all(0)
    count=int(compatible.sum());positive=int((compatible&scene_head).sum())
    return dict(feasible=count,head_possible=positive>0,head_guaranteed=count>0 and positive==count,
                empty=count==0,unknown=count==0 or 0<positive<count)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter()
    centers,pairs,head,scene_head,family,code,coverage=catalog('cuda')
    rng=np.random.default_rng(37);offset=1+len(centers)
    groups={'single_head':np.flatnonzero((family==1)&scene_head),'single_nonhead':np.flatnonzero((family==1)&~scene_head),
      'two_head':offset+np.flatnonzero(head[pairs].all(1)), 'two_nonhead':offset+np.flatnonzero(~head[pairs].any(1)),
      'two_mixed':offset+np.flatnonzero(head[pairs].sum(1)==1)}
    selected=np.concatenate([rng.choice(v,100,replace=False) for v in groups.values()])
    assert len(np.unique(selected))==500
    nr,dr=np.random.SeedSequence(41).spawn(2)
    noise=np.random.default_rng(nr).uniform(-.05,.05,(500,25));valid_np=np.random.default_rng(dr).random((500,25))>=.2
    truth=torch.as_tensor(scene_head,device='cuda');valid=torch.as_tensor(valid_np,device='cuda')
    outputs={};records=[]
    for law in ('nearest_supported','max_solid_angle_area','farthest_supported'):
        lawcode=code if law=='nearest_supported' else torch.zeros_like(code)
        for method in ('single','stationary','moving','moving_single_source_model'):
            template=lawcode[1:] if method.startswith('moving') else lawcode[1:2].expand(25,-1)
            # Baseline orientation matches the first jitter pose; only repeated
            # motion differs. Row0 is kept as an extra aligned geometry reference.
            sizes=family<=1 if method=='moving_single_source_model' else np.ones(len(family),bool)
            hypothesis=template[:,sizes];hypothesis_truth=truth[sizes]
            per=[]
            for j,index in enumerate(selected):
                raw=template[:,index].cpu().numpy()
                ranges=np.where(raw,2.55,3.55)+noise[j]
                observed=torch.as_tensor(ranges<3.05,device='cuda')
                keep=valid[j].clone()
                if method=='single':keep[1:]=False
                result=infer(hypothesis,observed,keep,hypothesis_truth)
                result.update(case=j,scene=int(index),true_head=bool(scene_head[index]))
                per.append(result)
            pos=[r for r in per if r['true_head']];neg=[r for r in per if not r['true_head']]
            count=lambda n,d:dict(numerator=int(n),denominator=int(d))
            outputs[law+'/'+method]=dict(head_coverage=count(sum(r['head_guaranteed'] for r in pos),len(pos)),
              false_head=count(sum(r['head_guaranteed'] for r in neg),len(neg)),unknown=sum(r['unknown'] for r in per),
              empty=sum(r['empty'] for r in per),median_feasible=float(np.median([r['feasible'] for r in per])))
            records.append(dict(law=law,method=method,rows=per))
    torch.cuda.synchronize();a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'observations.npz',selected=selected,noise=noise,valid=valid_np,near_codes=code.cpu().numpy(),centers=centers,pairs=pairs,scene_head=scene_head)
    write(a.output/'rows.json',records)
    write(a.output/'result.json',dict(status='PASS',results=outputs,scene_hypotheses=len(family),cases=500,coverage_max=float(coverage.max()),
      backend='CUDA geometry and feasibility',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-start,
      operator_sha256={n:sha(Path(__file__).with_name(n)) for n in ('tof_motion_scan.py','tof_jitter_geometry.py','contact_retina_spec.py','body_query_collection_labels.py')},
      observations_sha256=sha(a.output/'observations.npz'),rows_sha256=sha(a.output/'rows.json'),
      scope='Controlled finite matched hypothesis family. Zero false positives with complete family follow from true-scene inclusion, not empirical safety validation. No original B changes.'))
    print(outputs)


if __name__=='__main__':main()

"""Multi-object CUDA ray/native visible support; never sends truth to model inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from contact_retina_spec import BODY_BOXES, MAP_SHA

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.inference_mode()
def verify(root):
    root=root.resolve(); assert not (root/'verification.json').exists()
    receipt=read(root/'receipt.json');assert receipt['status']=='PASS' and receipt['source_unchanged']
    assert read(root/'process-release.json')['released']
    spec=read(root/'evaluator/spec.json');data=read(root/'model/dataset.json')
    assert sha(root/'evaluator/spec.json')==receipt['spec_sha256'] and spec['map_sha256']==MAP_SHA
    assert data['samples']==spec['samples'] and len(data['frames'])==len(spec['cases'])
    assert all(set(f)=={'sample_index','clip_id','frame_in_clip','rgb_path','time_s'} for f in data['frames'])
    assert torch.cuda.is_available();torch.set_num_threads(1);device='cuda';started=time.perf_counter()
    yy,xx=torch.meshgrid(torch.arange(360,device=device),torch.arange(640,device=device),indexing='ij')
    focal=320/np.tan(np.radians(50));rx,ry=(xx-319.5)/focal,(yy-179.5)/focal
    sample_at={s['frame_indices'][-1]:s for s in data['samples']}
    labels,support,object_support,rows={},{},{},[]
    for i,(frame,case) in enumerate(zip(data['frames'],spec['cases'])):
        assert frame==dict(sample_index=i,clip_id=case['clip_id'],frame_in_clip=case['frame_in_clip'],rgb_path=f'sample/{i:04d}.png',time_s=case['time_s'])
        native=torch.as_tensor(np.load(root/f'evaluator/native/{i:04d}.npy'),device=device)
        assert native.shape==(360,640)
        cam=case['camera']; p,y=np.radians([cam['pitch'],cam['yaw']])
        forward=torch.tensor([np.cos(p)*np.cos(y),np.cos(p)*np.sin(y),np.sin(p)],device=device)
        right=torch.tensor([-np.sin(y),np.cos(y),0],device=device)
        up=torch.tensor([-np.sin(p)*np.cos(y),-np.sin(p)*np.sin(y),np.cos(p)],device=device)
        direction=forward+rx[...,None]*right-ry[...,None]*up
        origin=torch.tensor([cam[k] for k in ('x','y','z')],device=device)
        points=origin+native[...,None]*direction
        assert int(((native>0)&((points[...,2]-.12).abs()<.02)).sum())>500
        object_masks={};counts={}
        for obj in case['objects']:
            center=torch.tensor(obj['center_m'],device=device);half=torch.tensor(obj['size_m'],device=device)/2
            a,b=(center-half-origin)/direction,(center+half-origin)/direction
            near=torch.minimum(a,b).amax(-1);far=torch.maximum(a,b).amin(-1)
            visible=(far>=near)&(near>0)&(native>0)&((native-near).abs()<.03)
            object_masks[obj['name']]=visible;counts[obj['name']]=int(visible.sum())
            assert counts[obj['name']]>=3, f'Invisible intended object {i} {obj["name"]}'
        visible=torch.zeros((360,640),device=device,dtype=torch.bool)
        for mask in object_masks.values():visible|=mask
        masks=[];target=[];body=case['wearer']
        for low,high in BODY_BOXES:
            q=visible&(points[...,0]>=body['x']+high[0])&(points[...,0]<=body['x']+high[0]+3)
            q&=(points[...,1]>=body['y']+low[1])&(points[...,1]<=body['y']+high[1])
            q&=(points[...,2]>=body['z']+low[2])&(points[...,2]<=body['z']+high[2])
            masks.append(q);target.append(int(q.sum()>=3))
        intended=[int('box' in object_masks),int('bar' in object_masks)]
        assert target==intended, f'Wrong/occluded intended query {i}: {target} vs {intended}'
        if i in sample_at:
            sid=sample_at[i]['sample_id'];labels[sid]=target+[-1,-1]
            support[sid]=F.adaptive_max_pool2d(torch.stack(masks)[None].float(),(18,32))[0].byte().cpu().numpy()
            for name in ('bar','box'):
                m=object_masks.get(name,torch.zeros_like(visible))
                object_support[sid+'__'+name]=F.adaptive_max_pool2d(m[None,None].float(),(18,32))[0,0].byte().cpu().numpy()
        rows.append(dict(sample_index=i,visible_pixels=counts,rgb_sha256=sha(root/'model'/frame['rgb_path']),native_sha256=sha(root/f'evaluator/native/{i:04d}.npy')))
    # Exact pose and retained object identity across each factorial quartet.
    for group in {s['group_id'] for s in spec['samples']}:
        members=[s for s in spec['samples'] if s['group_id']==group]
        assert len(members)==4
        cs=[spec['cases'][s['frame_indices'][-1]] for s in members]
        assert all(c['camera']==cs[0]['camera'] and c['wearer']==cs[0]['wearer'] for c in cs)
        for name in ('bar','box'):
            objs=[o for c in cs for o in c['objects'] if o['name']==name];assert len(objs)==2 and objs[0]==objs[1]
    (root/'evaluator/labels.json').write_text(json.dumps(dict(targets=labels),indent=2),encoding='utf-8')
    np.savez_compressed(root/'evaluator/support.npz',**support);np.savez_compressed(root/'evaluator/object_support.npz',**object_support)
    result=dict(status='PASS',frames=len(rows),samples=len(labels),rows=rows,backend='CUDA',device=torch.cuda.get_device_name(),elapsed_s=time.perf_counter()-started,dataset_sha256=sha(root/'model/dataset.json'),code_sha256=sha(Path(__file__)),scope='All intended task objects visible, labels match intended quartet; no all-scene free-space or causal-mechanism certification')
    (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);verify(p.parse_args().capture)

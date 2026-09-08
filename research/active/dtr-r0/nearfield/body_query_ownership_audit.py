"""Hash-bound native ownership versus frozen point readouts, no model inference."""
from pathlib import Path
import argparse,time
import numpy as np
import torch
import torch.nn.functional as F
from body_query_data import read,write,sha,truth,fresh_output
from body_query_model import fixed_projection,projection_weights,query_boxes,CALIBRATION
from worlds_verify import camera_points,corridor_masks


def membership(native):
    valid=torch.isfinite(native)&(native>0)&(native<100)
    p=camera_points(native,dict(x=0.,y=0.,z=0.,pitch=0.,yaw=0.,roll=0.))
    p[...,2]+=CALIBRATION['height_m']
    heads=corridor_masks(p,valid,dict(x=0.,y=0.,z=0.),3.)
    cells=[]
    for c,(lo,hi) in enumerate(query_boxes()):
        m=heads[c//6].clone()
        m&=(p[...,0]>=lo[0])&((p[...,0]<=hi[0]) if c%6//3==1 else (p[...,0]<hi[0]))
        m&=(p[...,1]>=lo[1])&((p[...,1]<=hi[1]) if c%3==2 else (p[...,1]<hi[1]))
        cells.append(m)
    cells=torch.stack(cells)
    assert torch.equal(cells.reshape(2,6,*native.shape).sum(1),heads.long())
    return cells,valid,p


def run(a):
    start=time.perf_counter();out=fresh_output(a.output)
    if not torch.cuda.is_available():raise RuntimeError('Native geometry requires GPU')
    torch.set_num_threads(1)
    grid,valid,_=fixed_projection();w=projection_weights(grid).cuda().reshape(12,27,576)
    xx=((grid[...,0]+1)*640/2).floor().long().clamp(0,639).cuda()
    yy=((grid[...,1]+1)*360/2).floor().long().clamp(0,359).cuda()
    ci=torch.arange(12,device='cuda')[:,None]
    world=read(a.world);manifest=read(a.cache/'manifest.json')
    assert sha(a.world)==manifest['world_verification_sha256']
    q2receipt=read(a.points/'receipt.json');hashes={str(a.world):sha(a.world)}
    results={}
    for role in ('train','dev','eval'):
        rec,gt=truth(a.cache,role);path=a.points/f'{role}-points.npz'
        assert sha(path)==q2receipt['arrays'][path.name]
        dump=np.load(path);assert np.array_equal(dump['valid'],valid.numpy())
        rows=[]
        for i,r in enumerate(rec['records']):
            index=r['sample_index'];path=a.native/f'{index:04d}.npy'
            h=sha(path);assert h==world['rows'][index]['native_sha256'];hashes[str(path)]=h
            n=torch.from_numpy(np.load(path)).cuda();cells,known,p=membership(n)
            assert np.array_equal(cells.sum((-2,-1)).clamp_max(3).cpu().numpy(),gt['counts'][i])
            owned=cells[ci,yy,xx];kn=known[yy,xx]
            local=torch.einsum('cpk,ck->cp',w,F.max_pool2d(cells.float()[:,None],20).flatten(1))
            # Ray categories are descriptive and mutually exclusive, not free-space labels.
            surface=p[yy,xx];x=surface[...,0];z=surface[...,2]
            category=torch.full_like(x,6,dtype=torch.int8) # other known surface
            category[(z>=.65)&(z<1.4)]=1;category[(z>=1.4)&(z<=1.85)]=2
            category[z<.65]=3;category[z>1.85]=4
            category[x>3.18]=5;category[~kn]=0
            rows.append((owned.cpu().numpy(),kn.cpu().numpy(),local.cpu().numpy(),category.cpu().numpy()))
        owned,known,local,category=[np.stack([r[j] for r in rows]) for j in range(4)]
        v=np.broadcast_to(valid.numpy(),owned.shape);owned&=v;known&=v;local*=v
        z=dump['point_logits'].astype(float);e=np.exp(z-z.max(-1,keepdims=True));strong=1-e[...,0]/e.sum(-1)>=.5
        cz=dump['cell_logits'].astype(float);ce=np.exp(cz-cz.max(-1,keepdims=True));miss=(gt['counts']>0)&(1-ce[...,0]/ce.sum(-1)<.5)
        y=gt['counts']>0
        def stats(cols):
            pos=y[:,cols];o=owned[:,cols];k=known[:,cols];l=local[:,cols];s=strong[:,cols];vv=v[:,cols]
            return dict(positive=int(pos.sum()),ray_covered=int((pos&o.any(-1)).sum()),
                local_covered=int((pos&(l>0).any(-1)).sum()),missed=int(miss[:,cols].sum()),
                missed_ray_covered=int((miss[:,cols]&o.any(-1)).sum()),
                missed_local_covered=int((miss[:,cols]&(l>0).any(-1)).sum()),
                missed_owned_strong=int((miss[:,cols]&(o&s).any(-1)).sum()),
                point_counts={name:dict(total=int(mask.sum()),strong=int((mask&s).sum())) for name,mask in
                    [('owned',o),('known_other',k&~o),('unknown',vv&~k),('local_positive',l>0)]})
        result=dict(all=stats(np.arange(12)),heads={})
        for h,head in enumerate(('BODY','HEAD')):
            result['heads'][head]={name:stats(np.arange(h*6+lo,h*6+hi)) for name,lo,hi in [('all',0,6),('near',0,3),('far',3,6)]}
        ids=np.array([r['condition']=='BODY_ONLY' for r in rec['records']])
        selected=strong[ids,6:]&v[ids,6:];cat=category[ids,6:]
        result['body_only_head_strong_rays']={name:int((selected&(cat==j)).sum()) for j,name in enumerate(['UNKNOWN','BODY_HEIGHT','HEAD_HEIGHT','LOW','ABOVE','FAR_GT3_18','OTHER'])}
        np.savez_compressed(out/f'{role}-ownership.npz',owned=owned,known=known,local=local,category=category,valid=valid.numpy())
        results[role]=result;print(role,result['all'],flush=True)
    torch.cuda.synchronize()
    write(out/'result.json',dict(roles=results,seconds=time.perf_counter()-start,training_steps=0,inference_frames=0))
    write(out/'receipt.json',dict(status='PASS',backend='CUDA',device=torch.cuda.get_device_name(),
        native_frames=320,count_matches=320,code_sha256=sha(Path(__file__)),inputs=hashes,
        result_sha256=sha(out/'result.json'),arrays={p.name:sha(p) for p in out.glob('*.npz')}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('native','world','cache','points','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

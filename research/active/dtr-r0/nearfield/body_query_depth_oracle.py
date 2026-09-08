"""Privileged sampled depth cell routing; preserves native count distribution units."""
from pathlib import Path
import argparse,time
import numpy as np
from body_query_model import fixed_projection,query_boxes
from body_query_data import read,write,sha,truth,fresh_output
from body_query_point_audit import near
from body_query_direct_readout import evaluate
from city_dev_selection import select_threshold


def cell_gates(d,known,valid,xx,yy,qx,scale=1.,offset=0.):
    x=d*scale+offset;f=640/(2*np.tan(np.deg2rad(50)));y=(xx-319.5)/f*x;z=1.7-(yy-179.5)/f*x
    intervals=[];owned=[]
    for c,(lo,hi) in enumerate(query_boxes()):
        m=(x[c]>=lo[0])&((x[c]<=hi[0]) if c%6//3==1 else (x[c]<hi[0]))
        intervals.append(m)
        owned.append(m&(y[c]>=lo[1])&((y[c]<=hi[1]) if c%3==2 else (y[c]<hi[1]))&(z[c]>=lo[2])&(z[c]<=hi[2]))
    def pool(p):return np.where(valid,np.where(known,p,1.),0.).max(-1)
    return dict(interval=pool(np.array(intervals)),owned=pool(np.array(owned)),
        gaussian=pool(np.exp(-.5*((x-qx)/.375)**2)))


def run(a):
    out=fresh_output(a.output);start=time.perf_counter();grid,v,xyz=fixed_projection();grid=grid.numpy();v=v.numpy();qx=xyz.numpy()[...,0]*3.18
    xx=np.floor((grid[...,0]+1)*320).astype(int).clip(0,639);yy=np.floor((grid[...,1]+1)*180).astype(int).clip(0,359)
    world=read(a.world);assert sha(a.world)==read(a.cache/'manifest.json')['world_verification_sha256']
    original=read(a.baseline/'selection.json')['B']['thresholds'];cached={};inputs={str(a.world):sha(a.world)};summary={};records={};targets={}
    for role in ('train','dev','eval'):
        rec,gt=truth(a.cache,role);records[role]=rec['records'];targets[role]=gt;path=a.baseline/f'B-{role}.npz';inputs[str(path)]=sha(path);counts=np.load(path)['counts'].astype(float)
        rows={};observed=[];unknown=[]
        for r in rec['records']:
            i=r['sample_index'];path=a.native/f'{i:04d}.npy';h=sha(path);assert h==world['rows'][i]['native_sha256'];inputs[str(path)]=h
            d=np.load(path)[yy,xx].astype(float);known=np.isfinite(d)&(d>0)&(d<100);observed.append(d);unknown.append(~known&v)
            gates=cell_gates(d,known,v,xx,yy,qx)
            for name,scale,offset in [('scale09',.9,0),('scale11',1.1,0),('offset_m02',1,-.2),('offset_p02',1,.2)]:
                gates[name]=cell_gates(d,known,v,xx,yy,qx,scale,offset)['owned']
            for name,g in gates.items():rows.setdefault(name,[]).append(g)
        observed=np.stack(observed);unknown=np.stack(unknown)
        np.savez_compressed(out/f'{role}-residuals.npz',observed=observed,signed=observed-qx,absolute=abs(observed-qx),unknown=unknown,valid=v,query_depth=qx)
        summary[role]=dict(unknown_samples=int(unknown.sum()),unknown_passthrough_cells=int(unknown.any(-1).sum()),valid_samples=int(v.sum()*len(rec['records'])))
        for name,rows_g in rows.items():
            g=np.stack(rows_g);p=counts.copy();p[...,1:]*=g[...,None];p[...,0]=1-p[...,1:].sum(-1)
            assert np.all(p>=-1e-7) and np.allclose(p.sum(-1),1)
            # Original float32 counts may sum a few ulps away from1; remove only roundoff.
            p=np.clip(p,0,1);p/=p.sum(-1,keepdims=True)
            # Pure veto cannot raise B evidence. Preserve original exact boundary
            # scores when all six gates pass, avoiding softmax roundoff changes.
            original_near=np.load(a.baseline/f'B-{role}.npz')['near'].astype(float)
            n=np.minimum(near(p),original_near)
            n=np.where((g.reshape(-1,2,6)==1).all(-1),original_near,n)
            cached[name,role]=(n,1-p[...,0]);np.savez_compressed(out/f'{name}-{role}.npz',gate=g,counts=p,near=n)
    names=list(dict.fromkeys(n for n,role in cached));selection={}
    for name in names:
        selection[name]=[select_threshold(cached[name,'dev'][0][:,h],targets['dev']['near'][:,h],min_count=8)['threshold'] for h in range(2)]
    write(out/'selection.json',selection);selected_hash=sha(out/'selection.json');results={}
    for name in names:
        results[name]={}
        for role in ('train','dev','eval'):
            n,q=cached[name,role];results[name][role]=evaluate(n,q,targets[role],records[role],selection[name])
            results[name][role]['original_B_cutoff_heads']=evaluate(n,q,targets[role],records[role],original)['heads']
    assert sha(out/'selection.json')==selected_hash
    write(out/'result.json',dict(methods=results,unknown=summary,seconds=time.perf_counter()-start))
    write(out/'receipt.json',dict(status='PASS',training_steps=0,inference_frames=0,backend='CPU',reason='TASK_NOT_GPU_SUITABLE sampled scalar geometry',inputs=inputs,code_sha256=sha(Path(__file__)),result_sha256=sha(out/'result.json'),selection_sha256=selected_hash))
    print({name:r['eval']['heads'] for name,r in results.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('native','world','cache','baseline','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

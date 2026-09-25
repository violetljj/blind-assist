"""Read-only identity closure and exact triangle/1cm voxel audit of saved pilot."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from cnh_track_a_generate import BOXES, check_unit


def exact_voxels(triangles, width=.01):
    """Closed triangle/cube SAT. Includes touched voxels on either grid face."""
    cells=set(); half=width/2
    for tri in np.asarray(triangles).reshape(-1,3,3):
        lo=np.floor(tri.min(0)/width-1e-9).astype(int)
        hi=np.floor(tri.max(0)/width+1e-9).astype(int)
        indices=np.array(list(itertools.product(*(range(lo[k],hi[k]+1) for k in range(3)))))
        centers=(indices+.5)*width
        edges=np.roll(tri,-1,axis=0)-tri
        axes=[*np.eye(3),np.cross(edges[0],edges[1])]
        axes += [np.cross(e,a) for e in edges for a in np.eye(3)]
        good=np.ones(len(indices),bool)
        for a in axes:
            if np.linalg.norm(a)<1e-12:continue
            proj=tri@a; mid=centers@a;radius=half*np.abs(a).sum()
            good &= (mid+radius>=proj.min()-1e-12)&(mid-radius<=proj.max()+1e-12)
        cells.update(tuple(map(int,v)) for v in indices[good])
    return cells


def run(folder):
    start=time.monotonic(); allsets=[];errors=[];counts={};seeds=[];hashes={};units=[]
    for path in sorted(folder.glob('unit[0-9][0-9].json')):
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        unit=json.loads(path.read_text());u=unit['unit'];units.append(u)
        expected_split='train' if u<6 else 'calib' if u<8 else 'audit'
        if unit['split']!=expected_split:errors.append(f'u{u}:split')
        if len(unit['configs'])!=20:errors.append(f'u{u}:config_count')
        for c in unit['configs']:
            key=f"{u}/{c['config']}";seeds.append(c['seed'])
            y=np.array(c['labels']);m=np.array(c['margins']);p=np.array(c['world_from_Q']);a=np.array(c['angles'])
            objects=c['objects'];ids=[o['id'] for o in objects]
            if len(ids)!=len(set(ids)) or ids!=c['object_ids']:errors.append(key+':ids')
            if not np.isfinite(m).all() or not np.isfinite(p).all():errors.append(key+':finite')
            if y.shape!=(12,6) or not np.array_equal(y,m.max(1)>=-1e-10):errors.append(key+':labels')
            boundary=(np.abs(m)<.05).any((1,2));main=(np.arange(12)>=3)&~boundary
            if not np.array_equal(boundary,c['boundary']) or not np.array_equal(main,c['main']):errors.append(key+':subset')
            r=p[:,:3,:3]
            if np.max(np.abs(r@r.transpose(0,2,1)-np.eye(3)))>1e-6:errors.append(key+':rotation')
            if np.max(np.abs(np.diff(p[:,2,3])-c['speed']*.2))>1e-6:errors.append(key+':speed')
            if not ((a>=[-15,-10,-30])&(a<=[10,10,30])).all() or np.abs(np.diff(a[:,2])).max()>12+1e-9:errors.append(key+':head_pose')
            for t in range(12):
                for q in range(6):
                    expected=np.array(ids)[m[t,:,q]>=-1e-10].tolist()
                    if c['contributors'][t][q]!=expected:errors.append(key+':contributor')
                    z=c['witness_z'][t][q]
                    if y[t,q] and (z is None or not .3-1e-8<=z<=3+1e-8):errors.append(key+':witness')
            tris=[]
            for o in objects:
                world=np.asarray(o['triangles_world'])
                if world.ndim!=3 or world.shape[1:]!=(3,3) or not np.isfinite(world).all():errors.append(key+':mesh')
                local=(world-p[7,:3,3])@r[7]
                if np.max(np.abs(local@r[7].T+p[7,:3,3]-world))>1e-6:errors.append(key+':roundtrip')
                cat=o['category'];counts[cat]=counts.get(cat,0)+1
                if cat in ('HEAD','BODY'):tris.append(local)
            if tris:allsets.append((u,c['config'],exact_voxels(np.concatenate(tris))))
    duplicates=[];near={}
    for i,(ua,ca,fa) in enumerate(allsets):
        for ub,cb,fb in allsets[i+1:]:
            intersection=len(fa&fb);union=len(fa)+len(fb)-intersection;j=intersection/max(1,union)
            if fa==fb:duplicates.append([ua,ca,ub,cb])
            if ua!=ub and j>=.98:
                near.setdefault((ua,ub),set()).add(ca);near.setdefault((ub,ua),set()).add(cb)
    near_failed=[dict(unit=a,other=b,count=len(cs)) for (a,b),cs in near.items() if len(cs)>=.9*sum(u==a for u,_,_ in allsets)]
    if len(seeds)!=len(set(seeds)):errors.append('seed_collision')
    return dict(schema='cnh.track-a.readonly-audit.v1',completed_units=units,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes=hashes,
        G0=dict(pass_gate=not errors,errors=errors,config_count=len(seeds),frame_count=len(seeds)*12),
        G1=dict(pass_gate=not duplicates and not near_failed,nonempty_configs=len(allsets),exact_duplicates=duplicates,near_failed=near_failed,
                definition='exact closed triangle/cube SAT at 1cm; HEAD/BODY related surfaces in anchor Q; low/background separately declared, no texture/ID'),
        category_objects=counts,wall_s=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=run(args.input);args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('G0','G1','category_objects','wall_s')}))

"""Track A bounded geometry-first pilot. Never renders past a failed gate."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from cnh_track_a_fov import rx, rz
from cnh_track_a_geometry import box_mesh, cylinder_mesh, signed_margin, clip_triangles

BOXES = np.array([[[x-.3,y[0],.3],[x+.3,y[1],3]] for x in (-.3,0,.3) for y in ((-.2,.42),(.42,.9))])
PATTERNS = [(0,0,0),(1,0,0),(1,1,0),(1,1,1),(0,1,1),(0,0,1),(1,0,1)]
# Independent fixed anchor schedule; no selection from emitted frame statistics.
SCHEDULE = [(0,0),(0,0),(1,0),(5,0),(2,0),(4,0),(0,1),(0,5),
            (0,2),(0,4),(3,3),(3,3),(6,2),(2,6),(4,1),(1,4),
            (1,0),(0,5),(5,0),(0,1)]


def ry(deg):
    a=np.deg2rad(deg)
    return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def head_path(rng):
    a=np.empty((12,3)); a[0]=rng.uniform([-15,-10,-30],[10,10,30])
    lo=np.array([-15,-10,-30]); hi=np.array([10,10,30])
    for t in range(1,12):
        step=rng.normal(0,[1,.6,1.5]); step[2]=np.clip(step[2],-12,12)
        v=a[t-1]+step
        v=np.where(v<lo,2*lo-v,v); v=np.where(v>hi,2*hi-v,v)
        a[t]=v
    return a


def surface_fingerprint(triangles):
    """Deterministic 1 cm voxel occupancy of surfaces; ignore rho and IDs."""
    occupied=set()
    for tri in triangles:
        n=max(1,int(np.ceil(max(np.linalg.norm(tri[i]-tri[j]) for i,j in ((0,1),(1,2),(2,0)))/.005)))
        for i in range(n+1):
            t=np.arange(n-i+1)/n
            pts=tri[0]+(tri[1]-tri[0])*(i/n)+t[:,None]*(tri[2]-tri[0])
            good=(pts[:,0]>=-.9)&(pts[:,0]<=.9)&(pts[:,2]>=.2)&(pts[:,2]<=4)
            occupied.update(tuple(map(int,v)) for v in np.floor(pts[good]/.01).astype(int))
    return occupied


def targets(pattern):
    return {0:[],1:[-.46],2:[-.16],3:[-.16,.16],4:[.16],5:[.46],6:[-.46,.46]}[pattern]


def make_object(split, center, rng, category, boundary=False, wide=False):
    # Disjoint composite surface templates, not merely renamed object IDs.
    x,y,z=center
    r=rng.uniform(.008,.024)
    half_x=.48 if wide else r
    h=.14 if category=='HEAD' else .11
    if split=='train':
        mesh=box_mesh(np.array(center)-[half_x,h/2,r],np.array(center)+[half_x,h/2,r])
        family='rectangular-prism'
    elif split=='calib':
        mesh=cylinder_mesh(np.array(center),r,half_x*2 if wide else h,axis=0 if wide else 1,sides=64)
        family='cylinder64'
    else:
        # Two detached short plates belong to one rigid composite object.
        mesh=np.concatenate([box_mesh(np.array(center)+[-half_x,-h/2,d-r/3],np.array(center)+[half_x,h/2,d+r/3]) for d in (-r,r)])
        family='paired-plate-composite'
    return dict(triangles=mesh, rho=float(rng.uniform(.1,.9)), family=family, category=category,
                center=list(map(float,center)), radius=float(r), wide=wide,suspended=category=='HEAD', boundary=boundary)


def describe_config(unit,ci,split,height,width):
    seed=20260925+1009*unit+1000003*(ci+1)
    rng=np.random.default_rng(seed)
    angles=head_path(rng); speed=float(rng.uniform(.8,1.4))
    origins=np.array([[0,-height,(t-7)*.2*speed] for t in range(12)])
    rotations=np.array([ry(a[2]) for a in angles])
    poses=np.repeat(np.eye(4)[None],12,axis=0); poses[:,:3,:3]=rotations; poses[:,:3,3]=origins
    wanted=np.array([PATTERNS[SCHEDULE[ci][q%2]][q//2] for q in range(6)])
    logs=[]
    for attempt in range(32):
        objects=[]
        for group,pattern in enumerate(SCHEDULE[ci]):
            for x in ([0.] if pattern==3 else targets(pattern)):
                z=(.70,1.50,2.50)[(ci+group)%3]+rng.uniform(-.12,.12)
                y=.08 if group==0 else .65
                x+=rng.uniform(-.025,.025)
                if ci>=16:
                    # Intentional front-boundary contact band at anchor.
                    z=.3+(.02 if ci in (16,17) else -.04)
                obj=make_object(split,[x,y,z],rng,'HEAD' if group==0 else 'BODY',ci>=16,pattern==3)
                obj['id']=len(objects)+1
                objects.append(obj)
        # A genuinely below-BODY obstacle, present also in all-zero anchors.
        low=box_mesh(np.array([.7,height-.25,1.3]),np.array([.85,height,1.55]))
        objects.append(dict(id=len(objects)+1,triangles=low,rho=float(rng.uniform(.1,.9)),family='low-background',category='LOW',center=[.775,height-.125,1.425],radius=.075,suspended=False,boundary=False))
        margins=np.array([[signed_margin(o['triangles'],*b) for b in BOXES] for o in objects])
        labels=(margins.max(0)>=-1e-10).astype(int)
        # Boundary anchors have separately designated physical side; target bit
        # is still checked and failures retained, never moved during trajectory.
        expected=wanted.copy()
        if ci in (18,19): expected[:]=0
        margin_ok=bool(np.all(np.abs(margins)>=.05)) if ci<16 else bool(np.any(np.abs(margins)<.05))
        if np.array_equal(labels,expected) and margin_ok:
            break
        logs.append(dict(attempt=attempt,labels=labels.tolist(),expected=expected.tolist(),margin_ok=margin_ok))
    else:
        return None,logs
    related=[o['triangles'] for o in objects if o['category'] in ('HEAD','BODY')]
    anchor_local=np.concatenate(related) if related else np.empty((0,3,3))
    # Freeze world meshes at anchor. Background world axes independent of head.
    for o in objects:
        o['triangles_world']=o['triangles']@rotations[7].T+origins[7]
    backgrounds=[box_mesh([-width,0,-3],[width,.1,7])]
    if unit%4!=3: backgrounds.append(box_mesh([-width,-height-2,-3],[-width+.1,0,7]))
    if unit%4==0: backgrounds.append(box_mesh([width-.1,-height-2,-3],[width,0,7]))
    if unit%4 in (1,2):
        backgrounds.append(box_mesh([width-.1,-height-2,-3],[width,0,3.5]))
        backgrounds.append(box_mesh([width-.1,-height-2,3.5],[width+3,0,3.6]))
    if unit%4==2: backgrounds.append(box_mesh([-width-3,-height-2,3.5],[-width,0,3.6]))
    all_objects=objects+[dict(id=100+i,triangles_world=t,rho=.5,family='background',category='BACKGROUND') for i,t in enumerate(backgrounds)]
    labels=[]; margins=[]; contributors=[]; witness_z=[]
    for t in range(12):
        row=[]; zrow=[]
        for o in all_objects:
            local=(o['triangles_world']-origins[t])@rotations[t]
            margins_q=[signed_margin(local,*b) for b in BOXES]
            row.append(margins_q)
            zq=[]
            for margin,b in zip(margins_q,BOXES):
                clipped=clip_triangles(local,*b) if margin>=-1e-10 else np.empty((0,3,3))
                if len(clipped):
                    zq.append(float(clipped[...,2].min()))
                elif margin>=-1e-10:
                    _,witness=signed_margin(local,*b,return_witness=True)
                    zq.append(float(witness[2]))
                else:
                    zq.append(None)
            zrow.append(zq)
        row=np.array(row)
        margins.append(row)
        labels.append((row.max(0)>=-1e-10).astype(int))
        contributors.append([np.array([o['id'] for o in all_objects])[row[:,q]>=-1e-10].tolist() for q in range(6)])
        witness_z.append([min([zrow[i][q] for i in range(len(row)) if zrow[i][q] is not None],default=None) for q in range(6)])
    margins=np.array(margins); labels=np.array(labels)
    boundary=(np.abs(margins)<.05).any(axis=(1,2))
    main=(np.arange(12)>=3)&~boundary
    return dict(unit=unit,config=ci,seed=seed,split=split,speed=speed,height=height,angles=angles.tolist(),world_from_Q=poses.tolist(),
        target_labels=expected.tolist(),labels=labels.tolist(),margins=margins.tolist(),object_ids=[o['id'] for o in all_objects],
        contributors=contributors,witness_z=witness_z,boundary=boundary.tolist(),main=main.tolist(),
        objects=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in o.items() if k!='triangles'} for o in all_objects],
        fingerprint=sorted(surface_fingerprint(anchor_local)), rejected=logs),logs


def check_unit(configs):
    y=np.concatenate([np.array(c['labels'])[c['main']] for c in configs])
    ids=np.concatenate([np.full(sum(c['main']),c['config']) for c in configs])
    n=len(y); failures=[]; metrics={'N_main':n,'total_frames':sum(len(c['labels']) for c in configs),'boundary_frames':sum(sum(c['boundary']) for c in configs)}
    if not n:return dict(**metrics,pass_gate=False,failures=['empty_main'])
    def count(name,mask,old,minimum_configs=0):
        actual=int(mask.sum()); numcfg=len(set(ids[mask].tolist())); need=math.ceil(old*n/160)
        metrics[name]=dict(actual=actual,required=need,configs=numcfg,required_configs=minimum_configs)
        if actual<need or numcfg<minimum_configs:failures.append(name)
    metrics['positive_counts']=y.sum(0).tolist(); metrics['positive_rates']=y.mean(0).tolist()
    if not ((y.mean(0)>=.15)&(y.mean(0)<=.85)).all():failures.append('positive_rate')
    for q in range(6):
        count(f'q{q}_positive',y[:,q]==1,24);count(f'q{q}_negative',y[:,q]==0,24)
    metrics['HB_disagreement']=[]
    for i in range(3):
        h,b=y[:,i*2],y[:,i*2+1]; d=float((h!=b).mean());metrics['HB_disagreement'].append(d)
        if d<.2:failures.append(f'HB_disagreement_{i}')
        count(f'H1B0_{i}',(h==1)&(b==0),8,2);count(f'H0B1_{i}',(h==0)&(b==1),8,2)
    head=y[:,::2]; body=y[:,1::2]
    count('HEAD_only',head.any(1)&~body.any(1),16,2);count('BODY_only',body.any(1)&~head.any(1),16,2)
    count('asymmetric',(y[:,:2]!=y[:,4:]).any(1),32,4)
    count('left_only',((y[:,:2]==1)&(y[:,4:]==0)).any(1),8)
    count('right_only',((y[:,:2]==0)&(y[:,4:]==1)).any(1),8)
    multi=np.concatenate([np.array([len(set(sum(cs,[])))>=2 for cs in c['contributors']])[c['main']] for c in configs])
    count('multi',multi,32,4);count('all_zero',~y.any(1),16,2)
    combos={''.join(map(str,row)):dict(frames=0,configs=set()) for row in y}
    for row,ci in zip(y,ids):
        entry=combos[''.join(map(str,row))];entry['frames']+=1;entry['configs'].add(int(ci))
    metrics['combinations']={k:dict(frames=v['frames'],configs=sorted(v['configs'])) for k,v in sorted(combos.items())}
    metrics['eligible_combinations']=sum(len(v['configs'])>=2 for v in combos.values())
    if metrics['eligible_combinations']<12:failures.append('eligible_combinations')
    metrics['phi']={}
    for a,b in ((0,1),(2,3),(4,5),(0,2),(4,2),(1,3),(5,3)):
        value=float(np.corrcoef(y[:,a],y[:,b])[0,1]) if y[:,a].std()>0 and y[:,b].std()>0 else None
        metrics['phi'][f'{a}-{b}']=value
        if value is None or abs(value)>.8:failures.append(f'phi_{a}_{b}')
    if any(((g[:,0]==0)&(g[:,1]==1)&(g[:,2]==0)).any() for g in (head,body)):failures.append('infeasible_010')
    zs=np.concatenate([np.array([[np.nan if z is None else z for z in row] for row in c['witness_z']])[c['main']] for c in configs])
    for lo,hi in ((.3,1),(1,2),(2,3.00000001)):
        mask=(zs>=lo)&(zs<hi)&(y==1); actual=int(mask.sum()); nc=len(set(ids[mask.any(1)].tolist()));need=math.ceil(24*n/160)
        metrics[f'distance_{lo}_{hi}']=dict(actual=actual,required=need,configs=nc,required_configs=3)
        if actual<need or nc<3:failures.append(f'distance_{lo}_{hi}')
    mm=np.concatenate([np.array(c['margins']).reshape(-1) for c in configs])
    metrics['margin_quantiles']=np.quantile(mm,[0,.1,.25,.5,.75,.9,1]).tolist()
    return dict(**metrics,pass_gate=not failures,failures=failures)


def check_coordinates(configs):
    max_orth=max_roundtrip=0.
    for c in configs:
        poses=np.array(c['world_from_Q'])
        for t,pose in enumerate(poses):
            pitch,roll,_=c['angles'][t]
            for mount in (0,-10):
                r=pose[:3,:3]@rx(pitch)@rz(roll)@rx(mount)
                max_orth=max(max_orth,float(np.abs(r.T@r-np.eye(3)).max()))
                point=np.array([.2,.6,1.7]); world=r@point+pose[:3,3]
                max_roundtrip=max(max_roundtrip,float(np.abs(r.T@(world-pose[:3,3])-point).max()))
    return dict(pass_gate=max_orth<=1e-6 and max_roundtrip<=1e-6,max_orth=max_orth,max_roundtrip_m=max_roundtrip,
                frames=sum(len(c['labels']) for c in configs),unknown=0)


def run(output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);started=time.monotonic();units=[];fingerprints=[];attempts=rejects=0
    source_files=['cnh_track_a_generate.py','cnh_track_a_geometry.py','cnh_track_a_fov.py','CNH_NEARFIELD_ACCEPTANCE_PLAN_V1_1_20260925.md']
    save(output/'manifest.json',dict(schema='cnh.track-a.geometry.v1.1',source_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in source_files},
        axes='X right, Y gravity down, Z horizontal head yaw',units='metres; degrees; seconds',seed=20260925,split={'train':list(range(6)),'calib':[6,7],'audit':[8,9,10,11]},
        scene='opaque complete procedural triangle surfaces',limit_units=12,limit_hours=4,limit_gib=20))
    reserved=[20360925+1009*i for i in range(64)]
    save(output/'confirmation-seed-reservation.json',dict(namespace='cnh-track-a-confirmation-v1',seeds=reserved,sha256=hashlib.sha256(json.dumps(reserved,separators=(',',':')).encode()).hexdigest(),generated_content=False))
    split_configs={'train':[],'calib':[],'audit':[]}
    for u in range(12):
        split='train' if u<6 else 'calib' if u<8 else 'audit'
        rng=np.random.default_rng(20260925+1009*u);height=float(rng.uniform(1.45,1.75));width=float(rng.uniform(4,6))
        configs=[];t0=time.monotonic()
        for ci in range(20):
            if time.monotonic()-started>4*3600:
                save(output/'result.json',dict(status='STOP_WALL_BUDGET',completed_units=units,downstream='NOT_RUN',wall_s=time.monotonic()-started));return
            c,logs=describe_config(u,ci,split,height,width);rejects+=len(logs);attempts+=len(logs)+(c is not None)
            if c is None:
                save(output/f'failure-unit{u:02d}-config{ci:02d}.json',dict(rejected=logs))
                save(output/'result.json',dict(status='STOP_GENERATION_FAIL',unit=u,config=ci,completed_units=units,attempts=attempts,rejections=rejects,wall_s=time.monotonic()-started,downstream='NOT_RUN'))
                return
            configs.append(c)
        checks=check_unit(configs)
        g0=check_coordinates(configs)
        metadata=dict(unit=u,split=split,shape=('straight','L','T','open')[u%4],height=height,width=width,configs=configs,checks=checks,G0=g0,wall_s=time.monotonic()-t0)
        save(output/f'unit{u:02d}.json',metadata)
        units.append({k:v for k,v in metadata.items() if k!='configs'})
        split_configs[split].extend([dict(c,config=u*20+c['config']) for c in configs])
        fingerprints.extend((u,c['config'],set(map(tuple,c['fingerprint']))) for c in configs if c['fingerprint'])
        print(json.dumps(dict(unit=u,N_main=checks['N_main'],G2=checks['pass_gate'],failures=checks['failures'],wall_s=metadata['wall_s'])),flush=True)
    duplicates=[];near={}
    for i,(ua,ca,fa) in enumerate(fingerprints):
        for ub,cb,fb in fingerprints[i+1:]:
            if ua==ub:continue
            j=len(fa&fb)/max(1,len(fa|fb))
            if j==1:duplicates.append([ua,ca,ub,cb])
            if j>=.98:near.setdefault((ua,ub),set()).add(ca);near.setdefault((ub,ua),set()).add(cb)
    near_fail=[dict(unit=a,other=b,count=len(cs)) for (a,b),cs in near.items() if len(cs)>=.9*sum(u==a for u,_,_ in fingerprints)]
    split_checks={s:check_unit(cs) for s,cs in split_configs.items()}
    for checks in split_checks.values():
        if checks['eligible_combinations']<20:
            checks['failures'].append('split_eligible_combinations_lt20');checks['pass_gate']=False
    g0=all(u['G0']['pass_gate'] for u in units)
    g1=not duplicates and not near_fail; g2=all(u['checks']['pass_gate'] for u in units) and all(c['pass_gate'] for c in split_checks.values())
    rate=rejects/max(1,attempts)
    status='GEOMETRY_READY' if g0 and g1 and g2 and rate<=.5 else 'STOP_GEOMETRY_GATE_FAIL'
    result=dict(status=status,planned_units=12,completed_geometry_units=len(units),completed_sensor_frames=0,units=units,
        G0=dict(pass_gate=g0),G1=dict(pass_gate=g1,duplicates=duplicates,near_fail=near_fail),G2=dict(pass_gate=g2,split_checks=split_checks),rejection_rate=rate,attempts=attempts,rejections=rejects,
        G3_sim='NOT_RUN',G4='NOT_RUN',G5='NOT_RUN',B0_B1='NOT_RUN',wall_s=time.monotonic()-started)
    save(output/'result.json',result)
    print(status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)

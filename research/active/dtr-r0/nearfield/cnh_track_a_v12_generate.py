"""Frozen v1.2 geometry-only generator. CLI execution creates one bounded cohort.

Trajectories are generated and saved before object candidates. MILP has public
geometry only; exact triangle margins, not solver status, decide acceptance.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
from cnh_track_a_generate import (BOXES, PATTERNS, make_object, head_path, ry,
                                 check_unit, check_coordinates, save)
from cnh_track_a_geometry import box_mesh, signed_margin, clip_triangles
from cnh_track_a_validate import exact_voxels

FAMILY='cnh-track-a-v12-20260925'
COMMON=((0,0),(1,0),(5,0),(2,0),(0,1),(0,5),(0,4))
EVEN=((3,3),(1,2),(2,1),(4,5),(5,4),(3,6),(6,3))
ODD=((6,6),(1,4),(4,1),(2,5),(5,2),(3,2),(2,3))
PURPOSES=('layout','trajectory','size_placement','material','noise','ego_motion')
STATE_PATTERN=((0,0,0),(1,0,0),(1,1,0),(0,1,1),(0,0,1),(0,0,0))


def purpose_seed(unit,config,purpose,candidate=0):
    if purpose not in PURPOSES: raise ValueError('Unknown seed purpose')
    message=f'{FAMILY}|{unit}|{config}|{purpose}|{candidate}'
    return int(hashlib.sha256(message.encode()).hexdigest()[:16],16)


def schedule(unit):
    family=COMMON+(EVEN if unit%2==0 else ODD)
    return family+family+((4,0),(0,2),(3,0),(0,3))


def trajectory(unit,config,height):
    seed=purpose_seed(unit,config,'trajectory')
    rng=np.random.default_rng(seed)
    angles=head_path(rng);speed=float(rng.uniform(.8,1.4))
    poses=np.repeat(np.eye(4)[None],12,axis=0)
    poses[:,:3,:3]=np.array([ry(a[2]) for a in angles])
    poses[:,:3,3]=[[0,-height,(t-7)*.2*speed] for t in range(12)]
    return dict(seed=seed,speed=speed,angles=angles.tolist(),world_from_Q=poses.tolist())


def object_templates(split,patterns,rng,material_rng):
    """Objects centred at local origin; retain vertical group and target state."""
    result=[]
    for group,pattern in enumerate(patterns):
        states={0:[],1:[1],2:[2],3:[None],4:[3],5:[4],6:[1,4]}[pattern]
        for state in states:
            obj=make_object(split,[0,0,0],rng,'HEAD' if group==0 else 'BODY',wide=state is None)
            obj.update(id=len(result)+1,group=group,target_state=state,
                       rho=float(material_rng.uniform(.1,.9)))
            result.append(obj)
    return result


class Program:
    def __init__(self):self.cost=[];self.low=[];self.high=[];self.integer=[];self.rows=[];self.lower=[];self.upper=[]
    def var(self,lo=0,hi=1,cost=0,integer=True):
        i=len(self.cost);self.cost.append(cost);self.low.append(lo);self.high.append(hi);self.integer.append(int(integer));return i
    def constraint(self,terms,lo=-np.inf,hi=np.inf):
        self.rows.append(terms);self.lower.append(lo);self.upper.append(hi)
    def conditional(self,terms,lo,hi,gate):
        # lo <= expression <= hi when gate=1, M=50 otherwise.
        a=dict(terms);a[gate]=a.get(gate,0)+50;self.constraint(a,hi=hi+50)
        a=dict(terms);a[gate]=a.get(gate,0)-50;self.constraint(a,lo=lo-50)
    def solve(self):
        rr=[];cc=[];vv=[]
        for i,row in enumerate(self.rows):
            for j,value in row.items():rr.append(i);cc.append(j);vv.append(value)
        matrix=sparse.coo_matrix((vv,(rr,cc)),shape=(len(self.rows),len(self.cost))).tocsc()
        return milp(np.asarray(self.cost),integrality=np.asarray(self.integer),
                    bounds=Bounds(self.low,self.high),constraints=LinearConstraint(matrix,self.lower,self.upper),
                    options=dict(time_limit=2.,mip_rel_gap=0.,presolve=True))


def plan_centres(templates,path,desired_depth,multi):
    """One <=2-second MILP. Continuous world X/Z and immutable world Y.

    Returns centres and solver-selected frame flags, or explicit no-incumbent.
    No trajectory updates; exact surface checks are separate and authoritative.
    """
    poses=np.asarray(path['world_from_Q']);anchor=poses[7]
    p=Program(); frames=list(range(3,12))
    mains={t:p.var(cost=-10000) for t in frames}
    targets={t:p.var(cost=-100+abs(t-7)) for t in frames}
    for t in frames:p.constraint({targets[t]:1,mains[t]:-1},hi=0)
    p.constraint({targets[t]:1 for t in frames},lo=2)
    p.constraint({mains[t]:1 for t in frames},lo=6 if multi else 2)
    variables=[]; info=[]; matches={t:[] for t in frames}
    for oi,obj in enumerate(templates):
        x,z=p.var(-8,8,integer=False),p.var(-4,8,integer=False)
        dev=p.var(0,20,cost=1,integer=False);variables.append((x,z))
        y=(.08 if obj['group']==0 else .65)+anchor[1,3]
        offsets=np.asarray(obj['triangles']).reshape(-1,3)@anchor[:3,:3].T
        statevars={}
        for t in frames:
            rot=poses[t,:3,:3];origin=poses[t,:3,3]
            off=offsets@rot
            hx=float(np.abs(off[:,0]).max());hz=float(np.abs(off[:,2]).max())
            # Row-vector world points @ rotation -> current Q.
            exprx={x:rot[0,0],z:rot[2,0]};constantx=(y-origin[1])*rot[1,0]-origin[0]*rot[0,0]-origin[2]*rot[2,0]
            exprz={x:rot[0,2],z:rot[2,2]};constantz=(y-origin[1])*rot[1,2]-origin[0]*rot[0,2]-origin[2]*rot[2,2]
            p.conditional(exprz,.350001+hz-constantz,2.949999-hz-constantz,mains[t])
            if obj['wide']:
                p.conditional(exprx,-hx+.050001-constantx,hx-.050001-constantx,mains[t])
                # For a spanning object, its surface must reach both outer boxes.
                positive={q:mains[t] for q in range(3)}
            else:
                states=[p.var() for _ in range(6)];statevars[t]=states
                p.constraint({**{s:1 for s in states},mains[t]:-1},lo=0,hi=0)
                intervals=[(-12,-.650001-hx),(-.549999+hx,-.350001-hx),
                           (-.249999+hx,-.050001-hx),(.050001+hx,.249999-hx),
                           (.350001+hx,.549999-hx),(.650001+hx,12)]
                for gate,(lo,hi) in zip(states,intervals):p.conditional(exprx,lo-constantx,hi-constantx,gate)
                p.constraint({targets[t]:1,states[obj['target_state']]:-1},hi=0)
                positive={q:[states[s] for s in range(6) if STATE_PATTERN[s][q]] for q in range(3)}
            # A two-object, two-query matching certifies distinct contributors.
            local=[]
            for q in range(3):
                match=p.var();local.append(match);globalq=2*q+obj['group']
                matches[t].append((oi,globalq,match))
                terms={match:1}
                for gate in ([positive[q]] if obj['wide'] else positive[q]):terms[gate]=terms.get(gate,0)-1
                p.constraint(terms,hi=0)
            p.constraint({m:1 for m in local},hi=1)
        # Public anchor depth L1 deviation, not an outcome-based target.
        rot=anchor[:3,:3];origin=anchor[:3,3]
        const=(y-origin[1])*rot[1,2]-origin[0]*rot[0,2]-origin[2]*rot[2,2]
        p.constraint({x:rot[0,2],z:rot[2,2],dev:-1},hi=desired_depth-const)
        p.constraint({x:-rot[0,2],z:-rot[2,2],dev:-1},hi=-desired_depth+const)
        info.append(dict(y=y))
    if multi:
        for t in frames:
            for q in range(6):p.constraint({m:1 for _,qq,m in matches[t] if qq==q},hi=1)
            p.constraint({**{m:1 for _,_,m in matches[t]},mains[t]:-2},lo=0)
    result=p.solve()
    if result.x is None:return None,dict(status=int(result.status),message=result.message,incumbent=False)
    centres=np.array([[result.x[x],d['y'],result.x[z]] for (x,z),d in zip(variables,info)]).reshape(-1,3)
    return centres,dict(status=int(result.status),message=result.message,incumbent=True,
        planned_main=[t for t in frames if result.x[mains[t]]>.5],
        planned_target=[t for t in frames if result.x[targets[t]]>.5],objective=float(result.fun))


def boundary_centres(templates,path,ci,rng):
    """Construct chosen front-face margin using actual mesh max-Z offset."""
    anchor=np.asarray(path['world_from_Q'])[7]
    margin=float(rng.uniform(.005,.04))*(1 if ci in (28,29) else -1)
    centres=[]
    for obj in templates:
        state=obj['target_state']
        x=0. if obj['wide'] else {1:-.45,2:-.15,3:.15,4:.45}[state]
        x+=float(rng.uniform(-.025,.025))
        y=(.08 if obj['group']==0 else .65)+float(rng.uniform(-.015,.015))
        z=.3+margin-float(np.asarray(obj['triangles'])[...,2].max())
        centres.append(np.array([x,y,z])@anchor[:3,:3].T+anchor[:3,3])
    query=2*(1 if templates[0]['target_state'] in (2,3,None) else 0)+templates[0]['group']
    return np.asarray(centres),dict(target_margin=margin,object_id=templates[0]['id'],query=query)


def evaluate_candidate(unit,ci,split,height,width,path,templates,centres):
    poses=np.asarray(path['world_from_Q']);anchor=poses[7]
    objects=[]
    for obj,center in zip(templates,centres):
        world=np.asarray(obj['triangles'])@anchor[:3,:3].T+center
        objects.append({**{k:v for k,v in obj.items() if k!='triangles'},
                        'triangles_world':world,'center':center.tolist()})
    low=box_mesh([.7,height-.25,1.3],[.85,height,1.55])@anchor[:3,:3].T+anchor[:3,3]
    objects.append(dict(id=len(objects)+1,triangles_world=low,rho=.5,family='low-background',category='LOW'))
    backgrounds=[box_mesh([-width,0,-3],[width,.1,7])]
    if unit%4!=3:backgrounds.append(box_mesh([-width,-height-2,-3],[-width+.1,0,7]))
    if unit%4==0:backgrounds.append(box_mesh([width-.1,-height-2,-3],[width,0,7]))
    if unit%4 in (1,2):
        backgrounds.extend([box_mesh([width-.1,-height-2,-3],[width,0,3.5]),box_mesh([width-.1,-height-2,3.5],[width+3,0,3.6])])
    if unit%4==2:backgrounds.append(box_mesh([-width-3,-height-2,3.5],[-width,0,3.6]))
    objects.extend(dict(id=100+i,triangles_world=mesh,rho=.5,family='background',category='BACKGROUND') for i,mesh in enumerate(backgrounds))
    margins=[];labels=[];contributors=[];witness=[]
    for pose in poses:
        local=[(o['triangles_world']-pose[:3,3])@pose[:3,:3] for o in objects]
        row=np.array([[signed_margin(mesh,*box) for box in BOXES] for mesh in local])
        margins.append(row);labels.append((row.max(0)>=-1e-10).astype(int))
        contributors.append([[o['id'] for o,m in zip(objects,row[:,q]) if m>=-1e-10] for q in range(6)])
        zrow=[]
        for q,box in enumerate(BOXES):
            zs=[]
            for mesh,m in zip(local,row[:,q]):
                if m>=-1e-10:
                    clipped=clip_triangles(mesh,*box)
                    zs.append(float(clipped[...,2].min()) if len(clipped) else float(signed_margin(mesh,*box,return_witness=True)[1][2]))
            zrow.append(min(zs) if zs else None)
        witness.append(zrow)
    margins=np.asarray(margins);labels=np.asarray(labels)
    boundary=(np.abs(margins)<.05).any((1,2));main=(np.arange(12)>=3)&~boundary
    patterns=schedule(unit)[ci];wanted=np.array([PATTERNS[patterns[q%2]][q//2] for q in range(6)])
    if ci>=30:wanted[:]=0
    return dict(unit=unit,config=ci,split=split,height=height,seed=path['seed'],speed=path['speed'],
        angles=path['angles'],world_from_Q=path['world_from_Q'],target_labels=wanted.tolist(),
        labels=labels.tolist(),margins=margins.tolist(),object_ids=[o['id'] for o in objects],
        contributors=contributors,witness_z=witness,boundary=boundary.tolist(),main=main.tolist(),
        objects=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in o.items()} for o in objects])


def distinct_contributors(contributors):
    return any(a!=b for qa,ids in enumerate(contributors) for qb,others in enumerate(contributors)
               if qa!=qb for a in ids for b in others)


def candidate_check(config,ci,boundary_plan=None):
    labels=np.asarray(config['labels']);main=np.asarray(config['main']);wanted=np.asarray(config['target_labels'])
    if ci<28:
        target=main&(labels==wanted).all(1)
        if target.sum()<2:return False,'target_combo_lt2_main'
        if ci%14>=7:
            multi=main&np.array([distinct_contributors(c) for c in config['contributors']])
            if multi.sum()<6:return False,'multi_distinct_query_lt6_main'
        return True,'exact_geometry_pass'
    value=np.asarray(config['margins'])[7,boundary_plan['object_id']-1,boundary_plan['query']]
    if abs(value-boundary_plan['target_margin'])>1e-4:return False,'boundary_exact_margin_mismatch'
    return True,'exact_boundary_pass'


def fingerprint(config):
    pose=np.asarray(config['world_from_Q'])[7]
    triangles=[(np.asarray(o['triangles_world'])-pose[:3,3])@pose[:3,:3]
               for o in config['objects'] if o['category'] in ('HEAD','BODY')]
    return exact_voxels(np.concatenate(triangles)) if triangles else set()


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+4*3600;units=[];attempts=rejects=0
    sets=[];buckets={};split_configs={s:[] for s in ('train','calib','audit')}
    source_names=('cnh_track_a_v12_generate.py','cnh_track_a_generate.py','cnh_track_a_geometry.py','cnh_track_a_validate.py','CNH_NEARFIELD_ACCEPTANCE_PLAN_V1_2_20260925.md')
    save(output/'manifest.json',dict(schema='cnh.track-a.geometry.v1.2',seed_family=FAMILY,
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in source_names},
        limit_units=12,configs_per_unit=32,frames_per_config=12,limit_hours=4,limit_gib=20,
        split={'train':list(range(6)),'calib':[6,7],'audit':[8,9,10,11]}))
    def stop(status,**extra):
        result=dict(status=status,completed_geometry_units=len(units),units=units,attempts=attempts,rejections=rejects,
                    rejection_rate=rejects/max(1,attempts),downstream='NOT_RUN',completed_sensor_frames=0,
                    G3_sim='NOT_RUN',G4='NOT_RUN',G5='NOT_RUN',readouts='NOT_RUN',
                    wall_s=time.monotonic()-started,**extra)
        save(output/'result.json',result);return result
    try:
        for u in range(12):
            split='train' if u<6 else 'calib' if u<8 else 'audit'
            layout_seed=purpose_seed(u,-1,'layout');rng=np.random.default_rng(layout_seed)
            height=float(rng.uniform(1.45,1.75));width=float(rng.uniform(4,6))
            paths=[trajectory(u,ci,height) for ci in range(32)]
            save(output/f'trajectories-unit{u:02d}.json',dict(layout_seed=layout_seed,height=height,width=width,paths=paths))
            configs=[];logs=[];unitstart=time.monotonic()
            for ci in range(32):
                accepted=None
                for attempt in range(32):
                    if time.monotonic()>deadline:return stop('STOP_WALL_BUDGET',unit=u,config=ci)
                    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>20*1024**3:return stop('STOP_STORAGE_BUDGET',unit=u,config=ci)
                    attempts+=1;seeds={purpose:purpose_seed(u,-1 if purpose=='layout' else ci,purpose,attempt if purpose in ('size_placement','material') else 0) for purpose in PURPOSES}
                    rng=np.random.default_rng(seeds['size_placement']);mrng=np.random.default_rng(seeds['material'])
                    templates=object_templates(split,schedule(u)[ci],rng,mrng)
                    if ci<28:
                        centres,planning=plan_centres(templates,paths[ci],(.8,1.6,2.4)[ci%3],ci%14>=7)
                        boundary_plan=None
                    else:
                        centres,boundary_plan=boundary_centres(templates,paths[ci],ci,rng);planning=dict(boundary=boundary_plan)
                    reason='solver_no_incumbent'
                    if centres is not None:
                        c=evaluate_candidate(u,ci,split,height,width,paths[ci],templates,centres)
                        good,reason=candidate_check(c,ci,boundary_plan)
                        if good:
                            fp=fingerprint(c);key=hashlib.sha256(json.dumps(sorted(fp),separators=(',',':')).encode()).hexdigest()
                            if fp and any(fp==old for old in buckets.get(key,[])):good=False;reason='global_exact_voxel_duplicate'
                        if good:
                            c.update(seeds=seeds,candidate=attempt,planning=planning,fingerprint=sorted(fp),
                                     planned_multi=bool(ci<28 and ci%14>=7),boundary_plan=boundary_plan,
                                     planned_patterns=list(schedule(u)[ci]),
                                     desired_anchor_depth=(.8,1.6,2.4)[ci%3],
                                     actual_anchor_depths=[float(((np.asarray(center)-np.asarray(paths[ci]['world_from_Q'])[7,:3,3])@np.asarray(paths[ci]['world_from_Q'])[7,:3,:3])[2]) for center in centres])
                            accepted=c
                            if fp:sets.append((u,ci,fp));buckets.setdefault(key,[]).append(fp)
                    entry=dict(config=ci,candidate=attempt,seeds=seeds,accepted=accepted is not None,reason=reason,planning=planning)
                    logs.append(entry)
                    if centres is not None and accepted is None:
                        folder=output/'rejected';folder.mkdir(exist_ok=True)
                        save(folder/f'unit{u:02d}-config{ci:02d}-candidate{attempt:02d}.json',dict(candidate_config=c,**entry))
                    save(output/f'candidates-unit{u:02d}.json',logs)
                    if accepted is not None:break
                    rejects+=1
                if accepted is None:return stop('STOP_GENERATION_FAIL',unit=u,config=ci)
                configs.append(accepted)
                save(output/f'unit{u:02d}.json',dict(unit=u,split=split,height=height,width=width,configs=configs,status='PARTIAL',layout_seed=layout_seed))
            checks=check_unit(configs);g0=check_coordinates(configs)
            metadata=dict(unit=u,split=split,shape=('straight','L','T','open')[u%4],height=height,width=width,
                          configs=configs,checks=checks,G0=g0,wall_s=time.monotonic()-unitstart,layout_seed=layout_seed)
            save(output/f'unit{u:02d}.json',metadata);units.append({k:v for k,v in metadata.items() if k!='configs'})
            split_configs[split].extend(dict(c,config=u*32+c['config']) for c in configs)
            print(json.dumps(dict(unit=u,G2=checks['pass_gate'],failures=checks['failures'],wall_s=metadata['wall_s'])),flush=True)
        near={}
        for i,(ua,ca,fa) in enumerate(sets):
            for ub,cb,fb in sets[i+1:]:
                if ua==ub:continue
                j=len(fa&fb)/max(1,len(fa|fb))
                if j>=.98:near.setdefault((ua,ub),set()).add(ca);near.setdefault((ub,ua),set()).add(cb)
        near_fail=[dict(unit=a,other=b,count=len(cs)) for (a,b),cs in near.items() if len(cs)>=.9*sum(u==a for u,_,_ in sets)]
        split_checks={s:check_unit(cs) for s,cs in split_configs.items()}
        for check in split_checks.values():
            if check['eligible_combinations']<20:check['failures'].append('split_eligible_combinations_lt20');check['pass_gate']=False
        g0=all(u['G0']['pass_gate'] for u in units);g1=not near_fail
        g2=all(u['checks']['pass_gate'] for u in units) and all(c['pass_gate'] for c in split_checks.values())
        return stop('GEOMETRY_READY' if g0 and g1 and g2 and rejects/max(1,attempts)<=.5 else 'STOP_GEOMETRY_GATE_FAIL',
                    G0=dict(pass_gate=g0),G1=dict(pass_gate=g1,duplicates=[],near_fail=near_fail),G2=dict(pass_gate=g2,split_checks=split_checks))
    except BaseException as exc:
        stop('STOP_EXCEPTION',error=repr(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();print(run(args.output)['status'])

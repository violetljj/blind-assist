"""Evaluator-only first-visible dark-bar zone coverage. No rendering or sampling."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path('E:/linnan/linnan');HERE=ROOT/'research/active/dtr-r0/nearfield'
OUT=ROOT/'artifacts.local/work/cnh-bar-background-reference-dev-20261009/geometry'
SHAPES=ROOT/'artifacts.local/work/cnh-aligned-shapes-dev-20261008'
START=time.monotonic();PRIOR=0.;PREP_COMMAND_SECONDS=2.93
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def budget():
    if PREP_COMMAND_SECONDS+PRIOR+time.monotonic()-START>120:raise TimeoutError('Cumulative geometry CPU-wall120s includes prepare/failures/check')
def scalar_first(origin,direction,boxes):
    direction=direction/np.linalg.norm(direction);best=np.inf;chosen=-1
    for i,b in enumerate(boxes):
        near=-np.inf;far=np.inf;valid=True
        for axis in range(3):
            if abs(direction[axis])<1e-14:
                if origin[axis]<b['lo'][axis] or origin[axis]>b['hi'][axis]:valid=False;break
            else:
                a=(b['lo'][axis]-origin[axis])/direction[axis];z=(b['hi'][axis]-origin[axis])/direction[axis]
                near=max(near,min(a,z));far=min(far,max(a,z))
        t=near if near>1e-10 else far
        if valid and far>=max(near,0) and t>1e-10 and t<best:best=t;chosen=i
    return chosen
def run():
    global PRIOR
    OUT.mkdir(parents=True,exist_ok=True)
    PRIOR=sum(json.loads(p.read_text())['execution_seconds'] for p in OUT.glob('failure_*.json'))
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve existing geometry PLAN')
    sys.path.insert(0,str(HERE))
    import cnh_proposal_attribution_scenes as G
    cached=json.loads((SHAPES/'PLAN.json').read_text());rows=cached['scene_rows']
    # Rebuild unchanged finite scene grid through pure source functions, no renderer imports.
    tree=ast.parse((HERE/'cnh_aligned_shapes_dev.py').read_text());env={}
    constant=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='BASE_BG' for v in n.targets))
    exec(compile(ast.Module(body=[constant],type_ignores=[]),'pure_background_constants','exec'),env)
    functions=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ['layouts','scene_rows']]
    exec(compile(ast.Module(body=functions,type_ignores=[]),'pure_scene_layout','exec'),env)
    assert env['scene_rows']()==rows
    ids=np.array([i for i,r in enumerate(rows) if r['family']=='horizontal' and r['rho']==.25 and r['variant'].endswith('thick0.04')],dtype=np.int64)
    assert len(ids)==44 and all(abs(rows[i]['hi'][1]-rows[i]['lo'][1]-.04)<1e-10 for i in ids)
    assert all(rows[i]['id']==i for i in ids)
    with np.load(SHAPES/'geometry.npz') as d:sensor=d['sensor'];query=d['public_query'];category=d['category'][ids]
    a=np.deg2rad(-10.);c,s=np.cos(a),np.sin(a);R=np.array([[1,0,0],[0,c,-s],[0,s,c]])
    expected=np.repeat(np.eye(4)[None],16,0);expected[:,:3,:3]=R;expected[:,2,3]=np.arange(16)*.16-2.4
    np.testing.assert_array_equal(sensor,expected)
    q=np.repeat(np.eye(4)[None],16,0);q[:,:3,:3]=R;np.testing.assert_array_equal(query,q)
    directions,weights=G.angular_rays(16)
    assert directions.shape==(8,8,256,3) and weights.shape==(8,8,256)
    np.testing.assert_allclose(np.linalg.norm(directions,axis=-1),1,rtol=0,atol=4e-16)
    source_paths=[Path(__file__),HERE/'cnh_aligned_shapes_dev.py',HERE/'cnh_aligned_boundary_dev.py',Path(G.__file__),Path(sys.modules[G.angular_rays.__module__].__file__)]
    inputs=[SHAPES/'PLAN.json',SHAPES/'geometry.npz',SHAPES/'evaluated.npz']
    source_hash={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in source_paths}
    save('PLAN.json',dict(task='Dark4cmbar evaluator-only first-hit zone coverage',scope='All44darkhorizontal0.25rho4cm scenes/16trueposes/8x8zones/256midpointsubrays, target first then every row background. No K4 duplication: geometry identical across replicas.',
        scene_ids=ids.tolist(),frames=list(range(16)),budget_CPU_wall_seconds=120,
        preliminary_inspection_command_seconds=PREP_COMMAND_SECONDS,preparation_seconds=time.monotonic()-START,
        source_sha256=source_hash,input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in inputs},
        normalization='Unweighted first_hit_count/256 in original zone y,x ordering; not solid-angle-weighted energy or visibility probability.',
        first_hit='Original CPU analytic slab raycast object_id==0. Closed AABB face intersection, t>1e-10, strict nearest-depth tie keeps earlier box, target ordered first; all backgrounds can occlude.',
        authority='Evaluator truth only for pollution and timely-event coverage evaluation. Not allowed into background reference calibration, availability threshold, zone selection or online score.',
        provenance='New geometric recomputation; physical cache contains no per-zone coverage. Sum compared with earlier evaluated total first_target_rays.',
        new_render=0,new_photons=0,new_inference=0,training=0,stop='Complete or cumulative120s; retain failures and do not alter scenes/subray grid/poses.'))
    with (OUT/'source_snapshot.py').open('xb') as f:f.write(Path(__file__).read_bytes())
    budget();counts=np.zeros((44,16,8,8),np.uint16);scalar_checks=0
    for j,i in enumerate(ids):
        row=rows[int(i)];boxes=[row,*row['background']]
        for frame in range(16):
            budget();world=directions@sensor[frame,:3,:3].T
            hits=G.raycast_boxes(sensor[frame,:3,3],world,boxes)['object_id']
            counts[j,frame]=np.count_nonzero(hits==0,axis=-1)
            if j in [0,22,43] and frame in [0,13,15]:
                flat=hits.ravel();rayix=[0,8192,16383]
                if (flat==0).any():rayix.append(int(np.flatnonzero(flat==0)[0]))
                for n in rayix:
                    assert scalar_first(sensor[frame,:3,3],world.reshape(-1,3)[n],boxes)==flat[n]
                    scalar_checks+=1
        if j%11==0:print('GEOMETRY',j+1,'/44',round(time.monotonic()-START,3),flush=True)
    total=counts.astype(np.int32).sum((2,3));fraction=counts.astype(np.float64)/256
    with np.load(SHAPES/'evaluated.npz') as d:
        has_total='first_target_rays' in d.files
        if has_total:np.testing.assert_array_equal(total,d['first_target_rays'][ids])
        np.testing.assert_array_equal(category,d['category'][ids])
    assert counts.max()<=256 and np.isfinite(fraction).all() and (fraction>=0).all() and (fraction<=1).all()
    np.testing.assert_array_equal(fraction*256,counts)
    budget();np.savez_compressed(OUT/'coverage.npz',scene_ids=ids,frames=np.arange(16),coverage=fraction,
        first_hit_count=counts,total_first_target_rays=total,sensor=sensor,public_query=query,category=category,
        directions_sensor=directions,subray_solid_angle_weights=weights,subrays_per_zone=np.array(256))
    execution=time.monotonic()-START;budget()
    save('result.json',dict(status='COMPLETE',execution_seconds=execution,cumulative_seconds=PREP_COMMAND_SECONDS+PRIOR+execution,
        budget_CPU_wall_seconds=120,pid=os.getpid(),scenes=44,frames=16,zones=64,subrays_per_zone=256,total_raycast_directions=44*16*64*256,
        source_scene_rows_and_backgrounds_parity='EXACT original immutable PLAN and current pure scene generator',poses_parity='BITWISE original cached sensor/public query and fixed -10deg schedule',
        evaluated_total_first_target_rays_parity='BITWISE all44x16' if has_total else 'NOT_AVAILABLE',scalar_slab_crosschecks=scalar_checks,
        fractions='first_hit_count/256, unweighted; original y,x/suby,subx midpoint order',
        coverage_shape=list(fraction.shape),payload_sha256=sha(OUT/'coverage.npz'),source_sha256=sha(Path(__file__)),
        scenes_visible_timely=int((total[:,:14]>0).any(1).sum()),nonzero_scene_frame_zones=int(np.count_nonzero(counts)),
        category_scene_counts={str(x):int(sum('/'.join(r)==x for r in category)) for x in sorted(set('/'.join(r) for r in category))},
        evaluator_only=True,new_expected_render=0,new_photons=0,new_inference=0,training=0,
        limitation='Finite AABB midpoint quadrature at ideal poses, not physical hardware coverage. All zones retained; no target-conditioned algorithm zone selection. New geometry only, no new measured observations.'))
    print('COMPLETE',round(PREP_COMMAND_SECONDS+PRIOR+execution,3),'PID',os.getpid())
if __name__=='__main__':
    try:run()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True)
        save(f'failure_{time.time_ns()}.json',dict(status='FAILED',execution_seconds=time.monotonic()-START,error=repr(e)))
        raise

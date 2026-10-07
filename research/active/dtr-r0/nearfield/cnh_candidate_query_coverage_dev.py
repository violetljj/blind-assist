"""CPU sparse coverage-interface diagnostic on OLD torso-proxy observations.

Not head-yaw coverage, full-volume coverage, deployed CLEAR, or a three-state run.
Candidate complete query defines its target frame; no scores/boxes/labels are read.
"""
import argparse
import hashlib
import json
import time
import unittest
from pathlib import Path

import numpy as np
import cnh_tristate_dev as R

ROOT = R.ROOT
BASE = ROOT/'artifacts.local/work/cnh-torso-bias-dev-20261007/native_motion'
OUT = BASE/'head_yaw/coverage'
SELECTION = ((99000,'pulse_pos'),(99001,'const_neg'))
FRAMES = (5,10,15)
ARMS = ('e1','torso','gait')
RANGE = float(np.sqrt(2.1**2+.29**2+.9**2))
MARGIN = 3.
BUDGET = 120.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def local_points():
    """150 pre-fixed boundary/midpoint samples; shared height seam kept per query."""
    points=[]; heights=[]
    for group,(lo,hi) in enumerate(R.HEIGHTS):
        for x in (-.29,-.145,0.,.145,.29):
            for y in (lo,(lo+hi)/2.,hi):
                for z in (.9,1.2,1.5,1.8,2.1):
                    points.append((x,y,z));heights.append(group)
    return np.asarray(points),np.asarray(heights,np.int8)


def candidate_frame(sensor,query):
    """sensor_from_world inverse: candidate local -> world via S @ inverse(Q)."""
    return np.asarray(sensor,float)@np.linalg.inv(np.asarray(query,float))


def world_points(points,frame):
    return np.asarray(points)@frame[:3,:3].T+frame[:3,3]


def local_from_world(points,pose):
    return (np.asarray(points)-pose[:3,3])@pose[:3,:3]


def fov_local(local,range_limit=RANGE,margin_deg=0.):
    p=np.asarray(local);z=p[...,2];edge=np.tan(np.radians(22.5-margin_deg))
    return ((z>0)&(np.abs(p[...,0])<=edge*z+R.EPS)&
            (np.abs(p[...,1])<=edge*z+R.EPS)&
            (np.linalg.norm(p,axis=-1)<=range_limit+R.EPS))


def nominal_from_candidate(sensor,candidate):
    """R3 adaptation: candidate yaw replaces pose-displacement heading.

    Hypothetical aligned sensor uses the historical sensor origin and pitch -10.
    Candidate target points retain their own complete pelvis-origin transform.
    This is a nominal core scope, not an observed-space certificate.
    """
    yaw=np.degrees(np.arctan2(candidate[0,2],candidate[2,2]))
    pose=np.eye(4);pose[:3,:3]=R.rotation(yaw)@R.rotation(-10,'x')
    pose[:3,3]=sensor[:3,3]
    return pose


def coverage_probe(sensor,queries,frame,history_sensor=None,points=None):
    """Return sparse/core diagnostic masks, NEVER authorization to output CLEAR."""
    sensor=np.asarray(sensor);queries=np.asarray(queries)
    observed=sensor if history_sensor is None else np.asarray(history_sensor)
    local,groups=local_points() if points is None else (np.asarray(points),None)
    estimated=candidate_frame(sensor,queries)
    world=world_points(local,estimated[frame])
    core=np.zeros(len(local),bool);fresh=np.zeros((2,len(local)),bool)
    for h in range(max(0,frame-3),frame+1):
        nominal=nominal_from_candidate(sensor[h],estimated[h])
        core|=fov_local(local_from_world(world,nominal),margin_deg=MARGIN)
        for arm,angles in enumerate(((0.,),(-15.,15.))):
            for angle in angles:
                pose=observed[h]@R.extrinsic(angle)
                fresh[arm]|=fov_local(local_from_world(world,pose))
    missing=core[None]&~fresh
    return dict(world=world,core=core,fresh=fresh,missing=missing,height_group=groups,
                sparse_core_pass=(core.any()&~missing.any(1)),
                full_query_coverage_verified=False)


def partition_state(alarm,low_score,full_query_covered,tracking_valid):
    """Dummy interface contract; not called by existing production/replay pipeline."""
    if alarm:return 'OBSTACLE'
    if low_score and full_query_covered and tracking_valid is True:return 'CLEAR'
    return 'UNKNOWN'


def boundary_countercheck():
    epsilon=1e-6;edge=np.tan(np.pi/8)
    points=np.array([[edge-epsilon,0,1],[edge+epsilon,0,1],
                     [0,0,RANGE-epsilon],[0,0,RANGE+epsilon]])
    return dict(epsilon_m=epsilon,points_sensor=points.tolist(),
                expected=[True,False,True,False],actual=fov_local(points).tolist())


def run():
    tick=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    result=dict(status='RUNNING',records=[],budget_wall_seconds=BUDGET)
    def check():
        if time.monotonic()-tick>=BUDGET:raise TimeoutError('Independent coverage stage120s reached')
    try:
        sources=[Path(__file__),Path(__file__).with_name('test_cnh_candidate_query_coverage_dev.py'),
                 Path(R.__file__),Path(__file__).with_name('cnh_tristate_dev_r2.py'),
                 Path(__file__).with_name('cnh_tristate_dev_r3_geometry.py')]
        for unit,_ in SELECTION:
            sources.extend((BASE/'units'/f'unit{unit}.npz',BASE/'input_sensitivity/units'/f'unit{unit}.npz'))
        plan=dict(task='CANDIDATE_QUERY_SPARSE_COVERAGE_INTERFACE_DEV_20261007',
            selection=[dict(unit=u,perturbation=n,configs=list(range(40))) for u,n in SELECTION],
            frames=list(FRAMES),arms=ARMS,observation_variants=['stored_sensor','stored_noisy_sensor'],
            budget_wall_seconds=BUDGET,stage_includes='Formal diagnostic, tests, input hashing and result saving',
            target='150 fixed local boundary/midpoint samples of R3 .9..2.1m, x+/-.29m, both height bands; not whole volume',
            transform='Candidate E=S_current@inv(query_current); point local->world->historical sensor; full SE3 translation retained',
            adapted_r3='Historical nominal heading from each candidate E, not past sensor displacement. Same sensor origin/pitch-10, margin3deg, inherited fixed radial sqrt(2.1²+.29²+.9²), history current and3past at5Hz.',
            prohibited='No candidate score read, score/risk/UNKNOWN curve, boxes, geometry labels, GPU or new render. No calibration or safety conclusion.',
            alarms='Dummy helper/tests preserve OBSTACLE priority; sparse pass does not satisfy full_query_covered. Existing pipeline does not call helper.',
            limits='OLD input_sensitivity/native torso-aligned proxy observations only; head_yaw path is storage routing, not new head-yaw evidence. Masks may legitimately agree. No independent population, unoccluded photons, full volume or real tracking-validity evidence.',
            input_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
        save(OUT/'PLAN.json',plan);check()
        boundary=boundary_countercheck();assert boundary['actual']==boundary['expected']
        result['boundary_countercheck']=boundary
        suite=unittest.defaultTestLoader.loadTestsFromName('test_cnh_candidate_query_coverage_dev')
        tests=unittest.TextTestRunner(verbosity=1).run(suite)
        result['tests']=dict(run=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),skipped=len(tests.skipped))
        if not tests.wasSuccessful():raise AssertionError('Coverage interface tests failed')
        check()
        for unit,name in SELECTION:
            with np.load(BASE/'units'/f'unit{unit}.npz',allow_pickle=False) as stream:
                old={k:stream[k] for k in ('sensor','noisy','e1_query','source_pid','source_clip','native_frame_index')}
            with np.load(BASE/'input_sensitivity/units'/f'unit{unit}.npz',allow_pickle=False) as stream:
                candidate={arm:stream[name+'/'+arm+'_query'] for arm in ('torso','gait')}
            candidate['e1']=old['e1_query']
            assert old['sensor'].shape==(40,16,4,4)
            for config in range(40):
                check()
                assert str(old['source_pid'][config]) in ('P06','P07','P08','P09','P10')
                for frame in FRAMES:
                    for observation,key in (('stored_sensor','sensor'),('stored_noisy_sensor','noisy')):
                        probes={arm:coverage_probe(old['sensor'][config],candidate[arm][config],frame,
                                                  old[key][config]) for arm in ARMS}
                        for arm,probe in probes.items():
                            for height in (0,1):
                                selection=probe['height_group']==height
                                for branch,label in enumerate(('single','dual')):
                                    result['records'].append(dict(unit=unit,perturbation=name,config=config,
                                        native_frame=int(old['native_frame_index'][config,frame]),sampled_frame=frame,
                                        observation=observation,arm=arm,branch=label,height=height,
                                        points=int(selection.sum()),core_points=int((probe['core']&selection).sum()),
                                        fresh_points=int((probe['fresh'][branch]&selection).sum()),
                                        missing_core_points=int((probe['missing'][branch]&selection).sum()),
                                        sparse_core_pass=bool((probe['core']&selection).any() and
                                                             not(probe['missing'][branch]&selection).any())))
                        result.setdefault('mask_comparisons',[]).append(dict(unit=unit,config=config,frame=frame,
                            observation=observation,torso_gait_query_equal=bool(np.allclose(candidate['torso'][config,frame],candidate['gait'][config,frame],rtol=0,atol=1e-12)),
                            torso_gait_core_equal=bool(np.array_equal(probes['torso']['core'],probes['gait']['core'])),
                            torso_gait_fresh_equal=bool(np.array_equal(probes['torso']['fresh'],probes['gait']['fresh']))))
        check();summaries={}
        for record in result['records']:
            key='/'.join(map(str,(record['observation'],record['arm'],record['branch'],record['height'])))
            cell=summaries.setdefault(key,dict(frame_probes=0,points=0,core_points=0,fresh_points=0,missing_core_points=0,sparse_core_pass_frames=0))
            cell['frame_probes']+=1
            for field in ('points','core_points','fresh_points','missing_core_points'):cell[field]+=record[field]
            cell['sparse_core_pass_frames']+=int(record['sparse_core_pass'])
        result.update(status='PASS',units=2,configs=80,unique_config_frames=240,summaries=summaries,
                      limits=plan['limits'],existing_pipeline_uses_partition_state=False,
                      full_query_coverage_verified=False,plan_sha256=sha(OUT/'PLAN.json'))
    except BaseException as exc:
        result['status']='FAILED';result['error']=repr(exc);raise
    finally:
        result['seconds']=time.monotonic()-tick
        save(OUT/'results.json',result)
        print(json.dumps({k:result[k] for k in ('status','seconds','tests') if k in result}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('run',));parser.parse_args();run()

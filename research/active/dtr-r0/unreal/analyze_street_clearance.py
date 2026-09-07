"""Reconstruct depth decisions and compare actual Development clearance runs.

Replay consumes model-only depth/poses; evaluator outcomes are read separately
by summarize. Replayed commands are never scored as new motion trajectories.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from street_live_policy import MotionPolicy, depth_corridors


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(root):
    rows, identities = [], {}
    for path in sorted((root/'model').glob('episode_*/episode.json')):
        episode = read(path)
        policies = [MotionPolicy('CANDIDATE_DEPTH'), MotionPolicy('CANDIDATE_CLEARANCE')]
        route = episode['route_frame']
        steps = []
        identities[str(path)] = sha(path)
        for frame in episode['frames']:
            pose, camera = frame['wearer_transform'], frame['camera_transform']
            delta = [pose['x']-route['center_xy_m'][0],pose['y']-route['center_xy_m'][1]]
            x = sum(a*b for a,b in zip(delta,route['forward_xy']))
            y = sum(a*b for a,b in zip(delta,route['right_xy']))
            depth_path = root/'model'/frame['depth_path']
            identities[str(depth_path)] = sha(depth_path)
            depth = np.load(depth_path,allow_pickle=False)
            commands = []
            for policy in policies:
                corridor = depth_corridors(depth,100.0,camera['z']-pose['z'],y,camera['pitch'],
                                           buffered=policy.mode=='CANDIDATE_CLEARANCE')
                command = policy.command(t=frame['time_s'],x=x,y=y,goal_x=8,
                                         dtr_risk=False,corridors=corridor)
                commands.append({k:command[k] for k in ('action','vx_mps','vy_mps','target_y_m')})
            steps.append({'sample_index':frame['sample_index'],'time_s':frame['time_s'],
                          'baseline':commands[0],'buffered':commands[1],
                          'changed':commands[0]!=commands[1]})
        rows.append({'episode_id':episode['episode_id'],'frames':len(steps),
                     'changed_decision_frames':sum(s['changed'] for s in steps),'steps':steps})
    # Validate the reconstruction against historical depth commands after all
    # model-only calls. Logged responses never become policy inputs.
    mismatches = []
    for row in rows:
        path = root/'evaluator/episodes'/(row['episode_id']+'.json')
        identities[str(path)] = sha(path)
        logged = read(path)['frames']
        for step, frame in zip(row['steps'],logged,strict=True):
            expected = frame['response']['command']
            if any(step['baseline'][k]!=expected[k] for k in step['baseline']):
                mismatches.append([row['episode_id'],step['sample_index']])
    if mismatches:
        raise ValueError(f'Baseline reconstruction differs: {mismatches[:8]}')
    return {'schema':'ue-clearance-fixed-pose-replay-v1','baseline_action_mismatches':mismatches,
            'evidence':'Same recorded depth and ego poses; changed commands only, no counterfactual motion score',
            'backend':'CPU / TASK_NOT_GPU_SUITABLE scalar geometry and receipt analysis',
            'frames':sum(r['frames'] for r in rows),'rows':rows,'input_sha256':identities}


def summarize(baseline, challenger):
    left, right = read(baseline/'evaluation.json'),read(challenger/'evaluation.json')
    a,b = read(baseline/'identity.json'),read(challenger/'identity.json')
    for key in ('map_sha256','render_config_sha256','scenario_selection','prediction_engine','action_footprint_state'):
        if a.get(key)!=b.get(key):
            raise ValueError('Comparison input mismatch: '+key)
    if read(baseline/'sensor-worker/backend.json')['model_sha256'] != read(challenger/'sensor-worker/backend.json')['model_sha256']:
        raise ValueError('Detector identity changed')
    for report in (left,right):
        if report['status']!='COMPLETE' or not report['all_open_loop_contrasts_pass']:
            raise ValueError('Comparison requires complete valid controls')
    if not read(challenger/'process-release.json')['released']:
        raise ValueError('Owned resources remain')
    rows=[]
    keys=('success','contact','first_contact_s','goal_reached','duration_s','applied_stationary_s',
          'path_length_m','measured_worker_latency_s')
    for pair in left['pairs']:
        other=next(p for p in right['pairs'] if p['scenario_id']==pair['scenario_id'])
        rows.append({'scenario_id':pair['scenario_id'],
                     'baseline':{k:pair['ASSISTED'][k] for k in keys},
                     'challenger':{k:other['ASSISTED'][k] for k in keys}})
    regressions=[r['scenario_id'] for r in rows if r['baseline']['success'] and not r['challenger']['success']]
    improvements=[r['scenario_id'] for r in rows if not r['baseline']['success'] and r['challenger']['success']]
    supported=unknown=frames=0
    for path in (challenger/'evaluator/episodes').glob('*.json'):
        ep=read(path)
        if ep['arm']!='ASSISTED':continue
        for frame in ep['frames']:
            command=frame['response']['command']
            frames+=1
            supported+=command['candidate_evaluation']['admitted_tracks']>0
            unknown+=command['candidate_evaluation']['admitted_tracks']==0
    return {'schema':'ue-clearance-development-comparison-v1',
            'evidence':'Historical rerendered Development comparison; not exact-pixel pairing or confirmation',
            'baseline':str(baseline),'challenger':str(challenger),
            'input_identity_checks':'Same bank, map, render configuration, model, cadence policy and predictor mode',
            'successes':[left['assisted_successes'],right['assisted_successes']],
            'denominators':[left['assisted_complete_denominator'],right['assisted_complete_denominator']],
            'improvements':improvements,'regressions':regressions,
            'assisted_frames':frames,'supported_footprint_frames':supported,'unknown_footprint_frames':unknown,
            'rows':rows,'default_promotion':False,
            'decision':'REJECT_FIXED_VARIANT_REGRESSION' if regressions else
                       'RETAIN_DEVELOPMENT_CHALLENGER' if 'challenge_late_stop_stop' in improvements else
                       'REJECT_FIXED_VARIANT_NO_TARGET_GAIN',
            'input_sha256':{str(root/name):sha(root/name) for root in (baseline,challenger)
                            for name in ('evaluation.json','identity.json','sensor-worker/backend.json','process-release.json')}}


def blocker(root):
    """Evaluator-only attribution of a recorded blocker to static scene geometry."""
    rows=[]
    identities={}
    definition=Path(__file__).with_name('build_street_lab.py')
    identities[str(definition)]=sha(definition)
    for eid in ('episode_0005','episode_0007'):
        ep_path=root/'model'/eid/'episode.json'
        depth_path=root/'model'/eid/'0050.npy'
        identities.update({str(p):sha(p) for p in (ep_path,depth_path)})
        frame=read(ep_path)['frames'][50]
        pose,camera=frame['wearer_transform'],frame['camera_transform']
        depth=np.load(depth_path,allow_pickle=False)
        h,w=depth.shape
        v,u=np.mgrid[2:h:4,2:w:4]
        d=depth[v,u]
        focal=w/(2*math.tan(math.radians(100)/2))
        right=(u-(w-1)/2)*d/focal
        up=-(v-(h-1)/2)*d/focal
        pitch=math.radians(camera['pitch'])
        forward=math.cos(pitch)*d-math.sin(pitch)*up
        height=camera['z']-pose['z']+math.sin(pitch)*d+math.cos(pitch)*up
        mask=(forward>.08)&(forward<12)&np.isfinite(forward)&(height>.065)&(height<1.85)&(abs(right)<.70)
        near=mask&(forward<1.4)
        surface_x=pose['x']+forward[near]
        rows.append({'episode_id':eid,'time_s':frame['time_s'],'query_samples':int(mask.sum()),
                     'front_q04_m':float(np.quantile(forward[mask],.04)),
                     'near_samples':int(near.sum()),
                     'near_world_x_m':[float(surface_x.min()),float(surface_x.max())],
                     'near_world_y_m':[float((pose['y']+right[near]).min()),float((pose['y']+right[near]).max())]})
        if abs(surface_x.min()-31.075)>.002:
            raise ValueError('Observed blocking plane differs from static planter definition')
    return {'schema':'ue-static-blocker-attribution-v1','rows':rows,'input_sha256':identities,
            'static_definition':'build_street_lab.py: tree planter center (32,-2.1,.28) m, size (1.85,1.85,.60) m',
            'planter_world_bounds_m':{'x':[31.075,32.925],'y':[-3.025,-1.175],'z':[-.02,.58]},
            'target_y_m':-1.2,'target_intersects_planter':True,
            'claim':'Evaluator-only analytic/static-depth attribution; not a new native collision measurement',
            'coverage_gap':'Late-stop scenario evaluator declares pedestrian only; static planter is outside its contact denominator'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('replay','summarize','blocker'))
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--challenger',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    artifacts=Path(__file__).resolve().parents[4]/'artifacts.local'
    if not args.output.resolve().is_relative_to(artifacts.resolve()):
        parser.error('Output must remain under artifacts.local')
    if args.command=='summarize' and args.challenger is None:
        parser.error('Summarize requires --challenger')
    result=(replay(args.baseline) if args.command=='replay' else blocker(args.baseline)
            if args.command=='blocker' else summarize(args.baseline,args.challenger))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','input_sha256')},indent=2))


if __name__=='__main__':main()

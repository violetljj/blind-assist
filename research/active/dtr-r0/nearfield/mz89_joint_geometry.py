"""One frozen analytic joint temporal geometry canary; no training or tuning."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import mz88_uncertainty_aware_association as old
from mz85_rotation_compensated_state import rotate_ray

TASK = 'mz89-joint-geometry-20260912'
SEED = 89012
DT = old.DT_S
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from research_backend import BackendCandidate, DeviceObservation, select_backend


def build_source():
    """New draws from a declared analytic family design; not new sensor realism."""
    rng = np.random.default_rng(SEED)
    source = []
    for family in old.FAMILIES:
        for variant in range(6):
            t = np.arange(30) * DT
            phase, freq, amp = rng.uniform(0, 6), rng.uniform(1.3, 2.5), rng.uniform(15, 25)
            yaw = amp * np.sin(freq*t + phase) + 0.15*amp*np.sin(0.6*freq*t + 0.4)
            pitch = 0.25*amp*np.sin(0.8*freq*t + phase + 0.6)
            yaw -= yaw[0]
            pitch -= pitch[0]
            sign = 1 if variant % 2 == 0 else -1
            inside, outside = rng.uniform(0.6, 3.2), rng.uniform(0.6, 4.2)
            initial, speed = rng.uniform(4.05, 4.55), rng.uniform(0.85, 1.12)
            z, lateral_speed = rng.uniform(2.45, 2.95), rng.uniform(1.45, 2.25)
            x0 = rng.uniform(2.55, 3.05)
            gap_start = int(rng.integers(9, 19))
            profile = old.stress_profile(variant)
            if profile['dropout']:
                start = int(rng.integers(7, 22))
                profile['dropout'] = tuple(range(start, start + len(profile['dropout'])))
            frames = []
            for j, seconds in enumerate(t):
                returns, radar = [], []
                known, truth = True, False
                height = 'BODY' if variant % 2 == 0 else 'HEAD'
                r, angle, vr, rcs, lateral = initial-speed*seconds, sign*inside, -speed, 0.9, False
                if family == 'boundary_inside':
                    angle = sign*(12-inside)
                    truth = r < 3.18
                elif family == 'boundary_outside':
                    angle, r, vr = sign*(12+outside), 2.85-0.1*seconds, -0.55
                elif family == 'lateral_crossing':
                    x = sign*(x0-lateral_speed*seconds)
                    future_x = x-sign*lateral_speed
                    r, angle = math.hypot(x,z), math.degrees(math.atan2(x,z))
                    future = math.degrees(math.atan2(future_x,z))
                    vr = (r-math.hypot(x+sign*lateral_speed*DT,z))/DT
                    lateral = True
                    truth = r < 3.6 and old.segment_intersects_corridor(angle,future)
                elif family == 'head_motion':
                    # Sparse outside-corridor returns plus independently moving central clutter.
                    angle, r = sign*(17+outside), 2.65
                    if (j+variant) % 7 in (1,2,3):
                        returns.append({'range':r,'theta':angle-yaw[j],'height':height})
                    if (j+variant) % 7 in (0,1,2,3,4):
                        radar.append({'range':2.55+0.04*math.sin(j),'theta':3*math.sin(j*0.4),
                                      'vr':-0.6 if j%7<4 else 0.3,'rcs':0.95})
                elif family == 'multi_target_gap':
                    truth = r < 3.18
                    angle = sign*(3+0.4*variant)
                    known = not gap_start <= j < gap_start+2+variant%3
                    returns.append({'range':r+0.26,'theta':-sign*5-yaw[j],'height':'BODY'})
                    radar.append({'range':r+0.26,'theta':-sign*5,'vr':-speed,'rcs':0.7})
                elif family == 'weak_radar':
                    angle, rcs = sign*1.5, 0.035
                    truth = r < 3.18
                if family != 'head_motion':
                    returns.insert(0, {'range':r,'theta':angle-yaw[j],'height':height,'lateral':lateral})
                    radar.insert(0, {'range':r,'theta':angle,'vr':vr,'rcs':rcs})
                frames.append({'index':j,'truth':bool(truth),'truth_height':height if truth else None,
                               'tof_known':known,'tof_objects':returns,'radar_objects':radar})
            source.append({'episode_id':f'e{len(source):03d}', 'family':family,'variant':variant,
                           'stress':profile['name'],'profile':profile,'frames':frames,
                           'true_yaw_deg':yaw.tolist(),'true_pitch_deg':pitch.tolist()})
    return source


def materialize(source):
    flat = old.flatten_source(source)
    obs = {k:flat[k] for k in ('episode_id','tof_known','tof_range_m','tof_theta_deg',
                              'tof_lateral','tof_height')}
    radar = old.radar_materializer(source,np.random.default_rng(SEED+1))
    for name,value in zip(('radar_range_m','radar_velocity','radar_angle','radar_valid'),radar):
        obs[name] = value
    delta_yaw, delta_pitch, validity = [], [], []
    for n, ep in enumerate(source):
        p = ep['profile']
        rng = np.random.default_rng(SEED+100+n)
        axes = []
        for axis,scale in (('yaw',1.0),('pitch',0.3)):
            pose = old.sample_with_offset(np.asarray(ep[f'true_{axis}_deg']),p['sync_ms'])
            delta = np.diff(np.r_[0.,pose]) + scale*p['bias_dps']*DT
            delta += rng.normal(0,scale*p['noise_dps']*DT,30)
            delta[0] += scale*p['extrinsic_deg']
            delta[list(p['dropout'])] = 0
            axes.append(delta)
        valid = np.ones(30,bool)
        valid[list(p['dropout'])] = False
        delta_yaw.extend(axes[0]); delta_pitch.extend(axes[1]); validity.extend(valid)
    obs.update(delta_yaw=np.asarray(delta_yaw),delta_pitch=np.asarray(delta_pitch),
               imu_valid=np.asarray(validity))
    evaluator = {k:flat[k] for k in ('episode_id','family','variant','truth','truth_height')}
    return obs,evaluator


def pose_loadings(delta, valid):
    """Rows share nuisance latents; missed increments stay uncertain after recovery."""
    n = len(delta)
    matrix = np.zeros((n,3+2*n))
    accumulated = np.zeros(3+2*n)
    last_rate = 0.
    for j in range(n):
        if valid[j]:
            last_rate = delta[j]/DT
            accumulated[3+j] = 2*DT
        else:
            accumulated[3+n+j] = DT*(abs(last_rate)+2)
        matrix[j] = accumulated
        matrix[j,0] = 2.0
        matrix[j,1] = 2.0*(j+1)*DT
        matrix[j,2] = 0.02*last_rate
    return matrix


def forecast_sigma(previous, current, elapsed, joint=True):
    a = 1/elapsed
    if joint:
        return float(np.linalg.norm((1+a)*current-a*previous))
    return float(math.sqrt((1+a)**2*np.dot(current,current)+a*a*np.dot(previous,previous)))


def matches(previous, current):
    """Mutually unique nearest observations; tuple=(range,angle,height,lateral)."""
    if not previous or not current:
        return {}
    cost = np.full((len(previous),len(current)),np.inf)
    for i,p in enumerate(previous):
        for j,c in enumerate(current):
            dr,da = abs(p[0]-c[0]),abs(p[1]-c[1])
            if dr <= 0.5 and da <= 15:
                cost[i,j] = (dr/0.5)**2+(da/15)**2
    result = {}
    for j in range(len(current)):
        i = int(np.argmin(cost[:,j]))
        value = cost[i,j]
        if (np.isfinite(value) and np.count_nonzero(np.isclose(cost[:,j],value)) == 1
                and np.argmin(cost[i]) == j and np.count_nonzero(np.isclose(cost[i],value)) == 1):
            result[j] = i
    return result


def compatible(r, angle, sigma, rr, rv, ra, valid):
    eligible = (valid & (rr<3.18) & (rv<=-0.35) & (np.abs(ra)+7<=12)
                & (np.abs(rr-r)<=0.35) & (np.abs(ra-angle)<=math.hypot(sigma,7)))
    return bool(eligible.any())


def joint_state(obs, yaw, pitch, joint=True):
    n = len(yaw)
    states = np.full(n,'CERTAIN_OUT',dtype='<U12')
    heights = np.full(n,'UNKNOWN',dtype='<U8')
    spatial = np.zeros(n,bool)
    sigma_now = np.zeros(n)
    sigma_future = np.full(n,np.nan)
    for ep in np.unique(obs['episode_id']):
        rows = np.flatnonzero(obs['episode_id']==ep)
        loadings = pose_loadings(obs['delta_yaw'][rows],obs['imu_valid'][rows])
        previous, previous_j = [], None
        for j,index in enumerate(rows):
            sigma = float(np.linalg.norm(loadings[j])); sigma_now[index] = sigma
            if not obs['tof_known'][index]:
                states[index] = 'MISSING'
                continue
            current = []
            for slot,r in enumerate(obs['tof_range_m'][index]):
                if np.isfinite(r):
                    angle,_ = rotate_ray(obs['tof_theta_deg'][index,slot],yaw[index],pitch[index])
                    current.append((r,angle,obs['tof_height'][index,slot],obs['tof_lateral'][index,slot]))
            elapsed = None if previous_j is None else (j-previous_j)*DT
            pairs = matches(previous,current) if elapsed is not None and elapsed<=0.3+1e-9 else {}
            for slot,(r,angle,height,lateral) in enumerate(current):
                state = old.point_state(angle,r,sigma) if r<3.18 else 'CERTAIN_OUT'
                if lateral and slot in pairs and r<3.6:
                    future = angle+(angle-previous[pairs[slot]][1])/elapsed
                    sf = forecast_sigma(loadings[previous_j],loadings[j],elapsed,joint)
                    sigma_future[index] = sf
                    state = old.stronger_state(state,old.crossing_state(angle,future,sigma,sf))
                states[index] = old.stronger_state(states[index],state)
                if state == 'CERTAIN_IN':
                    heights[index] = height
                elif state == 'UNCERTAIN':
                    spatial[index] |= compatible(r,angle,sigma,obs['radar_range_m'][index],
                        obs['radar_velocity'][index],obs['radar_angle'][index],obs['radar_valid'][index])
            previous,previous_j = current,j
    return {'state':states,'height':heights,'spatial':spatial,
            'sigma_now':sigma_now,'sigma_future':sigma_future}


def predict(obs):
    allowed = {'episode_id','tof_known','tof_range_m','tof_theta_deg','tof_lateral','tof_height',
               'radar_range_m','radar_velocity','radar_angle','radar_valid',
               'delta_yaw','delta_pitch','imu_valid'}
    if set(obs) != allowed:
        raise ValueError('Predictor accepts only the sealed observation schema')
    ids = obs['episode_id']
    yaw,sy = old.integrate_with_uncertainty(obs['delta_yaw'],obs['imu_valid'],ids)
    pitch,_ = old.integrate_with_uncertainty(obs['delta_pitch'],obs['imu_valid'],ids)
    radar = old.radar_expert(obs['radar_range_m'],obs['radar_velocity'],obs['radar_angle'],obs['radar_valid'],ids)
    legacy = old.tof_association(obs,yaw,pitch,sy)
    authority = old.apply_authority(legacy['state'],obs['tof_known'],radar,ids)
    hard = legacy['hard'] | (~obs['tof_known'] & radar)
    hold = hard.copy()
    for j in range(1,len(hard)):
        if ids[j]==ids[j-1] and not obs['tof_known'][j] and hard[j-1]:
            hold[j] = True
    values = {'hard_mz85':hard,'hard_hold':hold,'mz88_full':authority['full']}
    unknown = {'hard_mz85':~obs['tof_known'] & ~hard,'hard_hold':~obs['tof_known'] & ~hold,
               'mz88_full':np.isin(legacy['state'],['MISSING','UNCERTAIN']) & ~authority['full']}
    debug = {}
    for name,joint,spatial in (('marginal_spatial',False,True),('joint_coarse',True,False),('joint_spatial',True,True)):
        state = joint_state(obs,yaw,pitch,joint)
        certain = state['state']=='CERTAIN_IN'
        uncertain = state['state']=='UNCERTAIN'
        fallback = ~obs['tof_known'] & radar
        confirmation = uncertain & radar & (state['spatial'] if spatial else True)
        values[name] = certain | fallback | confirmation
        unknown[name] = np.isin(state['state'],['MISSING','UNCERTAIN']) & ~values[name]
        debug[name+'_height'] = np.where(certain,state['height'],'UNKNOWN')
        for key in ('state','spatial','sigma_now','sigma_future'):
            debug[name+'_'+key] = state[key]
    return values,unknown,debug


def intervals(mask):
    padded = np.r_[False,mask,False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded)==1),np.flatnonzero(np.diff(padded)==-1)))


def event_rows(ids, truth, values):
    events = []
    for ep in np.unique(ids):
        ix = np.flatnonzero(ids==ep)
        for start,end in intervals(truth[ix]):
            row = {'episode_id':str(ep),'start':int(start),'end_exclusive':int(end),'arms':{}}
            for name,v in values.items():
                segment = v[ix[start:end]]
                hits = np.flatnonzero(segment)
                row['arms'][name] = {'first':int(start+hits[0]) if hits.size else None,
                                     'fragments':len(intervals(segment))}
            events.append(row)
    return events


def evaluate(obs, evaluator, values, unknown):
    truth,ids = evaluator['truth'],obs['episode_id']
    metrics = {}
    for name,v in values.items():
        row = old.metrics(truth,v)
        row.update(UNKNOWN=int(unknown[name].sum()),positive_UNKNOWN=int((truth&unknown[name]).sum()),
                   false_segments=sum(len(intervals(v[ids==ep]&~truth[ids==ep])) for ep in np.unique(ids)))
        metrics[name] = row
    family = {f:{name:old.metrics(truth[evaluator['family']==f],v[evaluator['family']==f])
                 for name,v in values.items()} for f in old.FAMILIES}
    events = event_rows(ids,truth,values)
    for name in values:
        metrics[name]['missed_events'] = sum(r['arms'][name]['first'] is None for r in events)
        metrics[name]['fragments'] = sum(r['arms'][name]['fragments'] for r in events)
    base,cand,hold = (metrics[n] for n in ('hard_mz85','joint_spatial','hard_hold'))
    delays,misses = [],0
    for event in events:
        b,c = (event['arms'][name]['first'] for name in ('hard_mz85','joint_spatial'))
        if b is not None:
            if c is None: misses += 1
            else: delays.append((c-b)*DT)
    nominal = evaluator['variant']==0
    stressed = np.isin(evaluator['family'],['boundary_outside','head_motion']) & ~nominal
    baseline_fp = int((~truth & values['hard_mz85'] & stressed).sum())
    candidate_fp = int((~truth & values['joint_spatial'] & stressed).sum())
    gates = {
        'f1_at_least_both_controls':cand['F1']>=max(base['F1'],hold['F1']),
        'tp_loss_at_most_two':cand['TP']>=base['TP']-2,
        'fp_at_most_both_controls':cand['FP']<=min(base['FP'],hold['FP']),
        'boundary_head_fp_reduced_quarter':baseline_fp>0 and candidate_fp<=0.75*baseline_fp,
        'crossing_tp_loss_at_most_two':family['lateral_crossing']['joint_spatial']['TP']>=family['lateral_crossing']['hard_mz85']['TP']-2,
        'nominal_no_lost_tp_or_new_fp':not bool(((truth & values['hard_mz85'] & ~values['joint_spatial']) |
                                              (~truth & ~values['hard_mz85'] & values['joint_spatial']))[nominal].any()),
        'no_missed_baseline_event_or_delay_above_0p2':misses==0 and max(delays,default=0)<=0.2+1e-9,
        'false_segments_and_fragments_no_increase':cand['false_segments']<=base['false_segments'] and cand['fragments']<=base['fragments'],
        'strict_benefit_over_hold':cand['F1']>hold['F1'] or cand['FP']<hold['FP'] or cand['TP']>hold['TP'],
    }
    return {'metrics':metrics,'family_metrics':family,'events':events,'gates':gates,
            'timing':{'max_added_delay_s':max(delays,default=None),'missed_baseline_events':misses},
            'stressed_boundary_head_fp':{'baseline':baseline_fp,'candidate':candidate_fp}}


def run(output):
    output = output.resolve()
    if output.parent != (ROOT/'artifacts.local/work'/TASK).resolve() or output.exists():
        raise ValueError('Output must be a new direct child of the canonical MZ89 artifact root')
    output.mkdir(parents=True)
    start = time.perf_counter()
    for name in ('mz89_joint_geometry.py','mz88_uncertainty_aware_association.py',
                 'mz85_rotation_compensated_state.py','mz84_bidirectional_complementarity.py',
                 'MZ89_JOINT_GEOMETRY_20260912.md'):
        shutil.copyfile(HERE/name,output/name)
    source = build_source()
    obs,evaluator = materialize(source)
    old.write(output/'source.json',source)
    np.savez_compressed(output/'observations.npz',**obs)
    np.savez_compressed(output/'evaluator.npz',**evaluator)
    old.write(output/'source-receipt.json',{'status':'SEALED_BEFORE_PREDICTION',
              'seed':SEED,'episodes':len(source),'frames':len(obs['episode_id']),
              'hashes':{n:old.sha(output/n) for n in ('source.json','observations.npz','evaluator.npz')}})
    # Predictor receives only the serialized observation contract, not the source/truth.
    del source,evaluator,obs
    with np.load(output/'observations.npz',allow_pickle=False) as data:
        obs = dict(data)
    predictions = []
    def probe():
        predictions.append(predict(obs))
        return predictions[-1]
    select_backend('scalar-scoring',cpu=BackendCandidate('numpy-cpu','cpu',probe,
        lambda _:DeviceObservation('cpu','host CPU','numpy')),record_path=output/'backend.json',
        capabilities={'python':sys.version,'numpy':np.__version__,'scope':'scalar analytic scoring'})
    values,unknown,debug = predictions[0]
    np.savez_compressed(output/'predictions.npz',**values,
                        **{n+'_unknown':v for n,v in unknown.items()},**debug)
    old.write(output/'predictor-receipt.json',{'status':'SEALED_BEFORE_EVALUATION',
        'predictions_sha256':old.sha(output/'predictions.npz'),'training_steps':0,'threshold_searches':0})
    with np.load(output/'evaluator.npz',allow_pickle=False) as data:
        evaluator = dict(data)
    result = evaluate(obs,evaluator,values,unknown)
    state = debug['joint_spatial_state']
    allowed = (state=='CERTAIN_IN') | ((state=='UNCERTAIN') & debug['joint_spatial_spatial']) | ~obs['tof_known']
    integrity = {'unsupported_candidate_alerts':int((values['joint_spatial'] & ~allowed).sum()),
                 'noncertain_fabricated_height':int(((state!='CERTAIN_IN') & (debug['joint_spatial_height']!='UNKNOWN')).sum()),
                 'alert_unknown_overlap':sum(int((values[n]&unknown[n]).sum()) for n in values)}
    result['integrity'] = integrity
    result['gates']['authority_integrity'] = all(v==0 for v in integrity.values())
    result['decision'] = ('JOINT_GEOMETRY_CONTROLLED_GAIN' if all(result['gates'].values())
                          else 'JOINT_GEOMETRY_FULL_GATE_NOT_MET')
    old.write(output/'result.json',result)
    old.write(output/'receipt.json',{'status':'PASS','decision':result['decision'],
        'seconds':time.perf_counter()-start,'backend':'CPU TASK_NOT_GPU_SUITABLE scalar scoring',
        'hashes':{p.name:old.sha(p) for p in output.iterdir() if p.is_file()},
        'claim':'Constructed analytic Development only; no physical tube, sensor or safety claim.'})
    print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)

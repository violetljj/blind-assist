"""Common raw-sensor contract test for frozen MZ88/MZ89 geometric policies."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

import mz89_joint_geometry as previous
import mz90_observation_source as source_model

old = previous.old
TASK = 'mz90-observable-contract-20260912'
REGIMES = ('ideal','sensor_proxy')
RAW_KEYS = {'episode_id','time_s','tof_packet_received','tof_range_m','tof_theta_deg',
            'tof_range_sigma_m','tof_status','radar_packet_received','radar_range_m',
            'radar_velocity','radar_angle','radar_valid','delta_yaw','delta_pitch','imu_valid'}
ARM_NAMES = ('current_geometry','matched_hard','matched_hold','mz88_full',
             'marginal_spatial','joint_coarse','joint_spatial')


def adapt(raw):
    """No simulator truth, target motion hints, identities or stabilized Radar."""
    if set(raw)!=RAW_KEYS:
        raise ValueError('Unexpected raw observation schema')
    n = len(raw['episode_id'])
    yaw,_ = old.integrate_with_uncertainty(raw['delta_yaw'],raw['imu_valid'],raw['episode_id'])
    obs = {k:raw[k].copy() for k in ('episode_id','delta_yaw','delta_pitch','imu_valid')}
    obs['tof_range_m'] = np.full((n,2),np.nan)
    obs['tof_theta_deg'] = np.full((n,2),np.nan)
    for i in range(n):
        mask = (raw['tof_packet_received'][i] & (raw['tof_status'][i]==5)
                & np.isfinite(raw['tof_range_m'][i]) & (raw['tof_range_m'][i]>0)
                & np.isfinite(raw['tof_theta_deg'][i]))
        indexes = np.flatnonzero(mask)
        indexes = sorted(indexes,key=lambda j:(raw['tof_range_m'][i,j],raw['tof_theta_deg'][i,j]))[:2]
        for slot,j in enumerate(indexes):
            obs['tof_range_m'][i,slot] = raw['tof_range_m'][i,j]
            obs['tof_theta_deg'][i,slot] = raw['tof_theta_deg'][i,j]
    obs['tof_known'] = np.isfinite(obs['tof_range_m']).any(axis=1)
    # Constant eligibility: motion must be inferred from available history.
    obs['tof_lateral'] = np.isfinite(obs['tof_range_m'])
    obs['tof_height'] = np.full((n,2),'UNKNOWN',dtype='<U8')
    obs['radar_range_m'] = raw['radar_range_m'].copy()
    obs['radar_velocity'] = raw['radar_velocity'].copy()
    obs['radar_angle'] = (raw['radar_angle']+yaw[:,None]+180)%360-180
    obs['radar_valid'] = (raw['radar_valid'] & raw['radar_packet_received'][:,None]
        & np.isfinite(raw['radar_range_m']) & (raw['radar_range_m']>0)
        & np.isfinite(raw['radar_velocity']) & np.isfinite(raw['radar_angle']))
    return obs


def simple_controls(obs):
    n = len(obs['episode_id'])
    yaw,_ = old.integrate_with_uncertainty(obs['delta_yaw'],obs['imu_valid'],obs['episode_id'])
    pitch,_ = old.integrate_with_uncertainty(obs['delta_pitch'],obs['imu_valid'],obs['episode_id'])
    direct,matched = np.zeros(n,bool),np.zeros(n,bool)
    for ep in np.unique(obs['episode_id']):
        rows = np.flatnonzero(obs['episode_id']==ep)
        history,history_j = [],None
        for j,i in enumerate(rows):
            if not obs['tof_known'][i]:
                continue
            points = []
            for slot,r in enumerate(obs['tof_range_m'][i]):
                if np.isfinite(r):
                    angle,_ = previous.rotate_ray(obs['tof_theta_deg'][i,slot],yaw[i],pitch[i])
                    points.append((r,angle,'UNKNOWN',True))
            dt = None if history_j is None else (j-history_j)*previous.DT
            pairs = previous.matches(history,points) if dt is not None and dt<=0.3+1e-9 else {}
            for slot,(r,angle,_,_) in enumerate(points):
                current = r<3.18 and abs(angle)<=12
                direct[i] |= current
                matched[i] |= current
                if slot in pairs and r<3.6:
                    future = angle+(angle-history[pairs[slot]][1])/dt
                    matched[i] |= old.segment_intersects_corridor(angle,future)
            history,history_j = points,j
    radar = old.radar_expert(obs['radar_range_m'],obs['radar_velocity'],obs['radar_angle'],
                            obs['radar_valid'],obs['episode_id'])
    fallback = ~obs['tof_known'] & radar
    direct |= fallback
    matched |= fallback
    hold = matched.copy()
    for j in range(1,n):
        if obs['episode_id'][j]==obs['episode_id'][j-1] and not obs['tof_known'][j] and matched[j-1]:
            hold[j] = True
    return {'current_geometry':direct,'matched_hard':matched,'matched_hold':hold}


def predict(raw):
    obs = adapt(raw)
    values,unknown,debug = previous.predict(obs)
    values = {n:values[n] for n in ARM_NAMES if n in values}
    unknown = {n:unknown[n] for n in values}
    for name,value in simple_controls(obs).items():
        values[name] = value
        unknown[name] = ~obs['tof_known'] & ~value
    return obs,values,unknown,debug


def evaluate(ids, evaluator, values, unknown):
    truth = evaluator['truth']
    events = previous.event_rows(ids,truth,values)
    metrics = {}
    for name in ARM_NAMES:
        pred = values[name]
        row = old.metrics(truth,pred)
        row.update(UNKNOWN=int(unknown[name].sum()),positive_UNKNOWN=int((unknown[name]&truth).sum()),
            false_segments=sum(len(previous.intervals(pred[ids==ep]&~truth[ids==ep])) for ep in np.unique(ids)),
            missed_events=sum(e['arms'][name]['first'] is None for e in events),
            fragments=sum(e['arms'][name]['fragments'] for e in events))
        metrics[name] = row
    future = evaluator['future_only_truth']
    future_only = {name:{'TP':int((values[name]&future).sum()),'FN':int((~values[name]&future).sum())}
                   for name in ARM_NAMES}
    candidate,base,hold,marginal = (metrics[n] for n in ('joint_spatial','matched_hard','matched_hold','marginal_spatial'))
    delays,missed = [],0
    for e in events:
        b,c = (e['arms'][n]['first'] for n in ('matched_hard','joint_spatial'))
        if b is not None:
            if c is None: missed += 1
            else: delays.append((c-b)*previous.DT)
    transfer = {'adds_net_tp':candidate['TP']>marginal['TP'],
                'no_added_fp':candidate['FP']<=marginal['FP'],
                'no_added_false_segments':candidate['false_segments']<=marginal['false_segments'],
                'no_added_missed_events':candidate['missed_events']<=marginal['missed_events']}
    full = {'f1_at_least_all_simple':candidate['F1']>=max(metrics[n]['F1'] for n in ('current_geometry','matched_hard','matched_hold')),
            'tp_loss_le_two':candidate['TP']>=base['TP']-2,
            'fp_at_most_both_matched':candidate['FP']<=min(base['FP'],hold['FP']),
            'events_fragments_no_increase':candidate['missed_events']<=base['missed_events'] and candidate['fragments']<=base['fragments'],
            'no_lost_baseline_event_or_delay_over_0p2':missed==0 and max(delays,default=0)<=0.2+1e-9,
            'false_segments_no_increase':candidate['false_segments']<=base['false_segments'],
            'strict_benefit_over_hold':candidate['F1']>hold['F1'] or candidate['TP']>hold['TP'] or candidate['FP']<hold['FP']}
    paired = {'covariance_added_tp':int((truth&values['joint_spatial']&~values['marginal_spatial']).sum()),
              'covariance_lost_tp':int((truth&~values['joint_spatial']&values['marginal_spatial']).sum()),
              'covariance_added_fp':int((~truth&values['joint_spatial']&~values['marginal_spatial']).sum()),
              'covariance_removed_fp':int((~truth&~values['joint_spatial']&values['marginal_spatial']).sum()),
              'prediction_differences':int((values['joint_spatial']!=values['marginal_spatial']).sum())}
    return {'metrics':metrics,'future_only':future_only,'truth_positive':int(truth.sum()),
            'truth_events':len(events),'events':events,'transfer_gates':transfer,'full_gates':full,
            'covariance_transfer_pass':all(transfer.values()),'full_pass':all(full.values()),
            'paired_covariance':paired,'timing':{'max_added_delay_s':max(delays,default=None),'missed_baseline_events':missed}}


def availability(raw,obs):
    packet = raw['tof_packet_received']
    return {'frames':len(packet),'tof_packet_gaps':int((~packet).sum()),
            'tof_received_no_return':int((packet&~obs['tof_known']).sum()),
            'tof_any_valid':int(obs['tof_known'].sum()),
            'raw_valid_zones':int(((raw['tof_status']==5)&packet[:,None]).sum()),
            'selected_valid_returns':int(np.isfinite(obs['tof_range_m']).sum()),
            'radar_packet_gaps':int((~raw['radar_packet_received']).sum()),
            'radar_any_valid':int(obs['radar_valid'].any(axis=1).sum()),
            'imu_missing':int((~raw['imu_valid']).sum())}


def run(output):
    output = output.resolve()
    if output.exists() or output.parent!=(previous.ROOT/'artifacts.local/work'/TASK).resolve():
        raise ValueError('Output must be a new child of canonical MZ90 artifact root')
    output.mkdir(parents=True)
    started = time.perf_counter()
    files = ('mz90_observable_contract.py','mz90_observation_source.py','MZ90_LITERATURE_AND_PROTOCOL_20260912.md',
             'mz89_joint_geometry.py','mz88_uncertainty_aware_association.py','mz85_rotation_compensated_state.py',
             'mz84_bidirectional_complementarity.py')
    for name in files:
        shutil.copyfile(previous.HERE/name,output/name)
    source = source_model.build_source()
    old.write(output/'source.json',source)
    for regime in REGIMES:
        target = output/regime
        target.mkdir()
        raw,evaluator = source_model.materialize(source,regime)
        np.savez_compressed(target/'raw-observations.npz',**raw)
        np.savez_compressed(target/'evaluator.npz',**evaluator)
    old.write(output/'source-receipt.json',{'status':'BOTH_REGIMES_SEALED_BEFORE_PREDICTION',
        'hashes':{str(p.relative_to(output)):old.sha(p) for p in output.rglob('*') if p.is_file()}})
    del source,raw,evaluator
    summaries = {}
    for regime in REGIMES:
        target = output/regime
        with np.load(target/'raw-observations.npz',allow_pickle=False) as data: raw=dict(data)
        results = []
        def probe():
            results.append(predict(raw))
            return results[-1]
        previous.select_backend('scalar-scoring',cpu=previous.BackendCandidate('numpy-cpu','cpu',probe,
            lambda _:previous.DeviceObservation('cpu','host CPU','numpy')),record_path=target/'backend.json',
            capabilities={'numpy':np.__version__,'scope':'scalar sensor-contract scoring'})
        obs,values,unknown,debug = results[0]
        np.savez_compressed(target/'adapted-observations.npz',**obs)
        np.savez_compressed(target/'predictions.npz',**values,**{n+'_unknown':v for n,v in unknown.items()},**debug)
        old.write(target/'predictor-receipt.json',{'status':'SEALED_BEFORE_ANY_EVALUATOR_OPEN',
            'predictions_sha256':old.sha(target/'predictions.npz'),'training_steps':0,'searches':0})
        summaries[regime] = availability(raw,obs)
    # Both prediction payloads are now sealed. Only now open evaluator files.
    result = {'regimes':{},'observation_availability':summaries}
    truths = []
    for regime in REGIMES:
        target=output/regime
        with np.load(target/'evaluator.npz',allow_pickle=False) as data: evaluator=dict(data)
        with np.load(target/'predictions.npz',allow_pickle=False) as data: predictions=dict(data)
        with np.load(target/'adapted-observations.npz',allow_pickle=False) as data: obs=dict(data)
        values={n:predictions[n] for n in ARM_NAMES}
        unknown={n:predictions[n+'_unknown'] for n in ARM_NAMES}
        truth=evaluator['truth']; truths.append(truth)
        result['regimes'][regime]=evaluate(obs['episode_id'],evaluator,values,unknown)
        violations={'alert_unknown_overlap':sum(int((values[n]&unknown[n]).sum()) for n in ARM_NAMES),
                    'fabricated_height':sum(int((v!='UNKNOWN').sum()) for k,v in predictions.items() if k.endswith('_height')),
                    'populated_missing_tof':int(np.isfinite(obs['tof_range_m'][~obs['tof_known']]).sum())}
        if any(violations.values()): raise RuntimeError(f'Integrity violation: {violations}')
        result['regimes'][regime]['integrity']=violations
    assert np.array_equal(truths[0],truths[1]),'Paired truth mismatch'
    proxy=result['regimes']['sensor_proxy']
    if proxy['full_pass']:
        decision='OBSERVABLE_CONTRACT_CONTROLLED_FULL_GAIN'
    elif proxy['covariance_transfer_pass']:
        decision='OBSERVABLE_CONTRACT_COVARIANCE_ONLY_FULL_GATE_NOT_MET'
    else:
        decision='OBSERVABLE_CONTRACT_COVARIANCE_TRANSFER_NOT_ESTABLISHED'
    result['decision']=decision
    old.write(output/'result.json',result)
    receipt={'status':'PASS','decision':decision,'seconds':time.perf_counter()-started,
             'paired_truth_identical':True,'training_steps':0,'threshold_searches':0,
             'backend':'CPU scalar scoring; no persistent or paid allocation',
             'hashes':{str(p.relative_to(output)):old.sha(p) for p in output.rglob('*') if p.is_file()},
             'claim':'Uncalibrated horizontal simulated sensor contract only; no8x8 or physical collision performance.'}
    old.write(output/'receipt.json',receipt)
    print(json.dumps({'decision':decision,'regimes':{r:{k:v for k,v in result['regimes'][r].items() if k!='events'}
                      for r in REGIMES},'availability':summaries},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)

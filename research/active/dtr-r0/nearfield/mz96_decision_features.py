"""Observable-only frame features and fixed rule comparators for MZ96."""
import numpy as np
import mz94_radar_horizon as a
import mz95_coverage_authority as b


def build(raw):
    obs=a.contract.adapt(raw);n=len(obs['episode_id'])
    horizon,length=a.horizon_admission(obs)
    legacy=obs['radar_valid']&(obs['radar_range_m']<3.18)&(obs['radar_velocity']<=-.35)&(np.abs(obs['radar_angle'])<=20)
    expanded=obs['radar_valid']&(obs['radar_range_m']<3.6)&(obs['radar_velocity']<=-.35)&(np.abs(obs['radar_angle'])<=20)
    covered=b.coverage(obs,raw,legacy);covered_a=b.coverage(obs,raw,horizon)
    values={'matched_hold':a.contract.simple_controls(obs)['matched_hold']}
    values['range_only']=a.assemble_admission(obs,expanded)[0]
    values['horizon_A']=a.assemble_admission(obs,horizon)[0]
    values['coverage_B']=b.combine(obs,legacy,covered)[0]
    values['A_plus_B']=b.combine(obs,horizon,covered_a)[0]
    no_radar={**obs,'radar_valid':np.zeros_like(legacy)}
    tof=a.contract.simple_controls(no_radar)
    yaw,_=a.contract.old.integrate_with_uncertainty(obs['delta_yaw'],obs['imu_valid'],obs['episode_id'])
    features={}
    for k in range(2):
        features[f'tof{k}_range']=obs['tof_range_m'][:,k]
        features[f'tof{k}_bearing']=obs['tof_theta_deg'][:,k]+yaw
    ranked=np.full((n,2,3),np.nan)
    for i in range(n):
        ix=sorted(np.flatnonzero(obs['radar_valid'][i]),key=lambda j:(obs['radar_range_m'][i,j],obs['radar_angle'][i,j],obs['radar_velocity'][i,j]))[:2]
        for k,j in enumerate(ix):ranked[i,k]=[obs['radar_range_m'][i,j],obs['radar_angle'][i,j],obs['radar_velocity'][i,j]]
    for k in range(2):
        for j,name in enumerate(('range','bearing','radial_velocity')):features[f'radar{k}_{name}']=ranked[:,k,j]
    features.update(tof_valid_count=((raw['tof_status']==5)&raw['tof_packet_received'][:,None]).sum(1),
        radar_valid_count=obs['radar_valid'].sum(1),radar_qualified_count=legacy.sum(1),radar_horizon_count=horizon.sum(1),
        radar_track_length=length.max(1),tof_direct=tof['current_geometry'],tof_temporal_required=tof['matched_hard']&~tof['current_geometry'],
        tof_known=obs['tof_known'],tof_packet=raw['tof_packet_received'],coverage_legacy=covered,coverage_horizon=covered_a,
        range_disagreement=obs['tof_range_m'][:,0]-ranked[:,0,0],bearing_disagreement=(obs['tof_theta_deg'][:,0]+yaw-ranked[:,0,1]+180)%360-180,
        yaw_delta=raw['delta_yaw'],imu_valid=raw['imu_valid'])
    radar_state=a.hysteresis(legacy.any(1),obs['episode_id'])
    features['sensor_conflict']=obs['tof_known']&(tof['matched_hard']!=radar_state)
    for name,current in (('radar_support',legacy.any(1)),('tof_support',tof['matched_hard'])):
        for lag in (1,2,3):
            v=np.zeros(n)
            for i in range(lag,n):
                if obs['episode_id'][i-lag]==obs['episode_id'][i]:v[i]=current[i-lag]
            features[f'{name}_lag{lag}']=v
        age=np.full(n,10.)
        for i in range(n):
            age[i]=0 if current[i] else min(10.,age[i-1]+.1) if i and obs['episode_id'][i-1]==obs['episode_id'][i] else 10.
        features[f'{name}_age']=age
    names=list(features);matrix=np.column_stack([features[k] for k in names]).astype(np.float32)
    assert not np.isinf(matrix).any()
    return obs,matrix,names,values

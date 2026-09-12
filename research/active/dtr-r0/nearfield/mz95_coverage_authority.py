"""B: pointwise horizontal ToF coverage qualifies a negative branch authority."""
import numpy as np
from mz90_observable_contract import adapt, simple_controls, old
import mz90_observable_contract as contract
from mz84_bidirectional_complementarity import hysteresis


def coverage(obs, raw, admitted):
    n=len(obs['episode_id']);covered=np.zeros(n,bool);last=None
    yaw,_=old.integrate_with_uncertainty(obs['delta_yaw'],obs['imu_valid'],obs['episode_id'])
    for i in range(n):
        if i==0 or obs['episode_id'][i]!=obs['episode_id'][i-1]:last=None
        if admitted[i].any():last=i
        if last is None or i-last>1 or not obs['tof_known'][i]:continue
        bins=np.flatnonzero(raw['tof_packet_received'][i]&(raw['tof_status'][i]==5)
            &np.isfinite(raw['tof_range_m'][i])&np.isfinite(raw['tof_theta_deg'][i]))
        supports=np.flatnonzero(admitted[last]);all_covered=True
        for j in supports:
            r=obs['radar_range_m'][last,j]+obs['radar_velocity'][last,j]*(i-last)*.1
            a=obs['radar_angle'][last,j]
            matches=[k for k in bins if abs(raw['tof_range_m'][i,k]-r)<=.15 and
                     abs((raw['tof_theta_deg'][i,k]+yaw[i]-a+180)%360-180)<=5.625/2]
            if not matches:all_covered=False;break
        covered[i]=all_covered
    return covered


def combine(obs,admitted,qualified_coverage):
    tof=simple_controls({**obs,'radar_valid':np.zeros_like(obs['radar_valid'])})['matched_hard']
    radar=hysteresis(admitted.any(axis=1),obs['episode_id'])
    hard=tof|(radar&(~obs['tof_known']|~qualified_coverage))
    hold=hard.copy()
    for i in range(1,len(hold)):
        if obs['episode_id'][i]==obs['episode_id'][i-1] and not obs['tof_known'][i] and hard[i-1]:hold[i]=True
    # Valid ToF outside the Radar support region is not negative authority there.
    unknown=~hold&(~obs['tof_known']|(radar&~qualified_coverage))
    return hold,unknown


def predict(raw):
    obs=adapt(raw)
    admitted=obs['radar_valid']&(obs['radar_range_m']<3.18)&(obs['radar_velocity']<=-.35)&(np.abs(obs['radar_angle'])<=20)
    covered=coverage(obs,raw,admitted)
    candidate,unknown=combine(obs,admitted,covered)
    baseline=simple_controls(obs)['matched_hold']
    return obs,{'unchanged_control':baseline,'coverage_B':candidate},{'unchanged_control':~obs['tof_known']&~baseline,'coverage_B':unknown},dict(coverage=covered,admitted=admitted)

"""Causal cross-sensor angular calibration; frozen MZ91 decision rules."""
import numpy as np
from mz91_radar_localization import (contract, DT, WINDOW, BIAS_SIGMA, wrap, localize, risk, assemble)
import mz91_radar_localization as prior


def estimate_bias(raw):
    """Estimate Radar-minus-ToF bearing bias without source IDs or absent-return evidence."""
    n=len(raw['episode_id']); mean=np.zeros(n); variance=np.full(n,BIAS_SIGMA**2)
    count=np.zeros(n,int); measurements=np.full(n,np.nan); history=[]
    noise_variance=2**2+10**2/12+5.625**2/12
    for i in range(n):
        if i==0 or raw['episode_id'][i]!=raw['episode_id'][i-1]: history=[]
        history=[item for item in history if i-item[0]<=20]
        tof=np.flatnonzero(raw['tof_packet_received'][i]&(raw['tof_status'][i]==5)
             &np.isfinite(raw['tof_range_m'][i])&(raw['tof_range_m'][i]>0)&np.isfinite(raw['tof_theta_deg'][i]))
        radar=np.flatnonzero(raw['radar_packet_received'][i]&raw['radar_valid'][i]
             &np.isfinite(raw['radar_range_m'][i])&(raw['radar_range_m'][i]>0)&np.isfinite(raw['radar_angle'][i]))
        compatible=np.zeros((len(tof),len(radar)),bool)
        for t,k in enumerate(tof):
            for r,j in enumerate(radar):
                compatible[t,r]=(abs(raw['tof_range_m'][i,k]-raw['radar_range_m'][i,j])<=.15
                    and abs(wrap(raw['radar_angle'][i,j]-raw['tof_theta_deg'][i,k]))<=25)
        pairs=[(t,r) for t,r in zip(*np.nonzero(compatible))
               if compatible[t].sum()==1 and compatible[:,r].sum()==1]
        if len(pairs)==1:
            t,r=pairs[0]
            value=float(wrap(raw['radar_angle'][i,radar[r]]-raw['tof_theta_deg'][i,tof[t]]))
            history.append((i,value)); measurements[i]=value
        count[i]=len(history)
        if len(history)>=3:
            posterior=1/(1/BIAS_SIGMA**2+len(history)/noise_variance)
            mean[i]=posterior*sum(v for _,v in history)/noise_variance
            variance[i]=posterior+2.5**2
    return mean,variance,count,measurements


def calibrated_localize(history, pose_sigma, bias, variance):
    # Reproject the entire raw-stabilized history under the CURRENT common bias.
    # Mixing earlier corrected angles would create artificial angular velocity.
    corrected=[[i,r,wrap(a-bias),v] for i,r,a,v in history]
    mean,cov=localize(corrected,pose_sigma,True)
    cov[2,2]+=variance-BIAS_SIGMA**2
    return mean,cov


def localization_scores(obs, bias, variance):
    n = len(obs['episode_id'])
    probability = np.full((n,4),np.nan)
    length = np.zeros((n,4),int)
    _, pose = contract.old.integrate_with_uncertainty(obs['delta_yaw'],obs['imu_valid'],obs['episode_id'])
    tracks = []; gaps = 0
    for i in range(n):
        if i==0 or obs['episode_id'][i]!=obs['episode_id'][i-1]:
            tracks=[]; gaps=0
        if not obs['imu_valid'][i]:
            tracks=[]; gaps+=1
        sigma = np.hypot(pose[i],gaps*5.)
        tracks = [h for h in tracks if i-h[-1][0]<=3]
        slots = np.flatnonzero(obs['radar_valid'][i])
        points = [[i,float(obs['radar_range_m'][i,k]),float(obs['radar_angle'][i,k]),
                   float(obs['radar_velocity'][i,k])] for k in slots]
        cost = np.full((len(tracks),len(points)),np.inf)
        for j,h in enumerate(tracks):
            for k,p in enumerate(points):
                dt=(i-h[-1][0])*DT
                dr=abs(p[1]-(h[-1][1]+h[-1][3]*dt))
                da=abs(wrap(p[2]-h[-1][2])); dv=abs(p[3]-h[-1][3])
                if dr<=.5 and da<=15 and dv<=.5:
                    cost[j,k]=(dr/.5)**2+(da/15)**2+(dv/.5)**2
        used=set(); fresh=[]
        for k,p in enumerate(points):
            h=[]
            if len(tracks):
                column=cost[:,k]; j=int(np.argmin(column)); v=column[j]
                if np.isfinite(v) and np.count_nonzero(np.isclose(column,v))==1:
                    row=cost[j]
                    if int(np.argmin(row))==k and np.count_nonzero(np.isclose(row,row[k]))==1:
                        h=tracks[j]; used.add(j)
            h=(h+[p])[-WINDOW:]
            h=[p0 for p0 in h if i-p0[0]<=5]
            probability[i,slots[k]]=risk(*calibrated_localize(h,sigma,bias[i],variance[i]))
            length[i,slots[k]]=len(h)
            fresh.append(h)
        tracks=[h for j,h in enumerate(tracks) if j not in used]+fresh
    return probability,length


def predict(raw):
    obs=contract.adapt(raw)
    bias,variance,count,measurements=estimate_bias(raw)
    control,length0=prior.localization_scores(obs,True)
    calibrated,length=localization_scores(obs,bias,variance)
    values={};unknown={};debug=dict(bias_deg=bias,bias_variance=variance,
        calibration_count=count,paired_difference_deg=measurements,history_length=length)
    for name,score in (('mz91_control',control),('mz92_calibrated',calibrated)):
        values[name],unknown[name],debug[name+'_veto']=assemble(obs,score)
        debug[name+'_probability']=score
    return obs,values,unknown,debug

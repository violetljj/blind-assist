"""Separate observed occupancy from predictions based on uncertain lateral motion."""
import numpy as np
from mz91_radar_localization import DT, WINDOW, wrap, NORMAL, assemble
from mz92_cross_sensor_calibration import contract, calibrated_localize, estimate_bias

def motion_supported(mean, covariance, count):
    return count>=3 and mean[2]*mean[3]<0 and abs(mean[3])>1.96*np.sqrt(covariance[3,3])

def risk_components(mean, covariance):
    # Eigen factor supports the deliberately zero single-frame angular-rate variance.
    eigen, vectors = np.linalg.eigh(covariance)
    samples = mean+NORMAL@(vectors@np.diag(np.sqrt(np.maximum(0,eigen)))).T
    r, vr, angle, omega = samples.T
    angle, omega = np.radians(angle), np.radians(omega)
    x,z = r*np.sin(angle), r*np.cos(angle)
    vx,vz = vr*np.sin(angle)+r*np.cos(angle)*omega, vr*np.cos(angle)-r*np.sin(angle)*omega
    current = (z>.2)&(r>0)&(r<3.18)&(np.abs(np.degrees(angle))<=12)
    # Clip a continuous one-second constant-velocity trajectory to the wedge.
    lo, hi = np.zeros(len(r)), np.ones(len(r))
    possible = np.ones(len(r), bool)
    slope = np.tan(np.radians(12))
    for intercept, speed in ((z-.2,vz),(slope*z-x,slope*vz-vx),(slope*z+x,slope*vz+vx)):
        moving = np.abs(speed)>1e-12
        crossing = np.divide(-intercept,speed,out=np.zeros_like(speed),where=moving)
        lo = np.where(speed>1e-12,np.maximum(lo,crossing),lo)
        hi = np.where(speed< -1e-12,np.minimum(hi,crossing),hi)
        possible &= moving | (intercept>=0)
    future = possible&(lo<=hi)&(z+vz*((lo+hi)/2)>.2)&(z>.2)&(r>0)&(r<3.6)
    return float((current|future).mean()), float(current.mean())


def score_tracks(obs, bias, variance):
    n = len(obs['episode_id'])
    probability = np.full((n,4),np.nan)
    length = np.zeros((n,4),int)
    current = np.full((n,4),np.nan)
    supported = np.zeros((n,4),bool)
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
            mean,cov=calibrated_localize(h,sigma,bias[i],variance[i])
            probability[i,slots[k]],current[i,slots[k]]=risk_components(mean,cov)
            supported[i,slots[k]]=motion_supported(mean,cov,len(h))
            length[i,slots[k]]=len(h)
            fresh.append(h)
        tracks=[h for j,h in enumerate(tracks) if j not in used]+fresh
    return probability,current,supported,length


def predict(raw):
    obs=contract.adapt(raw)
    bias,variance,count,measurements=estimate_bias(raw)
    total,current,supported,length=score_tracks(obs,bias,variance)
    candidate=np.where(supported,total,current)
    values={};unknown={};debug=dict(total_probability=total,current_probability=current,
        supported_lateral_motion=supported,history_length=length,calibration_count=count)
    for name,score in (('mz92_control',total),('mz93_supported',candidate)):
        values[name],unknown[name],debug[name+'_veto']=assemble(obs,score)
    return obs,values,unknown,debug

"""A: deterministic causal CV Radar admission aligned with the route time horizon."""
import numpy as np
from mz91_radar_localization import contract, DT, WINDOW, wrap, localize
from mz84_bidirectional_complementarity import hysteresis


def corridor_hit(x,z,vx,vz,horizon=1.):
    lo,hi=0.,horizon; k=np.tan(np.radians(12))
    for a,b in ((z-.2,vz),(k*z-x,k*vz-vx),(k*z+x,k*vz+vx)):
        if abs(b)<1e-12:
            if a<0:return False
        elif b>0:lo=max(lo,-a/b)
        else:hi=min(hi,-a/b)
    return lo<=hi and z+vz*(lo+hi)/2>.2


def trajectory_admission(history):
    _,r,a,vr=history[-1]
    theta=np.radians(a);x,z=r*np.sin(theta),r*np.cos(theta)
    # GT includes t=0 in its future interval. Current corridor occupancy needs no velocity guess.
    if z>.2 and r<3.6 and abs(a)<=12:return True
    if len(history)<3 or history[-1][0]-history[0][0]<2:return False
    mean,_=localize(history,0.,True)
    r,vr,a,omega=mean;theta,omega=np.radians(a),np.radians(omega)
    x,z=r*np.sin(theta),r*np.cos(theta)
    vx=vr*np.sin(theta)+r*np.cos(theta)*omega
    vz=vr*np.cos(theta)-r*np.sin(theta)*omega
    return bool(0<r<3.6 and z>.2 and corridor_hit(x,z,vx,vz))


def assemble_admission(obs,admitted):
    tof=contract.simple_controls({**obs,'radar_valid':np.zeros_like(obs['radar_valid'])})['matched_hard']
    # Never call radar_expert here: it would silently restore the old raw gate.
    radar=hysteresis(admitted.any(axis=1),obs['episode_id'])
    hard=tof|(~obs['tof_known']&radar);hold=hard.copy()
    for i in range(1,len(hold)):
        if obs['episode_id'][i]==obs['episode_id'][i-1] and not obs['tof_known'][i] and hard[i-1]:hold[i]=True
    return hold,~obs['tof_known']&~hold


def horizon_admission(obs):
    n = len(obs['episode_id'])
    admitted = np.zeros((n,4),bool)
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
            admitted[i,slots[k]]=trajectory_admission(h)
            length[i,slots[k]]=len(h)
            fresh.append(h)
        tracks=[h for j,h in enumerate(tracks) if j not in used]+fresh
    return admitted,length


def predict(raw):
    obs=contract.adapt(raw)
    admitted,length=horizon_admission(obs)
    expanded=obs['radar_valid']&(obs['radar_range_m']<3.6)&(obs['radar_velocity']<=-.35)&(np.abs(obs['radar_angle'])<=20)
    values={};unknown={}
    for name,q in (('range_only_control',expanded),('horizon_A',admitted)):
        values[name],unknown[name]=assemble_admission(obs,q)
    return obs,values,unknown,dict(admitted=admitted,history_length=length)

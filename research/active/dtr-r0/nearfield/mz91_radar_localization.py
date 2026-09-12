"""Causal short-window Radar localization; shared angle bias never averages away."""
import numpy as np
import mz90_observable_contract as contract

DT = .1
WINDOW = 5
ANGLE_SIGMA = np.sqrt(2**2 + 10**2/12)
BIAS_SIGMA = 10.0
VETO_PROBABILITY = .1
_normal = np.random.default_rng(91012).normal(size=(256, 4))
NORMAL = np.concatenate((_normal, -_normal))


def wrap(a):
    return (a+180) % 360-180


def localize(history, pose_sigma, temporal=True):
    """Return (range, radial speed, angle, angular speed), covariance."""
    h = np.asarray(history[-WINDOW:])
    dt = (h[:, 0]-h[-1, 0])*DT
    design = np.column_stack((np.ones(len(h)), dt))
    if not temporal or len(h) < 3:
        return np.array([h[-1,1], h[-1,3], h[-1,2], 0.]), np.diag(
            [.06**2, .08**2, ANGLE_SIGMA**2+BIAS_SIGMA**2+pose_sigma**2, 0.])
    # Range positions plus independent Doppler observations constrain radial speed.
    precision = design.T@design/.06**2 + np.diag([0., len(h)/.08**2])
    rcov = np.linalg.inv(precision)
    radial = rcov@(design.T@h[:,1]/.06**2 + [0., h[:,3].sum()/.08**2])
    angles = h[-1,2]+wrap(h[:,2]-h[-1,2])
    # Weak fixed angular-rate prior; common offsets affect intercept, never 1/N.
    acov = np.linalg.inv(design.T@design/ANGLE_SIGMA**2+np.diag([0.,1/30**2]))
    angular = acov@(design.T@angles/ANGLE_SIGMA**2)
    acov[0,0] += BIAS_SIGMA**2+pose_sigma**2
    acov[1,1] += 2**2
    cov = np.zeros((4,4)); cov[:2,:2] = rcov; cov[2:,2:] = acov
    return np.r_[radial, angular], cov


def risk(mean, covariance):
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
    return float((current|future).mean())


def localization_scores(obs, temporal=True):
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
            probability[i,slots[k]]=risk(*localize(h,sigma,temporal))
            length[i,slots[k]]=len(h)
            fresh.append(h)
        tracks=[h for j,h in enumerate(tracks) if j not in used]+fresh
    return probability,length


def assemble(obs, probability):
    """Preserve ToF, two-of-three hysteresis and non-reseeding one-frame hold."""
    no_radar={**obs,'radar_valid':np.zeros_like(obs['radar_valid'])}
    tof=contract.simple_controls(no_radar)['matched_hard']
    qualifies=obs['radar_valid']&(obs['radar_range_m']<3.18)&(obs['radar_velocity']<=-.35)&(np.abs(obs['radar_angle'])<=20)
    accepted=qualifies&(probability>=VETO_PROBABILITY)
    radar=contract.old.radar_expert(obs['radar_range_m'],obs['radar_velocity'],obs['radar_angle'],accepted,obs['episode_id'])
    hard=tof|(~obs['tof_known']&radar)
    hold=hard.copy()
    for i in range(1,len(hold)):
        if obs['episode_id'][i]==obs['episode_id'][i-1] and not obs['tof_known'][i] and hard[i-1]:
            hold[i]=True
    return hold,~obs['tof_known']&~hold,qualifies&~accepted


def predict(raw):
    obs=contract.adapt(raw)
    values={}; unknown={}; debug={}
    for name,temporal in (('single_frame',False),('multi_frame',True)):
        score,length=localization_scores(obs,temporal)
        values[name],unknown[name],debug[name+'_veto']=assemble(obs,score)
        debug[name+'_probability']=score
        debug[name+'_history_length']=length
    return obs,values,unknown,debug

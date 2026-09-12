"""Observable-only nominal angle anchors. No labels, scene parameters or identities."""
import numpy as np

ZONE_HALF=45./8/2
RANGE_CLUSTER=.30
RANGE_MATCH=.35
ANGLE_MATCH=20.
ANGLE_SLACK=9.  # Nominal quantization half-step + 2 noise sigma; NOT a confidence bound.
WINDOW_S=1.
MAX_WIDTH=10.


def wrap(a):return (a+180)%360-180


def clusters(raw,i):
    good=(raw['tof_packet_received'][i] & (raw['tof_status'][i]==5)
          & np.isfinite(raw['tof_range_m'][i]) & (raw['tof_range_m'][i]>0)
          & np.isfinite(raw['tof_theta_deg'][i]))
    groups=[]
    for k in np.flatnonzero(good):
        if groups and k==groups[-1][-1]+1 and abs(raw['tof_range_m'][i,k]-raw['tof_range_m'][i,k-1])<=RANGE_CLUSTER:
            groups[-1].append(int(k))
        else:groups.append([int(k)])
    return [dict(zones=g,r=float(np.median(raw['tof_range_m'][i,g])),
        lo=float(min(raw['tof_theta_deg'][i,g])-ZONE_HALF),hi=float(max(raw['tof_theta_deg'][i,g])+ZONE_HALF)) for g in groups]


def frame_anchor(raw,i):
    groups=clusters(raw,i)
    valid=(raw['radar_packet_received'][i] & raw['radar_valid'][i]
           & np.isfinite(raw['radar_range_m'][i]) & (raw['radar_range_m'][i]>0)
           & np.isfinite(raw['radar_angle'][i]) & np.isfinite(raw['radar_velocity'][i]))
    slots=np.flatnonzero(valid);edges=np.zeros((len(groups),len(slots)),bool)
    for j,g in enumerate(groups):
        for k,s in enumerate(slots):
            edges[j,k]=(abs(g['r']-raw['radar_range_m'][i,s])<=RANGE_MATCH
                and abs(wrap(raw['radar_angle'][i,s]-(g['lo']+g['hi'])/2))<=ANGLE_MATCH)
    pairs=[]
    for j,k in zip(*np.nonzero(edges)):
        if edges[j].sum()!=1 or edges[:,k].sum()!=1:continue
        g=groups[j];s=int(slots[k]);d=float(wrap(raw['radar_angle'][i,s]-(g['lo']+g['hi'])/2))
        half=(g['hi']-g['lo'])/2+ANGLE_SLACK
        pairs.append(dict(slot=s,zones=g['zones'],lo=max(-20.,d-half),hi=min(20.,d+half)))
    status='ambiguous' if edges.any() else 'no_pair';interval=None
    if pairs:
        lo=max(p['lo'] for p in pairs);hi=min(p['hi'] for p in pairs)
        if lo<=hi:status='anchor';interval=(lo,hi)
        else:status='conflict'
    return interval,dict(status=status,cluster_count=len(groups),candidate_edges=int(edges.sum()),pairs=pairs)


def estimate(raw):
    n=len(raw['episode_id']);bias=np.full(n,np.nan);low=bias.copy();high=bias.copy();age=bias.copy()
    available=np.zeros(n,bool);applied=available.copy();records=[];history=[];cached=None
    for i in range(n):
        t=float(raw['time_s'][i])
        if i==0 or raw['episode_id'][i]!=raw['episode_id'][i-1]:history=[];cached=None
        # Strictly earlier validated anchors only; absence never refreshes their age.
        if cached is not None and t-cached[0]<=WINDOW_S+1e-9:
            available[i]=True;low[i],high[i]=cached[1:];bias[i]=(low[i]+high[i])/2;age[i]=t-cached[0]
        goodtof=bool(clusters(raw,i))
        validradar=raw['radar_valid'][i] & raw['radar_packet_received'][i]
        validradar &= np.isfinite(raw['radar_range_m'][i]) & (raw['radar_range_m'][i]>0)
        validradar &= np.isfinite(raw['radar_angle'][i]) & np.isfinite(raw['radar_velocity'][i])
        applied[i]=available[i] and not goodtof and bool(validradar.any())
        interval,record=frame_anchor(raw,i);record.update(frame=i,time_s=t)
        history=[h for h in history if t-h[0]<=WINDOW_S+1e-9]
        if record['status']=='conflict':history=[];cached=None
        elif interval is not None:
            history.append((t,*interval))
            lo=max(h[1] for h in history);hi=min(h[2] for h in history)
            if lo>hi:
                history=[];cached=None;record['status']='temporal_conflict'
            elif len(history)>=3 and t-history[0][0]>=.2-1e-9 and hi-lo<=MAX_WIDTH:
                cached=(t,lo,hi);record['status']='validated_anchor'
        records.append(record)
    return dict(bias_deg=bias,lower_deg=low,upper_deg=high,age_s=age,available=available,applied=applied),records


def correct(raw,trace):
    out={k:v.copy() for k,v in raw.items()}
    for i in np.flatnonzero(trace['applied']):
        valid=(raw['radar_valid'][i] & np.isfinite(raw['radar_angle'][i])
            & np.isfinite(raw['radar_range_m'][i]) & (raw['radar_range_m'][i]>0)
            & np.isfinite(raw['radar_velocity'][i]))
        out['radar_angle'][i,valid]=wrap(raw['radar_angle'][i,valid]-trace['bias_deg'][i])
    return out

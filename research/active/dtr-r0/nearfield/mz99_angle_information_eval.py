"""Explicit simulation geometry and causal warning-session evaluation."""
import numpy as np
import run_mz96_decision_head as common

DT=.1
BODY_HALF_X=.30
BODY_HALF_Z=.20
HORIZON=1.
MIN_LEAD=.5


def slab_interval(x,z,vx,vz,xlo,xhi,zlo,zhi,horizon):
    lo,hi=0.,float(horizon)
    for p,v,lower,upper in ((x,vx,xlo,xhi),(z,vz,zlo,zhi)):
        if abs(v)<1e-12:
            if not lower<=p<=upper:return None
        else:
            a,b=(lower-p)/v,(upper-p)/v
            lo=max(lo,min(a,b));hi=min(hi,max(a,b))
        if lo>hi+1e-10:return None
    return max(0.,lo),max(lo,hi)


def contact_interval(b,horizon=HORIZON):
    # Native extents are UE(world forward,lateral,vertical); evaluator x is lateral.
    hx=BODY_HALF_X+b['extent_m'][1];hz=BODY_HALF_Z+b['extent_m'][0]
    return slab_interval(b['x'],b['z'],b['vx'],b['vz'],-hx,hx,-hz,hz,horizon)


def route_current(b):
    hx=b['extent_m'][1];hz=b['extent_m'][0]
    return bool(b['x']+hx>=-BODY_HALF_X and b['x']-hx<=BODY_HALF_X
        and b['z']+hz>=.20 and b['z']-hz<=3.60)


def labels_from_geometry(geometry):
    route=[];body=[];now=[]
    for row in geometry:
        bounds=row['native_bounds']
        route.append(any(route_current(b) for b in bounds))
        body.append(any(contact_interval(b) is not None for b in bounds))
        now.append(any(contact_interval(b,0.) is not None for b in bounds))
    route,body,now=map(lambda v:np.asarray(v,bool),(route,body,now))
    return dict(current_route=route,body_1s=body,current_contact=now,strict_future_contact=body&~now)


def sessions(pred,ids):
    active=np.zeros(len(pred),bool);starts=np.zeros(len(pred),bool);on=False;empty=0
    for i in range(len(pred)):
        if i==0 or ids[i]!=ids[i-1]:on=False;empty=0
        if pred[i]:
            if not on:starts[i]=True
            on=True;empty=0
        elif on:
            empty+=1
            if empty>=4:on=False
        active[i]=on
    return active,starts


def session_metrics(pred,ids,truth):
    active,starts=sessions(pred,ids);duration=len(ids)*DT/60
    wholly_false=0;repeats=0;missed=0
    for ep in np.unique(ids):
        ix=np.flatnonzero(ids==ep)
        for lo,hi in common.previous.intervals(active[ix]):wholly_false+=int(not truth[ix[lo:hi]].any())
        for lo,hi in common.previous.intervals(truth[ix]):
            carried=int(active[ix[lo]] and not starts[ix[lo]])
            repeats+=max(0,carried+int(starts[ix[lo:hi]].sum())-1)
            missed+=int(not active[ix[lo:hi]].any())
    return dict(prompt_starts=int(starts.sum()),prompt_starts_per_minute=float(starts.sum()/duration),
        false_prompt_starts=int((starts&~truth).sum()),false_prompt_starts_per_minute=float((starts&~truth).sum()/duration),
        wholly_false_sessions=wholly_false,repeated_starts_within_truth_interval=repeats,
        rendered_missed_truth_intervals=missed,rendered_alert_frames=int(active.sum()))


def contact_events(ids,times,geometry):
    events=[]
    for ep in np.unique(ids):
        ix=np.flatnonzero(ids==ep);first=geometry[int(ix[0])]
        for slot,b in enumerate(first['native_bounds']):
            interval=contact_interval(b,float('inf'))
            if interval is None:continue
            contact=interval[0];end=float(times[ix[-1]])
            status='left_censored' if contact<HORIZON-1e-9 else 'right_censored' if contact>end+1e-9 else 'evaluable'
            events.append(dict(episode_id=str(ep),actor_index=slot,contact_time_s=contact,status=status))
    return events


def timing_metrics(pred,ids,times,events):
    active,starts=sessions(pred,ids);counts=dict(evaluable=0,left_censored=0,right_censored=0,timely=0,late=0,missed=0,
        timely_fresh_session=0,timely_carried_session=0)
    details=[]
    for e in events:
        status=e['status']
        if status!='evaluable':counts[status]+=1;continue
        counts['evaluable']+=1;t=e['contact_time_s'];same=ids==e['episode_id']
        window=same&(times>=t-HORIZON-1e-9)&(times<=t+1e-9)
        timely=window&(times<=t-MIN_LEAD+1e-9)
        hits=np.flatnonzero(active&window);timely_hits=np.flatnonzero(active&timely)
        result='timely' if len(timely_hits) else 'late' if len(hits) else 'missed';counts[result]+=1
        fresh=bool((starts&timely).any())
        if result=='timely':counts['timely_fresh_session' if fresh else 'timely_carried_session']+=1
        details.append(dict(**e,result=result,first_in_window_lead_s=float(t-times[hits[0]]) if len(hits) else None,
            fresh_start_in_timely_window=fresh))
    counts['timely_rate']=counts['timely']/counts['evaluable'] if counts['evaluable'] else None
    counts['details']=details
    return counts

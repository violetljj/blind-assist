"""Offline cut recalibration at the reference clear-slot cost for nuisance configurations."""
import numpy as np, glob, os
D='E:/linnan/linnan/artifacts.local/work/real-contract-v5-dev-20261011/nuisance'
cat=np.load('E:/linnan/linnan/artifacts.local/work/cnh-no-bin0-dr-retrain-dev-20261011/hold_grades_notifications.npz')['category']
RA,LT=14.973200828798356,4.625390338985158
def smooth(x):
    out=np.empty_like(x,dtype=float)
    for t in range(x.shape[-2]):
        y=x[...,max(0,t-4):t+1,:]; w=2.**np.arange(y.shape[-2]); out[...,t,:]=(y*w[:,None]).sum(-2)/w.sum()
    return out
def sel(a,c): return a.transpose(0,3,1,2)[cat==c]   # (events, k, frames)
def cut_for(score_clear,count):
    v=np.sort(score_clear.ravel())[::-1]
    return v[count] if count<len(v) else -np.inf   # strictly above count-th -> count slots (ties aside)
def evaluate(m,l,ra,lt):
    A=m>=ra; L=l>=lt; S=A|L; out={}
    for c in ('clear','pass'): out[c+'_slots']=int(sel(S,c).sum()); out[c+'_events']=float(sel(S,c)[...,:11].any(-1).mean())
    ce=sel(S,'contact')[...,:11].any(-1); out['contact_timely']=float(ce.mean())
    out['contact_timely_M3only']=float(sel(A,'contact')[...,:11].any(-1).mean())
    return out
ref=np.load(f'{D}/sim_original.npz'); m0=smooth(ref['m3_raw']); l0=smooth(ref['local_raw'])
nA=int(sel(m0>=RA,'clear').sum()); nL=int(sel((l0>=LT),'clear').sum())
print('reference clear slots raised',nA,'local',nL)
for f in sorted(glob.glob(f'{D}/*.npz')):
    d=np.load(f); m=smooth(d['m3_raw']); l=smooth(d['local_raw'])
    e0=evaluate(m,l,RA,LT)
    ra=max(RA,cut_for(sel(m,'clear'),nA)); lt=max(LT,cut_for(sel(l,'clear'),nL))
    e1=evaluate(m,l,ra,lt)
    print(f"\n== {os.path.basename(f)}")
    print(f"  original cuts : contact timely {e0['contact_timely']:.3f} (M3 {e0['contact_timely_M3only']:.3f}) | clear slots {e0['clear_slots']} events {e0['clear_events']:.3f} | pass events {e0['pass_events']:.3f}")
    print(f"  recalibrated  : raised {ra:.2f} local {lt:.2f} -> contact timely {e1['contact_timely']:.3f} (M3 {e1['contact_timely_M3only']:.3f}) | clear slots {e1['clear_slots']} events {e1['clear_events']:.3f} | pass events {e1['pass_events']:.3f}")

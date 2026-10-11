"""Inject real-like near-bin crosstalk tail + DC nuisance into the consumed no-bin0 DR sim hold.

Amplitudes are in normalized noise units, taken from the dorm pilot (4b centre zones after v4:
bin1 ~2.8, bin2 ~1.2, bin3 ~0.8; DC up to ~0.4). Per scene-replica draws: t1~U(0,3), bins 1/2/3 at
1/.43/.30 of t1 with per-zone factor U(.5,1.5); DC~U(-.2,.5) on bins 1..15. The 'v5 residual'
configuration models an imperfect real-side tail subtraction (t1~U(-.6,.9)) plus per-zone baseline
removal. Frozen default backbone (no_bin0) only; no training, thresholds are recalibrated offline.
Development description on a consumed hold.
"""
import sys,os,time
from pathlib import Path
os.chdir(Path(__file__).resolve().parent); sys.path.insert(0,'.')
import numpy as np
import cnh_no_bin0_dr_eval_20261011 as EV
W=EV.OUT/'data/hold'
OUTD=str(EV.C.ROOT/'artifacts.local/work/real-contract-v5-dev-20261011/nuisance')
os.makedirs(OUTD,exist_ok=True)
with np.load(W/'geometry.npz') as g: sensor,query=g['sensor'],g['public_query']
z0=np.asarray(EV.normalized(W)).astype(np.float32)   # (1536,2,16,8,8,16)
n,k=z0.shape[:2]
rng=np.random.default_rng(20261011)
# per scene-replica draws (constant over frames, like a session property)
t1=rng.uniform(0,3,(n,k,1,1,1)); zone=rng.uniform(.5,1.5,(n,k,1,8,8)); dc=rng.uniform(-.2,.5,(n,k,1,1,1))
def nuisance(z,tail_scale,resid,baseline):
    z=z.copy()
    tt=t1*tail_scale if not resid else rng.uniform(-.6,.9,(n,k,1,1,1))
    for b,f in ((1,1.),(2,.43),(3,.3)): z[...,b]+=tt*f*zone
    z[...,1:]+=dc[...,None]
    if baseline: z=z-np.median(z[...,3:16],axis=-1,keepdims=True)
    return z.astype(np.float16)
configs={'sim_original':z0.astype(np.float16),
         'real_like_nuisance':nuisance(z0,1.,False,False),
         'v5_residual(tail resid + baseline)':nuisance(z0,1.,True,True),
         'baseline_only_on_clean':(z0-np.median(z0[...,3:16],axis=-1,keepdims=True)).astype(np.float16)}
t=time.time()
for name,z in configs.items():
    r=EV.infer(z,sensor,query,arms=('no_bin0',))['no_bin0']
    np.savez_compressed(f"{OUTD}/{name.split('(')[0].strip()}.npz",m3_raw=r['m3_raw'],local_raw=r['local_raw'])
    print('done',name,round(time.time()-t,1),flush=True)

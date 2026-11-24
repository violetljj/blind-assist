"""Closed-form harmonic identities, synthetic head scans and observability controls."""
import hashlib
import json
import numpy as np
import sympy as sp
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from heading import CausalBodyHeadingFilter, angle, corridor_risk, harmonic
from prepare_lamar import HERE, OUT


def rotation(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s],[s,c]])


def symbolic():
    x,y,d,a,t=sp.symbols("x y d a t",real=True)
    e=lambda q:sp.Matrix([sp.cos(q),sp.sin(q)])
    r=sp.Matrix([[sp.cos(y),-sp.sin(y)],[sp.sin(y),sp.cos(y)]])
    assert (r*e(x)-e(x+y)).applyfunc(sp.trigsimp)==sp.zeros(2,1)
    assert (e(x+sp.pi)*sp.cos(t+d+sp.pi)-e(x)*sp.cos(t+d)).applyfunc(sp.trigsimp)==sp.zeros(2,1)
    assert (e(x)*sp.sin(d)*a).diff(x).dot(e(x)).simplify()==0
    ml,ap=sp.symbols("ml ap",positive=True)
    cov=((ap*e(x)*e(x).T)+(ml*e(x+sp.pi/2)*e(x+sp.pi/2).T))/2
    assert (cov.subs(ml,ap)-sp.eye(2)*ap/2).applyfunc(sp.trigsimp)==sp.zeros(2,2)
    return {"SO2_rotation_equivariance":True,"pi_symmetry_free_phases":True,
            "PCA_degenerate_equal_energies":True}


def synthetic():
    rng=np.random.default_rng(2191)
    ts=np.arange(0,54,.02)
    body=np.deg2rad(np.clip((ts-14)*3.,0,47))
    scan=np.deg2rad(45*np.sin(2*np.pi*ts/5)+10*(ts>24)*(ts<32))
    head=body+scan
    walking=(ts>=1)&(ts<39)
    phi=2*np.pi*ts
    g=np.zeros((len(ts),3))
    for i,t in enumerate(ts):
        g[i,:2]=rotation(body[i])@np.array([1.25*np.cos(2*phi[i]-.85),.62*np.cos(phi[i]+.12)])
    g[:,2]=1.4*np.cos(2*phi)
    acc=g*walking[:,None]+rng.normal(0,.05,g.shape)
    f=CausalBodyHeadingFilter()
    records=[]
    for i in range(201,len(ts),5):
        world_from_plane=rotation(head[i])
        past=acc[i-200:i+1].copy()
        past[:,:2]=past[:,:2]@world_from_plane
        gait=harmonic(ts[i-200:i+1],past)
        optical=world_from_plane.T@rotation(body[i]+np.deg2rad(rng.normal(0,2)))[:,0] if walking[i] else None
        direction,k=f.update(gait,optical,world_from_plane)
        pred=angle(world_from_plane@direction)
        err=abs((pred-np.rad2deg(body[i])+180)%360-180)
        records.append((ts[i],bool(walking[i]),abs(np.rad2deg(scan[i])),float(err),float(pred),k))
    walk=[r for r in records if r[1]]
    stopped=[r for r in records if r[0]>44]
    head_median=float(np.median([r[2] for r in walk]))
    mae=float(np.mean([r[3] for r in walk]))
    stop_drift=float(np.ptp(np.unwrap(np.deg2rad([r[4] for r in stopped])))*180/np.pi)
    assert head_median>20 and mae<6 and stop_drift<1e-8
    return {"walking_outputs":len(walk),"head_abs_median_deg":head_median,"fusion_MAE_deg":mae,
            "stop_after_window_flush_outputs":len(stopped),"held_world_drift_deg":stop_drift,
            "transition_max_error_deg":max(r[3] for r in records if 39<=r[0]<=43),
            "orientation_evidence":"Calibrated synthetic half-plane and 2 deg SD causal optical observation",
            "stop_concentration":max(r[5] for r in stopped)}


def check_corridor():
    assert abs(corridor_risk(.73,.2,-.3,0)-.2/np.pi)<1e-8
    x=np.linspace(0,np.pi,80)
    risks=np.array([corridor_risk(xx,.22,0,4.) for xx in x])
    assert np.max(np.diff(risks))<1e-8
    cut=brentq(lambda y:corridor_risk(y,.22,0,4.)-.12,0,np.pi)
    return {"uniform_probability_for_half_width_0p2_rad":.2/np.pi,
            "example_kappa_4_half_width_0p22_cost_threshold_0p12_bearing_cut_rad":cut}


if __name__=="__main__":
    with threadpool_limits(limits=1):
        report={"symbolic":symbolic(),"synthetic":synthetic(),"posterior_corridor":check_corridor(),
                "code_sha256":{p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in ("verify_theory.py","heading.py")}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"theory_verification.json").write_text(json.dumps(report,indent=1,allow_nan=False)+"\n")
    print(report)

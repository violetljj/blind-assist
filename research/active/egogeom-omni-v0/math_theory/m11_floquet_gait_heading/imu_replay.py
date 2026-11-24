"""Time-ordered IMU-only evidence without known pedestrian travel bearing."""
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from heading import harmonic
from prepare_lamar import WORK


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def world_samples(t,a,w):
    # Approximate gyro integration at ~100 Hz for Nymeria; phone uses native samples.
    n = max(1,int(round(.01/np.median(np.diff(t)))))
    nfull=len(t)//n*n
    blocks=t[:nfull].reshape(-1,n)
    valid=(blocks[:,-1]-blocks[:,0])<.05
    t=blocks[:,-1]
    a=a[:nfull].reshape(-1,n,3).mean(axis=1)
    w=w[:nfull].reshape(-1,n,3).mean(axis=1)
    g=Rotation.identity()
    start=t[0]
    align=None
    acc_sum=np.zeros(3)
    output=np.full_like(a,np.nan)
    for i,s in enumerate(t):
        if not valid[i] or (i and not 0<s-t[i-1]<.10):
            g=Rotation.identity(); align=None; acc_sum[:]=0.; start=s
            continue
        if i:
            g=g*Rotation.from_rotvec((w[i]+w[i-1])*.5*(s-t[i-1]))
        a_rot=g.apply(a[i])
        if align is None:
            acc_sum+=a_rot
            if s-start<2: continue
            align=Rotation.align_vectors([[0,0,1]],[acc_sum/np.linalg.norm(acc_sum)])[0]
        output[i]=align.apply(a_rot)
    return t,output


def windows(t,a,w):
    t,acc=world_samples(t,a,w)
    active=[]
    for end in np.arange(t[0]+6,t[-1],4.):
        sl=(t>end-4)&(t<=end)
        if not np.isfinite(acc[sl]).all(): continue
        h=harmonic(t[sl],acc[sl])
        if h is not None:
            active.append({"end":float(end-t[0]),"valid":h["valid"],"frequency":h["frequency"],
                "vertical":h["vertical"],"phase_margin":h["phase_margin"],
                "bearing":h["direction"].tolist()})
    valid=[h for h in active if h["valid"]]
    turns=[np.rad2deg(np.arccos(np.clip(np.dot(x["bearing"],y["bearing"]),-1,1)))
           for x,y in zip(valid,valid[1:]) if y["end"]-x["end"]<4.1]
    return {"admissible_windows":len(active),"gate_open":len(valid),
            "adjacent_valid":len(turns),"adjacent_bearing_change_deg":list(map(float,turns)),
            "windows":active}


def imu_evidence():
    ny=WORK/"cnh-nymeria-imu-causal-dev-20261008/final"
    raw=ny/"raw_right_60s.npz"
    params=ny/"causal_probe.json"
    p=json.loads(params.read_text())["calibration"]
    with np.load(raw) as f:
        valid=f["valid"]
        t=(f["capture_ns"][valid]-f["capture_ns"][0])*1e-9
        a=np.linalg.solve(p["accel_rectification"],(f["accel"][valid]-p["accel_bias"]).T).T
        w=np.linalg.solve(p["gyro_rectification"],(f["gyro"][valid]-p["gyro_bias"]).T).T
    nymeria={"raw_sha256":digest(raw),"calibration_sha256":digest(params),"valid_samples":len(t),**windows(t,a,w)}
    phone=WORK/"phone-new-capture-20261012"
    sessions=json.loads((phone/"import_receipt.json").read_text())["sessions"]
    records={}
    for s in sessions:
        path=phone/"raw"/s/"imu.raw.jsonl"
        data=[]
        with path.open(encoding="utf-8",errors="replace") as stream:
            for line in stream:
                try: row=json.loads(line)
                except ValueError: continue
                if row.get("type")=="imu_frame" and row.get("valid"):
                    data.append(row)
        times=np.array([r["device_us"] for r in data]);t=(times-times[0])*1e-6
        a=np.array([r["accel_g"] for r in data])*9.80665
        w=np.deg2rad([r["gyro_dps"] for r in data])
        records[s]={"raw_sha256":digest(path),"valid_samples":len(data),**windows(t,a,w)}
    return {"nymeria":nymeria,"phone18":records}

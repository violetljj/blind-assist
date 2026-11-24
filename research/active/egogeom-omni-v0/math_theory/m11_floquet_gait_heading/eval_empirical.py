"""Score local causal features first; join NavVis/forward references only afterwards."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

from heading import CausalBodyHeadingFilter, angle, harmonic, unit, visual_foe
from prepare_lamar import DATA, HERE, OUT, ROOT, WORK

HOST = ROOT/"research/active/hardware-bringup/host"
sys.path.insert(0,str(HOST))
from arkit_selective_heading_dev_20261116 import HALF_WIDTH, MIN_SOLID_ANGLE, in_corridor
from lamar_native_corridor_dev_20261116 import DepthSensor, episode_pairs, reliable_direction, walking_plane
from lamar_motion_gate_dev_20261116 import future_yaw, motion_facts

HEAD = np.array([1.,0.])
ARMS = ("head","strict_pose","pose04","FOE","IMU","fusion")


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def summarize(x):
    if not len(x):
        return {"n":0}
    x = np.asarray(x)
    return {"n":len(x),"p50":float(np.median(x)),"p90":float(np.quantile(x,.9)),
            "p95":float(np.quantile(x,.95))}


def rigs(raw):
    transforms = {}
    for line in (raw/"rigs.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        row = [p.strip() for p in line.split(",")]
        q = np.asarray(row[2:6],float)
        transforms[row[1]] = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    row = next(x for x in csv.reader((raw/"sensors.txt").read_text().splitlines(),skipinitialspace=True)
               if x[0]=="hetlf")
    fx, fy, cx, cy = map(float,row[-4:])
    return transforms, np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])


def lamar_acc(path, local, rot):
    # Only the accelerometer low-pass history is needed. Aggregate non-overlapping
    # 20-ms blocks, stamp with their last observed time, and break gaps.
    times, acc = [], []
    bucket = []
    prev = start = None
    with path.open() as f:
        for row in csv.reader(f,skipinitialspace=True):
            if len(row)<5 or row[1]!="accel_sensor":
                continue
            t = int(row[0])
            if prev is None or t-prev>50000 or t<=prev:
                bucket = []
                start = t
            bucket.append([float(v) for v in row[2:5]])
            if t-start>=20000:
                acc.append(np.mean(bucket,axis=0)); times.append(t)
                start=t; bucket=[]
            prev=t
    times, acc = np.array(times), np.asarray(acc)
    pi = local["timestamp_us"].searchsorted(times,side="right")-1
    keep = (pi>=0) & (times-local["timestamp_us"][pi]<=70000)
    rr = local["T_local_rig"][pi[keep],:3,:3] @ rot
    return times[keep], np.einsum("nij,nj->ni",rr,acc[keep]), local["component"][pi[keep]]


def infer_session(native, reverse_gait=False):
    src = native.stem
    inputs = OUT/"lamar_inputs"/src
    manifest = load(inputs/"manifest.json")
    root = DATA/"hololens/CAB"
    raw = root/"raw"/src
    prepared = root/"prepared"/native.name
    with np.load(prepared) as f:
        keys = ("ts","timestamp_us","T_local","T_local_rig","component","local_down","depth_index","depth_timestamp_us")
        loc = {k:f[k] for k in keys}
        sensor = DepthSensor(f["depth_lut"],f["T_depth_to_head"])
    rotations,k = rigs(raw)
    assert sha(inputs/"imu.txt")==manifest["sha256"]["imu.txt"]
    times, accel, comps = lamar_acc(inputs/"imu.txt",loc,rotations["accel_sensor"])
    stamps = loc["timestamp_us"]
    frames = {}
    estimator = CausalBodyHeadingFilter()
    last_time = last_component = None
    times_ms = {k:[] for k in ["imu_estimator","pixel_flow_and_signed_FOE","fusion_only"]}
    with np.load(native) as f:
        rows, images = f["depth_rows"],f["depth_mm"]
        for dr,img in zip(rows,images,strict=True):
            ix = loc["depth_index"][dr]
            if ix<0:
                continue
            plane = walking_plane(loc,ix)
            if plane is None:
                continue
            t = loc["depth_timestamp_us"][dr]
            assert stamps[ix]==t
            if last_time is None or not 0 < t-last_time <= 600000 or loc["component"][ix]!=last_component:
                estimator = CausalBodyHeadingFilter()
            last_time, last_component = t,loc["component"][ix]
            start,end = times.searchsorted([t-4_000_000,t],side="right")
            a = accel[start:end]
            valid_hist = len(a)>0 and (comps[start:end]==loc["component"][ix]).all() and t-times[end-1]<=80000
            t0=perf_counter_ns()
            gait = harmonic(times[start:end]/1e6,a@plane.T) if valid_hist else None
            if reverse_gait and gait is not None:
                gait["direction"] = -gait["direction"]
            times_ms["imu_estimator"].append((perf_counter_ns()-t0)/1e6)
            foe = None
            pair = manifest["pairs"].get(str(dr))
            if pair is not None:
                ids = stamps.searchsorted(pair,side="right")-1
                if ids.min()>=0 and (loc["component"][ids]==loc["component"][ix]).all() and max(pair-stamps[ids])<=70000:
                    cams = loc["T_local_rig"][ids,:3,:3]@rotations["hetlf"]
                    names = [str(s)+".jpg" for s in pair]
                    for name in names:
                        if sha(inputs/name)!=manifest["sha256"][name]:
                            raise ValueError("Gray member SHA mismatch")
                    gray = [cv2.imread(str(inputs/name),0) for name in names]
                    t0=perf_counter_ns()
                    v = visual_foe(*gray,k,cams[1].T@cams[0])
                    times_ms["pixel_flow_and_signed_FOE"].append((perf_counter_ns()-t0)/1e6)
                    if v is not None:
                        horizontal = plane[:2]@cams[1]@v
                        if np.linalg.norm(horizontal)>.3:
                            foe=unit(horizontal)
            strict = reliable_direction(loc,ix,plane)
            facts = motion_facts(loc,ix,plane)
            yaw = np.deg2rad(facts.get("yaw04",0.))
            pose04 = np.array([np.cos(yaw), np.sin(yaw)]) if not set(facts["failed"])&{"timestamp","component","speed"} else HEAD
            t0=perf_counter_ns()
            fusion, concentration = estimator.update(gait,foe,plane[:2].T)
            times_ms["fusion_only"].append((perf_counter_ns()-t0)/1e6)
            choices=(HEAD,strict if strict is not None else HEAD,pose04,foe if foe is not None else HEAD,
                     gait["direction"] if gait and gait["valid"] else HEAD,fusion)
            pose = np.eye(4)
            pose[:3,:3] = plane@loc["T_local"][ix,:3,:3]
            cloud=sensor.cloud(img.astype(np.float32)*.001,pose@sensor.head_from_depth)
            hf={arm:float(angle(v)) for arm,v in zip(ARMS,choices)}
            hits={arm:in_corridor(cloud,np.zeros(3),v,HALF_WIDTH)[0]>=MIN_SOLID_ANGLE
                  for arm,v in zip(ARMS,choices)}
            frames[int(dr)]={"yaw":hf,"hit":hits,"foe_available":foe is not None,
                    "imu_available":bool(gait and gait["valid"]),"pose_gate_open":strict is not None,
                    "cross":(gait["cross_real"]+1j*gait["cross_imag"]) if gait else None,
                    "kappa":concentration}
    outputs={}
    for dr,current in frames.items():
        if dr<2 or not all(dr-j in frames for j in (1,2)):
            continue
        ix=loc["depth_index"][dr-2:dr+1]
        if len(set(loc["component"][ix]))!=1 or np.diff(loc["depth_timestamp_us"][dr-2:dr+1]).max()>600000:
            continue
        outputs[dr]={**current,"hit":{a:np.sum([frames[dr-i]["hit"][a] for i in (2,1,0)],axis=0)>=2 for a in ARMS}}
    return outputs,times_ms,{p.name:sha(p) for p in (inputs/"manifest.json",raw/"rigs.txt",raw/"sensors.txt",prepared,native)
                             if p not in (prepared,native)}|{"prepared_npz":sha(prepared),"native_npz":sha(native)}


def score_visit(visit, preds, old, breakdown):
    src=visit["visit"]
    prepared=DATA/"hololens/CAB/prepared"/(src+".npz")
    with np.load(prepared) as f:
        world, stamps = f["T_wc"], f["timestamp_us"]
    scored={}
    for row in visit["rows"]:
        dr=row["depth_row"]
        old_key=(src,int(row["timestamp"]))
        if dr not in preds or old_key not in old:
            continue
        v=preds[dr]; prior=old[old_key]
        assert v["hit"]["head"].tolist()==prior["baseline"]
        i,j=stamps.searchsorted([int(row["timestamp"]),int(row["future_stamp"])])
        ref=future_yaw(world,i,j)
        assert abs(abs(ref)-row["direction_deg"])<=.001
        err={a:abs((v["yaw"][a]-ref+180.)%360.-180.) for a in ARMS}
        theta=np.deg2rad(ref)
        c=v["cross"]
        physical_phase=float(np.angle(c@np.array([np.cos(theta),np.sin(theta)]))) if c is not None else None
        scored[dr]={"timestamp_us":int(row["timestamp"]),"category":row["category"],"travel":row["travel"],
               "errors":err,"phase_vs_future_travel":physical_phase,"imu":v["imu_available"],
               "foe":v["foe_available"],"pose_gate":v["pose_gate_open"],
               "hit":{a:v["hit"][a].tolist() for a in ARMS}|{"old_veto":prior["candidate"]}}
    positive=[]
    for idx,ep in enumerate(episode_pairs(visit["rows"],"travel_occupied")):
        if not all(r["depth_row"] in scored for r in ep):
            continue
        flags={}
        for a in (*ARMS,"old_veto"):
            flags[a]=any(np.any(np.asarray(r["travel"])==1,where=scored[r["depth_row"]]["hit"][a]) for r in ep)
        b=breakdown[src,idx]
        assert flags["head"]==b["detected"]
        positive.append({"idx":idx,"reason":b["main_reason"],"alerts":flags})
    return list(scored.values()),positive


def summary(rows, eps):
    ret={}
    for arm in (*ARMS,"old_veto"):
        false=[r for r in rows if r["category"]=="head_occupied_travel_empty"]
        clear=[r for r in rows if r["category"]=="other_clear"]
        cases=[p for p in eps if not p["alerts"]["head"]]
        err=[r["errors"][arm] for r in rows] if arm in ARMS else []
        ret[arm]={"deg":summarize(err),"over_90_deg":sum(v>90 for v in err),
                  "occupied_detected":sum(e["alerts"][arm] for e in eps),"occupied_total":len(eps),
                  "misses_rescued":sum(e["alerts"][arm] for e in cases),"misses":len(cases),
                  "dir_rescued":sum(e["alerts"][arm] for e in cases if e["reason"]=="direction_offset"),
                  "dir_misses":sum(e["reason"]=="direction_offset" for e in cases),
                  "formerly_detected_lost":sum(e["alerts"]["head"] and not e["alerts"][arm] for e in eps),
                  "old_30_removed":sum(any(r["hit"]["head"]) and not any(r["hit"][arm]) for r in false),
                  "total_target_alerts":sum(any(r["hit"][arm]) for r in false),"target_times":len(false),
                  "target_new":sum(not any(r["hit"]["head"]) and any(r["hit"][arm]) for r in false),
                  "other_clear_alerts":sum(any(r["hit"][arm]) for r in clear),"other_clear_times":len(clear)}
    return ret


def run():
    from imu_replay import imu_evidence
    cv2.setNumThreads(1)
    inputs=sorted((DATA/"hololens/CAB/native_depth").glob("*.npz"))
    pred={}; timings={}; hashes={}
    for native in inputs:
        pred[native.stem],tt,hashes[native.stem]=infer_session(native)
        for k,v in tt.items(): timings.setdefault(k,[]).extend(v)
        print("causal",native.stem,len(pred[native.stem]),flush=True)
    geom_file=DATA/"screen-hololens/summary.json"
    old_file=DATA/"screen-hololens/causal_comparison.json"
    reasons_file=WORK/"lamar-miss-breakdown-dev-20261117/breakdown.json"
    old={(r["session"],int(r["timestamp_us"])):r for r in load(old_file)["predictions"]}
    reasons={(p["session"],p["episode"]):p for p in load(reasons_file)["episodes"]}
    all_rows=[]; all_eps=[]; per_visit={}; cases={}
    for visit in load(geom_file)["visits"]:
        src=visit["visit"]
        if src not in pred: continue
        rows,eps=score_visit(visit,pred[src],old,reasons)
        per_visit[src]=summary(rows,eps)
        cases[src]={"per_time":rows,"episodes":eps}
        all_rows.extend(rows); all_eps.extend(eps)
    result=summary(all_rows,all_eps)
    assert (len(all_rows),len(all_eps),result["head"]["misses"],result["head"]["dir_misses"],result["head"]["total_target_alerts"])==(456,32,15,7,30)
    phases=[r["phase_vs_future_travel"] for r in all_rows if r["imu"]]
    return {"nominal":result,"per_visit":per_visit,"cases":cases,"predictor_inputs":hashes,
            "paired_times":len(all_rows),"imu_available":len(phases),"FOE_available":sum(r["foe"] for r in all_rows),
            "imu_negative_phase_vs_reference":sum(p<0 for p in phases),
            "common_sensor_support":sum(r["imu"] and r["foe"] for r in all_rows),
            "paired_sensor_heading_errors":{a:summarize([r["errors"][a] for r in all_rows if r["imu"] and r["foe"]]) for a in ARMS},
            "cpu_ms_single_thread":{k:summarize(v) for k,v in timings.items()},
            "weak_imu_recordings":imu_evidence(),
            "raw_scoring_inputs":{p.name:sha(p) for p in (geom_file,old_file,reasons_file)},
            "code_sha256":{p.name:sha(p) for p in [*HERE.glob("*.py"),
                 *(HOST/k for k in ("lamar_native_corridor_dev_20261116.py","lamar_motion_gate_dev_20261116.py",
                                    "arkit_selective_heading_dev_20261116.py"))]},
            "distance_MAE_and_FA_per_walker_km":"not evaluable; sparse sampled reference times, no labelled clear-walk mileage"}


if __name__=="__main__":
    with threadpool_limits(limits=1):
        out=run()
    (OUT/"v5_empirical_report.json").write_text(json.dumps(out,indent=1,allow_nan=False)+"\n",encoding="utf8")
    print(out["nominal"],out["imu_available"],out["FOE_available"],flush=True)

"""Offline synchronized cross-device geometry. No predictor or phone-facing interface."""
import os
for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]="1"

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
M9 = HERE.parent
ROOT = HERE.parents[5]
DATA = ROOT / "artifacts.local/downloads/aria-gen2-pilot-20261117"
ART = ROOT / "artifacts.local/work/egogeom-omni-v0"
OUT = ART / "math_theory" / M9.name / HERE.name
sys.path.insert(0,str(M9 / "aria_surface_witness"))
from reference import Ground, centres, corridor_reference

cv2.setNumThreads(1)
V,U = np.indices((160,160))
EVAL = ART/"math_theory"/M9.name/"aria_alert_support/v5_aria_alert_support.json"
TOL = .15
TIME_GATE_NS = 16_666_667


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def write(p,obj):
    temp=p.with_suffix(".tmp")
    temp.write_text(json.dumps(obj,allow_nan=False,ensure_ascii=False,indent=1),encoding="utf8")
    temp.replace(p)


def q(x):
    a=np.asarray(x)
    return {"n":len(a),**dict(zip(("min","p50","p95","max"),map(float,np.percentile(a,[0,50,95,100]))))} if len(a) else {"n":0}


class Recording:
    def __init__(self,walk):
        self.walk=walk
        self.base=DATA/walk/"x/depth"
        traj=self.base.parent/"closed_loop_trajectory.csv"
        params=self.base/"pinhole_camera_parameters.json"
        raw=np.loadtxt(traj,delimiter=",",skiprows=1,usecols=[1,2],dtype=np.int64)
        t=raw[:,0]*1000
        utc=raw[:,1]
        assert np.all(np.diff(t)>0) and np.all(np.diff(utc)>=0)
        # A depth exposure falls exactly on a new shared-clock tick in this release.
        self.meta=json.loads(params.read_bytes())
        self.tracking=np.array([v["frameTimestampNs"] for v in self.meta],np.int64)
        loc=np.searchsorted(t,self.tracking)
        assert np.array_equal(t[loc],self.tracking) and np.all(utc[loc]!=utc[loc-1])
        self.utc=utc[loc]
        assert np.all(np.diff(self.utc)>0)
        with traj.open("rb") as f:
            next(f)
            uids={line.split(b",",1)[0].decode() for line in f}
        assert len(uids)==1
        self.graph=uids.pop()
        self.pos=np.array([v["T_world_camera"]["Translation"] for v in self.meta])
        self.rot=Rotation.from_quat([v["T_world_camera"]["QuaternionXYZW"] for v in self.meta]).as_matrix()
        self.k=np.array([v["camera"]["Parameters"] for v in self.meta])
        self.receipt={"graph_uid":self.graph,"n_depth":len(loc),"n_trajectory":len(t),
                      "trajectory_sha256":sha(traj),"metadata_sha256":sha(params),
                      "first_last_tracking_ns":self.tracking[[0,-1]].tolist(),"first_last_shared_ns":self.utc[[0,-1]].tolist(),
                      "clock_tick_step_ms":q(np.diff(self.utc)/1e6),
                      "exact_depth_to_shared_transition_pairs":len(loc)}
        self.hashes={}

    def match(self,stamp):
        i=int(np.searchsorted(self.utc,stamp))
        candidates=[j for j in (i-1,i) if 0<=j<len(self.utc)]
        j=min(candidates,key=lambda j:abs(int(self.utc[j])-int(stamp)))
        dt=int(self.utc[j])-int(stamp)
        return j if abs(dt)<=TIME_GATE_NS else None,dt

    @lru_cache(maxsize=24)
    def frame(self,idx):
        fid=self.meta[idx]["index"]
        data=(self.base/"depth"/f"depth_{fid:08d}.png").read_bytes()
        self.hashes[str(fid)]=hashlib.sha256(data).hexdigest()
        mm=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
        assert mm.shape==(512,512) and mm.dtype==np.uint16
        fx,fy,cx,cy=self.k[idx]*160/512
        ds=cv2.resize(mm,(160,160),interpolation=cv2.INTER_NEAREST).astype(np.float32)/1000
        ok=(ds>.3)&(ds<12)
        z=ds[ok]
        pc=np.stack(((U[ok]-cx)/fx*z,(V[ok]-cy)/fy*z,z),axis=-1)
        xyz=((self.rot[idx]@pc.T).T+self.pos[idx]).astype(np.float32)
        return Frame(mm,xyz)


@dataclass
class Frame:
    mm: np.ndarray
    points: np.ndarray


def classify_rays(world,observer,idx,frame):
    """Project world sites; full 3x3 native patch must agree to assert a surface/free ray."""
    camera=(world-observer.pos[idx])@observer.rot[idx]
    z=camera[:,2]
    fx,fy,cx,cy=observer.k[idx]
    pixel=np.rint(camera[:,:2]/np.maximum(z,1e-7)[:,None]*[fx,fy]+[cx,cy])
    inside=(z>.3)&(z<12)&np.all((pixel>=1)&(pixel<=510),axis=1)
    x,y=np.clip(pixel,1,510).astype(np.int32).T
    low=np.full(len(x),65535,np.uint16)
    high=np.zeros(len(x),np.uint16)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            val=frame.mm[y+dy,x+dx]
            np.minimum(low,val,out=low)
            np.maximum(high,val,out=high)
    valid=inside&(low>300)&(high<12000)
    radial_per_z=np.linalg.norm(camera,axis=1)/np.maximum(z,1e-7)
    lower=(low*.001-z)*radial_per_z
    upper=(high*.001-z)*radial_per_z
    return inside,valid,x+512*y,lower,upper


def view_summary(points,observer,idx,frame):
    # Deterministic 1:16 subsample of the target frame's valid 160x160 depth returns.
    grid=points[::16]
    inside,valid,_,lo,hi=classify_rays(grid,observer,idx,frame)
    return {"source_samples":len(grid),"in_peer_image":int(inside.sum()),"peer_valid":int(valid.sum()),
            "peer_surface_015":int((valid&(lo>=-TOL)&(hi<=TOL)).sum())}


def measure_cross(site,forward,peer,j,frame):
    # Reuse the original ground, voxel and support definitions with peer-only measurements.
    out={"range_m":None,"coverage":None,"free_stencil":0,"free_projectable":0,"peer_near":False,"peer_clear":False}
    if np.sum(frame.points[:,2]<site[2]-.8)<200:
        return out
    ground,_,_,dist,cov=corridor_reference([frame],site,forward)
    out["coverage"]=float(cov)
    if np.isfinite(dist):
        out["range_m"]=float(dist)
    out["peer_near"]=bool(np.isfinite(dist) and cov>=.6)
    # Same 7x3x2 support layout as M9's causal witness. Height comes only from peer points.
    z,x,h=np.meshgrid(np.arange(.5,4.,.5),[-.20,0.,.20],[.45,1.2],indexing="ij")
    side=np.array([forward[1],-forward[0]])
    xy=site[:2]+z.ravel()[:,None]*forward+x.ravel()[:,None]*side
    distance,index=ground.tree.query(xy,distance_upper_bound=.4)
    good=np.isfinite(distance)
    sites=np.column_stack((xy[good],ground.height[index[good]]+h.ravel()[good]))
    inside,valid,_,lo,_=classify_rays(sites,peer,j,frame)
    out["free_projectable"]=int(inside.sum())
    out["free_stencil"]=int((valid & (lo>.30)).sum())
    out["peer_clear"]=bool(not np.isfinite(dist) and cov>=.6 and out["free_stencil"]==42)
    return out


def target_test(cell_keys,peer,j,frame):
    inside,valid,pixel,lo,hi=classify_rays(centres(cell_keys),peer,j,frame)
    result={"n":len(cell_keys),"in_image":int(inside.sum()),"valid":int(valid.sum())}
    for tol,name in ((.15,"015"),(.30,"030")):
        agree=valid&(lo>=-tol)&(hi<=tol)
        beyond=valid&(lo>tol)
        nearer=valid&(hi<-tol)
        n_support=len(np.unique(pixel[agree]))
        possible=len(cell_keys)-int(beyond.sum())
        result[name]={"same_surface_cells":int(agree.sum()),"distinct_pixels":n_support,"beyond_cells":int(beyond.sum()),
                      "occluded_cells":int(nearer.sum()),"supports":n_support>=8,"rejects_cluster":possible<8}
        assert len(cell_keys)>=8 and not(n_support>=8 and possible<8)
    return result


def run_clip(path,who,peer):
    ci=int(path.stem.rsplit("_c",1)[1])
    with np.load(path) as pack:
        original=pack["eval_rows"]
        t=pack["poses_t"].astype(np.float64)
        r=pack["poses_R"]
        heading_available=pack["valid_heading"].astype(bool)
    ids=ci*900+6*np.arange(len(t))
    np.testing.assert_allclose(t,who.pos[ids],rtol=0,atol=1e-5)
    np.testing.assert_allclose(r,who.rot[ids],rtol=0,atol=1e-6)
    rows=[]
    for i,idx in enumerate(ids):
        j,dt=peer.match(who.utc[idx])
        d,c=original[i,:2]
        old="near" if c>=.6 and d<=4 else ("clear" if c>=.6 and d>4 else "unknown")
        row={"i":i,"source_idx":int(who.meta[idx]["index"]),"source_shared_ns":int(who.utc[idx]),"peer_idx":None,
             "class":old,"reference_m":float(d) if np.isfinite(d) else None,"peer":None,"cluster":None}
        rows.append(row)
        if j is None:
            continue
        v=peer.frame(j)
        row.update(peer_idx=int(peer.meta[j]["index"]),peer_shared_ns=int(peer.utc[j]),pair_error_ms=dt/1.e6,
                   baseline_m=float(np.linalg.norm(who.pos[idx]-peer.pos[j])),
                   axis_angle_deg=float(np.degrees(np.arccos(np.clip(who.rot[idx,:,2]@peer.rot[j,:,2],-1,1)))),
                   fov=view_summary(who.frame(int(idx)).points,peer,j,v))
        if not heading_available[i]:
            continue
        fw=t[min(i+2,len(t)-1),:2]-t[max(0,i-2),:2]
        fw/=np.linalg.norm(fw)
        row["peer"]=measure_cross(t[i],fw,peer,j,v)
        if old=="near":
            all_views=[who.frame(int(ids[k])) for k in range(max(0,i-4),min(len(t),i+5))]
            _,_,keys,own,cov=corridor_reference(all_views,t[i],fw)
            assert np.isclose(own,d,atol=1e-5) and np.isclose(cov,c,atol=1e-5)
            row["cluster"]=target_test(keys,peer,j,v)
    return rows


def aggregate(rows):
    cm={c:dict.fromkeys(("near","clear","unknown"),0) for c in ("near","clear","unknown")}
    statuses=Counter()
    fov=Counter()
    for r in rows:
        p=r["peer"] or {}
        state="near" if p.get("peer_near") else ("clear" if p.get("peer_clear") else "unknown")
        cm[r["class"]][state]+=1
        if r["peer_idx"] is not None:
            fov.update(r["fov"])
        if r["class"]=="near" and p.get("peer_near"):
            statuses["same_distance_055"]+=abs(r["reference_m"]-p["range_m"])<=.55
        if r["class"]=="near" and r["cluster"]:
            statuses["any_cluster_in_peer"]+=r["cluster"]["in_image"]>0
            for label in ("015","030"):
                for mode in ("supports","rejects_cluster"):
                    statuses[label+"_"+mode]+=r["cluster"][label][mode]
        if p.get("free_stencil")==42:
            statuses["complete_free_stencil"]+=1
    pairs=[r for r in rows if r["peer_idx"] is not None]
    return dict(n=len(rows),time_matched=len(pairs),class_matrix=cm,counts=dict(statuses),fov_pooled=dict(fov),
                baseline_m=q([r["baseline_m"] for r in pairs]),dt_abs_ms=q([abs(r["pair_error_ms"]) for r in pairs]),
                optical_axis_angle_deg=q([r["axis_angle_deg"] for r in pairs]))


def event_stats(rows,archive):
    out=[]
    for e in archive:
        ids=e["timely_hazard_frames"]
        if not ids:
            continue
        near=[rows[e["clip"]][i] for i in ids]
        assert all(r["class"]=="near" for r in near)
        flags={}
        for label in ("015","030"):
            sup=[bool(r["cluster"] and r["cluster"][label]["supports"]) for r in near]
            rej=[bool(r["cluster"] and r["cluster"][label]["rejects_cluster"]) for r in near]
            flags[label]={"supports_any":any(sup),"rejects_every":all(rej),"any_reject":any(rej)}
        def allowed(r):
            p=r["peer"] or {}
            return p.get("peer_near") and p["range_m"]>=e["dreq_m"]
        out.append({"clip":e["clip"],"walker":e["walker"],"start":e["event_start"],"ids":ids,
                    "cluster":flags,"peer_near_any":any((r["peer"] or {}).get("peer_near") for r in near),
                    "peer_timing_any":any(allowed(r) for r in near),
                    "same_distance_055_any":any((r["peer"] or {}).get("peer_near") and
                         abs(r["peer"]["range_m"]-r["reference_m"])<=.55 for r in near),
                    "peer_clear_every":all((r["peer"] or {}).get("peer_clear") for r in near)})
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    recordings=[Recording("walk_"+str(w)) for w in (0,1)]
    assert recordings[0].graph==recordings[1].graph
    pack_root=ART/"zeroshot_eval/aria_g2_20clips"
    packs=sorted(pack_root.glob("*.npz"))
    old=json.loads(EVAL.read_bytes())
    assert len(packs)==19
    all_rows={}
    for path in packs:
        assert sha(path)==old["truth_pack_sha256"][path.name]
        w=int(path.stem.split("_")[2])
        all_rows[path.stem]=run_clip(path,recordings[w],recordings[1-w])
        print(path.stem,flush=True)
    events=event_stats(all_rows,old["results"]["v4"]["events"])
    assert len(events)==129
    aggregate_by={}
    for kind in ("all","walk_0","walk_1"):
        rows=[r for name,v in all_rows.items() if kind=="all" or kind in name for r in v]
        aggregate_by[kind]=aggregate(rows)
    assert [sum(aggregate_by["all"]["class_matrix"][c].values()) for c in ("near","clear","unknown")]==[840,954,1056]
    result={"clock":{w.walk:w.receipt for w in recordings},"frames":all_rows,"summary":aggregate_by,"events":events,
            "source_depth_image_sha256":{w.walk:w.hashes for w in recordings},
            "input_ledger_sha256":sha(EVAL),"packs_sha256":old["truth_pack_sha256"],
            "code_sha256":{p.name:sha(p) for p in [Path(__file__),M9/"aria_surface_witness/reference.py"]}}
    write(OUT/"audit.json",result)
    print(aggregate_by["all"],flush=True)


if __name__=="__main__":
    main()

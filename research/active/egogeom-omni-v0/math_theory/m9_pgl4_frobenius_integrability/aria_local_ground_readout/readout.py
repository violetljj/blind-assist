"""Three matched Aria readouts with frozen per-clip folds and shared-clock exclusion."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import csv
import hashlib
import json
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
M9=HERE.parent
MATH=M9.parent
ART=MATH.parents[3]/"artifacts.local"
DATA=ART/"work/egogeom-omni-v0"
PACK=DATA/"zeroshot_eval/aria_g2_20clips"
STREAM=DATA/"math_theory/v41_stream/aria_gen2"
RAW=ART/"downloads/aria-gen2-pilot-20261117"
OUT=DATA/"math_theory"/M9.name/HERE.name
sys.path.insert(0,str(M9/"aria_surface_witness"))
from reference import Ground, centres, keys3, NATIVE_U, NATIVE_V

cv2.setNumThreads(1)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=1,allow_nan=False)+"\n",encoding="utf8")


def clock_map(path,timestamps):
    exposed={int(ts):i for i,ts in enumerate(timestamps)}
    shared=np.full(len(exposed),-1,dtype=np.int64)
    graphs=set()
    with path.open(newline="",encoding="utf8") as f:
        for r in csv.DictReader(f):
            i=exposed.get(int(r["tracking_timestamp_us"])*1000)
            if i is not None:
                assert shared[i]<0
                graphs.add(r["graph_uid"])
                shared[i]=int(r["utc_timestamp_ns"])
    # Each exposure has its own exact row, independent of repeated UTC on intermediary IMU ticks.
    assert len(graphs)==1 and np.all(shared>=0) and np.all(np.diff(shared)>0)
    return shared,next(iter(graphs)),0


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    folders={k:OUT/k for k in ("input","ratios","output")}
    for f in folders.values():
        f.mkdir(exist_ok=True)
    clips={}
    owners={}
    code_identity={p.name:sha(p) for p in [HERE/"PROTOCOL.md",Path(__file__),M9/"aria_surface_witness/reference.py"]}
    for walk in ("walk_0","walk_1"):
        base=RAW/walk/"x/depth"
        param=base/"pinhole_camera_parameters.json"
        trajectory=base.parent/"closed_loop_trajectory.csv"
        pose=json.loads(param.read_text(encoding="utf8"))[::6]
        stems=sorted(STREAM.glob("aria_"+walk+"_c*.npz"))
        used=pose[:150*len(stems)]
        stamps=np.array([p["frameTimestampNs"] for p in used],np.int64)
        utc,uid,lag=clock_map(trajectory,stamps)
        owners[walk]=dict(graph_uid=uid,max_time_transfer_lag_ns=lag,calibration_sha256=sha(param),clock_sha256=sha(trajectory))
        for ci,file in enumerate(stems):
            name=file.stem
            assert int(name.rsplit("_c",1)[1])==ci
            sl=slice(ci*150,(ci+1)*150)
            ms=used[sl]; ut=utc[sl]
            with np.load(file,allow_pickle=False) as p:
                model=p["depth_s4"].reshape(150,-1)
                r=p["poses_R"]; t=p["poses_t"]; fw=p["strict_fwd"][:,:2].astype(float)
            np.testing.assert_allclose(t,[m["T_world_camera"]["Translation"] for m in ms],atol=1e-5,rtol=0)
            from scipy.spatial.transform import Rotation
            expected=Rotation.from_quat([m["T_world_camera"]["QuaternionXYZW"] for m in ms]).as_matrix()
            np.testing.assert_allclose(r,expected,atol=1e-6,rtol=0)
            fw/=np.linalg.norm(fw,axis=1,keepdims=True)
            inputs=folders["input"]/(name+".npz")
            # The deployable observation side contains intrinsics, poses, headings, and frozen predictions only.
            np.savez(inputs,model=model,rot=r,pos=t,fw=fw,K=np.array([m["camera"]["Parameters"] for m in ms]))
            stereo=np.empty((150,72*96),np.uint16)
            hashes=hashlib.sha256()
            for i,m in enumerate(ms):
                path=base/"depth"/("depth_%08d.png"%m["index"])
                buf=path.read_bytes()
                hashes.update(hashlib.sha256(buf).digest())
                mm=cv2.imdecode(np.frombuffer(buf,np.uint8),cv2.IMREAD_UNCHANGED)
                assert mm.shape==(512,512) and mm.dtype==np.uint16
                stereo[i]=mm[NATIVE_V,NATIVE_U]
            oracle=folders["input"]/(name+"_stereo.npy")
            np.save(oracle,stereo)
            z=stereo.astype(np.float32)/1000.
            ok=(z>.3)&(z<12)&np.isfinite(model)&(model>0)
            ratios=(z[ok]/model[ok]).astype(np.float32)
            assert np.all(np.isfinite(ratios)) and np.min(ratios)>0
            np.save(folders["ratios"]/(name+".npy"),ratios)
            dt=int(np.median(np.diff(ut)))
            assert 190_000_000<dt<210_000_000
            clips[name]=dict(walk=walk,utc_start_ns=int(ut[0]),utc_stop_ns=int(ut[-1])+dt,
                             n=150,ratio_n=len(ratios),stream_sha256=sha(file),input_sha256=sha(inputs),
                             stereo_sha256=sha(oracle),ordered_png_sha256_digest=hashes.hexdigest())
    assert len(clips)==19 and owners["walk_0"]["graph_uid"]==owners["walk_1"]["graph_uid"]
    folds={}
    guard=30_000_000_000
    for name,q in clips.items():
        sources=[c for c,p in clips.items() if p["walk"]!=q["walk"] and
                 (p["utc_stop_ns"]<=q["utc_start_ns"]-guard or p["utc_start_ns"]>=q["utc_stop_ns"]+guard)]
        assert len(sources)>=3
        # Exact pooled pixel median. Imported only for this fold's other-wearer training records.
        values=np.concatenate([np.load(folders["ratios"]/(c+".npy"),mmap_mode="r") for c in sources])
        scale=float(np.median(values,overwrite_input=True))
        del values
        tag=scale.hex().replace(".","_").replace("+","p")
        folds[name]=dict(train=sources,scale=scale,scale_key=tag,
                         n_train_pixels=sum(clips[c]["ratio_n"] for c in sources),
                         min_gap_s=min(max(clips[c]["utc_start_ns"]-q["utc_stop_ns"],
                                           q["utc_start_ns"]-clips[c]["utc_stop_ns"])/1e9 for c in sources))
    dump(OUT/"inventory.json",dict(clips=clips,walks=owners,folds=folds,hash=code_identity))
    print("Folds",[(k,len(v["train"]),round(v["scale"],3)) for k,v in folds.items()],flush=True)


def corridor(cloud,position,fw):
    # Production local ground and voxel geometry; evaluate one heading and one history only.
    if np.count_nonzero(cloud[:,2]<position[2]-.8)<200:
        return np.nan,np.nan,0
    ground=Ground(cloud,position)
    rel=cloud[:,:2]-position[:2]
    z=rel@fw
    side=rel[:,0]*fw[1]-rel[:,1]*fw[0]
    bounds=(abs(side)<=.35)&(z>=.25)&(z<=4.05)
    p=cloud[bounds]
    height=ground.above(p)
    indices=np.unique(keys3(p[(height>=.15)&(height<=1.90)]))
    xyz=centres(indices)[:,:2]-position[:2]
    z=xyz@fw
    side=xyz[:,0]*fw[1]-xyz[:,1]*fw[0]
    z=np.sort(z[(abs(side)<=.30)&(z>=.30)&(z<=4.)])
    mass=np.searchsorted(z,z+.30,side="right")-np.arange(len(z))
    hit=np.flatnonzero(mass>=8)
    d=float(z[hit[0]]) if len(hit) else np.inf
    return d,ground.coverage(position,fw,d),int(mass.max(initial=0))


def readout(name,arm,scale=1.):
    tic=time.perf_counter()
    is_stereo=arm=="stereo"
    folder=OUT/"output"/arm
    folder.mkdir(parents=True,exist_ok=True)
    dest=folder/(name+".npz")
    cached = dest.exists()
    if cached:
        with np.load(dest) as result:
            if result["scale"] != scale:
                raise ValueError(f"Scale mismatch for {dest}")
            if str(result["code_sha256"]) == sha(Path(__file__)):
                return
    source=OUT/"input"/(name+".npz")
    with np.load(source) as a:
        r,t,fw,K=[a[k] for k in ("rot","pos","fw","K")]
        zsrc=(np.load(OUT/"input"/(name+"_stereo.npy")).astype(np.float32)/1000 if is_stereo else a["model"])
    times=[]; history=deque(maxlen=5); results=[]
    for i,z0 in enumerate(zsrc):
        tick=time.perf_counter_ns()
        z=z0.astype(np.float64)*scale
        ok=np.isfinite(z)&(z>.30)&(z<12.)
        fx,fy,cx,cy=K[i]
        points=np.stack(((NATIVE_U[ok]-cx)*z[ok]/fx,(NATIVE_V[ok]-cy)*z[ok]/fy,z[ok]),axis=1)
        history.append((points@r[i].T+t[i]).astype(np.float32))
        results.append(corridor(np.concatenate(history),t[i].astype(float),fw[i]))
        times.append((time.perf_counter_ns()-tick)/1e6)
    if cached:
        # Git normalizes line endings. Verify the numeric replay before reusing the original receipt.
        with np.load(dest) as previous:
            np.testing.assert_equal(np.asarray(results),previous["readout"])
        return
    np.savez(dest,readout=np.asarray(results),time_ms=np.asarray(times),scale=scale,code_sha256=sha(Path(__file__)))
    print(arm,name,round(time.perf_counter()-tic,2),flush=True)


def replay(which):
    data=json.loads((OUT/"inventory.json").read_text())
    if which in ("all","raw"):
        for name in data["clips"]:
            for arm in ("model","stereo"):
                readout(name,arm)
    if which in ("all","scaled"):
        for n,f in data["folds"].items():
            for name in [n,*f["train"]]:
                readout(name,f["scale_key"],f["scale"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","replay"])
    parser.add_argument("--which",choices=["all","raw","scaled"],default="all")
    a=parser.parse_args()
    with threadpool_limits(1):
        if a.action=="prepare":
            prepare()
        else:
            replay(a.which)

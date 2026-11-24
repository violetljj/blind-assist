"""Evaluator-only Aria factorial reference reconstruction. No prediction API."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[name]="1"
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
M9=HERE.parent
ROUTE=M9.parents[1]
REPO=ROUTE.parents[2]
ART=REPO/"artifacts.local/work/egogeom-omni-v0"
OUT=ART/"math_theory"/M9.name/HERE.name
WITNESS=M9/"aria_surface_witness"
sys.path.insert(0,str(WITNESS))
from reference import Ground, centres, keys3, read_native

STAGES=("centered_full","strict_full","centered_past","strict_past","strict_past_points_full_ground")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def evaluate_slice(pw,n_past,position,headings):
    """One ground fit, then exactly the frozen cell and voxel rules per heading.

    Prefix sidecar drops the future obstacle samples but keeps this same fitted ground.
    """
    out = np.full((len(headings),2,2),np.nan)
    if np.count_nonzero(pw[:,2]<position[2]-.8)<200:
        return out
    ground = Ground(pw,position)
    rel = pw[:,:2]-position[:2]
    take=np.zeros(len(pw),dtype=bool)
    for fw in headings:
        distance=rel@fw
        side=rel[:,0]*fw[1]-rel[:,1]*fw[0]
        take|=(abs(side)<=.35)&(distance>=.25)&(distance<=4.05)
    selected=np.flatnonzero(take)
    h=ground.above(pw[selected])
    selected=selected[(h>=.15)&(h<=1.90)]
    keys=keys3(pw[selected])
    for j,vox in enumerate((np.unique(keys),np.unique(keys[selected<n_past]))):
        points=centres(vox)[:,:2]-position[:2]
        for i,fw in enumerate(headings):
            z=points@fw
            side=points[:,0]*fw[1]-points[:,1]*fw[0]
            values=np.sort(z[(abs(side)<=.30)&(z>=.30)&(z<=4.)])
            hits=np.flatnonzero(np.searchsorted(values,values+.30,side="right")-np.arange(len(values))>=8)
            d=float(values[hits[0]]) if len(hits) else np.inf
            out[i,j]=(d,ground.coverage(position,fw,d))
    return out


def rebuild(path):
    t0=perf_counter()
    clip=path.stem
    walk,number=re.fullmatch(r"aria_(walk_[01])_c(\d+)",clip).groups()
    base=REPO/"artifacts.local/downloads/aria-gen2-pilot-20261117"/walk/"x/depth"
    params=base/"pinhole_camera_parameters.json"
    start=150*int(number)
    metas=json.loads(params.read_text())[::6][start:start+150]
    stream=ART/"math_theory/v4_stream/aria_gen2"/path.name
    with np.load(path,allow_pickle=False) as f:
        rot=f["poses_R"]; pos=f["poses_t"].astype(float)
        allowed=f["valid_heading"].astype(bool); old=f["eval_rows"]
    with np.load(stream,allow_pickle=False) as f:
        strict=f["strict_fwd"][:,:2].astype(float)
        assert np.array_equal(rot,f["poses_R"]) and np.array_equal(pos,f["poses_t"])
    strict/=np.linalg.norm(strict,axis=1,keepdims=True)
    # Criteria match pack_aria_g2_local.py, including clip starts.
    n=len(metas)
    assert n==len(rot)==150
    ii=np.arange(n)
    centered_v=pos[np.minimum(n-1,ii+2),:2]-pos[np.maximum(0,ii-2),:2]
    past_v=pos[np.where(ii>=2,ii,2),:2]-pos[np.maximum(0,ii-2),:2]
    center_norm=np.linalg.norm(centered_v,axis=1)
    assert np.array_equal(allowed,(np.linalg.norm(past_v,axis=1)>=.15)&(center_norm>=.15))
    centered=centered_v/np.where(center_norm==0,1,center_norm)[:,None]
    angles=np.degrees(np.arccos(np.clip(np.sum(centered*strict,axis=1),-1.,1.)))
    seg=np.r_[0.,np.linalg.norm(np.diff(pos[:,:2],axis=0),axis=1)]
    rows={s:np.full_like(old,np.nan) for s in STAGES}
    cache={};depth_hashes={}
    for i in np.flatnonzero(allowed):
        lo,hi=max(0,i-4),min(n,i+5)
        for j in list(cache):
            if j<lo: del cache[j]
        for j in range(lo,hi):
            if j not in cache:
                frame=read_native(base,metas[j],rot[j],pos[j])
                cache[j]=frame.points
                depth_hashes[str(metas[j]["index"])]=frame.digest
        pw=np.concatenate([cache[j] for j in range(lo,hi)])
        n_past=sum(len(cache[j]) for j in range(lo,i+1))
        dirs=(centered[i],strict[i])
        full=evaluate_slice(pw,n_past,pos[i],dirs)
        past=evaluate_slice(pw[:n_past],n_past,pos[i],dirs)
        for name,result in zip(STAGES,(full[0,0],full[1,0],past[0,0],past[1,0],full[1,1]),strict=True):
            if not np.isnan(result[1]):
                rows[name][i]=[*result,0.,seg[i]]
    replay=rows["centered_full"]
    errmask=np.isfinite(old)&np.isfinite(replay)
    max_error=float(np.max(np.abs(old[errmask]-replay[errmask]))) if errmask.any() else 0.
    np.testing.assert_allclose(old,replay,rtol=0,atol=1e-8,equal_nan=True,err_msg=clip)
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/(clip+".npz")
    np.savez_compressed(target,allowed=allowed,heading_angle_deg=angles,frozen=old,**rows)
    receipt={"clip":clip,"walk":walk,"raw_native_depth_sha256_by_index":depth_hashes,"native_parameters_sha256":sha(params),
             "frozen_pack_sha256":sha(path),"output_sha256":sha(target),"max_abs_packed_replay_error":max_error,
             "strict_fwd_array_sha256":hashlib.sha256(strict.tobytes()).hexdigest(),
             "production_pack_generator_sha256":sha(ROUTE.parent/"egoclear-v0/pack_aria_g2_local.py"),
             "geometry_helper_sha256":sha(WITNESS/"reference.py"),"rebuild_sha256":sha(Path(__file__))}
    (OUT/(clip+".json")).write_text(json.dumps(receipt,indent=1)+"\n",encoding="utf-8")
    print(clip,int(allowed.sum()),max_error,round(perf_counter()-t0,1),"s",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip",action="append",help="Optional exact frozen clip name; default all 19.")
    args=parser.parse_args()
    src=sorted((ART/"zeroshot_eval/aria_g2_20clips").glob("*.npz"))
    assert len(src)==19
    selected=[p for p in src if not args.clip or p.stem in args.clip]
    if args.clip and len(selected)!=len(set(args.clip)):
        raise ValueError("Unknown clip")
    with threadpool_limits(limits=1):
        for path in selected: rebuild(path)

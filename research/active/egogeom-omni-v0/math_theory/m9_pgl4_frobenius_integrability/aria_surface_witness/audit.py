"""Same-source stereo localization of immutable Aria six-view cluster members."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from reference import read_native, corridor_reference, keys3, NATIVE_U, NATIVE_V

HERE=Path(__file__).resolve().parent
SIBLING=HERE.parent/"aria_alert_support"
sys.path.insert(0,str(SIBLING))
from predict import ART, dump, sha
from support import axes,world_rays

OUT=ART/"math_theory"/HERE.parent.name/HERE.name
OLD=OUT.parent/"aria_alert_support"
NATIVE=ART.parents[1]/"downloads/aria-gen2-pilot-20261117"
FIELDS=("predicted","stereo_valid","axial","causal_corridor","causal_corridor_height","same_slab",
        "same_slab_axial","both_corridors_slab_axial","target_voxel_1","target_voxel_8","full_witness")
SLICE_FIELDS=("source_pixel_s4","native_pixel_512","pred_axial_mm","stereo_axial_mm","flags")
SLACK=.30


def counts(sorted_f,mask):
    prefix=np.r_[0,np.cumsum(mask)]
    return prefix[np.searchsorted(sorted_f,sorted_f+SLACK,side="right")]-prefix[np.arange(len(sorted_f))]


def two_ranges(s,other,mask):
    lo=np.maximum(s,other)-SLACK
    hi=np.minimum(s,other)
    ok=mask & (lo<=hi+1e-12)
    return lo[ok]-1e-12,hi[ok]+1e-12


def cover(s,lo,hi):
    return np.searchsorted(np.sort(lo),s,side="right")-np.searchsorted(np.sort(hi),s,side="left")


def distinct_coverage(s,other,mask,voxel):
    # Each distinct target voxel contributes at most once for any cluster starting at s.
    start,end=[],[]
    for code in np.unique(voxel[mask]):
        lo,hi=two_ranges(s,other,mask & (voxel==code))
        if not len(lo):
            continue
        order=np.argsort(lo)
        lo,hi=lo[order],hi[order]
        merged=np.maximum.accumulate(hi)
        br=np.flatnonzero(lo[1:]>merged[:-1])+1
        start.extend(lo[np.r_[0,br]])
        end.extend(merged[np.r_[br-1,len(lo)-1]])
    return cover(s,start,end)


def pred_mask(dep,K,R,t,fw):
    ray=world_rays(dep.shape,K,R)
    zz=dep.ravel()
    xyz=ray*zz[:,None]
    forward,side=axes(R,fw)
    f,lat=xyz[:,:2]@forward,xyz[:,:2]@side
    valid=np.isfinite(zz)&(zz>.25)&(zz<10.)
    strip=valid&(np.abs(lat)<=1.2)&(f>=.8)&(f<=4.5)
    floor=np.percentile(xyz[strip,2],8) if strip.sum()>=40 else -1.5
    ix=np.flatnonzero(valid&(xyz[:,2]>=floor+.15)&(xyz[:,2]<=floor+1.90))
    return ix,(xyz[ix,:2]+t[:2]),zz[ix]


def assess(ids,xy,zpred,native,ground,occupied,targets,origin,fw,tf,D,haz):
    side=np.array([fw[1],-fw[0]])
    rel=xy-origin[:2]
    forward=rel@fw
    keep=(abs(rel@side)<=.30)&(forward>=.30)&(forward<=4.)
    order=np.flatnonzero(keep)[np.argsort(forward[keep],kind="stable")]
    s,idx,z=forward[order],ids[order],zpred[order]
    if len(s)<18:
        return None
    mass=np.searchsorted(s,s+.30,side="right")-np.arange(len(s))
    cand=(mass>=18)&((abs(s-D)<=.55) if haz else True)
    if not cand.any():
        return None
    sl=native.source_range[idx]
    valid=(sl>.3)&(sl<12.)
    axial=valid&(abs(sl-z)<=.30)
    pw=native.source_xyz[idx]
    rel=pw[:,:2]-origin[:2]
    fref=rel@fw
    h=ground.above(pw)
    high=(h>=.15)&(h<=1.90)
    inc=valid&(abs(rel@side)<=.30)&(fref>=.30)&(fref<=4.)
    ftruth=rel@tf
    in_truth=((abs(rel@np.array([tf[1],-tf[0]]))<=.30)&(ftruth>=.30)&(ftruth<=4.))
    mask=inc&high
    tight=mask&axial
    two=cover(s,*two_ranges(s,fref,mask))
    three=cover(s,*two_ranges(s,fref,tight))
    four=cover(s,*two_ranges(s,fref,tight&in_truth))
    voxel=keys3(pw)
    # For clear frames targets are empty by definition. The same geometric ladder through
    # both_corridors_slab_axial still applies, without treating missing targets as evidence.
    touched=tight&in_truth&np.isin(voxel,targets) if haz else np.zeros_like(inc)
    distinct=distinct_coverage(s,fref,touched,voxel)
    stage_counts=[mass,counts(s,valid),counts(s,axial),counts(s,inc),counts(s,mask),two,three,four,distinct,distinct,distinct]
    thresholds=[18]*8+[1,8,8]
    hits={k:bool((cand&(a>=limit)).any()) for k,a,limit in zip(FIELDS,stage_counts,thresholds)}
    hits["full_witness"]=bool((cand&(four>=18)&(distinct>=8)).any())
    # Pick a single representative; every constituent remains addressable by original source pixel.
    jj=np.flatnonzero(cand)
    order=np.lexsort((s[jj],-mass[jj],-four[jj],-distinct[jj],-((four[jj]>=18)&(distinct[jj]>=8)).astype(int)))
    j=jj[order[0]]
    members=(s>=s[j])&(s<=s[j]+.30)
    flags=(valid.astype(np.uint8) | (axial.astype(np.uint8)<<1) | (inc.astype(np.uint8)<<2)
           | (high.astype(np.uint8)<<3) | (((fref>=s[j])&(fref<=s[j]+.30)).astype(np.uint8)<<4)
           | (in_truth.astype(np.uint8)<<5) | (np.isin(voxel,targets).astype(np.uint8)<<6))
    src=idx[members]
    arrays=(src.astype(np.int16),(NATIVE_V[src]*512+NATIVE_U[src]).astype(np.int32),
            np.rint(z[members]*1000).astype(np.int16),np.rint(sl[members]*1000).astype(np.uint16),flags[members])
    evidence={"start_m":float(s[j]),"points":int(mass[j]),"valid_same_pixels":int(stage_counts[1][j]),
              "axial_matches":int(stage_counts[2][j]),"causal_corridor_ground":int(stage_counts[4][j]),
              "strict_same_slab":int(four[j]),"unique_target_voxels":int(distinct[j]),
              "full":bool(four[j]>=18 and distinct[j]>=8)}
    return hits,evidence,arrays


def run(name):
    packed=ART/"zeroshot_eval/aria_g2_20clips"/(name+".npz")
    stream_dir=lambda ver: ART/"math_theory"/(ver+"_stream")/"aria_gen2"/(name+".npz")
    walk=name.removeprefix("aria_").rsplit("_c",1)[0]
    ci=int(name.rsplit("_c",1)[1])
    base=NATIVE/walk/"x/depth"
    params=base/"pinhole_camera_parameters.json"
    metadata=json.loads(params.read_text())[ci*900:(ci+1)*900:6]
    with np.load(packed) as p:
        orig,rows,R,t,K0=[p[k] for k in ("orig_hw","eval_rows","poses_R","poses_t","K_orig")]
    assert list(orig)==[512,512] and len(rows)==len(metadata)==150
    packed_bridge=np.median([r["camera"]["Parameters"] for r in metadata],axis=0)
    np.testing.assert_allclose(K0,packed_bridge,rtol=0,atol=1e-4)
    versions=("v4","v41")
    dp={}
    for ver in versions:
        with np.load(stream_dir(ver)) as p:
            dp[ver]=p["depth_s4"]
            if ver=="v4":
                K,heading=p["K"],p["strict_fwd"]
            else:
                np.testing.assert_array_equal(p["K"],K)
                np.testing.assert_array_equal(p["strict_fwd"],heading)
            np.testing.assert_array_equal(R,p["poses_R"])
            np.testing.assert_array_equal(t,p["poses_t"])
    t=t.astype(float)
    valid=(~np.isnan(rows[:,0]))&(rows[:,1]>=.6)
    ref_frames={}
    source_model={v:{} for v in versions}
    detail={v:[] for v in versions}
    raw={v:{k:[] for k in SLICE_FIELDS} for v in versions}
    offsets={v:[0] for v in versions}
    inputs={}
    checks={"scored":0,"max_distance_error_m":0.,"max_coverage_error":0.,"max_pose_error":0.}
    for i in range(len(t)):
        # Ten raw stereo grids at most, including the label-only future interval and t-5.
        for j in range(max(0,i-5),min(len(t),i+5)):
            if j not in ref_frames:
                native=read_native(base,metadata[j],R[j],t[j])
                ref_frames[j]=native
                inputs[str(metadata[j]["index"])]=native.digest
                checks["max_pose_error"]=max(checks["max_pose_error"],native.pose_error)
        ref_frames.pop(i-6,None)
        for ver in versions:
            source_model[ver].pop(i-6,None)
            source_model[ver][i]=pred_mask(dp[ver][i],K[i],R[i],t[i],heading[i])
        if not valid[i]:
            for v in versions:
                detail[v].append({"frame":i,"reference":"unknown"})
                offsets[v].append(offsets[v][-1])
            continue
        gt_heading=t[min(i+2,len(t)-1),:2]-t[max(0,i-2),:2]
        tf=gt_heading/np.linalg.norm(gt_heading)
        fw,_=axes(R[i],heading[i])
        around=[ref_frames[j] for j in range(max(0,i-4),min(len(t),i+5))]
        ground,occupied,target,Drep,cov=corridor_reference(around,t[i],tf)
        D=rows[i,0]
        assert np.isfinite(D)==np.isfinite(Drep),(name,i,D,Drep)
        err=abs(Drep-D) if np.isfinite(D) else 0.
        checks["scored"]+=1
        checks["max_distance_error_m"]=max(checks["max_distance_error_m"],err)
        checks["max_coverage_error"]=max(checks["max_coverage_error"],abs(cov-rows[i,1]))
        if err>1e-4 or abs(cov-rows[i,1])>1e-8:
            raise ValueError(f"Frozen reference bridge differed: {name} {i} {D} {Drep} {rows[i,1]} {cov}")
        haz=bool(D<=4)
        for v in versions:
            combined={key:False for key in FIELDS}
            representative=None
            ordering=None
            chosen=None
            for j in range(max(0,i-5),i+1):
                res=assess(*source_model[v][j],ref_frames[j],ground,occupied,target,t[i],fw,tf,D,haz)
                if res is None:
                    continue
                flags,one,arrays=res
                for k in flags:
                    combined[k]|=flags[k]
                key=(one["full"],one["unique_target_voxels"],one["strict_same_slab"],one["points"],-one["start_m"],j)
                if ordering is None or key>ordering:
                    ordering,representative,chosen=key,dict(one,source_clip_frame=j,native_frame=metadata[j]["index"]),arrays
            row={"frame":i,"reference":"hazard" if haz else "clear","target_distance_m":float(D) if haz else None,
                 "target_frozen_voxels":len(target),"hits":combined,"representative":representative}
            detail[v].append(row)
            size=0
            if chosen is not None:
                size=len(chosen[0])
                for k,arr in zip(SLICE_FIELDS,chosen):
                    raw[v][k].append(arr)
            offsets[v].append(offsets[v][-1]+size)
    out=OUT/"clips"
    out.mkdir(parents=True,exist_ok=True)
    prefix=out/name
    arrays={}
    for v in versions:
        arrays[v+"_offsets"]=np.array(offsets[v],np.int32)
        for k in SLICE_FIELDS:
            arrays[v+"_"+k]=np.concatenate(raw[v][k]) if raw[v][k] else np.zeros(0)
    np.savez_compressed(prefix.with_suffix(".npz"),**arrays)
    report={"name":name,"src_sha256":sha(packed),"stream_sha256":{v:sha(stream_dir(v)) for v in versions},
            "stereo_png_sha256":inputs,"params_sha256":sha(params),"checks":checks,"rows":detail,
            "pixel_receipt_sha256":sha(prefix.with_suffix(".npz"))}
    dump(prefix.with_suffix(".json"),report)
    print(name,checks,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("clips",nargs="*")
    args=parser.parse_args()
    names=args.clips or [p.stem for p in sorted((ART/"zeroshot_eval/aria_g2_20clips").glob("*.npz"))]
    for name in names:
        run(name)

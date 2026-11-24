"""Summarize all 30 Hz native time matches and the 5 Hz fixed-development audit."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from audit import OUT, Frame, Recording, TIME_GATE_NS, classify_rays, q, sha, write


def check_projection():
    rotation=Rotation.from_euler("yz",[.45,-.2]).as_matrix()
    pos=np.array([3.,-1.,2.])
    rec=SimpleNamespace(pos=pos[None],rot=rotation[None],k=np.array([[310.,308.,255.,250.]]))
    local=np.array([[.3,0.,2.],[-.2,.3,1.],[0.,-.2,3.],[.5,.5,-1.],[4.,0.,1.]])
    points=local@rotation.T+pos
    mm=np.full((512,512),2000,np.uint16)
    view,valid,idx,lo,hi=classify_rays(points,rec,0,Frame(mm,points))
    assert np.array_equal(view,[True,True,True,False,False]) and np.array_equal(view,valid)
    assert abs(lo[0])<1.e-12 and lo[1]>.30 and hi[2]<-.30
    y,x=divmod(idx[0],512)
    mm[y+1,x-1]=0
    assert not classify_rays(points,rec,0,Frame(mm,points))[1][0]
    return "Known transformed plane, occluded/far/behind/outside tests and invalid-neighbour rejection passed."


def full_clock():
    a,b=Recording("walk_0"),Recording("walk_1")
    j=np.searchsorted(b.utc,a.utc).clip(1,len(b.utc)-1)
    j-=abs(b.utc[j-1]-a.utc)<abs(b.utc[j]-a.utc)
    delta=b.utc[j]-a.utc
    paired=abs(delta)<=TIME_GATE_NS
    assert np.all(paired) and len(np.unique(j))==len(a.utc)
    distance=np.linalg.norm(b.pos[j]-a.pos,axis=-1)
    k=int(np.argmin(distance))
    return {"native_matched":int(paired.sum()),"absolute_clock_difference_ms":q(abs(delta)/1e6),
            "camera_baseline_m":q(distance),"closest_native_indices":[int(a.meta[k]["index"]),int(b.meta[j[k]]["index"])],
            "closest_shared_ns":[int(a.utc[k]),int(b.utc[j[k]])],"first_tracking_offset_1_minus_0_ns":int(b.tracking[0]-a.tracking[0]),
            "first_shared_offset_1_minus_0_ns":int(b.utc[0]-a.utc[0]),"shared_clock_samples":len(set(map(int,j)))}


def main():
    p=OUT/"audit.json"
    r=json.loads(p.read_bytes())
    names=list(r["frames"])
    result={"full_rate":full_clock(),"groups":{},"projection_check":check_projection()}
    for group in ("all","walk_0","walk_1"):
        clips={k:rows for k,rows in r["frames"].items() if group=="all" or group in k}
        events=[e for e in r["events"] if e["clip"] in clips]
        unique_slots={(e["clip"],i) for e in events for i in e["ids"]}
        unique_indices=[clips[name][i] for name,i in unique_slots]
        count=Counter()
        for e in events:
            for l in ("015","030"):
                f=e["cluster"][l]
                key="supported" if f["supports_any"] else ("all_rejected" if f["rejects_every"] else "unknown")
                count[l+"_"+key]+=1
                count[l+"_at_least_one_rejected"]+=f["any_reject"]
                count[l+"_mixture_supported_rejected"]+=f["supports_any"] and f["any_reject"]
            for label in ("peer_near_any","peer_timing_any","same_distance_055_any","peer_clear_every"):
                count[label]+=bool(e[label])
        detail={}
        for cls in ("near","clear","unknown"):
            rows=[v for vals in clips.values() for v in vals if v["class"]==cls]
            peers=[v["peer"] for v in rows if v["peer"]]
            overlap=[v["fov"] for v in rows if v.get("fov")]
            detail[cls]={"heading_and_pair":len(peers),
                         "peer_coverage_06":sum(v["coverage"] is not None and v["coverage"]>=.6 for v in peers),
                         "peer_coverage_without_near_or_free_proxy":sum(v["coverage"] is not None and v["coverage"]>=.6 and
                                                        not v["peer_near"] and not v["peer_clear"] for v in peers),
                         "stencil_points_geometrically_visible":q([v["free_projectable"] for v in peers]),
                         "stencil_points_witnessed_free":q([v["free_stencil"] for v in peers]),
                         "fov_world_samples":{k:sum(v[k] for v in overlap) for k in ("source_samples","in_peer_image","peer_valid","peer_surface_015")}}
            if cls=="near":
                detail[cls]["target_cells_no_peer_projection"]=sum(not v["cluster"] or v["cluster"]["in_image"]==0 for v in rows)
                detail[cls]["target_cells_any_free"]=sum(bool(v["cluster"] and v["cluster"]["015"]["beyond_cells"]) for v in rows)
        result["groups"][group]={"frames":r["summary"][group],"events_n":len(events),"events":dict(count),"details":detail,
                  "unique_timely_near":len(unique_indices),"unique_timely_surface_015":sum(bool(v["cluster"] and v["cluster"]["015"]["supports"]) for v in unique_indices),
                  "pack_first_last_shared_s":{k:[v[0]["source_shared_ns"]/1.e9,v[-1]["source_shared_ns"]/1.e9] for k,v in clips.items()}}
    result.update(sha256={"audit.json":sha(p),"summarize.py":sha(Path(__file__))})
    write(OUT/"summary.json",result)
    print("native",result["full_rate"])
    print("frames",result["groups"]["all"]["frames"])
    print("events",result["groups"]["all"]["events"])


if __name__=="__main__":
    main()

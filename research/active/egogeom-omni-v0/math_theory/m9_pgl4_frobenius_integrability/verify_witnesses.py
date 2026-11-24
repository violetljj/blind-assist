"""Replay all threshold witnesses in the frozen corridor reader, independent of selection."""
import json
from pathlib import Path

import numpy as np
from eval_empirical import ART, OUTPUT, corridor_dist_from_depth, sha, store


def main():
    files = {name:OUTPUT/(name+".json") for name in ("oracle_aria","ray_bound","box_bound")}
    primary, rays, boxes = [json.loads(p.read_text()) for p in files.values()]
    box = {(r["clip"],r["frame"]):r for r in boxes["results"]}
    ray = {(r["clip"],r["frame"]):r for r in rays["rows"]}
    by_name = {}
    for row in primary["rows"]:
        by_name.setdefault(row["clip"],[]).append(row)
    counts = {"witnessed":0,"impossible_any_depth":0,"impossible_additional_box":0,"unresolved":0}
    errors, max_diff = [], 0.
    for name,rows in by_name.items():
        with np.load(ART/"math_theory/v4_stream/aria_gen2"/(name+".npz")) as c:
            dep,K,R,fw = (c[k] for k in ("depth_s4","K","poses_R","strict_fwd"))
        for row in rows:
            i, target = row["frame"], row["target_m"]
            key=name,i
            arm, best = min(((a,q) for a,q in row["best"].items() if q["error_m"] is not None),
                            key=lambda item:item[1]["error_m"])
            extra=box.get(key,{})
            if "witness_coefficients" in extra:
                arm="box"
                best={"error_m":abs(extra["witness_readout_m"]-target),"coefficient":extra["witness_coefficients"],
                      "readout_m":extra["witness_readout_m"]}
            errors.append(best["error_m"])
            if best["error_m"]<=.55:
                v,u=np.indices(dep[i].shape,dtype=float)
                x,y=(4*u+.5-K[i,0,2])/K[i,0,0],(4*v+.5-K[i,1,2])/K[i,1,1]
                a,b,bx,by=best["coefficient"]
                disparity=a/dep[i].astype(float)+b+bx*x+by*y
                assert (disparity>0).all()
                full=np.empty((4*len(v),4*v.shape[1]),np.float32)
                full[::4,::4]=1/disparity
                out=corridor_dist_from_depth(full,None,None,K[i],R[i],fw[i],False)[0]
                assert abs(out-target)<=.55 and not extra.get("certified_impossible",False)
                max_diff=max(max_diff,abs(out-best["readout_m"]))
                counts["witnessed"]+=1
            elif ray[key]["unrestricted_support_upper"]<18:
                counts["impossible_any_depth"]+=1
            elif extra.get("certified_impossible",False):
                counts["impossible_additional_box"]+=1
            else:
                counts["unresolved"]+=1
    assert sum(counts.values())==840
    result={"counts":counts,"attained_mae_m":float(np.mean(errors)),"max_depth_difference_m":max_diff,
            "input_sha256":{k:sha(v) for k,v in files.items()},"code_sha256":sha(Path(__file__))}
    store(OUTPUT/"witness_audit.json",result)
    print(result)


if __name__=="__main__":
    main()

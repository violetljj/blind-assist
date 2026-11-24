"""Same six-view selector with evaluator stereo depth substituted for model depth."""
import json

from audit import (ART,NATIVE,OLD,OUT,FIELDS,assess,axes,corridor_reference,dump,pred_mask,read_native,sha)
from pathlib import Path
import numpy as np


def main():
    ledger=json.loads((OLD/"v5_aria_alert_support.json").read_text())["results"]["v4"]["events"]
    frames={}
    for src in sorted((ART/"zeroshot_eval/aria_g2_20clips").glob("*.npz")):
        name=src.stem
        a=name.removeprefix("aria_").rsplit("_c",1)
        walk,ci=a[0],int(a[1])
        base=NATIVE/walk/"x/depth"
        meta=json.loads((base/"pinhole_camera_parameters.json").read_text())[ci*900:(ci+1)*900:6]
        with np.load(src) as p:
            rows,R,t=p["eval_rows"],p["poses_R"],p["poses_t"].astype(float)
        with np.load(ART/"math_theory/v4_stream/aria_gen2"/src.name) as p:
            K,fw=p["K"],p["strict_fwd"]
        ref={}
        model={}
        flags=[]
        for i in range(len(t)):
            for j in range(max(0,i-5),min(len(t),i+5)):
                if j not in ref:
                    ref[j]=read_native(base,meta[j],R[j],t[j])
            ref.pop(i-6,None)
            model.pop(i-6,None)
            model[i]=pred_mask(ref[i].source_range.reshape(72,96),K[i],R[i],t[i],fw[i])
            if not (rows[i,1]>=.6 and not np.isnan(rows[i,0])):
                flags.append({"reference":"unknown"})
                continue
            tf=t[min(i+2,len(t)-1),:2]-t[max(0,i-2),:2]
            tf/=np.linalg.norm(tf)
            forward,_=axes(R[i],fw[i])
            around=[ref[j] for j in range(max(0,i-4),min(len(t),i+5))]
            ground,vox,target,D,cov=corridor_reference(around,t[i],tf)
            assert np.isclose(D,rows[i,0],atol=1e-4) and np.isclose(cov,rows[i,1])
            result={k:False for k in FIELDS}
            for j in range(max(0,i-5),i+1):
                one=assess(*model[j],ref[j],ground,vox,target,t[i],forward,tf,D,D<=4)
                if one:
                    for key in FIELDS:
                        result[key]|=one[0][key]
            flags.append({"reference":"hazard" if D<=4 else "clear","hits":result})
        frames[name]=flags
        print(name,flush=True)
    events=[{"name":r["clip"],"frame":r["event_start"],"walker":r["walker"],
             "hits":{key:any(frames[r["clip"]][i]["hits"][key] for i in r["timely_hazard_frames"])
                     for key in FIELDS}} for r in ledger if r["timely_hazard_frames"]]
    def summarize(rows):
        return {"n":len(rows),**{key:sum(r["hits"][key] for r in rows) for key in FIELDS}}
    summary={typ:summarize([r for rows in frames.values() for r in rows if r["reference"]==typ])
             for typ in ("hazard","clear")}
    summary["events"]=summarize(events)
    report={"summary":summary,"frames":frames,"events":events,
            "code_sha256":sha(Path(__file__)),"model_report_sha256":sha(OUT/"surface_witness_report.json")}
    dump(OUT/"reference_depth_control.json",report)
    print(summary,flush=True)


if __name__=="__main__":
    main()

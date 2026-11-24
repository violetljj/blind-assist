"""Produce observable-only Aria features for both backbones before any event evaluation."""
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter_ns

for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np
from support import CausalRayReadout

HERE=Path(__file__).resolve().parent
MATH=HERE.parents[1]
ART=MATH.parents[3]/"artifacts.local/work/egogeom-omni-v0"
OUT=ART/"math_theory"/HERE.parent.name/HERE.name
M6=MATH/"m6_evt_eprocess_risk"
sys.path.insert(0,str(M6))
from risk import InvariantEProcessBEV


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".partial")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(path)


def main():
    report={"code_sha256":{p.name:sha(p) for p in (Path(__file__),HERE/"support.py",M6/"risk.py")},"versions":{}}
    for version in ("v4","v41"):
        root=ART/"math_theory"/(version+"_stream")
        src=sorted((root/"aria_gen2").glob("*.npz"))
        assert len(src)==19
        out=OUT/version
        out.mkdir(exist_ok=True,parents=True)
        items=[]
        for cache in src:
            with np.load(cache) as obs:
                dep,dh,nrm,ls,K,R,t,fw=[obs[k] for k in ("depth_s4","dh_s4","nrm_s4","ls_s4","K","poses_R","poses_t","strict_fwd")]
                raw,one,bev=[obs[k] for k in ("r_hyb_strict","d_1f_strict","d_bev_strict")]
            engine,alternative=InvariantEProcessBEV(),CausalRayReadout()
            m6scores,values,times=[],[],[]
            for i in range(len(dep)):
                m6scores.append(engine.step(dep[i],dh[i],nrm[i],ls[i],K[i],R[i],t[i],fw[i]))
                start=perf_counter_ns()
                values.append(alternative.step(dep[i],K[i],R[i],t[i],fw[i]))
                times.append((perf_counter_ns()-start)/1.e6)
            new,known,current_cap,history_cap,live_hit=np.asarray(values).T
            dest=out/cache.name
            np.savez_compressed(dest,raw=raw,one=one,bev=bev,m6=m6scores,normalized=new,
                                known=known.astype(bool),current_cap=current_cap,history_cap=history_cap,
                                live_hit=live_hit.astype(bool),time_ms=times)
            items.append({"name":cache.stem,"stream_sha256":sha(cache),"output_sha256":sha(dest)})
        report["versions"][version]={"manifest_sha256":sha(root/"manifest.json"),"clips":items}
        print(version,len(items),flush=True)
    dump(OUT/"predictions.json",report)


if __name__=="__main__":
    main()

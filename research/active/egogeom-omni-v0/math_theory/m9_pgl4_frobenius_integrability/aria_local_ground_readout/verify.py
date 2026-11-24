"""Bridge the archived single-heading readout against the independent nine-frame reconstruction helper."""
import json
import sys
from readout import HERE,M9,OUT,NATIVE_U,NATIVE_V,sha,dump
import numpy as np

sys.path.insert(0,str(M9/"aria_reference_heading"))
from rebuild import evaluate_slice


def main():
    meta=json.loads((OUT/"inventory.json").read_text())
    assert meta["hash"]["PROTOCOL.md"]==sha(HERE/"PROTOCOL.md")
    m=json.loads((OUT/"results.json").read_text())
    checked=0
    maxerr=0.
    for name in meta["clips"]:
        with np.load(OUT/"input"/(name+".npz")) as obs:
            r,pos,k,fw,model=(obs[key] for key in ("rot","pos","K","fw","model"))
        stereo=np.load(OUT/"input"/(name+"_stereo.npy")).astype(np.float32)/1000.
        fold=meta["folds"][name]
        for arm,depths,s in [("model",model,1.),("stereo",stereo,1.),(fold["scale_key"],model,fold["scale"])]:
            p=OUT/"output"/arm/(name+".npz")
            assert sha(p)==m["prediction_sha256"][str(p.relative_to(OUT))]
            with np.load(p) as saved:
                d=saved["readout"]
            for i in (0,4,75,149):
                bins=[]
                for j in range(max(0,i-4),i+1):
                    z=depths[j].astype(float)*s
                    valid=(z>.3)&(z<12.)
                    x,y=np.vstack((NATIVE_U[valid],NATIVE_V[valid]))
                    fx,fy,cx,cy=k[j]
                    rays=np.column_stack(((x-cx)/fx,(y-cy)/fy,np.ones_like(x)))
                    bins.append(((rays*z[valid,None])@r[j].T+pos[j]).astype(np.float32))
                cloud=np.concatenate(bins)
                expected=evaluate_slice(cloud,len(cloud),pos[i].astype(float),[fw[i]])[0,0]
                np.testing.assert_allclose(expected,d[i,:2],equal_nan=True,rtol=0,atol=1e-8)
                good=np.isfinite(expected)&np.isfinite(d[i,:2])
                if good.any(): maxerr=max(maxerr,float(np.max(abs(expected[good]-d[i,:2][good]))))
                checked+=1
    dump(OUT/"readout_bridge.json",{"samples":checked,"max_abs_finite_difference":maxerr,
        "results_sha256":sha(OUT/"results.json"),"independent_helper_sha256":sha(M9/"aria_reference_heading/rebuild.py"),
        "checked_frames_each_clip":[0,4,75,149],"code_sha256":sha(HERE/"verify.py")})
    print("geometry bridge",checked,maxerr)


if __name__=="__main__":
    main()

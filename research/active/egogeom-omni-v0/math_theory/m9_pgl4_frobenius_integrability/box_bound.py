"""Branch-and-bound certificate over a bounded shared four-parameter box.

Refines every unresolved finite-grid miss after the unbounded-ray test.
Ground height includes the frozen 8th-percentile fallback.
"""
from collections import deque
import json
from pathlib import Path

import numpy as np
from eval_empirical import ART, OUTPUT, corridor_dist_from_depth, sha, store

EPS = .0002
ORIGINAL_BOX = np.array([[.4,2.5],[-.15,.15],[-.12,.12],[-.12,.12]])


class RayBox:
    def __init__(self, depth, intr, rotation, heading, target):
        v,u = np.indices(depth.shape,dtype=float)
        def rays(shift):
            return (4*u+shift-intr[0,2])/intr[0,0], (4*v+shift-intr[1,2])/intr[1,1]
        x,y = rays(2.)
        xyz = np.stack((x,y,np.ones_like(x)),axis=-1)@rotation.T
        f = np.array([*heading[:2],0.],float)
        if np.linalg.norm(f)<1e-4:
            f = rotation[:,2].copy()
            f[2] = 0.
        f /= max(np.linalg.norm(f),1e-6)
        self.fw = (xyz@f).ravel()
        self.side = np.abs(xyz@np.array([f[1],-f[0],0.])).ravel()
        self.vert = xyz[...,2].ravel()
        x,y=rays(.5)
        self.M = np.stack((1/depth.astype(float),np.ones_like(x),x,y),axis=-1).reshape(-1,4)
        self.Mabs = np.abs(self.M)
        self.target=target

    def viable(self, box):
        mid=self.M@box.mean(1)
        rad=self.Mabs@(np.diff(box,axis=1).ravel()/2)+EPS
        if np.any(mid+rad<=0):
            return False
        zl=np.maximum(.25-EPS,1/np.maximum(mid+rad,1e-12)-EPS)
        zu=np.minimum(10+EPS,1/np.maximum(mid-rad,1e-12)+EPS)
        ok=zu>=zl
        f0=np.minimum(self.fw*zl,self.fw*zu)-EPS
        f1=np.maximum(self.fw*zl,self.fw*zu)+EPS
        h0=np.minimum(self.vert*zl,self.vert*zu)-EPS
        h1=np.maximum(self.vert*zl,self.vert*zu)+EPS
        # Some rays can be discarded by the frozen depth mask.
        all_depth_valid=(mid-rad>1/(10-EPS)) & (mid+rad<1/(.25+EPS))
        must=all_depth_valid & (self.side*zu<1.2-EPS)&(f0>.8+EPS)&(f1<4.5-EPS)
        can=ok & (self.side*zl<=1.2+EPS)&(f1>=.8-EPS)&(f0<=4.5+EPS)
        n_min,n_max=int(must.sum()),int(can.sum())
        ground_lo=ground_hi=-1.5
        if n_max>=40:
            # Any actual ordered height is no smaller than the same rank in
            # the full possible set. Mandatory heights likewise bound from above.
            k_min=int(np.floor(.08*(max(n_min,40)-1)))
            ground_lo=np.partition(h0[can],k_min)[k_min]
            k_max=int(np.ceil(.08*(n_max-1)))
            ground_hi=(np.partition(h1[must],k_max)[k_max] if n_min>k_max else np.max(h1[can]))
            if n_min<40:
                ground_lo,ground_hi=min(ground_lo,-1.5),max(ground_hi,-1.5)
        ok &= (self.side*zl<=.30+EPS)&(h1>=ground_lo+.15-EPS)&(h0<=ground_hi+1.90+EPS)
        d0=np.maximum(.30-EPS,f0)
        d1=np.minimum(4.+EPS,f1)
        lower=np.maximum(max(.30,self.target-.55)-EPS,d0-.30-EPS)
        upper=np.minimum(min(4.,self.target+.55)+EPS,d1+EPS)
        ok &= (d1>=d0)&(upper>=lower)
        if int(ok.sum())<18:
            return False
        starts,ends=np.sort(lower[ok]),np.sort(upper[ok])
        return np.max(np.arange(1,len(starts)+1)-np.searchsorted(ends,starts,side="left"))>=18

    def search(self, score, target, max_depth=40, max_nodes=12000):
        stack=deque([(ORIGINAL_BOX,0)])
        nodes=unresolved=0
        while stack and nodes<max_nodes:
            box,level=stack.popleft()
            nodes+=1
            if not self.viable(box):
                continue
            coefficient = box.mean(1)
            distance = score(coefficient)
            if np.isfinite(distance) and abs(distance-target)<=.55:
                return {"certified_impossible":False,"witness_coefficients":coefficient.tolist(),
                        "witness_readout_m":distance,"boxes_examined":nodes}
            if level==max_depth:
                unresolved+=1
                continue
            axis=level%4
            center=box[axis].mean()
            left,right=box.copy(),box.copy()
            left[axis,1]=right[axis,0]=center
            stack.extend(((left,level+1),(right,level+1)))
        return {"certified_impossible":not stack and unresolved==0,"boxes_examined":nodes,
                "unresolved_leaves":unresolved,"unvisited_nodes":len(stack),"maximum_depth":max_depth}


def main():
    path=OUTPUT/"ray_bound.json"
    rows=json.loads(path.read_text())["rows"]
    results=[]
    # Check positives bracketing the target range before using the relaxation to exclude.
    oracle=json.loads((OUTPUT/"oracle_aria.json").read_text())["rows"]
    successful=[r for r in oracle if min(a["error_m"] for a in r["best"].values() if a["error_m"] is not None)<=.55]
    for item in successful[::30]:
        with np.load(ART/"math_theory/v4_stream/aria_gen2"/(item["clip"]+".npz")) as obs:
            i=item["frame"]
            model=RayBox(*[obs[k][i] for k in ("depth_s4","K","poses_R","strict_fwd")],item["target_m"])
        witness=min((q for q in item["best"].values() if q["error_m"] is not None),key=lambda q:q["error_m"])
        c=np.array(witness["coefficient"])
        assert model.viable(np.stack((c,c),axis=1))
    for item in rows:
        if item["grid_error_m"]<=.55 or item["box_support_upper"]<18:
            continue
        target=next(r["target_m"] for r in oracle if (r["clip"],r["frame"])==(item["clip"],item["frame"]))
        with np.load(ART/"math_theory/v4_stream/aria_gen2"/(item["clip"]+".npz")) as obs:
            i=item["frame"]
            depth,K,R,fw=[obs[k][i] for k in ("depth_s4","K","poses_R","strict_fwd")]
            model=RayBox(depth,K,R,fw,target)
        full=np.empty((4*depth.shape[0],4*depth.shape[1]),np.float32)
        def score(coefficient):
            disp=model.M@coefficient
            if np.any(disp<=0):
                return np.inf
            full[::4,::4]=(1/disp).reshape(depth.shape)
            return corridor_dist_from_depth(full,None,None,K,R,fw,False)[0]
        results.append({"clip":item["clip"],"frame":i,"scalar_certificate":item["certified_scalar_failure"],
                        **model.search(score,target)})
    store(OUTPUT/"box_bound.json",{"box":ORIGINAL_BOX.tolist(),"ray_bound_sha256":sha(path),
                                  "code_sha256":sha(Path(__file__)),"results":results})
    print("classified",len(results),"excluded",sum(r["certified_impossible"] for r in results),
          "rescued",sum("witness_coefficients" in r for r in results),flush=True)


if __name__=="__main__":
    main()

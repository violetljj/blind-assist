"""Exclude an overly near hypothesis without requiring a unique finite far depth."""
from functools import lru_cache
from itertools import combinations

import numpy as np

from holoset_rgb_contradiction_20261119 import RGBVerifier, match_pair


class RGBNearExcluder(RGBVerifier):
    def __init__(self, views):
        super().__init__(views)
        self.cones=lru_cache(maxsize=56)(self._cones)

    def _cones(self,source,dest):
        a,b=self.views[source],self.views[dest]
        return match_pair(self.image(source),self.image(dest),a.K,b.K,
                          np.linalg.inv(b.c2w)@a.c2w,return_hypotheses=True)

    def _pair(self,source,dest):
        return self.cones(source,dest)[0]

    def classify(self,source_rgb_ts,cutoff_ts,maps,mono):
        _,info=super().classify(source_rgb_ts,cutoff_ts,maps,mono)
        dates=info["witnesses"]
        if info["rgb_scale"] is None:
            return np.zeros(mono.shape,bool),info
        u,v=[np.rint((a-.5)/2).astype(int) for a in maps]
        bits,near,upper=[],[],[]
        for ts in dates:
            metric,support,nearest=self.cones(int(source_rgb_ts),ts)
            h,w=nearest.shape
            y,x=v.clip(0,h-1),u.clip(0,w-1)
            inside=(u>=0)&(u<w)&(v>=0)&(v<h)
            bits.append(support[:,y,x]*inside)
            near.append(nearest[y,x]*inside)
            # Only a resolved view whose entire depth interval is on the near side can veto.
            upper.append(metric[1,y,x]*inside)
        lower=np.zeros(mono.shape,np.float32)
        for i,j in combinations(range(len(dates)),2):
            if max(dates[i],dates[j])<=source_rgb_ts:
                continue
            common=np.any(np.bitwise_and(bits[i],bits[j]),axis=0)
            nearest=np.maximum(near[i],near[j])
            bound=np.divide(1.,nearest,out=np.zeros_like(nearest),where=common&(nearest>0))
            lower=np.maximum(lower,bound)
        pred=info["mono_rgb_m"]
        threshold=pred+np.maximum(.20,.20*pred)
        veto=np.max(upper,axis=0)*threshold>1
        valid=(lower>0)&~veto&np.isfinite(pred)&(pred>0)
        info.update(lower_m=np.where(valid,lower,np.nan), valid=valid)
        return valid & (lower>threshold),info

"""Check the conservative ray-count bound against sampled point clouds."""
import numpy as np
from predict import OUT, dump
from support import at, best, intervals


def main():
    rng=np.random.default_rng(2436)
    fw,side=np.array([1.,0.]),np.array([0.,1.])
    margin_checked=0
    for _ in range(400):
        rays=rng.normal(size=(220,3))
        offset=rng.uniform(-.8,.8,3)
        left,right=intervals(rays,offset,fw,side)
        z=rng.uniform(.251,9.99,len(rays))
        xy=offset[:2]+z[:,None]*rays[:,:2]
        s=rng.uniform(.30,3.7)
        inside=(np.abs(xy@side)<=.30)&(xy@fw>=s)&(xy@fw<=s+.30)
        exact=int(inside.sum())
        upper=int(at(left,right,s))
        assert upper>=exact
        assert best(left,right,s)>=upper
        margin_checked+=exact
    rays=np.tile(np.array([1.,0.,0.]),(18,1))
    current=intervals(-rays,np.zeros(3),fw,side)
    old=intervals(rays,np.array([-.4,0.,0.]),fw,side)
    assert at(*current,2.)==0 and at(*old,2.)==18
    assert at(*intervals(rays,np.array([0.,.31,0.]),fw,side),2.)==0
    dump(OUT/"geometry_verification.json",{"random_trials":400,"sampled_in_corridor_points":margin_checked,
         "facing_away_current":0,"prior_forward_view":18,"parallel_laterally_outside":0})
    print("ray bounds verified",margin_checked)


if __name__=="__main__":
    main()

import json
import math
from pathlib import Path

import cv2
import numpy as np
from metric_scale import MetricScale, View, triangulate


def main(dest=None):
    cv2.setNumThreads(1)
    rng = np.random.default_rng(20261124)
    K = np.array([[200.,0,95.5],[0,190.,71.5],[0,0,1.]])
    angle = math.radians(8)
    R1 = np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
    views = [View(np.empty(0),K,np.eye(3),np.zeros(3)),
             View(np.empty(0),K.copy(),R1,np.array([.3,.06,.02]))]
    pts = rng.uniform([-.5,-.5,2.],[.5,.5,4.],(40,3))
    def project(view):
        cam = (pts-view.t)@view.R
        img = cam@view.K.T
        return img[:,:2]/img[:,2:],cam[:,2]
    p0,_ = project(views[0])
    p1,z1 = project(views[1])
    tri,valid,_ = triangulate(p0,p1,*views)
    error = float(abs(tri[valid]-z1[valid]).max())
    assert valid.all() and error<1.e-9
    pure_rotation = View(np.empty(0),K,R1,np.zeros(3))
    rotation_pixels,_ = project(pure_rotation)
    _,bad,_ = triangulate(p0,rotation_pixels,views[0],pure_rotation)
    assert not bad.any()
    # Sliding parallel to a textured wall, a trajectory with zero depth-to-depth
    # information but valid RGB disparity. The actual metric depth is 4 m.
    full_K = K.copy()
    full_K[:2,:] *= 2
    full_K[:2,2] += .5
    texture = cv2.GaussianBlur(rng.integers(0,256,(144,192),np.uint8),(3,3),.55)
    tests = []
    for true in [.60,1.60]:
        est = MetricScale()
        cases = []
        depth = np.full((72,96),4/true,np.float32)
        ls = np.full_like(depth,-2.)
        for t in range(14):
            dx = .14*t
            mapping = np.array([[1,0,-K[0,0]*dx/4],[0,1,0]],np.float32)
            gray = cv2.warpAffine(texture,mapping,(192,144),borderMode=cv2.BORDER_CONSTANT)
            s,local,_ = est.step(gray,depth,ls,full_K,np.eye(3),np.array([dx,0,0]))
            cases.append(float(s))
            assert np.all(np.isfinite(local))
        assert abs(cases[-1]/true-1)<.08 and est.updates>=8
        tests.append({"truth":true,"estimates":cases,"updated":est.updates})
    if dest is None:
        dest = Path(__file__).resolve().parents[6]/"artifacts.local/work/egogeom-omni-v0/math_theory/m2_lie_group_scale_iekf/rgb_triangulation"
    dest.mkdir(exist_ok=True,parents=True)
    result = {"triangulation_max_abs_error_m":error,"pure_rotation_observations":int(sum(bad)),
              "metric_wall_motion":tests,"opencv":cv2.__version__}
    (dest/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(result)


if __name__=="__main__":
    main()

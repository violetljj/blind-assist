"""Exact compatibility, counterexamples and identifiability under simulations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp

from frobenius import camera_normal, edge_equations

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[4]/"artifacts.local/work/egogeom-omni-v0/math_theory"/HERE.name


def symbolic():
    x, y = sp.symbols("x y", real=True)
    Z = sp.Function("Z")(x,y)
    r = sp.Matrix([x,y,1])
    P = Z*r
    N = P.diff(x).cross(P.diff(y))
    omega = -N[:2,:]/N.dot(r)
    assert (omega - sp.Matrix([Z.diff(x),Z.diff(y)])/Z).applyfunc(sp.simplify)==sp.zeros(2,1)
    assert sp.simplify(omega[1].diff(x)-omega[0].diff(y))==0
    # Domains of the positive near roots are noted in THEORY.md.
    graphs = {"tilted_plane": 2/(1+x/4-y/8), "sphere_center_camera":3/sp.sqrt(1+x*x+y*y),
              "y_axis_cylinder_camera_on_axis":2/sp.sqrt(1+x*x)}
    for graph in graphs.values():
        w = omega.subs(Z,graph).doit()
        assert sp.simplify(w[1].diff(x)-w[0].diff(y))==0
    # A front-facing plane can have any distance, with identically zero curl.
    D = sp.exp(x)
    assert sp.diff(sp.log(D),x)==1
    b0,bx,by,a = sp.symbols("b0 bx by a", real=True, nonzero=True)
    H = sp.eye(4)
    H[3,:] = sp.Matrix([[bx,by,b0,a]])
    Xh = H*sp.Matrix([x*Z,y*Z,Z,1])
    assert sp.simplify(Xh[3]/Xh[2]-(a/Z+b0+bx*x+by*y))==0
    assert H.det()==a
    return {"graphs":list(graphs),"perspective_orthogonality":True,"closed_one_form":True,
            "PGL_action":True,"two_parallel_frontoparallel_planes_rank":2,
            "incompatible_example_curl":0,"incompatible_example_x_residual":1}


def synthetic():
    rng = np.random.default_rng(20261124)
    y,x = np.meshgrid(np.linspace(-.6,.6,51),np.linspace(-.8,.8,63),indexing="ij")
    d = .62+.16*x+.10*y+.1*x*x+.08*y*y-.11*x*y
    dx,dy = .16+.2*x-.11*y,.1+.16*y-.11*x
    normal = np.stack((dx,dy,d-x*dx-y*dy))
    normal /= np.linalg.norm(normal,axis=0)
    errs, with_anchor, noisy, rank = [], [], [], None
    for _ in range(100):
        a = rng.uniform(.5,2.)
        b = rng.uniform([-.15,-.12,-.12],[.15,.12,.12])
        raw = (d-b[0]-b[1]*x-b[2]*y)/a
        A,rhs,w = edge_equations(raw,x,y,normal)
        matrix = A * np.sqrt(w)[:,None]
        s = np.linalg.svd(matrix,compute_uv=False)
        rank = s.tolist()
        def fit(n):
            M,target,wgt = edge_equations(raw,x,y,n)
            return np.linalg.lstsq(M*np.sqrt(wgt)[:,None],target*np.sqrt(wgt),rcond=None)[0]
        for n, scores in ((normal,errs),(normal+.01*rng.normal(size=normal.shape),noisy)):
            bhat = fit(n)
            scores.append(float(np.linalg.norm(bhat-b/a)/max(1e-12,np.linalg.norm(b/a))))
        # One exact external axial-depth anchor at centre fixes a.
        bhat = np.linalg.lstsq(matrix,rhs*np.sqrt(w),rcond=None)[0]
        q = raw + bhat[0] + bhat[1]*x+bhat[2]*y
        ahat = d[25,31]/q[25,31]
        with_anchor.append(np.linalg.norm(np.r_[ahat, ahat*bhat]-np.r_[a,b])/np.linalg.norm(np.r_[a,b]))
    # World down = camera +Y, therefore the level-to-camera map is identity.
    R = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
    np.testing.assert_allclose(camera_normal(normal,R),normal)
    plane = np.zeros_like(normal)
    plane[2] = 1.
    A,b,w = edge_equations(np.ones_like(d),x,y,plane)
    assert np.linalg.matrix_rank(A)==2
    assert max(errs)<.02 and max(with_anchor)<.02
    return {"draws":100,"last_design_singular_values":rank,"relative_error_max_ratios":max(errs),
            "relative_error_max_four_parameters_with_metric_anchor":max(with_anchor),
            "relative_error_median_ratios_normal_noise_sd001":float(np.median(noisy)),
            "frontoparallel_rank":int(np.linalg.matrix_rank(A))}


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    result={"symbolic":symbolic(),"synthetic":synthetic(),
            "code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),HERE/"frobenius.py")}}
    (OUT/"theory_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(result)

"""Symbolic identities, conditional Gaussian bound and rigid image-pair controls."""
import os
for key in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key] = '1'

import json
import math

import cv2
import numpy as np
import sympy as sp
from scipy.stats import norm

from direct import affine_ic, gradients
from eval_empirical import base, decode, digest, HERE, OUT


def identities():
    x,y,X,Y,q,px,py,vx,vy,vz = sp.symbols('x y X Y q px py vx vy vz')
    pos = sp.Matrix([x,y])
    b = vz*pos - sp.Matrix([vx,vy])
    rho = q+px*(x-X)+py*(y-Y)
    flow = rho*b
    sub = {x:X,y:Y}
    J = flow.jacobian(pos).subs(sub)
    b0 = b.subs(sub)
    assert J == q*vz*sp.eye(2)+b0*sp.Matrix([[px,py]])
    comps = [J.trace(), J[1,0]-J[0,1], J[0,0]-J[1,1], J[1,0]+J[0,1]]
    assert sp.simplify((comps[0] - (sp.Matrix([px,py]).dot(flow.subs(sub)))/q)/2 - q*vz)==0
    nx,ny,nz,Z = sp.symbols('nx ny nz Z', nonzero=True)
    disparity=(nx*x+ny*y+nz)/(Z*(nx*X+ny*Y+nz))
    assert sp.simplify(disparity.subs(sub)-1/Z)==0
    assert sp.simplify(sp.diff(disparity,x)/disparity.subs(sub)-nx/(nx*X+ny*Y+nz))==0
    # Frontal plane with axial approach and slanted plane with lateral motion
    # share D=2. Four components separate shear if observable.
    d = comps[0].subs({q:1,X:0,Y:0})
    assert d.subs({vz:1,px:0,py:0})==d.subs({vz:0,px:1,vx:-2,py:0})==2
    return {'D_C_S1_S2':[str(sp.expand(c)) for c in comps],'plane_sign_and_off_axis_denominator':True,
            'lateral_only_positive_divergence':2,'scale_ratio_for_static_if_predicted_depth_is_sZ':'s'}


def render_trials(rng, count=384):
    side=81
    center=np.array([40.,40.])
    K=np.array([[90.,0,40.],[0,90.,40.],[0,0,1.]])
    results={}
    for kind in ('static_slant','approaching','crossing'):
        z_values = rng.uniform(6,9,count)
        hit=0
        velocities=[]
        ss=[]
        missed=0
        depth_false=0
        center_res=[]
        for z in z_values:
            # Stationary, known orientation camera moves 0.15 m axially and 0.03 m right.
            n = np.array([.60,.10,1.]) if kind=='static_slant' else np.array([0.,0.,1.])
            ego=np.array([.03,0.,.15])
            moving={'static_slant':[0.,0.,0.],'approaching':[0.,0.,-1.2],'crossing':[.4,0.,0.]}[kind]
            true_H = K @ (np.eye(3)+np.outer(np.asarray(moving)-ego,n)/z) @ np.linalg.inv(K)
            static_H = K @ (np.eye(3)-np.outer(ego,n)/z) @ np.linalg.inv(K)
            c,J=map_jac(static_H,center)
            true_c,J_true=map_jac(true_H,center)
            texture=cv2.GaussianBlur(rng.normal(127.,55.,(side,side)).astype(np.float32),(0,0),.85)
            clean=cv2.warpPerspective(texture,true_H,(side,side),flags=cv2.INTER_LINEAR)
            noise=1.
            im0=texture # conditional on a known, noise-free template
            im1=clean+rng.normal(0,noise,clean.shape).astype(np.float32)
            # Center supplied to isolate scale observability from correspondence failures.
            ell,sd,valid=affine_ic(im0,im1,gradients(im0),center[None],true_c[None],J[None],8.,False,8)
            center_res.append(float(np.linalg.norm(true_c-c)))
            if not valid[0]:
                missed+=1
                continue
            motion=-(z-.15)*np.expm1(-ell[0])/.2   # approach positive; finite perspective
            real_motion=6. if kind=='approaching' else 0.
            velocities.append(motion-real_motion)
            ss.append(float(sd[0]*(z-.15)*math.exp(-ell[0])/.2))
            hit+=int(abs(motion)>=.75)
            if kind=='static_slant':
                d0=z*(1+rng.uniform(-.15,.15))
                d1=(z-.15)*(1+rng.uniform(-.15,.15))
                depth_false+=int(abs(d1-d0+.15)/.2>=.75)
        err=np.array(velocities)
        results[kind]=dict(trials=count,supported=count-missed,radial_above_075=hit,
                           depth_difference_above_075=depth_false if kind=='static_slant' else None,
                           approach_bias_mps=float(err.mean()),rmse_mps=float(np.sqrt((err*err).mean())),
                           empirical_residual_model_se_mps=float(np.sqrt(np.mean(np.square(ss)))),
                           supplied_center_shift_px_median=float(np.median(center_res)))
    return results


def map_jac(H, center):
    v=H @ np.r_[center,1.]
    p=v[:2]/v[2]
    return p,(H[:2,:2]-p[:,None]*H[2,:2])/v[2]


def linear_bound(rng):
    # Local linear experiment uses an explicit independent Gaussian noise law.
    offs=np.stack(np.meshgrid(np.arange(-6,7),np.arange(-6,7)),axis=-1).reshape(-1,2)
    dx,dy=rng.normal(size=(2,len(offs)))
    p=np.column_stack([offs,np.ones(len(offs))])
    H=np.column_stack([dx[:,None]*p,dy[:,None]*p])
    c=np.array([.5,0,0,0,.5,0.])
    coef=c @ np.linalg.pinv(H)
    est=rng.normal(size=(10000,len(offs))) @ coef
    crlb=float(c @ np.linalg.inv(H.T@H) @ c)
    return {'samples':len(est),'kappa_variance':float(est.var(ddof=1)),'bound':crlb,
            'variance_over_bound':float(est.var(ddof=1)/crlb)}


def condition_grid():
    # Counterfactual angular extent; arbitrary real textures, no rider labels.
    spectra={}
    textures={}
    for p in sorted(base.SRC.glob('*.npz')):
        with np.load(p) as a:
            im=decode(a['rgb_blob'],a['rgb_offsets'],a['orig_hw'],0)
        with np.load(base.CACHE/p.name) as a:
            s4=a['gray_s4'][0]
        textures[p.stem]=digest(p)
        for im0 in (im,s4):
            h,w=im0.shape
            ix,iy=gradients(im0)
            for fov in (70,90,110):
                focal=w/2/math.tan(math.radians(fov/2))
                for z in range(3,13):
                    radius=int(math.floor(focal*.6/z/2))
                    for cy in (.35,.50,.65):
                        for cx in (.35,.50,.65):
                            x=int(cx*w); y=int(cy*h)
                            sl = np.s_[y-radius:y+radius+1,x-radius:x+radius+1]
                            gx,gy=ix[sl].ravel(),iy[sl].ravel()
                            off=np.stack(np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1)),axis=-1).reshape(-1,2)
                            phi=np.column_stack([off,np.ones(len(gx))])
                            design=np.column_stack([gx[:,None]*phi,gy[:,None]*phi])
                            # Residualize isotropic dilation against five free affine nuisances.
                            xcol=.5*(design[:,0]+design[:,4])
                            nuisance=np.column_stack([design[:,0]-design[:,4],design[:,1],design[:,2],design[:,3],design[:,5]])
                            dil=2*(xcol - nuisance @ np.linalg.lstsq(nuisance,xcol,rcond=1.e-8)[0])
                            info=float(dil@dil/16.) # known 4 DN independent residual noise
                            spectra.setdefault((w,fov,z),[]).append(info)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(12.,6.5),constrained_layout=True)
    FigureCanvasAgg(fig)
    axes=fig.subplots(2,3,sharex=True,sharey=True)
    records=[]
    for row,w in enumerate((96,384)):
        for col,fov in enumerate((70,90,110)):
            ax=axes[row,col]
            for hz in (5,10,15,30):
                series=[]
                for z in range(3,13):
                    inf=np.asarray(spectra[w,fov,z])
                    # Two-frame small-deformation Gaussian experiment, conditional on gradients.
                    ncp=6/z/hz*np.sqrt(inf)
                    power=norm.sf(norm.isf(.05)-ncp)
                    ok=power>=.95
                    finite=inf>1.e-10
                    sd=hz/np.sqrt(inf[finite]) if finite.any() else np.empty(0)
                    records.append(dict(width=w,FOV_deg=fov,Z_m=z,fps=hz,patches=len(inf),finite=int(finite.sum()),
                                        meets_95_percent_power=int(ok.sum()),median_power=float(np.median(power)),
                                        median_finite_rate_sd_s_inv=float(np.median(sd)) if len(sd) else None))
                    series.append(100*ok.mean())
                ax.plot(range(3,13),series,label=f'{hz} Hz',marker='.',linewidth=1.3)
            ax.set(title=f'{w} px width, FOV {fov} deg',ylim=(-2,102))
            ax.grid(alpha=.25)
            if row==1: ax.set_xlabel('distance (m)')
            if col==0: ax.set_ylabel('texture patches reaching 95% power (%)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Conditional two-frame information at 6 m/s; 1-sided size 5%, residual noise 4 DN\n216 CODa texture patches per point; ideal linear warp, no object or velocity labels')
    fig.savefig(OUT/'conditional_power.png',dpi=160)
    return dict(records=records,source_rgb_sha256=textures,sampling='first frame per clip, fixed 3x3 texture centers, no object labels')


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(7401)
    results={'symbolic':identities(),'rendered_paths':render_trials(rng),
             'conditional_linear_gaussian':linear_bound(rng),'texture_grid':condition_grid(),
             'source_code_sha256':{p.name:digest(p) for p in (HERE/'verify_theory.py', HERE/'direct.py', HERE/'eval_empirical.py')}}
    (OUT/'theory_verification.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in results.items() if k!='texture_grid'}))


if __name__=='__main__':
    main()

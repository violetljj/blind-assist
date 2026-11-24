"""Spherical differential identities and two-grid minimax calculations."""
import json
import math

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.ndimage import binary_erosion
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from atlas import (level_basis, project_kb, project_pinhole,project_mei,project_aria624,
                   mercator_geometry,mercator_pixels,yaw)
from eval_empirical import HERE,OUT,digest,oxford_inputs


def symbolic():
    eta,phi,psi=sp.symbols('eta phi psi',real=True)
    # Use latitude first, then dtheta/deta=cos(theta) and sech(eta)=cos(theta).
    theta=sp.symbols('theta',real=True)
    r=sp.Matrix([sp.cos(theta)*sp.sin(phi),sp.sin(theta),sp.cos(theta)*sp.cos(phi)])
    a=r.diff(phi); b=r.diff(theta)*sp.cos(theta)
    metric=(a.row_join(b).T*a.row_join(b)).applyfunc(sp.trigsimp)
    assert metric==sp.cos(theta)**2*sp.eye(2)
    curvature=sp.simplify(-sp.diff(sp.log(1/sp.cosh(eta)),eta,2)*sp.cosh(eta)**2)
    assert curvature==1
    R=sp.Matrix([[sp.cos(psi),0,sp.sin(psi)],[0,1,0],[-sp.sin(psi),0,sp.cos(psi)]])
    assert (R*r-r.subs(phi,phi+psi)).applyfunc(sp.trigsimp)==sp.zeros(3,1)
    # Area density times inverse metric cancels in 2D Dirichlet energy.
    assert sp.simplify(metric.det()/sp.cos(theta)**4)==1
    # X=rho*r, differentiate using an orthonormal chart tangent pair.
    rho,dp,dq,l=sp.symbols('rho dp dq l',nonzero=True)
    T1=sp.Matrix([rho*l,0,dp]); T2=sp.Matrix([0,rho*l,dq])
    N=T1.cross(T2)
    assert sp.simplify(-N[0]*l/N[2]-dp/rho)==0
    assert sp.simplify(-N[1]*l/N[2]-dq/rho)==0
    h,w=64,96
    m,pars=mercator_geometry(h,w,(50,50,48,32),np.eye(3))
    y,x=np.mgrid[:h,:w]
    uv=mercator_pixels(yaw(m,2.3*pars[0]),h,w,pars)
    err=float(np.abs(uv-np.stack([x+2.3,y],axis=-1)).max())
    assert err<1.e-10
    # A two-channel ReLU implemented by 1x1 convolution plus activation does
    # not commute with half-pixel linear interpolation at a sign boundary.
    a=np.array([-1.,1.])
    relu_error=float(np.maximum(a.mean(),0)-np.maximum(a,0).mean())
    assert relu_error==-.5
    # Even a single linear stride-2 identity kernel loses one-sample phase.
    impulse=np.array([1,0,0,0])
    assert np.roll(impulse,1)[::2].sum()!=impulse[::2].sum()
    ray=np.array([[0.,0.,1.],[.2,.1,1.]])
    z=np.zeros(4); k=np.array([300.,300.,120.,130.])
    p1,_=project_pinhole(ray,k); p2,_=project_mei(ray,k,0,z); assert np.allclose(p1,p2)
    p3,_=project_kb(ray,k,z,1.3)
    p4,_=project_aria624(ray,[300.,120.,130.,*np.zeros(12)],1.3); assert np.allclose(p3,p4)
    return dict(curvature=1,metric='cos(theta)^2 I',max_coordinate_yaw_error_px=err,
                relu_half_shift_difference=relu_error,stride_two_counterexample=True,
                radial_depth_to_normal_PDE=True,MEI_pinhole_and_Aria624_KB_limits=True)


def cap_table():
    out=[]
    for deg in (60,110,150):
        a=np.deg2rad(deg/2)
        sec=1/np.cos(a)
        def theta(v):
            return np.arccos(np.clip(np.cos(a)/np.cos(v),-1,1))
        eq=quad(lambda v:theta(v),0,a)[0]/(a*a)
        merc=quad(lambda v:np.arcsinh(np.tan(theta(v))),0,a)[0]/(a*np.arcsinh(np.tan(a)))
        st=1/(np.cos(a/2)**2)
        out.append(dict(FOV=deg,gnomonic=[sec,sec**2,np.pi/4],equirectangular=[sec,sec,eq],
                        mercator=[1,sec,merc],stereographic=[1,st,np.pi/4],chebyshev=[1,st,np.pi/4]))
    return out


def solve(x,y,region):
    # Node-centered five-point Laplacian in isothermal coordinates. Interface
    # nodes of the physical mask are clamped to zero (staircase approximation).
    dx=x[1]-x[0]; dy=y[1]-y[0]
    active=binary_erosion(region,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    jj,ii=np.where(active)
    lab=np.full(region.shape,-1,int); lab[active]=np.arange(len(ii))
    cols=[lab[active]]; rows=[lab[active]]; vals=[np.full(len(ii),2/dx**2+2/dy**2)]
    for sy,sx,weight in ((0,1,1/dx**2),(0,-1,1/dx**2),(1,0,1/dy**2),(-1,0,1/dy**2)):
        aux=lab[jj+sy,ii+sx]; keep=aux>=0
        cols.append(aux[keep]); rows.append(lab[active][keep]); vals.append(np.full(keep.sum(),-weight))
    matrix=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(ii),len(ii))).tocsr()
    rhs=1/np.cosh(y[jj])**2
    sol=spsolve(matrix,rhs)
    assert sol.min()>0
    return dict(interior_nodes=int(len(sol)),max_log_inverse_scale=float(sol.max()),
                scale_ratio=float(np.exp(sol.max())),residual_linf=float(np.max(np.abs(matrix@sol-rhs))))


def domains():
    import sys
    from eval_empirical import ROOT
    sys.path.insert(0,str(ROOT/'research/active/hardware-bringup/host'))
    from spires_insert_dev_20261117 import K0,D0,R_CV
    frame=next(r for r in oxford_inputs() if r['key']=='3400')
    basis=level_basis(frame['down'])
    rec={}
    for nx in (101,201):
        x=np.linspace(-np.deg2rad(75),np.deg2rad(75),nx)
        y=np.linspace(*np.arcsinh(np.tan(np.deg2rad([-55,35]))),nx)
        rec['spec_rect_'+str(nx)]=solve(x,y,np.ones((nx,nx),bool))
        x=np.linspace(-np.pi/2,np.pi/2,nx)
        y=np.linspace(-2.8,2.8,2*nx-1)
        ph,et=np.meshgrid(x,y); sech=1/np.cosh(et)
        r=np.stack([sech*np.sin(ph),np.tanh(et),sech*np.cos(ph)],axis=-1)
        xy,mask=project_kb(r@basis@R_CV.T,K0,D0,np.deg2rad(80))
        mask &= (xy[...,0]>=0)&(xy[...,0]<=1439)&(xy[...,1]>=0)&(xy[...,1]<=1079)
        assert not (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
        rec['oxford_3400_'+str(nx)]=solve(x,y,mask)
        rec['oxford_3400_'+str(nx)]['mercator_scale_ratio']=float(sech[mask].max()/sech[mask].min())
    return rec


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    res=dict(symbolic=symbolic(),cap_table=cap_table(),dirichlet=domains(),
             source_code_sha256={p.name:digest(p) for p in (HERE/'verify_theory.py',HERE/'eval_empirical.py',HERE/'atlas.py')})
    (OUT/'theory_verification.json').write_text(json.dumps(res,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(res))


if __name__=='__main__':
    main()

"""Earth-level directions, inverse sensor maps, and exact support at pixel centers."""
import cv2
import numpy as np


def unit(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True), 1.e-12)


def level_basis(down):
    down=unit(np.asarray(down,float))
    forward=np.array([0.,0.,1.])-down[2]*down
    if np.linalg.norm(forward)<1.e-3:
        forward=np.array([0.,-1.,0.])+down[1]*down
    forward=unit(forward)
    return np.stack([unit(np.cross(down,forward)),down,forward])


def pin_rays(h,w,k):
    y,x=np.mgrid[:h,:w]+.5
    return unit(np.stack([(x-k[2])/k[0],(y-k[3])/k[1],np.ones_like(x)],axis=-1))


def yaw(rays,angle):
    c,s=np.cos(angle),np.sin(angle)
    return rays @ np.array([[c,0,-s],[0,1,0],[s,0,c]])


def project_pinhole(ray,k):
    z=ray[...,2]
    p=ray[...,:2]/np.maximum(z[...,None],1.e-9)
    return p*np.asarray(k[:2])+np.asarray(k[2:]), z>0


def project_kb(ray,k,coeff,limit):
    r=np.linalg.norm(ray[...,:2],axis=-1)
    theta=np.arctan2(r,ray[...,2])
    t2=theta*theta
    a,b,c,d=coeff
    q=theta*(1+t2*(a+t2*(b+t2*(c+t2*d))))
    scale=np.ones_like(r)
    np.divide(q,r,out=scale,where=r>1.e-12)
    return ray[...,:2]*scale[...,None]*np.asarray(k[:2])+np.asarray(k[2:]), theta<limit


def project_mei(ray,k,xi,dist):
    # Unified central sphere followed by Brown radial/tangential distortion.
    X,Y,Z=np.moveaxis(unit(ray),-1,0)
    den=Z+xi
    # For xi>1 the back branch folds before den=0.
    valid=(den>1.e-9)&(Z>-(1/xi if xi>1 else xi))
    x=X/np.maximum(den,1.e-9); y=Y/np.maximum(den,1.e-9)
    k1,k2,p1,p2=dist
    r=x*x+y*y; d=1+k1*r+k2*r*r
    q=np.stack([x*d+2*p1*x*y+p2*(r+2*x*x),y*d+2*p2*x*y+p1*(r+2*y*y)],axis=-1)
    return q*np.asarray(k[:2])+np.asarray(k[2:]),valid


def mercator_geometry(h,w,k,basis):
    # Match horizontal pinhole FOV and average ray center. Use the same step
    # on both axes for conformality. Image rows point down.
    step=(np.arctan((w-k[2])/k[0])-np.arctan(-k[2]/k[0]))/w
    pin_center=unit(np.array([(w/2-k[2])/k[0],(h/2-k[3])/k[1],1.]))@basis.T
    phi0=np.arctan2(pin_center[0],pin_center[2])
    eta0=np.arctanh(np.clip(pin_center[1],-.999999,.999999))
    y,x=np.mgrid[:h,:w]+.5
    phi=phi0+(x-w/2)*step
    eta=eta0+(y-h/2)*step
    sech=1/np.cosh(eta)
    ray=np.stack([sech*np.sin(phi),np.tanh(eta),sech*np.cos(phi)],axis=-1)
    return ray,(step,phi0,eta0)


def mercator_pixels(ray,h,w,geometry):
    step,phi0,eta0=geometry
    phi=np.arctan2(ray[...,0],ray[...,2])-phi0
    phi=np.arctan2(np.sin(phi),np.cos(phi))
    eta=np.arctanh(np.clip(ray[...,1],-.999999999,.999999999))
    return np.stack([phi/step+w/2-.5,(eta-eta0)/step+h/2-.5],axis=-1)


def pull(im,p,allowed,nearest=False):
    h,w=im.shape[:2]
    valid=allowed & np.isfinite(p).all(axis=-1)&(p[...,0]>=0)&(p[...,0]<=w-1)&(p[...,1]>=0)&(p[...,1]<=h-1)
    out=cv2.remap(im,p.astype(np.float32),None,cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT)
    out[~valid]=0
    return out,valid


def project_aria624(ray, params, limit):
    """Fisheye624 native forward branch, 15 parameters in the Aria SDK order."""
    f,cx,cy,*dist=params
    if len(dist)!=12:
        raise ValueError('Expected six radial, two tangential and four prism coefficients')
    r=np.linalg.norm(ray[...,:2],axis=-1)
    theta=np.arctan2(r,ray[...,2]); t2=theta*theta
    poly=np.polynomial.polynomial.polyval(t2,[1.,*dist[:6]])
    scale=np.divide(theta*poly,r,out=np.ones_like(r),where=r>1.e-12)
    x,y=np.moveaxis(ray[...,:2]*scale[...,None],-1,0)
    p0,p1,s0,s1,s2,s3=dist[6:]
    r2=x*x+y*y
    u=x + (2*x*x+r2)*p0+2*x*y*p1+s0*r2+s1*r2*r2
    v=y + (2*y*y+r2)*p1+2*x*y*p0+s2*r2+s3*r2*r2
    return f*np.stack([u,v],axis=-1)+[cx,cy], (ray[...,2]>0)&(theta<limit)

"""Exact coordinate extrema of a rotated square angular cone's radial shell.

Analytic geometry only. No response law, target identity or temporal accumulation.
"""
import math
import numpy as np
from tof_jitter_geometry import rotation
from contact_retina_spec import BODY_BOXES


def directional_extrema(coefficients, slope):
    # f(u,v)=(a+b*u+c*v)/sqrt(1+u*u+v*v), on [-s,s]^2.
    # Compact domain: extrema occur at corners, edge stationary points or an
    # interior stationary point. Enumerate all, including negative extrema.
    a,b,c=map(float,coefficients);s=slope
    points=[(u,v) for u in (-s,s) for v in (-s,s)]
    for k in (-s,s):
        if a+b*k != 0:
            v=c*(1+k*k)/(a+b*k)
            if -s<=v<=s:points.append((k,v))
        if a+c*k != 0:
            u=b*(1+k*k)/(a+c*k)
            if -s<=u<=s:points.append((u,k))
    if a != 0:
        u,v=b/a,c/a
        if -s<=u<=s and -s<=v<=s:points.append((u,v))
    values=[(a+b*u+c*v)/math.sqrt(1+u*u+v*v) for u,v in points]
    return min(values),max(values)


def support(range_m,valid,uncertainty_m,pitch_deg,yaw_deg,eye_height_m=1.7):
    if not valid:return dict(status='UNKNOWN',supported=[],bounds=None)
    if any(v is None or not math.isfinite(v) for v in (range_m,uncertainty_m,eye_height_m)):
        raise ValueError('Finite range inputs required')
    if range_m<=0 or uncertainty_m<0:raise ValueError('Invalid range uncertainty')
    lo=max(0.,range_m-uncertainty_m);hi=range_m+uncertainty_m
    slope=math.tan(math.radians(7.5))/math.sqrt(2)
    bounds=[]
    for row in rotation(pitch_deg,yaw_deg):
        mn,mx=directional_extrema(row,slope)
        values=[lo*mn,lo*mx,hi*mn,hi*mx]
        bounds.append((min(values)-1e-12,max(values)+1e-12))
    lower,upper=np.array(bounds).T;lower[2]+=eye_height_m;upper[2]+=eye_height_m
    names=[]
    for kind,(bl,bh) in zip(('BODY','HEAD'),BODY_BOXES):
        for half,start in zip(('NEAR','FAR'),(bh[0],bh[0]+1.5)):
            if lower[0]>=start and upper[0]<=start+1.5 and all(lower[k]>=bl[k] and upper[k]<=bh[k] for k in (1,2)):
                names.append(kind+'_'+half)
    return dict(status='POSITIVE_SUPPORT' if names else 'UNKNOWN',supported=names,bounds=[lower.tolist(),upper.tolist()])

"""Conservative interval projection; conclusions conditional on its proxy/bounds."""
import copy
import itertools
import math

CORRIDOR = ((.2,3.6),(-.3,.3),(.4,2.05))
RANGE_ERROR_M = .15
BOX_ERROR_PX = 2.
PITCH_ERROR_DEG = .5


def add(a,b): return (a[0]+b[0],a[1]+b[1])
def sub(a,b): return (a[0]-b[1],a[1]-b[0])
def mul(a,b):
    values=[x*y for x in a for y in b]
    return min(values),max(values)


def div(a,b):
    if b[0]<=0<=b[1]:raise ValueError('Interval division through zero')
    return mul(a,tuple(sorted((1/b[0],1/b[1]))))


def square(a): return (0. if a[0]<=0<=a[1] else min(a[0]**2,a[1]**2),max(a[0]**2,a[1]**2))


def trig(a, cosine=False):
    fn=math.cos if cosine else math.sin
    values=[fn(a[0]),fn(a[1])]
    offset=0. if cosine else math.pi/2
    for k in range(math.ceil((a[0]-offset)/math.pi),math.floor((a[1]-offset)/math.pi)+1):values.append(fn(offset+k*math.pi))
    return min(values),max(values)


def projection_envelope(u,v,intr,ranges,pitches,yaws,height):
    """Returns enclosing XYZ intervals, including interior trig extrema."""
    if ranges[0]<=0 or intr['fx']<=0 or intr['fy']<=0: return None
    if not all(math.isfinite(x) for interval in (u,v,ranges,pitches,yaws) for x in interval):return None
    if any(a>b for a,b in (u,v,ranges,pitches,yaws)):return None
    a=((u[0]-intr['cx'])/intr['fx'],(u[1]-intr['cx'])/intr['fx'])
    b=((intr['cy']-v[1])/intr['fy'],(intr['cy']-v[0])/intr['fy'])
    p=tuple(map(math.radians,pitches));y=tuple(map(math.radians,yaws))
    cp,sp=trig(p,True),trig(p)
    f=sub(cp,mul(b,sp));up=add(sp,mul(b,cp))
    if f[0]<=0:return None
    bearing=add(tuple(math.atan(x) for x in div(a,f)),y)
    h2=add(square(f),square(a));h=tuple(math.sqrt(x) for x in h2)
    z=add((height,height),mul(ranges,div(up,h)))
    return (mul(ranges,trig(bearing,True)),mul(ranges,trig(bearing)),z)


def disjoint(box): return any(hi<clo or lo>chi for (lo,hi),(clo,chi) in zip(box,CORRIDOR))
def enclosed(box): return all(lo>=clo and hi<=chi for (lo,hi),(clo,chi) in zip(box,CORRIDOR))


def classify(box,r,row,yaw):
    if len(box)!=4 or not all(math.isfinite(v) for v in box):return dict(state='AMBIGUOUS',reason='invalid_box')
    x1,y1,x2,y2=box
    if x2-x1<=2*BOX_ERROR_PX or y2-y1<=2*BOX_ERROR_PX:return dict(state='AMBIGUOUS',reason='thin_or_invalid_box')
    dyaw=.5+.2*row['time_s']
    ranges=(r-RANGE_ERROR_M,r+RANGE_ERROR_M)
    pitches=(row['camera_pitch_deg']-PITCH_ERROR_DEG,row['camera_pitch_deg']+PITCH_ERROR_DEG)
    yaws=(yaw-dyaw,yaw+dyaw);intr=row['rgb_intrinsics'];height=row['camera_in_body_m'][2]
    outer=projection_envelope((x1-BOX_ERROR_PX,x2+BOX_ERROR_PX),(y1-BOX_ERROR_PX,y2+BOX_ERROR_PX),intr,ranges,pitches,yaws,height)
    if outer is None:return dict(state='AMBIGUOUS',reason='invalid_projection')
    details=dict(outer_xyz=outer,range_bounds=ranges,yaw_bounds=yaws,pitch_bounds=pitches)
    if disjoint(outer):return dict(state='OUT',**details)
    xs=(x1+BOX_ERROR_PX,(x1+x2)/2,x2-BOX_ERROR_PX)
    ys=(y1+BOX_ERROR_PX,(y1+y2)/2,y2-BOX_ERROR_PX)
    for u,v in itertools.product(xs,ys):
        witness=projection_envelope((u,u),(v,v),intr,ranges,pitches,yaws,height)
        if witness is not None and enclosed(witness):return dict(state='IN',witness_pixel=[u,v],witness_xyz=witness,**details)
    return dict(state='AMBIGUOUS',reason='overlap_not_decided',**details)


def refine(row,nominal):
    """Consumes observable-derived cache only; never mutates the previous arm."""
    result=copy.deepcopy(nominal);support=bool(nominal['tof_support'])
    for a in result['associations']:
        a['nominal_support']=a['refined_support']
        if a['state']=='ASSOCIATED':
            k=a['slot'];geometry=classify(result['proposals'][a['proposal']],row['radar_range_m'][k],row,result['integrated_yaw_deg'])
            a['geometry']=geometry
            a['refined_support']=geometry['state']=='IN' or (geometry['state']=='AMBIGUOUS' and a['baseline_support'])
        else:
            a['geometry']=dict(state='UNAVAILABLE')
            a['refined_support']=a['baseline_support']
        support |= bool(a['refined_support'])
    result['candidate']=bool(support);result['candidate_state']='ALERT' if support else 'UNKNOWN'
    return result

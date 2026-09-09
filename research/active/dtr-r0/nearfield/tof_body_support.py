"""Conservative positive body-range support from an ideal aligned ToF cone.

Engineering geometry, not a VL53L1X response/validity model. No negative clearance.
"""
import math
from contact_retina_spec import BODY_BOXES


def support(range_m, valid, diagonal_fov_deg, uncertainty_m, eye_height_m=1.7):
    if not valid:
        return dict(status='UNKNOWN',supported=[],bounds=None)
    values=(range_m,diagonal_fov_deg,uncertainty_m,eye_height_m)
    if any(v is None or not math.isfinite(v) for v in values):raise ValueError('Finite calibrated inputs required')
    if range_m<=0 or uncertainty_m<0 or not 0<diagonal_fov_deg<90:raise ValueError('Invalid range/cone')
    lo=max(0.,range_m-uncertainty_m);hi=range_m+uncertainty_m
    slope=math.tan(math.radians(diagonal_fov_deg/2))/math.sqrt(2)
    axis_sine=slope/math.sqrt(1+slope*slope)
    # Exact AABB of the aligned angular square's radial shell. Containment is
    # sufficient, conservative evidence that ANY returning direction is inside.
    lower=[lo/math.sqrt(1+2*slope*slope),-hi*axis_sine,eye_height_m-hi*axis_sine]
    upper=[hi,hi*axis_sine,eye_height_m+hi*axis_sine]
    supported=[]
    for kind,(body_low,body_high) in zip(('BODY','HEAD'),BODY_BOXES):
        for half,(start,end) in zip(('NEAR','FAR'),((body_high[0],body_high[0]+1.5),(body_high[0]+1.5,body_high[0]+3))):
            if (lower[0]>=start and upper[0]<=end and lower[1]>=body_low[1] and upper[1]<=body_high[1]
                    and lower[2]>=body_low[2] and upper[2]<=body_high[2]):supported.append(kind+'_'+half)
    return dict(status='POSITIVE_SUPPORT' if supported else 'UNKNOWN',supported=supported,bounds=[lower,upper])


def add_positive_support(rgb_range_flags, observation):
    """Never erase a visual event or claim absence from a single return."""
    out=list(rgb_range_flags)
    if len(out)!=4:raise ValueError('BODY_NEAR,BODY_FAR,HEAD_NEAR,HEAD_FAR required')
    for i,name in enumerate(('BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR')):
        if name in observation['supported']:out[i]=True
    return out

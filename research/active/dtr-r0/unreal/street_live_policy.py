"""Sensor-only local motion response for the UE laboratory.

This is an experimental controller alongside retained DTR, not an App promotion.
Depth supplies unclassified low/static obstacles; DTR supplies dynamic route risk.
No actor IDs, future trajectories, scenario role, or collision labels are inputs.
"""
import math
import numpy as np


def depth_corridors(depth, fov, camera_height, ego_y=0.0, pitch_degrees=0.0):
    h,w=depth.shape
    fy=w/(2*math.tan(math.radians(fov)/2))
    v,u=np.mgrid[2:h:4,2:w:4]
    x=depth[v,u]
    y=(u-(w-1)/2)*x/fy
    camera_up=-(v-(h-1)/2)*x/fy
    pitch=math.radians(pitch_degrees)
    z=camera_height+math.sin(pitch)*x+math.cos(pitch)*camera_up
    x=math.cos(pitch)*x-math.sin(pitch)*camera_up
    valid=(x>.08)&(x<12)&np.isfinite(x)
    obstacle=valid&(z>.065)&(z<1.85)
    corridors={}
    for target in (0.0,-.72,.72,-.52,.52):
        # Lateral offsets are in the route frame; samples in camera-right coordinates.
        samples=x[obstacle & (np.abs(y-(target-ego_y))<.30)]
        corridors[str(target)]=float(np.quantile(samples,.04)) if len(samples)>=8 else 12.0
    # Use the same wearer corridor as side selection. A wider front query
    # incorrectly treated the adjacent narrow-passage wall as a frontal block.
    central=x[obstacle & (np.abs(y)<.30)]
    nearest=float(np.quantile(central,.04)) if len(central)>=8 else None
    return {'clearance_m':corridors,'front_obstacle_m':nearest,
            'valid_fraction':float(valid.mean()),'height_filter_m':[.065,1.85],
            'source':'OBSERVED_FORWARD_DEPTH_ONLY'}


class MotionPolicy:
    MODES=('JOINT','DTR_ONLY','DEPTH_ONLY')

    def __init__(self, mode='JOINT'):
        if mode not in self.MODES: raise ValueError(f'Unknown controller mode: {mode}')
        self.mode=mode
        self.target_y=0.0
        self.pass_until_x=None
        self.last_risk_t=-100.0

    def command(self, *, t, x, y, goal_x, dtr_risk, corridors):
        # Remove the entire depth control channel, including its validity stop,
        # side selection and return-path geometry, in the DTR-only ablation.
        if self.mode=='DTR_ONLY':
            corridors={'clearance_m':{str(k):12.0 for k in (0.0,-.72,.72,-.52,.52)},
                       'front_obstacle_m':None,'valid_fraction':1.0}
        if self.mode=='DEPTH_ONLY': dtr_risk=False
        result=self._command_active(t=t,x=x,y=y,goal_x=goal_x,dtr_risk=dtr_risk,corridors=corridors)
        result.update(controller_mode=self.mode,dtr_route_risk=bool(dtr_risk),
                      depth_near_risk=corridors['front_obstacle_m'] is not None and corridors['front_obstacle_m']<2.8)
        return result

    def _command_active(self, *, t, x, y, goal_x, dtr_risk, corridors):
        clear=corridors['clearance_m']
        front=corridors['front_obstacle_m']
        near=front is not None and front<2.8
        risk=bool(dtr_risk or near)
        if risk: self.last_risk_t=t
        if x>=goal_x:
            return {'vx_mps':0.0,'vy_mps':0.0,'action':'ARRIVED','risk':risk,'target_y_m':y}
        if corridors['valid_fraction']<.2:
            return {'vx_mps':0.0,'vy_mps':0.0,'action':'WAIT_UNKNOWN_DEPTH','risk':risk,'target_y_m':y}
        if near and self.pass_until_x is None:
            sides=sorted((-.72,.72,-.52,.52),key=lambda side:clear[str(side)],reverse=True)
            side=sides[0]
            if clear[str(side)]>min(3.3,front+1.0):
                self.target_y=side
                self.pass_until_x=x+front+1.0
        if self.pass_until_x is not None and clear['0.0']<12.0:
            # Keep the last observed obstacle position until we have passed it.
            # Leaving the camera frustum is not evidence that the return is clear.
            self.pass_until_x=max(self.pass_until_x,x+clear['0.0']+1.0)
        if self.pass_until_x is not None and x>self.pass_until_x and clear['0.0']>2.8:
            self.target_y=0.0
            self.pass_until_x=None
        error=self.target_y-y
        vy=max(-.65,min(.65,error*2))
        if near and front<.9:
            # Do not sidestep into a crossing person already immediately ahead.
            vx=0.0
            vy=0.0
            action='BRAKE_IMMINENT'
        elif abs(error)>.09:
            vx=.35 if front is None or front>1.0 else 0.0
            action='SIDESTEP'
        elif near and front<1.4:
            vx=0.0
            action='BRAKE'
        elif dtr_risk and self.pass_until_x is None:
            vx=0.0
            action='WAIT_DYNAMIC'
        elif t-self.last_risk_t<.4 and self.pass_until_x is None:
            vx=.35
            action='CAUTIOUS_RESUME'
        else:
            vx=1.0
            action='WALK'
        return {'vx_mps':vx,'vy_mps':vy,'action':action,'risk':risk,
                'dtr_route_risk':bool(dtr_risk),'depth_near_risk':near,'target_y_m':self.target_y}

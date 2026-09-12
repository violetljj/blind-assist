"""Small fresh controlled single-camera multimodal simulation source."""
import json
from pathlib import Path
import random

def source():
    rng=random.Random(107013);frames=[]
    families=('body','head','pole','offroute','clear','turning')
    for family in families:
        for variant in range(2):
            episode=f'{family}_{variant}'
            if family=='body': objects=[dict(name='shape0',center_m=[2.7+variant*.2,0.,1.05],size_m=[.18,.4,.8])]
            elif family=='head': objects=[dict(name='shape0',center_m=[2.7+variant*.2,0.,1.7],size_m=[.12,.3,.16])]
            elif family=='pole': objects=[dict(name='shape0',center_m=[2.7+variant*.2,.08,1.05],size_m=[.06,.06,2.1])]
            elif family=='offroute': objects=[dict(name='shape0',center_m=[2.5+variant*.3,.45+variant*.2,1.2],size_m=[.18,.16,1.2])]
            elif family=='clear': objects=[]
            else: objects=[dict(name='shape0',center_m=[2.6,.08,1.05],size_m=[.18,.25,.7]),dict(name='shape1',center_m=[2.9,.65,1.65],size_m=[.15,.28,.22])]
            for obj in objects:obj.update(texture_seed=rng.randrange(2**30),texture_grid=[3,6])
            sensor_seed=rng.randrange(2**30)
            for j in range(8):
                import math
                t=j*.25;x=j*.125;yaw=12*math.sin(1.2*t) if family=='turning' else 0.
                frames.append(dict(id=f'{episode}_{j:02d}',episode=episode,family=family,time_s=t,
                    camera=dict(x=x,y=0.,z=1.7,yaw=yaw,pitch=-3.,roll=0.),body_origin_m=[x,0.,0.],
                    objects=objects,sensor_seed=sensor_seed,wearer_speed=.5,
                    radar_ghost=dict(x=-.12,z=2.4) if variant==1 else None))
    return dict(schema='mz107-single-rgb-native-sensor-canary-v1',seed=107013,
        authority='FRESH_CONTROLLED_UE_RGB_NATIVE_COLLISION_TOF_HYPOTHETICAL_RADAR_IMU_NOT_HARDWARE_OR_RF',
        rig=dict(width=640,height=360,hfov_deg=70.,tof_hfov_deg=45.,rgb_camera_count=1),
        source_classes='6 families x2 scenes x8 time samples; no train/test split; development canary',
        frames=frames)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve source specification')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(source(),indent=2)+'\n')

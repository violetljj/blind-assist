"""Fixed 12-scene single-RGB canary with inside-edge and outside-edge pairs.

Uses unchanged MZ107 renderer and sensor generator; all 96 frames are retained.
Scenario construction is disclosed Development, not a blind confirmation panel.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def source():
    rng=random.Random(109013)
    scenes=[]
    # Paired inner edges 0.29m (1cm inside the corridor) and 0.41m (outside).
    # All geometry, seeds and frame counts are fixed before this one capture.
    for k,(forward,lateral,width) in enumerate(((3.2,.38,.18),(3.2,-.38,.18),(3.2,.50,.18),(3.2,-.50,.18))):
        assert math.isclose(abs(lateral)-width/2,.29 if k<2 else .41,abs_tol=1e-12)
        scenes.append(dict(episode=f'boundary_{k}',family='boundary',yaw_amplitude=7.,
            ghost=None,objects=[dict(name='shape0',center_m=[forward,lateral,1.2],size_m=[.12,width,1.1])]))
    for k in range(2):
        scenes.append(dict(episode=f'head_{k}',family='head',yaw_amplitude=0.,ghost=None,
            objects=[dict(name='shape0',center_m=[2.8+k*.3,.1*(-1 if k else 1),1.7],size_m=[.12,.26,.16])]))
    for k in range(2):
        scenes.append(dict(episode=f'pole_{k}',family='pole',yaw_amplitude=0.,ghost=None,
            objects=[dict(name='shape0',center_m=[2.9+k*.3,.08*(-1 if k else 1),1.05],size_m=[.055,.055,2.1])]))
    for k in range(2):
        sign=1 if k==0 else -1
        scenes.append(dict(episode=f'multitarget_{k}',family='multitarget',yaw_amplitude=9.*sign,
            ghost=dict(x=.12*sign,z=2.4),objects=[
                dict(name='shape0',center_m=[3.1,.06*sign,1.05],size_m=[.18,.24,.7]),
                dict(name='shape1',center_m=[3.4,.42*sign,1.7],size_m=[.12,.12,.18])]))
    for k in range(2):
        scenes.append(dict(episode=f'clear_{k}',family='clear',yaw_amplitude=0.,
            ghost=dict(x=-.12,z=2.4) if k else None,objects=[]))
    frames=[]
    for scene in scenes:
        objects=scene['objects']
        for obj in objects:obj.update(texture_seed=rng.randrange(2**30),texture_grid=[3,6])
        sensor_seed=rng.randrange(2**30)
        for j in range(8):
            t=j*.25;x=j*.125;yaw=scene['yaw_amplitude']*math.sin(1.2*t)
            frames.append(dict(id=f"{scene['episode']}_{j:02d}",episode=scene['episode'],family=scene['family'],time_s=t,
                camera=dict(x=x,y=0.,z=1.7,yaw=yaw,pitch=-3.,roll=0.),body_origin_m=[x,0.,0.],objects=objects,
                sensor_seed=sensor_seed,wearer_speed=.5,radar_ghost=scene['ghost']))
    assert len(frames)==96 and len({f['episode'] for f in frames})==12
    return dict(schema='mz109-single-rgb-interval-extent-canary-v1',seed=109013,
        authority='FRESH_CONTROLLED_UE_RGB_NATIVE_COLLISION_TOF_HYPOTHETICAL_RADAR_IMU_NOT_HARDWARE_OR_RF',
        rig=dict(width=640,height=360,hfov_deg=70.,tof_hfov_deg=45.,rgb_camera_count=1),
        source_classes='2inside-edge+2outside-edge boundary+2head+2pole+2multitarget+2clear; 8 frames each; disclosed Development; no adaptive selection',
        fixed_before_capture=True,frames=frames)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[4]
    if not args.output.resolve().is_relative_to((root/'artifacts.local').resolve()):raise ValueError('Canonical artifacts only')
    if args.output.exists():raise ValueError('Preserve frozen source')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(source(),indent=2)+'\n')
    freeze=dict(spec_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frames=96,episodes=12,
        selection='FIXED_SEED109013_ALL_SCENES_ALL_FRAMES_NO_OUTCOME_SELECTION')
    args.output.with_name('freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')

"""Frozen NF-G6 construction: 12 posed static windows, 36 frames maximum."""
import argparse
import json
from pathlib import Path


def specification():
    camera = dict(x=26., y=0., z=1.82, pitch=-10., yaw=0., roll=0.)
    clips=[]
    def box(name,x,z,size,material):
        return dict(name=name,kind='cube',center_m=[x,0.,z],size_m=size,material='/Game/StreetLab/Materials/'+material)
    def add(motion,distance=None,control=None):
        clip_id=f'clip_{len(clips):02d}'
        objects=[]
        if distance is not None:
            objects=[box('bar',26+distance,1.72,[.08,1.2,.08],'Bronze')]
        if control=='flat_wall_mark':
            objects=[box('distant_wall',32.04,2.12,[.08,4.,4.],'Limestone'),box('flush_wall_mark',31.999,1.72,[.002,1.2,.08],'Charcoal')]
        poses=[]
        for i in range(3):
            pose=dict(camera)
            if motion=='forward': pose['x']+=i*.1
            elif motion=='lateral': pose['y']+=(i-1)*.1
            elif motion=='yaw': pose['yaw']+=(i-1)*2.
            poses.append(pose)
        clips.append(dict(clip_id=clip_id,motion=motion,distance_m=distance,control=control,objects=objects,poses=poses))
    for motion in ('forward','lateral'):
        for distance in (1.,2.,3.,6.): add(motion,distance)
    for distance in (2.,6.): add('yaw',distance)
    add('forward',control='flat_wall_mark')
    add('forward',control='no_added_obstacle')
    cases=[]
    for clip in clips:
        for i,pose in enumerate(clip['poses']):
            cases.append(dict(name=f"{clip['clip_id']}_f{i}",clip_id=clip['clip_id'],frame_in_clip=i,camera=pose,objects=clip['objects']))
    assert len(cases)==36
    return dict(schema='nf-g6-temporal-structure-spec-v1',sampling='POSED_STATIC_WINDOWS_NOT_REALTIME',pose_authority='IDEAL_PRIVILEGED_EGOPOSE',object_motion='NONE_FIXED_WORLD_GEOMETRY_PER_CLIP',map_sha256='cf35e5c9df54cd0f781f09ea8105fe8ef6078ed0822d4e594d64216e79a254fb',clips=clips,cases=cases)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[4]
    target=args.output.resolve()
    assert target.is_relative_to((root/'artifacts.local').resolve()) and not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(specification(),indent=2),encoding='utf-8')

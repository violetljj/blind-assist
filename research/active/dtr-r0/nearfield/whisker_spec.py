"""NF-G8 fresh grouped RGB trajectories; evaluator-only box geometry."""
import argparse
import json
import math
from pathlib import Path
import random

from contact_retina_spec import MAP_SHA, BODY_BOXES


def specification():
    rng = random.Random(80907)
    clips, cases, samples = [], [], []
    for group in range(128):
        split = 'train' if group < 80 else 'val' if group < 104 else 'test'
        # Each group's geometry, appearance, and all motion counterfactuals stay together.
        bx, by = 26 + rng.uniform(-.3, .3), rng.uniform(-.7, .7)
        distance = rng.uniform(2.7, 3.5)
        height = rng.uniform(1.57, 1.77) if group % 2 else rng.uniform(1.24, 1.30)
        length, thickness = rng.uniform(.7, 1.5), rng.uniform(.05, .12)
        shape = ('bar', 'panel', 'pole', 'bar')[group % 4]
        size = [.10, length, thickness]
        if shape == 'panel':
            size = [.12, rng.uniform(.65, 1.1), .32]
        elif shape == 'pole':
            size = [.12, .12, rng.uniform(.5, .8)]
        center = [bx + distance, by + rng.uniform(-.10, .10), .12 + height]
        material = '/Game/StreetLab/Materials/' + ('Bronze', 'Charcoal', 'Limestone')[group % 3]
        speed = rng.uniform(.6, 1.1)
        for variant in range(2):
            control = 'near' if variant == 0 else ('far', 'lateral', 'above')[group % 3]
            c, s = list(center), list(size)
            if control == 'far':
                far = rng.uniform(6., 8.)
                scale = far / distance
                c = [bx + far, by + (c[1] - by) * scale, 1.82 + (c[2] - 1.82) * scale]
                s = [s[0], s[1] * scale, s[2] * scale]
            elif control == 'lateral':
                c[1] = by + (1 if group % 2 else -1) * (s[1] / 2 + .55)
            elif control == 'above':
                c[2] = 2.15 + s[2] / 2
            objects = [dict(name='target', kind='cube', center_m=c, size_m=s, material=material)]
            for motion in ('approach', 'stop', 'static', 'rotation'):
                clip_id = f'g{group:03d}_v{variant}_{motion}'
                clips.append(dict(clip_id=clip_id, group_id=f'g{group:03d}', split=split,
                                  control=control, motion=motion, shape=shape, objects=objects))
                indices = []
                for j in range(6):
                    t = j * .2
                    travel = speed * (min(t, .4) if motion == 'stop' else t if motion == 'approach' else .4)
                    moving = motion == 'approach' or (motion == 'stop' and j < 2)
                    phase = 'stop' if motion == 'stop' and j >= 2 else motion
                    yaw = (t - .5) * (24 if group % 2 else -24) if motion == 'rotation' else 0.
                    body = dict(x=bx + travel, y=by, z=.12, pitch=0., yaw=0., roll=0.)
                    camera = dict(x=body['x'], y=by, z=1.82, pitch=-5., yaw=yaw, roll=0.)
                    idx = len(cases)
                    indices.append(idx)
                    cases.append(dict(name=f'{clip_id}_f{j}', clip_id=clip_id, frame_in_clip=j,
                                      camera=camera, wearer=body, time_s=t,
                                      speed_m_s=speed if moving else 0., phase=phase, objects=objects))
                    if j >= 2:
                        samples.append(dict(sample_id=f'{clip_id}_t{j}', clip_id=clip_id,
                                            group_id=f'g{group:03d}', split=split, frame_indices=indices[-3:]))
    assert len(clips) == 1024 and len(cases) == 6144 and len(samples) == 4096
    return dict(schema='nf-g8-whisker-spec-v1', map_sha256=MAP_SHA, seed=80907,
                clips=clips, cases=cases, samples=samples, body_boxes=BODY_BOXES,
                labels={'targets': {}}, query_range_m=3.,
                sampling='SIMULATED_5_HZ_SETTLED_POSES_NOT_WALLCLOCK_CAMERA',
                scope='CURRENT_VISIBLE_TASK_BOX_IN_FORWARD_BODY_CORRIDOR_AND_CLOSING; NOT_ALL_SCENE_FREE_SPACE',
                split_scope='128 parameter groups disjoint; one shared frozen Willow background; shape families shared')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(specification(), indent=2), encoding='utf-8')

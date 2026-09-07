"""Fixed paired NF-G7 trajectories and analytic body-box contact labels."""
import argparse
import json
import math
from pathlib import Path
import random

MAP_SHA = 'cf35e5c9df54cd0f781f09ea8105fe8ef6078ed0822d4e594d64216e79a254fb'
HORIZONS = (1., 2., 3.)
# Offsets relative to the body's floor point, with positive X forward.
BODY_BOXES = (([-.18, -.28, .65], [.18, .28, 1.4]),
              ([-.13, -.18, 1.4], [.13, .18, 1.85]))


def first_contact(body, speed, objects, bounds):
    """Continuous constant-X sweep of an axis-aligned body box against boxes."""
    if speed <= 0:
        raise ValueError('This experiment requires positive constant forward speed')
    low, high = bounds
    best = math.inf
    for obj in objects:
        a = [c - s / 2 for c, s in zip(obj['center_m'], obj['size_m'])]
        b = [c + s / 2 for c, s in zip(obj['center_m'], obj['size_m'])]
        if any(body[k] + high[k] < a[k] or body[k] + low[k] > b[k] for k in (1, 2)):
            continue
        enter = (a[0] - body[0] - high[0]) / speed
        leave = (b[0] - body[0] - low[0]) / speed
        if leave >= max(0., enter):
            best = min(best, max(0., enter))
    return None if not math.isfinite(best) else best


def specification():
    rng = random.Random(70917)
    clips, cases, samples, targets, contacts = [], [], [], {}, {}
    materials = ('Bronze', 'Charcoal', 'Limestone')
    for group in range(24):
        group_id = f'g{group:02d}'
        split = 'train' if group < 12 else 'val' if group < 16 else 'test'
        speed = rng.uniform(.85, 1.25)
        distance = rng.uniform(2.7, 4.1)
        # Upper torso controls stay above the floor after far-view scaling.
        height = rng.uniform(1.61, 1.78) if group % 2 == 0 else rng.uniform(1.24, 1.30)
        thickness = rng.uniform(.045, .095)
        length = rng.uniform(.85, 1.35)
        center_y = rng.uniform(-.09, .09)
        phase = rng.uniform(0, 2 * math.pi)
        yaw_amp, pitch0 = rng.uniform(1., 5.), rng.uniform(-10., -3.)
        material = '/Game/StreetLab/Materials/' + materials[group % 3]
        for variant in range(2):
            clip_id = f'{group_id}_v{variant}'
            control = 'near_contact' if variant == 0 else ('far_bar', 'wall_mark', 'near_miss')[group % 3]
            x, y, z = 26. + distance, center_y, .12 + height
            size = [.08, length, thickness]
            objects = []
            if control in ('far_bar', 'wall_mark'):
                far = rng.uniform(7., 8.5)
                scale = far / distance
                x, y, z = 26. + far, center_y * scale, 1.82 + (z - 1.82) * scale
                size = [.08 if control == 'far_bar' else .004, length * scale, thickness * scale]
                if control == 'wall_mark':
                    objects.append(dict(name='wall', kind='cube', center_m=[x + .062, y, 2.5],
                                        size_m=[.12, 5., 5.], material='/Game/StreetLab/Materials/Limestone'))
            elif control == 'near_miss':
                y += (1 if group % 2 else -1) * (length / 2 + .50)
            objects.append(dict(name='target', kind='cube', center_m=[x, y, z], size_m=size, material=material))
            clip = dict(clip_id=clip_id, group_id=group_id, split=split, control=control,
                        speed_m_s=speed, objects=objects, initial_distance_m=distance,
                        body_part='HEAD' if group % 2 == 0 else 'BODY')
            clips.append(clip)
            indices = []
            for j in range(16):
                t = j / 10
                body = dict(x=26. + speed*t, y=0., z=.12, pitch=0., yaw=0., roll=0.)
                camera = dict(x=body['x'], y=.025*math.sin(2*math.pi*1.7*t + phase),
                              z=1.82 + .025*math.cos(2*math.pi*1.7*t + phase),
                              pitch=pitch0 + 1.5*math.sin(2*math.pi*.7*t + phase),
                              yaw=yaw_amp*math.sin(2*math.pi*.6*t + phase), roll=0.)
                idx = len(cases)
                indices.append(idx)
                cases.append(dict(name=f'{clip_id}_f{j:02d}', clip_id=clip_id, frame_in_clip=j,
                                  camera=camera, wearer=body, time_s=t, speed_m_s=speed, objects=objects))
                if j >= 7:
                    sample_id = f'{clip_id}_t{j:02d}'
                    samples.append(dict(sample_id=sample_id, clip_id=clip_id, group_id=group_id,
                                        split=split, frame_indices=indices[-8:]))
                    contact = [first_contact([body[k] for k in ('x', 'y', 'z')], speed, objects, box)
                               for box in BODY_BOXES]
                    targets[sample_id] = [[int(c is not None and c <= h) for h in HORIZONS] for c in contact]
                    contacts[sample_id] = contact
    assert len(cases) == 768 and len(samples) == 432
    return dict(schema='nf-g7-contact-retina-spec-v1', map_sha256=MAP_SHA,
                sampling='SIMULATED_10_HZ_POSED_TRAJECTORY_NOT_WALLCLOCK_SENSOR',
                seed=70917, clips=clips, cases=cases, samples=samples,
                labels=dict(targets=targets, first_contact_s=contacts), body_boxes=BODY_BOXES,
                label_scope='TASK_OBJECT_BOXES_CONSTANT_BODY_X_VELOCITY_NOT_HUMAN_MESH_OR_ALL_SCENE_CONTACT')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    target = args.output.resolve()
    assert target.is_relative_to((Path(__file__).resolve().parents[4]/'artifacts.local').resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(specification(), stream, indent=2, allow_nan=False)

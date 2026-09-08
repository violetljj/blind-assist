"""One-site attached-fixture Development collection: 250 matched quartets."""
import copy
import random

from contextual_geometry import intrusion_metrics, target_contact
from contextual_sampling import FAMILIES, RELATIONS
from contextual_scene import preview, scene


SEED = 20260908
SCHEMA = 'city-contextual-headspace-1000-v1'
SITE_ID = 'street200_frontage_x-76_y15'
HEAD_HEIGHTS = dict(crossbar=1.67, cabinet=1.78, oblique_rod=1.78, hanging_sign=1.67)


def collection(template):
    """Build exactly 1,000 requests and fail if any intended state is infeasible.

    Within each quartet the camera, wearer, material, thickness and tilt are
    fixed. The scene constructor changes the fixture height and its attached
    supports together. All groups reuse the same street site; groups are not
    independent worlds or a source-disjoint train/test split.
    """
    result = preview(template)
    rng = random.Random(SEED)
    cases = []
    for index in range(250):
        family = FAMILIES[index % len(FAMILIES)]
        group_id = f'{SITE_ID}:quartet_{index:03d}'
        distance = rng.uniform(.8, 2.8)
        lateral = rng.uniform(-.1, .1)
        eye_height = rng.uniform(1.6, 1.8)
        pitch = rng.uniform(-8., 6.)
        tilt = rng.uniform(15., 35.) if family == 'oblique_rod' else 0.
        thickness = rng.uniform(.03, .08)
        height_jitter = rng.uniform(-.005, .005)
        heights = dict(CLEAR=2.2, BODY_ONLY=1.05, HEAD_ONLY=HEAD_HEIGHTS[family], BOTH=1.4)
        for relation in RELATIONS:
            height = heights[relation] + height_jitter
            world = scene(family, height, tilt, thickness)
            target_front = min(o['center_m'][0] - o['size_m'][0] / 2
                               for o in world['objects'] if o['target_part'])
            camera = dict(x=target_front - .18 - distance,
                          y=world['route_y_m'] - lateral,
                          z=world['floor_z_m'] + eye_height, pitch=pitch, yaw=0., roll=0.)
            wearer = dict(x=camera['x'], y=camera['y'], z=world['floor_z_m'],
                          pitch=0., yaw=0., roll=0.)
            objects = copy.deepcopy(world['objects'])
            contact = target_contact(objects, wearer)
            if contact['relation'] != relation:
                raise ValueError(f'Infeasible attached source {group_id}/{relation}: '
                                 f'computed {contact["relation"]}')
            name = f'contextual1000_g{index:03d}_{relation.lower()}'
            condition = dict(condition_id=name, family=family, desired_relation=relation,
                             subset='quartet', hard_kind=None, source_site_id=SITE_ID,
                             split_group_id=SITE_ID, counterfactual_parent_id=group_id,
                             distance_m=distance, lateral_offset_m=lateral,
                             eye_height_m=eye_height, camera_pitch_deg=pitch,
                             tilt_degrees=tilt, thickness_m=thickness,
                             height_m=height, common_height_jitter_m=height_jitter,
                             lighting_profile='daylight', boundary=False,
                             route_lateral_delta_m=0., occluder=False)
            cases.append(dict(name=name, group_id=group_id, variant_id=relation,
                              condition=condition, source_site_id=SITE_ID,
                              camera=camera, wearer=wearer, floor_z_m=world['floor_z_m'],
                              floor_check=False, objects=objects, scene_context=world['context'],
                              effective_target_height_m=height, geometric_contact=contact,
                              geometric_intrusion=intrusion_metrics(objects, wearer),
                              sun_intensity_scale=1., skylight_intensity_scale=1.))
    result.update(schema=SCHEMA, cases=cases, seed=SEED,
                  pair_export_mode='native_async', export_appearance=False,
                  export_dependencies=False, settling_ticks=32, settling_interval_s=0.,
                  exposure_ev100=12.,
                  purpose='1000_ATTACHED_FIXTURE_MATCHED_QUARTET_DEVELOPMENT_NO_MODEL_TRAINING',
                  suite_contract=dict(frames=1000, quartets=250, states_per_quartet=4,
                      geometry_counts={relation: 250 for relation in RELATIONS},
                      family_group_counts=dict(crossbar=63, cabinet=63, oblique_rod=62, hanging_sign=62),
                      source_worlds=1, source_site_id=SITE_ID,
                      body_boxes_version='contact_retina_spec',
                      intervention='Fixture height plus physically attached support adjustment only within each quartet',
                      split='Keep the entire source site together; quartet IDs are not independent-world split authority',
                      authority='Exact attached assembly cuboid geometry; visible native depth gaps remain in the denominator',
                      intrusion='Projected lateral occupancy somewhere in the horizon, not volume or risk severity'),
                  scope='Single-site synthetic attached-fixture Development. No independent-world claim, model training or safety score.')
    return result

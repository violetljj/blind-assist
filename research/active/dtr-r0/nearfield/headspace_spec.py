"""Eight compound geometry controls; target-only exact box sweeps, not mesh truth."""
import copy
from contact_retina_spec import BODY_BOXES, first_contact


def parts(family):
    """Local metre geometry. These are explicit cuboid proxies, not realistic assets."""
    shapes = {
        'horizontal_bar': [([0, 0, 0], [.08, 1.2, .08])],
        'low_branch': [([0, 0, 0], [.09, 1.2, .08]),
                       ([.15, -.18, .05], [.35, .07, .07]),
                       ([-.12, .25, -.04], [.3, .06, .06])],
        'side_sign': [([0, 0, 0], [.12, .9, .26]),
                      ([0, .58, .1], [.08, .4, .05])],
        'scaffold_tube': [([0, 0, 0], [.025, 1.5, .025])],
        'open_cabinet_door': [([0, 0, 0], [.035, .55, .28]),
                              ([-.045, .2, 0], [.07, .06, .05])],
        'stair_underside': [([-.24, 0, -.08], [.24, 1.0, .12]),
                            ([0, 0, 0], [.24, 1.0, .12]),
                            ([.24, 0, .08], [.24, 1.0, .12])],
        'open_window': [([0, 0, -.12], [.04, .55, .04]),
                        ([0, 0, .12], [.04, .55, .04]),
                        ([0, -.255, 0], [.04, .04, .24]),
                        ([0, .255, 0], [.04, .04, .24])],
        'hanging_rope': [([-.12, -.3, .04], [.018, .6, .018]),
                         ([0, 0, 0], [.018, .06, .10]),
                         ([.12, .3, -.04], [.018, .6, .018]),
                         ([-.06, -.015, .04], [.14, .018, .018]),
                         ([.06, .015, -.04], [.14, .018, .018])],
    }
    return [dict(name=f'{family}_{i}', center_m=c, size_m=s)
            for i, (c, s) in enumerate(shapes[family])]


FAMILIES = ('horizontal_bar', 'low_branch', 'side_sign', 'scaffold_tube',
            'open_cabinet_door', 'stair_underside', 'open_window', 'hanging_rope')
RELATIONS = ('BODY_ONLY', 'BOTH', 'HEAD_ONLY', 'CLEAR')


def target_contact(objects, wearer, horizon=3.):
    """Exact union of unrotated solid cuboids, closed-boundary constant-X sweep."""
    if horizon <= 0:
        raise ValueError('Positive horizon required')
    for obj in objects:
        if 'mesh_asset' in obj or 'primitive_asset' in obj or obj.get('rotation_deg'):
            raise ValueError('Analytic control accepts unrotated solid cubes only')
    origin = [wearer[k] for k in ('x', 'y', 'z')]
    if any(wearer.get(k, 0) != 0 for k in ('yaw', 'pitch', 'roll')):
        raise ValueError('Analytic control requires a straight world-X route')
    contacts = [first_contact(origin, 1., objects, box) for box in BODY_BOXES]
    hits = [d is not None and d <= horizon for d in contacts]
    return dict(relation={(False, False): 'CLEAR', (True, False): 'BODY_ONLY',
                          (False, True): 'HEAD_ONLY', (True, True): 'BOTH'}[tuple(hits)],
                first_contact_distance_m=[d if hit else None for d, hit in zip(contacts, hits)],
                authority='TARGET_CUBOID_UNION_ONLY_NOT_ALL_SCENE_OR_VISIBLE_SUPPORT')


def specification(template):
    """Freeze each target world before sampling two views; siblings move Z only."""
    base = next(c for c in template['cases'] if c['name'] == 'west_sidewalk')
    if base.get('objects') or base['camera']['yaw'] != 0 or base['camera']['roll'] != 0:
        raise ValueError('Empty world-X west sidewalk view required')
    floor = float(base['floor_z_m'])
    # Fixed scene anchor, independent of near/far view sampling below.
    anchor = [float(base['camera']['x']) + 2., float(base['camera']['y']), floor]
    worlds, cases = [], []
    for family in FAMILIES:
        local = parts(family)
        bottom = min(p['center_m'][2] - p['size_m'][2]/2 for p in local)
        top = max(p['center_m'][2] + p['size_m'][2]/2 for p in local)
        body, head = BODY_BOXES
        heights = (body[1][2] - .12 - top, body[1][2],
                   head[0][2] + .10 - bottom, head[1][2] + .15 - bottom)
        for relation, height in zip(RELATIONS, heights):
            objects = copy.deepcopy(local)
            for obj in objects:
                obj['center_m'] = [obj['center_m'][0]+anchor[0],
                                   obj['center_m'][1]+anchor[1],
                                   obj['center_m'][2]+floor+height]
            world_id = f'headspace_{family}_{relation}'
            worlds.append(dict(world_id=world_id, family=family, objects=objects,
                               intervention=dict(axis='z', translation_m=height)))
            front = min(o['center_m'][0]-o['size_m'][0]/2 for o in objects)
            for view, gap in (('near', .65), ('far', 2.25)):
                camera = copy.deepcopy(base['camera'])
                camera['x'] = front - BODY_BOXES[0][1][0] - gap
                wearer = dict(x=camera['x'], y=camera['y'], z=floor, yaw=0., pitch=0., roll=0.)
                contact = target_contact(objects, wearer)
                if contact['relation'] != relation:
                    raise ValueError(f'{world_id}: computed {contact} differs from requested {relation}')
                group = f'headspace_{family}_{view}'
                cases.append(dict(name=f'{group}_{relation}', group_id=group,
                    group_type='SAME_OBJECT_HEIGHT_QUARTET', variant_id=relation,
                    world_id=world_id, family=family, camera=camera, wearer=wearer,
                    floor_z_m=floor, floor_check=False, objects=copy.deepcopy(objects),
                    expected_target_contact=contact))
    result = copy.deepcopy(template)
    result.update(schema='city-headspace-controls-v1', cases=cases, worlds=worlds,
        export_dependencies=False, settling_ticks=32, settling_interval_s=0.,
        suite_contract=dict(groups=16, frames=64, families=list(FAMILIES),
            relations=list(RELATIONS), body_boxes=BODY_BOXES, horizon_m=3.,
            geometry='CUBOID_COMPOUND_PROXIES_NOT_REAL_ASSETS; exact component union, never whole-object bbox',
            clear='No target intersection within 3m; not certified all-scene free space',
            sampling='Two straight-route views per fixed world; height-only counterfactual worlds',
            boundary='Controlled engineering extension, not complete persistent-world/NavMesh generation or training',
            model_contract='RGB/calibration only; worlds, names, expected contacts and grouping evaluator-only'))
    return result

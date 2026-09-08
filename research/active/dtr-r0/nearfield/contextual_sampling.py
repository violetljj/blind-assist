"""Deterministic contextual scene requests, not evaluated geometry or risk labels.

All lengths are metres in a floor-relative frame: X forward, Y lateral, Z up.
Desired relations and boundary membership are sampling intent only. The scene
builder must derive actual contact from the complete attached geometry.
"""
from copy import deepcopy
from itertools import product

from contact_retina_spec import BODY_BOXES


FAMILIES = ('crossbar', 'cabinet', 'oblique_rod', 'hanging_sign')
RELATIONS = ('CLEAR', 'BODY_ONLY', 'HEAD_ONLY', 'BOTH')
POSITIONS = {'LEFT': -.1, 'CENTER': 0., 'RIGHT': .1}
DISTANCES_M = (1., 2.5)
HEIGHTS_M = {'BODY_ONLY': 1.1, 'BOTH': 1.4, 'HEAD_ONLY': 1.65, 'CLEAR': 2.2}
HARD_COUNTS = {'body_head_boundary': 8, 'head_clear_boundary': 6,
               'thin': 4, 'oblique': 4, 'occlusion': 4, 'light': 2,
               'pitch': 2, 'path': 2}


def conditions(site_id='city_contextual_site_00'):
    """Return 96 factorial core conditions and 32 explicitly allocated controls.

    Core counterfactual parents bind every relation, lateral position and distance
    from one source family. Hard parents bind their matched factor sweep. Split
    groups bind all core and hard conditions from the same source family/site;
    callers must retain this grouping when partitioning generated views.
    """
    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError('A nonempty source site identifier is required')
    result = []

    def add(family, desired_relation, position='CENTER', distance_m=1.,
            hard_kind=None, parent=None, **changes):
        subset = 'hard' if hard_kind else 'core'
        row = dict(condition_id=f'{site_id}_{subset}_{len(result):03d}',
                   source_site_id=site_id, family=family,
                   counterfactual_parent_id=f'{site_id}:{parent or family + ":core"}',
                   split_group_id=f'{site_id}:{family}', subset=subset,
                   hard_kind=hard_kind, desired_relation=desired_relation,
                   position=position, lateral_offset_m=POSITIONS[position],
                   distance_m=distance_m, height_m=HEIGHTS_M[desired_relation],
                   tilt_degrees=30. if family == 'oblique_rod' else 0.,
                   thickness_m=.06, camera_pitch_deg=0.,
                   lighting_profile='daylight', occluder=False,
                   route_lateral_delta_m=0., boundary=False, boundary_name=None)
        row.update(changes)
        result.append(row)

    for family, relation, position, distance in product(
            FAMILIES, RELATIONS, POSITIONS, DISTANCES_M):
        add(family, relation, position, distance)

    for family in FAMILIES:
        for relation, height in (('BODY_ONLY', 1.38), ('HEAD_ONLY', 1.42)):
            add(family, relation, hard_kind='body_head_boundary',
                parent=f'{family}:body_head_boundary', height_m=height,
                boundary=True, boundary_name='BODY_HEAD')
    for family in ('crossbar', 'cabinet', 'hanging_sign'):
        for relation, height in (('HEAD_ONLY', 1.83), ('CLEAR', 1.87)):
            add(family, relation, hard_kind='head_clear_boundary',
                parent=f'{family}:head_clear_boundary', height_m=height,
                boundary=True, boundary_name='HEAD_CLEAR')
    for family, thickness in product(('crossbar', 'oblique_rod'), (.01, .02)):
        add(family, 'HEAD_ONLY', hard_kind='thin', parent=f'{family}:thin',
            thickness_m=thickness)
    for tilt in (15., 30., 45., 60.):
        add('oblique_rod', 'HEAD_ONLY', hard_kind='oblique',
            parent='oblique_rod:oblique', tilt_degrees=tilt)
    for family, occluder in product(('cabinet', 'hanging_sign'), (False, True)):
        add(family, 'HEAD_ONLY', hard_kind='occlusion',
            parent=f'{family}:occlusion', occluder=occluder)
    for lighting in ('daylight', 'darker'):
        add('hanging_sign', 'HEAD_ONLY', hard_kind='light',
            parent='hanging_sign:light', lighting_profile=lighting)
    for pitch in (-15., 10.):
        add('crossbar', 'HEAD_ONLY', hard_kind='pitch',
            parent='crossbar:pitch', camera_pitch_deg=pitch)
    for lateral in (-.35, .35):
        add('cabinet', 'HEAD_ONLY', hard_kind='path',
            parent='cabinet:path', route_lateral_delta_m=lateral)
    return result


def specification(site_id='city_contextual_site_00'):
    """Pure manifest; intentionally contains no contact or truth annotations."""
    return dict(schema='nf-contextual-sampling-v1',
                coordinate_frame='FLOOR_RELATIVE_X_FORWARD_Y_LATERAL_Z_UP',
                relation_authority='DESIRED_SAMPLING_INTENT_NOT_GEOMETRY_TRUTH',
                boundary_authority='ORTHOGONAL_SAMPLING_FLAG_NOT_FIFTH_RELATION',
                geometry_policy=('Heights are advisory target centers; panel thickness '
                                 'and attached supports may make intended cells infeasible. '
                                 'Retain those cells as coverage gaps; never detach a '
                                 'physical support to force the desired relation.'),
                body_boxes=deepcopy(BODY_BOXES),
                conditions=conditions(site_id))

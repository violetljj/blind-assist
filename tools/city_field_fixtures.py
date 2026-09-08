"""Reusable controlled synthetic maintenance assemblies for CitySample collection.

These are Engine one-metre cubes, not natural CitySample obstacle assets. The
material palette is reused from nearfield/contextual_scene.py; this module does
not assert live asset availability or native pavement clearance. The caller must
admit a real ground surface and adequate side space before placing the portal.
anchor_m is (x, y, floor_z) at the portal; approach is local positive X. Stable
instance IDs describe assembly parts across matched conditions and camera views.
"""
import importlib.util
import math
import re
from pathlib import Path

_source = Path(__file__).resolve().parents[1] / 'research/active/dtr-r0/nearfield/contextual_scene.py'
_spec = importlib.util.spec_from_file_location('_city_field_contextual_palette', _source)
_palette = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_palette)
MATERIALS = dict(_palette.MATERIALS)
KINDS = ('clear', 'thin_pole', 'body_protrusion', 'head_bar', 'suspended_sign')
PROVENANCE = 'CONTROLLED_SYNTHETIC_SUPPORTED_ASSEMBLY_NOT_NATURAL_CITY_ASSET'


def _pose(anchor_m, yaw_deg):
    if len(anchor_m) != 3 or not all(math.isfinite(float(v)) for v in (*anchor_m, yaw_deg)):
        raise ValueError('anchor_m and yaw_deg must be finite')
    angle = math.radians(yaw_deg)
    return math.cos(angle), math.sin(angle)


def fixture_objects(kind, anchor_m, yaw_deg, instance_prefix):
    """Return capture-compatible cube objects; clear retains the identical portal.

    All support_parent references are object names, except ground. Feet and
    uprights sit outside both sweeps. The common top beam is above 1.85 m.
    Dimensions are engineering placement geometry, not rendered visibility truth.
    """
    if kind not in KINDS:
        raise ValueError('Unknown fixture condition: ' + str(kind))
    if not isinstance(instance_prefix, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', instance_prefix):
        raise ValueError('instance_prefix must be a safe stable site identifier')
    co, si = _pose(anchor_m, yaw_deg)
    objects = []

    def box(part, center, size, material='metal', parent='ground', target=False):
        x, y, z = center
        name = instance_prefix + '__' + part
        objects.append(dict(name=name, instance_id=name,
            center_m=[anchor_m[0]+co*x-si*y, anchor_m[1]+si*x+co*y, anchor_m[2]+z],
            size_m=list(size), rotation_deg=dict(pitch=0., yaw=float(yaw_deg), roll=0.),
            material_asset=MATERIALS[material],
            support_parent=parent if parent == 'ground' else instance_prefix+'__'+parent,
            target_part=target, geometry_provenance=PROVENANCE))

    for side in ('left', 'right'):
        y = -.85 if side == 'left' else .85
        box(side+'_foot', (0., y, .05), (.50, .32, .10), 'black')
        box(side+'_upright', (0., y, 1.30), (.08, .08, 2.50), parent=side+'_foot')
    box('top_beam', (0., 0., 2.53), (.10, 1.78, .12), parent='left_upright')

    if kind == 'thin_pole':
        box('pole_foot', (0., 0., .025), (.20, .20, .05), 'black')
        box('thin_pole', (0., 0., 1.025), (.045, .045, 2.), parent='pole_foot', target=True)
    elif kind == 'body_protrusion':
        box('body_clamp', (0., -.85, .95), (.14, .14, .16), 'brass', 'left_upright')
        # Butt joint at clamp's inner Y face (-.78), rather than overlapping the
        # clamp/upright. Overlap made isolated arm rays hit a different component
        # only 2.5 cm nearer, failing the unchanged strict identity check.
        box('body_arm', (0., -.365, .95), (.09, .83, .12), parent='body_clamp', target=True)
    elif kind == 'head_bar':
        for side in ('left', 'right'):
            box('head_clamp_'+side, (0., -.85 if side == 'left' else .85, 1.65),
                (.14, .14, .14), 'brass', side+'_upright')
        box('head_crossbar', (0., 0., 1.65), (.065, 1.70, .065), parent='head_clamp_left', target=True)
    elif kind == 'suspended_sign':
        for side in ('left', 'right'):
            box('hanger_'+side, (0., -.30 if side == 'left' else .30, 2.185),
                (.028, .028, .69), 'black', 'top_beam')
        box('sign_panel', (0., 0., 1.68), (.075, .78, .32), 'blue', 'hanger_left', True)
    validate_fixture(objects, anchor_m, yaw_deg)
    return objects


def corridor_objects(anchor_m, yaw_deg, instance_prefix, length_m=12., width_m=1.):
    """Grounded side guardrails for an explicitly controlled narrow passage.

    width_m is the unobstructed distance between the inward edges of all parts,
    including feet. The corridor spans local -length/2 .. +length/2; it is not a
    claim that the native City street is narrow. Needs admitted flat pavement.
    """
    if not all(math.isfinite(v) for v in (length_m, width_m)) or length_m <= 0 or width_m <= .56:
        raise ValueError('Corridor requires positive length and width greater than BODY sweep')
    if not isinstance(instance_prefix, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', instance_prefix):
        raise ValueError('instance_prefix must be a safe stable site identifier')
    co, si = _pose(anchor_m, yaw_deg)
    objects = []

    def box(part, xyz, size, parent='ground', material='metal'):
        x,y,z = xyz
        name = instance_prefix+'__corridor_'+part
        objects.append(dict(name=name, instance_id=name,
            center_m=[anchor_m[0]+co*x-si*y, anchor_m[1]+si*x+co*y, anchor_m[2]+z],
            size_m=list(size), rotation_deg=dict(pitch=0., yaw=float(yaw_deg), roll=0.),
            material_asset=MATERIALS[material], support_parent='ground' if parent == 'ground'
            else instance_prefix+'__corridor_'+parent, target_part=False,
            geometry_provenance=PROVENANCE))

    spans = max(1, math.ceil(length_m/3.))
    for side in ('left', 'right'):
        y = (-1 if side == 'left' else 1)*(width_m/2+.10)
        for i in range(spans+1):
            x = -length_m/2 + i*length_m/spans
            foot, post = f'{side}_foot_{i}', f'{side}_post_{i}'
            box(foot, (x,y,.04), (.30,.20,.08), material='black')
            box(post, (x,y,.60), (.07,.07,1.12), foot)
        for level,z in (('lower',.45), ('upper',1.10)):
            box(f'{side}_{level}_rail', (0.,y,z), (length_m,.07,.07), f'{side}_post_0')
    hits = validate_fixture(objects, anchor_m, yaw_deg)
    if hits['BODY'] or hits['HEAD']:
        raise ValueError('Corridor intrudes into centre sweep')
    return objects


def _local_bounds(obj, anchor_m, yaw_deg):
    co, si = _pose(anchor_m, yaw_deg)
    rotation = obj['rotation_deg']
    if abs(rotation['pitch']) > 1e-8 or abs(rotation['roll']) > 1e-8 or abs(
            (rotation['yaw']-yaw_deg+180) % 360-180) > 1e-8:
        raise ValueError('Fixture cubes must align with route yaw')
    center = obj['center_m']; size = obj['size_m']
    if len(center) != 3 or len(size) != 3 or not all(math.isfinite(float(v)) for v in (*center, *size)) or min(size) <= 0:
        raise ValueError('Invalid cube geometry')
    dx, dy = center[0]-anchor_m[0], center[1]-anchor_m[1]
    local = [co*dx+si*dy, -si*dx+co*dy, center[2]-anchor_m[2]]
    return ([v-s/2 for v,s in zip(local,size)], [v+s/2 for v,s in zip(local,size)])


def _touches(a, b, tolerance=1e-7):
    return all(a[0][i] <= b[1][i]+tolerance and b[0][i] <= a[1][i]+tolerance for i in range(3))


def validate_fixture(objects, anchor_m, yaw_deg):
    """Reject floating/cyclic supports using actual transformed cube bounds.

    This checks geometric attachment, not load-bearing engineering or contact
    with unknown native pavement. Returns parts intersecting a through-portal
    BODY (width .56 m, z .65..1.4) or HEAD (.36 m, z 1.4..1.85) sweep.
    """
    by_name = {o['name']: o for o in objects}
    if len(by_name) != len(objects) or len({o['instance_id'] for o in objects}) != len(objects):
        raise ValueError('Duplicate fixture identity')
    bounds = {name: _local_bounds(o, anchor_m, yaw_deg) for name,o in by_name.items()}
    for name, obj in by_name.items():
        parent = obj['support_parent']
        if bounds[name][0][2] < -1e-7:
            raise ValueError('Fixture penetrates the admitted floor: ' + name)
        if parent == 'ground':
            if abs(bounds[name][0][2]) > 1e-7:
                raise ValueError('Ground support is floating: ' + name)
        elif parent not in bounds or not _touches(bounds[name], bounds[parent]):
            raise ValueError('Detached support: ' + name)
        seen = {name}
        while parent != 'ground':
            if parent in seen:
                raise ValueError('Cyclic support graph')
            seen.add(parent)
            parent = by_name[parent]['support_parent']
    contacts = {}
    for band, width, low, high in (('BODY', .56, .65, 1.4), ('HEAD', .36, 1.4, 1.85)):
        query = ([-.5, -width/2, low], [.5, width/2, high])
        contacts[band] = sorted(name for name, bound in bounds.items() if _touches(bound, query))
    return contacts

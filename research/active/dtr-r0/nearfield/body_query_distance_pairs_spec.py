"""Freeze 32 rigid forward-translation pairs from accepted 5000-frame sources.

Scalar metadata/analytic work only. Native visible-count labels remain authoritative;
this generator never reads predictions, starts capture, or trains a model.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from body_query_5000_spec import FAMILIES, read, sha
from contextual_geometry import target_contact
from contact_retina_spec import BODY_BOXES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def local_objects(case):
    """Undo the source's rigid yaw placement for the analytic cuboid union."""
    objects = copy.deepcopy(case['objects'])
    wearer = case['wearer']
    angle = math.radians(wearer['yaw'])
    co, si = math.cos(angle), math.sin(angle)
    for obj in objects:
        x, y, z = obj['center_m']
        x, y = x-wearer['x'], y-wearer['y']
        obj['center_m'] = [co*x+si*y, -si*x+co*y, z-wearer['z']]
        obj['rotation_deg']['yaw'] -= wearer['yaw']
    return objects


def preflight(case):
    objects = local_objects(case)
    result = target_contact(objects, dict(x=0., y=0., z=0.))
    assert result['relation'] == 'HEAD_ONLY', (case['name'], result)
    witnesses = set(result['object_witness_names']['HEAD'])
    intervals = [(o['center_m'][0]-o['size_m'][0]/2,
                  o['center_m'][0]+o['size_m'][0]/2)
                 for o in objects if o['name'] in witnesses]
    extent = [min(a for a, _ in intervals), max(b for _, b in intervals)]
    boundary = BODY_BOXES[1][1][0]+1.5
    if case['declared_range'] == 'near':
        assert 0.13 < extent[0] and extent[1] < boundary, (case['name'], extent)
    else:
        assert boundary < extent[0] and extent[1] < 3.13, (case['name'], extent)
    return dict(name=case['name'], relation=result['relation'],
                head_witness_forward_extent_m=extent, head_range_boundary_m=boundary,
                head_witnesses=sorted(witnesses),
                authority='ANALYTIC_SUPPLIED_SOLIDS_ONLY_NATIVE_VISIBILITY_PENDING')


def validate_pair(near, far):
    """Exact structural equality, permitting only declared translation metadata."""
    a, b = copy.deepcopy(near), copy.deepcopy(far)
    angle = math.radians(a['camera']['yaw'])
    delta = b['condition']['distance_m']-a['condition']['distance_m']
    expected = [delta*math.cos(angle), delta*math.sin(angle), 0.]
    for oa, ob in zip(a['objects'], b['objects'], strict=True):
        actual = [v-u for u, v in zip(oa['center_m'], ob['center_m'])]
        assert all(abs(v-e) < 1e-9 for v, e in zip(actual, expected)), actual
        assert oa['center_m'][2] == ob['center_m'][2]
        oa.pop('center_m'); ob.pop('center_m')
    for c in (a, b):
        c.pop('name'); c.pop('declared_range')
        c['condition'].pop('condition_id'); c['condition'].pop('distance_m')
    assert a == b, near['pair_id']
    return expected


def make_pair(raw, near_distance, far_distance, source_hash):
    pair_id = 'bqdist-'+raw['site_id']+'-'+raw['condition']['family']
    result = []
    for endpoint, distance in (('near', near_distance), ('far', far_distance)):
        c = copy.deepcopy(raw)
        delta = distance-raw['condition']['distance_m']
        angle = math.radians(c['camera']['yaw'])
        for obj in c['objects']:
            obj['center_m'][0] += delta*math.cos(angle)
            obj['center_m'][1] += delta*math.sin(angle)
        c.pop('local_geometry_sha256', None)
        c.update(name=pair_id+'-'+endpoint, pair_id=pair_id, group_id=pair_id,
                 parent_group_id=pair_id, declared_range=endpoint,
                 geometry_frame='Source rigid world geometry translated only along camera forward',
                 source_case_name=raw['name'], source_spec_sha256=source_hash,
                 source_case_sha256=digest(raw), source_role='CONSUMED_DEVELOPMENT_DISTANCE_PAIR')
        c['condition'].update(condition_id=c['name'], counterfactual_parent_id=pair_id,
                              subset='strict_distance_pair', query_control='HEAD_ONLY',
                              distance_m=distance)
        result.append(c)
    translation = validate_pair(*result)
    checks = [preflight(c) for c in result]
    return result, dict(pair_id=pair_id, source_case_name=raw['name'],
        source_case_sha256=digest(raw), source_spec_sha256=source_hash,
        near_distance_m=near_distance, far_distance_m=far_distance,
        forward_translation_world_m=translation,
        dimensions=[dict(name=o['name'], size_m=o['size_m'], rotation_deg=o['rotation_deg'],
                        material_asset=o.get('material_asset'), center_height_above_floor_m=o['center_m'][2]-raw['floor_z_m'])
                    for o in raw['objects']], analytic_checks=checks,
        strict_pair_equality='PASS')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    artifact_root = (Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    assert a.output.resolve().is_relative_to(artifact_root) and not a.output.exists()
    sites_path = a.source_root/'accepted-sites-primary-v1.json'
    sites = read(sites_path)
    regions = sites['regions'][:4]
    assert [r['region_id'] for r in regions] == ['big01', 'big03', 'big05', 'big06']
    specs, checks = [], []
    for region in regions:
        source_path = a.source_root/'full-specs-primary-v1'/(region['region_id']+'.json')
        source = read(source_path)
        source_hash = sha(source_path)
        spec = copy.deepcopy(source)
        spec.update(cases=[], schema='body-query-distance-pairs-v1',
                    purpose='32_STRICT_DISTANCE_PAIRS_FROZEN_EXPANDED_B_NO_TRAINING',
                    source_role='CONSUMED_DEVELOPMENT_DISTANCE_PAIR',
                    scope='Controlled same-source Development diagnostic; no generalization or safety claim',
                    provenance=dict(source_spec=str(source_path.resolve()), source_spec_sha256=source_hash,
                                    accepted_sites_sha256=sha(sites_path), generator_sha256=sha(__file__)))
        for rank, site in enumerate(region['accepted_sites'][:2]):
            for family in FAMILIES:
                candidates = [c for c in source['cases'] if c['site_id'] == site['site_id']
                              and c['condition']['family'] == family and c['variant_id'] == 'HEAD_ONLY']
                assert len(candidates) == 1
                pair, check = make_pair(candidates[0], (1., 1.3)[rank], (2.1, 2.6)[rank], source_hash)
                spec['cases'].extend(pair); checks.append(check)
        assert len(spec['cases']) == 16
        specs.append((region['region_id'], spec))
    assert len(checks) == 32
    a.output.mkdir(parents=True)
    outputs = []
    for region_id, spec in specs:
        path = a.output/(region_id+'.json')
        path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
        outputs.append(dict(region_id=region_id, spec=str(path.resolve()), sha256=sha(path), frames=16))
    report = dict(status='SPEC_ONLY_NOT_CAPTURED', frames=64, pairs=32,
                  selection='First four primary regions, first two accepted sites in saved order; original per-site HEAD_ONLY family templates; no predictions accessed',
                  source_schema='body-query-5000-source-v1', outputs=outputs,
                  source_root=str(a.source_root.resolve()), accepted_sites_sha256=sha(sites_path),
                  geometry_preflight='64_HEAD_ONLY_AND_EXCLUSIVE_INTENDED_HEAD_DISTANCE_EXTENTS_PASS',
                  strict_pair_equality='32_PASS', pair_details=checks,
                  limitations=['Native scene BODY/HEAD evidence and near/far exclusivity must be confirmed after capture.',
                               'Fixture assembly including supports translates rigidly; native background and camera remain fixed.',
                               'Original per-site dimensions retained; same site ranks reuse templates across regions.',
                               'No UE capture, inference, training, or generalization evaluation performed by this generator.'])
    (a.output/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'pair_details'}))


if __name__ == '__main__':
    main()

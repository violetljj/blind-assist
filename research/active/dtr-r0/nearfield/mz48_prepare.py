"""Freeze the 2,560-frame multi-site rich-object source and 64-frame canary."""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


FAMILIES = ('pipe', 'ladder', 'pouch', 'birch', 'oblique_rod')
RELATIONS = ('HEAD_ONLY', 'BODY_ONLY', 'BOTH', 'CLEAR')
SCALES = (.75, .90, 1.05, 1.20)
NEAR = (.85, .90, .95, 1.00)
FAR = (1.90, 2.10, 2.30, 2.50)
LATERAL = (-.06, .02, .06, -.02)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def rotation(degrees):
    p, y, r = (math.radians(degrees[k]) for k in ('pitch', 'yaw', 'roll'))
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    return np.array([[cp*cy, sr*sp*cy-cr*sy, -cr*sp*cy-sr*sy],
                     [cp*sy, sr*sp*sy+cr*cy, -cr*sp*sy+sr*cy], [sp, -sr*cp, cr*cp]])


def world(camera, floor, point):
    yaw = math.radians(camera['yaw']); x, y, z = point
    return [camera['x']+x*math.cos(yaw)-y*math.sin(yaw),
            camera['y']+x*math.sin(yaw)+y*math.cos(yaw), floor+z]


def prepare(root, output):
    work = root / 'artifacts.local/work'
    original_path = work / 'mz42-rich-objects-20260910/spec-v2/spec.json'
    metadata_path = work / 'mz42-rich-objects-20260910/returned-v1/metadata-v1/metadata.json'
    admission_path = work / 'mz36-new-source-20260910/admission-v1/result.json'
    source = read(original_path); metadata = read(metadata_path); admission = read(admission_path)
    assert sha(original_path) == 'cc76970fe3f358c4cffca24d423cb1d81a73720d609a4e7dc21b8f3067cf7600'
    assert sha(metadata_path) == '0f420f4c0005419e3049b7fec30a7f34dd18f33ed82322fe806181a01e8a18c8'
    assert admission['status'] == 'PASS'
    inputs = {str(p): sha(p) for p in (original_path, metadata_path, admission_path, Path(__file__))}
    templates = {(c['condition']['family'], c['variant_id']): c for c in source['cases']}
    sites, region_specs = [], {}
    for region in ('dense_candidate_05', 'dense_candidate_06'):
        path = work / 'mz36-new-source-20260910/full-specs-v1' / (region + '.json')
        region_specs[region] = read(path); inputs[str(path)] = sha(path)
        candidates = {c['site_id']: c for c in region_specs[region]['cases']}
        for yaw in (0, 180, -90, 90):
            candidates_yaw = []
            for site_id, case in sorted(candidates.items()):
                rows = [r for r in admission['records'] if r['site_id'] == site_id]
                if case['camera']['yaw'] == yaw and rows and all(r['group_accepted'] and all(r['known']) for r in rows):
                    assert case['camera']['pitch'] == case['camera']['roll'] == 0
                    candidates_yaw.append(case)
            assert candidates_yaw, (region, yaw)
            sites.append(candidates_yaw[0])
    assert len({s['site_id'] for s in sites}) == 8
    assert 'mz36_dense_candidate_05_site_003' not in {s['site_id'] for s in sites}
    cases, canary_pairs, audit_pairs = [], set(), set()
    for site_index, site in enumerate(sites):
        for j in range(4):
            canary_pairs.add((site_index, (site_index+j) % 5, j, (site_index+j) % 2, (site_index+2*j) % 4))
        missing_family = (site_index+4) % 5
        audit_pairs.add((site_index, missing_family, site_index % 4, site_index % 2, site_index % 4))
        for family_index, family in enumerate(FAMILIES):
            for relation_index, relation in enumerate(RELATIONS):
                original = templates[(family, relation)]
                for range_index, declared_range in enumerate(('near', 'far')):
                    for profile in range(4):
                        front = (NEAR if declared_range == 'near' else FAR)[profile]
                        target_objects, target_bounds = [], []
                        for i, obj in enumerate(o for o in original['objects'] if o['target_part']):
                            target = copy.deepcopy(obj)
                            if family != 'oblique_rod':
                                scale = obj['scale'] * SCALES[profile]
                                asset = metadata['assets'][family]
                                local = np.array(list(itertools.product(*zip(asset['bounds_min_cm'], asset['bounds_max_cm'])))) * (.01*scale)
                                rotated = local @ rotation(obj['rotation_deg']).T
                                low, high = rotated.min(0), rotated.max(0)
                                lateral = LATERAL[profile] + (obj['center_m'][1]-original['camera']['y'])
                                if relation == 'CLEAR':
                                    lateral = 1.10 + LATERAL[profile] + i*.08
                                base_z = obj['center_m'][2]-original['floor_z_m']
                                anchor = np.array([front+i*.10, lateral, base_z])
                                origin = anchor - np.array([low[0], (low[1]+high[1])/2, low[2]])
                                low, high = low+origin, high+origin
                                target['scale'] = scale
                                target['placement'] = 'actor_origin'
                            else:
                                origin = np.array([front+.10, obj['center_m'][1]-original['camera']['y']+LATERAL[profile],
                                                   obj['center_m'][2]-original['floor_z_m']])
                                half = np.array(obj['size_m'])/2
                                rotated = np.array(list(itertools.product(*zip(-half, half)))) @ rotation(obj['rotation_deg']).T
                                low, high = rotated.min(0)+origin, rotated.max(0)+origin
                            target['center_m'] = world(site['camera'], site['floor_z_m'], origin)
                            target['rotation_deg']['yaw'] += site['camera']['yaw']
                            target_objects.append(target)
                            target_bounds.append(dict(name=target['name'], camera_aligned_min_m=low.tolist(), camera_aligned_max_m=high.tolist()))
                            assert low[0] > (.18 if declared_range == 'near' else 1.68)
                            assert high[0] < (1.63 if declared_range == 'near' else 3.13)
                        # A state-independent scaffold occurs in every family/relation equally.
                        supports = []
                        for obj in templates[('oblique_rod', 'HEAD_ONLY')]['objects']:
                            if obj['target_part']:
                                continue
                            support = copy.deepcopy(obj)
                            sign = -1 if obj['name'].endswith('_-1') else 1
                            x = front+.10 + (1.50 if obj['name'].startswith('rear_') else .75 if obj['name'].startswith('side_rail') else 0.)
                            z = 1.50 if obj['name'].startswith('clamp') else obj['center_m'][2]-templates[('oblique_rod', 'HEAD_ONLY')]['floor_z_m']
                            y = sign*.80
                            support['center_m'] = world(site['camera'], site['floor_z_m'], [x, y, z])
                            support['rotation_deg']['yaw'] = site['camera']['yaw']
                            assert abs(y)-obj['size_m'][1]/2 > .28
                            supports.append(support)
                        assert len(supports) == 12
                        pair_key = (site_index, family_index, relation_index, range_index, profile)
                        pair_id = f"mz48-{site['site_id']}-{family}-{relation}-{declared_range}-p{profile}"
                        expected = {'HEAD_ONLY': [0, 1], 'BODY_ONLY': [1, 0], 'BOTH': [1, 1], 'CLEAR': [0, 0]}[relation]
                        events = [0]*4
                        events[range_index], events[2+range_index] = expected
                        for supported in (False, True):
                            case = {k: copy.deepcopy(site[k]) for k in ('camera', 'wearer', 'floor_z_m', 'site_id', 'region_id')}
                            context = 'supported' if supported else 'unsupported'
                            case.update(name=pair_id+'-'+context, group_id=pair_id, pair_id=pair_id,
                                variant_id=relation, declared_range=declared_range, source_role='DEV_ONLY',
                                floor_check=False, probe_native_floor=True, profile_id=profile, support_context=context,
                                expected_events=events, placement_bounds=target_bounds, canary=pair_key in canary_pairs,
                                native_audit_sample=pair_key in canary_pairs or pair_key in audit_pairs,
                                objects=copy.deepcopy(target_objects)+(copy.deepcopy(supports) if supported else []),
                                condition=dict(family=family, desired_relation=relation, condition_id=pair_id+'-'+context))
                            cases.append(case)
    assert len(cases) == 2560 and sum(c['canary'] for c in cases) == 64
    assert sum(c['native_audit_sample'] for c in cases) == 80
    for i in range(0, len(cases), 2):
        assert cases[i]['pair_id'] == cases[i+1]['pair_id']
        assert cases[i]['objects'] == [o for o in cases[i+1]['objects'] if o['target_part']]
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    for region in region_specs:
        subset = [c for c in cases if c['region_id'] == region and c['canary']]
        shards.append(('canary-'+region, region, subset, True))
    for site in sites:
        subset = [c for c in cases if c['site_id'] == site['site_id'] and not c['canary']]
        assert len(subset) == 312
        shards.append(('main-'+site['site_id'], site['region_id'], subset, False))
    shard_rows = []
    for name, region, subset, canary in shards:
        spec = {k: copy.deepcopy(v) for k, v in source.items() if k not in ('cases', 'provenance')}
        spec.update(schema='mz48-rich-kilotier-v1', cases=subset,
            world_partition_region_m=region_specs[region]['world_partition_region_m'],
            scope='Controlled synthetic multi-site rich shapes; native labels independent from intended relations',
            provenance=dict(inputs=inputs, fixed_total_budget=2560, stage='canary' if canary else 'main'))
        path = output / 'shards' / (name+'.json'); write(path, spec)
        shard_rows.append(dict(shard_id=name, frames=len(subset), canary=canary, path=str(path), sha256=sha(path)))
    write(output/'manifest.json', dict(status='FROZEN_BEFORE_CAPTURE', inputs=inputs,
        total_frames=2560, canary_frames=64, main_frames=2496, native_audit_frames=80,
        sites=[{k:s[k] for k in ('site_id', 'region_id', 'camera', 'floor_z_m')} for s in sites],
        families=list(FAMILIES), relations=list(RELATIONS), scales=SCALES, near_front_m=NEAR, far_front_m=FAR,
        lateral_offset_m=LATERAL, target_pair_dictionary_equality=True,
        support_recipe='Twelve original components, fixed clamp1.5m, lateral+-0.8m; state-independent in all families',
        shards=shard_rows, new_model_inference_frames=0, training_steps=0,
        canary_gate='All64 source/floor/target-placement checks pass; actual labels and paired native masks reported; root visual admission before main',
        resume='Never repeat successful frame IDs; mechanically resume only untouched frozen shards after proving prior process release'))
    print(json.dumps(dict(status='FROZEN_BEFORE_CAPTURE', frames=2560, canary=64, shards=len(shards),
                          manifest_sha256=sha(output/'manifest.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.output.resolve())

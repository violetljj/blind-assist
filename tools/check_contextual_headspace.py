"""Retain native visibility gaps separately from full assembly geometry labels."""
import argparse
import collections
import json
import sys
from pathlib import Path
from make_city_obstacle_suite import REPO, under_artifacts, read, sha
sys.path.insert(0,str(REPO/'research/active/dtr-r0/nearfield'))
from contextual_geometry import target_contact, intrusion_metrics
from contextual_collection1000 import SCHEMA as COLLECTION1000_SCHEMA, SITE_ID


def evaluate(spec, native):
    """Pure geometry/coverage check; native visibility gaps remain explicit rows."""
    full1000 = spec['schema'] == COLLECTION1000_SCHEMA
    expected = 1000 if full1000 else 128
    if (spec['schema'] not in ('city-contextual-headspace-128-v1', COLLECTION1000_SCHEMA)
            or len(spec['cases'])!=expected or len(native['rows'])!=expected):
        raise ValueError('Incomplete contextual source')
    if native['status'] != 'PASS':
        raise ValueError('Unverified native capture')
    if full1000 and len({c['name'] for c in spec['cases']}) != 1000:
        raise ValueError('Collection1000 requires 1000 unique frame identities')
    relations={(0,0):'CLEAR',(1,0):'BODY_ONLY',(0,1):'HEAD_ONLY',(1,1):'BOTH'}
    rows=[]
    for case,observed in zip(spec['cases'],native['rows']):
        if case['name']!=observed['name']:
            raise ValueError('Row identity differs')
        geometry=target_contact(case['objects'],case['wearer'])
        if geometry!=case['geometric_contact']:
            raise ValueError('Geometry reference differs')
        if full1000 and intrusion_metrics(case['objects'],case['wearer']) != case['geometric_intrusion']:
            raise ValueError('Intrusion reference differs')
        visible=relations.get(tuple(observed['body_head_visible_targets']),'UNKNOWN')
        rows.append(dict(name=case['name'],family=case['condition']['family'],
            subset=case['condition']['subset'],hard_kind=case['condition']['hard_kind'],
            intended=case['condition']['desired_relation'],actual_geometry=geometry['relation'],
            visible_relation=visible,agrees=visible==geometry['relation'],
            visible_pixels=observed['visible_pixels_per_height'],unknown_depth_pixels=observed['unknown_depth_pixels']))
    if full1000:
        groups=collections.defaultdict(list)
        for case, row in zip(spec['cases'],rows):
            if case['condition']['desired_relation'] != case['geometric_contact']['relation']:
                raise ValueError('Collection1000 contains an infeasible intended state')
            if (case['source_site_id'] != SITE_ID
                    or case['condition']['source_site_id'] != SITE_ID
                    or case['condition']['split_group_id'] != SITE_ID
                    or case['condition']['counterfactual_parent_id'] != case['group_id']):
                raise ValueError('Collection1000 source/group binding differs')
            groups[case['group_id']].append(case)
            row.update(group_id=case['group_id'],source_site_id=case['source_site_id'])
        if len(groups) != 250:
            raise ValueError('Collection1000 requires 250 quartets')
        for cases in groups.values():
            if (len(cases) != 4
                    or {c['geometric_contact']['relation'] for c in cases} != set(relations.values())
                    or len({c['condition']['family'] for c in cases}) != 1
                    or any(c['camera'] != cases[0]['camera'] or c['wearer'] != cases[0]['wearer'] for c in cases)):
                raise ValueError('Collection1000 quartet coverage or matched views differ')
        family_counts=dict(collections.Counter(cases[0]['condition']['family'] for cases in groups.values()))
        if family_counts != dict(crossbar=63,cabinet=63,oblique_rod=62,hanging_sign=62):
            raise ValueError('Collection1000 family allocation differs')
        counts=dict(collections.Counter(r['actual_geometry'] for r in rows))
        if counts != {r:250 for r in relations.values()}:
            raise ValueError('Collection1000 class balance differs')
        return dict(status='PASS' if all(r['agrees'] for r in rows) else 'VISIBILITY_GAP',
            frames=1000,quartets=250,geometry_counts=counts,family_group_counts=family_counts,
            native_visible_agreement=sum(r['agrees'] for r in rows),
            native_visibility_gaps=sum(not r['agrees'] for r in rows),rows=rows,
            scope='1000 attached fixture frames in 250 matched quartets at one street site. '
                  'Native visibility gaps retained in the full denominator; no independent-world, model-training or safety claim.')
    core=[r for r in rows if r['subset']=='core']
    if len(core)!=96 or len(rows)-len(core)!=32:
        raise ValueError('Incorrect core/hard allocation')
    result=dict(status='PASS' if all(r['agrees'] for r in core) else 'CORE_VISIBILITY_GAP',
        frames=128,core_frames=96,hard_frames=32,
        core_visible_agreement=sum(r['agrees'] for r in core),
        hard_visible_agreement=sum(r['agrees'] for r in rows if r['subset']=='hard'),
        geometry_core_counts=dict(collections.Counter(r['actual_geometry'] for r in core)),
        hard_groups=dict(collections.Counter(r['hard_kind'] for r in rows if r['subset']=='hard')),
        rows=rows,
        scope='Attached cuboid assembly sweep versus all-scene visible native depth, single street Development. '
              'Hard gaps stay in denominator. Intent and boundary flags are not assigned truth. No model or safety score.')
    return result


def check(root):
    root=under_artifacts(root)
    spec_path=root/'source/spec.json'
    spec,native=read(spec_path),read(root/'world-verification.json')
    if native['status']!='PASS' or read(root/'receipt.json')['spec_sha256']!=sha(spec_path):
        raise ValueError('Unverified source/capture')
    result=evaluate(spec,native)
    result.update(spec_sha256=sha(spec_path),native_report_sha256=sha(root/'world-verification.json'))
    with (root/'contextual-check.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    return {k:v for k,v in result.items() if k!='rows'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True)
    print(json.dumps(check(p.parse_args().capture)))

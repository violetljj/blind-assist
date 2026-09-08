"""Retain native visibility gaps separately from full assembly geometry labels."""
import argparse
import collections
import json
import sys
from pathlib import Path
from make_city_obstacle_suite import REPO, under_artifacts, read, sha
sys.path.insert(0,str(REPO/'research/active/dtr-r0/nearfield'))
from contextual_geometry import target_contact


def check(root):
    root=under_artifacts(root)
    spec_path=root/'source/spec.json'
    spec,native=read(spec_path),read(root/'world-verification.json')
    if native['status']!='PASS' or read(root/'receipt.json')['spec_sha256']!=sha(spec_path):
        raise ValueError('Unverified source/capture')
    if spec['schema']!='city-contextual-headspace-128-v1' or len(spec['cases'])!=128 or len(native['rows'])!=128:
        raise ValueError('Incomplete contextual128 source')
    relations={(0,0):'CLEAR',(1,0):'BODY_ONLY',(0,1):'HEAD_ONLY',(1,1):'BOTH'}
    rows=[]
    for case,observed in zip(spec['cases'],native['rows']):
        if case['name']!=observed['name']:
            raise ValueError('Row identity differs')
        geometry=target_contact(case['objects'],case['wearer'])
        if geometry!=case['geometric_contact']:
            raise ValueError('Geometry reference differs')
        visible=relations.get(tuple(observed['body_head_visible_targets']),'UNKNOWN')
        rows.append(dict(name=case['name'],family=case['condition']['family'],
            subset=case['condition']['subset'],hard_kind=case['condition']['hard_kind'],
            intended=case['condition']['desired_relation'],actual_geometry=geometry['relation'],
            visible_relation=visible,agrees=visible==geometry['relation'],
            visible_pixels=observed['visible_pixels_per_height'],unknown_depth_pixels=observed['unknown_depth_pixels']))
    core=[r for r in rows if r['subset']=='core']
    if len(core)!=96 or len(rows)-len(core)!=32:
        raise ValueError('Incorrect core/hard allocation')
    result=dict(status='PASS' if all(r['agrees'] for r in core) else 'CORE_VISIBILITY_GAP',
        frames=128,core_frames=96,hard_frames=32,
        core_visible_agreement=sum(r['agrees'] for r in core),
        hard_visible_agreement=sum(r['agrees'] for r in rows if r['subset']=='hard'),
        geometry_core_counts=dict(collections.Counter(r['actual_geometry'] for r in core)),
        hard_groups=dict(collections.Counter(r['hard_kind'] for r in rows if r['subset']=='hard')),
        rows=rows,spec_sha256=sha(spec_path),native_report_sha256=sha(root/'world-verification.json'),
        scope='Attached cuboid assembly sweep versus all-scene visible native depth, single street Development. '
              'Hard gaps stay in denominator. Intent and boundary flags are not assigned truth. No model or safety score.')
    with (root/'contextual-check.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    return {k:v for k,v in result.items() if k!='rows'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True)
    print(json.dumps(check(p.parse_args().capture)))

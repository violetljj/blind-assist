"""Fixed-fixture radial views; exact native floor preflight precedes final export."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def build(config, source, receipt=None):
    route=config['route_id']
    originals={c['variant']:c for c in source['cases'] if c.get('route_id')==route and c.get('variant') in config['variants']}
    assert set(originals)==set(config['variants'])
    clear=originals['clear']
    feet=[o for o in clear['objects'] if o['name'].endswith(('_left_foot','_right_foot'))]
    assert len(feet)==2
    anchor=[sum(o['center_m'][j] for o in feet)/2 for j in range(2)]
    anchor.append(feet[0]['center_m'][2]-feet[0]['size_m'][2]/2)
    base_yaw=clear['camera']['yaw']
    measured={p['case']:p for p in receipt['native_floor_probes']} if receipt else {}
    spec=copy.deepcopy(source)
    spec.update(cases=[],native_targets=[t for t in source['native_targets'] if t.get('route_id')==route],
                export_native_inventory=False,export_appearance=False,export_controlled_targets=True,
                collection_scope='Consumed same-route fixed-geometry approach Development; no model fitting.')
    spec.pop('inventory_indices',None)
    for az in config['approach_azimuth_deg']:
        heading=base_yaw+az
        variants=config['variants'] if receipt else ['clear']
        for variant in variants:
            for distance in config['distance_m']:
                key=f'az{az:+g}_d{distance:g}'
                preflight_name='floor_'+key
                radians=math.radians(heading)
                x=anchor[0]-distance*math.cos(radians);y=anchor[1]-distance*math.sin(radians)
                floor=anchor[2]
                if receipt:
                    probe=measured[preflight_name]
                    assert probe['hit'],preflight_name
                    px,py,floor=probe['point_m']
                    assert math.hypot(px-x,py-y)<.01,preflight_name
                    assert abs(floor-anchor[2])<.5,'Review unexpected native floor before final export: '+preflight_name
                case=copy.deepcopy(originals[variant])
                case.update(name=(variant+'_'+key if receipt else preflight_name),
                    camera=dict(x=x,y=y,z=floor+config['camera_height_m'],yaw=heading,pitch=config['pitch_deg'],roll=0.),
                    floor_z_m=floor,distance_m=distance,approach_azimuth_deg=az,
                    sequence_id=variant+f'_az{az:+g}',route_distance_m=max(config['distance_m'])-distance,
                    pair_id=route+'_'+key,variant=variant,variant_id=variant,
                    floor_authority='EXACT_XY_NATIVE_PREFLIGHT_HIT' if receipt else 'NOMINAL_PENDING_NATIVE_PREFLIGHT',
                    fixture_anchor_m=anchor,source_case_name=originals[variant]['name'])
                assert case['objects']==originals[variant]['objects'],'Never move original fixture geometry'
                spec['cases'].append(case)
    return spec

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'experiments/city-field/approach-v1.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',type=Path,help='Completed preflight capture directory; omission produces21 clear poses')
    a=p.parse_args();out=a.output.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve())
    cfg=read(a.config);source_path=ROOT/cfg['source_spec'];source=read(source_path)
    assert sha(source_path)==cfg['source_spec_sha256']
    receipt=None
    if a.preflight:
        receipt=read(a.preflight/'receipt.json');assert receipt['status']=='PASS'
        assert read(a.preflight/'source-integrity.json')['unchanged']
        preflight_spec=read(a.preflight/'source/spec.json')
        assert preflight_spec==build(cfg,source),'Preflight poses differ from frozen recipe'
    spec=build(cfg,source,receipt)
    out.mkdir(parents=True,exist_ok=True)
    mode='final' if receipt else 'preflight'
    target=out/(mode+'-spec.json')
    with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(spec,f,indent=2);f.write('\n')
    manifest=dict(schema='city-field-approach-inputs-v1',mode=mode,frames=len(spec['cases']),
        source_spec=cfg['source_spec'],source_spec_sha256=sha(source_path),config_sha256=sha(a.config),
        spec_sha256=sha(target),fixture_geometry='EXACT_ORIGINAL_CASE_OBJECTS_AND_TARGET_IDS',
        preflight_receipt_sha256=sha(a.preflight/'receipt.json') if receipt else None,
        azimuth_convention='Camera heading original_yaw+azimuth; camera XY=anchor-distance*[cos(heading),sin(heading)], faces anchor.',
        scope='Consumed west TRAIN route Development; static settled poses, not continuous-motion evidence.')
    with (out/(mode+'-inputs.json')).open('x',encoding='utf-8',newline='\n') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(manifest))

if __name__=='__main__':main()

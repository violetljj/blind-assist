"""Build west-only street collection with condition-specific physical props.

The three sampled segments are disconnected. Native mesh copies are controlled
placements, not claims that those instances occur naturally at the fixture site.
"""
import argparse
import json
from pathlib import Path
from city_field_fixtures import fixture_objects, MATERIALS

ROOT = Path(__file__).resolve().parents[1]


def build():
    plan = json.loads((ROOT/'experiments/city-field/native-field-v1.json').read_bytes())
    plan.update(plan_id='city-collection-field-v2', require_all_splits=False,
                collection_scope='Three disconnected west Development segments; static samples, not a continuous traversed walk.')
    plan['regions'] = [r for r in plan['regions'] if r['region_id']=='west']
    plan['routes'] = [r for r in plan['routes'] if r['region_id']=='west']
    for route in plan['routes']:
        rid=route['route_id']
        if route['scene_type']=='sidewalk':
            for w in route['waypoints']:
                w['camera']['y']=-.2
                w['floor_authority']='PENDING_V2_EXACT_XY_PREFLIGHT'
            route['status']='PENDING'
        camera=route['waypoints'][3]['camera']; floor=route['waypoints'][3]['floor_z_m']
        anchor=[camera['x']+2,camera['y'],floor]
        route['fixture_anchor_m']=anchor
        def native(part, asset, zoffset=0):
            name=rid+'__'+part
            return dict(name=name,instance_id=name,mesh_asset=asset,scale=[1,1,1],
                center_m=[anchor[0],anchor[1],anchor[2]+zoffset],
                rotation_deg=dict(pitch=0,yaw=0,roll=0),support_parent='ground',target_part=True,
                geometry_provenance='CONTROLLED_PLACEMENT_OF_ORIGINAL_CITYSAMPLE_MESH')
        def cabinet_box(part,z,size,parent,target):
            name=rid+'__'+part
            return dict(name=name,instance_id=name,center_m=[anchor[0],anchor[1],anchor[2]+z],
                size_m=size,rotation_deg=dict(pitch=0,yaw=0,roll=0),
                material_asset=MATERIALS['metal'],support_parent=parent,target_part=target,
                geometry_provenance='CONTROLLED_GROUNDED_ROADSIDE_UTILITY_CABINET')
        variants={'clear':[],
            'thin_pole':[native('thin_pole','/Game/Megascans/3D_Assets/Metal_Bollard_04/ujcheg1ba_LOD0.ujcheg1ba_LOD0',.00164)],
            'body_protrusion':[
                cabinet_box('cabinet_plinth',.06,[.40,.55,.12],'ground',False),
                cabinet_box('body_arm',.645,[.35,.50,1.05],rid+'__cabinet_plinth',True)]}
        for kind in ('head_bar','suspended_sign'):
            variants[kind]=fixture_objects(kind,anchor,0,rid)
        route['fixture_variants']=variants
        route['condition_description']={
            'clear':'Original city and route boundary objects only; no overhead fixture.',
            'thin_pole':'Controlled copy of original metal bollard, grounded.',
            'body_protrusion':'Controlled roadside utility cabinet on a grounded low plinth: body-height occupied volume, not a cantilever. Exact cube collision geometry.',
            'head_bar':'Controlled maintenance stand with low crossbar; synthetic supported geometry.',
            'suspended_sign':'Controlled hanging panel supported by maintenance stand and two hangers.'}
        route['admission']={'status':'PENDING_V2_PREVIEW','reason_pending':'Exact shifted sidewalk floor and ordinary mesh fixtures need rendered review.'}
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--admission',type=Path,help='Reviewed source preview receipt; exact seven sidewalk floor probes required')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/city-field/native-field-v2.json')
    args=parser.parse_args();out=args.output;plan=build()
    if args.admission:
        receipt=json.loads(args.admission.read_bytes())
        if receipt['status']!='PASS' or not receipt.get('source_unchanged'): raise ValueError('Invalid source preview')
        probes={r['case']:r for r in receipt['native_floor_probes']}
        route=plan['routes'][0]
        for i,w in enumerate(route['waypoints']):
            row=probes[f'west_sidewalk_path_{i:03d}']
            if not row['hit']: raise ValueError('Missing floor')
            x,y,z=row['point_m'];camera=w['camera']
            if abs(camera['x']-x)>.001 or abs(camera['y']-y)>.001: raise ValueError('Floor coordinates disagree')
            w.update(floor_z_m=z,floor_authority='V2_EXACT_XY_NATIVE_FLOOR_PROBE')
            camera['z']=z+1.7
        # Regenerate fixtures with the measured middle-site floor.
        dz=route['waypoints'][3]['floor_z_m']-route['fixture_anchor_m'][2]
        route['fixture_anchor_m'][2]+=dz
        for objects in route['fixture_variants'].values():
            for obj in objects: obj['center_m'][2]+=dz
        for route in plan['routes']:
            route['status']='ADMITTED'
            route['admission']={'status':'ADMITTED','evidence':str(args.admission),
                'scope':'Shifted sidewalk exact floors and preview; intersection and controlled corridor reuse inspected v1 geometry. Static visual admission, not controller traversal.'}
        plan['admission']={'status':'ADMITTED','evidence':str(args.admission),
            'scope':'Seven shifted sidewalk camera samples on clear paving; prior native crosswalk and controlled plaza corridor retained. Three disconnected static segments, no controller traversal claim.'}
    out.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(out)

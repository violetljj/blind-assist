"""Deterministic fresh-region crossbars; no model or outcome access."""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

from body_query_distance_pairs_spec import preflight


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def target(distance, lateral, height, width, thickness, roll, scale=1.):
    return dict(name='head_bar', instance_id='head_bar', target_part=True,
                center_m=[distance, lateral*scale, 1.7+(height-1.7)*scale],
                size_m=[.06*scale, width*scale, thickness*scale],
                rotation_deg=dict(pitch=0., yaw=0., roll=roll))


def corners(obj):
    r = math.radians(obj['rotation_deg']['roll'])
    co, si = math.cos(r), math.sin(r)
    values = []
    for a,b,c in itertools.product((-0.5,0.5), repeat=3):
        x,y,z = (v*s for v,s in zip((a,b,c), obj['size_m']))
        values.append([obj['center_m'][0]+x,
                       obj['center_m'][1]+co*y+si*z,
                       obj['center_m'][2]-si*y+co*z])
    return values


def projection(obj):
    f = 320/math.tan(math.radians(50))
    return [[319.5+f*y/x, 179.5-f*(z-1.7)/x] for x,y,z in corners(obj)]


def assembly(bar):
    objects = [bar]
    roll = math.radians(bar['rotation_deg']['roll'])
    # Posts meet the rolled bar's endpoints, remain grounded and preserve
    # physically meaningful floor context in the matched-size arm.
    for sign in (-1,1):
        y = bar['center_m'][1]+sign*bar['size_m'][1]*.5*math.cos(roll)
        z = bar['center_m'][2]-sign*bar['size_m'][1]*.5*math.sin(roll)
        assert abs(y)-.025 > .28, 'Post enters BODY corridor'
        objects.append(dict(name=f'post_{sign}', center_m=[bar['center_m'][0],y,z/2],
                            size_m=[.05,.05,z], rotation_deg=dict(pitch=0.,yaw=0.,roll=0.)))
    return objects


def make_pair(site, region_id, arm, variant):
    near,far = ((1.05,2.25),(1.25,2.55))[variant]
    rank = site['rank']
    lateral = (-.10,0.,.10)[rank%3]
    height = (1.66,1.70,1.72)[rank%3]
    width = (1.20,1.35)[variant]
    thickness = (.05,.065)[variant]
    roll = (-3.,0.,3.)[rank%3]
    pid = f'{region_id}-s{rank:03}-{arm}-v{variant}'
    yaw = site['yaw_deg']; co,si = math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
    x,y = site['camera_xy_m']; floor = site['floor_z_m']
    result, local = [], []
    for endpoint,distance in (('near',near),('far',far)):
        scale = far/near if arm=='matched' and endpoint=='far' else 1.
        bar = target(distance,lateral,height,width,thickness,roll,scale)
        objects = assembly(bar); local.append(copy.deepcopy(objects))
        for obj in objects:
            ox,oy,oz = obj['center_m']
            obj['center_m'] = [x+co*ox-si*oy,y+si*ox+co*oy,floor+oz]
            obj['rotation_deg']['yaw'] = yaw
        case = dict(name=pid+'-'+endpoint, pair_id=pid, group_id=pid,
                    region_id=region_id, site_id=site['site_id'], arm=arm,
                    declared_range=endpoint, variant_id='HEAD_ONLY',
                    source_role='FRESH_CONTROLLED_DEVELOPMENT_NO_FITTING',
                    floor_z_m=floor, floor_check=False, probe_native_floor=True,
                    camera=dict(x=x,y=y,z=floor+1.7,yaw=yaw,pitch=0.,roll=0.),
                    wearer=dict(x=x,y=y,z=floor,yaw=yaw,pitch=0.,roll=0.),
                    condition=dict(family='crossbar',distance_m=distance,
                                   desired_relation='HEAD_ONLY',query_control='HEAD_ONLY'),
                    objects=objects)
        preflight(case)
        result.append(case)
    projections = [projection(objects[0]) for objects in local]
    error = max(abs(a-b) for p,q in zip(*projections) for a,b in zip(p,q))
    if arm=='matched':
        assert error < 1e-10, error
    else:
        assert local[0][0]['size_m']==local[1][0]['size_m']
    return result, dict(pair_id=pid, arm=arm, analytic_corner_error_px=error,
                        local_objects=local, projected_corners=projections)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sites', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    canonical = (Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    assert a.output.resolve().is_relative_to(canonical)
    data = json.loads(a.sites.read_bytes())
    assert data['status']=='SOURCE_REVIEW_SELECTED'
    checks, outputs = [], []
    for region in data['regions']:
        assert len(region['accepted_sites'])==30
        spec = json.loads(Path(region['template_spec_file']).read_bytes())
        for key in ('source_floor_grid','native_targets','route'):
            spec.pop(key,None)
        spec.update(cases=[],schema='body-query-fresh-size-v1',
                    scope='Fresh geographic controlled Development; shared map/assets; no fitting',
                    export_native_inventory=True,inventory_indices=[0],
                    export_hlod_membership=False,export_dependencies=False,
                    export_appearance=False,pair_export_mode='native_async',
                    export_controlled_targets=True,native_targets=[dict(target_id='head_bar')],
                    provenance=dict(sites_sha256=digest(a.sites),generator_sha256=digest(__file__)))
        for site in region['accepted_sites']:
            for arm in ('ordinary','matched'):
                for variant in (0,1):
                    cases,check = make_pair(site,region['region_id'],arm,variant)
                    spec['cases'].extend(cases); checks.append(check)
        path = a.output/(region['region_id']+'.json'); write(path,spec)
        outputs.append(dict(region_id=region['region_id'],spec_file=path.name,
                            spec_sha256=digest(path),frames=len(spec['cases'])))
    assert len(checks)==240
    write(a.output/'manifest.json',dict(status='FROZEN_SPEC_NATIVE_ADMISSION_PENDING',
          regions=outputs,pairs=240,frames=480,checks=checks))


if __name__=='__main__':
    main()

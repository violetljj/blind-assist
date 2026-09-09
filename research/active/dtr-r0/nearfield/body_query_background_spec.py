"""Move exact historical crossbar pairs by translation only; no predictions."""
import argparse
import copy
import json
import math
from pathlib import Path

from body_query_collection_labels import read, write, sha
from body_query_distance_pairs_spec import preflight


def relocate(raw, site, region, source_index):
    case = copy.deepcopy(raw)
    assert raw['camera']['yaw'] == site['yaw_deg']
    delta = [site['camera_xy_m'][0]-raw['camera']['x'],
             site['camera_xy_m'][1]-raw['camera']['y'], site['floor_z_m']-raw['floor_z_m']]
    for key in ('camera', 'wearer'):
        for axis, value in zip(('x', 'y', 'z'), delta):
            case[key][axis] += value
    for obj in case['objects']:
        obj['center_m'] = [a+b for a,b in zip(obj['center_m'], delta)]
    case['floor_z_m'] += delta[2]
    for a,b in zip(raw['objects'],case['objects'],strict=True):
        assert {k:v for k,v in a.items() if k!='center_m'} == {k:v for k,v in b.items() if k!='center_m'}
        for j,axis in enumerate(('x','y','z')):
            assert abs((a['center_m'][j]-raw['camera'][axis])-(b['center_m'][j]-case['camera'][axis]))<1e-9
    for axis in ('pitch','yaw','roll'):
        assert raw['camera'][axis]==case['camera'][axis]
    for obj in case['objects']:
        if obj.get('target_part'):
            assert obj['name']=='adjustable_cross_member'
            obj['instance_id']=obj['name']  # Native export identity; no physical change.
    assert abs(case['camera']['z']-case['floor_z_m']-1.7)<1e-9
    pid='bqbackground-'+region+'-'+site['site_id']
    case.update(name=pid+'-'+raw['declared_range'],pair_id=pid,group_id=pid,parent_group_id=pid,
                site_id=site['site_id'],region_id=region,arm='background',
                source_role='CONSUMED_DEVELOPMENT_BACKGROUND_DIAGNOSTIC',
                original_pair_id=raw['pair_id'],original_case_name=raw['name'],original_index=source_index,
                probe_native_floor=True,geometry_frame='Original camera wearer and complete fixture translated together')
    # Keep old geometry/provenance; identity changes do not drive rendering.
    case['condition']['condition_id']=case['name']
    case['condition']['counterfactual_parent_id']=pid
    case.pop('physical_endpoint_sha256',None)
    preflight(case)
    return case, delta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sites','historical','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    canonical=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    assert a.output.resolve().is_relative_to(canonical)
    sites=read(a.sites);rows=read(a.historical/'frame_metadata.json')
    candidates={yaw:[] for yaw in (0,90,180,-90)};cache={};lookup={r['name']:i for i,r in enumerate(rows)}
    for i,row in enumerate(rows):
        if (row['source_partition'],row['family'],row['endpoint'])!=('eval','crossbar','near'):continue
        path=Path(row['rgb_path']).parents[2]/'source/spec.json'
        if path not in cache:cache[path]=read(path)
        raw=next(c for c in cache[path]['cases'] if c['name']==row['name'])
        if raw['paired_variation_slot']!='base':continue
        far=next(c for c in cache[path]['cases'] if c['pair_id']==raw['pair_id'] and c['declared_range']=='far')
        candidates[raw['camera']['yaw']].append((raw,far,path))
    checks=[];outputs=[];used=set()
    for region in sites['regions']:
        assert len(region['accepted_sites'])==30
        # Original render settings, only geographic streaming rectangle changes.
        spec=copy.deepcopy(next(iter(cache.values())))
        template=read(region['template_spec_file'])
        spec.update(cases=[],schema='body-query-background-only-v1',
                    world_partition_region_m=template['world_partition_region_m'],
                    scope='Consumed background-only diagnostic; shared map assets; no fitting',
                    export_controlled_targets=True,native_targets=[dict(target_id='adjustable_cross_member')],
                    provenance=dict(sites_sha256=sha(a.sites),historical_metadata_sha256=sha(a.historical/'frame_metadata.json'),generator_sha256=sha(__file__)))
        for site in region['accepted_sites']:
            near,far,path=candidates[site['yaw_deg']].pop(0)
            assert near['pair_id'] not in used;used.add(near['pair_id'])
            translated=[];deltas=[]
            for raw in (near,far):
                case,delta=relocate(raw,site,region['region_id'],lookup[raw['name']]);translated.append(case);deltas.append(delta)
            assert deltas[0]==deltas[1]
            assert translated[0]['camera']==translated[1]['camera']
            spec['cases'].extend(translated)
            checks.append(dict(pair_id=translated[0]['pair_id'],original_pair_id=near['pair_id'],
                               original_indices=[lookup[c['name']] for c in (near,far)],
                               source_spec=str(path.resolve()),source_spec_sha256=sha(path),
                               delta_m=deltas[0],objects=len(near['objects']),translation_invariants='PASS'))
        path=a.output/(region['region_id']+'.json');write(path,spec)
        outputs.append(dict(region_id=region['region_id'],spec_file=path.name,spec_sha256=sha(path),frames=len(spec['cases'])))
    assert len(checks)==60
    write(a.output/'manifest.json',dict(status='FROZEN_SPEC_NATIVE_ADMISSION_PENDING',pairs=60,frames=120,regions=outputs,
          selection='All60 previous sites; first unused historical EVAL crossbar BASE pair per matching absolute yaw in metadata order; no prediction access',checks=checks))


if __name__=='__main__':main()

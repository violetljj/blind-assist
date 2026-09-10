"""New XY proposals from two retained source-only floor scouts; no model outcomes."""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path

from mz5_ensemble_readout import read, write, sha


def prepare(root, output):
    root = root.resolve()
    if output.exists() or not output.resolve().is_relative_to((root/'artifacts.local').resolve()):
        raise ValueError('A fresh artifacts.local output is required')
    old = root/'artifacts.local/work/body-query-fresh-size-20260909'
    sources = [old/'empty-probe-specs-v1/site-candidates.json',
               old/'empty-probe-specs-v2/site-candidates.json', old/'accepted-sites-v1.json']
    historical = [read(p) for p in sources]
    current = historical[1]
    helper = root/'tools/body_query_5000_sites.py'
    loader = importlib.util.spec_from_file_location('retained_floor_candidates', helper)
    module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    bindings = {str(p):sha(p) for p in sources+[helper]}
    regions = []; specs = {}
    for region in current['regions']:
        name = region['region_id']
        assert name in ('dense_candidate_05', 'dense_candidate_06')
        paths = region['original_sources']
        for entry in paths.values():
            assert sha(entry['path']) == entry['sha256']
            bindings[entry['path']] = entry['sha256']
        spec, grid, inventory = [read(paths[k]['path']) for k in ('template_spec','floor_grid','primitive_inventory')]
        assert read(paths['source_receipt']['path'])['status'] == 'PASS'
        assert spec['scope'] == 'SOURCE_ONLY_RECONNAISSANCE_NOT_RESEARCH_COHORT'
        meshes = {c['component_path']:c.get('static_mesh_asset','')
                  for actor in inventory.get('loaded_actor_paths',[]) for c in actor.get('primitive_components',[])}
        rows = []
        for row in grid['rows']:
            mesh = meshes.get(row.get('component_path',''),'').lower()
            joined = row.get('component_path','').lower()+' '+mesh
            if mesh and 'ground_collision' not in joined and any(t in mesh for t in ('sidewalk','road_','_road')):
                rows.append(row)
        pool = module.candidates(rows,meshes)
        exclusions = set()
        for document in historical:
            for previous in document['regions']:
                if previous['region_id'] != name: continue
                for key in ('primary_sites','reserve_sites','accepted_sites'):
                    exclusions.update(tuple(s['camera_xy_m']) for s in previous.get(key,[]))
        assert exclusions
        available = [s for s in pool if min(math.dist(s['camera_xy_m'],p) for p in exclusions) >= 6.-1e-7]
        sites = copy.deepcopy(module.select(available,15,separation=6.))
        for rank, site in enumerate(sites,1):
            site.update(site_id=f'mz36_{name}_site_{rank:03}', rank=rank,
                        prior_xy_distance_m=min(math.dist(site['camera_xy_m'],p) for p in exclusions))
        row = dict(region_id=name, prior_xy_count=len(exclusions), eligible_before=len(pool),
                   eligible_after=len(available), candidate_sites=sites, original_sources=paths,
                   prior_xy=sorted(exclusions), status='PROPOSED' if len(sites)==15 else 'NOT_EVALUABLE')
        regions.append(row)
        if len(sites) != 15: continue
        probe = copy.deepcopy(spec)
        for key in ('source_floor_grid','native_targets','export_controlled_targets'): probe.pop(key,None)
        probe.update(cases=[],export_hlod_membership=False,export_native_inventory=True,inventory_indices=[0],
                     pair_export_mode='native_async',settling_ticks=32,first_use_settling_ticks=64,
                     scope='SOURCE_ONLY_RECONNAISSANCE_NOT_RESEARCH_COHORT')
        for site in sites:
            x,y = site['camera_xy_m']
            probe['cases'].append(dict(name=site['site_id']+'_empty',site_id=site['site_id'],region_id=name,
                source_role='MZ36_NEW_XY_SOURCE_ONLY',objects=[],probe_native_floor=True,
                source_floor_z_m=site['floor_z_m'],camera=dict(x=x,y=y,z=site['floor_z_m']+1.7,
                    yaw=site['yaw_deg'],pitch=0,roll=0)))
        specs[name] = probe
    output.mkdir(parents=True)
    manifest = dict(status='PROPOSED' if all(r['status']=='PROPOSED' for r in regions) else 'NOT_EVALUABLE',
        scope='New XY on two historically scouted CitySample regions; shared assets/background, no region-independent claim',
        regions=regions,inputs=bindings,generator_sha256=sha(__file__),model_predictions=0,
        selection='15 ranked sites per region, >=6m from all previous proposed XY and each other; first10 passing empty source review',
        full_capture_budget=dict(sites=20,frames_per_site=20,frames=400))
    write(output/'site-candidates.json',manifest)
    entries = []
    if manifest['status']=='PROPOSED':
        for name,spec in specs.items():
            path=output/(name+'.json');write(path,spec)
            entries.append(dict(region_id=name,spec_file=path.name,spec_sha256=sha(path),frames=15))
    write(output/'manifest.json',dict(status=manifest['status'],regions=entries,frames=sum(e['frames'] for e in entries),
                                    candidate_manifest_sha256=sha(output/'site-candidates.json')))
    print(json.dumps(dict(status=manifest['status'],regions=[{k:r[k] for k in ('region_id','prior_xy_count','eligible_before','eligible_after','status')} for r in regions])))
    return manifest['status']=='PROPOSED'


def full(root, review_path, output):
    from body_query_5000_spec import fixture_catalog, site_cases
    root=root.resolve()
    if output.exists() or not output.resolve().is_relative_to((root/'artifacts.local').resolve()):
        raise ValueError('Fresh artifacts.local output required')
    task=root/'artifacts.local/work/mz36-new-source-20260910'
    candidates=read(task/'empty-specs-v1/site-candidates.json');review=read(review_path)
    assert review['status']=='PASS' and len(review['regions'])==2
    fixture=root/'artifacts.local/work/body-query-v1-20260908/source-v3/spec-all-v1.json'
    catalog=fixture_catalog(read(fixture));entries=[];built=[]
    for region_index, reviewed in enumerate(review['regions']):
        name=reviewed['region_id'];selected=reviewed['selected_site_ids']
        assert len(selected)==10 and len(set(selected))==10
        source=next(r for r in candidates['regions'] if r['region_id']==name)
        accepted=[r['site_id'] for r in reviewed['rows'] if r['accepted']][:10]
        assert selected==accepted
        spec=read(task/'empty-specs-v1'/f'{name}.json')
        spec.update(cases=[],source_role='DEV_ONLY',export_controlled_targets=True,
            schema='mz36-new-xy-controlled-v1',scope='New XY controlled Development; previously consumed regions and shared CitySample assets; frozen models, no fitting')
        for rank,site_id in enumerate(selected):
            site=next(s for s in source['candidate_sites'] if s['site_id']==site_id)
            for case in site_cases(catalog,dict(region_id=name,split='dev'),site,1000+region_index*10+rank):
                for key in ('name','group_id','parent_group_id'):
                    case[key]=case[key].replace('bq5000-','mz36-')
                case['condition']['condition_id']=case['name']
                case['condition']['counterfactual_parent_id']=case['group_id']
                spec['cases'].append(case)
        assert len(spec['cases'])==200
        spec['provenance']=dict(generator_sha256=sha(__file__),retained_site_cases_sha256=sha(Path(__file__).with_name('body_query_5000_spec.py')),
            fixture_file=str(fixture),fixture_sha256=sha(fixture),fixture_role='TRAIN_ONLY',review_sha256=sha(review_path),rank_offset=1000)
        built.append((name,spec))
    output.mkdir(parents=True)
    for name,spec in built:
        path=output/f'{name}.json';write(path,spec)
        entries.append(dict(region_id=name,spec_file=path.name,spec_sha256=sha(path),frames=len(spec['cases'])))
    write(output/'manifest.json',dict(status='FROZEN_SPEC_NOT_CAPTURED',regions=entries,frames=400,review_sha256=sha(review_path),
        frozen_models_sha256=sha(task/'frozen-models.json'),protocol_sha256=sha(Path(__file__).with_name('MZ36_NEW_SOURCE_PROTOCOL_20260910.md'))))
    print(json.dumps(dict(status='FROZEN_SPEC_NOT_CAPTURED',frames=400,sites=20)))
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review',type=Path)
    a=p.parse_args();raise SystemExit(0 if (full(a.root,a.review,a.output) if a.review else prepare(a.root,a.output)) else 2)

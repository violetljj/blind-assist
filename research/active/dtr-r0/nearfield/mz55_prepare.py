"""Freeze exactly2560 MZ55 cases from measured existing mesh bounds; no UE."""
import argparse
import copy
import itertools
from pathlib import Path
import numpy as np
from mz48_prepare import read, sha, write, rotation, world


def prepare(root, output):
    work = root/'artifacts.local/work'
    design_root = work/'mz52-diverse-source-design-20260911'
    inputs = {}
    def bound(path, expected=None):
        digest = sha(path); assert expected is None or digest == expected, path
        inputs[str(path)] = digest
        return read(path)
    design = bound(design_root/'design.json')
    evidence = bound(design_root/'mesh-evidence.json')
    for path, digest in evidence['inputs'].items():
        assert sha(Path(path)) == digest
        inputs[path] = digest
    for name in ('PROPOSAL.md',):
        inputs[str(design_root/name)] = sha(design_root/name)
    for name in ('mz55_prepare.py', 'mz48_prepare.py', 'MZ55_DIVERSE_MESH_SOURCE_20260911.md'):
        path = Path(__file__).with_name(name); inputs[str(path)] = sha(path)
    original_path = work/'mz42-rich-objects-20260910/spec-v2/spec.json'
    original = bound(original_path, 'cc76970fe3f358c4cffca24d423cb1d81a73720d609a4e7dc21b8f3067cf7600')
    prior_root = work/'mz48-rich-kilotier-20260911'
    prior = bound(prior_root/'spec-v1/manifest.json')
    runtime = bound(prior_root/'primary-runtime-plugin.json')
    runtime_config = bound(work/'mz36-new-source-20260910/runtime-config.json')
    region_specs = {}
    for row in prior['shards'][:2]:
        spec = bound(Path(row['path']), row['sha256'])
        region_specs[spec['cases'][0]['region_id']] = spec
    sites = sorted(prior['sites'], key=lambda s:s['site_id'])
    assert [s['site_id'].removeprefix('mz36_dense_candidate_') for s in sites] == design['sites']
    families = design['families']; relations = design['relations']
    templates = {c['variant_id']:c for c in original['cases'] if c['condition']['family']=='oblique_rod'}
    assets = {}
    for family in families[:-1]:
        item = evidence['assets'][family]
        path = Path(item['asset_file']); assert sha(path) == item['asset_sha256']
        inputs[str(path)] = item['asset_sha256']
        assets[family] = {k:item[k] for k in ('asset','asset_sha256','asset_bytes','bounds_min_m','bounds_max_m')}
    rod_material = templates['HEAD_ONLY']['objects'][0]['material_asset']
    material_path = root/'artifacts.local/unreal/CitySample/Content'/(rod_material.removeprefix('/Game/')+'.uasset')
    inputs[str(material_path)] = sha(material_path)
    canary = {(i,j,(i+j)%4,(i+j)%2,(i+2*j)%5) for i in range(8) for j in range(4)}
    extra_audit = {(i,0,i%4,i%2,(i+1)%5) for i in range(8)}
    assert canary.isdisjoint(extra_audit)
    cases = []
    for si, site in enumerate(sites):
        region_case = next(c for c in region_specs[site['region_id']]['cases'] if c['site_id']==site['site_id'])
        for fi, family in enumerate(families):
            for ri, relation in enumerate(relations):
                for di, distance in enumerate(('near','far')):
                    for pi in range(5):
                        front = design['front_anchors_m'][distance][pi]
                        lateral = design['lateral_offsets_m'][pi]
                        targets, bounds = [], []
                        if family != 'retained_rod':
                            asset = assets[family]; scale = design['uniform_scales'][family][pi]
                            rot = dict(pitch=90. if family=='square_grille' else 0.,
                                yaw=(0. if family=='square_grille' else 90.)+design['new_mesh_yaw_offsets_degrees'][pi], roll=0.)
                            corners = np.array(list(itertools.product(*zip(asset['bounds_min_m'],asset['bounds_max_m']))))*scale
                            rotated = corners@rotation(rot).T; lo,hi = rotated.min(0),rotated.max(0)
                            heights = design['new_mesh_center_heights_m'][relation]
                            heights = heights if isinstance(heights,list) else [heights]
                            for ti,height in enumerate(heights):
                                origin = np.array([front+ti*.05,lateral,height])-np.array([lo[0],(lo[1]+hi[1])/2,(lo[2]+hi[2])/2])
                                low,high = lo+origin,hi+origin
                                mesh = asset['asset']+'.'+asset['asset'].split('/')[-1]
                                target = dict(name=family+'_'+str(ti),mesh_asset=mesh,scale=scale,placement='actor_origin',
                                    target_part=True,center_m=world(site['camera'],site['floor_z_m'],origin),
                                    rotation_deg=dict(rot,yaw=rot['yaw']+site['camera']['yaw']))
                                targets.append(target);bounds.append(dict(name=target['name'],camera_aligned_min_m=low.tolist(),camera_aligned_max_m=high.tolist()))
                                if relation=='VISIBLE_NONINTRUDING': assert low[2]>1.85
                                elif height<1.4: assert low[2]>=.65 and high[2]<=1.4
                                else: assert low[2]>=1.4 and high[2]<=1.85
                        else:
                            old = templates['CLEAR' if relation=='VISIBLE_NONINTRUDING' else relation]
                            for obj in (o for o in old['objects'] if o['target_part']):
                                target=copy.deepcopy(obj)
                                origin=np.array([front+.10,obj['center_m'][1]-old['camera']['y']+lateral,
                                    obj['center_m'][2]-old['floor_z_m']])
                                half=np.array(obj['size_m'])/2
                                points=np.array(list(itertools.product(*zip(-half,half))))@rotation(obj['rotation_deg']).T+origin
                                target['center_m']=world(site['camera'],site['floor_z_m'],origin)
                                target['rotation_deg']['yaw']+=site['camera']['yaw']
                                targets.append(target);bounds.append(dict(name=target['name'],camera_aligned_min_m=points.min(0).tolist(),camera_aligned_max_m=points.max(0).tolist()))
                        supports=[]
                        support_template=templates['HEAD_ONLY']
                        for obj in (o for o in support_template['objects'] if not o['target_part']):
                            support=copy.deepcopy(obj);sign=-1 if obj['name'].endswith('_-1') else 1
                            x=front+.10+(1.50 if obj['name'].startswith('rear_') else .75 if obj['name'].startswith('side_rail') else 0.)
                            z=1.50 if obj['name'].startswith('clamp') else obj['center_m'][2]-support_template['floor_z_m']
                            y=sign*.80;assert abs(y)-obj['size_m'][1]/2>.28
                            support['center_m']=world(site['camera'],site['floor_z_m'],[x,y,z]);support['rotation_deg']['yaw']=site['camera']['yaw']
                            supports.append(support)
                        assert len(supports)==12
                        key=(si,fi,ri,di,pi);pair=f"mz55-{site['site_id']}-{family}-{relation}-{distance}-p{pi}"
                        expected={'HEAD_ONLY':[0,1],'BODY_ONLY':[1,0],'BOTH':[1,1],'VISIBLE_NONINTRUDING':[0,0]}[relation]
                        events=[0]*4;events[di],events[2+di]=expected
                        role=design['roles'][design['sites'][si]]
                        for supported in (False,True):
                            context='supported' if supported else 'unsupported'
                            case={k:copy.deepcopy(region_case[k]) for k in ('camera','wearer','floor_z_m','site_id','region_id')}
                            case.update(name=pair+'-'+context,group_id=pair,pair_id=pair,variant_id=relation,declared_range=distance,
                                source_role='DEV_ONLY',training_role=role,floor_check=False,probe_native_floor=True,profile_id=pi,
                                support_context=context,expected_events=events,expected_collapsed_relation=expected,
                                placement_bounds=copy.deepcopy(bounds),canary=key in canary,native_audit_sample=key in canary or key in extra_audit,
                                objects=copy.deepcopy(targets)+(copy.deepcopy(supports) if supported else []),
                                condition=dict(family=family,desired_relation=relation,condition_id=pair+'-'+context))
                            cases.append(case)
    assert len(cases)==len({c['name'] for c in cases})==2560
    assert sum(c['canary'] for c in cases)==64 and sum(c['native_audit_sample'] for c in cases)==80
    for a,b in zip(cases[::2],cases[1::2]):
        assert a['pair_id']==b['pair_id'] and a['training_role']==b['training_role']
        assert a['objects']==[o for o in b['objects'] if o['target_part']]
    output.mkdir(parents=True,exist_ok=False)
    shards=[]
    for region in region_specs:
        shards.append(('canary-'+region,region,[c for c in cases if c['region_id']==region and c['canary']],True))
    for site in sites:
        shards.append(('main-'+site['site_id'],site['region_id'],[c for c in cases if c['site_id']==site['site_id'] and not c['canary']],False))
    rows=[]
    for name,region,subset,is_canary in shards:
        assert len(subset)==(32 if is_canary else 312)
        spec={k:copy.deepcopy(v) for k,v in region_specs[region].items() if k not in ('cases','provenance')}
        spec.update(schema='mz55-diverse-mesh-v1',cases=subset,
            scope='Controlled diverse rigid meshes; consumed sites; native truth independent of collapsed relation intent',
            purpose='MZ55_SOURCE_ONLY_DIVERSE_MESH',provenance=dict(inputs=inputs,fixed_total_budget=2560,stage='canary' if is_canary else 'main'))
        path=output/'shards'/(name+'.json');write(path,spec)
        rows.append(dict(shard_id=name,frames=len(subset),canary=is_canary,path=str(path),sha256=sha(path)))
    manifest=dict(status='FROZEN_BEFORE_CAPTURE',experiment='mz55-diverse-mesh-source-20260911',inputs=inputs,
        total_frames=2560,new_mesh_frames=1920,rod_frames=640,canary_frames=64,main_frames=2496,native_audit_frames=80,
        families=families,relations=relations,sites=sites,role_counts=design['role_counts'],assets=assets,
        rod_material=dict(asset=rod_material,sha256=inputs[str(material_path)]),runtime_package=runtime,
        runtime_config_sha256=sha(work/'mz36-new-source-20260910/runtime-config.json'),shards=rows,
        canary_gate=dict(source_valid=64,intent_min=56,pair_invariant=32,all_new_family_relation_present=True,visual_all64_required=True),
        new_model_inference_frames=0,training_steps=0,captures_executed=0,
        host_policy='Worker canary only until root admission; primary only after GPU free and exclusive unstarted shard allocation')
    write(output/'manifest.json',manifest)
    write(output/'receipt.json',dict(status='PASS_PREPARED_NOT_CAPTURED',inputs=inputs,frames=2560,canary_frames=64,
        outputs={str(p.relative_to(output)):sha(p) for p in output.rglob('*.json')},new_captures=0))
    print('PASS_PREPARED_NOT_CAPTURED',sha(output/'manifest.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.root.resolve(),a.output.resolve())

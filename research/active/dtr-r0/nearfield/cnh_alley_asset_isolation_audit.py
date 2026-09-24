"""Read-only pre-capture audit of the authored alley asset pools.

Checks actual saved clutter actors against the site plan, and compares the
planned insertion pool across roles and proposed splits. Package dependency
overlap is reported conservatively; it is not a rendered-material proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_package(entry):
    return entry['asset_path'].split('.')[0]


def audit(plan, catalog, build_run, development_capture_root=None,
          verify_local_bytes=False, repo_root=None):
    build_run=Path(build_run)
    compact=read(build_run/'compact-family-audit.json')
    family_path=build_run/'family-receipt.json'
    family=read(family_path)
    generated=read(build_run/'generated-plan.json')
    loaded=read(build_run/'loaded-assets.json')
    if compact['family_receipt_sha256']!=sha(family_path):
        raise ValueError('Family receipt hash differs from compact audit')
    if compact['status']!='PASS_NATIVE_AUTHORING_FAMILY_ISOLATION_AND_MEASURED_CLEARANCE':
        raise ValueError('Native authoring audit is not PASS')
    sites=plan['sites']
    generated_sites={s['site_id']:s for s in generated['sites']}
    family_sites={s['site_id']:s for s in family['sites']}
    map_rows={s['site_id']:s for s in compact['maps']}
    if len(sites)!=9 or len(generated_sites)!=9 or len(family_sites)!=9 or len(map_rows)!=9:
        raise ValueError('Expected the same nine authored sites in all receipts')
    if set(generated_sites)!=set(family_sites)!=set(map_rows)!={s['site_id'] for s in sites}:
        raise ValueError('Site IDs differ across plan and receipts')

    owners={}
    source_roots=defaultdict(set)
    clutter_counts=Counter()
    physical_sites=defaultdict(set)
    selected=defaultdict(set)
    verified_maps=0
    for site in sites:
        site_id=site['site_id']
        split=site['proposed_split']
        if split not in ('train','dev','test'):
            raise ValueError('Unknown proposed split: '+split)
        actual=generated_sites[site_id]
        recorded=family_sites[site_id]
        if any(actual[k]!=site[k] for k in ('proposed_split','clutter_assets','insert_assets','physical_site_id')):
            raise ValueError('Generated plan differs from site assignment: '+site_id)
        if any(recorded[k]!=site[k] for k in ('proposed_split','clutter_assets','insert_assets','physical_site_id')):
            raise ValueError('Family receipt differs from site assignment: '+site_id)
        if not map_rows[site_id]['passed'] or map_rows[site_id]['map_asset']!=actual['map_asset']:
            raise ValueError('Map authoring acceptance differs: '+site_id)
        if verify_local_bytes:
            map_file=(Path(repo_root)/catalog['source_project']).parent/'Content'/(
                actual['map_asset'].removeprefix('/Game/')+'.umap')
            if sha(map_file)!=map_rows[site_id]['map_sha256']:
                raise ValueError('Saved map bytes differ from audited hash: '+site_id)
            verified_maps+=1
        receipt=read(build_run/(site_id+'-geometry-receipt.json'))
        if receipt['status']!='SAVED_AUTHORING_GEOMETRY_REQUIRES_VISUAL_REVIEW':
            raise ValueError('Map was not saved: '+site_id)
        if receipt['proposed_split']!=split or receipt['map_asset']!=actual['map_asset']:
            raise ValueError('Geometry receipt assignment differs: '+site_id)
        clutter=[a for a in receipt['actors'] if a['role']=='clutter']
        if len(clutter)!=site['clutter_count']:
            raise ValueError('Actual clutter count differs: '+site_id)
        if {a['asset'] for a in clutter}!=set(site['clutter_assets']):
            raise ValueError('Actual clutter pool differs: '+site_id)
        clutter_counts[split]+=len(clutter)
        physical_sites[split].add(site['physical_site_id'])
        for role, keys, table in (
            ('distractor',{a['asset'] for a in clutter},catalog['meshes']),
            ('insert',set(site['insert_assets']),catalog['insert_assets']),
        ):
            for key in sorted(keys):
                entry=table[key]
                if role=='distractor' and entry['role']!='DISTRACTOR':
                    raise ValueError('Actual clutter lacks DISTRACTOR role: '+key)
                source=source_package(entry)
                load_key=key if role=='distractor' else 'insert__'+key
                if loaded['assets'][load_key]['source']!=entry['asset_path']:
                    raise ValueError('Native loaded source differs: '+key)
                for kind,value in (('semantic',entry['family_id']),('source_kit',entry['source_family_root']),('mesh',source)):
                    identity=(kind,value)
                    prior=owners.setdefault(identity,split)
                    if prior!=split:
                        raise ValueError('Cross-split '+kind+' overlap: '+value)
                selected[split].add((role,key))
                source_roots[split].add(source)
    if compact['clutter_count']!=sum(clutter_counts.values()):
        raise ValueError('Compact clutter count differs from actor receipts')
    if any(physical_sites[a]&physical_sites[b] for a,b in (('train','dev'),('train','test'),('dev','test'))):
        raise ValueError('Physical sites cross proposed splits')

    verified_source_files=0
    if verify_local_bytes:
        for role,key in set().union(*selected.values()):
            entry=(catalog['meshes'] if role=='distractor' else catalog['insert_assets'])[key]
            provenance=entry.get('provenance',{})
            path=entry.get('local_uasset') or provenance['local_uasset']
            expected=entry.get('uasset_sha256') or provenance['sha256']
            if sha(Path(repo_root)/path)!=expected:
                raise ValueError('Source mesh bytes differ from audited hash: '+role+'/'+key)
            verified_source_files+=1

    graph={}
    overbroad={}
    for site in sites:
        closure=read(build_run/(site['site_id']+'-dependency-closure.json'))
        if closure['status']!='COMPLETE_HARD_AND_SOFT_PACKAGE_CLOSURE':
            raise ValueError('Incomplete package dependency closure: '+site['site_id'])
        split=site['proposed_split']
        extra=sorted(set(closure['roots'])&set().union(*(v for k,v in source_roots.items() if k!=split)))
        if extra:overbroad[site['site_id']]=extra
        for row in closure['packages']:
            package=row['package']
            deps=set(row['dependencies'])
            if package in graph and graph[package]!=deps:
                raise ValueError('Conflicting dependency graph for '+package)
            graph[package]=deps

    def reachable(root):
        pending=[root]
        found=set()
        while pending:
            package=pending.pop()
            if package in found:continue
            found.add(package)
            pending.extend(graph.get(package,()))
        return found

    package_sets={split:set().union(*(reachable(root) for root in roots))
                  for split,roots in source_roots.items()}
    overlap={}
    for a,b in (('train','dev'),('train','test'),('dev','test')):
        shared=sorted(package_sets[a]&package_sets[b])
        overlap[a+'__'+b]=dict(package_count=len(shared),
            non_engine_packages=[p for p in shared if not p.startswith(('/Engine/','/Script/'))])
    development_capture={}
    if development_capture_root is not None:
        capture_root=Path(development_capture_root)
        for site in sites:
            site_id=site['site_id']
            spec_path=capture_root/site_id/'capture/source/spec.json'
            if site['proposed_split']=='test':
                if spec_path.exists():raise ValueError('Test capture already present: '+site_id)
                continue
            spec=read(spec_path)
            expected={i:catalog['insert_assets'][key]['asset_path']
                      for i,key in zip((1,254),site['insert_assets'])}
            actual={row['id']:row['mesh_asset'] for row in spec['assets']}
            if actual!=expected or len(spec['assets'])!=2:
                raise ValueError('Captured insertion pair differs from split pool: '+site_id)
            if spec['data_role']!='Development' or spec['benchmark_eligible']:
                raise ValueError('Captured split authority differs: '+site_id)
            layouts=spec['layouts']
            if len(layouts)!=1 or layouts[0]['physical_site_id']!=site['physical_site_id']:
                raise ValueError('Capture physical site differs: '+site_id)
            clips=layouts[0]['clips']
            if [(clip['id'],len(clip['poses'])) for clip in clips]!=[
                ('centre',40),('boundary',40),('outside',40),('removed',40)]:
                raise ValueError('Capture sampling differs: '+site_id)
            development_capture[site_id]=dict(split=site['proposed_split'],
                target=site['insert_assets'][0],distractor=site['insert_assets'][1],
                clips=4,frames=sum(len(clip['poses']) for clip in clips),
                spec_sha256=sha(spec_path))
    return dict(schema='cnh-alley-asset-isolation-audit-v1',
        status='DECLARED_AND_ACTUAL_FAMILIES_DISJOINT_VISUAL_DEPENDENCY_NOT_PROVEN',
        formal_test_capture_admission=False,site_count=len(sites),
        actual_clutter_count=dict(sorted(clutter_counts.items())),
        used_assets={split:{role:sorted(key for r,key in keys if r==role)
                            for role in ('distractor','insert')}
                     for split,keys in sorted(selected.items())},
        semantic_source_kit_mesh_cross_split_overlap=0,
        physical_site_cross_split_overlap=0,
        verified_local_map_files=verified_maps,
        verified_local_source_mesh_files=verified_source_files,
        audited_development_capture=development_capture,
        overbroad_historical_closure_roots=overbroad,
        conservative_package_overlap=overlap,
        limitation='Package dependencies include generic shaders and material branches; effective rendered visible dependencies and final capture insertions require a separate bound audit.',
        evidence=dict(compact_audit_sha256=sha(build_run/'compact-family-audit.json'),
                      family_receipt_sha256=sha(family_path)))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan',type=Path,default=Path(__file__).with_name('cnh_street_alley_demo.json'))
    parser.add_argument('--catalog',type=Path,default=Path(__file__).with_name('cnh_street_alley_assets.json'))
    parser.add_argument('--build-run',type=Path,required=True)
    parser.add_argument('--development-capture-root',type=Path)
    parser.add_argument('--verify-local-bytes',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=audit(read(args.plan),read(args.catalog),args.build_run,args.development_capture_root,
                 args.verify_local_bytes,Path(__file__).resolve().parents[4])
    payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload,encoding='utf-8')
    print(payload)


if __name__=='__main__':main()

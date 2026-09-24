"""Join native LOD0 material inspection to the R3a split plan, without test capture."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(plan,catalog,loaded,native,terminal,native_path,development_capture_root=None):
    if native['status']!='NATIVE_SOURCE_LOD0_MATERIAL_INSPECTION_COMPLETE':
        raise ValueError('Native material inspection incomplete')
    if terminal['status']!='PASS_READ_ONLY_NATIVE_MATERIAL_INSPECTION' or not (
        terminal['protected_file_hashes_unchanged'] and terminal['process_released'] and terminal['temp_released']):
        raise ValueError('Native read-only/process integrity failed')
    if terminal['native_receipt_sha256']!=sha(native_path):
        raise ValueError('Native material receipt hash differs')
    declared={}
    for site in plan['sites']:
        split=site['proposed_split']
        for role,keys,table in [('distractor',site['clutter_assets'],catalog['meshes']),
                                ('insert',site['insert_assets'],catalog['insert_assets'])]:
            for key in keys:
                ident=(split,role,key)
                declared[ident]=table[key]['asset_path']
    if len(declared)!=15:raise ValueError('Expected fifteen used role/source assignments')
    observed={}
    owned=defaultdict(lambda:defaultdict(set))
    by_asset={}
    for row in native['assets']:
        ident=(row['split'],row['role'],row['key'])
        if ident in observed or ident not in declared or row['mesh']!=declared[ident]:
            raise ValueError('Native source identity differs from declared split pool: '+str(ident))
        observed[ident]=row
        load_key=row['key'] if row['role']=='distractor' else 'insert__'+row['key']
        source_slots=loaded['assets'][load_key]['materials']
        if row['lod_index']!=0 or not row['sections']:
            raise ValueError('LOD0 section mapping missing: '+str(ident))
        slots={m['slot']:m for m in row['materials']}
        if {s['material_slot'] for s in row['sections']}!=set(slots):
            raise ValueError('Inspected material slots differ from LOD0 sections: '+str(ident))
        details=[]
        for slot,material in sorted(slots.items()):
            if material['interface']!=source_slots[slot]['material']:
                raise ValueError('Loaded native material differs from authoring receipt: '+str(ident))
            owner=dict(role=row['role'],key=row['key'],slot=slot)
            for texture in material['editor_used_textures']:
                owned[row['split'],'editor_used_texture'][texture].add((row['role'],row['key'],slot))
            for parameter in material['effective_texture_parameters']:
                if parameter['texture']:
                    owned[row['split'],'effective_texture_parameter'][parameter['texture']].add((row['role'],row['key'],slot))
            owned[row['split'],'material_interface'][material['interface']].add((row['role'],row['key'],slot))
            for parent in material['parent_chain'][1:]:
                owned[row['split'],'material_parent'][parent['path']].add((row['role'],row['key'],slot))
            details.append(dict(owner,interface=material['interface'],
                editor_used_texture_count=len(material['editor_used_textures']),
                effective_texture_parameters=material['effective_texture_parameters'],
                effective_static_switches=material['effective_static_switches'],
                editor_used_textures=material['editor_used_textures'],
                parent_chain=material['parent_chain']))
        by_asset['/'.join(ident)]=dict(lod0_section_count=len(row['sections']),
            source_nanite_enabled=row['source_nanite_enabled'],materials=details)
    if set(observed)!=set(declared):raise ValueError('Native inspection omitted a used source asset')

    def entries(a,b,kind):
        left=owned[a,kind];right=owned[b,kind]
        return [dict(path=p,
            left_owners=[dict(role=r,key=k,slot=s) for r,k,s in sorted(left[p])],
            right_owners=[dict(role=r,key=k,slot=s) for r,k,s in sorted(right[p])])
            for p in sorted(set(left)&set(right))]

    pairs={a+'__'+b:{kind:entries(a,b,kind) for kind in (
        'editor_used_texture','effective_texture_parameter','material_interface','material_parent')}
        for a,b in [('train','dev'),('train','test'),('dev','test')]}
    for groups in pairs.values():
        used={row['path'] for row in groups['editor_used_texture']}
        params={row['path'] for row in groups['effective_texture_parameter']}
        groups['used_and_effective_parameter']=sorted(used&params)
        groups['used_without_named_parameter']=sorted(used-params)
        groups['effective_parameter_not_editor_used']=sorted(params-used)
    overlap_count={pair:{kind:len(values) for kind,values in groups.items()}
                   for pair,groups in pairs.items()}
    shared_used=any(groups['editor_used_texture'] for groups in pairs.values())
    development_derived=[]
    if development_capture_root is not None:
        capture_root=Path(development_capture_root)
        for site in plan['sites']:
            if site['proposed_split']=='test':continue
            source=read(capture_root/site['site_id']/'capture/raw-manifest.json')
            assets=source['derived_assets']
            if set(assets)!={'1','254'}:
                raise ValueError('Development insertion identities differ: '+site['site_id'])
            for identifier,key in zip(('1','254'),site['insert_assets']):
                entry=assets[identifier]
                if (entry['source_mesh']!=catalog['insert_assets'][key]['asset_path'] or
                    entry['saved'] or not entry['source_unchanged'] or
                    entry['configuration']['textures_and_appearance']!='DUPLICATED_GRAPH_RETAINED' or
                    any(m.get('overrides_preserved') is False for m in entry['materials'])):
                    raise ValueError('Development derived material receipt differs: '+site['site_id']+'/'+key)
                development_derived.append(dict(site_id=site['site_id'],key=key,
                    source_mesh=entry['source_mesh'],source_unchanged=True,
                    material_overrides_preserved=True,derived_assets_unsaved=True))
    return dict(schema='cnh-alley-visible-dependency-intersection-v1',
        status=('FAIL_SHARED_EDITOR_USED_TEXTURE_DEPENDENCIES' if shared_used
                else 'NO_SHARED_EDITOR_USED_TEXTURES_SOURCE_SCOPE_ONLY'),
        formal_test_capture_admission=False,
        native_api_scope='MaterialEditingLibrary.GetMaterialUsedTextures at current material quality; compiled texture expression values, shader platform not explicitly selected; no pixel contribution test',
        actual_lod='LOD0 material sections of fifteen source meshes; authoring clutter and controlled insert capture both force LOD0',
        split_pairs=pairs,overlap_counts=overlap_count,assets=by_asset,
        development_derived_insertions=development_derived,
        evidence=dict(native_receipt_sha256=sha(native_path),
                      protected_file_count=terminal['protected_file_count'],
                      protected_file_hashes_unchanged=True,
                      map_or_frame_capture='NOT_RUN'))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=Path(__file__).with_name('cnh_street_alley_demo.json'))
    parser.add_argument('--catalog',type=Path,default=Path(__file__).with_name('cnh_street_alley_assets.json'))
    parser.add_argument('--loaded',type=Path,required=True)
    parser.add_argument('--development-capture-root',type=Path)
    args=parser.parse_args()
    native_path=args.native_run/'native-materials.json'
    result=summarize(read(args.plan),read(args.catalog),read(args.loaded),read(native_path),
                     read(args.native_run/'terminal.json'),native_path,args.development_capture_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],overlap_counts=result['overlap_counts'],
                          output=str(args.output))))


if __name__=='__main__':main()

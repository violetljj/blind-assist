"""One-pose loaded-component compiled-material capability probe; no admission.

Snapshots an exact collector hook without modifying the legacy collector or map.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]


def probe_materials(u, api, output):
    """Query actual component material interfaces, never asset-name guesses."""
    started=time.monotonic()
    capability=getattr(getattr(u,'BlindAssistCaptureLibrary',None),'get_material_geometry_capability',None)
    materials={}; components=[]; errors=[]
    for actor in sorted(api.get_all_level_actors(),key=lambda a:a.get_path_name()):
        for component in sorted(actor.get_components_by_class(u.PrimitiveComponent),key=lambda c:c.get_path_name()):
            path=component.get_path_name()
            identity=dict(actor=actor.get_path_name(),component=path,component_class=component.get_class().get_name())
            try:
                slots=component.get_num_materials()
                uses=[]
                for slot in range(slots):
                    material=component.get_material(slot)
                    key=material.get_path_name() if material is not None else path+':null:'+str(slot)
                    uses.append(dict(slot=slot,material=key))
                    if key in materials:
                        materials[key]['component_slot_uses']+=1
                        continue
                    row=dict(material=key,component_slot_uses=1)
                    if material is None:
                        row['native_report']=dict(data_status='UNKNOWN',capability='NULL_COMPONENT_MATERIAL')
                    elif capability is None:
                        row['native_report']=dict(data_status='UNKNOWN',capability='NATIVE_API_UNAVAILABLE')
                    else:
                        try:
                            raw=capability(material)
                            decoded=json.loads(raw)
                            if not isinstance(decoded,dict):
                                raise ValueError('Native capability must return a JSON object')
                            row['native_report']=decoded
                            row['native_json']=raw
                        except Exception as exc:
                            row['native_report']=dict(data_status='UNKNOWN',capability='NATIVE_QUERY_FAILED',error=str(exc))
                    materials[key]=row
                components.append(dict(identity,material_slots=slots,materials=uses))
            except Exception as exc:
                errors.append(dict(identity,error=str(exc)))
    reports=[r['native_report'] for r in materials.values()]
    available=sum(r.get('data_status')=='AVAILABLE' for r in reports)
    result=dict(schema='cnh-compiled-material-probe-v1',status='CAPABILITY_PROBE_COMPLETE_NO_ADMISSION',
        benchmark_eligible=False,source_admission='NOT_RUN',authority='ACTUAL_LOADED_COMPONENT_MATERIAL_NATIVE_COMPILED_QUERY',
        api_available=capability is not None,unique_material_count=len(materials),component_count=len(components),
        compiled_available_count=available,unresolved_material_count=len(reports)-available,
        counts_by_capability=dict(Counter(r.get('capability','UNKNOWN') for r in reports)),
        counts_by_data_status=dict(Counter(r.get('data_status','UNKNOWN') for r in reports)),
        counts_by_status=dict(Counter(r.get('status','NOT_SUPPLIED') for r in reports)),
        materials=list(materials.values()),components=components,component_errors=errors,
        limitations='Loaded primitives only. Compiled material flags do not certify world completeness, mesh precision, motion bounds, or WPO aggregate extent.',
        wall_s=time.monotonic()-started)
    with Path(output).open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result


def run(args):
    journal=os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    if not journal or json.loads(Path(journal).read_text(encoding='utf-8-sig')).get('state')!='running':
        raise RuntimeError('Use governed research-ue execution')
    if not math.isfinite(args.timeout) or not 0<args.timeout<=600:
        raise ValueError('Probe timeout must be positive and no greater than 600 seconds')
    root=(REPO/'artifacts.local').resolve()
    for name in ('output','result'):
        path=getattr(args,name).resolve()
        if not path.is_relative_to(root) or path==root or path.exists():
            raise ValueError('Fresh '+name+' strictly under artifacts.local required')
        setattr(args,name,path)
    spec=json.loads(args.spec.read_text(encoding='utf-8-sig'))
    if spec.get('map_asset') not in ('/Game/BAResearchSlice/Street200V7','/Game/Map/Small_City_LVL'):
        raise ValueError('Only the two already selected source maps are allowed')
    if spec.get('scope')!='SOURCE_ONLY_RECONNAISSANCE_NOT_RESEARCH_COHORT' or len(spec.get('cases',[]))!=1:
        raise ValueError('Exactly one source-only pose required; no formal layout admission')
    if spec.get('export_native_inventory') is not True or spec.get('inventory_indices') != [0]:
        raise ValueError('Existing native inventory hook required')
    inputs=args.output.with_name(args.output.name+'-inputs')
    inputs.mkdir(parents=True,exist_ok=False)
    original_path=HERE/'city_pcg_capture.py'
    original=original_path.read_text(encoding='utf-8')
    hook="            native_inventory = inventory(u, api, spec['cases'][index]['camera'], 35.)"
    if original.count(hook)!=1:
        raise RuntimeError('Collector hook changed; review before executing')
    patched=original.replace(hook,hook+"\n            from cnh_route_scene_probe import probe as cnh_scene_probe\n"
        "            cnh_scene_probe(u, api, spec['cases'][index]['camera'], 8., OUT/f'evaluator/cnh-scene-{index:04d}')\n"
        "            from cnh_route_material_probe_check import probe_materials\n"
        "            probe_materials(u, api, OUT/'evaluator/material-capabilities.json')")
    args.capture_source=inputs/'city_pcg_capture.py'
    args.capture_source.write_text(patched,encoding='utf-8')
    sys.path.insert(0,str(REPO/'tools'))
    import run_city_pcg_capture as collector
    original_run=collector.run_owned

    def with_probe(command,env,output,timeout):
        launch_path=output/'launch.json'
        launch=json.loads(launch_path.read_text())
        for name in ('cnh_route_scene_probe.py','cnh_route_material_probe_check.py'):
            target=output/'source'/name
            shutil.copy2(HERE/name,target)
            launch['source_hashes'][name]=hashlib.sha256(target.read_bytes()).hexdigest()
        command=[v+',ProceduralMeshComponent' if v.startswith('-EnablePlugins=') else v for v in command]
        launch.update(command=command,scope='ONE_EXISTING_POSE_COMPILED_MATERIAL_CAPABILITY_ONLY',
            collector_original_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),journal=journal)
        launch_path.write_text(json.dumps(launch,indent=2),encoding='utf-8')
        return original_run(command,env,output,timeout)

    collector.run_owned=with_probe
    try:
        collector.capture(args)
        material_path=args.output/'evaluator/material-capabilities.json'
        material=json.loads(material_path.read_text())
        summary={key:value for key,value in material.items() if key not in ('materials','components')}
        summary.update(output=str(args.output),map_asset=spec['map_asset'],
            material_receipt_sha256=hashlib.sha256(material_path.read_bytes()).hexdigest(),
            source_probe=str(args.output/'evaluator/cnh-scene-0000/scene-probe.json'))
        with args.result.open('x',encoding='utf-8') as stream:
            json.dump(summary,stream,indent=2,allow_nan=False);stream.write('\n')
        return summary
    finally:
        collector.run_owned=original_run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','spec','plugin','engine','output','result'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ddc-path',type=Path)
    parser.add_argument('--timeout',type=float,default=600)
    run(parser.parse_args())

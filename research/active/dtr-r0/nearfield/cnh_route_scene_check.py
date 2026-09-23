"""Run a source-only mesh capability probe through the existing native collector.

This exports RGB-D plus an inventory, NOT seven CNH passes or an admitted layout.
The shared collector and saved maps are unchanged; an exact hook is snapshotted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def run(args):
    if not os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL'):
        raise RuntimeError('Use tools/ba.ps1 run research-ue -RunSpec')
    root = (REPO/'artifacts.local').resolve()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(root) or args.output == root or args.output.exists():
        raise ValueError('Fresh output strictly under artifacts.local required')
    args.result = args.result.resolve()
    if not args.result.is_relative_to(root) or args.result.exists():
        raise ValueError('Fresh result strictly under artifacts.local required')
    spec = json.loads(args.spec.read_text(encoding='utf-8-sig'))
    if spec.get('map_asset') not in ('/Game/BAResearchSlice/Street200V7', '/Game/Map/Small_City_LVL'):
        raise ValueError('This probe is bounded to the two selected scene sources')
    if spec.get('scope') != 'SOURCE_ONLY_RECONNAISSANCE_NOT_RESEARCH_COHORT' or len(spec['cases']) != 1:
        raise ValueError('Exactly one preflight pose per source; not the four-layout comparison')
    if spec.get('export_native_inventory') is not True or spec.get('inventory_indices') != [0]:
        raise ValueError('Inventory hook required')
    inputs = args.output.with_name(args.output.name + '-inputs')
    inputs.mkdir(parents=True, exist_ok=False)
    original = (HERE/'city_pcg_capture.py').read_text(encoding='utf-8')
    hook = "            native_inventory = inventory(u, api, spec['cases'][index]['camera'], 35.)"
    if original.count(hook) != 1:
        raise RuntimeError('Collector changed; review hook before running')
    patched = original.replace(hook, hook + "\n            from cnh_route_scene_probe import probe as cnh_scene_probe\n"
        "            cnh_scene_probe(u, api, spec['cases'][index]['camera'], 8., OUT/f'evaluator/cnh-scene-{index:04d}')")
    args.capture_source = inputs/'city_pcg_capture.py'
    args.capture_source.write_text(patched, encoding='utf-8')
    sys.path.insert(0, str(REPO/'tools'))
    import run_city_pcg_capture as collector
    original_run = collector.run_owned

    def with_probe(command, env, output, timeout):
        target = output/'source/cnh_route_scene_probe.py'
        shutil.copy2(HERE/'cnh_route_scene_probe.py', target)
        launch_path = output/'launch.json'
        launch = json.loads(launch_path.read_text())
        launch['source_hashes'][target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
        command = [v+',ProceduralMeshComponent' if v.startswith('-EnablePlugins=') else v for v in command]
        launch.update(command=command, scope='PRECOMPARISON_SOURCE_CAPABILITY_ONLY',
                      collector_original_sha256=hashlib.sha256((HERE/'city_pcg_capture.py').read_bytes()).hexdigest())
        launch_path.write_text(json.dumps(launch, indent=2), encoding='utf-8')
        return original_run(command, env, output, timeout)

    collector.run_owned = with_probe
    try:
        collector.capture(args)
        result = json.loads((args.output/'evaluator/cnh-scene-0000/scene-probe.json').read_text())
        with args.result.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
    finally:
        collector.run_owned = original_run


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ('project', 'spec', 'plugin', 'engine', 'output', 'result'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--ddc-path', type=Path)
    p.add_argument('--timeout', type=float, default=600)
    run(p.parse_args())

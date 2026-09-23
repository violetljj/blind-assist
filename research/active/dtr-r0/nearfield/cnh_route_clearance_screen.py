"""Bounded prefilter diagnostic on consumed scene-probe snapshots, not admission.

The old exported mesh AABBs do not bound WPO or unloaded/native unsupported
components. Report possible candidates separately; never upgrade this metadata.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

from cnh_route_capture import basis
from cnh_route_native_clearance import evaluate


def screen(capture):
    capture = Path(capture)
    source_path = capture/'source/spec.json'
    probe_path = capture/'evaluator/cnh-scene-0000/scene-probe.json'
    source = json.loads(source_path.read_text(encoding='utf-8-sig'))
    probe = json.loads(probe_path.read_text())
    camera = source['cases'][0]['camera']
    forward, right, _ = basis(camera)
    # Fixed 16 positions, selected without reading their clearance outcomes.
    offsets = list(itertools.product((-1.5, -.5, .5, 1.5), (-.45, -.15, .15, .45)))
    random.Random(20260924).shuffle(offsets)
    # Legacy probe lacks editor-only flags. UE SceneCaptureComponent::OnRegister
    # creates this exact hidden visualization mesh; retain each exclusion.
    editor_proxies = [row for row in probe['instances']
        if row['actor_path'].rsplit('.', 1)[-1].startswith('SceneCapture2D_')
        and row['mesh']['asset_path'] == '/Engine/EditorMeshes/MatineeCam_SM.MatineeCam_SM']
    proxy_ids = {row['id'] for row in editor_proxies}
    entities = [dict(id=f"{row['component_path']}:{row['instance_index']}",
        bounds_min_m=row['bounds_min_m'], bounds_max_m=row['bounds_max_m'],
        bounds_conservative=False, deformation_bounded=False) for row in probe['instances'] if row['id'] not in proxy_ids]
    results = []
    for ordinal, (along, lateral) in enumerate(offsets):
        clips = []
        for name, side in (('centre', 0.), ('boundary', .12), ('outside', .30), ('removed', 0.)):
            poses = []
            for travel in (-.5, .5):
                pose = dict(camera)
                for axis, key in enumerate(('x', 'y', 'z')):
                    pose[key] += (along+travel)*forward[axis]+(lateral+side)*right[axis]
                poses.append(pose)
            clips.append(dict(id=name, trajectory_model='piecewise_linear_fixed_orientation', poses=poses))
        manifest = dict(native_coverage_complete=False, native_entities=entities, clips=clips)
        result = evaluate(manifest)
        for box in result['swept_query_aabbs']:
            for point in itertools.product(*zip(box['min_m'], box['max_m'])):
                if sum((point[i]-probe['camera_m'][i])**2 for i in range(3)) > probe['radius_m']**2:
                    raise ValueError('Diagnostic query extends outside the original probe selection sphere')
        results.append(dict(candidate=ordinal, offsets_m=[along,lateral], clips=clips, result=result))
    possible = sum(row['result']['broadphase_clear'] for row in results)
    return dict(schema='cnh-consumed-source-clearance-screen-v1', status='COMPLETED_DIAGNOSTIC',
        scope='FIXED_16_POSE_CANDIDATES_WITHIN_ONE_CONSUMED_SITE_NOT_INDEPENDENT_LAYOUTS',
        sampling_seed=20260924, candidates=len(results), possible_clear_candidates=possible,
        bounds_conflict_candidates=len(results)-possible, admitted_layouts=0,
        native_coverage_complete=False,
        limitation='Exported LOD bounds do not bound WPO, unsupported primitives or unloaded geometry; possible clear is not PASS',
        unsupported_primitives=len(probe['unsupported_primitives']),
        excluded_editor_proxies=[dict(actor_path=row['actor_path'], component_path=row['component_path'],
            mesh=row['mesh']['asset_path'], reason='UE_SCENECAPTURE_ONREGISTER_HIDDEN_VISUALIZATION_PROXY') for row in editor_proxies],
        source_spec_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest(), results=results)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--street', type=Path, required=True)
    p.add_argument('--city', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = dict(street=screen(args.street), city=screen(args.city),
        backend='CPU_BOUNDED_AABB_PREFILTER_NO_GPU_WORKLOAD',
        source_hashes={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('cnh_route_clearance_screen.py', 'cnh_route_native_clearance.py', 'cnh_route_capture.py')})
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')

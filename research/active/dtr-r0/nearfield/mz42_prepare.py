"""Freeze20 real-mesh/control frames from actual UE metadata and one admitted site."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(root, metadata_path, output):
    prior = root / 'artifacts.local/work/mz36-new-source-20260910'
    metadata = read(metadata_path)
    assert metadata['status'] == 'PASS' and set(metadata['assets']) == {'pipe', 'ladder', 'pouch', 'birch'}
    review_path = prior / 'empty-review-v1.json'
    review = read(review_path)
    region = next(row for row in review['regions'] if row['region_id'] == 'dense_candidate_05')
    site = region['selected_site_ids'][0]
    assert site == 'mz36_dense_candidate_05_site_001'
    source_path = prior / 'full-specs-v1/dense_candidate_05.json'
    source = read(source_path)
    template = next(case for case in source['cases'] if case['site_id'] == site)
    assert template['camera']['yaw'] == 0
    spec = {key: copy.deepcopy(value) for key, value in source.items() if key not in ('cases', 'provenance')}
    spec.update(schema='mz42-rich-mesh-engineering-v1', cases=[], source_role='DEV_ONLY',
        scope='Controlled real-mesh diversity engineering; shared admitted site, rigid scaled objects; no model or sensor fidelity claim')
    transforms = {}
    variants = ('HEAD_ONLY', 'BODY_ONLY', 'BOTH', 'CLEAR')
    for family in ('pipe', 'ladder', 'pouch', 'birch'):
        info = metadata['assets'][family]
        extent = [(high-low)/100 for low, high in zip(info['bounds_min_cm'], info['bounds_max_cm'])]
        longest = max(range(3), key=lambda i: extent[i])
        # For upright forms, face the original X/Z ladder/tree plane toward camera X.
        rotation = dict(pitch=90. if longest == 2 else 0., yaw=90. if longest in (0, 2) else 0., roll=0.)
        rotated = [extent[1], extent[0], extent[2]] if longest == 0 else [extent[1], extent[2], extent[0]] if longest == 2 else extent
        scale = min(.35/rotated[0], .95/rotated[1], .30/rotated[2])
        transforms[family] = dict(source_extents_m=extent, rotation_deg=rotation, scale=scale,
                                  rotated_extents_m=[value*scale for value in rotated])
        for variant in variants:
            name = f'mz42-{site}-{family}-{variant}'
            case = {key: copy.deepcopy(template[key]) for key in ('camera', 'wearer', 'floor_z_m', 'floor_check', 'probe_native_floor')}
            case.update(name=name, group_id=f'mz42-{site}-{family}', site_id=site, region_id='dense_candidate_05',
                variant_id=variant, source_role='DEV_ONLY', declared_range='near', objects=[],
                condition=dict(family=family, desired_relation=variant, condition_id=name,
                               placement_authority='BOUNDS_FOR_PLACEMENT_ONLY; LABELS_FROM_NATIVE_SURFACES'))
            # Keep two actors in every variant so object count cannot identify BOTH.
            positions = [(0., 1.48), (.08, 1.48)] if variant == 'HEAD_ONLY' else [(0., .90), (.08, .90)] if variant == 'BODY_ONLY' else [(0., .90), (0., 1.48)]
            if variant == 'CLEAR':
                positions = [(1., .90), (1., 1.48)]
            for number, (lateral, bottom) in enumerate(positions):
                case['objects'].append(dict(name=f'{family}_{number}', mesh_asset=info['asset'], scale=scale,
                    rotation_deg=rotation, placement='bounds_front_center_floor', target_part=True,
                    center_m=[case['camera']['x']+1.2+number*.10, case['camera']['y']+lateral, case['floor_z_m']+bottom]))
            spec['cases'].append(case)
    for variant in variants:
        original = next(case for case in source['cases'] if case['site_id']==site and
                        case['condition']['family']=='oblique_rod' and case['variant_id']==variant)
        case = copy.deepcopy(original)
        case['original_control_case'] = original['name']
        case['name'] = 'mz42-control-' + original['name']
        case['group_id'] = 'mz42-control-' + original['group_id']
        spec['cases'].append(case)
    assert len(spec['cases']) == 20
    bindings = {str(path): sha(path) for path in (metadata_path, review_path, source_path, Path(__file__))}
    spec['provenance'] = dict(inputs=bindings, frames=20, new_mesh_frames=16, retained_control_frames=4,
        source_site=site, render_policy='Native material slots; no RGB-only edits; same calibrated camera and native depth',
        transformed_whole_mesh='Uniform scale and rigid rotation; scaled birch is a woody-form proxy, not natural leaf coverage')
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'spec.json'
    path.write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    manifest = dict(status='FROZEN_BEFORE_CAPTURE', frames=20, spec_sha256=sha(path), inputs=bindings,
                    site=site, transforms=transforms, labels='PENDING_NATIVE_RENDER', model_inference_frames=0, training_steps=0)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'metadata', 'output'):
        parser.add_argument('--' + argument, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.metadata.resolve(), args.output.resolve())

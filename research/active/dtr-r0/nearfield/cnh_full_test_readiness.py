"""Offline readiness inventory for the frozen 164-layout CNH test scope.

Reads authoring configuration and source-plan metadata only. It never reads
captured frames, test labels, predictions, or model outcomes, and cannot launch
capture. The output is a quota schedule, not a benchmark or test capture spec.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


ENVIRONMENTS = ('sidewalk', 'intersection', 'plaza', 'alley')
FAMILIES_PER_ENVIRONMENT = {'head': 16, 'rod': 16, 'body': 3, 'wall': 3, 'low': 3}
SEED = 20260924


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slots():
    rng = random.Random(SEED)
    rows = []
    for environment in ENVIRONMENTS:
        families = [family for family, count in FAMILIES_PER_ENVIRONMENT.items() for _ in range(count)]
        rng.shuffle(families)
        for ordinal, family in enumerate(families):
            rows.append(dict(slot_id=f'test-{environment}-{ordinal:02d}',
                             split='test', environment=environment, family=family,
                             clips=['centre', 'boundary', 'outside', 'removed'],
                             poses_per_clip=40, nominal_samples=160,
                             physical_site_id=None, map_asset=None, target_source_family=None,
                             distractor_source_families=None, layout_geometry=None,
                             capture_spec=None, status='QUOTA_ONLY_NO_TEST_PAYLOAD'))
    assert len(rows) == 164
    assert len({row['slot_id'] for row in rows}) == 164
    assert Counter(row['environment'] for row in rows) == {name: 41 for name in ENVIRONMENTS}
    assert Counter(row['family'] for row in rows) == {'head':64,'rod':64,'body':12,'wall':12,'low':12}
    return rows


def inventory(repo):
    repo = Path(repo).resolve()
    code = repo/'research/active/dtr-r0/nearfield'
    plan_doc = code/'CNH_ROUTE_COMPARISON_PLAN_20260924.md'
    catalog = code/'cnh_street_alley_assets.json'
    pilot_code = code/'cnh_route_pilot.py'
    primitive_code = code/'cnh_route_spec.py'
    authoring = repo/'artifacts.local/work/cnh-route-comparison-20260924/plan/street-alley-generator-v3r2.json'
    old_source = repo/'artifacts.local/work/cnh-route-comparison-20260924/plan/source-manifest.json'
    project = repo/'artifacts.local/unreal/CitySample'
    config = json.loads(authoring.read_text(encoding='utf-8-sig'))
    assets = json.loads(catalog.read_text(encoding='utf-8-sig'))
    old = json.loads(old_source.read_text(encoding='utf-8-sig'))
    if 'raise NotImplementedError' not in pilot_code.read_text(encoding='utf-8'):
        raise RuntimeError('Realistic benchmark pilot implementation changed; reassess readiness')
    from cnh_route_spec import ENVIRONMENTS as CURRENT_GENERATOR_ENVIRONMENTS
    if tuple(CURRENT_GENERATOR_ENVIRONMENTS) != ENVIRONMENTS:
        raise RuntimeError('Current primitive generator environment declaration differs from frozen outdoor categories')
    sites = config['sites']
    if len(sites) != 9 or Counter(s['proposed_split'] for s in sites) != {'train':3,'dev':3,'test':3}:
        raise ValueError('Expected the nine R3a authoring sites, three per prospective split')
    candidates = []
    for site in sites:
        if site['proposed_split'] != 'test':
            continue
        path = project/'Content/BAResearchAlley'/(site['site_id']+'.umap')
        if not path.is_file():
            raise FileNotFoundError(path)
        candidates.append(dict(site_id=site['site_id'], physical_site_id=site['physical_site_id'],
                               topology=site['topology'], map_asset=config['namespace']+'/'+site['site_id'],
                               map_sha256=sha(path), insert_asset_keys=site['insert_assets'],
                               distractor_asset_keys=site['clutter_assets'],
                               authority='PROSPECTIVE_AUTHORING_MAP_NOT_FORMAL_LAYOUT'))
    if len({row['physical_site_id'] for row in candidates}) != 3:
        raise ValueError('Three distinct prospective test physical sites required')
    insert_keys = sorted({key for row in candidates for key in row['insert_asset_keys']})
    distractor_keys = sorted({key for row in candidates for key in row['distractor_asset_keys']})
    for key in insert_keys:
        if assets['insert_assets'][key]['proposed_split'] != 'test':
            raise ValueError('Test authoring insert key crosses declared split: '+key)
    old_categories = sorted({key.split('/',1)[1] for key in old['split_environment_counts']
                             if key.startswith('test/')})
    schedule = slots()
    return dict(schema='cnh-full-test-readiness-v1', status='BLOCKED_REALISTIC_164_NOT_AUTHORED',
                authority='OFFLINE_QUOTA_SCHEDULE_AND_INPUT_INVENTORY_ONLY',
                no_capture=True, test_frames_read=0, test_labels_read=0, predictions_read=0,
                frozen=dict(seed=SEED, environments=list(ENVIRONMENTS),
                            families_per_environment=FAMILIES_PER_ENVIRONMENT,
                            layouts_per_environment=41, test_layouts=164, clips_per_layout=4,
                            nominal_poses_per_clip=40, nominal_time_samples=26240,
                            train_layouts=180, dev_layouts=40, total_layouts=384),
                schedule=schedule,
                source_inventory=dict(
                    r3a_test_authoring_sites=candidates,
                    test_insert_source_families={key: assets['insert_assets'][key]['source_family_root']
                                                 for key in insert_keys},
                    test_distractor_source_families={key: assets['meshes'][key]['source_family_root']
                                                     for key in distractor_keys},
                    authentic_test_capture_specs=0, admitted_formal_test_layouts=0,
                    authored_test_map_candidates_by_environment=dict(sidewalk=0,intersection=0,
                                                                     plaza=0,alley=3),
                    stale_primitive_manifest_test_environments=old_categories,
                    stale_primitive_manifest_is_usable_for_realistic_capture=False,
                    current_primitive_generator_environments=list(CURRENT_GENERATOR_ENVIRONMENTS),
                    current_primitive_generator_is_benchmark_source=False,
                    street200_physical_site_count=1,
                    expansion_v3_map_exists=(project/'Content/BAResearchExpansion/BrickServiceA_V3.umap').is_file(),
                    expansion_v3_authority='ONE_AUTHORING_PREVIEW_NOT_TEST_ADMISSION'),
                gates=[
                    'Author and screen 41 actual layouts in each of four outdoor environments, preserving 164 quota slots and unique physical-site partition identity.',
                    'Assign real source obstacle families for all five classes and same-class distractors, with original/derived family and rendered used-texture isolation across train/dev/test.',
                    'Freeze actual four-clip trajectories, source meshes, transforms, background geometry, NIR and appearance sampling, seed, hashes and layout labels before rendering.',
                    'Verify native 15cm clearance, 30/75mm and 2pct geometry checks, visibility, complete seven-pass format, and exact labels; retain failed layouts and denominator.',
                    'Use the governed realistic benchmark capture path and sealed evaluator-only labels; keep model/threshold selection blind to test labels and outcomes.',
                    'Use measured 20-train-layout seven-pass throughput/storage with 1.3x engineering margin against the fixed 250GiB and 24 GPU-hour limits.',
                    'Once asset, geometry, capacity and protected-output gates pass, collect the frozen test once into sealed storage; open labels and outcomes only after model and threshold seals. No outcome-driven slot replacement or quota change.'
                ],
                planning_estimate=dict(nominal_test_time_samples=26240,
                                       nominal_seven_pass_outputs=183680,
                                       preliminary_net_capture_hours_at_1_to_2_samples_per_s=[3.64,7.29],
                                       preliminary_raw_gb_at_2_to_3_mb_per_sample=[52.48,78.72],
                                       estimate_authority='FROZEN_PROTOCOL_ROUGH_RANGE_NOT_MEASURED_SEVEN_PASS_CAPACITY'),
                provenance={str(path.relative_to(repo)):sha(path)
                            for path in (plan_doc,catalog,pilot_code,primitive_code,authoring,old_source)},
                blockers=dict(missing_formal_layouts=164,
                              missing_formal_layouts_by_environment={name:41 for name in ENVIRONMENTS},
                              test_obstacle_class_coverage='UNPROVEN_FROM_TWO_DECLARED_INSERT_KITS',
                              rendered_asset_isolation='FINAL_CAPTURE_MATERIAL_AUDIT_PENDING',
                              full_source_scene_and_geometry='NOT_FROZEN',
                              protected_benchmark_capture_and_label_path='NOT_IMPLEMENTED'))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = inventory(args.repo)
    if args.output.exists():
        raise FileExistsError('Refuse to overwrite a frozen readiness receipt: '+str(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], slots=len(result['schedule']),
                          output=str(args.output), test_frames_read=0)))


if __name__ == '__main__':
    main()

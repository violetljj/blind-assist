"""Audit declared source identities under the shared-background CNH contract.

Metadata lint only: this does not establish visibility, geometry or capture admission.
Identity lists must come from source/dependency exports, never just renamed bundles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

POLICY = 'OBSTACLE_FAMILY_DISJOINT_SHARED_BACKGROUND_V1'
SPLITS = {'train', 'dev', 'test'}
ROLES = {'target', 'distractor', 'background'}


def identities(row, key):
    values = row[key]
    if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f'{key} requires nonempty exported identities')
    if len(values) != len(set(values)):
        raise ValueError(f'Duplicate {key}')
    return set(values)


def audit(source):
    if source.get('isolation_policy') != POLICY:
        raise ValueError('Explicit shared-background policy required')
    assets, layouts = source['assets'], source['layouts']
    if not assets or not layouts:
        raise ValueError('Assets and layouts are required')
    by_id = {}
    for asset in assets:
        key = asset['id']
        if not isinstance(key, str) or not key or key in by_id:
            raise ValueError('Asset IDs must be unique nonempty strings')
        by_id[key] = asset
        identities(asset, 'family_ids')
        identities(asset, 'visual_dependency_ids')
        if asset.get('dependency_closure_complete') is not True:
            raise ValueError('Incomplete dependency closure: ' + key)

    seen_layouts, restricted, used = set(), set(), {}
    physical = {}
    violations = []
    for layout in layouts:
        key, split = layout['id'], layout['split']
        if not isinstance(key, str) or not key or key in seen_layouts or split not in SPLITS:
            raise ValueError('Unique layout ID and valid split required')
        seen_layouts.add(key)
        # Conservative actual-visible instance superset includes distant HLOD leaves.
        # A site key additionally keeps different poses of the same site together.
        for field in ('site_ids', 'visible_physical_instance_ids'):
            for identity in identities(layout, field):
                physical.setdefault((field, identity), set()).add(split)
        if layout.get('physical_visibility_closure_complete') is not True:
            raise ValueError('Incomplete physical visibility closure: ' + key)
        if not layout['uses']:
            raise ValueError('Layout asset uses are required')
        for use in layout['uses']:
            asset_id, role = use['asset_id'], use['role']
            if asset_id not in by_id or role not in ROLES:
                raise ValueError('Unknown asset or role')
            if type(use['intersects_query']) is not bool:
                raise ValueError('Query intersection must be adjudicated, not unknown')
            used.setdefault(asset_id, set()).add(split)
            if role != 'background' or use['intersects_query']:
                restricted.add(asset_id)

    # Restriction follows both family and dependency identity into background uses.
    # This catches renamed/derived assets and role laundering across splits.
    restriction_keys = set()
    for key in restricted:
        asset = by_id[key]
        for field in ('family_ids', 'visual_dependency_ids'):
            restriction_keys.update((field, value) for value in asset[field])
    shared = []
    for field in ('family_ids', 'visual_dependency_ids'):
        owners = {}
        for key, splits in used.items():
            for identity in by_id[key][field]:
                entry = owners.setdefault(identity, {'splits': set(), 'assets': set()})
                entry['splits'].update(splits)
                entry['assets'].add(key)
        for identity, entry in sorted(owners.items()):
            if len(entry['splits']) <= 1:
                continue
            row = dict(kind=field, identity=identity, splits=sorted(entry['splits']), assets=sorted(entry['assets']))
            if (field, identity) in restriction_keys:
                violations.append(dict(code='RESTRICTED_ASSET_OVERLAP', **row))
            else:
                shared.append(row)
    for (field, identity), splits in sorted(physical.items()):
        if len(splits) > 1:
            violations.append(dict(code='PHYSICAL_SOURCE_OVERLAP', kind=field, identity=identity, splits=sorted(splits)))
    return dict(status='FAIL_DECLARED_SPLIT_CONTRACT' if violations else 'PASS_DECLARED_SPLIT_CONTRACT_ONLY',
        isolation_policy=POLICY, benchmark_eligible=False,
        authority='DECLARED_METADATA_ONLY_NOT_GEOMETRY_VISIBILITY_OR_CAPTURE_ADMISSION',
        layout_count=len(layouts), restricted_assets=sorted(restricted),
        shared_background_identities=shared, violations=violations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.manifest.read_text(encoding='utf-8-sig')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    raise SystemExit(1 if result['violations'] else 0)

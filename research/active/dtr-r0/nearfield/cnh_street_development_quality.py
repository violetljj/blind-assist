"""Per-layout Development ledger; pending admission never becomes formal access.

Optional result JSON: {capture_manifest_sha256, results: [{layout_id, metric,
status, scope, evidence}]}. Metrics are geometry, energy_convergence,
label_precision; status is PASS/FAIL/PENDING. Results are recorded with source
hashes, not independently re-evaluated. Any recorded FAIL quarantines its layout;
a later PASS does not erase it. Use a new capture generation after a repair.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

METRICS = ('geometry', 'energy_convergence', 'label_precision')
SCOPE = 'STREET_DEVELOPMENT_PILOT_NOT_BENCHMARK'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def identity(row):
    return row['layout_id'], row['clip_id'], row['pose_index']


def build_ledger(spec, manifest, transport, manifest_sha256, result_documents=()):
    if spec.get('scope') != SCOPE or spec.get('data_role') != 'Development':
        raise ValueError('Explicit Street Development pilot required')
    layouts = spec['layouts']
    ids = [r['layout_id'] for r in layouts]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate layout IDs')
    rows = manifest.get('frames', [])
    if any(r['layout_id'] not in ids for r in rows):
        raise ValueError('Manifest contains undeclared layout')
    if any(str(r.get('split', '')).lower() in ('test', 'locked_test', 'blind')
           or r.get('data_role') != 'Development' for r in rows):
        raise ValueError('Only explicitly Development frames may enter ledger')
    reports = transport.get('frames', [])
    common_faults = []
    if transport.get('status') != 'PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY':
        common_faults.append('TRANSPORT_NOT_FINALIZED')
    if transport.get('frame_count') != len(reports):
        common_faults.append('TRANSPORT_RECEIPT_INTERNAL_COUNT_MISMATCH')
    entries = {}
    for layout in layouts:
        lid = layout['layout_id']
        captured = [r for r in rows if r['layout_id'] == lid]
        checked = [r for r in reports if r['layout_id'] == lid]
        expected = {(lid, clip['id'], i) for clip in layout['clips'] for i in range(len(clip['poses']))}
        issues = list(common_faults)
        keys = [identity(r) for r in captured]
        if set(keys) != expected or len(keys) != len(expected):
            issues.append('LAYOUT_FRAME_MEMBERSHIP_MISMATCH')
        if Counter(identity(r) for r in checked) != Counter(keys):
            issues.append('LAYOUT_TRANSPORT_MEMBERSHIP_MISMATCH')
        by_key = {identity(r): r for r in checked}
        for row in captured:
            checked_row = by_key.get(identity(row), {})
            if any(checked_row.get(k) != row.get(k) for k in ('id', 'folder', 'physical_site_id', 'environment_category', 'data_role', 'nominal_time_s', 'asset_ids')):
                issues.append('FRAME_TRANSPORT_METADATA_MISMATCH')
            if not checked_row.get('hashes'):
                issues.append('FRAME_TRANSPORT_HASH_RECEIPT_MISSING')
            if row.get('physical_site_id') != layout.get('physical_site_id'):
                issues.append('FRAME_PHYSICAL_SITE_MISMATCH')
        entries[lid] = dict(layout_id=lid, environment_category=layout.get('environment_category'),
            physical_site_id=layout.get('physical_site_id'), frame_count=len(captured), expected_frames=len(expected),
            transport=dict(status='FAIL' if issues else 'PASS_RECEIPT_MATCH_ONLY', issues=sorted(set(issues)),
                payload_rehash='NOT_RUN_LEDGER_MATCHES_EXISTING_TRANSPORT_RECEIPT'),
            checks={key: dict(status='PENDING', evidence=[]) for key in METRICS},
            asset_isolation=dict(status='DEVELOPMENT_ONLY_NOT_ESTABLISHED',
                observed_asset_ids=sorted({str(a) for r in captured for a in r.get('asset_ids', [])}),
                reason='One shared Street200 block and reused insertion assets; instance IDs are not asset-family provenance'),
            formal_eligible=False, benchmark_eligible=False)
    for document in result_documents:
        if document.get('capture_manifest_sha256') != manifest_sha256:
            raise ValueError('Quality result belongs to another capture manifest')
        for result in document.get('results', []):
            lid, metric, status = result['layout_id'], result['metric'], result['status']
            if lid not in entries or metric not in METRICS or status not in ('PASS', 'FAIL', 'PENDING'):
                raise ValueError('Unknown layout, metric or status')
            if not result.get('scope') or not result.get('evidence'):
                raise ValueError('Explicit result scope and evidence reference required')
            check = entries[lid]['checks'][metric]
            check['evidence'].append(result)
            if check['status'] != 'FAIL':
                check['status'] = status
    for entry in entries.values():
        failed = entry['transport']['status'] == 'FAIL' or any(c['status'] == 'FAIL' for c in entry['checks'].values())
        entry['disposition'] = 'QUARANTINE_AFFECTED_LAYOUT' if failed else 'AVAILABLE_FOR_PROVISIONAL_DEVELOPMENT'
        entry['pending_checks'] = [k for k, c in entry['checks'].items() if c['status'] == 'PENDING']
    return dict(schema='cnh-street-development-quality-v1', status='DEVELOPMENT_LEDGER_ONLY',
        capture_manifest_sha256=manifest_sha256, source_gate_record=transport.get('source_gate', 'UNREPORTED'),
        source_gate_blocks_development=False, formal_eligible=False, benchmark_eligible=False,
        physical_site_count=len({r.get('physical_site_id') for r in layouts}),
        independent_sites_not_layout_count=True, temporal_authority=manifest.get('temporal_authority', 'UNREPORTED'),
        layouts=list(entries.values()), counts=dict(Counter(r['disposition'] for r in entries.values())))


def run(capture, output, results=()):
    capture = Path(capture).resolve(strict=True)
    sources = {name: capture/name for name in ('source/spec.json', 'raw-manifest.json', 'format-receipt.json')}
    ledger = build_ledger(read(sources['source/spec.json']), read(sources['raw-manifest.json']),
        read(sources['format-receipt.json']), digest(sources['raw-manifest.json']), [read(p) for p in results])
    ledger['input_sha256'] = {name: digest(path) for name, path in sources.items()}
    ledger['result_documents'] = [dict(path=str(Path(p).resolve()), sha256=digest(p)) for p in results]
    ledger['code_sha256'] = digest(__file__)
    with Path(output).open('x', encoding='utf-8') as stream:
        json.dump(ledger, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--result', type=Path, action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(run(args.capture, args.output, args.result), indent=2))

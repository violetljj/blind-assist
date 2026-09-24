"""Read-only storage inventory and dry-run compaction plan. Never deletes files.

Keep canonical RGB PNG, depth EXR + validity, float16 normal/albedo + validity,
instance IDs, camera/inserted geometry and all other evidence. Retain isolated
depth first/last pose per layout/clip. Planned removals are BLOCKED until numeric
conversion readback and consumers/audits no longer require each original.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

CANONICAL = ('left.png', 'right.png', 'depth_left.exr', 'depth_right.exr',
    'depth_left_valid.npy', 'depth_right_valid.npy', 'normal_left.npy',
    'normal_left_valid.npy', 'albedo_left.npy', 'albedo_left_valid.npy', 'instance_left.png')
TRANSPORT_REPLACEMENTS = {'depth_left.transport.npy': ['depth_left.exr', 'depth_left_valid.npy'],
    'depth_right.transport.npy': ['depth_right.exr', 'depth_right_valid.npy'],
    'normal_left.transport.npy': ['normal_left.npy', 'normal_left_valid.npy'],
    'albedo_left.transport.npy': ['albedo_left.npy', 'albedo_left_valid.npy']}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def crosscheck_ids(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['layout_id'], row['clip_id']].append(row)
    return {row['id'] for group in groups.values()
            for row in (min(group, key=lambda r: r['pose_index']), max(group, key=lambda r: r['pose_index']))}


def plan(capture):
    root = Path(capture).resolve(strict=True)
    manifest_path = root/'raw-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    receipt_path = root/'format-receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    if receipt.get('status') != 'PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY':
        raise ValueError('Finalized source transport required before inventory')
    rows = manifest['frames']; sample_ids = crosscheck_ids(rows)
    receipt_rows = {r['id']: r for r in receipt.get('frames', [])}
    frames = []; total_bytes = 0; candidate_bytes = 0
    for row in rows:
        folder = (root/row['folder']).resolve(strict=True)
        if not folder.is_relative_to(root):
            raise ValueError('Frame folder escapes capture')
        # This tool follows no payload symlinks/junctions, even though it never deletes.
        logical = root/row['folder']
        if logical.is_symlink() or (hasattr(logical, 'is_junction') and logical.is_junction()):
            raise ValueError('Frame reparse point prohibited')
        missing = [name for name in CANONICAL if not (folder/name).is_file()]
        files = []
        for path in sorted(folder.iterdir()):
            if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
                raise ValueError('Payload reparse point prohibited')
            if not path.is_file():
                continue
            digest = sha(path); size = path.stat().st_size; total_bytes += size
            old_hash = receipt_rows.get(row['id'], {}).get('hashes', {}).get(path.name)
            match = 'MATCH' if old_hash == digest else 'MISMATCH' if old_hash else 'NOT_RECORDED'
            action = 'KEEP'; blockers = []; replacements = []
            if path.name in TRANSPORT_REPLACEMENTS:
                action = 'PLAN_REMOVE_BLOCKED'; replacements = TRANSPORT_REPLACEMENTS[path.name]
                blockers = ['CANONICAL_NUMERIC_ROUNDTRIP_VERIFICATION_REQUIRED',
                    'TRANSPORT_DEPENDENT_GEOMETRY_AND_LABEL_AUDITS_MUST_COMPLETE_OR_MIGRATE',
                    'CONSUMER_AND_FINALIZER_REPLAY_MIGRATION_REQUIRED']
                if path.name.startswith(('normal_', 'albedo_')):
                    blockers.append('FLOAT32_TO_FLOAT16_QUANTIZATION_DISCLOSURE_AND_ERROR_CHECK_REQUIRED')
            elif path.name.startswith('isolated_depth_') and path.name.endswith('.transport.npy'):
                if row['id'] not in sample_ids:
                    action = 'PLAN_REMOVE_BLOCKED'
                    blockers = ['ALL_FRAME_ID_LABEL_PRECISION_AUDIT_MUST_COMPLETE',
                        'FIRST_LAST_PER_CLIP_CROSSCHECK_ONLY_NOT_FULL_LABEL_REPRODUCIBILITY']
                else:
                    action = 'KEEP_CROSSCHECK_FIRST_LAST_PER_LAYOUT_CLIP'
            if action == 'PLAN_REMOVE_BLOCKED':
                candidate_bytes += size
                if missing:
                    blockers.append('CANONICAL_FILES_MISSING')
                if match != 'MATCH':
                    blockers.append('SOURCE_TRANSPORT_HASH_NOT_VERIFIED')
            files.append(dict(path=path.relative_to(root).as_posix(), bytes=size, sha256=digest,
                original_transport_hash_status=match, action=action, replacements=replacements, blockers=blockers))
        frames.append(dict(frame_id=row['id'], layout_id=row['layout_id'], clip_id=row['clip_id'],
            pose_index=row['pose_index'], crosscheck_sample=row['id'] in sample_ids,
            missing_canonical=missing, files=files))
    return dict(schema='cnh-street-storage-dry-run-v1', status='DRY_RUN_ONLY_NO_DELETIONS',
        executable=False, files_deleted=0, bytes_reclaimed=0, capture_root=str(root),
        source_manifest_sha256=sha(manifest_path), source_format_sha256=sha(receipt_path), code_sha256=sha(__file__),
        canonical_frame_files=list(CANONICAL),
        retained_other_evidence='ALL_NONCANDIDATES_UNCHANGED_INCLUDING_CAMERA_INSERTED_GEOMETRY_RECEIPTS_EVALUATOR_SOURCE',
        original_format_receipt='IMMUTABLE_HISTORICAL_TRANSPORT_RECEIPT_NOT_CURRENT_PAYLOAD_INVENTORY',
        future_cleanup='Requires new canonical inventory with hashes, retained original hashes, numeric conversion checks and actual deletion receipt; this plan grants no deletion authority',
        frame_bytes=total_bytes, planned_removal_bytes=candidate_bytes,
        retained_frame_bytes_if_unblocked=total_bytes-candidate_bytes,
        frame_count=len(frames), crosscheck_frame_count=len(sample_ids), frames=frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = plan(args.capture)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'frames'}, indent=2))

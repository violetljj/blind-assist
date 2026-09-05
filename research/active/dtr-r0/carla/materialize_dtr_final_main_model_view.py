"""Derive main-only R1 C2 model input without auxiliary observations or truth.

The source header may name twelve episodes. Only the ten main episode payloads
are opened. Alignment membership and dependent receipt hashes are rebuilt; raw
alignment projections, plans, pixels, and observations other than receipt links
are preserved. This does not admit any source or authorize model inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import dtr_carla_rgbd_model_adapter as adapter

MAIN_IDS = tuple(f'ep_{i:02d}' for i in range(1,11))
SOURCE_IDS = set(MAIN_IDS) | {'ep_11','ep_12'}


def read(path):
    value=json.loads(path.read_text(encoding='utf-8'))
    adapter.assert_sanitized_model_value(value)
    return value


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False)
        handle.write('\n')


def materialize(source: Path, output: Path, *, hardlink: bool=False):
    source=source.resolve(strict=True)
    output=output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('source_output_overlap')
    manifest=read(source/'manifest.json')
    adapter._exact_fields(manifest,adapter.C2_ROOT_FIELDS,'main_view_root')
    if manifest['schema_version'] != adapter.C2_ROOT_SCHEMA:
        raise ValueError('main_view_requires_C2_v2')
    for key, filename in [('rgbd_alignment_receipt','rgbd_alignment_receipt.json'),
                          ('model_contract','model_contract.json'),
                          ('camera_calibration','camera_calibration.json')]:
        if manifest[key]['path'] != filename:
            raise ValueError('main_view_noncanonical_root_reference')
    references={row['episode_id']:row for row in manifest['episodes']}
    if set(references)!=SOURCE_IDS or len(manifest['episodes'])!=12:
        raise ValueError('main_view_requires_exact_twelve_source_episodes')
    alignment_context=adapter._parse_c2_alignment_receipt(source,manifest['rgbd_alignment_receipt'],manifest['experiment_id'])
    adapter._validate_c2_model_contract(source,reference=manifest['model_contract'],
                                      alignment=alignment_context,sync_explicit=True)
    if set(alignment_context['episodes']) != SOURCE_IDS:
        raise ValueError('main_view_alignment_membership')
    alignment=alignment_context['receipt']
    alignment['episodes']=[row for row in alignment['episodes'] if row['episode_id'] in MAIN_IDS]
    alignment['receipt_sha256']=hashlib.sha256(adapter.canonical_json_bytes(
        {k:v for k,v in alignment.items() if k!='receipt_sha256'})).hexdigest().upper()
    output.mkdir(parents=True,exist_ok=False)
    lineage={'schema':'dtr-r1-main-model-view-lineage-v1',
        'source_manifest_sha256':adapter.sha256_file(source/'manifest.json'),
        'source_alignment_file_sha256':alignment_context['file_sha256'],
        'source_episode_manifest_sha256':{},'selected_episode_ids':list(MAIN_IDS),
        'storage':'hardlinks' if hardlink else 'copies',
        'authority':'DERIVED_MAIN_INPUT_VIEW_NOT_SOURCE_ADMISSION',
        'implementation_sha256':adapter.sha256_file(Path(__file__))}

    def copy_reference(relative,expected):
        path=adapter.resolve_model_path(source,relative,'main_view_source')
        adapter.validate_file_hash(path,expected,'main_view_source')
        target=output/relative
        if Path(relative).is_absolute() or not target.resolve().is_relative_to(output):
            raise ValueError('main_view_output_escape')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            adapter.validate_file_hash(target,expected,'main_view_reused_reference')
        elif hardlink:
            os.link(path,target)
        else:
            # Exclusive creation prevents accidentally overwriting an existing file.
            with path.open('rb') as incoming,target.open('xb') as outgoing:
                shutil.copyfileobj(incoming,outgoing)

    alignment_relative=manifest['rgbd_alignment_receipt']['path']
    write(output/alignment_relative,alignment)
    alignment_file_hash=adapter.sha256_file(output/alignment_relative)
    manifest['rgbd_alignment_receipt'].update(receipt_sha256=alignment['receipt_sha256'],sha256=alignment_file_hash)
    model_contract=read(adapter.resolve_model_path(source,manifest['model_contract']['path'],'main_view_contract'))
    model_contract['rgbd_alignment'].update(receipt_sha256=alignment['receipt_sha256'],file_sha256=alignment_file_hash)
    write(output/manifest['model_contract']['path'],model_contract)
    manifest['model_contract']['sha256']=adapter.sha256_file(output/manifest['model_contract']['path'])
    calibration=manifest['camera_calibration']
    copy_reference(calibration['path'],calibration['sha256'])
    selected=[]
    for episode_id in MAIN_IDS:
        reference=references[episode_id]
        expected_manifest=f'episodes/{episode_id}/manifest.json'
        if reference['manifest_path']!=expected_manifest:
            raise ValueError('main_view_noncanonical_episode_path')
        path=adapter.resolve_model_path(source,expected_manifest,'main_view_episode')
        adapter.validate_file_hash(path,reference['manifest_sha256'],'main_view_episode')
        episode=read(path)
        lineage['source_episode_manifest_sha256'][episode_id]=reference['manifest_sha256']
        plan=episode['issued_plan']
        if plan['path']!=f'plans/{episode_id}.json':
            raise ValueError('main_view_noncanonical_plan_path')
        copy_reference(plan['path'],plan['file_sha256'])
        observations=path.parent/'observations.jsonl'
        adapter.validate_file_hash(observations,episode['observations_sha256'],'main_view_observations')
        rows=[]
        for line in observations.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row=json.loads(line)
            adapter.assert_sanitized_model_value(row)
            if row['episode_id']!=episode_id:
                raise ValueError('main_view_observation_episode')
            for modality,directory in [('wearable_rgb','rgb'),('metric_depth','depth')]:
                image=row[modality]
                if image['path']!=f'episodes/{episode_id}/{directory}/{row["sample_index"]:06d}.png':
                    raise ValueError('main_view_noncanonical_payload_path')
                copy_reference(image['path'],image['sha256'])
            row['frame_alignment']['receipt_sha256']=alignment['receipt_sha256']
            rows.append(row)
        destination=output/f'episodes/{episode_id}/observations.jsonl'
        with destination.open('x',encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        episode['rgbd_alignment']['receipt_sha256']=alignment['receipt_sha256']
        episode['observations_sha256']=adapter.sha256_file(destination)
        write(output/expected_manifest,episode)
        reference['manifest_sha256']=adapter.sha256_file(output/expected_manifest)
        selected.append(reference)
    manifest['episodes']=selected
    write(output/'manifest.json',manifest)
    contract=adapter.load_model_contract(output,validate_payload_hashes=True)
    if tuple(e.episode_id for e in contract.episodes)!=MAIN_IDS:
        raise ValueError('main_view_derived_membership')
    for episode in contract.episodes:
        for observation in episode.observations:
            for path in (observation.rgb.path,observation.depth.path):
                if not path.resolve().is_relative_to(output) or path.is_symlink():
                    raise ValueError('main_view_external_payload')
    lineage.update(derived_manifest_sha256=contract.manifest_sha256,
                   frames=sum(len(e.observations) for e in contract.episodes))
    write(output/'main_view_lineage.json',lineage)
    return lineage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hardlink',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[4]
    if not args.output.resolve().is_relative_to((repo/'artifacts.local').resolve()):
        parser.error('Output must remain under canonical artifacts.local')
    print(json.dumps(materialize(args.source_model_root,args.output,hardlink=args.hardlink),indent=2))


if __name__=='__main__':
    main()

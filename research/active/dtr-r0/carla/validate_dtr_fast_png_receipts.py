"""Independently decode the three Development fast-PNG capture shards.

Uses Pillow, never the capture encoder. Opens only implementation receipts,
raw capture result metadata, pixel journals and PNG payloads; no evaluator rows.
The resulting receipt verifies encoding integrity, not a model/source score.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

SHARDS=(('FINAL_A','depth'),('FINAL_B','wearable'),('FINAL_B','depth'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest().upper()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition,label):
    if not condition:
        raise ValueError(label)


def relative_file(root,relative):
    value=Path(relative)
    require(not value.is_absolute() and '..' not in value.parts,'payload_path_escape')
    path=(root/value).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(),'payload_path_escape')
    return path


def validate_shard(root,group,sensor):
    group_root=root/'raw'/group
    shard=(group_root/'shards'/sensor).resolve(strict=True)
    receipt=group_root/'encoder-receipts'/sensor
    implementation_path=receipt/'implementation.json'
    outcome_path=receipt/'outcome.json'
    journal_path=receipt/'raw-pixels.jsonl'
    result_path=shard/'result.json'
    implementation=read(implementation_path)
    outcome=read(outcome_path)
    result=read(result_path)
    require(implementation['schema']=='dtr-c2-fast-png-implementation-v1','implementation_schema')
    require(outcome['schema']=='dtr-c2-fast-png-outcome-v1','outcome_schema')
    require(implementation['sensor']==outcome['sensor']==result['sensor']==sensor,'sensor_identity')
    require(implementation['authority']=='SYNTHETIC_DEVELOPMENT_COMPOSITE_NOT_FRESH_CONFIRMATION','composite_authority')
    require(set(implementation['files'])=={'base_capture','encoder','wrapper','protocol'},'implementation_membership')
    for name,reference in implementation['files'].items():
        require(sha(Path(reference['path']))==reference['sha256'],f'implementation_hash:{name}')
    require(outcome['status']=='COMPLETE' and outcome['exit_code']==0,'capture_not_complete')
    require(outcome['implementation_receipt_sha256']==sha(implementation_path),'implementation_receipt_hash')
    require(outcome['raw_pixel_journal_sha256']==sha(journal_path),'journal_hash')
    require(outcome['base_result_sha256']==sha(result_path),'base_result_binding')
    require(result['status']=='DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE','base_status')
    require(all(value is True for value in result['checks'].values()),'base_checks')
    require(result['capture_script_sha256']==implementation['files']['base_capture']['sha256'],'base_capture_identity')
    require(result['protocol_sha256']==implementation['files']['protocol']['sha256'],'protocol_identity')
    expected=result['payload_count']
    require(isinstance(expected,int) and expected>0,'expected_payload_count')
    require(outcome['payload_count']==expected,'outcome_count')
    seen=set()
    count=0
    total_bytes=0
    with journal_path.open(encoding='utf-8') as handle:
        for line in handle:
            require(bool(line.strip()),'empty_journal_row')
            row=json.loads(line)
            require(set(row)=={'payload_path','world_frame','sensor_timestamp_s','width','height',
                              'raw_rgba_sha256','png_sha256','png_bytes'},'journal_row_schema')
            path=relative_file(shard,row['payload_path'])
            relative=path.relative_to(shard).as_posix()
            require(relative not in seen,'duplicate_payload')
            require(path.suffix.lower()=='.png','payload_not_png')
            seen.add(relative)
            require(isinstance(row['world_frame'],int),'world_frame_type')
            require(math.isfinite(row['sensor_timestamp_s']),'timestamp_nonfinite')
            require(path.stat().st_size==row['png_bytes'],'png_size')
            require(sha(path)==row['png_sha256'],'png_hash')
            with Image.open(path) as image:
                require(image.format=='PNG','decoder_format')
                require(image.size==(row['width'],row['height']),'decoder_dimensions')
                require(image.mode=='RGBA','decoder_not_native_rgba')
                # Force independent full decompression; do not call fast_sensor_png.
                raw=image.convert('RGBA').tobytes()
            require(hashlib.sha256(raw).hexdigest().upper()==row['raw_rgba_sha256'],'decoded_rgba_hash')
            count+=1
            total_bytes+=row['png_bytes']
    require(count==expected,'journal_expected_count')
    actual={path.relative_to(shard).as_posix() for path in shard.rglob('*.png')}
    require(seen==actual,'payload_membership')
    return {'group':group,'sensor':sensor,'status':'PASS','expected_payload_count':expected,
        'validated_payload_count':count,'png_bytes':total_bytes,
        'implementation_receipt_sha256':sha(implementation_path),'outcome_receipt_sha256':sha(outcome_path),
        'raw_pixel_journal_sha256':sha(journal_path),'base_result_sha256':sha(result_path),
        'implementation_hashes':{name:value['sha256'] for name,value in implementation['files'].items()}}


def validate(root):
    root=Path(root).resolve(strict=True)
    output=root/'fast-png-validation.json'
    require(not output.exists(),'validation_output_already_exists')
    rows=[validate_shard(root,group,sensor) for group,sensor in SHARDS]
    value={'schema':'dtr-c2-fast-png-independent-validation-v1','status':'PASS',
        'authority':'DEVELOPMENT_ENCODING_INTEGRITY_ONLY',
        'decoder':'Pillow.Image RGBA independent full decompression',
        'validator_sha256':sha(Path(__file__)),'shards':rows,
        'shard_count':len(rows),'validated_payload_count':sum(row['validated_payload_count'] for row in rows)}
    with output.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False)
        handle.write('\n')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(validate(args.root),indent=2))

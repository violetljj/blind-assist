"""Development-only C2 capture wrapper: native scene code, lossless fast PNG.

The base capture module is not edited and its result hashes remain literal.
Additional implementation identity and raw RGBA pixel digests live separately
under encoder-receipts/<sensor>/; they must accompany any composite provenance.
This entry point does not restore frozen R1 or fresh-confirmation authority.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np
import fast_sensor_png


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest().upper()


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False)
        handle.write('\n')


class FastPngWriter:
    """The seven metadata fields exactly match native save_image."""
    def __init__(self,shard_root,journal):
        self.shard_root=Path(shard_root).resolve()
        self.journal=journal
        self.count=0
        self.encode_write_seconds=0.
        self.digest_seconds=0.

    def __call__(self,image,path):
        path=Path(path)
        resolved=path.resolve()
        if not resolved.is_relative_to(self.shard_root):
            raise ValueError('fast_png_output_outside_owned_shard')
        started=time.perf_counter()
        encoded=fast_sensor_png.encode_bgra(image.raw_data,image.width,image.height)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:
            handle.write(encoded)
        self.encode_write_seconds+=time.perf_counter()-started
        started=time.perf_counter()
        rgba=np.frombuffer(image.raw_data,dtype=np.uint8).reshape(image.height,image.width,4)[:,:,[2,1,0,3]]
        # Hash the actual serialized pixel order, independent of PNG encoding.
        digest=hashlib.sha256(rgba.tobytes()).hexdigest().upper()
        encoded_hash=hashlib.sha256(encoded).hexdigest().upper()
        self.digest_seconds+=time.perf_counter()-started
        metadata={'path':str(path),'bytes':path.stat().st_size,'sha256':encoded_hash,
            'world_frame':int(image.frame),'sensor_timestamp_s':float(image.timestamp),
            'width':int(image.width),'height':int(image.height)}
        self.journal.write(json.dumps({'payload_path':resolved.relative_to(self.shard_root).as_posix(),
            'world_frame':metadata['world_frame'],'sensor_timestamp_s':metadata['sensor_timestamp_s'],
            'width':metadata['width'],'height':metadata['height'],
            'raw_rgba_sha256':digest,'png_sha256':encoded_hash,'png_bytes':metadata['bytes']},
            sort_keys=True,allow_nan=False)+'\n')
        self.journal.flush()
        self.count+=1
        return metadata


def main():
    base=importlib.import_module('capture_dtr_carla_c2_rich_scene')
    # Parse with the unchanged native parser. main() receives the same argv,
    # including host/port/rpc-timeout; only save_image is replaced below.
    args=base.parse_args()
    output=args.output_root.resolve(strict=True)
    shard=output/'shards'/args.sensor
    if shard.exists():
        raise FileExistsError(f'refusing shard overwrite: {shard}')
    files={'base_capture':Path(base.__file__).resolve(),
           'encoder':Path(fast_sensor_png.__file__).resolve(),
           'wrapper':Path(__file__).resolve(),
           'protocol':args.protocol.resolve(strict=True)}
    hashes={name:sha(path) for name,path in files.items()}
    receipt_root=output/'encoder-receipts'/args.sensor
    receipt_root.mkdir(parents=True,exist_ok=False)
    write_json(receipt_root/'implementation.json',{
        'schema':'dtr-c2-fast-png-implementation-v1','sensor':args.sensor,
        'authority':'SYNTHETIC_DEVELOPMENT_COMPOSITE_NOT_FRESH_CONFIRMATION',
        'files':{name:{'path':str(path),'sha256':hashes[name]} for name,path in files.items()},
        'base_result_capture_hash_means':'UNCHANGED_BASE_MODULE_ONLY_SEE_THIS_COMPOSITE_RECEIPT',
        'pixel_encoding':'native BGRA reordered to RGBA; PNG8 RGBA zlib level1 filter0',
        'pixel_equivalence':'LOSSLESS_NATIVE_BUFFER_ENCODING_NOT_CROSS_RUN_PIXEL_IDENTITY'})
    original=base.save_image
    started=time.perf_counter()
    outcome={'schema':'dtr-c2-fast-png-outcome-v1','sensor':args.sensor,'status':'FAILED'}
    journal_path=receipt_root/'raw-pixels.jsonl'
    writer=None
    try:
        with journal_path.open('x',encoding='utf-8') as journal:
            writer=FastPngWriter(shard,journal)
            base.save_image=writer
            code=base.main()
        if {name:sha(path) for name,path in files.items()} != hashes:
            raise RuntimeError('fast_png_implementation_changed_during_capture')
        outcome.update(status='COMPLETE' if code==0 else 'BASE_CAPTURE_FAILED',exit_code=code)
        return code
    except BaseException as exc:
        outcome['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        base.save_image=original
        outcome.update(elapsed_seconds=time.perf_counter()-started,
            implementation_receipt_sha256=sha(receipt_root/'implementation.json'),
            raw_pixel_journal_sha256=sha(journal_path) if journal_path.exists() else None,
            payload_count=writer.count if writer else 0,
            encode_write_seconds=writer.encode_write_seconds if writer else 0.,
            digest_seconds=writer.digest_seconds if writer else 0.,
            base_result_sha256=sha(shard/'result.json') if (shard/'result.json').is_file() else None)
        write_json(receipt_root/'outcome.json',outcome)


if __name__=='__main__':
    raise SystemExit(main())

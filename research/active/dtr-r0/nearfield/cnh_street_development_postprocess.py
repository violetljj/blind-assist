"""Bounded CPU postprocessing of a completed Street Development capture.

Runs two-frame baseline interface smoke, per-layout pending quality ledger and
all-frame canonical encoding verification. Writes new outputs only; never deletes
raw data and never converts encoding equality into independent label precision.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from cnh_street_development_baseline import run as baseline, sha
from cnh_street_development_quality import run as quality
from cnh_street_canonical_verify import verify


def run(capture, output):
    journal = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    if not journal or json.loads(Path(journal).read_text(encoding='utf-8-sig')).get('state') != 'running':
        raise RuntimeError('Governed research-ue execution required')
    capture = Path(capture).resolve(strict=True)
    output = Path(output).resolve()
    artifact_root = (Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    if not output.is_relative_to(artifact_root) or output == artifact_root or output.is_relative_to(capture):
        raise ValueError('Output must be a separate artifacts.local evidence directory')
    if any((output/name).exists() for name in ('result.json', 'baseline', 'quality-ledger.json', 'canonical-verification.json')):
        raise FileExistsError('Never overwrite an existing postprocessing attempt')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(schema='cnh-street-development-postprocess-v1', status='RUNNING',
        evidence_boundary='DEVELOPMENT_INTERFACE_AND_ENCODING_ONLY_NO_TASK_ACCURACY_OR_HARDWARE_CLAIM',
        backend='CPU_NUMPY_IO', cpu_reason='Two-frame small-array interface smoke and lossless readback/hashing; no training, CUDA or UE',
        capture=str(capture), source_manifest_sha256=sha(capture/'raw-manifest.json'),
        source_format_sha256=sha(capture/'format-receipt.json'), files_deleted=0,
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in (
            'cnh_street_development_postprocess.py', 'cnh_street_development_baseline.py',
            'cnh_street_development_quality.py', 'cnh_street_canonical_verify.py', 'cnh_street_storage_plan.py')})
    try:
        first = baseline(capture, output/'baseline', max_frames=2)
        result['baseline'] = dict(status=first['status'], frames=len(first['frames']),
            receipt='baseline/receipt.json', calibration=first['hardware_validity'], clearance=first['clearance'])
        ledger = quality(capture, output/'quality-ledger.json')
        result['quality'] = dict(status=ledger['status'], counts=ledger['counts'],
            formal_eligible=ledger['formal_eligible'], physical_site_count=ledger['physical_site_count'])
        encoded = verify(capture, output/'canonical-verification.json')
        result['canonical'] = {key:encoded[key] for key in ('status', 'frame_count', 'max_attribute_quantization_error',
            'candidate_bytes', 'label_precision', 'files_deleted')}
        result['status'] = 'PASS_DEVELOPMENT_INTERFACE_AND_CANONICAL_ENCODING_ONLY'
    except BaseException:
        result.update(status='FAIL_POSTPROCESS', error=traceback.format_exc())
        raise
    finally:
        result['wall_s'] = time.monotonic()-started
        with (output/'result.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.capture, args.output), indent=2))

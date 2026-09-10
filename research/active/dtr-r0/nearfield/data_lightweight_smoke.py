"""CPU source-loader parity and byte-preservation checks for compact captures."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import time
import warnings
import zipfile

import numpy as np
from PIL import Image

from data_lightweight import CompactSource, FORMAT, MANIFEST, digest_file, pack_source, verify_source


def check_equal(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def run(source, archive, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    nearfield = Path(__file__).resolve().parent
    detail_code = nearfield / 'mz16_detail_cache.py'
    inference_code = nearfield / 'mz40_frozen_inference.py'
    crop = next(ast.literal_eval(node.value) for node in ast.parse(detail_code.read_text(encoding='utf-8')).body
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'CROP' for target in node.targets))
    started = time.perf_counter()
    samples = []
    with CompactSource(source) as plain, CompactSource(archive) as packed:
        native_paths = [name for name in packed.paths() if '/evaluator/native/' in name and name.endswith('.npy')]
        scenes = sorted({name.split('/capture/')[0] for name in native_paths})
        for scene in scenes:
            frames = sorted(name for name in native_paths if name.startswith(scene+'/capture/'))
            for native_path in dict.fromkeys([frames[0], frames[len(frames)//2], frames[-1]]):
                stem = Path(native_path).stem
                prefix = native_path.split('/evaluator/native/')[0]
                support_path = prefix + '/evaluator/world_support/' + stem + '.npy'
                image_path = prefix + '/model/sample/' + stem + '.png'
                native_plain = plain.load_array(native_path)
                native_packed = packed.load_array(native_path)
                check_equal(native_plain, native_packed)
                assert native_packed.dtype == np.float32 and native_packed.shape == (360,640)
                support_plain = plain.load_array(support_path)
                support_packed = packed.load_array(support_path)
                check_equal(support_plain, support_packed)
                assert support_packed.dtype == np.int8 and support_packed.shape == (2,360,640)
                with packed.open(native_path) as stream:
                    check_equal(native_plain, np.load(stream, allow_pickle=False))
                with Image.open(Path(source)/image_path) as original_image:
                    original_image = original_image.convert('RGB')
                    compact_image = packed.load_image(image_path).convert('RGB')
                    check_equal(np.asarray(original_image), np.asarray(compact_image))
                    # Match the actual CPU input transforms in frozen MZ40.
                    low_original = np.asarray(original_image.resize((256,144), Image.Resampling.BOX))
                    low_compact = np.asarray(compact_image.resize((256,144), Image.Resampling.BOX))
                    high_original = np.asarray(original_image.crop(crop))
                    high_compact = np.asarray(compact_image.crop(crop))
                    check_equal(low_original, low_compact)
                    check_equal(high_original, high_compact)
                samples.append(dict(native_path=native_path, image_path=image_path,
                                    native_dtype=str(native_packed.dtype), native_shape=list(native_packed.shape),
                                    support_dtype=str(support_packed.dtype), support_shape=list(support_packed.shape),
                                    low_input_sha256=hashlib.sha256(low_compact.tobytes()).hexdigest(),
                                    high_input_sha256=hashlib.sha256(high_compact.tobytes()).hexdigest(),
                                    array_bytes_equal=True, image_pixels_equal=True,
                                    mz40_low_high_input_bytes_equal=True, file_like_numpy_load=True))
        assert plain.read_json('completion.json') == packed.read_json('completion.json')
        path_rejections = []
        for illegal in ('../escape.npy', '/absolute.npy', 'C:/outside.npy', 'a//b.npy', 'a/./b.npy'):
            try:
                with packed.open(illegal):
                    pass
            except ValueError:
                path_rejections.append(illegal)
            else:
                raise AssertionError(f'Unsafe relative path accepted: {illegal}')
    compatibility_seconds = time.perf_counter()-started

    with tempfile.TemporaryDirectory(prefix='owned-smoke-', dir=output) as temp:
        owned = Path(temp).resolve()
        assert owned.is_relative_to(output) and owned != output
        raw = owned/'source'
        raw.mkdir()
        bits = np.array([0,0x80000000,0x7f800000,0xff800000,0x7fc00001,0x7fa12345,0x3f800000], dtype='<u4')
        np.save(raw/'special.npy', bits.view('<f4'), allow_pickle=False)
        fixture = owned/'fixture.source.zip'
        fixture_pack = pack_source(raw, fixture)
        fixture_verify = verify_source(fixture, raw)
        with CompactSource(fixture) as reader:
            restored = reader.load_array('special.npy')
        assert np.array_equal(restored.view('<u4'), bits)
        frozen_hash = digest_file(fixture)
        try:
            pack_source(raw, fixture)
        except FileExistsError:
            overwrite_rejected = True
        else:
            raise AssertionError('Existing package was overwritten')
        assert digest_file(fixture) == frozen_hash

        bad_checks = []
        for kind in ('traversal', 'duplicate'):
            bad = owned/(kind+'.zip')
            names = ['../bad.npy'] if kind == 'traversal' else ['same.npy', 'same.npy']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                with zipfile.ZipFile(bad, 'w') as z:
                    for name in names:
                        z.writestr(name, b'test')
                    z.writestr(MANIFEST, json.dumps(dict(format=FORMAT, entries=[])))
            try:
                with CompactSource(bad):
                    pass
            except ValueError:
                bad_checks.append(kind)
            else:
                raise AssertionError(f'Invalid {kind} package accepted')
    assert not owned.exists()
    result = dict(status='PASS', backend='CPU NumPy/Pillow; TASK_NOT_GPU_SUITABLE; no model fit or GPU rerun',
                  source=str(Path(source).resolve()), archive=str(Path(archive).resolve()),
                  sampled_frames=len(samples), scene_count=len(scenes), samples=samples,
                  compatibility_seconds=compatibility_seconds, crop_xyxy=crop,
                  preprocessing_code_sha256={str(detail_code):digest_file(detail_code), str(inference_code):digest_file(inference_code)},
                  json_reader_equal=True, path_rejections=path_rejections,
                  malformed_packages_rejected=bad_checks,
                  float32_nan_payload_infinity_signed_zero_bits_preserved=True,
                  fixture_roundtrip=fixture_verify, existing_package_overwrite_rejected=overwrite_rejected,
                  temporary_fixture_released=True, extracted_capture_files=0, source_mutations=0)
    (output/'loader-smoke.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key not in {'samples', 'preprocessing_code_sha256'}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.archive, args.output)

"""Adapt accepted expanded collection labels to the RGB-only QueryRGB cache.

This is CPU I/O and integrity checking, not inference or geometry relabeling.
EVAL payloads stay evaluator-side; no prediction outcomes are evaluated here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
from PIL import Image

from body_query_data import QueryRGB, fresh_output, read, sha, truth, write
from body_query_model import CALIBRATION


EXPECTED = {'train': 2500, 'dev': 1000, 'eval': 1500}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(dataset, output):
    started = time.perf_counter()
    root = Path(dataset).resolve(strict=True)
    source = read(root / 'manifest.json')
    index_hash = sha(root / 'index.json')
    require(source['status'] == 'PASS', 'Source dataset must be accepted')
    require(index_hash == source['index_sha256'], 'Source index hash mismatch')
    require(source['label_contract']['calibration'] == CALIBRATION, 'Calibration changed')
    frames = read(root / 'index.json')['frames']
    require(Counter(r['source_role'] for r in frames) ==
            {k.upper() + '_ONLY': v for k, v in EXPECTED.items()}, 'Source role denominator changed')
    require(len({r['frame_id'] for r in frames}) == 5000, 'Frame IDs must be unique')
    out = fresh_output(output)
    manifest = dict(schema='body-query-cache-v1', adapter_schema='body-query-expanded-data-v1',
                    status='BUILDING', calibration=CALIBRATION, source_dataset=str(root),
                    source_index_sha256=index_hash, source_manifest_sha256=sha(root / 'manifest.json'),
                    source_label_contract=source['label_contract'], partitions={}, groups={},
                    input='RGB only, fixed calibration',
                    label_semantics='Observed native visible-pixel counts capped3; zero is not certified empty space; support UNKNOWN=-1',
                    scope='Synthetic expanded same-world Development; source roles and groups preserved; shared assets/background possible')
    write(out / 'manifest.json', manifest)
    seen_rgb, seen_sites, seen_regions = {}, {}, {}

    def save(relative, values):
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values, allow_pickle=False)
        return dict(path=relative, sha256=sha(path))

    try:
        for role, size in EXPECTED.items():
            rows = [r for r in frames if r['source_role'] == role.upper() + '_ONLY']
            images = np.empty((size, 144, 256, 3), dtype=np.uint8)
            near = np.empty((size, 2), dtype=np.int64)
            counts = np.empty((size, 12), dtype=np.int64)
            support = np.empty((size, 2, 18, 32), dtype=np.int8)
            raw_counts, records = [], []
            for j, row in enumerate(rows):
                ident = row['frame_id']
                require(row['status'] == 'PASS', f'Unaccepted source frame {ident}')
                for mapping, key in ((manifest['groups'], row['group_id']),
                                     (seen_sites, row['site_id']), (seen_regions, row['region_id']),
                                     (seen_rgb, row['rgb_sha256'])):
                    require(mapping.setdefault(key, role) == role, f'Cross-role source key: {key}')
                image_path = Path(row['rgb_file']).resolve(strict=True)
                label_path = Path(row['label_file']).resolve(strict=True)
                require(sha(image_path) == row['rgb_sha256'], f'RGB hash mismatch: {ident}')
                require(sha(label_path) == row['labels']['sha256'], f'Label hash mismatch: {ident}')
                camera = row['camera']
                require(abs(camera['pitch']) < 1e-5 and abs(camera.get('roll', 0)) < 1e-5 and
                        abs(camera['z'] - row['floor_probe']['declared_floor_z_m'] - 1.7) < 1e-4,
                        f'Fixed-camera contract mismatch: {ident}')
                with np.load(label_path, allow_pickle=False) as data:
                    for key, shape in (('near', (2,)), ('counts', (12,)), ('raw_counts', (12,))):
                        require(data[key].shape == shape and data[key].dtype == np.int64 and
                                np.array_equal(data[key], row[key]), f'Label/index mismatch: {ident}/{key}')
                    require(np.all(data['raw_counts'] >= 0) and
                            np.array_equal(data['raw_counts'].clip(max=3), data['counts']),
                            f'Native count semantics mismatch: {ident}')
                    require(np.array_equal((data['counts'].reshape(2, 6).sum(1) >= 3).astype(np.int64),
                                           data['near']), f'Near aggregation mismatch: {ident}')
                    mask = data['support']
                    require(mask.shape == (2, 360, 640) and mask.dtype == np.int8 and
                            np.isin(mask, [-1, 0, 1]).all(), f'Invalid native support: {ident}')
                    require(np.array_equal((mask == 1).sum((1, 2)), data['raw_counts'].reshape(2, 6).sum(1)),
                            f'Native support/count totals mismatch: {ident}')
                    require(np.array_equal(mask[0] == -1, mask[1] == -1) and
                            int((mask[0] == -1).sum()) == row['unknown_native_pixels'],
                            f'UNKNOWN preservation mismatch: {ident}')
                    blocks = mask.reshape(2, 18, 20, 32, 20)
                    pooled = np.where((blocks == 1).any((2, 4)), 1,
                                      np.where((blocks == -1).any((2, 4)), -1, 0)).astype(np.int8)
                    require(data['support_pooled'].dtype == np.int8 and
                            np.array_equal(pooled, data['support_pooled']), f'Pooled support mismatch: {ident}')
                    near[j], counts[j], support[j] = data['near'], data['counts'], data['support_pooled']
                    raw_counts.append(data['raw_counts'].tolist())
                with Image.open(image_path) as image:
                    require(image.size == (640, 360), f'Native RGB size mismatch: {ident}')
                    images[j] = np.asarray(image.convert('RGB').resize((256, 144), Image.Resampling.BOX))
                records.append(dict(sample_index=ident, source_sample_index=row['sample_index'],
                                    name=row['name'], group_id=row['group_id'], source_role=row['source_role'],
                                    condition=row['condition'], family=row['family'], site=row['site_id'],
                                    region_id=row['region_id'], declared_range=row['declared_range'],
                                    actual_near=row['near'], source_rgb_sha256=row['rgb_sha256'],
                                    source_native_sha256=row['native_sha256'], source_labels_sha256=row['labels']['sha256']))
                if (j + 1) % 500 == 0:
                    print(f'{role}: {j + 1}/{size} hash-bound frames cached', flush=True)
            ids = [r['frame_id'] for r in rows]
            manifest['partitions'][role] = dict(rgb=save(f'model/{role}/rgb.npy', images),
                                               sample_indices=ids, frames=size,
                                               groups=len({r['group_id'] for r in rows}))
            folder = 'supervision' if role == 'train' else 'evaluator'
            record = dict(records=records, sample_indices=ids,
                          near=save(f'{folder}/{role}/near.npy', near),
                          counts=save(f'{folder}/{role}/counts.npy', counts),
                          support=save(f'{folder}/{role}/support.npy', support))
            write(out / folder / f'{role}.json', record)
            if role == 'train':
                write(out / 'evaluator/train.json', record)
            write(out / f'evaluator/{role}-raw-counts.json', dict(records=records, raw_counts=raw_counts))
        require(sha(root / 'index.json') == index_hash, 'Source index changed during cache build')
        manifest.update(status='PASS', backend='CPU', backend_reason='TASK_NOT_GPU_SUITABLE',
                        seconds=time.perf_counter() - started,
                        adapter_sha256=sha(Path(__file__)), sites=seen_sites, regions=seen_regions,
                        integrity_checks=dict(rgb_hashes=5000, label_hashes=5000, label_semantics=5000,
                                              source_index_hash=True, role_group_site_region_disjoint=True),
                        eval_outcome_scoring=False)
        write(out / 'manifest.json', manifest)
        for role, size in EXPECTED.items():
            rgb = QueryRGB(out, role)
            record, arrays = truth(out, role, training=(role == 'train'))
            require(len(rgb.ids) == size and rgb.ids == record['sample_indices'] and
                    arrays['near'].shape == (size, 2) and arrays['counts'].shape == (size, 12) and
                    arrays['support'].shape == (size, 2, 18, 32), 'Cache reader compatibility failed')
        write(out / 'adapter-validation.json', dict(status='PASS', partitions=EXPECTED,
              checks='Source hashes, copied native labels, UNKNOWN pooling, split isolation, QueryRGB/truth readback',
              inference=False, eval_outcome_scoring=False, seconds=time.perf_counter() - started))
        return manifest
    except Exception as exc:
        manifest['status'] = 'FAIL'
        write(out / 'manifest.json', manifest)
        write(out / 'failure.json', dict(error=repr(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.dataset, args.output)
    print('PASS:', result['integrity_checks'])

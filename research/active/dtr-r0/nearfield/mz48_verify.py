"""Primary saved-source checks; no fit, inference, extraction or recapture."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from data_lightweight import CompactSource, verify_source


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def masks(depth):
    rows, cols = np.indices(depth.shape)
    focal = 320 / np.tan(np.pi * 50 / 180)
    right = (cols - 319.5) / focal
    up = (179.5 - rows) / focal
    axial = depth.astype(np.float64)
    height = 1.7 + axial * up
    lateral = axial * right
    observable = np.isfinite(axial) & (axial > 0) & (axial < 100)
    observable &= axial * np.sqrt(1 + right**2 + up**2) <= 4
    events = []
    for front, width, bottom, top in [(.18, .28, .65, 1.4), (.13, .18, 1.4, 1.85)]:
        inside = observable & (abs(lateral) <= width) & (height >= bottom) & (height <= top)
        events.extend([inside & (axial >= front) & (axial < front + 1.5),
                       inside & (axial >= front + 1.5) & (axial <= front + 3)])
    return np.stack(events)


def verify(folder):
    archive = folder / 'training.source.zip'
    returned = read(folder / 'return.json')
    assert sha(archive) == returned['zip_sha256']
    full_hash_check = verify_source(archive)
    result = read(folder / 'dataset-v1/result.json')
    with CompactSource(archive) as source:
        metadata = source.read_json('evaluator/metadata.json')
        records, pairs = metadata['records'], metadata['pairs']
        packets = dict(np.load(io.BytesIO(source.read_bytes('model/packets.npz')), allow_pickle=False))
        labels = dict(np.load(io.BytesIO(source.read_bytes('evaluator/labels.npz')), allow_pickle=False))
        n = len(records)
        assert n == result['frames']
        assert packets['ranges'].shape == packets['valid'].shape == (n, 64, 2)
        for name, shape in [('truth', (n, 4)), ('known', (n, 4)),
                            ('source_presence', (n, 64, 2, 49)),
                            ('query_presence', (n, 64, 2, 49, 4)), ('cell_known', (n, 64, 49))]:
            assert labels[name].shape == shape and labels[name].dtype == np.bool_
        assert not (labels['query_presence'] & ~labels['source_presence'][..., None]).any()
        assert not (labels['source_presence'] & ~packets['valid'][..., None]).any()
        assert np.isfinite(packets['ranges']).all()
        predictor = source.read_json('model/predictor.json')
        assert predictor['rgb_files'] == [r['rgb'] for r in records]
        audited, observed = {}, []
        for i, row in enumerate(records):
            assert row['index'] == i
            np.testing.assert_array_equal(labels['truth'][i], row['event_truth'])
            np.testing.assert_array_equal(labels['known'][i], [row['source_valid']] * 4)
            assert source.entries[row['rgb']]['sha256'] == row['rgb_sha256']
            assert source.load_image(row['rgb']).size == (640, 360)
            if row['native_audit_sample']:
                key = f'evaluator/native/{i:04d}.npy'
                assert source.entries[key]['sha256'] == row['native_sha256']
                depth = source.load_array(key)
                event = masks(depth)
                np.testing.assert_array_equal(event.sum((1, 2)), row['event_counts'])
                np.testing.assert_array_equal(event.sum((1, 2)) >= 3, labels['truth'][i])
                audited[i] = depth, event
        for pair in pairs:
            a, b = pair['indices']
            assert records[a]['pair_id'] == records[b]['pair_id'] == pair['pair_id']
            assert records[a]['role'] == records[b]['role']
            assert records[a]['target_actors'] == records[b]['target_actors']
            assert pair['target_dictionary_equal'] and pair['actual_target_receipt_equal']
            if a in audited and b in audited:
                da, ma = audited[a]; db, mb = audited[b]
                diff = (ma != mb).sum((1, 2)).tolist()
                union = ma.any(0) | mb.any(0)
                error = float(abs(da[union] - db[union]).max()) if union.any() else 0.
                assert diff == pair['changed_event_mask_pixels']
                assert error == pair['event_depth_max_abs_m']
                observed.append(dict(pair_id=pair['pair_id'], changed_pixels=diff, max_depth_error_m=error))
        check = dict(status='PASS', archive_sha256=sha(archive), package_hash_check=full_hash_check,
                     frames=n, native_independently_checked=len(audited), paired_native_checks=observed,
                     label_shapes={k:list(v.shape) for k,v in labels.items()}, result=result,
                     cpu_reason='TASK_NOT_GPU_SUITABLE: package hashing and small saved source audit',
                     model_inference_frames=0, training_steps=0)
    (folder / 'primary-verify.json').write_text(json.dumps(check, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:check[k] for k in ('status', 'frames', 'native_independently_checked')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=Path, required=True)
    args = parser.parse_args()
    verify(args.shard)

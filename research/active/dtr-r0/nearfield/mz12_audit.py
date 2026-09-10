"""Independent scalar audit of frozen MZ12 existing-data replay."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from mz5_ensemble_readout import load_npz, read, write, sha
from mz11_audit import scalar_metrics


def grouped(logits, truth, rows, key):
    status = {}
    for prediction, target, row in zip(logits, truth, rows):
        correct = all(bool(prediction[q] >= 0) == bool(target[q]) for q in range(4))
        status[row[key]] = status.get(row[key], True) and correct
    return dict(correct=sum(status.values()), total=len(status))


def main(root, inventory, run):
    ir, receipt, start = read(inventory/'receipt.json'), read(run/'receipt.json'), read(run/'start.json')
    assert ir['status'] == receipt['status'] == 'PASS'
    for parent, record in [(inventory, ir), (run, receipt)]:
        for name, digest in record['outputs'].items():
            assert sha(parent/name) == digest, name
    for path, digest in ir['inputs'].items():
        assert sha(path) == digest
    for path, digest in receipt['frozen_hashes'].items():
        assert sha(path) == digest
    work = root/'artifacts.local/work'
    for name, digest in receipt['context_hashes'].items():
        parent = work/('body-query-10000-b-20260909/run-v1' if name == 'NEW-step2000.pt' else 'body-query-context-decoder-20260909/run-v1')
        assert sha(parent/name) == digest
    assert sha(Path(__file__).with_name('mz12_existing_replay.py')) == start['source_sha256']
    assert sha(Path(__file__).with_name('MZ12_EXISTING_DATA_PROTOCOL_20260910.md')) == start['protocol_sha256']
    assert sha(inventory/'receipt.json') == start['inventory_receipt_sha256']
    assert sha(inventory/'selected.json') == receipt['selection_sha256'] == start['selection_sha256']
    assert receipt['training_steps'] == start['training_steps'] == receipt['threshold_changes'] == 0
    rows = read(inventory/'selected.json')
    assert len(rows) == receipt['frames'] == 3000
    assert Counter(r['dataset'] for r in rows) == {'relation10000': 2000, 'distance5000': 1000}
    sources = {'relation10000': read(work/'body-query-10000-20260909/final-dataset-v2/index.json')['frames'],
               'distance5000': read(work/'body-query-distance-5000-20260909/final-dataset-v1/index.json')['records']}
    expected_order = [(dataset, i) for dataset, records in sources.items() for i, r in enumerate(records)
                      if (r.get('source_role') == 'DEV_ONLY' or r.get('source_partition') == 'dev')]
    assert [(r['dataset'], r['index']) for r in rows] == expected_order
    for row in rows:
        original = sources[row['dataset']][row['index']]
        distance = row['dataset'] == 'distance5000'
        assert original['accepted'] if distance else original['status'] == 'PASS'
        expected = dict(dataset=row['dataset'], index=row['index'],
            rgb=original['rgb_path' if distance else 'rgb_file'], rgb_sha=original['rgb_sha256'],
            native=original['native_path' if distance else 'native_file'], native_sha=original['native_sha256'],
            site=original['site_id'], group=original['pair_id' if distance else 'group_id'],
            family=original['family'], condition='HEAD_ONLY' if distance else original['condition'],
            role='DEV_ONLY', camera=original['camera'])
        assert row == expected
    a, result = load_npz(run/'predictions.npz'), read(run/'result.json')
    np.testing.assert_array_equal(a['dataset'], [r['dataset'] for r in rows])
    np.testing.assert_array_equal(a['truth'], a['native_counts'] >= 3)
    cutoff = np.load(work/'mz11-selective-addition-20260910/run-v1/threshold.npy', allow_pickle=False)
    gate = torch.load(work/'mz11-selective-addition-20260910/run-v1/gate.pt', map_location='cpu', weights_only=True)
    weight, bias = gate['weight'].numpy(), gate['bias'].numpy()
    max_score_error, preserved = 0., 0
    for i in range(3000):
        for q in range(4):
            baseline, source, available = a['MZ5'][i,q], a['SOURCE'][i,q], bool(a['available'][i,q])
            eligible = baseline < 0 and source >= 0 and available
            accepted = eligible and a['gate_score'][i,q] >= cutoff[q]
            assert bool(a['accepted'][i,q]) == accepted
            assert a['MZ11'][i,q] == (source if accepted else baseline)
            if baseline >= 0:
                assert a['MZ11'][i,q] == baseline
                preserved += 1
            feature = np.array([source/4 if available else 0., np.log1p(a['candidate_count'][i,q])/6,
                                a['candidate_zones'][i,q]/8, a['candidate_returns'][i,q]/8], dtype=np.float32)
            replay = sum(float(feature[j])*float(weight[q,j]) for j in range(4))+float(bias[q])
            max_score_error = max(max_score_error, abs(replay-float(a['gate_score'][i,q])))
    assert max_score_error < 1e-4
    audited = {}
    for dataset in sources:
        ids = [i for i, r in enumerate(rows) if r['dataset'] == dataset]
        selected = [rows[i] for i in ids]
        truth = a['truth'][ids]
        cohort = {}
        for arm in ['MZ5', 'ADAPTED', 'SOURCE', 'MZ11']:
            z = a[arm][ids]
            near, far = [0, 0], [0, 0]
            for pred, target in zip(z, truth):
                for part, q in enumerate([0, 2]):
                    near[part] += int(pred[q] >= 0 and target[q+1] and not target[q])
                    far[part] += int(pred[q+1] >= 0 and target[q] and not target[q+1])
            families = {}
            for family in sorted({r['family'] for r in selected}):
                member = [i for i,r in enumerate(selected) if r['family'] == family]
                families[family] = scalar_metrics(z[member], truth[member])
            cohort[arm] = dict(metrics=scalar_metrics(z, truth), groups=grouped(z, truth, selected, 'group'),
                sites=grouped(z, truth, selected, 'site'), wrong_far_near_only=far,
                wrong_near_far_only=near, families=families)
        tp, fp, lost = [0]*4, [0]*4, 0
        for i in ids:
            for q in range(4):
                before, after = a['MZ5'][i,q] >= 0, a['MZ11'][i,q] >= 0
                tp[q] += int(after and not before and a['truth'][i,q])
                fp[q] += int(after and not before and not a['truth'][i,q])
                lost += int(before and not after)
        cohort['changes'] = dict(added_tp=tp, added_fp=fp, lost_positive_bits=lost)
        cohort['observation'] = dict(no_packet=sum(bool(a['no_packet'][i]) for i in ids),
            unknown_pixel_total=sum(int(a['unknown_pixels'][i]) for i in ids))
        assert cohort == result[dataset], dataset
        audited[dataset] = cohort['changes']
    favorable = all(not any(c['added_fp']) and not c['lost_positive_bits'] for c in audited.values()) and sum(sum(c['added_tp'][1::2]) for c in audited.values()) > 0
    assert favorable == result['favorable_transfer']
    parity = read(run/'parity.json')
    assert parity['status'] == 'PASS' and parity['frames'] == 16
    assert parity['visual_max_error'] < 1e-4 and parity['range_max_error_m'] < 1e-6
    write(run/'audit.json', dict(status='PASS', backend='CPU_SCALAR_NO_FIT',
        metadata_rows=3000, composition_bits=12000, arm_prediction_bits=48000,
        baseline_positive_bits_preserved=preserved, max_scalar_gate_score_error=max_score_error,
        changes=audited, favorable_transfer=favorable, integration_parity=parity,
        boundary='Integration parity is16 consumed oldTRAIN frames; selected3000 are Development; no extraction repeated.',
        source_sha256=sha(Path(__file__)), receipt_sha256=sha(run/'receipt.json')))
    print('PASS', dict(changes=audited, favorable_transfer=favorable, max_score_error=max_score_error), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['root', 'inventory', 'run']:
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    main(args.root, args.inventory, args.run)

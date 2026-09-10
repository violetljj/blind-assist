"""Scalar row recount and receipt verification for the no-fit diagnostic."""
import argparse
from pathlib import Path
import numpy as np
from mz5_ensemble_readout import read, write, sha


def main(root, probe):
    receipt, result, admission, start = [read(probe / n) for n in
        ['receipt.json', 'result.json', 'admission.json', 'start.json']]
    for n, h in receipt['outputs'].items(): assert sha(probe / n) == h
    for n, h in receipt['code_sha256'].items(): assert sha(Path(__file__).with_name(n)) == h
    protocol = Path(__file__).with_name('MZ17_WITNESS_OBJECTIVE_PROTOCOL_20260910.md')
    assert sha(protocol) == start['inputs'][str(protocol)]
    batches = np.load(probe / 'batches.npy')
    original = np.load(root / 'artifacts.local/work/mz15-shared-support-20260910/run-v1/batches.npy')
    np.testing.assert_array_equal(batches, original[:64])
    rows = read(probe / 'rows.json')
    assert len(rows) == batches.size * 4 == 4096
    names = ['BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR']
    for i, row in enumerate(rows):
        assert row['frame_id'] == batches.flatten()[i // 4]
        assert row['output'] == names[i % 4] and row['draw_index'] == i // 4
        wrong = [w for w in row['winners'] if w['known'] and not w['query_positive']]
        assert len(wrong) == row['known_incorrect_winners']
        for w in row['winners']:
            np.testing.assert_allclose(w['total_gradient'], w['local_gradient'] + w['weighted_query_gradient'], rtol=2e-6, atol=1e-10)
        for key, gradient, comparison in [('query_up_wrong', 'weighted_query_gradient', -1),
            ('local_down_wrong', 'local_gradient', 1), ('total_up_wrong', 'total_gradient', -1)]:
            assert row[key] == sum(comparison * w[gradient] > 0 for w in wrong)
    unique = {}
    for name in ['ALL', *names]:
        subset = rows if name == 'ALL' else [r for r in rows if r['output'] == name]
        positive = [r for r in subset if r['truth'] and r['supervised']]
        bad = [r for r in positive if r['known_incorrect_winners']]
        s = result['summary'] if name == 'ALL' else result['by_output'][name]
        assert s['gt_positive_supervised'] == len(positive)
        assert s['known_incorrect_winner_queries'] == len(bad)
        assert s['total_raises_wrong_queries'] == sum(r['total_up_wrong'] > 0 for r in bad)
        assert s['unknown_winner_queries'] == sum(r['unknown_winners'] > 0 for r in positive)
        assert s['tied_winner_queries'] == sum(r['tie_count'] > 1 for r in positive)
        unique[name] = dict(positive_frame_queries=len({(r['frame_id'], r['output']) for r in positive}),
                           wrong_frame_queries=len({(r['frame_id'], r['output']) for r in bad}))
    s = result['summary']
    proceed = s['known_incorrect_winner_queries'] / s['gt_positive_supervised'] >= .10 and s['total_raises_wrong_queries'] / s['known_incorrect_winner_queries'] >= .50
    assert proceed == admission['proceed'] == result['admission']['passed']
    assert receipt['training_steps'] == 0 and receipt['optimizer_created'] is False
    assert receipt['parameter_and_buffer_unchanged'] and receipt['checkpoint_unchanged']
    assert not (probe.parent / 'run-v1').exists()
    write(probe / 'audit.json', dict(status='PASS', row_count=len(rows), draws=int(batches.size),
        unique_frames=int(np.unique(batches).size), unique_frame_queries=unique, admission=proceed,
        training='not launched', checks='Hashes, batch identity, scalar row gradients and counts; repeated draws disclosed'))
    print('PASS', len(rows), 'rows; admission', proceed, unique)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--probe', type=Path, required=True)
    args = p.parse_args(); main(args.root, args.probe)

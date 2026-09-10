"""Stored-output scalar audit of the two matched local support fits."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import read, write, sha, load_npz
from mz11_audit import scalar_metrics


def main(root, run):
    receipt, start, result = (read(run/f) for f in ['receipt.json', 'start.json', 'result.json'])
    for name, digest in receipt['outputs'].items(): assert sha(run/name) == digest
    for name, digest in start['code_sha256'].items():
        path=run/'mz15_train.original.py' if name=='mz15_train.py' else Path(__file__).with_name(name)
        assert sha(path) == digest
    repaired=run/'stress-repair-v1'
    repair=read(repaired/'receipt.json')
    for name,digest in repair['outputs'].items():assert sha(repaired/name)==digest
    assert sha(Path(__file__).with_name('mz15_train.py'))==repair['corrected_train_source_sha256']
    result=read(repaired/'result.json')
    assert sha(Path(__file__).with_name('MZ15_SHARED_SUPPORT_PROTOCOL_20260910.md')) == start['protocol_sha256']
    batches = np.load(run/'batches.npy')
    assert sha(run/'batches.npy') == start['batches_sha256'] and batches.shape == (1200, 16)
    work = root/'artifacts.local/work'
    old = load_npz(work/'mz8-attribution-20260910/cache-v5/observations.npz')
    assert (old['role'][batches[:, :4]] == 'TRAIN_ONLY').all()
    for sl, lo, hi in [(slice(4, 8), 3700, 8700), (slice(8, 12), 8700, 11200), (slice(12, 16), 3500, 3700)]:
        assert ((batches[:, sl] >= lo) & (batches[:, sl] < hi)).all()
    init1, init2 = (torch.load(run/(arm+'-initial.pt'), weights_only=True) for arm in ['SHARED', 'QUERY'])
    for key in init1:
        if key.startswith('head.2.'):
            torch.testing.assert_close(init2[key], init1[key].expand_as(init2[key]), rtol=0, atol=0)
        else:
            torch.testing.assert_close(init2[key], init1[key], rtol=0, atol=0)
    a = load_npz(repaired/'predictions.npz')
    old_arrays=load_npz(run/'predictions.npz')
    for key in a:
        if '/stress' in key and key.split('/')[-1] in ['winning_source','winning_query','best_true']:continue
        np.testing.assert_array_equal(a[key],old_arrays[key])
    rows = read(work/'mz15-shared-support-20260910/cache-v1/selected.json')
    assert not {r['site'] for r in rows[:7500]} & {r['site'] for r in rows[7500:]}
    trace = load_npz(work/'mz14-evidence-trace-20260910/run-v2/traces.npz')
    bits = 0
    for arm in ['SHARED', 'QUERY']:
        cutoff = np.load(run/(arm+'-cutoff.npy'))
        for q in range(4):
            b, t, z, s = (a[arm+'/DEV/'+k] for k in ['baseline', 'truth', 'raw', 'support'])
            e = (b[:, q] < 0) & s[:, q]; neg = e & ~t[:, q]
            expected = np.nextafter(np.float64(z[neg, q].max()), np.inf) if neg.any() else (-np.inf if e.any() else np.inf)
            assert cutoff[q] == expected
        for name, reported in result['metrics'][arm].items():
            v = {k.split('/')[-1]: val for k, val in a.items() if k.startswith(arm+'/'+name+'/')}
            for i in range(len(v['truth'])):
                for q in range(4):
                    margin = float(v['raw'][i, q])-cutoff[q] if v['support'][i, q] else -1e6
                    accept = v['baseline'][i, q] < 0 and v['support'][i, q] and margin >= 0
                    assert accept == v['accepted'][i, q]
                    assert margin == v['margin'][i, q]
                    assert v['candidate'][i, q] == (margin if accept else v['baseline'][i, q])
                    bits += 1
            assert scalar_metrics(v['baseline'], v['truth']) == reported['baseline']
            assert scalar_metrics(v['candidate'], v['truth']) == reported['candidate']
            assert scalar_metrics(v['margin'], v['truth']) == reported['standalone']
            assert (v['accepted'] & v['truth']).sum(0).tolist() == reported['added_tp']
            assert (v['accepted'] & ~v['truth']).sum(0).tolist() == reported['added_fp']
            if name in result['thin'][arm]:
                assert scalar_metrics(v['candidate'][100:125], v['truth'][100:125]) == result['thin'][arm][name]
        for name in ['relation10000', 'distance5000']:
            r = [r for r in rows if r['dataset'] == name]
            z, b, t = (a[arm+'/'+name+'/'+k] for k in ['candidate', 'baseline', 'truth'])
            for field in ['site', 'group', 'family']:
                units = {}
                for i, row in enumerate(r): units.setdefault(row[field], []).append(i)
                for unit, ii in units.items():
                    assert result['groups'][arm][name][field][unit] == dict(frames=len(ii), baseline=scalar_metrics(b[ii], t[ii]), candidate=scalar_metrics(z[ii], t[ii]))
            added = (trace[name+'/previous'] >= 0) & (trace[name+'/baseline'] < 0) & trace[name+'/truth']
            lost = added & (trace[name+'/oracle_source'] < 0)
            assert (lost & (z >= 0)).sum(0).tolist() == result['six_oracle_lost'][arm][name]['recovered']
        assert result['fits'][arm]['steps'] == 1200
        m, th = result['metrics'][arm], result['thin'][arm]
        names = ['DEV', 'clean', 'stress', 'relation10000', 'distance5000']
        expected = dict(no_added_fp=all(sum(m[k]['added_fp']) == 0 for k in names),
            placement_far_gain=all(sum(m[k]['added_tp'][1::2]) > 0 for k in ['relation10000', 'distance5000']),
            thin_clean=sum(th['clean']['tp'][1::2]) >= 48, thin_stress=sum(th['stress']['tp'][1::2]) >= 48,
            baseline_near_retained=all(all(m[k]['candidate']['tp'][q] >= m[k]['baseline']['tp'][q] for q in [0, 2]) for k in names))
        expected['useful_effect'] = all(expected.values())
        expected['wrong_correspondence_reduces_far_gain'] = sum(sum(m[k]['added_tp'][1::2]) for k in ['clean', 'relation10000', 'distance5000']) > sum(sum(m[k+'_wrong']['added_tp'][1::2]) for k in ['clean', 'relation10000', 'distance5000'])
        assert expected == result['gates'][arm]
    write(run/'audit.json', dict(status='PASS', scalar_bits=bits, matched_initialization='PASS',
        training_partition_and_batch_identity='PASS', old_dev_cutoffs='PASS',
        compose_and_baseline_positive_retention='PASS', family_group_site='PASS', gates=result['gates'],
        scope='Stored-array check; consumed Development, not independent confirmation'))
    print('PASS', bits, result['gates'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(); main(a.root, a.run)

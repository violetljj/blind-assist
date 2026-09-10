"""Independent scalar checks of frozen trace partitions and diagnostic masks."""
import argparse
from pathlib import Path
import numpy as np
from mz5_ensemble_readout import read, write, sha, load_npz
from mz11_audit import scalar_metrics


def main(run):
    receipt = read(run/'receipt.json')
    for name, digest in receipt['outputs'].items():
        assert sha(run/name) == digest
    assert sha(Path(__file__).with_name('mz14_evidence_trace.py')) == receipt['source_sha256']
    result, a = read(run/'result.json'), load_npz(run/'traces.npz')
    rows = read(run/'selected.json')
    bits, added_false, intervention = 0, [], {}
    for name in ['DEV', 'clean', 'relation10000', 'distance5000']:
        v = {k.split('/', 1)[1]: value for k, value in a.items() if k.startswith(name+'/')}
        r = result['cohorts'][name]
        for arm, key in [('source', 'source'), ('oracle_source', 'evaluator_oracle_source'), ('oracle_query', 'evaluator_oracle_query')]:
            assert scalar_metrics(v[arm], v['truth']) == r[key]
        assert np.all(v['oracle_query'] <= v['oracle_source'])
        assert np.all(v['oracle_source'] <= v['source'])
        fn_parts = np.zeros((3, 4), dtype=int)
        fp_parts = np.zeros((3, 4), dtype=int)
        for i, truth in enumerate(v['truth']):
            for q in range(4):
                bits += 1
                selected = int(v['selected_query_pixels'][i, q])
                eligible = int(v['eligible_query_pixels'][i, q])
                count = int(v['correct_candidate_count'][i, q])
                assert 0 <= eligible <= selected and (count > 0) == (eligible > 0)
                win_source = int(v['winning_source_pixels'][i, q])
                win_query = int(v['winning_query_pixels'][i, q])
                assert 0 <= win_query <= win_source
                if not v['available'][i, q]:
                    assert v['source'][i, q] == -1e6
                if truth[q] and v['source'][i, q] < 0:
                    category = 0 if selected == 0 else (1 if count == 0 else 2)
                    fn_parts[category, q] += 1
                if v['previous'][i, q] >= 0 and v['baseline'][i, q] < 0 and not truth[q]:
                    category = 0 if win_source == 0 else (1 if win_query == 0 else 2)
                    fp_parts[category, q] += 1
                    added_false.append((name, i, q))
        for part, key in zip(fn_parts, ['source_fn_no_selected_pixels', 'source_fn_selected_but_no_correct_candidate', 'source_fn_correct_candidate_below_threshold']):
            assert part.tolist() == r[key]
        for part, key in zip(fp_parts, ['added_fp_winner_no_source', 'added_fp_winner_source_outside_query', 'added_fp_winner_query_under_truth_pixel_budget']):
            assert part.tolist() == r[key]
        assert fn_parts.sum(0).tolist() == ((v['source'] < 0) & v['truth']).sum(0).tolist()
        assert fp_parts.sum(0).tolist() == r['added_fp']
        assert result['parity'][name]['sign_mismatches'] == []
        assert len([r for r in rows if r['cohort'] == name]) == len(v['truth'])
        wrong_winner = v['truth'] & (v['source'] >= 0) & (v['winning_query_pixels'] == 0)
        added_tp = (v['previous'] >= 0) & (v['baseline'] < 0) & v['truth']
        intervention[name] = dict(
            source_tp_lost=((v['source'] >= 0) & v['truth'] & (v['oracle_source'] < 0)).sum(0).tolist(),
            added_tp_lost=(added_tp & (v['oracle_source'] < 0)).sum(0).tolist(),
            tp_wrong_winner_strictly_above_correct=(wrong_winner & (v['source'] > v['oracle_query'])).sum(0).tolist(),
            tp_wrong_winner_tied_with_correct=(wrong_winner & (v['source'] == v['oracle_query'])).sum(0).tolist())
    cases = read(run/'cases.json')
    assert set(added_false) <= {(r['cohort'], r['row'], r['q']) for r in cases}
    added_rows = [r for r in cases if (r['cohort'], r['row'], r['q']) in set(added_false)]
    families = {}
    for r in added_rows:
        family = r.get('family', 'sequence_or_old_dev')
        f = families.setdefault(family, dict(bits=0, no_source=0, source_outside_query=0,
            query_under_budget=0, survives_source_oracle=0, survives_query_oracle=0, groups=set(), sites=set()))
        f['bits'] += 1
        f['no_source'] += r['winning_source_pixels'] == 0
        f['source_outside_query'] += r['winning_source_pixels'] > 0 and r['winning_query_pixels'] == 0
        f['query_under_budget'] += r['winning_query_pixels'] > 0
        f['survives_source_oracle'] += r['oracle_source'] >= 0
        f['survives_query_oracle'] += r['oracle_query'] >= 0
        f['groups'].add(r.get('group', (r['cohort'], r['row'])))
        f['sites'].add(r.get('site', r['cohort']))
    for f in families.values():
        f['groups'], f['sites'] = len(f['groups']), len(f['sites'])
    write(run/'audit.json', dict(status='PASS', scalar_bits=bits,
        added_false_bits=len(added_false), families=families, intervention=intervention,
        checks=['output hashes', 'frozen inference sign parity', 'scalar metrics',
            'oracle monotonicity', 'coverage hierarchy', 'exhaustive FP/FN partitions', 'case coverage'],
        limitation='Native masks are evaluator-only. Oracle query presence uses true membership, not an achievable model.'))
    print('PASS', bits, families)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    main(p.parse_args().run)

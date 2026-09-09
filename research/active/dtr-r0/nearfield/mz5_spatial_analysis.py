"""Paired scoring of completed MZ5 predictions; never fits or changes a model."""
import argparse
from pathlib import Path
import numpy as np

from mz3_error_attribution import read, load, sha, write, metrics, compare

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'


def run(output):
    output = Path(output).resolve()
    if (output/'analysis.json').exists():
        raise FileExistsError('Preserve completed analysis')
    rec = read(output/'receipt.json')
    assert rec['status'] == 'PASS' and rec['predictions_sha256'] == sha(output/'predictions.npz')
    assert rec['metrics_sha256'] == sha(output/'metrics.json')
    data, original_scores = load(output/'predictions.npz'), read(output/'metrics.json')
    old_path = WORK/'mz1-tiny-fusion-20260910/run-v1'
    old = load(old_path/'predictions.npz')
    assert sha(old_path/'predictions.npz') == read(old_path/'receipt.json')['predictions_sha256']
    for key in ('role', 'truth', 'original_alerts', 'joint'):
        np.testing.assert_array_equal(data[key], old[key])
    for arm, scores in original_scores.items():
        for role, expected in scores.items():
            mask = data['role'] == role
            assert metrics(data[arm+'_flags'][mask], data['truth'][mask]) == expected
    ev = data['role'] == 'EVAL_ONLY'
    truth = data['truth'][ev]
    assert len(truth) == 1500
    index = WORK/'body-query-5000-20260909/dataset-v1/index.json'
    assert sha(index) == rec['source_index_sha256']
    rows = [r for r in read(index)['frames'] if r['source_role']=='EVAL_ONLY']
    ids = np.array([r['frame_id'] for r in rows])
    expected = np.array([np.array(r['counts']).reshape(4, 3).sum(1)>=3 for r in rows])
    np.testing.assert_array_equal(truth, expected)
    ensemble_dir = WORK/'mz5-fixed-ensemble-20260910/run-v1'
    er = read(ensemble_dir/'receipt.json')
    assert sha(ensemble_dir/'predictions.npz') == er['output_sha256']['predictions.npz']
    ensemble = load(ensemble_dir/'predictions.npz')
    for key, expected in (('truth', truth), ('frame_id', ids), ('original_alerts', data['original_alerts'][ev])):
        np.testing.assert_array_equal(ensemble[key], expected)
    predictions = {a: data[a+'_flags'][ev] for a in original_scores}
    predictions.update(LOCAL_PERMUTED=data['LOCAL_PERMUTED_flags'],
        MZ1_FUSION=old['FUSION_flags'][ev], MZ1_TOF=old['TOF_ONLY_flags'][ev],
        MZ1_RGB=old['RGB_ONLY_flags'][ev], ENSEMBLE=ensemble['ENSEMBLE_flags'], MZ0=ensemble['MZ0_flags'])
    scores = {name: metrics(flags, truth) for name, flags in predictions.items()}
    conditions = np.array([r['condition'] for r in rows])
    strata = {name: np.array([r[name] for r in rows]) for name in ('condition', 'region_id', 'family')}
    strata['condition_range'] = np.array([r['condition']+'/'+r['declared_range'] for r in rows])
    strata['negative'] = np.where(~truth.any(1), 'NEGATIVE', 'POSITIVE')
    comparisons = {}
    pairs = [('LOCAL_FUSION', b) for b in ('POOLED_FUSION', 'LOCAL_RGB_ONLY', 'ENSEMBLE', 'MZ1_FUSION', 'MZ0')]
    pairs += [('LOCAL_PERMUTED', 'LOCAL_FUSION')]
    for candidate, base in pairs:
        comparisons[candidate+'_vs_'+base] = dict(
            overall=compare(predictions[candidate], predictions[base], truth, ids),
            strata={name: {value: compare(predictions[candidate][mask], predictions[base][mask], truth[mask], ids[mask])
                for value in sorted(set(values)) for mask in [values == value]} for name, values in strata.items()})
    strata_scores = {name: {value: {arm: metrics(flags[mask], truth[mask]) for arm, flags in predictions.items()}
        for value in sorted(set(values)) for mask in [values==value]} for name, values in strata.items()}
    cross = lambda s: s['body_to_head']['numerator']+s['head_to_body']['numerator']
    local, pooled, rgb = [scores[a] for a in ('LOCAL_FUSION', 'POOLED_FUSION', 'LOCAL_RGB_ONLY')]
    head_near = strata_scores['condition_range']['HEAD_ONLY/near']
    criterion = dict(exact_above_pooled=local['spatial_exact']['numerator']>pooled['spatial_exact']['numerator'],
        exact_above_rgb=local['spatial_exact']['numerator']>rgb['spatial_exact']['numerator'],
        head_only_near_above_pooled=head_near['LOCAL_FUSION']['spatial_exact']['numerator']>head_near['POOLED_FUSION']['spatial_exact']['numerator'],
        wrong_far_not_worse_than_pooled=local['wrong_far']['numerator']<=pooled['wrong_far']['numerator'],
        cross_body_not_worse_than_pooled=cross(local)<=cross(pooled))
    criterion['all'] = all(criterion.values())
    write(output/'analysis.json', dict(status='PASS', scores=scores, comparisons=comparisons,
        strata_scores=strata_scores, predeclared_necessary_criterion=criterion,
        prediction_sha256=sha(output/'predictions.npz'), original_alert_parity=5000,
        script_sha256=sha(__file__), source_index_sha256=sha(index),
        placement_reason='TASK_NOT_GPU_SUITABLE', scope='Consumed controlled Development; descriptive paired comparison'))
    print({name: dict(exact=s['spatial_exact']['numerator'], wrong_far=s['wrong_far']['numerator'], cross=cross(s),
        head_only_near=head_near[name]['spatial_exact']['numerator']) for name, s in scores.items()})
    print('criterion', criterion)
    print({name: {k: v['overall'][k] for k in ('gained', 'lost')} for name, v in comparisons.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run(parser.parse_args().run)

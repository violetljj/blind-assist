"""Conditional, unit-resampled contrasts for the frozen raw-radial probe.

No fitting or threshold selection. These intervals describe consumed Development
predictions with the fitted models and thresholds held fixed.
"""
import json
import numpy as np
import cnh_direction_information_dev as I

OUT = I.R.WORK / 'cnh-raw-radial-information-dev-20261007'
CONTRASTS = [('current', 'pair_only'), ('current_support', 'current'),
             ('temporal', 'current_support'), ('temporal', 'pair_only'),
             ('current', 'original_center'), ('temporal', 'original_center')]


def main():
    with np.load(OUT / 'ledger.npz') as data:
        ledger = dict(data)
    uid = ledger['unit']; contact = ledger['contact']; control = ledger['control']
    units, inverse = np.unique(uid, return_inverse=True)
    source_rows = I.R.read(I.H.R3 / 'rows.json')
    strata = {r['unit']: (r['batch'], r['mode'], r['turn'])
              for r in source_rows if r['unit'] in set(units.tolist())}
    rng = np.random.default_rng(2026100721)
    weights = np.zeros((1000, len(units)), dtype=np.int32)
    for stratum in sorted(set(strata.values())):
        ids = np.array([j for j, u in enumerate(units) if strata[u] == stratum])
        for b, draw in enumerate(rng.integers(0, len(ids), (1000, len(ids)))):
            weights[b, ids] = np.bincount(draw, minlength=len(ids))
    event_counts = np.bincount(inverse, weights=contact, minlength=len(units))
    control_counts = np.bincount(inverse, weights=control * 10, minlength=len(units))
    output = {}
    for mode in ('calibrated', 'matched_eval_descriptive'):
        for cap in ('.025', '.050'):
            for candidate, baseline in CONTRASTS:
                prefix = f'{mode}/0{cap}/'
                a = ledger['alarm/' + prefix + candidate]
                b = ledger['alarm/' + prefix + baseline]
                ta = ledger['timely_excluding_warmup/' + prefix + candidate]
                tb = ledger['timely_excluding_warmup/' + prefix + baseline]
                d = (ta.astype(int) - tb.astype(int)) * contact
                fa = (a[:, 2:12].sum(1) - b[:, 2:12].sum(1)) * control
                du = np.bincount(inverse, weights=d, minlength=len(units))
                fu = np.bincount(inverse, weights=fa, minlength=len(units))
                boot_d = weights @ du; boot_f = weights @ fu
                output[prefix + candidate + '-' + baseline] = dict(
                    timely_gain=int(d.sum()), rescued=int((d > 0).sum()),
                    lost=int((d < 0).sum()), events=int(contact.sum()),
                    timely_gain_ci95=np.quantile(boot_d, [.025, .975]).tolist(),
                    timely_rate_gain_pp_ci95=np.quantile(
                        100 * boot_d / (weights @ event_counts), [.025, .975]).tolist(),
                    false_alarm_interval_delta=int(fa.sum()),
                    control_intervals=int(control.sum()) * 10,
                    false_alarm_rate_delta_pp_ci95=np.quantile(
                        100 * boot_f / (weights @ control_counts), [.025, .975]).tolist(),
                    fold_timely_gain={str(f): int(d[ledger['fold'] == f].sum())
                                      for f in range(3)})
    result = dict(
        scope='Consumed Development, paired complete-unit stratified bootstrap; '
              'models and thresholds fixed, not refitted, no selection correction',
        repetitions=1000, seed=2026100721,
        source_ledger_sha256=I.R.sha(OUT / 'ledger.npz'),
        contrasts=output)
    with (OUT / 'contrasts.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    for key, value in output.items():
        if key.startswith('calibrated/0.025/'):
            print(key, value)


if __name__ == '__main__':
    main()

"""Paired conditional contrasts, no model fitting or threshold selection."""
import json
import numpy as np
import cnh_direction_information_dev as I

OUT = I.R.WORK / 'cnh-training-support-dev-20261007'
PAIRS = [('source_repeat', 'pair_only'), ('hb_aug', 'source_repeat'),
         ('hb_aug', 'pair_only'), ('hb_aug', 'original_center')]


def bootstrap_weights(unit, strata, seed):
    units, inverse = np.unique(unit, return_inverse=True)
    rng = np.random.default_rng(seed)
    weights = np.zeros((1000, len(units)), np.int32)
    for group in sorted(set(strata.values())):
        ids = np.array([j for j, u in enumerate(units) if strata[int(u)] == group])
        for b, draw in enumerate(rng.integers(0, len(ids), (1000, len(ids)))):
            weights[b, ids] = np.bincount(draw, minlength=len(ids))
    return units, inverse, weights


def compare(ledger, units, inverse, weights, mask, akey, bkey):
    event = ledger['contact'] & mask
    control = ledger['control'] & mask
    a = ledger['timely_excluding_warmup/' + akey]
    b = ledger['timely_excluding_warmup/' + bkey]
    delta = (a.astype(int) - b.astype(int)) * event
    fa_delta = (ledger['alarm/' + akey][:, 2:12].sum(1) -
                ledger['alarm/' + bkey][:, 2:12].sum(1)) * control
    event_mass = np.bincount(inverse, weights=event, minlength=len(units))
    control_mass = np.bincount(inverse, weights=control * 10, minlength=len(units))
    event_diff = np.bincount(inverse, weights=delta, minlength=len(units))
    control_diff = np.bincount(inverse, weights=fa_delta, minlength=len(units))
    ed, cd = weights @ event_mass, weights @ control_mass
    be, bc = weights @ event_diff, weights @ control_diff
    # Missing event/control bootstrap draws cannot be called zero-risk draws.
    valid_event, valid_control = ed > 0, cd > 0
    return dict(events=int(event.sum()), controls=int(control.sum()),
        timely_gain=int(delta.sum()), rescued=int((delta > 0).sum()),
        lost=int((delta < 0).sum()),
        timely_gain_ci95=np.quantile(be, [.025, .975]).tolist(),
        timely_gain_pp_ci95=np.quantile(100 * be[valid_event] / ed[valid_event], [.025, .975]).tolist(),
        false_alarm_interval_delta=int(fa_delta.sum()),
        false_alarm_delta_pp_ci95=np.quantile(100 * bc[valid_control] / cd[valid_control], [.025, .975]).tolist(),
        missing_event_draws=int((~valid_event).sum()), missing_control_draws=int((~valid_control).sum()),
        fold_timely_gain={str(f): int(delta[ledger['fold'] == f].sum()) for f in range(3)})


def main():
    result = dict(scope='Consumed Development conditional paired unit bootstrap; '
                       'models/thresholds fixed, no refit or selection correction. '
                       'HB structure participates in training on different units.',
                  draws=1000, results={}, source_sha256={})
    rows = I.R.read(I.H.R3 / 'rows.json')
    original_strata = {int(r['unit']): (r['batch'], r['mode'], r['turn']) for r in rows}
    for domain, filename in [('original', 'ledger.npz'), ('hb', 'hb_ledger.npz')]:
        with np.load(OUT / filename) as source:
            z = dict(source)
        uid = z['unit']
        if domain == 'original':
            strata = {int(u): original_strata[int(u)] for u in np.unique(uid)}
            groups = {'all': np.ones(len(uid), bool)}
        else:
            strata = {}
            for u in np.unique(uid):
                folds = np.unique(z['fold'][uid == u]); assert len(folds) == 1
                strata[int(u)] = int(folds[0])
            groups = {tag: z['tag'] == tag for tag in ('H', 'B', 'HB')}
        units, inverse, weights = bootstrap_weights(uid, strata, 2026100722)
        result['source_sha256'][filename] = I.R.sha(OUT / filename)
        for mode in ('calibrated', 'matched_eval_descriptive'):
            for cap in (.025, .05):
                prefix = f'{mode}/{cap:.3f}/'
                for candidate, baseline in PAIRS:
                    akey, bkey = prefix + candidate, prefix + baseline
                    if 'alarm/' + akey not in z:
                        continue
                    for group, mask in groups.items():
                        key = f'{domain}/{group}/{prefix}{candidate}-{baseline}'
                        result['results'][key] = compare(z, units, inverse, weights, mask, akey, bkey)
    with (OUT / 'contrasts.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    for key, value in result['results'].items():
        if '/calibrated/0.025/hb_aug-source_repeat' in key:
            print(key, value)


if __name__ == '__main__':
    main()

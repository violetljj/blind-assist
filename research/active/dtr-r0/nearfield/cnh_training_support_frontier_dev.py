"""Posthoc HB-control threshold diagnostic; never a deployable calibration."""
import json
import time
import numpy as np
import cnh_training_support_dev as M


def save(name, data):
    with (M.OUT / name).open('x', encoding='utf8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    start = time.monotonic()
    save('HB_FRONTIER_PLAN.json', dict(
        scope='Posthoc diagnostic using consumed HB evaluation controls to choose '
              'per-fold thresholds; not train/cal/test performance or a new candidate',
        question='Does HB augmentation retain a timely-detection advantage '
                 'when both models use the same HB-only false-alarm cap?',
        caps=[.025, .05], methods=list(M.METHODS),
        threshold='Lowest threshold satisfying each fold HB-control cap '
                  'over output intervals2:12; ties retained, >= alarms',
        budget_cpu_wall_seconds=30, no_fitting=True,
        source_sha256=M.R.sha(M.OUT / 'hb_ledger.npz'),
        source_code_sha256=M.R.sha(__file__)))
    with np.load(M.OUT / 'hb_ledger.npz') as data:
        z = dict(data)
    contact, control, deadline, tag, fold = [z[k] for k in
                                            ('contact', 'control', 'deadline', 'tag', 'fold')]
    results = {}; thresholds = []; outputs = {}
    for cap in (.025, .05):
        for name in M.METHODS:
            score = z['score/' + name]; alarm = np.zeros(score.shape, bool)
            for f in range(3):
                mask = fold == f
                fitmask = mask & control & (tag == 'HB')
                theta = M.P.threshold_at_cap(score[fitmask, 2:12], cap)
                alarm[mask] = score[mask] >= theta
                assert alarm[fitmask, 2:12].mean() <= cap + 1e-12
                thresholds.append(dict(cap=cap, method=name, fold=f, threshold=float(theta),
                    fit_control_intervals=int(fitmask.sum()) * 10,
                    fit_false_alarm_intervals=int(alarm[fitmask, 2:12].sum())))
            key = f'{cap:.3f}/{name}'
            results[key] = {t: M.Q.metrics(alarm, contact, control, deadline, tag == t)
                            for t in ('H', 'B', 'HB')}
            timely, post = M.Q.event_states(alarm, contact, deadline)
            results[key]['paired'] = M.paired_structure_counts(post, contact, tag)
            outputs['alarm/' + key] = alarm
            outputs['timely_excluding_warmup/' + key] = post
            print(key, results[key]['HB'])
    elapsed = time.monotonic() - start
    if elapsed > 30:
        raise TimeoutError('30s diagnostic budget exceeded')
    np.savez_compressed(M.OUT / 'hb_frontier_ledger.npz', **outputs)
    save('hb_frontier_descriptive.json', dict(scope='HB evaluation-control fitted '
        'thresholds; descriptive only, no deployable calibration claim',
        seconds=elapsed, thresholds=thresholds, metrics=results))


if __name__ == '__main__':
    main()

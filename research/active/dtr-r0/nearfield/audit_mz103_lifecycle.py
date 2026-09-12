"""Post-outcome attribution of frozen MZ103 native results; no policy changes."""
from pathlib import Path
import json
import numpy as np
import mz101_spatial as m
from run_mz101_spatial import metrics, segments, sha
from run_mz103_depth_frontend import TASK, write


def audit():
    result = {'authority': 'POSTHOC_CONSUMED_DIAGNOSTIC_NOT_CANDIDATE_SELECTION', 'panels': {}}
    for panel in ('mz101', 'mz102'):
        root = TASK / 'native-v1' / panel
        data = np.load(root / 'predictions.npz')
        gt = np.load(root / 'truth.npy')
        obs = json.loads((root / 'observations.json').read_text())
        ids = np.array([o['episode'] for o in obs])
        raw = data['replacement_union_support'] > 0
        native, old = data['replacement_union'], data['sgbm_union']
        # Exact Boolean equivalence is stronger than aggregate scores: all raw
        # native queries match truth; its final errors equal truth passed through
        # the unchanged lifecycle. This is not a new runnable candidate policy.
        np.testing.assert_array_equal(raw, gt)
        np.testing.assert_array_equal(m.hysteresis(gt, ids), native)
        entries, exits = set(), set()
        for episode in dict.fromkeys(ids):
            ix = np.flatnonzero(ids == episode)
            for part in range(2):
                for start, end in segments(gt[ix, part]):
                    entries.add((int(ix[start]), part))
                    if end < len(ix):
                        exits.add((int(ix[end]), part))
        misses = {tuple(map(int, x)) for x in np.argwhere(~native & gt)}
        false = {tuple(map(int, x)) for x in np.argwhere(native & ~gt)}
        assert misses == entries
        assert false == exits
        lost = []
        for i, j in np.argwhere(old & ~native & gt):
            assert (int(i), int(j)) in entries
            previous_same_episode = i > 0 and ids[i-1] == ids[i]
            previous_false_support = bool(previous_same_episode and
                data['sgbm_union_support'][i-1, j] > 0 and not gt[i-1, j])
            prior_indices = [k for k in range(int(i)) if ids[k] == ids[i]]
            prior_false = [k for k in prior_indices if
                           data['sgbm_union_support'][k,j] > 0 and not gt[k,j]]
            assert prior_false
            carried_false = bool(not previous_false_support and previous_same_episode
                                 and old[i-1,j] and not gt[i-1,j])
            assert previous_false_support or carried_false
            lost.append(dict(frame=obs[i]['id'], part=m.PARTS[j],
                earlier_false_support_frames=[obs[k]['id'] for k in prior_false],
                immediate_previous_false_support=previous_false_support,
                previously_activated_false_alert_carried=carried_false,
                current_native_support=True, native_confirmed_next_frame=bool(
                    i+1 < len(obs) and ids[i+1] == ids[i] and native[i+1, j])))
        row = dict(input_sha256=sha(root/'predictions.npz'),
            raw_native_equals_gt=True, final_native_equals_hysteresis_gt=True,
            fn_event_entry_count=len(misses), fp_first_exit_count=len(false),
            left_censored_entry_count=sum(obs[i]['time_s'] == 0 for i,j in entries),
            old_tp_lost_with_prior_false_support=lost,
            raw_scores={k: metrics(data[k+'_support'] > 0, gt, obs)
                        for k in ('tof', 'sgbm_union', 'replacement_union')})
        result['panels'][panel] = row
    write(TASK / 'lifecycle-attribution.json', result)
    print(json.dumps({k: {x: v[x] for x in ('raw_native_equals_gt',
          'final_native_equals_hysteresis_gt', 'fn_event_entry_count',
          'fp_first_exit_count', 'left_censored_entry_count')}
          for k,v in result['panels'].items()}, indent=2))


if __name__ == '__main__':
    audit()

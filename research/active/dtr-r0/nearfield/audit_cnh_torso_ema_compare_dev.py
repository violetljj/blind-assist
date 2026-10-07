"""Independent CPU audit; shares the comparison's cumulative analyze budget.

Recomputes saved evidence without rendering, GPU inference or parameter search.
"""
import time
import numpy as np
import cnh_torso_ema_compare_dev as E
import cnh_real_head_confirm as RC
import cnh_cvr_pilot as CP


def manual_smooth(raw):
    raw = np.asarray(raw, dtype=np.float64)
    result = np.zeros_like(raw)
    for t in range(13):
        length = min(t+1, 5)
        denominator = (1 << length) - 1
        for lag in range(length):
            result[..., t, :] += raw[..., t-lag, :] * (1 << (length-lag-1)) / denominator
    return result


def audit():
    tick = time.monotonic()
    with E.stage('analyze') as check:
        result = E.B.A.read(E.OUT/'result.json')
        assert result['status'] == 'COMPLETE'
        with np.load(E.OUT/'ledger.npz') as f:
            ledger = {k: f[k] for k in f.files}
        with np.load(E.Y.OUT/'ledger.npz') as f:
            parent = {k: f[k] for k in f.files}
        np.testing.assert_array_equal(ledger['score'][:, :, :4], parent['score'])
        for key in ('unit', 'config', 'contact', 'control', 'deadline'):
            np.testing.assert_array_equal(ledger[key], parent[key])
        assert ledger['score'].shape == (5, 2, 5, 3840, 13)
        assert int(ledger['contact'].sum()) == 229 and int(ledger['control'].sum()) == 384
        source = E.B.A.read(E.OUT/'PLAN.json')
        for name, expected in source['hashes'].items():
            check()
            # Source text corrections must have an explicit immutable correction receipt.
            actual = E.B.A.sha(E.B.A.ROOT/name)
            if actual != expected:
                correction = E.B.A.read(E.OUT/'input_record_correction.json')
                assert correction.get('source_hash_before') == expected
                assert correction.get('source_hash_after') == actual
        raw_count = 0
        query_count = 0
        max_smooth_delta = 0.
        for unit in E.Y.UNITS:
            check()
            path = E.OUT/'units'/f'unit{unit}.npz'
            assert E.B.A.sha(path) == result['provenance']['unit_sha256'][str(unit)]
            mask = ledger['unit'] == unit
            np.testing.assert_array_equal(ledger['config'][mask], np.arange(40))
            with np.load(E.Y.M.OUT/'units'/f'unit{unit}.npz') as old, np.load(E.Y.OUT/'units'/f'unit{unit}.npz') as phy, np.load(path) as ema:
                pelvis_origin = old['travel'][..., :3, 3]
                for ni, name in enumerate(E.Y.NAMES):
                    check()
                    prefix = '' if name == 'zero' else name+'/'
                    data = old if name == 'zero' else phy
                    sensor = data[prefix+'sensor']; noisy = data[prefix+'noisy']
                    q = ema[name+'/ema_query']; rel = ema[name+'/ema_rel']; heading = ema[name+'/ema_heading']
                    for c in range(40):
                        np.testing.assert_array_equal(rel[c], RC.ema_rel(noisy[c]))
                        expected_rotation = np.stack([RC.rel_query(r)[:3, :3] for r in rel[c]])
                        np.testing.assert_array_equal(q[c, :, :3, :3], expected_rotation)
                        expected_heading = RC.wrap(np.array([RC.A_yaw(n) for n in noisy[c]])-rel[c])
                        np.testing.assert_array_equal(heading[c], expected_heading)
                        world_rotation = np.stack([CP.rotation(float(h), 'y') for h in heading[c]])
                        recovered = sensor[c, :, :3, 3] - np.einsum('fij,fj->fi', world_rotation, q[c, :, :3, 3])
                        np.testing.assert_allclose(recovered, pelvis_origin[c], atol=1e-12, rtol=0)
                    query_count += 40*16
                    for ai, arm in enumerate(E.ARMS):
                        oldarm = 'corrected_gait' if name == 'zero' and arm == 'gait' else arm
                        raw = ema[name+'/ema_raw'] if arm == 'ema' else data[prefix+oldarm+'_raw']
                        smooth = manual_smooth(raw).max(-1)
                        scores = np.stack((smooth[0], smooth[1:].max(0)))
                        target = ledger['score'][ni, :, ai][:, mask]
                        delta = float(np.abs(scores-target).max())
                        max_smooth_delta = max(max_smooth_delta, delta)
                        np.testing.assert_allclose(scores, target, atol=1e-12, rtol=0)
                        raw_count += 1
        contact = ledger['contact']; control = ledger['control']; deadline = ledger['deadline']
        for ni, name in enumerate(E.Y.NAMES):
            for si, sensor in enumerate(E.A.SENSORS):
                check()
                key = name+'/'+sensor
                cell = result['metrics'][key]
                decisions = []
                for ai, arm in enumerate(E.ARMS):
                    th = cell[arm]['threshold']; theta = th['threshold']
                    values = ledger['score'][ni, si, ai]
                    alarm = values >= theta
                    actual = int(alarm[control].sum())
                    assert actual == th['actual_fa_count']
                    assert th['residual_fa_count'] == th['target_fa_count'] - actual
                    assert th['target_fa_count'] == 124
                    assert actual <= 124
                    assert int((values[control] >= np.nextafter(theta, -np.inf)).sum()) > 124
                    main = np.zeros(3840, bool); full = main.copy()
                    for row in np.flatnonzero(contact):
                        full[row] = bool(alarm[row, :int(deadline[row])+1].any())
                        main[row] = bool(alarm[row, 2:int(deadline[row])+1].any())
                    np.testing.assert_array_equal(main, ledger[key+'/timely_main'][ai])
                    np.testing.assert_array_equal(full, ledger[key+'/timely_all'][ai])
                    assert int(main.sum()) == cell[arm]['timely_main']
                    assert int(full.sum()) == cell[arm]['timely_all']
                    for cost, sl, denominator in [('startup', slice(0, 2), 768), ('main', slice(2, 12), 3840), ('all13', slice(None), 4992)]:
                        count = int(alarm[control, sl].sum())
                        assert count == cell[arm]['false_alarm'][cost]['count']
                        assert denominator == cell[arm]['false_alarm'][cost]['denominator']
                    decisions.append((main, full))
                for base, bi in [('ema', 4), ('e1', 1), ('torso', 2)]:
                    for field, ix in [('main', 0), ('all_before', 1)]:
                        candidate = decisions[3][ix]; baseline = decisions[bi][ix]
                        gain = int((contact & candidate & ~baseline).sum())
                        loss = int((contact & baseline & ~candidate).sum())
                        ref = cell['gait_minus_'+base][field]
                        assert (gain, loss, gain-loss) == (ref['gains'], ref['losses'], ref['diff'])
        for sensor in E.A.SENSORS:
            diffs = [result['metrics'][name+'/'+sensor]['gait_minus_ema']['main']['diff'] for name in E.Y.NAMES[1:]]
            assert diffs == result['decisions'][sensor]['diffs']
            assert E.decision(diffs) == result['decisions'][sensor]['decision']
        parity = E.B.A.read(E.OUT/'first_unit_parity.json')
        assert parity['status'] == 'PASS' and parity['max_abs'] == 0.
        check()
        report = dict(status='PASS', units=96, raw_arrays=raw_count, ema_query_frames=query_count,
                      max_manual_smooth_delta=max_smooth_delta, threshold_cells=50,
                      first_unit_actual_e1_parity=parity,
                      backend_limit='Stored E1 parity is actual-run evidence; a separate runtime probe does not retroactively identify run backend.',
                      seconds=time.monotonic()-tick)
        E.B.save(E.OUT/'audit_result.json', report)
        print(report, flush=True)
        return report


if __name__ == '__main__':
    audit()

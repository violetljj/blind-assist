"""Independent CPU audit of the completed query-perturbation screen.

Read-only evidence inputs; exclusive audit_independent.json receipt. Does not
import the candidate evaluator/runner or perform projection/model inference.
Usage: python -B cnh_query_perturb_audit_dev.py [--root OUT] [--cap-seconds 237]
Run once only after raw.npz, metadata.npz and result.json are complete.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import cnh_tristate_dev as R

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-query-perturb-train-dev-20261008'
NAMES = ('frozen_M3_5', 'frozen_M3_seed0', 'Control', 'Aug')
QUERIES = ('exact', 'E1')


def audit(root=OUT, cap_seconds=237.):
    root = Path(root)
    if not 0 < cap_seconds <= 237:
        raise ValueError('Remaining CPU audit budget is at most237 seconds')
    dest = root/'audit_independent.json'
    if dest.exists():
        raise FileExistsError('Preserve existing independent audit receipt')
    began = time.monotonic()
    checks = []
    report = dict(status='FAILED', checks=checks, cap_seconds=cap_seconds,
        scope='Independent original R.smooth + numpy recount; inputs/labels/orders/hash audit. No candidate evaluator functions or GPU work.',
        geometry_boundary='Reuses frozen evaluator geometry files; does not rerender scenes or independently prove geometry correctness.')

    def check():
        if time.monotonic()-began >= cap_seconds:
            raise TimeoutError('Remaining CPU audit cap reached')

    def equal(a, b, label):
        check()
        if a != b:
            raise AssertionError(f'{label}: actual={a!r} reported={b!r}')

    def same(a, b, label):
        check()
        if not np.array_equal(a, b):
            raise AssertionError(label)

    def read(path):
        check()
        return json.loads(Path(path).read_text(encoding='utf8'))

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while True:
                check()
                block = f.read(8*1024*1024)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    def threshold(values, budget):
        values = np.asarray(values, dtype=float).ravel()
        if not len(values) or not np.isfinite(values).all():
            raise AssertionError('Missing/nonfinite strict-clear scores')
        boundary = float(np.sort(values)[len(values)-budget-1])
        theta = float(np.nextafter(boundary, np.inf))
        actual = int((values >= theta).sum())
        predecessor = int((values >= boundary).sum())
        if actual > budget or predecessor <= budget:
            raise AssertionError('Integer whole-tie minimality')
        return dict(threshold=theta, target_fa_intervals=budget,
            actual_fa_intervals=actual, residual_fa_intervals=budget-actual,
            boundary_score=boundary, boundary_tie_count=int((values == boundary).sum()),
            predecessor_fa_intervals=predecessor,
            cost_status='COST_MATCHED' if actual == budget else 'NOT_COST_MATCHED')

    try:
        plan = read(root/'PLAN.json')
        equal(sha(root/'PLAN.json'), (root/'PLAN.sha256').read_text().strip(), 'frozen PLAN hash')
        equal({k: plan['training'][k] for k in ('seed', 'rows', 'epochs', 'batch', 'lr', 'weight_decay')},
              dict(seed=0, rows=27456, epochs=2, batch=256, lr=3e-4, weight_decay=1e-4), 'frozen training recipe')
        result = read(root/'result.json')
        with np.load(root/'raw.npz', allow_pickle=False) as z:
            raw = z['raw']; raw_unit = z['unit']; raw_config = z['config']
        with np.load(root/'metadata.npz', allow_pickle=False) as z:
            md = {k: z[k] for k in z.files}
        equal(raw.shape, (4, 2, 3, 1280, 13, 2), 'raw axes')
        if not np.isfinite(raw).all():
            raise AssertionError('Nonfinite raw')
        unit, config = md['unit'], md['config']
        same(raw_unit, unit, 'raw/metadata unit mapping')
        same(raw_config, config, 'raw/metadata config mapping')
        ids = list(zip(unit.tolist(), config.tolist()))
        cal_units = list(range(98000, 98008)); eval_units = list(range(99000, 99024))
        equal(plan['evaluation']['calibration_units'], cal_units, 'PLAN calibration units')
        equal(plan['evaluation']['evaluation_units'], eval_units, 'PLAN evaluation units')
        equal(len(set(ids)), 1280, 'unique clip identities')
        equal(set(ids), {(u, c) for u in cal_units+eval_units for c in range(40)}, 'cohort identities')
        # Cache geometry members once, matching the inherited truth definition.
        import cnh_tristate_event_dev as E
        lineage = read(root/'checks/evaluator_input_lineage.json')
        equal(lineage['status'], 'PASS', 'supplemental evaluator lineage status')
        equal(lineage['new_PLAN_pre_run_coverage'], False, 'honest supplemental provenance timing')
        old_plan_path = ROOT/lineage['old_plan_path']
        equal(sha(old_plan_path), lineage['old_plan_sha256'], 'old frozen evaluator PLAN hash')
        old_hashes = {p.replace('\\', '/'): h for p, h in read(old_plan_path)['hashes'].items()}
        expected_lineage = {p.relative_to(ROOT).as_posix() for p in E.GEOMETRY}
        expected_lineage.add((E.R3/'rows.json').relative_to(ROOT).as_posix())
        equal({entry['path'] for entry in lineage['inputs']}, expected_lineage, 'all4 geometry and R3 rows lineage')
        equal(len(lineage['inputs']), 5, 'unique supplemental lineage entries')
        for entry in lineage['inputs']:
            p = entry['path']
            equal(entry['expected_old_sha256'], old_hashes[p], 'older frozen expected hash '+p)
            equal(entry['match'], True, 'supplemental lineage match '+p)
            equal(entry['current_sha256'], entry['expected_old_sha256'], 'supplemental observed hash '+p)
            equal(sha(ROOT/p), entry['expected_old_sha256'], 'current evaluator payload lineage '+p)
        report['supplemental_lineage_sha256'] = sha(root/'checks/evaluator_input_lineage.json')
        checks.append('All4 geometry and R3 rows match old frozen PLAN via supplemental post-training/pre-truth-join manifest; new PLAN did not prefreeze them')
        lookup = {key: i for i, key in enumerate(ids)}
        category = np.full((1280, 2), 'MISSING', dtype='<U20')
        clear = np.zeros((1280, 2), bool); covered = clear.copy()
        fraction = np.full((1280, 2), np.nan)
        for path in E.GEOMETRY:
            check()
            with np.load(path, allow_pickle=False) as z:
                g = {k: z[k] for k in ('unit', 'config', 'query', 'ref_category', 'clear_all', 'covered', 'reference_fraction')}
            for j, key in enumerate(zip(g['unit'].tolist(), g['config'].tolist())):
                i = lookup.get(key)
                if i is None:
                    continue
                q = int(g['query'][j])
                equal(category[i, q], 'MISSING', 'duplicate truth entry')
                category[i, q] = g['ref_category'][j]; clear[i, q] = g['clear_all'][j]
                covered[i, q] = g['covered'][j]; fraction[i, q] = g['reference_fraction'][j]
        if (category == 'MISSING').any():
            raise AssertionError('Missing truth')
        same(covered[:, 0], covered[:, 1], 'covered query parity')
        np.testing.assert_allclose(fraction[:, 0], fraction[:, 1], equal_nan=True)
        contact = (covered & np.isin(category, E.CATS)).any(1)
        control = clear.all(1)
        deadline = np.searchsorted(R.FRAMES, fraction[:, 0]+1e-10, side='right')-1
        same(contact, md['contact'], 'contact source')
        same(control, md['control'], 'strict-clear source')
        same(deadline, md['deadline'], 'causal deadline source')
        cal, ev = unit < 99000, unit >= 99000
        ec, cc = ev & control, cal & control
        equal((int((cal & contact).sum()), int((ev & contact).sum()), int(cc.sum()), int(ec.sum())), (23, 47, 65, 224), 'frozen denominators')
        budget, cal_budget = int(ec.sum())//4, int(cc.sum())//4
        equal(budget, 56, 'evaluation dynamic budget'); equal(cal_budget, 16, 'calibration dynamic budget')
        checks.append('1280 raw rows and evaluator metadata independently joined; cal23/65 eval47/224; budgets16/56')
        # Apply only original smoothing, independently fuse branch and height.
        sm = R.smooth(raw).max(-1)
        scores = np.stack((sm[:, :, 0], sm[:, :, 1:3].max(2)), axis=2)
        outputs = np.arange(13)[None]

        def verify_summary(score, theta, selected, saved, label):
            alarm = score >= theta
            evt = contact & selected
            before = (outputs <= deadline[:, None]) & evt[:, None]
            active = alarm & before
            main_event = active & (outputs >= 2)
            timely = main_event.any(1); any_before = active.any(1)
            first = np.where(timely, main_event.argmax(1), -1)
            for k, value in dict(events=int(evt.sum()), timely=int(timely.sum()), any_before=int(any_before.sum()),
                    startup_only_timely=int((any_before & ~timely).sum())).items():
                equal(value, saved[k], label+'/'+k)
            ctl = control & selected
            main = alarm[ctl, 2:12]; clips = int(ctl.sum()); count = int(main.sum())
            fa = dict(interval_count=count, interval_denominator=clips*10,
                interval_rate=count/(clips*10), alarm_seconds_proxy=count*.2,
                control_seconds_proxy=clips*2., mergeclip_episode_count=int(main.any(1).sum()),
                control_clip_denominator=clips,
                maximal_positive_run_count=int(main[:, 0].sum()+(main[:, 1:] & ~main[:, :-1]).sum()))
            for k, v in fa.items():
                equal(v, saved['false_alarm'][k], label+'/false_alarm/'+k)
            ledger = [dict(unit=int(unit[i]), config=int(config[i]), deadline_output=int(deadline[i]),
                first_main_output=int(first[i]), timely=bool(timely[i]), any_before=bool(any_before[i])) for i in np.flatnonzero(evt)]
            equal(ledger, saved['event_ledger'], label+'/paired event identities')
            return timely[ev & contact]

        primary_cells = []
        for si, sensor in enumerate(('single', 'dual')):
            saved = result['sensors'][sensor]; flags = {}; matched = {}
            for mi, name in enumerate(NAMES):
                for qi, query in enumerate(QUERIES):
                    key = f'{name}/{query}'; x = scores[mi, qi, si]
                    th = threshold(x[ec, 2:12], budget)
                    equal(th, saved['primary'][key]['threshold_selection'], sensor+'/'+key+'/primary threshold')
                    flags[name, query] = verify_summary(x, th['threshold'], ev, saved['primary'][key], sensor+'/'+key+'/primary')
                    matched[name, query] = th['residual_fa_intervals'] == 0
                    ct = threshold(x[cc, 2:12], cal_budget)
                    secondary = saved['secondary'][key]
                    equal(ct, secondary['threshold_selection'], sensor+'/'+key+'/cal-only threshold')
                    verify_summary(x, ct['threshold'], ev, secondary['evaluation'], sensor+'/'+key+'/cal fixed eval')
                    verify_summary(x, ct['threshold'], cal, secondary['calibration'], sensor+'/'+key+'/cal actual')
                    primary_cells.append(dict(sensor=sensor, model=name, query=query,
                        actual_fa_intervals=th['actual_fa_intervals'], residual_fa_intervals=th['residual_fa_intervals']))
            pairs = {}
            for query in QUERIES:
                for a, b in (('Aug', 'Control'), ('Aug', 'frozen_M3_5'), ('Aug', 'frozen_M3_seed0'), ('Control', 'frozen_M3_5')):
                    aa, bb = flags[a, query], flags[b, query]
                    rescued, lost = int((aa & ~bb).sum()), int((~aa & bb).sum())
                    key = f'{query}/{a}-vs-{b}'
                    p = dict(rescues=rescued, losses=lost, net=rescued-lost,
                        both_timely=int((aa & bb).sum()), both_not_timely=int((~aa & ~bb).sum()))
                    equal(p, saved['paired'][key], sensor+'/'+key+'/paired')
                    pairs[key] = p
            exact = pairs['exact/Aug-vs-frozen_M3_5']
            gates = dict(E1_Aug_gt_Control=pairs['E1/Aug-vs-Control']['net'] >= 1,
                E1_Aug_gt_frozen_M3_5=pairs['E1/Aug-vs-frozen_M3_5']['net'] >= 1,
                exact_paired_preserved=exact['losses'] <= 2 and exact['net'] >= -2)
            required = (('Aug', 'E1'), ('Control', 'E1'), ('frozen_M3_5', 'E1'), ('Aug', 'exact'), ('frozen_M3_5', 'exact'))
            cost = all(matched[k] for k in required)
            status = 'NOT_COST_MATCHED' if not cost else 'PASS' if all(gates.values()) else 'DO_NOT_ADVANCE'
            equal(gates, saved['decision']['conditions'], sensor+'/three authorized gates')
            equal(status, saved['decision']['status'], sensor+'/decision')
        pc, sc = result['primary_contract'], result['secondary_contract']
        for k, value in dict(eval_fa_interval_budget=56, eval_control_clips=224, eval_control_intervals=2240,
                alarm_seconds_proxy_budget=56*.2, control_seconds_proxy=448.).items():
            equal(value, pc[k], 'primary contract/'+k)
        for k, value in dict(calibration_fa_interval_budget=16, calibration_control_clips=65, calibration_intervals=650).items():
            equal(value, sc[k], 'secondary contract/'+k)
        report['primary_cells'] = primary_cells
        checks.append('All16 primary and16 calibration thresholds independently sorted; actual intervals/episodes/runs, event ledgers, paired and3 gates exact')
        # Original input row order, fixed M3 labels and deterministic deltas.
        old_root = ROOT/'artifacts.local/work/cnh-near-range-20261001/data/train'
        parts = sorted(old_root.glob('features_c*.npy'))
        all_meta = []
        for p in parts:
            with np.load(p.with_name(p.name.replace('features_', 'metadata_').replace('.npy', '.npz')), allow_pickle=False) as z:
                all_meta.append({k: z[k] for k in ('unit', 'config', 'frame')})
        old_meta = {k: np.concatenate([m[k] for m in all_meta]) for k in all_meta[0]}
        with np.load(root/'inputs/rows.npz', allow_pickle=False) as z:
            rows = {k: z[k] for k in z.files}
        for k in old_meta:
            same(rows[k], old_meta[k], 'original training '+k)
        equal(len(rows['unit']), 27456, 'training row count')
        with np.load(ROOT/'artifacts.local/work/cnh-margin-labels-20261002/train_labels.npz', allow_pickle=False) as z:
            labels = z['M3'].astype(np.float32)
        same(rows['labels'].view(np.uint32), labels.view(np.uint32), 'fixed original M3 labels bytes')
        delta_cache = {}
        for u, c in zip(rows['unit'], rows['config']):
            key = int(u), int(c)
            if key not in delta_cache:
                rng = np.random.default_rng([2026100811, *key])
                delta_cache[key] = 0. if rng.random() < .25 else float(np.clip(rng.normal(0., 10.), -20., 20.))
        expected_delta = np.array([delta_cache[int(u), int(c)] for u, c in zip(rows['unit'], rows['config'])])
        same(expected_delta, rows['delta'], 'deterministic fixed per unit/config delta')
        parent = ROOT/'artifacts.local/work/cnh-temporal-readout-20261004/inputs/train'
        with np.load(parent/'rows.npz', allow_pickle=False) as z:
            pm = {k: z[k] for k in ('unit', 'config', 'frame', 'domain')}
        parent_lookup = {(int(u), int(c), int(f)): i for i, (u, c, f, d) in enumerate(zip(pm['unit'], pm['config'], pm['frame'], pm['domain'])) if d == 0}
        same(rows['parent_row'], np.array([parent_lookup[int(u), int(c), int(f)] for u, c, f in zip(rows['unit'], rows['config'], rows['frame'])]), 'history parent row mapping')
        control_vox = np.load(root/'inputs/Control.npy', mmap_mode='r')
        aug_vox = np.load(root/'inputs/Aug.npy', mmap_mode='r')
        equal(control_vox.shape, (27456, 3, 24, 17, 33), 'Control voxel shape')
        equal(aug_vox.shape, control_vox.shape, 'Aug voxel shape')
        equal(str(control_vox.dtype), 'float16', 'Control dtype'); equal(str(aug_vox.dtype), 'float16', 'Aug dtype')
        offset = 0
        for p in parts:
            original = np.load(p, mmap_mode='r')
            for j in range(0, len(original), 128):
                check(); stop = min(j+128, len(original))
                same(np.asarray(original[j:stop]).view(np.uint16), np.asarray(control_vox[offset+j:offset+stop]).view(np.uint16), 'Control original cached bytes')
                zero = rows['delta'][offset+j:offset+stop] == 0
                same(np.asarray(aug_vox[offset+j:offset+stop])[zero].view(np.uint16), np.asarray(control_vox[offset+j:offset+stop])[zero].view(np.uint16), 'Aug zero delta exact cached bytes')
            offset += len(original)
        equal(offset, 27456, 'cached original rows')
        inputs = read(root/'inputs/complete.json')
        equal(int((rows['delta'] == 0).sum()), inputs['zero_rows'], 'zero rows')
        equal(hashlib.sha256(labels.tobytes()).hexdigest(), inputs['labels_sha256'], 'labels receipt hash')
        equal(hashlib.sha256(np.stack([rows[k] for k in ('unit', 'config', 'frame')], -1).tobytes()).hexdigest(), inputs['order_sha256'], 'input order receipt hash')
        input_hashes = {arm: sha(root/f'inputs/{arm}.npy') for arm in ('Control', 'Aug')}
        for arm, value in input_hashes.items():
            equal(value, inputs[arm+'_sha256'], arm+' input hash')
        checks.append('27456 original rows and parent histories mapped; original labels/delta; entire Control and every zero Aug voxel byte exact')
        baseline = ROOT/'artifacts.local/work/cnh-margin-labels-20261002/models/M3/model_seed0.pt'
        baseline_hash = sha(baseline)
        rng = np.random.default_rng(0)
        expected_orders = [hashlib.sha256(rng.permutation(27456).tobytes()).hexdigest() for _ in range(2)]
        for arm in ('Control', 'Aug'):
            receipt = read(root/f'models/{arm}/complete.json')
            equal(receipt['status'], 'COMPLETE', arm+' fit status'); equal(receipt['seed'], 0, arm+' seed')
            equal(receipt['baseline_sha256'], baseline_hash, arm+' initialization hash')
            equal(receipt['input_sha256'], input_hashes[arm], arm+' fit input hash')
            equal(receipt['model_sha256'], sha(root/f'models/{arm}/model.pt'), arm+' final model hash')
            equal(len(receipt['history']), 2, arm+' epochs')
            for i, h in enumerate(receipt['history']):
                equal((h['epoch'], h['updates'], h['rows'], h['order_sha256']), (i+1, 108, 27456, expected_orders[i]), arm+' epoch contract')
        checks.append('Both seed0 weight initializations hash-match M3; two epochs each27456 rows/108 updates and independently reproduced common orders')
        # Frozen source/input hashes and copied source snapshots. Stream large
        # payloads, check budget between blocks; no all-array RAM duplicate.
        checked_hashes = 0
        for path, expected in plan['hashes'].items():
            equal(sha(ROOT/path), expected, 'frozen source/input hash '+path)
            checked_hashes += 1
            if path.endswith('.py') and (root/'source'/Path(path).name).exists():
                equal(sha(root/'source'/Path(path).name), expected, 'frozen source snapshot '+path)
        report['frozen_hashes_verified'] = checked_hashes
        report['evidence_sha256'] = {p: sha(root/p) for p in ('raw.npz', 'metadata.npz', 'result.json', 'PLAN.json')}
        report['source_sha256'] = sha(Path(__file__))
        checks.append('PLAN source/input hashes and archived source snapshots match; raw/metadata/result identity sealed')
        report['status'] = 'PASS'
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['seconds'] = time.monotonic()-began
        with dest.open('x', encoding='utf8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--cap-seconds', type=float, default=237.)
    args = parser.parse_args()
    receipt = audit(args.root, args.cap_seconds)
    print(json.dumps(dict(status=receipt['status'], seconds=receipt['seconds'], checks=len(receipt['checks'])), ensure_ascii=False))

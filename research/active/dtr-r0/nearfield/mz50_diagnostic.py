"""CPU-only diagnosis of frozen MZ50 scores; no fit, inference or new cutoffs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from mz50_source import _bound

QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
CONDITIONS = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
MODES = ('OPEN', 'GATED')
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def counts(mask):
    return mask.sum(0).astype(int).tolist()


class Bindings:
    def __init__(self):
        self.inputs = {}

    def __call__(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        assert expected is None or digest == expected, f'Input hash mismatch: {path}'
        self.inputs[str(path)] = digest
        return path

    def check(self):
        for path, digest in self.inputs.items():
            assert sha(Path(path)) == digest, f'Input changed: {path}'


def auxiliary_labels(source_task, records, index, bind):
    """Read only ten small source-bound auxiliary NPZs; never open source RGB/native ZIPs."""
    presence = np.empty((2560, 64, 49, 4), dtype=bool)
    known = np.empty((2560, 64, 49), dtype=bool)
    assigned = np.zeros(2560, dtype=bool)
    for shard in index['shards']:
        paths = {key: _bound(source_task, shard[key]) for key in ('auxiliary', 'auxiliary_receipt')}
        for key, path in paths.items():
            bind(path, shard[key]['sha256'])
        receipt = read(paths['auxiliary_receipt'])
        assert receipt['status'] == 'PASS' and receipt['output_sha256'] == shard['auxiliary']['sha256']
        rows = np.array([i for i, r in enumerate(records) if r['shard_id'] == shard['shard_id']])
        assert len(rows) == receipt['frames'] and not assigned[rows].any()
        with np.load(paths['auxiliary'], allow_pickle=False) as aux:
            p, k = aux['cell_event_presence'], aux['cell_known']
        assert p.shape == (len(rows), 64, 49, 4) and k.shape == (len(rows), 64, 49)
        assert p.dtype == k.dtype == np.bool_ and not (p & ~k[..., None]).any()
        for i in rows:
            record = records[i]
            native = receipt['native_inputs'][record['source_index']]
            assert native['frame_id'] == record['frame_id'] and native['native_sha256'] == record['native_sha256']
            presence[i], known[i] = p[record['source_index']], k[record['source_index']]
        assigned[rows] = True
    assert assigned.all()
    return presence.reshape(2560, 3136, 4), known.reshape(2560, 3136)


def run(task):
    started = time.perf_counter()
    task = task.resolve(strict=True)
    out, source = task / 'diagnostic-v1', task / 'run-v1'
    assert not (out / 'receipt.json').exists(), 'Final diagnostic receipt is already sealed'
    bind = Bindings()
    snapshot = read(bind(out / 'input-snapshot.json'))
    for row in snapshot['files']:
        path = bind(task / row['path'], row['sha256'])
        assert path.stat().st_size == row['bytes']
    receipt = read(source / 'receipt.json')
    score = read(task / 'score-v1/result.json')
    score_receipt = read(task / 'score-v1/receipt.json')
    assert sha(task / 'score-v1/result.json') == score_receipt['outputs']['result.json']
    assert receipt['status'] == 'PASS' and receipt['total_steps'] == 600 and receipt['fits'] == 1
    assert receipt['readouts_same_weights'] and receipt['full_source_frames'] == 2560
    for name in ('predictions.npz', 'groups.json', 'OPEN-cutoff.npy', 'GATED-cutoff.npy'):
        assert sha(source / name) == receipt['outputs'][name]
    for name in ('mz50_source.py', 'mz50_echo_independent.py', 'mz50_train.py'):
        expected = next(value for path, value in receipt['inputs'].items() if Path(path).name == name)
        bind(Path(__file__).with_name(name), expected)
    with np.load(source / 'predictions.npz', allow_pickle=False) as archive:
        p = {key: archive[key] for key in archive.files}
    group_file = read(source / 'groups.json')
    records = group_file['records']
    groups = {key: np.array(value, dtype=np.int64) for key, value in group_file['groups'].items()}
    assert {key: len(value) for key, value in groups.items()} == dict(fit=1280, calibration=256, heldout_site=640, nonfit_family=384)
    assert sorted(np.concatenate(list(groups.values())).tolist()) == list(range(2560))
    assert len({r['frame_id'] for r in records}) == 2560
    np.testing.assert_array_equal(p['mz48/frame_ids'], [r['frame_id'] for r in records])
    owner = {}
    for name, ids in groups.items():
        for i in ids:
            assert owner.setdefault(records[i]['pair_id'], name) == name
    groups['nonfit'] = np.sort(np.concatenate([groups['heldout_site'], groups['nonfit_family']]))
    groups['noncal'] = np.sort(np.concatenate([groups['fit'], groups['nonfit']]))
    source_task = task.parent / 'mz48-rich-kilotier-20260911'
    index = read(bind(source_task / 'source-index.json', receipt['source_index_sha256']))
    assert index['status'] == 'COMPLETE' and index['schema'] == 'mz48-source-index-v1'
    native, local_known = auxiliary_labels(source_task, records, index, bind)
    cutoffs = {mode: np.load(source / (mode + '-cutoff.npy'), allow_pickle=False) for mode in MODES}
    assert all(cut.shape == (4,) for cut in cutoffs.values())
    truth48, known48 = p['mz48/truth'], p['mz48/known']
    np.testing.assert_array_equal(truth48, [r['event_truth'] for r in records])

    def arrays(cohort, condition, mode, key):
        return p[f'{cohort}/{condition}/{mode}/{key}']

    def base(cohort, condition):
        return p[f'{cohort}/{condition}/MZ37']

    def winner_details(cohort, condition, mode, i, q):
        supported = bool(arrays(cohort, condition, mode, 'support')[i, q])
        address = int(arrays(cohort, condition, mode, 'winner')[i, q])
        result = dict(support=supported, saved_winner=address,
                      raw=float(arrays(cohort, condition, mode, 'raw')[i, q]),
                      saved_cutoff=float(cutoffs[mode][q]),
                      margin=float(arrays(cohort, condition, mode, 'raw')[i, q] - cutoffs[mode][q]))
        if supported:
            result.update(zone=address // 49, cell=address % 49)
            if cohort == 'mz48':
                result.update(winner_cell_known=bool(local_known[i, address]),
                              winner_native_query=bool(native[i, address, q]),
                              winner_native_any_query=native[i, address].tolist())
        else:
            result['winner_interpretation'] = 'No candidate; argmax placeholder is not a selected cell'
        return result

    def event(cohort, condition, i, q):
        frame_id = p[cohort + '/frame_ids'][i].item()
        result = dict(cohort=cohort, index=int(i), frame_id=frame_id, query=QUERIES[q],
                      truth=bool(p[cohort + '/truth'][i, q]), known=bool(p[cohort + '/known'][i, q]),
                      MZ37=float(base(cohort, condition)[i, q]))
        result.update({mode: winner_details(cohort, condition, mode, i, q) for mode in MODES})
        if cohort == 'mz48':
            result.update({key: records[i][key] for key in ('family', 'range', 'relation', 'site_id', 'support_context', 'pair_id')})
            result['native_any_query_witness'] = bool(native[i, :, q].any())
        return result

    # Existing calibration domain and existing extrema only. No cutoff function,
    # nextafter, candidate cutoff, refit, threshold sweep or new predictions.
    calibration = groups['calibration']

    def collect(mode, key):
        return np.concatenate([arrays('DEV', 'DROP_CLOSE', mode, key),
                               arrays('mz48', 'DROP_CLOSE', mode, key)[calibration]])

    cal_truth = np.concatenate([p['DEV/truth'], truth48[calibration]])
    cal_known = np.concatenate([p['DEV/known'], known48[calibration]])
    cal_base = np.concatenate([base('DEV', 'DROP_CLOSE'), base('mz48', 'DROP_CLOSE')[calibration]])
    assert cal_truth.shape == (1256, 4) and cal_known.all()
    calibration_extrema = {}
    for mode in MODES:
        raw, support = collect(mode, 'raw'), collect(mode, 'support')
        calibration_extrema[mode] = []
        for q in range(4):
            negatives = support[:, q] & (cal_base[:, q] < 0) & ~cal_truth[:, q]
            ids = np.flatnonzero(negatives)
            maximum = float(raw[ids, q].max())
            peak_ids = ids[raw[ids, q] == maximum]
            assert cutoffs[mode][q] > maximum
            assert cutoffs[mode][q] - maximum <= np.finfo(np.float64).eps * max(1., abs(maximum)) * 2
            providers = []
            for i in peak_ids:
                cohort, local = ('DEV', int(i)) if i < 1000 else ('mz48', int(calibration[i - 1000]))
                providers.append(event(cohort, 'DROP_CLOSE', local, q))
            row = dict(query=QUERIES[q], existing_cutoff=float(cutoffs[mode][q]),
                       negative_maximum=maximum, peak_providers=providers, domains={})
            for name, start, stop in (('old_DEV', 0, 1000), ('rich_calibration', 1000, 1256)):
                subset = ids[(ids >= start) & (ids < stop)]
                value = dict(frames=stop - start, negative_eligible=int(len(subset)), maxima=[])
                if len(subset):
                    top = float(raw[subset, q].max())
                    for i in subset[raw[subset, q] == top]:
                        cohort, local = ('DEV', int(i)) if i < 1000 else ('mz48', int(calibration[i - 1000]))
                        value['maxima'].append(event(cohort, 'DROP_CLOSE', local, q))
                    value['maximum_raw'] = top
                row['domains'][name] = value
            calibration_extrema[mode].append(row)

    replay_checks = dict(known_saved_bits=0, open_gated_supported_bits=0, all_baseline_positives_preserved=True)
    changed, noncal = {}, {mode: np.zeros(2, dtype=np.int64) for mode in MODES}
    noncal_by_cohort = {}
    for cohort in COHORTS:
        y, k = p[cohort + '/truth'], p[cohort + '/known']
        changed[cohort] = {}
        for condition in CONDITIONS:
            opened = arrays(cohort, condition, 'OPEN', 'raw')
            gated = arrays(cohort, condition, 'GATED', 'raw')
            gate = arrays(cohort, condition, 'GATED', 'support')
            assert arrays(cohort, condition, 'OPEN', 'support').all()
            assert (opened[gate] >= gated[gate]).all()
            replay_checks['open_gated_supported_bits'] += int(gate.sum())
            before = base(cohort, condition)
            changed[cohort][condition] = {}
            for mode in MODES:
                raw = arrays(cohort, condition, mode, 'raw')
                support = arrays(cohort, condition, mode, 'support')
                candidate = arrays(cohort, condition, mode, 'candidate')
                # Verify the saved decisions at their already sealed cutoff.
                added = (candidate >= 0) & (before < 0)
                np.testing.assert_array_equal(added, (before < 0) & support & (raw.astype(float) >= cutoffs[mode]))
                assert ((candidate >= 0) | (before < 0)).all()
                gained, false = added & y & k, added & ~y & k
                changed[cohort][condition][mode] = dict(tp_added=counts(gained), fp_added=counts(false))
                scored = score['comparisons'][cohort][condition]['all'][mode]
                assert counts(gained) == scored['tp_gained'] and counts(false) == scored['fp_added']
                if condition == 'DROP_CLOSE' and cohort != 'DEV':
                    ids = groups['noncal'] if cohort == 'mz48' else np.arange(len(y))
                    increment = np.array([gained[ids].sum(), false[ids].sum()])
                    noncal[mode] += increment
                    noncal_by_cohort.setdefault(cohort, {})[mode] = dict(tp_added=int(increment[0]), fp_added=int(increment[1]))
                replay_checks['known_saved_bits'] += int(k.sum())
    for mode in MODES:
        assert noncal[mode].tolist() == [score['gates'][mode]['all_noncal_added_tp'], score['gates'][mode]['all_noncal_added_fp']]
    assert p['mz36/attempted_known'].shape == (400, 4) and int((~p['mz36/attempted_known']).sum()) == 80

    native_witness = native.any(1)
    rich_details, disagreements, examples = {}, {}, {}
    for condition in CONDITIONS:
        prefix = 'mz48/' + condition + '/'
        before = base('mz48', condition)
        gate = arrays('mz48', condition, 'GATED', 'support')
        positives = {mode: arrays('mz48', condition, mode, 'candidate') >= 0 for mode in MODES}
        gained = {mode: positives[mode] & (before < 0) & truth48 & known48 for mode in MODES}
        winning, winner_known = {}, {}
        for mode in MODES:
            winner = arrays('mz48', condition, mode, 'winner')
            assert winner.min() >= 0 and winner.max() < 3136
            winning[mode] = np.take_along_axis(native, winner[:, None, :], 1)[:, 0]
            winning[mode] &= arrays('mz48', condition, mode, 'support')
            np.testing.assert_array_equal(winning[mode], arrays('mz48', condition, mode, 'winning_native'))
            winner_known[mode] = np.take_along_axis(local_known, winner, 1) & arrays('mz48', condition, mode, 'support')
        rich_details[condition], disagreements[condition] = {}, {}
        for group, ids in groups.items():
            residual = truth48 & known48 & (before < 0)
            row = dict(frames=int(len(ids)), residual_mz37_fn=counts(residual[ids]),
                       residual_with_gated_candidate=counts((residual & gate)[ids]),
                       residual_with_native_witness=counts((residual & native_witness)[ids]),
                       residual_native_but_no_gated_candidate=counts((residual & native_witness & ~gate)[ids]),
                       positive_without_native_crop_witness=counts((truth48 & known48 & ~native_witness)[ids]),
                       modes={})
            for mode in MODES:
                mode_row = dict(tp_added=counts(gained[mode][ids]),
                    gained_tp_winning_native=counts((gained[mode] & winning[mode])[ids]),
                    gained_tp_winner_known=counts((gained[mode] & winner_known[mode])[ids]),
                    gained_tp_without_gated_candidate=counts((gained[mode] & ~gate)[ids]),
                    gained_tp_no_native_crop_witness=counts((gained[mode] & ~native_witness)[ids]),
                    positive_score_extent=[])
                for q in range(4):
                    ii = ids[truth48[ids, q] & known48[ids, q] & arrays('mz48', condition, mode, 'support')[ids, q]]
                    raw = arrays('mz48', condition, mode, 'raw')[ii, q]
                    mode_row['positive_score_extent'].append(dict(query=QUERIES[q], supported_positive_events=int(len(ii)),
                        maximum_raw=float(raw.max()) if len(raw) else None,
                        median_raw=float(np.median(raw)) if len(raw) else None,
                        above_existing_cutoff=int((raw.astype(float) >= cutoffs[mode][q]).sum())))
                row['modes'][mode] = mode_row
                if group in score['local_witnesses'][condition]:
                    for key in ('gained_tp_winning_native', 'gained_tp_without_gated_candidate'):
                        assert mode_row[key] == score['local_witnesses'][condition][group][mode][key]
            rich_details[condition][group] = row
            a = positives['OPEN'] & ~positives['GATED'] & known48
            b = positives['GATED'] & ~positives['OPEN'] & known48

            def difference(ii):
                return dict(frames=int(len(ii)), OPEN_only_tp=counts((a & truth48)[ii]),
                            GATED_only_tp=counts((b & truth48)[ii]), OPEN_only_fp=counts((a & ~truth48)[ii]),
                            GATED_only_fp=counts((b & ~truth48)[ii]))

            detail = dict(total=difference(ids), strata={})
            for field in ('family', 'range', 'site_id', 'relation', 'support_context'):
                values = sorted({str(records[i][field]) for i in ids})
                detail['strata'][field] = {value: difference(np.array([i for i in ids if str(records[i][field]) == value]))
                                           for value in values}
            disagreements[condition][group] = detail
        nonfit_mask = np.zeros(2560, dtype=bool)
        nonfit_mask[groups['nonfit']] = True
        events = []
        differing = (positives['OPEN'] != positives['GATED']) & known48 & nonfit_mask[:, None]
        for i, q in zip(*np.where(differing)):
            row = event('mz48', condition, int(i), int(q))
            row['only_positive_readout'] = 'OPEN' if positives['OPEN'][i, q] else 'GATED'
            events.append(row)
        examples[condition] = events

    legacy_false_alerts = []
    for cohort in ('relation10000', 'distance5000', 'rich', 'mz36'):
        for mode in MODES:
            added = arrays(cohort, 'DROP_CLOSE', mode, 'candidate') >= 0
            added &= (base(cohort, 'DROP_CLOSE') < 0) & ~p[cohort + '/truth'] & p[cohort + '/known']
            for i, q in zip(*np.where(added)):
                row = event(cohort, 'DROP_CLOSE', int(i), int(q))
                row['added_by'] = mode
                legacy_false_alerts.append(row)

    result = dict(status='PASS', event_order=list(QUERIES), existing_cutoffs={m: cutoffs[m].tolist() for m in MODES},
        calibration=dict(frames=1256, old_DEV=1000, rich_calibration=256, extrema=calibration_extrema),
        replay_checks=replay_checks, additions_by_cohort=changed,
        all_noncal_additions={m: dict(tp=int(noncal[m][0]), fp=int(noncal[m][1])) for m in MODES},
        noncal_by_cohort=noncal_by_cohort,
        noncal_definition='relation2000 + distance1000 + older rich44 + MZ36 admitted380 (20 attempts remain UNKNOWN) + MZ48 fit1280 + MZ48 nonfit1024; fit is included and is not held-out evidence',
        rich_native_details=rich_details, rich_disagreements=disagreements,
        nonfit_disagreement_events=examples, legacy_drop_false_alerts=legacy_false_alerts,
        mz36_attempts=400, mz36_unknown_bits=80,
        native_label_scope='MZ48 only; independent native angular cell presence/known. Old DEV maxima have no independent native cell annotation in this diagnostic. No gated candidate is not proof of free space.',
        scientific_scope='One controlled Development fit on consumed sites; shared weights; no natural-scene, calibrated-hardware or safety conclusion',
        execution=dict(backend='CPU saved outputs and small auxiliary arrays; TASK_NOT_GPU_SUITABLE',
                       fit_steps=0, inference_frames=0, new_cutoffs=0, threshold_searches=0,
                       dense_feature_reads=0, native_depth_reads=0, RGB_reads=0, source_ZIP_reads=0))
    write(out / 'result.json', result)
    make_report(out, result)
    bind.check()
    assert 'torch' not in sys.modules
    own_code_hash = sha(Path(__file__))
    final = dict(status='PASS', inputs=bind.inputs, input_snapshot_sha256=sha(out / 'input-snapshot.json'),
                 code_sha256=own_code_hash, outputs={name: sha(out / name) for name in ('result.json', 'REPORT.md')},
                 seconds=time.perf_counter() - started, backend='CPU', torch_imported=False,
                 scientific_output_binding='FINAL; do not rewrite this receipt or its bound report/results',
                 **{key: value for key, value in result['execution'].items() if key != 'backend'})
    write(out / 'receipt.json', final)
    print(json.dumps(dict(status='PASS', seconds=final['seconds'], all_noncal=result['all_noncal_additions'],
                         primary=result['rich_disagreements']['DROP_CLOSE']['nonfit']['total'],
                         code_sha256=own_code_hash)))


def make_report(out, result):
    calibration = result['calibration']['extrema']
    drop = result['rich_native_details']['DROP_CLOSE']
    differences = result['rich_disagreements']['DROP_CLOSE']['nonfit']
    lines = ['# MZ50 固定输出机制诊断', '',
        '同一 fitted field 下，OPEN 原始最大分数在所有有 GATED 候选的位置均不低于 GATED；'
        '校准后更差来自更广候选域产生的负例极值和更高门限，不能解释为 OPEN 拥有较低的原始最大分数。', '',
        '校准仍是已执行的 1,000 旧 DEV + 256 rich calibration；这里只定位已有最大负例和使用已封定门限，'
        '没有重新校准、替换校准源、搜索门限或运行模型。', '',
        '| query | OPEN cutoff | GATED cutoff | OPEN 最大负例 provider | 该帧 GATED support/raw |',
        '| --- | ---: | ---: | --- | --- |']
    for q, name in enumerate(QUERIES):
        provider = calibration['OPEN'][q]['peak_providers'][0]
        lines.append(f'| {name} | {result["existing_cutoffs"]["OPEN"][q]:.6f} | '
                     f'{result["existing_cutoffs"]["GATED"][q]:.6f} | '
                     f'{provider["cohort"]} frame {provider["frame_id"]} | '
                     f'{provider["GATED"]["support"]} / {provider["GATED"]["raw"]:.1f} |')
    lines += ['', '两个 readout 的四个最终门限 provider 均是旧 DEV；rich calibration 没有决定任何一个门限。'
        'OPEN 的四个峰值均没有对应 GATED 候选，-20 是无候选占位，winner=0 也不能当真实选中位置。', '',
        '| query | OPEN 负例候选数（旧/rich） | GATED 负例候选数（旧/rich） | rich OPEN 最大负例 raw | rich winner known/native query |',
        '| --- | ---: | ---: | ---: | --- |']
    for q, name in enumerate(QUERIES):
        o, g = calibration['OPEN'][q]['domains'], calibration['GATED'][q]['domains']
        peak = o['rich_calibration']['maxima'][0]['OPEN']
        lines.append(f'| {name} | {o["old_DEV"]["negative_eligible"]}/{o["rich_calibration"]["negative_eligible"]} | '
                     f'{g["old_DEV"]["negative_eligible"]}/{g["rich_calibration"]["negative_eligible"]} | '
                     f'{o["rich_calibration"]["maximum_raw"]:.6f} | '
                     f'{peak["winner_cell_known"]}/{peak["winner_native_query"]} |')
    lines += ['', '这些是校准集中 MZ37 尚未报阳性且标签为负的候选数。旧 DEV 峰值没有在本诊断重建独立 native cell 真值；'
        '不能仅凭无 GATED 候选把它们称为不可见或证明安全。', '',
        '| MZ48 DROP_CLOSE 子集 | OPEN 新增 TP | GATED 新增 TP | OPEN 新增 TP 中 native winner / 无 gated 候选 | GATED 新增 TP 中 native winner |',
        '| --- | ---: | ---: | --- | ---: |']
    for group in ('fit', 'calibration', 'heldout_site', 'nonfit_family', 'nonfit'):
        o, g = (drop[group]['modes'][mode] for mode in MODES)
        lines.append(f'| {group} ({drop[group]["frames"]}) | {sum(o["tp_added"])} | {sum(g["tp_added"])} | '
                     f'{sum(o["gained_tp_winning_native"])}/{sum(o["gained_tp_without_gated_candidate"])} | '
                     f'{sum(g["gained_tp_winning_native"])} |')
    lines += ['', 'MZ48 nonfit 共 1,024 帧：OPEN 相对 MZ37 新增 4 TP / 0 FP，GATED 新增 71 TP / 1 FP。'
        '两者共有 2 个新增 TP；OPEN 独有 2 个，GATED 独有 69 个。OPEN 的 4 个新增 TP 均有 native winner，'
        '其中 2 个没有 GATED 候选；GATED 的 71 个新增 TP 中 52 个 winning cell 有 native query 证据。'
        '其余 winning cell 不是该 query 的 native witness，不等于全帧不存在 intrusion。', '',
        '| nonfit 分层 | OPEN-only TP | GATED-only TP | OPEN-only FP | GATED-only FP |',
        '| --- | ---: | ---: | ---: | ---: |']
    for field in ('family', 'range', 'site_id'):
        for value, row in differences['strata'][field].items():
            totals = [sum(row[key]) for key in ('OPEN_only_tp', 'GATED_only_tp', 'OPEN_only_fp', 'GATED_only_fp')]
            lines.append(f'| {field}: {value} | ' + ' | '.join(map(str, totals)) + ' |')
    lines += ['', '逐 query、context、relation、三个 profile 的完整分歧及 frame/pair IDs 在 result.json。'
        '所有已有 MZ37 阳性均保留。', '',
        '| query | nonfit MZ37 FN | 有 native witness 但无 GATED 候选 | OPEN 实际救回（无 GATED） |',
        '| --- | ---: | ---: | ---: |']
    for q, name in enumerate(QUERIES):
        row = drop['nonfit']
        lines.append(f'| {name} | {row["residual_mz37_fn"][q]} | '
                     f'{row["residual_native_but_no_gated_candidate"][q]} | '
                     f'{row["modes"]["OPEN"]["gained_tp_without_gated_candidate"][q]} |')
    lines += ['', '这说明真实的 missing-echo 机会存在，OPEN 也有两个独有且 native-backed 的转移命中；'
        '但放开全角域最大池化并未有效覆盖大多数缺口。负例极值不是通过删掉校准难例就可以忽略的：'
        '它们揭示了当前一次 rich-only fit 对旧域的误报风险。', '',
        '| 非校准统计组成（DROP_CLOSE） | OPEN +TP/+FP | GATED +TP/+FP |',
        '| --- | ---: | ---: |']
    for cohort, rows in result['noncal_by_cohort'].items():
        o, g = rows['OPEN'], rows['GATED']
        label = cohort + (' (fit1280 + nonfit1024)' if cohort == 'mz48' else '')
        lines.append(f'| {label} | {o["tp_added"]}/{o["fp_added"]} | {g["tp_added"]}/{g["fp_added"]} |')
    lines += ['', '**all-noncal 包含本次 fit 的 1,280 帧**，所以 OPEN +18 TP/+2 FP、GATED +197 TP/+20 FP '
        '不能整体称为非拟合泛化。MZ36 仍为 400 次尝试，20 未准入帧的 80 UNKNOWN 保留。', '',
        '下一步机制选择：保留 rich/native 空间监督与 GATED 读出作为有增益但有 FP 成本的组件；'
        '它支持“当前 head/监督配合候选几何能新增检测”，并没有独立隔离出 richer data 或 native 标签的因果贡献。'
        '取消回波门槛只证明少数可行命中，尚未证明整体有效。最值得针对的是 OPEN 对无候选区域的错误极值：'
        '在后续独立注册中，用训练来源匹配的已知负例约束与只依赖可观测输入/固定几何的 query 局部候选或拒识机制，'
        '保留 missing-echo 通路而限制不相关峰值。不要把 native UNKNOWN 直接改成负标签，也不要从校准集移除难例来制造收益。', '',
        '范围：本次单 seed、受控 Development、已消费站点。没有自然场景、校准硬件或安全结论。'
        '本诊断新 fit/inference/cutoff/threshold-search/dense/native-depth/RGB 读取均为 0；'
        '仅重读已存分数、记录与十片小型 auxiliary 标签。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    run(parser.parse_args().task)

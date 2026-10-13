"""Index-only receipt pairing on consumed captures; no inference or exposure-clock mapping."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import time

from check_frozen_fusion_inputs import read_index

ROOT = Path(__file__).resolve().parents[4]
MAX_AGE_NS = 100_000_000
MODES = ('rgb_frozen_nearest', 'rgb_causal', 'imu_frozen_saved', 'imu_causal', 'imu_nearest_diagnostic')
WAIT_MS = (0, 20, 40, 60, 100)
SOURCE_FILES = (
    'app/src/main/java/com/linnan/blindassist/MobileFusionInputBuffer.kt',
    'app/src/main/java/com/linnan/blindassist/MobileFusionSession.kt',
    'app/src/main/java/com/linnan/blindassist/vision/MobileFusionCamera.kt',
    'research/active/hardware-bringup/host/frozen_fusion_eval_v1.py',
    'research/active/hardware-bringup/host/frozen_fusion_eval_v2.py',
    'research/active/hardware-bringup/host/real_phone_imu_replay.py',
    'research/active/hardware-bringup/host/check_frozen_fusion_inputs.py',
    'research/active/hardware-bringup/host/report_causal_pairing.py',
)


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return dict(n=0, p50=None, p95=None, minimum=None, maximum=None)

    def percentile(q):
        i = (len(ordered) - 1) * q
        lo, hi = math.floor(i), math.ceil(i)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (i - lo)

    return dict(n=len(ordered), p50=percentile(.5), p95=percentile(.95),
                minimum=ordered[0], maximum=ordered[-1])


def nearest(records, receipt):
    """numpy.argmin semantics from frozen v1: first index-order record wins a tie."""
    return min(records, key=lambda row: abs(receipt - row['received_elapsed_ns']), default=None)


def causal(records, receipt):
    """Buffer.lastOrNull predicate on index order, with both 0 and 100 ms included.

    All indexed qualifying callbacks are assumed available. Actual buffer arrival,
    future-callback eviction and cross-stream epoch changes are not replayed.
    """
    return next((row for row in reversed(records)
                 if 0 <= receipt - row['received_elapsed_ns'] <= MAX_AGE_NS), None)


def wait_join(records, receipt, wait_ms):
    """Ideal fixed-deadline receipt join; still uses the unchanged absolute 100 ms gate.

    At zero wait use the buffer's lastOrNull exactly. With waiting, equal-distance
    ties prefer a past receipt, then the last index record. No future callback
    beyond the decision deadline is visible to this policy.
    """
    if wait_ms == 0:
        return causal(records, receipt)
    candidates = [(i, row) for i, row in enumerate(records)
                  if -wait_ms * 1_000_000 <= receipt - row['received_elapsed_ns'] <= MAX_AGE_NS
                  and abs(receipt - row['received_elapsed_ns']) <= MAX_AGE_NS]
    selected = min(candidates, key=lambda item: (abs(receipt - item[1]['received_elapsed_ns']),
                   item[1]['received_elapsed_ns'] > receipt, -item[0]), default=None)
    return selected[1] if selected else None


def has_gravity(row):
    gravity = row.get('gravity_unit')
    return (isinstance(gravity, list) and len(gravity) == 3
            and all(type(v) in (int, float) and math.isfinite(v) for v in gravity)
            and math.hypot(*gravity) > 0)


def pair(row, receipt, session, kind, rule):
    if row is None:
        return dict(matched=False, reason='no_saved_reference' if rule == 'saved' else 'no_eligible_receipt')
    age_ns = receipt - row['received_elapsed_ns']
    matched = rule == 'saved' or abs(age_ns) <= MAX_AGE_NS
    key = (f"{session}|{row['boot_id']}|{row['source_frame_id']}"
           if kind == 'rgb' and row.get('boot_id') is not None and row.get('source_frame_id') is not None
           else f"{session}|index:{row['index']}")
    result = dict(matched=matched, record_index=row['index'], record_key=f"{session}|{row['index']}",
                  frame_key=key, age_ms=age_ns / 1e6, future=age_ns < 0,
                  within_causal_100ms=0 <= age_ns <= MAX_AGE_NS)
    if not matched:
        result['reason'] = 'nearest_receipt_outside_100ms'
    if kind == 'imu':
        result['gravity_available'] = has_gravity(row)
        result['device_id'] = row.get('device_id')
        result['boot_id'] = row.get('boot_id')
    return result


def session_frames(name, tof, rgb, imu):
    valid_tof = [row for row in tof if row.get('valid_frame')]
    valid_imu = [row for row in imu if row.get('valid_frame')]
    imu_by_index = {row['index']: row for row in valid_imu}
    frames = []
    for ordinal, tof_row in enumerate(valid_tof):
        receipt = tof_row['received_elapsed_ns']
        reference = (tof_row.get('imu_alignment') or {}).get('imu_record_index')
        saved = imu_by_index.get(reference)
        if reference is not None and saved is None:
            raise ValueError(f'{name}: ToF {tof_row["index"]} has broken saved IMU reference {reference}')
        row = dict(session=name, tof_index=tof_row['index'], valid_cnh_ordinal=ordinal,
                   sequence=tof_row['sequence'], received_elapsed_ns=receipt,
                   rgb_frozen_nearest=pair(nearest(rgb, receipt), receipt, name, 'rgb', 'nearest'),
                   rgb_causal=pair(causal(rgb, receipt), receipt, name, 'rgb', 'causal'),
                   imu_frozen_saved=pair(saved, receipt, name, 'imu', 'saved'),
                   imu_causal=pair(causal(valid_imu, receipt), receipt, name, 'imu', 'causal'),
                   imu_nearest_diagnostic=pair(nearest(valid_imu, receipt), receipt, name, 'imu', 'nearest'))
        for wait_ms in WAIT_MS:
            row[f'rgb_wait_{wait_ms}ms'] = pair(wait_join(rgb, receipt, wait_ms), receipt, name, 'rgb', 'wait')
        frames.append(row)
    timeline = {}
    for stream, rows in (('tof', valid_tof), ('rgb', rgb), ('imu', valid_imu)):
        steps = [b['received_elapsed_ns'] - a['received_elapsed_ns'] for a, b in zip(rows, rows[1:])]
        timeline[stream] = dict(records=len(rows), receipt_backward_edges=sum(v < 0 for v in steps),
                                receipt_equal_edges=sum(v == 0 for v in steps),
                                receipt_step_ms=quantiles([v / 1e6 for v in steps]))
    timeline['imu']['identity_changes'] = sum((a.get('device_id'), a.get('boot_id')) !=
            (b.get('device_id'), b.get('boot_id')) for a, b in zip(valid_imu, valid_imu[1:]))
    timeline['rgb']['frame_identity_fallback_records'] = sum(
        row.get('boot_id') is None or row.get('source_frame_id') is None for row in rgb)
    timeline['tof']['invalid_records_retained'] = len(tof) - len(valid_tof)
    return frames, timeline


def mode_summary(frames, mode):
    selected = [row[mode] for row in frames if row[mode]['matched']]
    counts = Counter(row['record_key'] for row in selected)
    result = dict(matched=len(selected), missing=len(frames) - len(selected), denominator_cnh=len(frames),
                  coverage_pct=100 * len(selected) / len(frames) if frames else None,
                  future_matches=sum(row['future'] for row in selected),
                  signed_receipt_age_ms=quantiles([row['age_ms'] for row in selected]),
                  absolute_receipt_separation_ms=quantiles([abs(row['age_ms']) for row in selected]),
                  unique_records=len(counts), records_used_multiple_times=sum(n > 1 for n in counts.values()),
                  tof_uses_of_reused_records=sum(n for n in counts.values() if n > 1),
                  extra_uses_beyond_first=sum(n - 1 for n in counts.values()),
                  maximum_uses_of_one_record=max(counts.values(), default=0))
    if mode.startswith('imu_'):
        result['with_gravity'] = sum(row['gravity_available'] for row in selected)
    else:
        result['distinct_reported_rgb_frames'] = len({row['frame_key'] for row in selected})
        result['extra_reported_frame_uses_beyond_first'] = len(selected) - result['distinct_reported_rgb_frames']
    return result


def transitions(frames, left, right):
    counts = Counter()
    extra_separation = []
    for frame in frames:
        a, b = frame[left], frame[right]
        if a['matched'] and b['matched']:
            same = a['record_key'] == b['record_key']
            counts['both_same_record' if same else 'both_different_record'] += 1
            counts['both_same_frame_id' if a['frame_key'] == b['frame_key'] else 'both_different_frame_id'] += 1
            if not same:
                counts['changed_from_future' if a['future'] else 'changed_from_nonfuture'] += 1
                counts['changed_with_equal_receipt_time'] += a['age_ms'] == b['age_ms']
            extra_separation.append(abs(b['age_ms']) - abs(a['age_ms']))
        else:
            counts['left_only' if a['matched'] else 'right_only' if b['matched'] else 'neither'] += 1
    fields = ('both_same_record', 'both_different_record', 'both_same_frame_id', 'both_different_frame_id',
              'changed_from_future', 'changed_from_nonfuture', 'changed_with_equal_receipt_time',
              'left_only', 'right_only', 'neither')
    return dict(denominator_cnh=len(frames), **{key: counts[key] for key in fields},
                added_absolute_receipt_separation_ms=quantiles(extra_separation))


def potential_windows(frames, mode):
    histogram, qualified_histogram = Counter(), Counter()
    count = repeat = one_id_multiple_votes = qualified = qualified_repeat = 0
    for i in range(2, len(frames)):
        window = frames[i - 2:i + 1]
        if len({row['session'] for row in window}) != 1:
            continue
        selected = [row[mode]['frame_key'] for row in window if row[mode]['matched']]
        distinct = len(set(selected))
        count += 1
        histogram[str(distinct)] += 1
        repeat += len(selected) > distinct
        one_id_multiple_votes += len(selected) >= 2 and distinct == 1
        sequence_contiguous = all(b['sequence'] == a['sequence'] + 1 for a, b in zip(window, window[1:]))
        imu_available = all(row['imu_causal']['matched'] and row['imu_causal']['gravity_available'] for row in window)
        if sequence_contiguous and imu_available:
            qualified += 1
            qualified_histogram[str(distinct)] += 1
            qualified_repeat += len(selected) > distinct
    return dict(three_adjacent_valid_cnh_windows=count, distinct_rgb_ids={str(i): histogram[str(i)] for i in range(4)},
                windows_reusing_rgb=repeat, at_least_two_matched_but_one_rgb_id=one_id_multiple_votes,
                sequence_contiguous_with_causal_imu_windows=qualified,
                qualified_distinct_rgb_ids={str(i): qualified_histogram[str(i)] for i in range(4)},
                qualified_windows_reusing_rgb=qualified_repeat,
                interpretation='Potential index windows only; no ready/pose/scale gate or actual worker/vote observation. Different IDs do not establish statistical independence.')


def summarize(frames):
    waits = {}
    for wait_ms in WAIT_MS:
        key = f'rgb_wait_{wait_ms}ms'
        waits[str(wait_ms)] = dict(**mode_summary(frames, key), planned_fixed_hold_ms=wait_ms,
                additional_maximum_wait_ms=wait_ms,
                selected_receipt_wait_lower_bound_ms=quantiles(
                    [max(0, -row[key]['age_ms']) for row in frames if row[key]['matched']]),
                from_zero_wait=transitions(frames, 'rgb_wait_0ms', key),
                potential_three_cnh_windows=potential_windows(frames, key))
    return dict(denominator_cnh=len(frames), modes={key: mode_summary(frames, key) for key in MODES},
                rgb_nearest_to_causal=transitions(frames, 'rgb_frozen_nearest', 'rgb_causal'),
                imu_saved_to_causal=transitions(frames, 'imu_frozen_saved', 'imu_causal'),
                causal_rgb_and_imu_gravity_available=sum(row['rgb_causal']['matched'] and
                    row['imu_causal']['matched'] and row['imu_causal']['gravity_available'] for row in frames),
                potential_three_cnh_windows={key: potential_windows(frames, key)
                                            for key in ('rgb_frozen_nearest', 'rgb_causal')},
                bounded_wait_rgb=waits)


def report_text(report):
    total = report['aggregate']; modes = total['modes']; n = total['denominator_cnh']
    change = total['rgb_nearest_to_causal']
    lines = ['# 18 段已消费录制：最近与因果接收配对', '',
             f"全部 {n} 个有效 CNH：最近 RGB {modes['rgb_frozen_nearest']['matched']}/{n}，因果 RGB {modes['rgb_causal']['matched']}/{n}；少 {change['left_only']} 帧。共同可配对中，换图 {change['both_different_record']} 帧、同图 {change['both_same_record']} 帧。", '',
             '只读 tof.index.jsonl、imu.index.jsonl、rgb.jsonl；未读标签、图像/传感器 payload 或执行模型。', '',
             '## 接收时间与覆盖', '',
             '年龄 = ToF Android 接收时间 − 对应源 Android 接收时间，单位 ms；负数表示未来收到的源帧。百分位采用全量配对帧线性插值，缺失保留在 CNH 分母。它不是曝光差，也不是处理延迟。', '',
             '| 配对 | 可配对 / CNH | 未来帧 | 年龄 P50 / P95 (ms) | 绝对间隔 P50 / P95 (ms) |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name in MODES:
        m = modes[name]; age = m['signed_receipt_age_ms']; absolute = m['absolute_receipt_separation_ms']
        fmt = lambda v: '-' if v is None else f'{v:.3f}'
        lines.append(f"| {name} | {m['matched']}/{n} | {m['future_matches']} | {fmt(age['p50'])} / {fmt(age['p95'])} | {fmt(absolute['p50'])} / {fmt(absolute['p95'])} |")
    lines += ['', '冻结 v1 的 RGB 用绝对最近帧（等距取索引中首条），≤100 ms 才进入后续分支；IMU 使用采集时保存的引用，**并非离线最近 IMU**。imu_nearest_diagnostic 仅为说明差异的旁证。', '',
              '手机 MobileFusionInputBuffer 对 RGB/IMU 均取插入顺序中最后一条年龄在 [0,100] ms 的记录，队列容量分别 16/64；IMU boot/device 改变时清空两路。这里的 causal 假设所有符合接收时间的索引回调当时已可用。', '',
              '## 按片段', '',
              '| 开始 (UTC+8) | CNH | 最近RGB | 因果RGB | 同图 / 换图 / 失配 | 因果RGB年龄 P50 / P95 (ms) | 保存IMU / 因果IMU | 最近复用 / 因果复用额外次数 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for session in report['sessions']:
        s = session['summary']; m = s['modes']; c = s['rgb_nearest_to_causal']; age = m['rgb_causal']['signed_receipt_age_ms']
        local = (datetime.strptime(session['session'][8:23], '%Y%m%dT%H%M%S') + timedelta(hours=8)).strftime('%H:%M:%S')
        lines.append(f"| {local} | {s['denominator_cnh']} | {m['rgb_frozen_nearest']['matched']} | {m['rgb_causal']['matched']} | {c['both_same_record']} / {c['both_different_record']} / {c['left_only']} | {fmt(age['p50'])} / {fmt(age['p95'])} | {m['imu_frozen_saved']['with_gravity']} / {m['imu_causal']['with_gravity']} | {m['rgb_frozen_nearest']['extra_uses_beyond_first']} / {m['rgb_causal']['extra_uses_beyond_first']} |")
    lines += ['', '## 可能的三帧投票重复', '',
              '以下窗口是各片段连续三个有效 CNH，尚未经过模型就绪、姿态、尺度与实时工作队列筛选；只是潜在投票输入，不是已经发生的提醒。', '',
              '| 规则 | RGB 额外复用次数 | 三CNH窗口数 | 不同RGB ID数 0 / 1 / 2 / 3 | 有重复RGB窗口 | 至少两次匹配但只有一张图 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key in ('rgb_frozen_nearest', 'rgb_causal'):
        w = total['potential_three_cnh_windows'][key]; h = w['distinct_rgb_ids']
        lines.append(f"| {key} | {modes[key]['extra_uses_beyond_first']} | {w['three_adjacent_valid_cnh_windows']} | {' / '.join(str(h[str(i)]) for i in range(4))} | {w['windows_reusing_rgb']} | {w['at_least_two_matched_but_one_rgb_id']} |")
    lines += ['', '## 有限等待的在线可实现候选', '',
              'ToF 接收后固定等待 w ms，在该 deadline 前已接收的 RGB 中选择距 ToF 接收时间最近、绝对差≤100 ms 的一条；等距优先过去图，其次索引最后一条。0 ms 精确采用手机 lastOrNull 规则。每帧计划额外等待上限为 w；所选图的到达等待下界仅是 max(0, RGB接收−ToF接收)，不是实测端到端延迟，也不是已实现的提前返回策略。', '',
              '| 固定等待/额外上限 (ms) | RGB覆盖 / CNH | 相对0ms 新增 / 丢失 / 换图 | 所选图到达等待下界 P50 / P95 (ms) | RGB额外复用 / 重复窗口 |',
              '| ---: | ---: | ---: | ---: | ---: |']
    for wait_ms in WAIT_MS:
        candidate = total['bounded_wait_rgb'][str(wait_ms)]; c = candidate['from_zero_wait']
        arrival = candidate['selected_receipt_wait_lower_bound_ms']
        lines.append(f"| {wait_ms} | {candidate['matched']}/{n} | {c['right_only']} / {c['left_only']} / {c['both_different_record']} | {fmt(arrival['p50'])} / {fmt(arrival['p95'])} | {candidate['extra_uses_beyond_first']} / {candidate['potential_three_cnh_windows']['windows_reusing_rgb']} |")
    lines += ['', '## 解释范围', '',
              '配对索引与全部分母见 per_cnh.jsonl / report.json。索引不会记录跨流并发回调实际先后、匹配时队列是否已被未来回调挤出、epoch 失效、ToF/RGB 工作队列丢弃、推理是否就绪或 750 ms 处理过期。因此结果不能冒充完整在线回放，也不能证明提醒准确率。IMU gravity_available 只检查字段可用，未执行设备标定、姿态旋转或姿态门控。', '',
              '本次不改变 100 ms 门限、不改变 v1/v2 或手机投票。原始设备时钟保持独立，不将 Android 接收匹配称为曝光同步。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expect-cnh', type=int, default=1818)
    args = parser.parse_args()
    cpu_started, wall_started = time.process_time(), time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    contract = dict(goal='Compare frozen versus mobile receipt-time input pairing on consumed captures',
                    cpu_budget_seconds=180, gpu_seconds=0, download_bytes=0,
                    data_role='Consumed 2026-10-12 Development; index-only engineering description',
                    raw=str(args.raw.resolve()), expected_cnh=args.expect_cnh,
                    max_age_ns=MAX_AGE_NS, payloads_read=False, labels_read=False, inference_run=False,
                    fixed_wait_candidate_ms=WAIT_MS,
                    source_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_FILES})
    (args.out / 'task_record.json').write_text(json.dumps(contract, indent=2), encoding='utf8')
    sessions, all_frames = [], []
    for folder in sorted(args.raw.glob('capture-*')):
        if time.process_time() - cpu_started > 180:
            raise TimeoutError('180 CPU-second budget reached; no further sessions opened')
        if not folder.is_dir():
            continue
        rows = {stream: [row for _, row in read_index(folder / filename)] for stream, filename in
                (('tof', 'tof.index.jsonl'), ('imu', 'imu.index.jsonl'), ('rgb', 'rgb.jsonl'))}
        frames, timeline = session_frames(folder.name, rows['tof'], rows['rgb'], rows['imu'])
        sessions.append(dict(session=folder.name, timeline=timeline, summary=summarize(frames)))
        all_frames.extend(frames)
    if len(all_frames) != args.expect_cnh:
        raise ValueError(f'Expected all {args.expect_cnh} valid CNH frames, got {len(all_frames)}; no subset claim')
    report = dict(contract=contract, sessions=sessions, aggregate=summarize(all_frames),
                  cpu_seconds=time.process_time() - cpu_started, wall_seconds=time.perf_counter() - wall_started)
    with (args.out / 'per_cnh.jsonl').open('w', encoding='utf8') as output:
        for frame in all_frames:
            output.write(json.dumps(frame) + '\n')
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    (args.out / 'REPORT.md').write_text(report_text(report), encoding='utf8')
    with (args.out / 'sessions.csv').open('w', newline='', encoding='utf8') as output:
        writer = csv.writer(output)
        writer.writerow(['session', 'valid_cnh', 'rgb_nearest', 'rgb_causal', 'same_record', 'changed_record',
                         'nearest_only', 'neither', 'causal_age_p50_ms', 'causal_age_p95_ms',
                         'imu_saved_with_gravity', 'imu_causal_with_gravity'])
        for session in sessions:
            s = session['summary']; m = s['modes']; c = s['rgb_nearest_to_causal']
            age = m['rgb_causal']['signed_receipt_age_ms']
            writer.writerow([session['session'], s['denominator_cnh'], m['rgb_frozen_nearest']['matched'],
                             m['rgb_causal']['matched'], c['both_same_record'], c['both_different_record'],
                             c['left_only'], c['neither'], age['p50'], age['p95'],
                             m['imu_frozen_saved']['with_gravity'], m['imu_causal']['with_gravity']])
    with (args.out / 'wait_candidates.csv').open('w', newline='', encoding='utf8') as output:
        writer = csv.writer(output)
        writer.writerow(['session', 'valid_cnh', 'fixed_wait_ms', 'matched', 'added_vs_zero', 'lost_vs_zero',
                         'changed_record_vs_zero', 'record_extra_uses', 'source_frame_extra_uses',
                         'selected_arrival_wait_p50_ms', 'selected_arrival_wait_p95_ms'])
        for session in sessions + [dict(session='ALL', summary=report['aggregate'])]:
            for wait_ms in WAIT_MS:
                c = session['summary']['bounded_wait_rgb'][str(wait_ms)]; t = c['from_zero_wait']
                age = c['selected_receipt_wait_lower_bound_ms']
                writer.writerow([session['session'], session['summary']['denominator_cnh'], wait_ms,
                                 c['matched'], t['right_only'], t['left_only'], t['both_different_record'],
                                 c['extra_uses_beyond_first'], c['extra_reported_frame_uses_beyond_first'],
                                 age['p50'], age['p95']])
    print(json.dumps(dict(sessions=len(sessions), valid_cnh=len(all_frames), cpu_seconds=report['cpu_seconds'],
                         wall_seconds=report['wall_seconds'], rgb=report['aggregate']['rgb_nearest_to_causal'],
                         imu=report['aggregate']['imu_saved_to_causal']), indent=2))


if __name__ == '__main__':
    main()

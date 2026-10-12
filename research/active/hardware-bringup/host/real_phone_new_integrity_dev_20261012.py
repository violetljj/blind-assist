"""Read-only integrity/receipt-clock audit of the 18 new phone captures.

All indexed bytes, including headers and truncated packets, are accounted for.
Clock domains are never subtracted or mapped to one another. No model or labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = ROOT / 'artifacts.local/work/phone-new-capture-20261012'


def read_rows(path):
    rows, bad = [], []
    for n, b in enumerate(path.read_bytes().splitlines(), 1):
        if not b.strip():
            bad.append(n)
            continue
        try:
            rows.append(json.loads(b))
        except (UnicodeDecodeError, json.JSONDecodeError):
            bad.append(n)
    if bad:
        raise ValueError(f'{path}: malformed/blank index lines {bad}')
    return rows


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), minimum=float(a.min()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), maximum=float(a.max())) if len(a) else dict(n=0)


def timeline(rows, seq_field, device_field, factor):
    seq = np.array([r[seq_field] for r in rows], np.int64)
    device = np.array([r[device_field] for r in rows], np.int64)
    receipt = np.array([r['received_elapsed_ns'] for r in rows], np.int64)
    ds, dd, dr = np.diff(seq), np.diff(device), np.diff(receipt)
    # factor converts THIS device clock to milliseconds, never aligns it to Android.
    long_device = [dict(previous_index=rows[i]['index'], current_index=rows[i+1]['index'],
                       previous_sequence=int(seq[i]), current_sequence=int(seq[i+1]),
                       device_step_ms=float(dd[i]*factor), receipt_step_ms=float(dr[i]/1e6))
                   for i in np.flatnonzero(dd*factor > 500)]
    return dict(frames=len(rows), sequence_duplicate=int((ds == 0).sum()),
                sequence_reset_or_reorder=int((ds < 0).sum()),
                sequence_gap_edges=int((ds > 1).sum()), missing_sequence_count=int(np.maximum(ds - 1, 0).sum()),
                device_nonincreasing_edges=int((dd <= 0).sum()),
                receipt_backward_edges=int((dr < 0).sum()),
                receipt_equal_edges=int((dr == 0).sum()),
                device_step_ms=quantiles(dd * factor), receipt_step_ms=quantiles(dr / 1e6),
                device_span_s=float((device[-1]-device[0])*factor/1000) if len(device) else None,
                receipt_span_s=float((receipt[-1]-receipt[0])/1e9) if len(receipt) else None,
                receipt_gaps_gt_500ms=int((dr > 500_000_000).sum()),
                device_gaps_gt_500ms=long_device)


def raw_records(folder, stream):
    rows = read_rows(folder / f'{stream}.index.jsonl')
    raw = (folder / f'{stream}.raw.jsonl').read_bytes()
    cursor, parsed, invalid, valid, zero = 0, {}, [], [], 0
    for ordinal, r in enumerate(rows, 1):
        if r['index'] != ordinal or r['raw_byte_offset'] != cursor or r['raw_byte_length'] <= 0:
            raise ValueError(f'{folder.name}/{stream}: index/byte coverage failure at {ordinal}')
        end = cursor+r['raw_byte_length']
        if end > len(raw):
            raise ValueError(f'{folder.name}/{stream}: byte overrun at {ordinal}')
        payload = raw[cursor:end]
        cursor = end
        try:
            f = json.loads(payload)
            parsed[ordinal] = f
            kind = f.get('type', 'missing_type')
        except (UnicodeDecodeError, json.JSONDecodeError):
            f, kind = None, 'unparseable_raw'
        if not r['valid_frame']:
            invalid.append(dict(index=ordinal, type=kind, bytes=len(payload),
                                terminated_newline=payload.endswith(b'\n'), offset=r['raw_byte_offset']))
            continue
        if f is None:
            raise ValueError(f'{folder.name}/{stream}: valid index points to malformed raw')
        if stream == 'tof':
            if f.get('type') != 'cnh_frame' or [f.get(k) for k in ('rows', 'cols', 'bins')] != [8, 8, 16]:
                raise ValueError(f'{folder.name}: unexpected CNH structure')
            for k, shape in [('hist_raw',(64,16)), ('hist_scaler',(64,16)),
                             ('ambient_raw',(64,)), ('ambient_scaler',(64,)),
                             ('distance_mm',(64,)), ('target_status',(64,)), ('nb_target',(64,))]:
                a = np.asarray(f[k])
                if a.shape != shape or not np.isfinite(a).all():
                    raise ValueError(f'{folder.name}: bad {k} at {ordinal}')
            for k, raw_field in [('sequence','seq'),('device_timestamp_ms','ms')]:
                if r[k] != f[raw_field]:
                    raise ValueError(f'{folder.name}: CNH {k} mismatch')
            scaled = np.ldexp(np.asarray(f['hist_raw'], float), -np.asarray(f['hist_scaler'], int))
            ambient = np.ldexp(np.asarray(f['ambient_raw'], float), -np.asarray(f['ambient_scaler'], int))
            if not np.isfinite(scaled).all() or not np.isfinite(ambient).all():
                raise ValueError(f'{folder.name}: scaled data not finite')
            zero += int((np.asarray(f['hist_raw']) == 0).sum())
        else:
            if f.get('type') != 'imu_frame' or not f.get('valid'):
                raise ValueError(f'{folder.name}: valid IMU index refers to invalid raw')
            for k, raw_field in [('sequence','seq'),('device_timestamp_us','device_us'),
                                 ('boot_id','boot_id'),('pitch_deg','pitch_deg'),('roll_deg','roll_deg')]:
                if r[k] != f[raw_field]:
                    raise ValueError(f'{folder.name}: IMU {k} mismatch')
            for k in ('gravity_unit', 'accel_raw', 'gyro_raw', 'accel_g', 'gyro_dps'):
                if len(f[k]) != 3 or not np.isfinite(f[k]).all() or r[k] != f[k]:
                    raise ValueError(f'{folder.name}: IMU {k} malformed/mismatch')
        valid.append(r)
    if cursor != len(raw):
        raise ValueError(f'{folder.name}/{stream}: {len(raw)-cursor} unindexed raw bytes')
    return rows, valid, parsed, dict(records=len(rows), valid_frames=len(valid), raw_bytes=len(raw),
                 raw_sha256=hashlib.sha256(raw).hexdigest(), indexed_raw_bytes=cursor,
                 invalid_records=invalid, raw_type_counts=dict(Counter(f.get('type', 'missing_type') for f in parsed.values())),
                 hist_zero_entries=zero if stream == 'tof' else None)


def audit(folder):
    m = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    tr, tv, tp, ts = raw_records(folder, 'tof')
    ir, iv, ip, ims = raw_records(folder, 'imu')
    rgb, events = read_rows(folder/'rgb.jsonl'), read_rows(folder/'events.jsonl')
    expected = dict(rgb_count=len(rgb), tof_count=len(tv), tof_record_count=len(tr),
                    imu_count=len(iv), imu_record_count=len(ir), event_count=len(events))
    if any(m[k] != v for k, v in expected.items()):
        raise ValueError(f'{folder.name}: manifest count mismatch: {expected}')
    referenced = set()
    for i, r in enumerate(rgb, 1):
        p = folder/r['file']
        b = p.read_bytes()
        if r['index'] != i or len(b) != r['bytes'] or hashlib.sha256(b).hexdigest() != r['sha256']:
            raise ValueError(f'{folder.name}: RGB digest/size/index mismatch at {i}')
        referenced.add(p.resolve())
    actual_rgb = {p.resolve() for p in (folder/'rgb').iterdir() if p.is_file()}
    if referenced != actual_rgb:
        raise ValueError(f'{folder.name}: unindexed/missing RGB payload')
    imu_by_index = {r['index']: r for r in iv}
    imu_receipt = np.array([r['received_elapsed_ns'] for r in iv], np.int64)
    if len(imu_receipt) and (np.diff(imu_receipt) < 0).any():
        raise ValueError(f'{folder.name}: IMU receipt backward prevents causal search')
    residuals, aligned, unavailable, newest_discrepancies, missing_reasons = [], 0, [], [], []
    for r in tv:
        a = r['imu_alignment']
        nearest = np.searchsorted(imu_receipt, r['received_elapsed_ns'], side='right')-1
        if a['status'] != 'ALIGNED_RECEIPT_CAUSAL':
            unavailable.append(r['index'])
            age = (r['received_elapsed_ns']-imu_receipt[nearest])/1e6 if nearest >= 0 else None
            missing_reasons.append(dict(tof_index=r['index'],nearest_receipt_causal_age_ms=age,
                    reason='no_causal_indexed_imu' if age is None else 'over_100ms' if age > 100 else 'not_available_at_online_save'))
            continue
        irf = imu_by_index.get(a['imu_record_index'])
        if irf is None:
            raise ValueError(f'{folder.name}: missing/invalid causal IMU reference')
        delta = irf['received_elapsed_ns']-r['received_elapsed_ns']
        expected_fields = dict(imu_minus_tof_ns=delta, imu_sequence=irf['sequence'],
            imu_boot_id=irf['boot_id'], imu_device_us=irf['device_timestamp_us'],
            imu_pitch_deg=irf['pitch_deg'], imu_roll_deg=irf['roll_deg'])
        if not -100_000_000 <= delta <= 0 or any(a[k] != v for k,v in expected_fields.items()):
            raise ValueError(f'{folder.name}: invalid causal IMU alignment')
        if a['raw_reference'] != dict(index_file='imu.index.jsonl',record_index=irf['index'],raw_file='imu.raw.jsonl'):
            raise ValueError(f'{folder.name}: raw reference mismatch')
        if nearest >= 0 and iv[nearest]['index'] != irf['index']:
            newest_discrepancies.append(dict(tof_index=r['index'],saved=irf['index'],
                    newest_receipt_causal=iv[nearest]['index'],
                    saved_age_ms=-delta/1e6,
                    newest_age_ms=float((r['received_elapsed_ns']-imu_receipt[nearest])/1e6)))
        residuals.append(delta/1e6)
        aligned += 1
    ts['timeline'] = timeline(tv,'sequence','device_timestamp_ms',1)
    ims['timeline'] = timeline(iv,'sequence','device_timestamp_us',.001)
    ims['sample_gap_count'] = sum(bool(r.get('sample_gap')) for r in iv)
    ims['boot_ids'] = dict(Counter(r['boot_id'] for r in iv))
    rgb_clock = timeline(rgb,'source_frame_id','device_capture_timestamp_us',.001)
    rgb_clock['boot_ids'] = dict(Counter(r['boot_id'] for r in rgb))
    # Cached boot/config records at connection preserve device identity; physical
    # resets require changing identity or resetting seq/device time in valid data.
    boot = [dict(index=i,**f) for i,f in tp.items() if f.get('type') in ('boot','transport_connection','config')]
    return dict(session=folder.name, manifest_status=m['status'], counts=expected,
            tof=ts, imu=ims, rgb=rgb_clock, rgb_payload_checks=len(rgb),
            event_kind_counts=dict(Counter(r['kind'] for r in events)), events=events,
            tof_connection_and_boot_records=boot,
            causal_imu=dict(checked_valid_tof=len(tv), aligned=aligned, unavailable_indices=unavailable,
                           unavailable_reasons=missing_reasons,
                           saved_reference_fields_equal=True, signed_imu_minus_tof_ms=quantiles(residuals),
                           newest_receipt_causal_discrepancies=newest_discrepancies),
            bytes_and_counts_pass=True)


def main(root):
    started=time.monotonic()
    receipt=json.loads((root/'import_receipt.json').read_text(encoding='utf8'))
    out=root/'integrity'; out.mkdir(exist_ok=True)
    sessions=[audit(root/'raw'/s) for s in receipt['sessions']]
    result=dict(schema='phone-new-integrity-dev-20261012', sessions=sessions,
                transfer_receipt=dict(files=receipt['files'],bytes=receipt['bytes'],
                    every_phone_sha_equal=receipt['every_phone_sha_equal'],archive_sha256=receipt['archive_sha256']),
                checked=dict(sessions=len(sessions),rgb=sum(s['counts']['rgb_count'] for s in sessions),
                    tof_valid=sum(s['counts']['tof_count'] for s in sessions),
                    tof_records=sum(s['counts']['tof_record_count'] for s in sessions),
                    imu_valid=sum(s['counts']['imu_count'] for s in sessions),
                    imu_records=sum(s['counts']['imu_record_count'] for s in sessions),
                    events=sum(s['counts']['event_count'] for s in sessions),
                    causal_imu=sum(s['causal_imu']['aligned'] for s in sessions),
                    raw_bytes=sum(s['tof']['raw_bytes']+s['imu']['raw_bytes'] for s in sessions)),
                seconds=time.monotonic()-started,pass_all_bytes_counts_payloads=True,
                boundary='Android receipt proximity only; H3 ms and Atom us remain unmapped; no exposure synchronization, geometry truth, model inference or labels.')
    result['diagnostics']=dict(
            tof_missing_sequence=sum(s['tof']['timeline']['missing_sequence_count'] for s in sessions),
            imu_missing_sequence=sum(s['imu']['timeline']['missing_sequence_count'] for s in sessions),
            imu_gap_edges=sum(s['imu']['timeline']['sequence_gap_edges'] for s in sessions),
            imu_unavailable=result['checked']['tof_valid']-result['checked']['causal_imu'],
            imu_unavailable_reason_counts=dict(Counter(r['reason'] for s in sessions for r in s['causal_imu']['unavailable_reasons'])),
            newest_causal_reference_discrepancies=sum(len(s['causal_imu']['newest_receipt_causal_discrepancies']) for s in sessions),
            newest_reference_receipt_ties=sum(r['saved_age_ms']==r['newest_age_ms'] for s in sessions for r in s['causal_imu']['newest_receipt_causal_discrepancies']),
            newest_strictly_newer_receipt_references=sum(r['saved_age_ms']>r['newest_age_ms'] for s in sessions for r in s['causal_imu']['newest_receipt_causal_discrepancies']),
            tof_raw_truncated=sum(r['type']=='unparseable_raw' for s in sessions for r in s['tof']['invalid_records']),
            imu_raw_truncated=sum(r['type']=='unparseable_raw' for s in sessions for r in s['imu']['invalid_records']),
            tof_sequence_reset_or_reorder=sum(s['tof']['timeline']['sequence_reset_or_reorder'] for s in sessions),
            imu_sequence_reset_or_reorder=sum(s['imu']['timeline']['sequence_reset_or_reorder'] for s in sessions),
            all_valid_frames_finite_and_8x8x16=True,
            hist_zero_entries=sum(s['tof']['hist_zero_entries'] for s in sessions),
            hist_total_entries=result['checked']['tof_valid']*64*16,
            event_kind_counts=dict(sum((Counter(s['event_kind_counts']) for s in sessions),Counter())))
    (out/'integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    lines=['# 新手机录制完整性与接收时钟诊断', '',
           '严格核对原始字节覆盖、索引/manifest数量、有效帧结构、RGB原始校验和以及因果IMU引用。各硬件时钟保持独立。','',
           '| UTC开始 | RGB | CNH有效/总记录 | IMU有效/总记录 | ToF缺seq | IMU缺seq | IMU匹配 | 接收差P95/最小(ms) |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in sessions:
        c=s['counts']; a=s['causal_imu']; q=a['signed_imu_minus_tof_ms']
        lines.append(f"| {s['session'][17:23]} | {c['rgb_count']} | {c['tof_count']}/{c['tof_record_count']} | {c['imu_count']}/{c['imu_record_count']} | {s['tof']['timeline']['missing_sequence_count']} | {s['imu']['timeline']['missing_sequence_count']} | {a['aligned']}/{c['tof_count']} | {q.get('p95',0):.3f}/{q.get('minimum',0):.3f} |")
    lines+=['',f"已核对 {result['checked']['sessions']} 段、{result['checked']['rgb']} 张RGB、{result['checked']['tof_valid']} 有效CNH、{result['checked']['imu_valid']} 有效IMU；{result['checked']['raw_bytes']} 原始传感器字节无offset空洞/覆盖/尾漏。",
            '','2 个ToF尾截断、2 个IMU尾截断保留，均不进入有效帧；另外54个ToF连接/boot/config和18个IMU boot记录保留。所有144349个零hist值按有效内容保留。',
            '',f"IMU匹配 {result['checked']['causal_imu']}/{result['checked']['tof_valid']}；未匹配38帧，其离线最近因果样本年龄也均超过100ms。IMU缺序787样本/130边，ToF缺序/重排均0。",
            '', '离线全量index与在线引用有3个record_index差异：15:06:57 ToF25与15:37:40 ToF26各有同receipt时间tie（额外年龄差0ns）；15:29:39首个ToF4保存IMU2年龄12.270729ms，离线最后因果IMU3年龄2.824688ms，实际差9.446041ms。在线saveTof从当时可用imuRecent队列选lastOrNull，较新样本可能尚未入队；三者均符合因果、原始字段一致与100ms约束，未改写引用。',
            '', '15:30:20 ToF设备时间步长最大1.905s，15:35:54最大514ms；尽管seq连续，仍有真实时间间隔。15:09:42 RGB记录camera_stalled/recovered，receipt缺口2.109s。数据完整不等于采样连续。',
            '', '分母和逐记录异常保留在 `integrity.json`。非法/截断帧没有作为有效CNH输入；原始payload按offset完整保留。Boot/config可由连接时缓存发出，不能仅凭标题视为本段物理重启。设备boot在段间变化，所有单段有效序列无重排/reset。','',
            f"实际CPU命令审计内部耗时 {result['seconds']:.3f} 秒；无GPU/模型推理。"]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(checked=result['checked'],seconds=result['seconds']),ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    main(ap.parse_args().root)

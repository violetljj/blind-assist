"""Engineering completeness checks; does not run inference or change frozen scoring."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
import math
from pathlib import Path

CLASSES = {'calibration', 'quiet', 'ahead', 'head', 'side', 'unknown'}
JUDGED = {'quiet', 'ahead', 'head', 'side'}
REQUIRED = ('tof.raw.jsonl', 'tof.index.jsonl', 'imu.index.jsonl', 'rgb.jsonl')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    # Match frozen v1's UTF-8 reader: accepting a BOM here would give a false PASS.
    text = path.read_text(encoding='utf-8')
    if text.startswith('\ufeff'):
        raise ValueError(f'{path.name}: UTF-8 BOM is unsupported by the frozen evaluator; save labels as UTF-8 without BOM before committing')
    return json.loads(text, object_pairs_hook=unique_object)


def read_index(path):
    rows = []
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=unique_object)
                if not isinstance(row, dict):
                    raise ValueError('expected a JSON object')
            except ValueError as exc:
                raise ValueError(f'{path.name}:{line_number}: {exc}; re-import the original export, do not silently drop the record') from exc
            rows.append((line_number, row))
    if not rows:
        raise ValueError(f'{path.name}: no index records')
    return rows


def integer(value, where, minimum=0):
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError(f'{where}: expected an integer in [{minimum}, 2^63)')
    return value


def numeric_array(value, shape, where, integral=False):
    if shape:
        if not isinstance(value, list) or len(value) != shape[0]:
            raise ValueError(f'{where}: expected array shape {shape}')
        for item in value:
            numeric_array(item, shape[1:], where, integral)
    elif type(value) not in (int, float) or not math.isfinite(value) or (integral and type(value) is not int):
        raise ValueError(f'{where}: expected finite {"integer" if integral else "numeric"} values')


def valid_index(rows, filename):
    valid = []
    for ordinal, (line, row) in enumerate(rows, 1):
        where = f'{filename}:{line}'
        if row.get('index') != ordinal or type(row.get('index')) is not int:
            raise ValueError(f'{where}: missing/duplicate/out-of-order record index; re-import the complete export')
        if type(row.get('valid_frame')) is not bool:
            raise ValueError(f'{where}: valid_frame must be boolean')
        if row['valid_frame']:
            integer(row.get('received_elapsed_ns'), f'{where} received_elapsed_ns')
            valid.append((line, row))
    return valid


def inspect_session(folder):
    """Validate only captured input bytes/indices; never imports an evaluator/model.

    Invalid transport/header/tail records remain indexed and are not CNH frames.
    Missing IMU or RGB receipt matches are reported, never filled or relabeled.
    """
    tof_rows = read_index(folder / 'tof.index.jsonl')
    imu_rows = read_index(folder / 'imu.index.jsonl')
    rgb_rows = read_index(folder / 'rgb.jsonl')
    tof_valid = valid_index(tof_rows, 'tof.index.jsonl')
    imu_valid = valid_index(imu_rows, 'imu.index.jsonl')
    if not tof_valid:
        raise ValueError('tof.index.jsonl: no valid CNH frames; this segment cannot be replayed')
    imu = {}
    for line, row in imu_valid:
        where = f'imu.index.jsonl:{line} gravity_unit'
        gravity = row.get('gravity_unit')
        # Schema 1 predates additive gravity; the frozen loader keeps NaN pose.
        # Presence of a valid IMU record and availability of gravity are distinct.
        if gravity is not None:
            numeric_array(gravity, (3,), where)
            if math.hypot(*gravity) == 0:
                raise ValueError(f'{where}: zero vector cannot define pose')
        imu[row['index']] = row

    cursor = 0
    raw_size = (folder / 'tof.raw.jsonl').stat().st_size
    with (folder / 'tof.raw.jsonl').open('rb') as raw:
        for line, row in tof_rows:
            where = f'tof.index.jsonl:{line}'
            offset = integer(row.get('raw_byte_offset'), f'{where} raw_byte_offset')
            length = integer(row.get('raw_byte_length'), f'{where} raw_byte_length', 1)
            if offset != cursor or offset + length > raw_size:
                raise ValueError(f'{where}: raw byte range is missing, overlapping or beyond tof.raw.jsonl; re-import the complete export')
            payload = raw.read(length)
            cursor += length
            if not row['valid_frame']:
                continue  # Preserve boot/config and recorded truncated transport tails.
            try:
                frame = json.loads(payload, object_pairs_hook=unique_object)
                if not isinstance(frame, dict) or frame.get('type') != 'cnh_frame' or [frame.get(k) for k in ('rows', 'cols', 'bins')] != [8, 8, 16]:
                    raise ValueError('expected an 8x8x16 cnh_frame')
                for key in ('hist_raw', 'hist_scaler'):
                    numeric_array(frame.get(key), (64, 16), key, key.endswith('scaler'))
                for key in ('ambient_raw', 'ambient_scaler', 'distance_mm', 'target_status'):
                    numeric_array(frame.get(key), (64,), key, key.endswith('scaler'))
                for values, scalers in zip(frame['hist_raw'] + [frame['ambient_raw']], frame['hist_scaler'] + [frame['ambient_scaler']]):
                    if not all(math.isfinite(math.ldexp(v, -s)) for v, s in zip(values, scalers)):
                        raise ValueError('decoded CNH/ambient values are not finite')
                for key, raw_key in (('sequence', 'seq'), ('device_timestamp_ms', 'ms')):
                    integer(frame.get(raw_key), raw_key)
                    if row.get(key) != frame[raw_key]:
                        raise ValueError(f'{key} does not match raw {raw_key}')
            except (ValueError, OverflowError) as exc:
                raise ValueError(f'{where}: invalid referenced CNH payload: {exc}') from exc
    if cursor != raw_size:
        raise ValueError(f'tof.raw.jsonl: {raw_size - cursor} bytes have no index record; re-import the complete export')

    missing_imu, missing_gravity = 0, 0
    for line, row in tof_valid:
        alignment = row.get('imu_alignment') or {}
        if not isinstance(alignment, dict):
            raise ValueError(f'tof.index.jsonl:{line}: imu_alignment must be an object or null')
        reference = alignment.get('imu_record_index')
        if reference is None:
            missing_imu += 1
        elif type(reference) is not int or reference not in imu:
            raise ValueError(f'tof.index.jsonl:{line}: IMU reference {reference!r} has no valid indexed IMU record')
        elif imu[reference].get('gravity_unit') is None:
            missing_gravity += 1

    rgb_times = []
    resolved_folder = folder.resolve()
    for line, row in rgb_rows:
        where = f'rgb.jsonl:{line}'
        rgb_times.append(integer(row.get('received_elapsed_ns'), f'{where} received_elapsed_ns'))
        filename = row.get('file')
        if not isinstance(filename, str) or not filename:
            raise ValueError(f'{where}: missing RGB file path')
        path = (folder / filename).resolve()
        if not path.is_relative_to(resolved_folder):
            raise ValueError(f'{where}: RGB path escapes this capture; re-import the original export')
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'{where}: missing/empty RGB payload {filename}; re-import the original export')
        payload = path.read_bytes()
        if 'bytes' in row and (type(row['bytes']) is not int or row['bytes'] != len(payload)):
            raise ValueError(f'{where}: RGB byte count differs for {filename}; re-import the original export')
        if 'sha256' in row and row['sha256'] != hashlib.sha256(payload).hexdigest():
            raise ValueError(f'{where}: RGB SHA256 differs for {filename}; re-import the original export')

    # Use only Android receipt clocks, as v1 does; this is not exposure sync.
    sorted_rgb_times = sorted(rgb_times)
    outside = 0
    for _, row in tof_valid:
        stamp = row['received_elapsed_ns']
        j = bisect_left(sorted_rgb_times, stamp)
        nearby = sorted_rgb_times[max(0, j - 1):j + 1]
        outside += min(abs(t - stamp) for t in nearby) > 100_000_000
    warnings = []
    if missing_imu:
        warnings.append(f'{missing_imu}/{len(tof_valid)} valid CNH frames have no IMU reference; frozen replay retains unavailable pose')
    legacy_imu = sum(row.get('gravity_unit') is None for row in imu.values())
    if legacy_imu:
        warnings.append(f'{legacy_imu}/{len(imu_valid)} valid IMU records have no gravity_unit (supported legacy schema); {missing_gravity}/{len(tof_valid)} CNH frames reference them and retain unavailable pose')
    if outside:
        warnings.append(f'{outside}/{len(tof_valid)} valid CNH frames have no RGB Android receipt within 100 ms; RGB branch unavailable if ready')
    diagnostics = dict(tof_records=len(tof_rows), valid_cnh_frames=len(tof_valid),
                       invalid_tof_records_retained=len(tof_rows) - len(tof_valid),
                       indexed_tof_bytes=cursor, valid_imu_records=len(imu_valid),
                       cnh_without_imu_reference=missing_imu, imu_without_gravity=legacy_imu,
                       cnh_without_usable_gravity=missing_imu + missing_gravity, rgb_records=len(rgb_rows),
                       rgb_payloads_checked=len(rgb_rows), cnh_without_rgb_receipt_match=outside)
    return diagnostics, warnings


def inspect_inputs(raw, sessions):
    errors, warnings, diagnostics = [], [], {}
    if not sessions:
        errors.append('no capture-* session directories found')
    for name in sorted(sessions):
        missing = False
        for filename in REQUIRED:
            path = raw / name / filename
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f'{name}: missing/empty {filename}')
                missing = True
        if not missing:
            try:
                diagnostics[name], notes = inspect_session(raw / name)
                warnings.extend(f'{name}: {note}' for note in notes)
            except (OSError, ValueError) as exc:
                errors.append(f'{name}: {exc}')
    return errors, warnings, diagnostics


def check_inputs_only(raw):
    sessions = {p.name for p in raw.iterdir() if p.is_dir() and p.name.startswith('capture-')}
    errors, warnings, diagnostics = inspect_inputs(raw, sessions)
    return dict(status='INCOMPLETE' if errors else 'INPUTS_ONLY_COMPLETE', sessions=len(sessions),
                errors=errors, warnings=warnings, input_diagnostics=diagnostics,
                labels_checked=False, evaluator_reports_checked=False,
                note='Input-only engineering check; labels, class coverage and scientific criteria were NOT checked. No image decoding or model inference.')


def check(raw, labels_path, reports=None):
    labels = read_json(labels_path)
    if not isinstance(labels, dict) or not labels:
        raise ValueError('labels must be a nonempty session-name object')
    sessions = {p.name for p in raw.iterdir() if p.is_dir() and p.name.startswith('capture-')}
    errors, warnings, input_diagnostics = inspect_inputs(raw, sessions & set(labels))
    if sessions != set(labels):
        errors.append(f'session mismatch: unlabeled={sorted(sessions - set(labels))}; missing={sorted(set(labels) - sessions)}')
    classes = set()
    for name, label in labels.items():
        if not isinstance(label, dict) or label.get('class') not in CLASSES:
            errors.append(f'{name}: invalid class'); continue
        classes.add(label['class'])
    for cls in ('ahead', 'quiet'):
        if cls not in classes:
            errors.append(f'no {cls} label: overall claim cannot be evaluated')
    if 'calibration' not in classes:
        warnings.append('no calibration segment: frozen evaluator uses default head height 1.35 m')
    summaries = {}
    if reports:
        versions = (1, 2, 3) if (reports / 'frozen_fusion_eval_v3.json').exists() else (1, 2)
        versions += (6,) if (reports / 'frozen_fusion_eval_v6.json').exists() else ()
        if 3 not in versions:
            warnings.append('v3 report absent: frozen v3 (registered 2026-10-13) was not run on this batch')
        for version in versions:
            report = read_json(reports / f'frozen_fusion_eval_v{version}.json')
            rows = report['sessions']
            if set(rows) != set(labels):
                errors.append(f'v{version}: report sessions differ from labels')
            for name, label in labels.items():
                row = rows.get(name)
                if not isinstance(label, dict) or row is None:
                    continue
                if row.get('cls') != label.get('class'):
                    errors.append(f'v{version} {name}: class differs from frozen label')
                if label.get('class') in JUDGED and (row.get('ready_frames', 0) <= 0 or not isinstance(row.get('pass'), bool)):
                    errors.append(f'v{version} {name}: not evaluable; must not be omitted from interpretation')
            summaries[f'v{version}'] = report.get('summary')
    return dict(status='INCOMPLETE' if errors else 'COMPLETE', sessions=len(labels), errors=errors,
                warnings=warnings, input_diagnostics=input_diagnostics, evaluator_summaries=summaries,
                note='Input/readout completeness only; COMPLETE does not mean scientific criteria passed, labels were committed, images decoded successfully, or device clocks synchronized.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--labels', type=Path, help='Check labels and input compatibility before frozen evaluation')
    mode.add_argument('--input-only', action='store_true', help='Check an imported capture batch before labeling; never checks scientific criteria')
    parser.add_argument('--reports', type=Path, help='Also check existing v1 AND v2 outputs (and v3 when present); never runs inference')
    args = parser.parse_args()
    if args.input_only and args.reports:
        parser.error('--reports requires --labels; --input-only cannot verify evaluation completeness')
    try:
        result = check_inputs_only(args.raw) if args.input_only else check(args.raw, args.labels, args.reports)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = dict(status='INCOMPLETE', errors=[str(exc)])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] in ('COMPLETE', 'INPUTS_ONLY_COMPLETE') else 2


if __name__ == '__main__':
    raise SystemExit(main())

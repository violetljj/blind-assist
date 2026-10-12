"""Engineering completeness checks; does not run inference or change frozen scoring."""
from __future__ import annotations

import argparse
import json
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
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_object)


def check(raw, labels_path, reports=None):
    labels = read_json(labels_path)
    if not isinstance(labels, dict) or not labels:
        raise ValueError('labels must be a nonempty session-name object')
    errors, warnings = [], []
    sessions = {p.name for p in raw.iterdir() if p.is_dir() and p.name.startswith('capture-')}
    if sessions != set(labels):
        errors.append(f'session mismatch: unlabeled={sorted(sessions - set(labels))}; missing={sorted(set(labels) - sessions)}')
    classes = set()
    for name, label in labels.items():
        if not isinstance(label, dict) or label.get('class') not in CLASSES:
            errors.append(f'{name}: invalid class'); continue
        classes.add(label['class'])
        if name not in sessions:
            continue
        for filename in REQUIRED:
            path = raw / name / filename
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f'{name}: missing/empty {filename}')
    for cls in ('ahead', 'quiet'):
        if cls not in classes:
            errors.append(f'no {cls} label: overall claim cannot be evaluated')
    if 'calibration' not in classes:
        warnings.append('no calibration segment: frozen evaluator uses default head height 1.35 m')
    summaries = {}
    if reports:
        for version in (1, 2):
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
                warnings=warnings, evaluator_summaries=summaries,
                note='Completeness only; COMPLETE does not mean scientific criteria passed or labels were committed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--reports', type=Path, help='Also check existing v1 AND v2 outputs; never runs inference')
    args = parser.parse_args()
    try:
        result = check(args.raw, args.labels, args.reports)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = dict(status='INCOMPLETE', errors=[str(exc)])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())

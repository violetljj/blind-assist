"""Explicit Atom-only IMU measurement; raw evidence and repeatable tilt summary."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import time
from atom_imu import ImuFeed, derive_mounting, unit, clock_metadata

ROOT = Path(__file__).resolve().parents[4]


def summarize(path):
    records = [json.loads(line) for line in (Path(path)/'receipts.jsonl').read_text(encoding='utf-8').splitlines()]
    frames = [r for r in records if r.get('valid_frame')]
    if not frames:
        return dict(status='NO_VALID_IMU_FRAMES', frames=0)
    first, last = frames[0]['sensor'], frames[-1]['sensor']
    domains = sorted(set(r.get('host_receipt_clock_domain', 'UNSPECIFIED') for r in frames))
    receipt_clock = clock_metadata() if domains == [clock_metadata()['domain']] else {'domains':domains,'domain':'UNSPECIFIED_OR_MIXED','provenance':'original per-record domains; no current clock inferred for old receipts'}
    elapsed = (last['device_us']-first['device_us'])/1e6
    stats = {}
    for key in ('pitch_deg', 'roll_deg'):
        values = [r['sensor'][key] for r in frames]
        beginning = [r['sensor'][key] for r in frames if r['sensor']['device_us']-first['device_us'] <= 5_000_000]
        ending = [r['sensor'][key] for r in frames if last['device_us']-r['sensor']['device_us'] <= 5_000_000]
        stats[key] = dict(mean_deg=statistics.mean(values), sample_std_deg=statistics.stdev(values) if len(values)>1 else None,
                          min_deg=min(values), max_deg=max(values),
                          drift_last5s_minus_first5s_deg=statistics.mean(ending)-statistics.mean(beginning) if elapsed>10 else None,
                          drift_unavailable_reason='first/last 5s windows overlap when device span <=10s' if elapsed<=10 else None,
                          first5s_frames=len(beginning), last5s_frames=len(ending))
    gravity = unit([statistics.mean(r['sensor']['gravity_unit'][axis] for r in frames) for axis in range(3)])
    return dict(status='MEASURED', frames=len(frames), device_elapsed_seconds=elapsed,
                host_receipt_clock=receipt_clock,
                observed_sample_hz=(len(frames)-1)/elapsed if elapsed>0 else None,
                distinct_boot_ids=sorted(set(r['sensor']['boot_id'] for r in frames)),
                atom_device_id=first['device_id'], gravity_mean_unit=gravity, attitude=stats,
                sample_gap_frames=sum(r['sensor'].get('sample_gap') is True for r in frames),
                accel_correction_frames=sum(r['sensor'].get('accel_correction') is True for r in frames),
                evidence_scope='Real Atom sensor receipt statistics; native BMI270 tilt, optical mounting not inferred',
                drift_rule='last 5 device seconds mean minus first 5 device seconds mean')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Atom IMU采样与水平/俯角安装候选；不刷写、不连接ToF')
    commands = parser.add_subparsers(dest='command', required=True)
    record = commands.add_parser('record')
    record.add_argument('--host', required=True)
    record.add_argument('--port', type=int, default=3336)
    record.add_argument('--seconds', type=float, default=60)
    record.add_argument('--out', type=Path, required=True)
    record.add_argument('--phase', default='static')
    derive = commands.add_parser('derive')
    derive.add_argument('--level', type=Path, required=True, help='record目录的summary.json')
    derive.add_argument('--down', type=Path, required=True)
    derive.add_argument('--mounting-id', required=True)
    derive.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.out.resolve().is_relative_to((ROOT/'artifacts.local').resolve()):
        parser.error('输出必须在canonical artifacts.local内')
    if args.command == 'derive':
        level = json.loads(args.level.read_text(encoding='utf-8-sig'))
        down = json.loads(args.down.read_text(encoding='utf-8-sig'))
        if level['atom_device_id'] != down['atom_device_id']:
            parser.error('水平/下俯记录不是同一Atom')
        result = derive_mounting(level['gravity_mean_unit'], down['gravity_mean_unit'], atom_device_id=level['atom_device_id'],
                                 mounting_id=args.mounting_id, evidence=f'{args.level.resolve()} ; {args.down.resolve()}')
        result['level_reference']['summary_path'] = str(args.level.resolve())
        result['down_reference']['summary_path'] = str(args.down.resolve())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print('已保存未核验安装候选；需明确水平光轴、上仰方向及正对墙检查后才能verified=true。')
        return 0
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds必须是正数')
    feed = ImuFeed(args.host, args.out, args.port)
    try:
        feed.start()
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            time.sleep(min(.2, max(0, deadline-time.perf_counter())))
    finally:
        feed.close()
    result = summarize(args.out)
    result.update(phase=args.phase, requested_host_seconds=args.seconds, output=str(args.out.resolve()))
    (args.out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['frames'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

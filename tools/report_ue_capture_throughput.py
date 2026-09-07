"""Summarize measured UE depth transport; never equate probe time to production."""
import argparse
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(root):
    receipt = read(root/'receipt.json')
    if read(root/'completion.json')['status'] != 'PASS' or receipt['status'] != 'PASS':
        raise ValueError('Only completed acquisitions can be summarized')
    rows = receipt['depth_exports']
    times = receipt['actual_capture_monotonic_s']
    intervals = [b-a for a,b in zip(times,times[1:])]
    result = dict(frames=receipt['frame_count'], mode=rows[0]['mode'],
                  cadence=receipt.get('cadence','tick'),
                  script_wall_s=receipt['wall_elapsed_s'],
                  process_wall_s=read(root/'process-release.json')['wall_elapsed_s'],
                  steady_elapsed_s=times[-1]-times[0],
                  steady_fps=(len(times)-1)/(times[-1]-times[0]),
                  median_frame_s=statistics.median(intervals),
                  source_unchanged=receipt['source_unchanged'])
    result['settling_policy'] = receipt.get('settling_policy', 'full')
    if 'pair_exports' in receipt:
        result['pair_export'] = receipt['pair_exports']['mode']
        result['pair_profile'] = receipt['pair_exports']['profile']
        result['steady_including_drain_s'] = receipt['exports_drained_monotonic_s'] - times[0]
        result['steady_including_drain_fps'] = (len(times)-1) / result['steady_including_drain_s']
    if 'rgb_exports' in receipt:
        result['rgb_export'] = receipt['rgb_exports']['mode']
        result['rgb_profile'] = receipt['rgb_exports']['profile']
        result['steady_including_drain_s'] = receipt['exports_drained_monotonic_s'] - times[0]
        result['steady_including_drain_fps'] = (len(times)-1) / result['steady_including_drain_s']
    if 'exports_drained_monotonic_s' in receipt and receipt.get('profiles'):
        start = receipt.get('acquisition_started_monotonic_s', times[0] - receipt['profiles'][0]['total_s'])
        result['acquisition_including_warmup_and_drain_s'] = receipt['exports_drained_monotonic_s'] - start
        result['acquisition_including_warmup_and_drain_fps'] = len(times) / result['acquisition_including_warmup_and_drain_s']
        result['capture_time_semantics'] = receipt.get('capture_time_semantics', 'GPU_PAIR_SUBMISSION' if 'pair_exports' in receipt else 'SYNCHRONOUS_READBACK_COMPLETE')
    if rows[0]['mode'].endswith('probe') or rows[0]['mode']=='probe':
        result['production_throughput']=False
        modes=list(rows[0]['measurements'])
        result['median_export_s']={m:statistics.median(r['measurements'][m]['total_s'] for r in rows[1:]) for m in modes}
        if 'bytes_equal' in rows[0]:
            result['same_target_byte_parity']=all(r['bytes_equal'] for r in rows)
    else:
        result['production_throughput']=True
        result['median_export_s']=statistics.median(r['total_s'] for r in rows[1:])
    if receipt.get('pair_exports', {}).get('mode', '').endswith('probe'):
        result['production_throughput'] = False
    if receipt.get('rgb_exports', {}).get('mode', '').endswith('probe'):
        result['production_throughput'] = False
    if (root/'transport.json').is_file():
        transport=read(root/'transport.json')
        result['median_host_conversion_s']=statistics.median(r['conversion_s'] for r in transport['rows'])
        result['same_target_byte_parity']=all(r['bytes_equal'] is True for r in transport['rows']) if 'probe' in result['mode'] else None
        result['frames_converted']=transport['frames']
        result['bytes_per_frame']=statistics.mean(r['depth_bytes'] for r in transport['rows'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result={str(root):summarize(root) for root in args.roots}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

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
    if rows[0]['mode'].endswith('probe') or rows[0]['mode']=='probe':
        result['production_throughput']=False
        modes=list(rows[0]['measurements'])
        result['median_export_s']={m:statistics.median(r['measurements'][m]['total_s'] for r in rows[1:]) for m in modes}
        if 'bytes_equal' in rows[0]:
            result['same_target_byte_parity']=all(r['bytes_equal'] for r in rows)
    else:
        result['production_throughput']=True
        result['median_export_s']=statistics.median(r['total_s'] for r in rows[1:])
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

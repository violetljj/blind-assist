"""Bounded 20-layout engineering pilot; never launches scientific fits or tests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback

def require_benchmark_source(source):
    if source.get('scene_layer') != 'REALISTIC_MODULAR':
        raise ValueError('20-layout benchmark pilot requires realistic modular assets; primitive and legacy plans are unit-test fixtures only')
    # A role string must not authorize the existing primitive-only renderer.
    raise NotImplementedError('Realistic modular capture, dependency-disjoint asset manifest and distractor admission are not implemented yet; do not dispatch UE')


def run(spec, output, timeout):
    source=json.loads(Path(spec).read_text(encoding='utf-8-sig'))
    require_benchmark_source(source)
    from cnh_route_launch import launch, write
    from cnh_route_geometry import inspect
    from cnh_route_pack import pack
    rows=source['frames']
    if len(rows)!=3200 or len({r['layout_id'] for r in rows})!=20 or any(r['split']!='train' for r in rows):
        raise ValueError('Pilot requires exactly the frozen 20 training layouts and 3200 frames')
    output=Path(output);started=time.monotonic()
    report=dict(status='RUNNING',scope='ENGINEERING_20_TRAIN_LAYOUTS_NO_MODEL_OR_TEST_ACCESS',
        expected_frames=3200,expected_layouts=20,model_results='NOT_RUN',
        full_collection='NOT_STARTED',owner='cnh-route-comparison-20260924')
    try:
        report['capture']=launch(spec,output,timeout)
        report['geometry']=inspect(output,pilot_sample=True)
        report['archives']=pack(output)
        engine=json.loads((output/'engine-receipt.json').read_text())
        render_forecast=engine['wall_s']/3200*61440*1.3/3600
        report['render_forecast_gpu_hours_with30pct']=render_forecast
        report['observed_capture_gpu_wall_s']=report['capture']['editor_wall_s']
        report['native_sample_rate_hz']=3200/engine['wall_s']
        checks=dict(all_native_passes=report['capture']['frame_count']==3200,
            geometry=report['geometry']['status']=='PASS_GEOMETRIC_CANARY',
            lossless_archives=report['archives']['status']=='PASS_LOSSLESS_ARCHIVES_VERIFIED_ORIGINALS_RETAINED',
            render_budget=render_forecast<=24)
        report['checks']=checks
        report['status']='ENGINEERING_CHECKS_RECORDED_PENDING_SENSOR_INTEGRATION_AND_STORAGE_REVIEW' if all(checks.values()) else 'ENGINEERING_GATE_FAILED'
        report['next_action']='Review archive forecast and actual-mesh sensor integration before any full collection; no automatic budget expansion'
    except BaseException:
        report.update(status='FAIL',error=traceback.format_exc())
        raise
    finally:
        report['total_wall_s']=time.monotonic()-started
        if output.exists():
            write(output/'pilot-engineering-report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout-sec',type=float,default=10800)
    a=p.parse_args();print(json.dumps(run(a.spec,a.output,a.timeout_sec),indent=2))

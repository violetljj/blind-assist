"""Create a spatial camera sequence on an unchanged native City Sample map.

This is settled static-world capture, not a real-time pedestrian simulation.
Waypoints are x,y,z camera positions in metres; z must be checked against native
floor probes. Only existing map actors are loaded, with no inserted obstacles.
"""
import argparse
import json
import math
from pathlib import Path

from run_city_pcg_capture import artifact_file
from ue_native_capture import REPO, file_hash


def waypoint(value):
    row = [float(v) for v in value.split(',')]
    if len(row) != 3 or not all(math.isfinite(v) for v in row):
        raise argparse.ArgumentTypeError('Expected three finite metre coordinates x,y,z')
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--waypoint', type=waypoint, action='append', required=True)
    p.add_argument('--step-m', type=float, default=.5)
    p.add_argument('--region-margin-m', type=float, default=25.)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if len(args.waypoint) < 2 or not math.isfinite(args.step_m) or args.step_m <= 0:
        p.error('At least two waypoints and a positive finite step are required')
    if not math.isfinite(args.region_margin_m) or args.region_margin_m <= 0:
        p.error('Region margin must be positive and finite')
    project = artifact_file(args.project, 'Project')
    map_file = artifact_file(project.parent/'Content/Map/Small_City_LVL.umap', 'Map')
    output = args.output.resolve()
    if not output.is_relative_to((REPO/'artifacts.local').resolve()) or output.exists():
        p.error('Output must be a new file under artifacts.local')
    cases, distance = [], 0.
    for segment, (a, b) in enumerate(zip(args.waypoint, args.waypoint[1:])):
        length = math.dist(a, b)
        if math.hypot(b[0]-a[0], b[1]-a[1]) <= 0:
            p.error('Each segment must have horizontal movement')
        steps = math.ceil(length/args.step_m)
        yaw = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
        for i in range(steps + (segment == len(args.waypoint)-2)):
            position = [x+(y-x)*i/steps for x, y in zip(a, b)]
            cases.append(dict(name=f'route_{len(cases):04d}',
                camera=dict(zip(('x', 'y', 'z'), position), pitch=-5., yaw=yaw, roll=0.),
                route_distance_m=distance+length*i/steps, objects=[], probe_native_floor=True))
        distance += length
    points = args.waypoint
    spec = dict(map_asset='/Game/Map/Small_City_LVL', map_file=str(map_file),
        map_sha256=file_hash(map_file), cases=cases,
        world_partition_region_m=dict(
            min=[min(v[k] for v in points)-margin for k, margin in enumerate((args.region_margin_m, args.region_margin_m, 5))],
            max=[max(v[k] for v in points)+margin for k, margin in enumerate((args.region_margin_m, args.region_margin_m, 30))]),
        settling_ticks=32, first_use_settling_ticks=64, settling_interval_s=0.,
        native_full_detail_only=True,
        pair_export_mode='native_probe', route=dict(length_m=distance, waypoints_m=points,
            max_step_m=args.step_m, timing='SPATIAL_SEQUENCE_NO_SIMULATED_TIME_OR_DYNAMIC_ACTORS'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(spec=str(output), frames=len(cases), distance_m=distance)))


if __name__ == '__main__':
    main()

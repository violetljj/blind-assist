"""Describe head motion in two HoloSet passes. All offsets are relative to the first low-texture alert in each recording."""
import argparse
import json
from pathlib import Path

import numpy as np

SEQUENCES = ("research-center-seq3", "research-center-seq4")
SECOND = 10_000_000


def measure(source, name, cases):
    folder = source / name
    first = min(row["rgb_ts"] for row in cases)
    path = next(folder.glob("*pv.txt"))
    observations = []
    for line in path.read_text().splitlines()[1:]:
        raw = line.split(",")
        t = (int(raw[0]) - first) / SECOND
        if not -1.5 <= t <= 6.5:
            continue
        pose = np.array(raw[3:], dtype=float).reshape(4, 4)
        rotation = pose[:3, :3]
        if not (np.allclose(rotation.T @ rotation, np.eye(3), atol=3.e-5)
                and np.isclose(np.linalg.det(rotation), 1., atol=3.e-5)):
            raise ValueError(f"Invalid pose near passage {raw[0]}")
        forward = -rotation[:, 2]
        pitch = float(np.degrees(np.arcsin(np.clip(forward[1], -1, 1))))
        observations.append([t, *pose[:3, 3], pitch])
    t, x, y, z, pitch = np.array(observations).T
    # Use only independently supported local floors from the registered depth screen.
    floor_pairs = []
    for line in (folder / "geometry.jsonl").read_text().splitlines():
        row = json.loads(line)
        dt = (row["rgb_ts"] - first) / SECOND
        if -2 <= dt <= 8 and "floor_y" in row:
            floor_pairs.append((dt, row["floor_y"]))
    floor_t, floor_y = np.asarray(sorted(floor_pairs)).T
    t_grid = np.arange(-1., 6.001, .1)
    xz = np.column_stack([np.interp(t_grid, t, v) for v in (x, z)])
    delta = xz - xz[10]           # chord between t=0 and 6 s
    length = np.linalg.norm(delta[-1])
    forward = delta[-1] / length
    side = np.array([forward[1], -forward[0]])
    lateral = delta @ side
    speeds = np.linalg.norm(xz[15:] - xz[10:-5], axis=1) / .5
    progress = np.diff(delta[10:] @ forward)
    eye = np.interp(t_grid, t, y) - np.interp(t_grid, floor_t, floor_y)
    start_height = float(np.median(eye[:11]))
    # Suppress single-pose noise and part of ordinary stride bobbing.
    med_t = np.arange(0., 4.51, .1)
    med_eye = np.array([np.median(eye[np.abs(t_grid-q) <= .21]) for q in med_t])
    dip = int(np.argmin(med_eye))
    lower = float(med_eye[dip])
    pre_pitch = float(np.median(pitch[(t >= -1.) & (t <= 0.)]))
    min_pitch = float(np.min(pitch[(t >= 0.) & (t <= 4.5)]))
    return dict(
        first_alert_rgb_ts=first, alert_relative_s=[(r["rgb_ts"]-first)/SECOND for r in cases],
        pv_pose_count=len(t), largest_pose_gap_s=float(np.max(np.diff(t))),
        floor_estimates=len(floor_y), floor_span_m=float(np.ptp(floor_y)),
        max_gap_between_floor_estimates_s=float(np.max(np.diff(floor_t))),
        displacement_0_to_6s_m=float(length), horizontal_path_0_to_6s_m=float(np.linalg.norm(np.diff(xz[10:],axis=0),axis=1).sum()),
        min_horizontal_speed_halfsecond_mps=float(speeds.min()), median_horizontal_speed_halfsecond_mps=float(np.median(speeds)),
        min_forward_displacement_per_100ms_m=float(np.min(progress)),
        max_side_departure_from_chord_m=float(np.max(np.abs(lateral[10:]))),
        side_departure_time_s=float(t_grid[10+np.argmax(np.abs(lateral[10:]))]),
        pre_camera_height_m=start_height, low_camera_height_m=lower,
        low_camera_time_s=float(med_t[dip]), drop_m=start_height-lower,
        post_camera_height_m=float(np.median(eye[t_grid >= 4.5])),
        pre_camera_pitch_deg=pre_pitch, most_downward_pitch_deg=min_pitch,
        flanking_floor_levels_m=floor_y.tolist(),
        profile_at_10hz=dict(relative_s=np.round(t_grid, 1).tolist(),
                             world_x_m=xz[:, 0].tolist(), world_z_m=xz[:, 1].tolist(),
                             camera_height_above_floor_m=eye.tolist(),
                             camera_pitch_deg=np.interp(t_grid, t, pitch).tolist(),
                             side_departure_from_chord_m=lateral.tolist()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("head_data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = json.loads((args.head_data / "alert-review/cases.json").read_bytes())
    result = {s: measure(args.head_data, s, [r for r in cases if r["seq"] == s]) for s in SEQUENCES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

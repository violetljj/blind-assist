"""Attribute recorded native-depth alarms against the published, asynchronous CAB scan.

Run from the repository root with the original LaMAR replay environment.
The independent map is an offline diagnostic only.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from arkit_selective_heading_dev_20261116 import BANDS, FAR, HALF_WIDTH, MIN_SOLID_ANGLE, NEAR
from lamar_native_corridor_dev_20261116 import DepthSensor, episode_pairs, walking_plane
from lamar_navvis_corridor_dev_20261116 import MAX_RANGE, NavVis


WORK = Path("artifacts.local/work")
SOURCE = WORK / "lamar-independent-heading-dev-20261116-codex"
SCREEN = SOURCE / "screen-hololens"
CAB = SOURCE / "hololens/CAB"
KINDS = ("head_occupied_travel_empty", "other_clear")
ARMS = ("original", "world_gravity", "remove_wall_compatible", "remove_lateral_reference",
        "remove_lateral_or_unmapped")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def episodes():
    comparison = read(SCREEN / "causal_comparison.json")
    pred = {(p["session"], p["timestamp_us"]): p["baseline"] for p in comparison["predictions"]}
    visits = read(SCREEN / "summary.json")["visits"]
    later = read(WORK / "lamar-motion-gate-dev-20261116-codex/whole_alert_case_check.json")
    removed = {(e["session"], tuple(ts)) for e in later["removed_episodes"] for ts in e["timestamps"]}
    totals = Counter()
    events = []
    for v in visits:
        session = v["visit"]
        for kind in KINDS:
            for ep in episode_pairs(v["rows"], kind):
                if not all((session, r["timestamp"]) in pred for r in ep):
                    continue
                totals[kind] += 1
                firing = []
                for r in ep:
                    bands = pred[session, r["timestamp"]]
                    if any(bands):
                        rec = {k: r[k] for k in ("timestamp", "depth_row", "direction_deg", "head",
                                                "travel", "head_robust", "travel_robust")}
                        rec["baseline"] = bands
                        firing.append(rec)
                if firing:
                    ts = tuple(r["timestamp"] for r in ep)
                    events.append(dict(session=session, category=kind, timestamps=ts, alarms=firing,
                                       later_gate_episode_disappears=(session, ts) in removed))
    assert dict(totals) == dict(head_occupied_travel_empty=20, other_clear=34)
    assert len(events) == 20 and sum(e["later_gate_episode_disappears"] for e in events) == 3
    return totals, events


def compact_mesh(centres):
    """Preserve every triangle that can be hit within eight metres of an audited ray origin."""
    with np.load(SCREEN / "cropped_mesh.npz") as mesh:
        vertices, faces = mesh["vertices"], mesh["faces"]
    chosen = []
    for start in range(0, len(faces), 120_000):
        part = faces[start:start + 120_000]
        points = vertices[part]
        lo, hi = points.min(axis=1), points.max(axis=1)
        keep = np.zeros(len(part), dtype=bool)
        for centre in centres:
            keep |= ((lo <= centre + MAX_RANGE + .02) & (hi >= centre - MAX_RANGE - .02)).all(axis=1)
        chosen.append(part[keep])
    indices, inverse = np.unique(np.concatenate(chosen), return_inverse=True)
    return vertices[indices], inverse.reshape(-1, 3).astype(np.int32)


def in_band(p, band):
    low, high = band
    return ((p[:, 0] >= NEAR) & (p[:, 0] <= FAR) & (np.abs(p[:, 1]) <= HALF_WIDTH)
            & (-p[:, 2] >= low) & (-p[:, 2] < high))


def load_session(session):
    keys = ("T_wc", "T_local", "ts", "local_down", "component", "depth_index", "depth_timestamp_us")
    with np.load(CAB / "prepared" / f"{session}.npz") as data:
        local = {k: data[k] for k in keys}
        sensor = DepthSensor(data["depth_lut"], data["T_depth_to_head"])
    return local, sensor


def raw_frame(data, sensor, row, image, scan, vertices, faces):
    i = data["depth_index"][row]
    assert i >= 0
    plane = walking_plane(data, i)
    local_head = np.eye(4)
    local_head[:3, :3] = plane @ data["T_local"][i, :3, :3]
    local_pose = local_head @ sensor.head_from_depth
    head_world = data["T_wc"][i]
    world_pose = head_world @ sensor.head_from_depth
    forward = head_world[:, 2][:2].copy()
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0], 0.])
    world_axes = np.array([[*forward, 0.], left, [0., 0., 1.]])
    native = image.astype(np.float32) * .001
    valid = (sensor.valid(native) & np.isfinite(sensor.weights))[::2, ::2]
    rays = sensor.rays[::2, ::2][valid]
    weights = 4. * sensor.weights[::2, ::2][valid]
    depth = native[::2, ::2][valid]
    local_ray = rays @ local_pose[:3, :3].T
    native_xyz = local_ray * depth[:, None] + local_pose[:3, 3]
    world_ray = rays @ world_pose[:3, :3].T
    world_xyz = world_ray * depth[:, None] + (world_pose[:3, 3] - head_world[:3, 3])
    world_xyz = world_xyz @ world_axes.T
    outputs = []
    for band in BANDS:
        hit = in_band(native_xyz, band)
        rays_world = np.ascontiguousarray(world_ray[hit], dtype=np.float32)
        w = weights[hit]
        wall = lateral = unknown = np.zeros(len(w), dtype=bool)
        if len(w):
            ranges = scan.distance(world_pose[:3, 3], rays_world)
            origins = np.broadcast_to(world_pose[:3, 3], rays_world.shape).astype(np.float32)
            face = scan.scene.run(origins, rays_world, query="INTERSECT")
            good = (ranges > .08) & (ranges < MAX_RANGE) & (face >= 0)
            ref = local_pose[:3, 3] + local_ray[hit] * np.where(good, ranges, 0.)[:, None]
            lateral = good & (np.abs(ref[:, 1]) > HALF_WIDTH)
            unknown = ~good
            normal = np.zeros_like(ref)
            tri = vertices[faces[face[good]]]
            cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            normal[good] = cross / np.maximum(1.e-12, np.linalg.norm(cross, axis=1, keepdims=True))
            low, high = band
            wall = (lateral & (np.abs(normal @ left) >= np.cos(np.deg2rad(30.)))
                    & (ref[:, 0] >= NEAR) & (ref[:, 0] <= FAR)
                    & (-ref[:, 2] >= low) & (-ref[:, 2] < high)
                    & (np.abs(ranges - depth[hit]) <= np.maximum(.10, .05 * ranges)))
        values = dict(original=w.sum(), world_gravity=weights[in_band(world_xyz, band)].sum(),
                      remove_wall_compatible=w[~wall].sum(), remove_lateral_reference=w[~lateral].sum(),
                      remove_lateral_or_unmapped=w[~(lateral | unknown)].sum())
        outputs.append(dict(support_pixels=len(w), wall_compatible_pixels=int(wall.sum()),
                            lateral_reference_pixels=int(lateral.sum()), unmapped_pixels=int(unknown.sum()),
                            solid_angle_sr={k: float(v) for k, v in values.items()},
                            votes={k: bool(v >= MIN_SOLID_ANGLE) for k, v in values.items()}))
    return outputs


def main(out):
    totals, events = episodes()
    audited = [e for e in events if e["category"] == "other_clear"]
    sessions = {s: load_session(s) for s in sorted({e["session"] for e in audited})}
    selected = {}
    centres = []
    for session, (data, sensor) in sessions.items():
        rows = sorted({r["depth_row"] - offset for e in audited if e["session"] == session
                       for r in e["alarms"] for offset in (2, 1, 0)})
        selected[session] = rows
        centres.extend((data["T_wc"][data["depth_index"][rows]] @ sensor.head_from_depth)[:, :3, 3])
    vertices, faces = compact_mesh(centres)
    scan = NavVis(vertices, faces)
    observations = {}
    for session, rows in selected.items():
        data, sensor = sessions[session]
        with np.load(CAB / "native_depth" / f"{session}.npz") as archive:
            ids, times = archive["depth_rows"], archive["timestamp_us"]
            assert np.array_equal(data["depth_timestamp_us"][ids], times)
            images = archive["depth_mm"]
            for row in rows:
                ix, = np.flatnonzero(ids == row)
                observations[session, row] = raw_frame(data, sensor, row, images[ix], scan, vertices, faces)
    for e in events:
        for a in e["alarms"]:
            a["any_matching_positive_head_band"] = any(f and t == 1 for f, t in zip(a["baseline"], a["head"]))
            if e["category"] != "other_clear":
                continue
            fr = [observations[e["session"], a["depth_row"] - offset] for offset in (2, 1, 0)]
            votes = {arm: [sum(frame[b]["votes"][arm] for frame in fr) >= 2 for b in range(2)]
                     for arm in ARMS}
            assert votes["original"] == a["baseline"]
            a["three_frame_output"] = votes
            a["frames_oldest_first"] = fr
    direction = [a for e in events if e["category"] == KINDS[0] for a in e["alarms"]]
    later_cases = [e for e in audited if not e["later_gate_episode_disappears"]]
    summary = dict(
        same_band_head_positive_timestamps=[sum(a["any_matching_positive_head_band"] for a in direction),
                                             len(direction)],
        halfwidth_robust_same_band_head_positive_timestamps=sum(
            any(f and t == 1 for f, t in zip(a["baseline"], a["head_robust"])) for a in direction),
        other_clear_episodes_erased={
            arm: sum(all(not any(a["three_frame_output"][arm]) for a in e["alarms"]) for e in audited)
            for arm in ARMS},
        surviving_later_gate_other_clear_episodes=len(later_cases),
        surviving_later_gate_other_clear_episodes_all_alarms_without_current_vote=sum(
            all(not any(b["votes"]["original"] for b in a["frames_oldest_first"][-1]) for a in e["alarms"])
            for e in later_cases))
    protocol = dict(
        source=str(SCREEN), native=str(CAB / "native_depth"), mesh="2022-06-21 CAB NavVis",
        near_far_m=[NEAR, FAR], bands_down_m=BANDS, half_width_m=HALF_WIDTH,
        threshold_sr=MIN_SOLID_ANGLE, reference_hit_m=[.08, MAX_RANGE],
        strict_wall_compatible="Same native ray hits map outside lateral corridor, inside same forward and "
            "height limits, with normal within 30 degrees of either lateral axis and range difference "
            "<= max(0.10 m, 5% of map range). This is a geometric proxy with no semantic or temporal identity.",
        world_gravity="Use unchanged native depth and published poses, with world vertical in place of "
            "tracked gravity. The map and the subsequent trajectory are never used by the original alert.",
        unit="Existing retrospective episodes, not independent people or collisions. Only the five "
            "other_clear baseline false alarms receive the ray intervention.")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ray_attribution.json").open("w", encoding="utf-8") as f:
        json.dump(dict(protocol=protocol, summary=summary, travel_clear_paired_episodes=totals,
                       audit_arms=ARMS, events=events), f, indent=2)
    for e in audited:
        print(e["session"], [(a["timestamp"], a["three_frame_output"]) for a in e["alarms"]])
    print("Same-band positive head ref", sum(any(a["any_matching_positive_head_band"] for a in e["alarms"])
          for e in events if e["category"] == KINDS[0]), "of 15 head-occupied travel-clear false alarms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORK / "sidewall-attribution-20261118/lamar")
    main(parser.parse_args().out)

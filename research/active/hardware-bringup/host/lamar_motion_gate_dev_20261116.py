"""Diagnose causal direction and native free-space evidence on the consumed LaMAR clips."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from arkit_selective_heading_dev_20261116 import (
    HALF_WIDTH, MARGIN, MIN_SOLID_ANGLE, angle_deg, in_corridor, samples,
)
from lamar_native_corridor_dev_20261116 import (
    DepthSensor, episode_pairs, evaluate_pass, predictions, reliable_direction, walking_plane,
)


HEAD = np.array([1., 0.])
ORIGIN = np.zeros(3)
MOTION_VALIDITY = {"timestamp", "component", "speed"}
TARGET = "head_occupied_travel_empty"
POSITIVE = "travel_occupied"
CLEAR = "other_clear"
LOCAL_KEYS = ("ts", "T_local", "component", "local_down", "depth_index", "depth_timestamp_us")


def signed_yaw(vector):
    return float(np.rad2deg(np.arctan2(vector[1], vector[0])))


def motion_facts(data, index, plane):
    times, poses, components = (data[key] for key in ("ts", "T_local", "component"))
    targets = times[index] - np.array([.8, .4])
    past = np.searchsorted(times[:index], targets, side="right") - 1
    if (past < 0).any() or (targets - times[past] > .07).any():
        return dict(failed=["timestamp"])
    window = slice(past[0], index + 1)
    if (components[window] != components[index]).any() or np.diff(times[window]).max() > .22:
        return dict(failed=["component"])
    picks = np.r_[past, index]
    chords = np.diff(poses[picks, :3, 3], axis=0) @ plane[:2].T
    speeds = np.linalg.norm(chords, axis=1) / np.diff(times[picks])
    yaw_long = signed_yaw(chords.sum(axis=0))
    yaw_short = signed_yaw(chords[-1])
    bend = angle_deg(*chords)
    reasons = (("speed", speeds.min() < .25), ("angle_low", abs(yaw_long) < 7.5),
               ("angle_high", abs(yaw_long) > 30.), ("inconsistent", bend > 15.))
    return dict(failed=[reason for reason, failed in reasons if failed], yaw08=yaw_long, yaw04=yaw_short,
                past_bend_deg=float(bend), speeds_mps=speeds.tolist())


def heading_evidence(sensor, observations, yaw):
    rotation = np.deg2rad(yaw)
    direction = np.array([np.cos(rotation), np.sin(rotation)])
    width = HALF_WIDTH + MARGIN
    target = samples(ORIGIN, direction, width)
    votes = [in_corridor(cloud, ORIGIN, direction, width)[0] for _, _, cloud in observations]
    seen = [sensor.clear_mask(target, image, pose) for image, pose, _ in observations]
    return dict(current_hit=(votes[-1] > 0).tolist(), history_hit=(np.max(votes, axis=0) > 0).tolist(),
                current_clear=seen[-1].mean(axis=1).tolist(),
                history_clear=np.logical_or.reduce(seen).mean(axis=1).tolist())


def view_history(data, sensor, recorder, depth_row):
    pose_index = data["depth_index"][depth_row]
    plane = walking_plane(data, pose_index)
    local_poses = data["T_local"]
    component = data["component"][pose_index]
    origin = local_poses[pose_index, :3, 3]
    views = []
    for i in range(depth_row - 2, depth_row + 1):
        past = data["depth_index"][i]
        assert 0 <= past <= pose_index and data["component"][past] == component
        relative = np.eye(4)
        relative[:3, :3] = plane @ local_poses[past, :3, :3]
        relative[:3, 3] = plane @ (local_poses[past, :3, 3] - origin)
        sensor_pose = relative @ sensor.head_from_depth
        depth = recorder[i].astype(np.float32) * .001
        views.append((depth, sensor_pose, sensor.cloud(depth, sensor_pose)))
    return pose_index, plane, views


def native_features(data, sensor, images, fixed):
    stamps = data["depth_timestamp_us"]
    result = {}
    for depth_row, (baseline, candidate, stage) in fixed.items():
        pose_index, plane, views = view_history(data, sensor, images, depth_row)
        motion = motion_facts(data, pose_index, plane)
        strict = reliable_direction(data, pose_index, plane)
        assert (strict is None) == bool(motion["failed"])
        if strict is not None:
            assert abs(signed_yaw(strict) - motion["yaw08"]) < 1e-10
        occupied, near = in_corridor(views[-1][2], ORIGIN, HEAD, HALF_WIDTH)
        eligible = baseline & (occupied >= MIN_SOLID_ANGLE) & (near < MIN_SOLID_ANGLE)
        evidence = {}
        if eligible.any() and not (MOTION_VALIDITY & set(motion["failed"])):
            for arm in ("yaw08", "yaw04"):
                evidence[arm] = heading_evidence(sensor, views, motion[arm])
        result[int(stamps[depth_row])] = dict(depth_row=depth_row, baseline=baseline.tolist(),
                                              old_candidate=candidate.tolist(), old_stage=stage,
                                              eligible=eligible.tolist(), motion=motion, evidence=evidence)
    return result


def future_yaw(pose_world, index, future):
    head = pose_world[index, :2, 2]
    travel = pose_world[future, :2, 3] - pose_world[index, :2, 3]
    return signed_yaw([head @ travel, head[0] * travel[1] - head[1] * travel[0]])


def attach_reference(visit, features, prepared, data, sensor, images):
    with np.load(prepared, allow_pickle=False) as source:
        world, times = source["T_wc"], source["timestamp_us"]
    row_id = {int(t): i for i, t in enumerate(times)}
    joined = []
    for label in visit["rows"]:
        key = int(label["timestamp"])
        if key not in features:
            continue
        row = features[key]
        reference = future_yaw(world, row_id[key], row_id[label["future_stamp"]])
        assert abs(abs(reference) - label["direction_deg"]) <= .001
        # This direction is used only after the native features have been produced.
        if any(row["eligible"]):
            _, _, views = view_history(data, sensor, images, row["depth_row"])
            row["evidence"]["future"] = heading_evidence(sensor, views, reference)
        joined.append(dict(label, **row, future_yaw=reference))
    return dict(visit=visit["visit"], rows=joined)


def direction_stats(rows):
    available = [row for row in rows if "yaw08" in row["motion"]]
    reference = np.array([r["future_yaw"] for r in available])
    result = dict(total=len(rows), paired=len(available), models={})
    for name in ("head", "yaw04", "yaw08"):
        yaw = np.zeros(len(available)) if name == "head" else np.array([r["motion"][name] for r in available])
        error = np.abs((yaw - reference + 180.) % 360. - 180.)
        if name == "head":
            base_error = error
        if len(error):
            result["models"][name] = dict(median_deg=float(np.median(error)), p90_deg=float(np.quantile(error, .9)),
                                           within_10deg=int((error <= 10.).sum()),
                                           closer_than_head=int((error + 1e-9 < base_error).sum()),
                                           worsens_gt5deg=int((error > base_error + 5.).sum()))
    return result


def arms():
    cases = [dict(name="fixed", direction="yaw08", view="current", coverage=.95, motion="fixed"),
             dict(name="raise_angle_ceiling", direction="yaw08", view="current", coverage=.95, motion="drop_upper")]
    for direction in ("yaw08", "yaw04", "future"):
        for view in ("current", "history"):
            for coverage in (.95, .75, 0.):
                cases.append(dict(name=f"{direction}_{view}_{coverage:.2f}", direction=direction, view=view,
                                  coverage=coverage, motion="none"))
    return cases


def allowed_to_veto(row, arm):
    failed = set(row["motion"]["failed"])
    if arm["motion"] == "fixed" and failed:
        return np.zeros(2, dtype=bool)
    if arm["motion"] == "drop_upper" and failed - {"angle_high"}:
        return np.zeros(2, dtype=bool)
    evidence = row["evidence"].get(arm["direction"])
    if evidence is None:
        return np.zeros(2, dtype=bool)
    hit = np.asarray(evidence[arm["view"] + "_hit"])
    fraction = np.asarray(evidence[arm["view"] + "_clear"])
    veto = np.asarray(row["eligible"]) & ~hit & (fraction >= arm["coverage"])
    if arm.get("whole_alert") and (np.asarray(row["baseline"]) & ~veto).any():
        return np.zeros(2, dtype=bool)
    return veto


def first_alert_delays(geometry, scored):
    delays = []
    for visit in geometry["visits"]:
        observed = scored.get(visit["visit"], {})
        for episode in episode_pairs(visit["rows"], POSITIVE):
            if not all(row["depth_row"] in observed for row in episode):
                continue
            first_base = first_candidate = None
            for row in episode:
                truth = np.asarray(row["travel"]) == 1
                baseline, candidate, _ = observed[row["depth_row"]]
                if first_base is None and (baseline & truth).any():
                    first_base = row["time"]
                if (candidate & truth).any():
                    first_candidate = row["time"]
                    break
            if first_base is not None and first_candidate is not None and first_candidate > first_base:
                delays.append(round(first_candidate - first_base, 6))
    return delays


def score(geometry, features, arm, robust=False):
    if robust:
        geometry = dict(visits=[dict(visit, rows=[dict(row, category=row["robust_category"], travel=row["travel_robust"])
                                                 for row in visit["rows"]]) for visit in geometry["visits"]])
    labels = {v["visit"]: {r["depth_row"]: r for r in v["rows"]} for v in geometry["visits"]}
    scored = {}
    changes = Counter()
    for session, rows in features.items():
        scored[session] = decisions = {}
        for row in rows:
            base = np.asarray(row["baseline"])
            veto = allowed_to_veto(row, arm)
            candidate = base & ~veto
            if arm["name"] == "fixed":
                np.testing.assert_array_equal(candidate, row["old_candidate"])
            label = labels[session][row["depth_row"]]
            decisions[row["depth_row"]] = (base, candidate, None)
            truth = np.asarray(label["travel"])
            changes["known_positive"] += int((veto & (truth == 1)).sum())
            changes["known_negative"] += int((veto & (truth == 0)).sum())
            changes["unknown"] += int((veto & (truth == -1)).sum())
    return dict(episodes=evaluate_pass(geometry, scored), bands=dict(changes),
                first_alert_delays_s=first_alert_delays(geometry, scored))


def episode_improvement(result):
    episodes = result["episodes"]
    return (episodes[TARGET]["lost_alert"], episodes[CLEAR]["lost_alert"])


def rank_if_supported(result):
    episodes = result["episodes"]
    if result["bands"].get("known_positive", 0) or episodes[POSITIVE]["lost_alert"]:
        return (0, 0)
    return episode_improvement(result)


def held_out(geometry, features, candidates):
    results = []
    for session in features:
        train = dict(visits=[v for v in geometry["visits"] if v["visit"] in features and v["visit"] != session])
        train_features = {k: v for k, v in features.items() if k != session}
        best = max(candidates, key=lambda arm: rank_if_supported(score(train, train_features, arm)))
        train_summary = score(train, train_features, best)
        if rank_if_supported(train_summary) == (0, 0):
            results.append(dict(held=session, choice="fixed", training_improvement=False))
            continue
        test = dict(visits=[v for v in geometry["visits"] if v["visit"] == session])
        results.append(dict(held=session, choice=best["name"], training_improvement=True,
                            test=score(test, {session: features[session]}, best)))
    return results


def motion_summary(rows):
    blocked = [r for r in rows if r["category"] == TARGET and any(r["baseline"]) and r["old_stage"] == "motion_gate"]
    false = [r for r in rows if r["category"] == TARGET and any(r["baseline"])]
    positive = [r for r in rows if r["category"] == POSITIVE]
    true = [r for r in positive if any(p and t == 1 for p, t in zip(r["baseline"], r["travel"], strict=True))]
    first = Counter(r["motion"]["failed"][0] for r in blocked)
    all_reasons = Counter(reason for row in blocked for reason in row["motion"]["failed"])
    assert len(blocked) == 27 and len(false) == 30
    return dict(first_fail=dict(first), all_fail=dict(all_reasons), failures=direction_stats(blocked),
                all_false_alerts=direction_stats(false), occupied=direction_stats(positive),
                correct_positive_alerts=direction_stats(true))


def run(args):
    start = perf_counter()
    previous = json.loads((args.source / "screen-hololens/causal_comparison.json").read_text(encoding="utf-8"))
    geometry = json.loads((args.source / "screen-hololens/summary.json").read_text(encoding="utf-8"))
    old = {(r["session"], int(r["timestamp_us"])): r for r in previous["predictions"]}
    native_root = args.source / "hololens/CAB"
    features = {}
    for visit in geometry["visits"]:
        session = visit["visit"]
        native = native_root / "native_depth" / f"{session}.npz"
        if not native.exists():
            continue
        prepared = native_root / "prepared" / f"{session}.npz"
        with np.load(prepared, allow_pickle=False) as source:
            local = {k: source[k] for k in LOCAL_KEYS}
            sensor = DepthSensor(source["depth_lut"], source["T_depth_to_head"])
        stamps = local["depth_timestamp_us"]
        fixed = {i: pred for i, pred in predictions(prepared, native).items() if (session, int(stamps[i])) in old}
        with np.load(native, allow_pickle=False) as recorded:
            images = dict(zip(recorded["depth_rows"], recorded["depth_mm"], strict=True))
        causal = native_features(local, sensor, images, fixed)
        for stamp, current in causal.items():
            prior = old[session, stamp]
            assert (current["baseline"], current["old_candidate"], current["old_stage"]) == (
                prior["baseline"], prior["candidate"], prior["gate"])
        joined = attach_reference(visit, causal, prepared, local, sensor, images)
        features[session] = joined["rows"]
        print(session, len(joined["rows"]), flush=True)
    rows = [r for visit in features.values() for r in visit]
    assert len(rows) == len(old) == 456
    motion = motion_summary(rows)
    policies = arms()
    comparisons = {}
    for arm in policies:
        nominal = score(geometry, features, arm)
        robust = score(geometry, features, arm, robust=True)
        comparisons[arm["name"]] = dict(arm=arm, nominal=nominal, robust=robust)
        episode = nominal["episodes"]
        print(arm["name"], episode[TARGET]["lost_alert"], episode[CLEAR]["lost_alert"],
              episode[POSITIVE]["lost_alert"], nominal["bands"], flush=True)
    eligible = [arm for arm in policies if arm["direction"] != "future"]
    transfer = held_out(geometry, features, eligible)
    result = dict(motion=motion, comparisons=comparisons, held_out=transfer, cases=features,
                  wall_seconds=round(perf_counter() - start, 3))
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)


def scan_clearance_thresholds(args):
    start = perf_counter()
    cached = json.loads(args.cached_features.read_text(encoding="utf-8"))
    geometry = json.loads((args.source / "screen-hololens/summary.json").read_text(encoding="utf-8"))
    features = cached["cases"]
    grid_size = samples(ORIGIN, HEAD, HALF_WIDTH + MARGIN).shape[1]
    fixed = arms()[:1]
    causal = []
    frontier = []
    for direction in ("yaw08", "yaw04", "future"):
        for view in ("current", "history"):
            these = [dict(name=f"{direction}_{view}_{count}of{grid_size}", direction=direction,
                          view=view, coverage=count / grid_size, required_sample_count=count, motion="none",
                          whole_alert=args.whole_alert)
                     for count in range(grid_size, -1, -1)]
            series = [dict(arm=arm, **score(geometry, features, arm)) for arm in these]
            admissible = [point for point in series if point["bands"].get("known_positive", 0) == 0]
            best = max(admissible, key=rank_if_supported, default=None)
            no_miss = [point for point in series if not point["episodes"][POSITIVE]["lost_alert"]]
            until_first_alert = [point for point in no_miss if not point["first_alert_delays_s"]]
            frontier.append(dict(direction=direction, view=view, series=series, best=best,
                                 best_no_episode_miss=max(no_miss, key=episode_improvement, default=None),
                                 best_no_episode_miss_or_delay=max(until_first_alert, key=episode_improvement, default=None)))
            if direction != "future":
                causal.extend(these)
            if best is not None:
                print(direction, view, best["arm"]["required_sample_count"], rank_if_supported(best), flush=True)
    transfer = held_out(geometry, features, fixed + causal)
    for fold in transfer:
        print(fold["held"], fold["choice"], rank_if_supported(fold["test"]) if fold["training_improvement"] else None, flush=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(dict(frontier=frontier, held_out=transfer, grid_size=grid_size, whole_alert=args.whole_alert,
                       wall_seconds=round(perf_counter() - start, 3)), out, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cached-features", type=Path, help="sweep all attainable clearance counts from saved features")
    parser.add_argument("--whole-alert", action="store_true", help="require evidence to cancel every active height band")
    args = parser.parse_args()
    if args.cached_features:
        scan_clearance_thresholds(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

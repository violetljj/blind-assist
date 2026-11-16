"""Small real-depth replay: current frame, 0.8 s history, and ray-cleared history."""

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from arkit_selective_heading_dev_20261116 import BANDS, FAR, HALF_WIDTH, MIN_SOLID_ANGLE, NEAR
from lamar_native_corridor_dev_20261116 import DepthSensor, episode_pairs, walking_plane


ARMS = ("single", "memory", "cleared")
HISTORY_S = 0.80
MAX_GAP_S = 0.32


@dataclass
class Frame:
    row: int
    time: float
    component: int
    cloud: np.ndarray
    weights: np.ndarray
    origin: np.ndarray
    plane: np.ndarray
    witnessed_free: Callable[[np.ndarray], np.ndarray]
    person: np.ndarray | None = None
    person_motion_at: np.ndarray | None = None


def follows(previous, current):
    return (current.row == previous.row + 1 and current.component == previous.component
            and 0 < current.time - previous.time <= MAX_GAP_S)


def corridor_bands(points, origin, plane):
    ahead, lateral, up = ((points - origin) @ plane.T).T
    down = -up
    inside = (np.abs(lateral) <= HALF_WIDTH) & (ahead >= NEAR) & (ahead <= FAR)
    return tuple(inside & (down >= low) & (down < high) for low, high in BANDS)


def replay(frames, keep_trace=False, person_history_s=None):
    if person_history_s is not None and not 0 <= person_history_s <= HISTORY_S:
        raise ValueError("Person history must be between zero and the original history duration")
    history = deque()
    recent_votes = deque(maxlen=3)
    result = {}
    previous = None
    for frame in frames:
        if previous is None or not follows(previous, frame):
            history.clear()
            recent_votes.clear()
        previous = frame
        while history and frame.time - history[0][0].time > HISTORY_S + 1e-8:
            history.popleft()

        now_bands = corridor_bands(frame.cloud, frame.origin, frame.plane)
        single = np.array([frame.weights[band].sum() >= MIN_SOLID_ANGLE for band in now_bands])
        raw_memory = single.copy()
        cleared_memory = single.copy()
        person_memory = single.copy()
        for past, kept in history:
            past_bands = corridor_bands(past.cloud, frame.origin, frame.plane)
            expired_person = (person_history_s is not None and past.person is not None
                              and frame.time - past.time > person_history_s + 1e-8)
            if expired_person:
                expired = past.person
                if past.person_motion_at is not None:
                    expired = expired & (past.person_motion_at <= frame.time)
            for heights, band in enumerate(past_bands):
                occupied = past.weights[band].sum() >= MIN_SOLID_ANGLE
                raw_memory[heights] |= occupied
                if person_history_s is not None:
                    retained = occupied
                    if occupied and expired_person:
                        retained = past.weights[band & ~expired].sum() >= MIN_SOLID_ANGLE
                    person_memory[heights] |= retained
                candidate = np.flatnonzero(band & kept)
                if len(candidate):
                    free = frame.witnessed_free(past.cloud[candidate])
                    kept[candidate[free]] = False
                    cleared_memory[heights] |= past.weights[candidate[~free]].sum() >= MIN_SOLID_ANGLE

        arms = [single, raw_memory, cleared_memory]
        if person_history_s is not None:
            if np.any(single & ~person_memory) or np.any(person_memory & ~raw_memory):
                raise AssertionError("Person history changed current evidence or added support")
            arms.append(person_memory)
        recent_votes.append(np.array(arms))
        if len(recent_votes) == 3:
            votes = np.count_nonzero(recent_votes, axis=0) >= 2
            if np.any(votes[0] & ~votes[2]) or np.any(votes[2] & ~votes[1]):
                raise AssertionError("History filtering changed current evidence or added support")
            item = {"time": frame.time, "alarm": votes.tolist()}
            if keep_trace:
                item["raw"] = recent_votes[-1].tolist()
                item["history_frames"] = len(history)
            result[frame.row] = item
        history.append((frame, np.ones(len(frame.cloud), dtype=bool)))
    return result


def lamar_frames(prepared, native, person_mask_for_cloud=None):
    keys = ("T_local", "ts", "local_down", "component", "depth_index", "depth_timestamp_us")
    with np.load(prepared, allow_pickle=False) as source:
        data = {key: source[key] for key in keys}
        sensor = DepthSensor(source["depth_lut"], source["T_depth_to_head"])
    with np.load(native, allow_pickle=False) as source:
        rows, stamps, images = source["depth_rows"], source["timestamp_us"], source["depth_mm"]
    if not np.array_equal(stamps, data["depth_timestamp_us"][rows]) or np.any(np.diff(rows) <= 0):
        raise ValueError(f"Depth rows and timestamps do not match for {prepared.stem}")
    for row, stamp, depth_mm in zip(rows, stamps, images, strict=True):
        index = data["depth_index"][row]
        plane = walking_plane(data, index) if index >= 0 else None
        if plane is None:
            continue
        head_pose = data["T_local"][index]
        depth_pose = head_pose @ sensor.head_from_depth
        depth = depth_mm.astype(np.float32) * .001
        cloud, weights = sensor.cloud(depth, depth_pose)

        def clear(points, image=depth, pose=depth_pose):
            return sensor.clear_mask(points, image, pose)

        person = person_mask_for_cloud(int(row), int(stamp), cloud) if person_mask_for_cloud else None
        yield Frame(int(row), int(stamp) * 1e-6, int(data["component"][index]), cloud, weights,
                    head_pose[:3, 3], plane, clear, person)


def evaluate_lamar(labels, predictions):
    episodes = []
    for visit in labels["visits"]:
        visits_pred = predictions.get(visit["visit"], {})
        if not visits_pred:
            continue
        for category in ("travel_occupied", "head_occupied_travel_empty", "other_clear"):
            for segment in episode_pairs(visit["rows"], category):
                if not all(row["depth_row"] in visits_pred for row in segment):
                    continue
                reference = np.array([row["travel"] for row in segment]) > 0
                alarms = np.array([visits_pred[row["depth_row"]]["alarm"] for row in segment])
                correct_band = alarms & reference[:, None, :] if category == "travel_occupied" else alarms
                times = np.array([row["time"] for row in segment])
                first = [float(times[np.flatnonzero(m)[0]]) if m.any() else None
                         for m in correct_band.any(axis=2).T]
                episodes.append(dict(session=visit["visit"], category=category, times=times.tolist(),
                                     alarm_frames=correct_band.any(axis=2).sum(axis=0).tolist(), first=first))
    summary = {}
    for category in ("travel_occupied", "head_occupied_travel_empty", "other_clear"):
        selected = [item for item in episodes if item["category"] == category]
        summary[category] = dict(
            episodes=len(selected), times=sum(len(e["times"]) for e in selected),
            sessions=len({e["session"] for e in selected}),
            **{arm: dict(episodes_with_alarm=sum(e["first"][i] is not None for e in selected),
                         times_with_alarm=sum(e["alarm_frames"][i] for e in selected),
                         new_alarm=sum(e["first"][0] is None and e["first"][i] is not None for e in selected),
                         earlier=sum(e["first"][0] is not None and e["first"][i] is not None
                                     and e["first"][i] < e["first"][0] for e in selected))
               for i, arm in enumerate(ARMS)},
        )
    return dict(summary=summary, episodes=episodes)


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lamar", type=Path, help="Existing consumed LaMAR pilot work directory")
    parser.add_argument("output", type=Path, help="Empty task-owned results directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.lamar / "hololens" / "CAB"
    predictions = {}
    for native in sorted((source / "native_depth").glob("*.npz")):
        predictions[native.stem] = replay(lamar_frames(source / "prepared" / native.name, native), keep_trace=True)
        print(native.stem, len(predictions[native.stem]), "evaluable time steps", flush=True)
    save(args.output / "lamar_predictions.json", predictions)
    labels = json.loads((args.lamar / "screen-hololens" / "summary.json").read_text(encoding="utf-8"))
    result = evaluate_lamar(labels, predictions)
    save(args.output / "lamar_comparison.json", result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

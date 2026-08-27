"""Run the CODa pose-compensated multiclass ceiling for DTR R2.

Only source-native 3-D boxes, persistent instance IDs, synchronized timestamps,
and dense ego poses are consumed.  No RGB or LiDAR detector is run.  Future
boxes are reserved for event truth after all causal predictions are produced.

The algorithm sees each box as a conservative circular footprint.  Truth is
stricter: the future source-native oriented rectangle must intersect the
wearer's actual future path capsule.  This keeps circularization error visible
as false alerts instead of building the same approximation into the labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from dtr_r0 import Arm, CausalFrame, DTRConfig, EgoPose, Observation, Prediction, Signal, run_arm
from dtr_r1 import FROZEN_R1_CONFIG, run_r1_arm
from dtr_r2 import FROZEN_R2_CONFIG, run_r2_arm
from jrdb_native_ceiling import (
    ArmAccumulator,
    alert_segments,
    overlaps,
    ratio,
    r1_dominance_decision,
    score_arm,
)


SCHEMA = "dtr-r2-coda-native-ceiling-v1"
CLAIM_CEILING = "PUBLIC_REAL_PRIVILEGED_MULTICLASS_NATIVE_BOX_POSE_CEILING_ONLY"
HORIZON_S = 3.0
ROUTE_HALF_WIDTH_M = 0.65
MINIMUM_HISTORY_S = 0.20
TIMESTAMP_TOLERANCE_S = 0.05
MAXIMUM_CONTIGUOUS_GAP_S = 0.15

DYNAMIC_CLASS_GROUP = {
    "Pedestrian": "pedestrian",
    "Bike": "micromobility",
    "Scooter": "micromobility",
    "Motorcycle": "micromobility",
    "Segway": "micromobility",
    "Skateboard": "micromobility",
    "Car": "vehicle",
    "Golf Cart": "vehicle",
    "Truck": "vehicle",
    "Pickup Truck": "vehicle",
    "Delivery Truck": "vehicle",
    "Service Vehicle": "vehicle",
    "Utility Vehicle": "vehicle",
    "Bus": "vehicle",
    "Cart": "temporary_obstacle",
}


@dataclass(frozen=True)
class PoseRow:
    time_s: float
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class CodaSample:
    frame_index: int
    time_s: float
    forward_m: float
    left_m: float
    relative_world_x_m: float
    relative_world_y_m: float
    radius_m: float
    length_m: float
    width_m: float
    yaw_rad: float
    class_name: str
    object_group: str

    @property
    def distance_m(self) -> float:
        return math.hypot(self.forward_m, self.left_m)

    @property
    def tube_threshold_m(self) -> float:
        return ROUTE_HALF_WIDTH_M + self.radius_m

    @property
    def lateral_half_extent_m(self) -> float:
        return (
            abs(math.sin(self.yaw_rad)) * self.length_m / 2.0
            + abs(math.cos(self.yaw_rad)) * self.width_m / 2.0
        )

    @property
    def forward_half_extent_m(self) -> float:
        return (
            abs(math.cos(self.yaw_rad)) * self.length_m / 2.0
            + abs(math.sin(self.yaw_rad)) * self.width_m / 2.0
        )

    def footprint_clearance_m(self) -> float:
        """Shortest ground-plane distance from ego origin to oriented box."""

        cosine = math.cos(self.yaw_rad)
        sine = math.sin(self.yaw_rad)
        delta_x = -self.forward_m
        delta_y = -self.left_m
        box_x = delta_x * cosine + delta_y * sine
        box_y = -delta_x * sine + delta_y * cosine
        outside_x = max(0.0, abs(box_x) - self.length_m / 2.0)
        outside_y = max(0.0, abs(box_y) - self.width_m / 2.0)
        return math.hypot(outside_x, outside_y)


@dataclass(frozen=True)
class CodaTruthEvent:
    start_index: int
    end_index: int
    contact_index: int
    category: str
    object_group: str
    class_name: str


@dataclass
class GroupAccumulator:
    total: Counter[str] = field(default_factory=Counter)
    recalled: Counter[str] = field(default_factory=Counter)
    escalated: Counter[str] = field(default_factory=Counter)
    alert_segments: Counter[str] = field(default_factory=Counter)
    false_alert_segments: Counter[str] = field(default_factory=Counter)
    evaluated_frames: Counter[str] = field(default_factory=Counter)

    def merge(self, other: "GroupAccumulator") -> None:
        self.total.update(other.total)
        self.recalled.update(other.recalled)
        self.escalated.update(other.escalated)
        self.alert_segments.update(other.alert_segments)
        self.false_alert_segments.update(other.false_alert_segments)
        self.evaluated_frames.update(other.evaluated_frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            group: {
                "events": self.total[group],
                "recalled": self.recalled[group],
                "recall": ratio(self.recalled[group], self.total[group]),
                "escalated": self.escalated[group],
                "escalation_rate": ratio(self.escalated[group], self.total[group]),
                "alert_segments": self.alert_segments[group],
                "false_alert_segments": self.false_alert_segments[group],
                "evaluated_frames": self.evaluated_frames[group],
            }
            for group in sorted(
                self.total.keys()
                | self.alert_segments.keys()
                | self.evaluated_frames.keys()
            )
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quaternion_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def read_poses(path: Path) -> list[PoseRow]:
    poses: list[PoseRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(item) for item in line.split()]
        if len(values) != 8:
            raise ValueError(f"unexpected CODa pose row in {path}")
        time_s, x_m, y_m, _z_m, qw, qx, qy, qz = values
        poses.append(PoseRow(time_s, x_m, y_m, quaternion_yaw(qw, qx, qy, qz)))
    return poses


def read_sequence(
    dataset_root: Path,
    sequence: str,
) -> tuple[dict[str, list[CodaSample]], dict[str, Any], dict[str, str]]:
    bbox_dir = dataset_root / "3d_bbox" / "os1" / sequence
    timestamp_path = dataset_root / "timestamps" / f"{sequence}.txt"
    global_pose_path = dataset_root / "poses" / "dense_global" / f"{sequence}.txt"
    local_pose_path = dataset_root / "poses" / "dense" / f"{sequence}.txt"
    metadata_path = dataset_root / "metadata" / f"{sequence}.json"
    pose_path = global_pose_path if global_pose_path.exists() else local_pose_path
    required = (bbox_dir, timestamp_path, pose_path, metadata_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing CODa inputs: {missing}")

    timestamps = [
        float(line)
        for line in timestamp_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    poses = read_poses(pose_path)
    if len(poses) != len(timestamps):
        raise ValueError("CODa dense pose and synchronized timestamp counts differ")

    tracks: dict[str, list[CodaSample]] = {}
    class_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    occlusion_counts: Counter[str] = Counter()
    bbox_digest = hashlib.sha256()
    label_frames = 0
    objects_seen = 0
    objects_used = 0
    pose_timestamp_mismatches = 0
    for path in sorted(
        bbox_dir.glob("*.json"),
        key=lambda item: int(item.stem.rsplit("_", 1)[-1]),
    ):
        frame_index = int(path.stem.rsplit("_", 1)[-1])
        if frame_index >= len(timestamps):
            raise ValueError(f"CODa frame {frame_index} exceeds synchronized source")
        raw = path.read_bytes()
        bbox_digest.update(path.name.encode("utf-8"))
        bbox_digest.update(b"\0")
        bbox_digest.update(raw)
        payload = json.loads(raw)
        label_frames += 1
        time_s = timestamps[frame_index]
        pose = poses[frame_index]
        if abs(pose.time_s - time_s) > TIMESTAMP_TOLERANCE_S:
            pose_timestamp_mismatches += 1
        cosine = math.cos(pose.yaw_rad)
        sine = math.sin(pose.yaw_rad)
        for box in payload.get("3dbbox", []):
            objects_seen += 1
            class_name = str(box["classId"])
            class_counts[class_name] += 1
            object_group = DYNAMIC_CLASS_GROUP.get(class_name)
            if object_group is None:
                continue
            forward_m = float(box["cX"])
            left_m = float(box["cY"])
            length_m = float(box["l"])
            width_m = float(box["w"])
            yaw_rad = float(box["y"])
            if not all(
                math.isfinite(value)
                for value in (forward_m, left_m, length_m, width_m, yaw_rad)
            ) or min(length_m, width_m) <= 0.0:
                continue
            relative_world_x_m = forward_m * cosine - left_m * sine
            relative_world_y_m = forward_m * sine + left_m * cosine
            sample = CodaSample(
                frame_index=frame_index,
                time_s=time_s,
                forward_m=forward_m,
                left_m=left_m,
                relative_world_x_m=relative_world_x_m,
                relative_world_y_m=relative_world_y_m,
                radius_m=max(0.15, 0.5 * math.hypot(length_m, width_m)),
                length_m=length_m,
                width_m=width_m,
                yaw_rad=yaw_rad,
                class_name=class_name,
                object_group=object_group,
            )
            tracks.setdefault(str(box["instanceId"]), []).append(sample)
            objects_used += 1
            group_counts[object_group] += 1
            attributes = box.get("labelAttributes", box.get("labelCategoryAttributes", {}))
            occlusion_counts[str(attributes.get("isOccluded", "Unknown"))] += 1
    for samples in tracks.values():
        samples.sort(key=lambda item: (item.frame_index, item.time_s))

    source = {
        "label_frames": label_frames,
        "objects_seen": objects_seen,
        "dynamic_objects_used": objects_used,
        "native_dynamic_identities": len(tracks),
        "class_box_counts": dict(sorted(class_counts.items())),
        "dynamic_group_box_counts": dict(sorted(group_counts.items())),
        "dynamic_occlusion_counts": dict(sorted(occlusion_counts.items())),
        "pose_timestamp_mismatch_frames": pose_timestamp_mismatches,
        "pose_authority": "dense_global" if pose_path == global_pose_path else "dense_local",
    }
    hashes = {
        "bbox_canonical_stream_sha256": bbox_digest.hexdigest(),
        "timestamps_sha256": sha256_file(timestamp_path),
        "poses_sha256": sha256_file(pose_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    return tracks, source, hashes


def contiguous_segments(samples: Sequence[CodaSample]) -> Iterable[list[CodaSample]]:
    current: list[CodaSample] = []
    for sample in samples:
        if current and (
            sample.frame_index != current[-1].frame_index + 1
            or sample.time_s <= current[-1].time_s
            or sample.time_s - current[-1].time_s > MAXIMUM_CONTIGUOUS_GAP_S
        ):
            yield current
            current = []
        current.append(sample)
    if current:
        yield current


def causal_frames(track_id: str, samples: Sequence[CodaSample]) -> list[CausalFrame]:
    origin = samples[0].time_s
    identity_pose = EgoPose(0.0, 0.0, 0.0, 0.0)
    return [
        CausalFrame(
            time_s=sample.time_s - origin,
            ego_pose=identity_pose,
            observations=(
                Observation(
                    track_id=track_id,
                    forward_m=sample.relative_world_x_m,
                    left_m=sample.relative_world_y_m,
                    radius_m=sample.radius_m,
                ),
            ),
            person_detection_count=1,
        )
        for sample in samples
    ]


def future_hits(
    samples: Sequence[CodaSample],
) -> tuple[list[bool | None], list[int | None]]:
    truth: list[bool | None] = []
    contacts: list[int | None] = []
    final_time = samples[-1].time_s
    for index, sample in enumerate(samples):
        hit: int | None = None
        future_index = index
        while (
            future_index < len(samples)
            and samples[future_index].time_s - sample.time_s <= HORIZON_S + 1e-9
        ):
            if samples[future_index].footprint_clearance_m() <= ROUTE_HALF_WIDTH_M:
                hit = future_index
                break
            future_index += 1
        has_full_future = final_time - sample.time_s >= HORIZON_S - TIMESTAMP_TOLERANCE_S
        truth.append(True if hit is not None else False if has_full_future else None)
        contacts.append(hit)
    return truth, contacts


def classify_event(samples: Sequence[CodaSample], start: int, contact: int) -> str:
    first = samples[start]
    hit = samples[contact]
    if (
        abs(first.left_m) > ROUTE_HALF_WIDTH_M + first.lateral_half_extent_m
        and abs(hit.left_m) <= ROUTE_HALF_WIDTH_M + hit.lateral_half_extent_m
    ):
        return "lateral_crossing"
    if (
        first.forward_m > ROUTE_HALF_WIDTH_M + first.forward_half_extent_m
        and abs(first.left_m) <= ROUTE_HALF_WIDTH_M + first.lateral_half_extent_m
        and hit.forward_m < first.forward_m
    ):
        return "oncoming_corridor"
    return "other_close_approach"


def truth_events(
    samples: Sequence[CodaSample],
    truth: Sequence[bool | None],
    contacts: Sequence[int | None],
) -> tuple[list[CodaTruthEvent], list[bool]]:
    known = [
        value is not None and sample.time_s - samples[0].time_s + 1e-9 >= MINIMUM_HISTORY_S
        for sample, value in zip(samples, truth)
    ]
    events: list[CodaTruthEvent] = []
    index = 0
    while index < len(samples):
        if not known[index] or truth[index] is not True:
            index += 1
            continue
        if index == 0 or not known[index - 1]:
            while index < len(samples) and known[index] and truth[index] is True:
                index += 1
            continue
        if truth[index - 1] is True:
            index += 1
            continue
        start = index
        while index + 1 < len(samples) and known[index + 1] and truth[index + 1] is True:
            index += 1
        end = index
        contact = contacts[start]
        if contact is not None:
            events.append(
                CodaTruthEvent(
                    start_index=start,
                    end_index=end,
                    contact_index=contact,
                    category=classify_event(samples, start, contact),
                    object_group=samples[start].object_group,
                    class_name=samples[start].class_name,
                )
            )
        index += 1
    return events, known


def score_groups(
    samples: Sequence[CodaSample],
    predictions: Sequence[Prediction],
    events: Sequence[CodaTruthEvent],
    known: Sequence[bool],
    truth: Sequence[bool | None],
) -> GroupAccumulator:
    result = GroupAccumulator()
    group = samples[0].object_group
    segments = [
        segment
        for segment in alert_segments(predictions)
        if any(known[index] for index in range(segment.start_index, segment.end_index + 1))
    ]
    result.alert_segments[group] += len(segments)
    result.false_alert_segments[group] += sum(
        not any(
            known[index] and truth[index] is True
            for index in range(segment.start_index, segment.end_index + 1)
        )
        for segment in segments
    )
    result.evaluated_frames[group] += sum(known)
    for event in events:
        result.total[event.object_group] += 1
        if any(overlaps(segment, event.start_index, event.contact_index) for segment in segments):
            result.recalled[event.object_group] += 1
        if any(
            predictions[index].signal is Signal.ESCALATE
            for index in range(event.start_index, event.contact_index + 1)
        ):
            result.escalated[event.object_group] += 1
    return result


def evaluate_segment(
    track_id: str,
    samples: Sequence[CodaSample],
    config: DTRConfig,
    arms: Sequence[Arm],
) -> tuple[dict[Arm, ArmAccumulator], dict[Arm, GroupAccumulator], int, float]:
    if len(samples) < 2 or samples[-1].time_s - samples[0].time_s < MINIMUM_HISTORY_S + HORIZON_S:
        return {}, {}, 0, 0.0
    truth, contacts = future_hits(samples)
    events, known = truth_events(samples, truth, contacts)
    if not any(known):
        return {}, {}, 0, 0.0
    frames = causal_frames(track_id, samples)
    predictions = {
        arm: (
            run_r1_arm(frames)
            if arm is Arm.D_R1_OCCUPANCY_CONSENSUS
            else run_r2_arm(frames, config)
            if arm is Arm.E_R2_GUARDED_CONSENSUS
            else run_arm(frames, arm, config)
        )
        for arm in arms
    }
    scored = {
        arm: score_arm(samples, arm_predictions, events, known, truth)
        for arm, arm_predictions in predictions.items()
    }
    grouped = {
        arm: score_groups(samples, arm_predictions, events, known, truth)
        for arm, arm_predictions in predictions.items()
    }
    known_indices = [index for index, value in enumerate(known) if value]
    exposure_s = (
        samples[known_indices[-1]].time_s - samples[known_indices[0]].time_s
        if len(known_indices) > 1
        else 0.0
    )
    return scored, grouped, len(events), exposure_s


def parse_sequence_root(value: str) -> tuple[str, Path]:
    sequence, separator, root = value.partition("=")
    if not separator or not sequence or not root:
        raise argparse.ArgumentTypeError("expected SEQUENCE=DATASET_ROOT")
    return sequence, Path(root)


def compare_arms(
    pooled: dict[str, dict[str, Any]],
    challenger: Arm,
    comparator: Arm,
) -> dict[str, Any]:
    left = pooled[challenger.value]
    right = pooled[comparator.value]
    recall_delta = left["critical_event_recall"] - right["critical_event_recall"]
    false_delta = right["false_alert_segments"] - left["false_alert_segments"]
    return {
        "status": (
            "DOMINATES"
            if recall_delta >= -1e-12 and false_delta > 0
            else "DOES_NOT_DOMINATE"
        ),
        "challenger": challenger.value,
        "comparator": comparator.value,
        "critical_recall_delta": recall_delta,
        "false_alert_segment_delta": false_delta,
        "false_alert_segment_reduction_fraction": ratio(
            false_delta, right["false_alert_segments"]
        ),
        "clear_rate_delta": left["clear_rate"] - right["clear_rate"],
    }


def evaluate(sequence_roots: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    config = DTRConfig(
        route_horizon_s=HORIZON_S,
        route_half_width_m=ROUTE_HALF_WIDTH_M,
        nominal_wearer_speed_mps=0.0,
    )
    arms = (
        Arm.B2_RADIAL_TTC,
        Arm.C_ROUTE_INTERSECTION,
        Arm.D_R1_OCCUPANCY_CONSENSUS,
        Arm.E_R2_GUARDED_CONSENSUS,
    )
    pooled = {arm: ArmAccumulator() for arm in arms}
    pooled_groups = {arm: GroupAccumulator() for arm in arms}
    sequence_results = []
    totals = Counter()
    for sequence, dataset_root in sequence_roots:
        tracks, source, hashes = read_sequence(dataset_root, sequence)
        sequence_arms = {arm: ArmAccumulator() for arm in arms}
        sequence_groups = {arm: GroupAccumulator() for arm in arms}
        track_segments = 0
        evaluable_segments = 0
        critical_events = 0
        exposure_s = 0.0
        for track_id, samples in tracks.items():
            for segment_index, segment in enumerate(contiguous_segments(samples)):
                track_segments += 1
                scored, grouped, events, segment_exposure_s = evaluate_segment(
                    f"{sequence}/{track_id}/{segment_index}", segment, config, arms
                )
                if not scored:
                    continue
                evaluable_segments += 1
                critical_events += events
                exposure_s += segment_exposure_s
                for arm in arms:
                    sequence_arms[arm].merge(scored[arm])
                    pooled[arm].merge(scored[arm])
                    sequence_groups[arm].merge(grouped[arm])
                    pooled_groups[arm].merge(grouped[arm])
        sequence_results.append(
            {
                "sequence": sequence,
                "dataset_root": str(dataset_root.resolve()),
                "source": source,
                "input_hashes": hashes,
                "contiguous_track_segments": track_segments,
                "evaluable_track_segments": evaluable_segments,
                "critical_events": critical_events,
                "track_segment_exposure_s": exposure_s,
                "arms": {
                    arm.value: {
                        **sequence_arms[arm].to_dict(include_escalation=True),
                        "by_object_group": sequence_groups[arm].to_dict(),
                    }
                    for arm in arms
                },
            }
        )
        totals.update(
            sequences=1,
            label_frames=source["label_frames"],
            dynamic_objects_used=source["dynamic_objects_used"],
            native_dynamic_identities=source["native_dynamic_identities"],
            contiguous_track_segments=track_segments,
            evaluable_track_segments=evaluable_segments,
            critical_events=critical_events,
        )
        print(
            f"CODa {sequence}: events={critical_events}, "
            f"evaluable_tracks={evaluable_segments}, frames={source['label_frames']}"
        )

    pooled_dict = {
        arm.value: {
            **pooled[arm].to_dict(include_escalation=True),
            "by_object_group": pooled_groups[arm].to_dict(),
        }
        for arm in arms
    }
    return {
        "schema": SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "source": {
            "dataset": "UT Campus Object Dataset (CODa)",
            "official_page": "https://amrl.cs.utexas.edu/coda/",
            "dataset_license": "CC BY-NC-SA 4.0 plus dataset terms",
            "selection": "explicit sequence roots: "
            + ", ".join(sequence for sequence, _ in sequence_roots),
            "acquisition": "HTTP Range extraction of native boxes/poses/timestamps only; no RGB or LiDAR payload",
        },
        "protocol": {
            "causal_algorithm_inputs": [
                "current-and-past source-native 3-D boxes",
                "persistent source-native instance IDs",
                "synchronized timestamps",
                "dense ego yaw used to express relative tracks in a fixed world orientation",
            ],
            "truth_only_after_prediction": "future oriented-box intersection with actual future ego path capsule",
            "algorithm_footprint": "circumscribed circle of current source-native length/width",
            "truth_footprint": "source-native oriented rectangle",
            "route_horizon_s": HORIZON_S,
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "dynamic_classes": dict(sorted(DYNAMIC_CLASS_GROUP.items())),
            "r1_config": FROZEN_R1_CONFIG.to_dict(),
            "r2_config": FROZEN_R2_CONFIG.to_dict(),
            "r0_config": {
                "minimum_history_s": MINIMUM_HISTORY_S,
                "nominal_wearer_speed_mps": 0.0,
                "relative_track_authority": "pose-rotated ego-relative native boxes",
            },
        },
        "totals": dict(totals),
        "pooled_arms": pooled_dict,
        "r1_vs_r0": r1_dominance_decision(pooled_dict),
        "r2_vs_r0": compare_arms(
            pooled_dict,
            Arm.E_R2_GUARDED_CONSENSUS,
            Arm.C_ROUTE_INTERSECTION,
        ),
        "r2_vs_r1": compare_arms(
            pooled_dict,
            Arm.E_R2_GUARDED_CONSENSUS,
            Arm.D_R1_OCCUPANCY_CONSENSUS,
        ),
        "sequences": sequence_results,
        "limitations": [
            "Privileged native boxes, identities, and poses are an algorithm ceiling, not a deployable detector result.",
            "The selected campus robot sequences do not establish natural-distribution or safety performance.",
            "Circularized algorithm footprints are conservative for long vehicles; oriented rectangles remain truth-only.",
            "Static walls, drop-offs, and vertical head clearance are outside this dynamic-box run.",
            "Risk score/support remain descriptive and uncalibrated.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-root",
        action="append",
        required=True,
        type=parse_sequence_root,
        metavar="SEQUENCE=ROOT",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.sequence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["r2_vs_r0"], indent=2))


if __name__ == "__main__":
    main()

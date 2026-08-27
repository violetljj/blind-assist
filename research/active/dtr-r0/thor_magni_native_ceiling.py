"""Run the THOR-MAGNI exact global-track/ego privileged ceiling for DTR-R0.

The two arms receive the same current-and-past QTM person and camera-wearer
centroids. Wearer route yaw is derived only from past motion. Future global
centroids are opened only by the evaluator to determine synchronized entry into
the actual wearer tube within three seconds.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from dtr_r0 import Arm, CausalFrame, DTRConfig, EgoPose, Observation, run_arm
from dtr_r1 import FROZEN_R1_CONFIG, run_r1_arm
from dtr_r2 import FROZEN_R2_CONFIG, run_r2_arm
from jrdb_native_ceiling import (
    ArmAccumulator,
    decision,
    future_hits,
    score_arm,
    sha256_file,
    truth_events,
    r1_dominance_decision,
    successor_comparison,
)


SCHEMA = "dtr-r0-thor-magni-native-ceiling-v1"
R1_SCHEMA = "dtr-r1-thor-magni-native-ceiling-v1"
R2_SCHEMA = "dtr-r2-thor-magni-native-ceiling-v1"
CLAIM_CEILING = "PUBLIC_REAL_EXACT_GLOBAL_TRACK_AND_EGO_PRIVILEGED_CEILING_ONLY"
MANIFEST_GLOB = "thor_magni_window_manifest_d7-r1-thor-magni-window-pupil-*.json"
HORIZON_S = 3.0
ROUTE_HALF_WIDTH_M = 0.65
PERSON_RADIUS_M = 0.30
QTM_SAMPLE_STRIDE = 10
ROUTE_YAW_LOOKBACK_S = 0.50
ROUTE_YAW_MINIMUM_SPAN_S = 0.20
ROUTE_YAW_MINIMUM_SPEED_MPS = 0.25
MAXIMUM_SAMPLED_GAP_S = 0.15


@dataclass(frozen=True)
class RawRow:
    frame_index: int
    time_s: float
    ego_xy_m: tuple[float, float] | None
    targets_xy_m: dict[str, tuple[float, float] | None]

    @property
    def completeness(self) -> int:
        return int(self.ego_xy_m is not None) + sum(
            value is not None for value in self.targets_xy_m.values()
        )


@dataclass(frozen=True)
class ThorSample:
    frame_index: int
    time_s: float
    ego_x_m: float
    ego_y_m: float
    target_x_m: float
    target_y_m: float
    route_yaw_rad: float
    radius_m: float = PERSON_RADIUS_M

    @property
    def forward_m(self) -> float:
        delta_x = self.target_x_m - self.ego_x_m
        delta_y = self.target_y_m - self.ego_y_m
        cosine = math.cos(self.route_yaw_rad)
        sine = math.sin(self.route_yaw_rad)
        return delta_x * cosine + delta_y * sine

    @property
    def left_m(self) -> float:
        delta_x = self.target_x_m - self.ego_x_m
        delta_y = self.target_y_m - self.ego_y_m
        cosine = math.cos(self.route_yaw_rad)
        sine = math.sin(self.route_yaw_rad)
        return -delta_x * sine + delta_y * cosine

    @property
    def distance_m(self) -> float:
        return math.hypot(
            self.target_x_m - self.ego_x_m,
            self.target_y_m - self.ego_y_m,
        )

    @property
    def tube_threshold_m(self) -> float:
        return ROUTE_HALF_WIDTH_M + self.radius_m


@dataclass(frozen=True)
class SessionData:
    targets: tuple[str, ...]
    rows: tuple[RawRow, ...]
    route_yaws: tuple[float | None, ...]
    counts: dict[str, int]


def optional_float(value: str) -> float | None:
    stripped = value.strip()
    if stripped in {"", "N/A"}:
        return None
    parsed = float(stripped)
    return parsed if math.isfinite(parsed) else None


def xy_from_row(
    row: Sequence[str], columns: tuple[int, int]
) -> tuple[float, float] | None:
    x = optional_float(row[columns[0]])
    y = optional_float(row[columns[1]])
    if x is None or y is None:
        return None
    return x / 1000.0, y / 1000.0


def causal_route_yaws(rows: Sequence[RawRow]) -> list[float | None]:
    yaws: list[float | None] = []
    history_start = 0
    for index, row in enumerate(rows):
        while (
            history_start < index
            and rows[history_start].time_s
            < row.time_s - ROUTE_YAW_LOOKBACK_S - 1e-9
        ):
            history_start += 1
        if row.ego_xy_m is None:
            yaws.append(None)
            continue
        prior = next(
            (
                rows[candidate]
                for candidate in range(history_start, index)
                if rows[candidate].ego_xy_m is not None
            ),
            None,
        )
        if prior is None or prior.ego_xy_m is None:
            yaws.append(None)
            continue
        span_s = row.time_s - prior.time_s
        if span_s + 1e-9 < ROUTE_YAW_MINIMUM_SPAN_S:
            yaws.append(None)
            continue
        delta_x = row.ego_xy_m[0] - prior.ego_xy_m[0]
        delta_y = row.ego_xy_m[1] - prior.ego_xy_m[1]
        speed_mps = math.hypot(delta_x, delta_y) / span_s
        if speed_mps < ROUTE_YAW_MINIMUM_SPEED_MPS:
            yaws.append(None)
            continue
        yaws.append(math.atan2(delta_y, delta_x))
    return yaws


def read_session(path: Path, camera_body: str) -> SessionData:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        metadata = [next(reader) for _ in range(16)]
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        body_names = tuple(value for value in metadata[12][1:] if value)
        targets = tuple(
            body
            for body in body_names
            if body.startswith("Helmet_") and body != camera_body
        )
        required = [
            "Frame",
            "Time",
            f"{camera_body} Centroid_X",
            f"{camera_body} Centroid_Y",
            *[
                f"{body} Centroid_{axis}"
                for body in targets
                for axis in "XY"
            ],
        ]
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"missing THOR-MAGNI columns in {path}: {missing}")
        ego_columns = (
            index[f"{camera_body} Centroid_X"],
            index[f"{camera_body} Centroid_Y"],
        )
        target_columns = {
            body: (
                index[f"{body} Centroid_X"],
                index[f"{body} Centroid_Y"],
            )
            for body in targets
        }
        by_frame: dict[int, RawRow] = {}
        source_rows = 0
        duplicate_rows = 0
        for values in reader:
            source_rows += 1
            frame_index = int(values[index["Frame"]])
            candidate = RawRow(
                frame_index=frame_index,
                time_s=float(values[index["Time"]]),
                ego_xy_m=xy_from_row(values, ego_columns),
                targets_xy_m={
                    body: xy_from_row(values, columns)
                    for body, columns in target_columns.items()
                },
            )
            current = by_frame.get(frame_index)
            if current is not None:
                duplicate_rows += 1
            if current is None or candidate.completeness > current.completeness:
                by_frame[frame_index] = candidate

    ordered = sorted(by_frame.values(), key=lambda item: (item.frame_index, item.time_s))
    if not ordered:
        raise ValueError(f"empty THOR-MAGNI scenario: {path}")
    first_frame = ordered[0].frame_index
    sampled = [
        row
        for row in ordered
        if (row.frame_index - first_frame) % QTM_SAMPLE_STRIDE == 0
    ]
    strictly_ordered: list[RawRow] = []
    nonincreasing_rows = 0
    for row in sampled:
        if strictly_ordered and row.time_s <= strictly_ordered[-1].time_s:
            nonincreasing_rows += 1
            continue
        strictly_ordered.append(row)
    yaws = causal_route_yaws(strictly_ordered)
    return SessionData(
        targets=targets,
        rows=tuple(strictly_ordered),
        route_yaws=tuple(yaws),
        counts={
            "source_rows": source_rows,
            "duplicate_rows": duplicate_rows,
            "unique_rows": len(ordered),
            "sampled_rows": len(strictly_ordered),
            "nonincreasing_sampled_rows": nonincreasing_rows,
            "ego_observed_sampled_rows": sum(
                row.ego_xy_m is not None for row in strictly_ordered
            ),
            "route_eligible_sampled_rows": sum(yaw is not None for yaw in yaws),
        },
    )


def target_samples(data: SessionData, body: str) -> list[ThorSample]:
    samples: list[ThorSample] = []
    for row, route_yaw in zip(data.rows, data.route_yaws):
        target_xy = row.targets_xy_m[body]
        if row.ego_xy_m is None or target_xy is None or route_yaw is None:
            continue
        samples.append(
            ThorSample(
                frame_index=row.frame_index,
                time_s=row.time_s,
                ego_x_m=row.ego_xy_m[0],
                ego_y_m=row.ego_xy_m[1],
                target_x_m=target_xy[0],
                target_y_m=target_xy[1],
                route_yaw_rad=route_yaw,
            )
        )
    return samples


def contiguous_segments(samples: Sequence[ThorSample]) -> Iterable[list[ThorSample]]:
    current: list[ThorSample] = []
    for sample in samples:
        if current and (
            sample.time_s <= current[-1].time_s
            or sample.time_s - current[-1].time_s
            > MAXIMUM_SAMPLED_GAP_S + 1e-9
        ):
            yield current
            current = []
        current.append(sample)
    if current:
        yield current


def causal_frames(track_id: str, samples: Sequence[ThorSample]) -> list[CausalFrame]:
    origin = samples[0].time_s
    return [
        CausalFrame(
            time_s=sample.time_s - origin,
            ego_pose=EgoPose(
                x_m=sample.ego_x_m,
                y_m=sample.ego_y_m,
                body_yaw_rad=sample.route_yaw_rad,
                sensor_yaw_rad=sample.route_yaw_rad,
            ),
            observations=(
                Observation(
                    track_id=track_id,
                    forward_m=sample.forward_m,
                    left_m=sample.left_m,
                    radius_m=sample.radius_m,
                ),
            ),
            person_detection_count=1,
        )
        for sample in samples
    ]


def evaluate_segment(
    track_id: str,
    samples: Sequence[ThorSample],
    config: DTRConfig,
    arms: Sequence[Arm] = (Arm.B2_RADIAL_TTC, Arm.C_ROUTE_INTERSECTION),
) -> tuple[dict[Arm, ArmAccumulator], int, float]:
    if (
        len(samples) < 2
        or samples[-1].time_s - samples[0].time_s
        < config.minimum_track_span_s + HORIZON_S
    ):
        return {}, 0, 0.0
    truth, contacts = future_hits(samples)
    events, known = truth_events(
        samples, truth, contacts, config.minimum_track_span_s
    )
    if not any(known):
        return {}, 0, 0.0
    frames = causal_frames(track_id, samples)
    scored = {
        arm: score_arm(
            samples,
            (
                run_r1_arm(frames)
                if arm is Arm.D_R1_OCCUPANCY_CONSENSUS
                else run_r2_arm(frames, config)
                if arm is Arm.E_R2_GUARDED_CONSENSUS
                else run_arm(frames, arm, config)
            ),
            events,
            known,
            truth,
        )
        for arm in arms
    }
    known_indices = [index for index, value in enumerate(known) if value]
    exposure_s = (
        samples[known_indices[-1]].time_s - samples[known_indices[0]].time_s
        if len(known_indices) > 1
        else 0.0
    )
    return scored, len(events), exposure_s


def evaluate(
    manifest_dir: Path,
    include_r1: bool = False,
    include_r2: bool = False,
) -> dict[str, Any]:
    include_successor = include_r1 or include_r2
    manifest_paths = sorted(manifest_dir.glob(MANIFEST_GLOB))
    if not manifest_paths:
        raise FileNotFoundError(f"no THOR-MAGNI manifests in {manifest_dir}")
    config = DTRConfig(
        route_horizon_s=HORIZON_S,
        route_half_width_m=ROUTE_HALF_WIDTH_M,
    )
    arms = (
        Arm.B2_RADIAL_TTC,
        Arm.C_ROUTE_INTERSECTION,
        *((Arm.D_R1_OCCUPANCY_CONSENSUS,) if include_successor else ()),
        *((Arm.E_R2_GUARDED_CONSENSUS,) if include_r2 else ()),
    )
    pooled = {arm: ArmAccumulator() for arm in arms}
    totals = {
        "sessions": 0,
        "source_rows": 0,
        "duplicate_rows": 0,
        "unique_rows": 0,
        "sampled_rows": 0,
        "nonincreasing_sampled_rows": 0,
        "ego_observed_sampled_rows": 0,
        "route_eligible_sampled_rows": 0,
        "target_identities": 0,
        "target_samples": 0,
        "contiguous_track_segments": 0,
        "evaluable_track_segments": 0,
        "critical_events": 0,
        "track_segment_exposure_s": 0.0,
    }
    session_results = []
    input_files = []
    for session_index, manifest_path in enumerate(manifest_paths, start=1):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenario_path = Path(str(manifest["scenario_csv"]))
        actual_hash = sha256_file(scenario_path)
        if actual_hash != manifest["scenario_csv_sha256"]:
            raise ValueError(f"scenario hash mismatch: {scenario_path}")
        data = read_session(scenario_path, str(manifest["camera_body"]))
        session_arms = {arm: ArmAccumulator() for arm in arms}
        session_segments = 0
        evaluable_segments = 0
        critical_events = 0
        exposure_s = 0.0
        target_sample_count = 0
        for body in data.targets:
            samples = target_samples(data, body)
            target_sample_count += len(samples)
            for segment_index, segment in enumerate(contiguous_segments(samples)):
                session_segments += 1
                scored, event_count, segment_exposure_s = evaluate_segment(
                    f"{manifest['source_session_id']}/{body}/{segment_index}",
                    segment,
                    config,
                    arms,
                )
                if not scored:
                    continue
                evaluable_segments += 1
                critical_events += event_count
                exposure_s += segment_exposure_s
                for arm, metrics in scored.items():
                    session_arms[arm].merge(metrics)
                    pooled[arm].merge(metrics)
        session_results.append(
            {
                "source_session_id": manifest["source_session_id"],
                "file_id": manifest["file_id"],
                "camera_body": manifest["camera_body"],
                "target_identities": len(data.targets),
                "target_samples": target_sample_count,
                "contiguous_track_segments": session_segments,
                "evaluable_track_segments": evaluable_segments,
                "critical_events": critical_events,
                "track_segment_exposure_s": exposure_s,
                "source": data.counts,
                "arms": {
                    arm.value: metrics.to_dict(include_escalation=include_successor)
                    for arm, metrics in session_arms.items()
                },
            }
        )
        totals["sessions"] += 1
        for name, value in data.counts.items():
            totals[name] += value
        totals["target_identities"] += len(data.targets)
        totals["target_samples"] += target_sample_count
        totals["contiguous_track_segments"] += session_segments
        totals["evaluable_track_segments"] += evaluable_segments
        totals["critical_events"] += critical_events
        totals["track_segment_exposure_s"] += exposure_s
        input_files.append(
            {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "scenario_csv": str(scenario_path.resolve()),
                "scenario_csv_sha256": actual_hash,
            }
        )
        print(
            f"[{session_index}/{len(manifest_paths)}] {manifest['file_id']}: "
            f"events={critical_events}, tracks={evaluable_segments}",
            flush=True,
        )

    pooled_dict = {
        arm.value: metrics.to_dict(include_escalation=include_successor)
        for arm, metrics in pooled.items()
    }
    result = {
        "schema_version": R2_SCHEMA if include_r2 else R1_SCHEMA if include_r1 else SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "source": {
            "dataset": "THOR-MAGNI public Pupil 19-session intake",
            "manifest_directory": str(manifest_dir.resolve()),
            "coordinate_contract": (
                "QTM global metric XY centroids; camera_body is ego and other "
                "Helmet_* bodies are person targets"
            ),
            "input_access": (
                "current-and-past QTM centroids only; wearer route yaw uses past motion"
            ),
            "truth_access": (
                "future synchronized global centroids used only by evaluator"
            ),
            "files": input_files,
        },
        "configuration": {
            "qtm_source_fps": 100.0,
            "evaluation_fps": 100.0 / QTM_SAMPLE_STRIDE,
            "route_horizon_s": HORIZON_S,
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "person_radius_m": PERSON_RADIUS_M,
            "track_window_s": config.track_window_s,
            "minimum_track_span_s": config.minimum_track_span_s,
            "nominal_wearer_speed_mps": config.nominal_wearer_speed_mps,
            "clear_grace_s": config.clear_grace_s,
            "route_yaw_lookback_s": ROUTE_YAW_LOOKBACK_S,
            "route_yaw_minimum_span_s": ROUTE_YAW_MINIMUM_SPAN_S,
            "route_yaw_minimum_speed_mps": ROUTE_YAW_MINIMUM_SPEED_MPS,
            "maximum_sampled_gap_s": MAXIMUM_SAMPLED_GAP_S,
            "r1": FROZEN_R1_CONFIG.to_dict() if include_successor else None,
            "r2": FROZEN_R2_CONFIG.to_dict() if include_r2 else None,
        },
        "coverage": totals,
        "pooled": pooled_dict,
        "decision": decision(pooled_dict),
        "r1_decision": r1_dominance_decision(pooled_dict),
        "r2_vs_r0": successor_comparison(
            pooled_dict,
            Arm.E_R2_GUARDED_CONSENSUS,
            Arm.C_ROUTE_INTERSECTION,
        ),
        "r2_vs_r1": successor_comparison(
            pooled_dict,
            Arm.E_R2_GUARDED_CONSENSUS,
            Arm.D_R1_OCCUPANCY_CONSENSUS,
        ),
        "by_session": session_results,
        "limitations": [
            "QTM centroids are privileged geometry, not detector or tracker output.",
            "Evaluation is conditional on a moving wearer whose route yaw is observable from past motion.",
            "Person radius is fixed at 0.30 m; route-entry truth is geometric, not human-authored alertability or safety truth.",
            "Alert lifecycle is scored per target track; simultaneous target alerts are not merged into a product utterance stream.",
            "No RGB detector, tracker, phone metric depth, Android runtime, or user study is evaluated.",
        ],
    }
    if not include_successor:
        result["configuration"].pop("r1")
        result.pop("r1_decision")
    if not include_r2:
        result["configuration"].pop("r2")
        result.pop("r2_vs_r0")
        result.pop("r2_vs_r1")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-r1",
        action="store_true",
        help="Add the fixed robust occupancy-consensus challenger.",
    )
    parser.add_argument(
        "--include-r2",
        action="store_true",
        help="Add R1 plus the fixed half-horizon guarded R2 successor.",
    )
    args = parser.parse_args()
    result = evaluate(
        args.manifest_dir,
        include_r1=args.include_r1,
        include_r2=args.include_r2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "coverage": result["coverage"],
                "decision": result["decision"],
                "r1_decision": result.get("r1_decision"),
                "r2_vs_r0": result.get("r2_vs_r0"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

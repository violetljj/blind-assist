from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class PairSupportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PairSupportError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class CandidateFrame:
    parent_id: str
    video_id: str
    sensor_timestamp_ns: int
    pose_valid: bool
    source_path: str


def _pose_is_valid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in value)
        and all(
            isinstance(item, (int, float)) and math.isfinite(float(item))
            for row in value
            for item in row
        )
    )


def load_candidate_frames(root: Path) -> list[CandidateFrame]:
    require(root.is_dir(), f"candidate-input root missing: {root}")
    files = sorted(root.rglob("*.json"), key=lambda path: path.as_posix())
    require(files, f"candidate-input root is empty: {root}")
    frames: list[CandidateFrame] = []
    identities: set[tuple[str, str, int]] = set()
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PairSupportError(f"candidate input unreadable: {path}") from exc
        schema = payload.get("schema")
        require(
            isinstance(schema, str) and "candidate_input" in schema,
            f"not a candidate-input record: {path}",
        )
        parent_id = payload.get("parent_id")
        video_id = payload.get("video_id")
        timestamp = payload.get("sensor_timestamp_ns")
        require(isinstance(parent_id, str) and parent_id, f"parent missing: {path}")
        require(isinstance(video_id, str) and video_id, f"video missing: {path}")
        require(
            isinstance(timestamp, int) and timestamp >= 0,
            f"timestamp invalid: {path}",
        )
        identity = (parent_id, video_id, timestamp)
        require(identity not in identities, f"duplicate frame identity: {identity}")
        identities.add(identity)
        frames.append(
            CandidateFrame(
                parent_id=parent_id,
                video_id=video_id,
                sensor_timestamp_ns=timestamp,
                pose_valid=_pose_is_valid(payload.get("camera_to_world_4x4")),
                source_path=path.relative_to(root).as_posix(),
            )
        )
    return frames


def audit_pair_support(
    label: str,
    root: Path,
    *,
    passive_window_s: float = 1.0,
    extended_window_s: float = 3.0,
) -> dict[str, Any]:
    require(label, "cohort label is empty")
    require(0.0 < passive_window_s <= extended_window_s, "pair windows invalid")
    frames = load_candidate_frames(root)
    grouped: dict[tuple[str, str], list[CandidateFrame]] = defaultdict(list)
    for frame in frames:
        grouped[(frame.parent_id, frame.video_id)].append(frame)

    passive_ns = round(passive_window_s * 1_000_000_000)
    extended_ns = round(extended_window_s * 1_000_000_000)
    adjacent_pair_count = 0
    passive_pair_count = 0
    passive_pose_pair_count = 0
    extended_pair_count = 0
    passive_parents: set[str] = set()
    positive_gaps_ns: list[int] = []
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: row.sensor_timestamp_ns)
        for previous, current in zip(ordered, ordered[1:]):
            gap_ns = current.sensor_timestamp_ns - previous.sensor_timestamp_ns
            require(gap_ns > 0, "timestamps must increase within parent/video")
            adjacent_pair_count += 1
            positive_gaps_ns.append(gap_ns)
            if gap_ns <= extended_ns:
                extended_pair_count += 1
            if gap_ns <= passive_ns:
                passive_pair_count += 1
                passive_parents.add(current.parent_id)
                if previous.pose_valid and current.pose_valid:
                    passive_pose_pair_count += 1

    identity_rows = [
        {
            "parent_id": frame.parent_id,
            "video_id": frame.video_id,
            "sensor_timestamp_ns": frame.sensor_timestamp_ns,
            "pose_valid": frame.pose_valid,
            "source_path": frame.source_path,
        }
        for frame in sorted(
            frames,
            key=lambda row: (
                row.parent_id,
                row.video_id,
                row.sensor_timestamp_ns,
            ),
        )
    ]
    return {
        "label": label,
        "candidate_input_root": root.as_posix(),
        "frame_count": len(frames),
        "parent_count": len({frame.parent_id for frame in frames}),
        "video_count": len(grouped),
        "pose_valid_frame_count": sum(frame.pose_valid for frame in frames),
        "adjacent_pair_count": adjacent_pair_count,
        "passive_window_s": passive_window_s,
        "pairs_within_passive_window": passive_pair_count,
        "pose_valid_pairs_within_passive_window": passive_pose_pair_count,
        "parents_with_passive_pair": len(passive_parents),
        "extended_window_s": extended_window_s,
        "pairs_within_extended_window": extended_pair_count,
        "minimum_positive_gap_s": (
            min(positive_gaps_ns) / 1_000_000_000 if positive_gaps_ns else None
        ),
        "candidate_identity_sequence_sha256": hashlib.sha256(
            canonical_json_bytes(identity_rows)
        ).hexdigest().upper(),
        "decision": (
            "PASSIVE_PAIR_SUPPORT_AVAILABLE"
            if passive_pose_pair_count > 0
            else "NOT_EVALUABLE_PASSIVE_WINDOW_PAIR_SUPPORT"
        ),
    }


def audit_cohorts(
    cohorts: Iterable[tuple[str, Path]],
    *,
    passive_window_s: float = 1.0,
    extended_window_s: float = 3.0,
) -> dict[str, Any]:
    rows = [
        audit_pair_support(
            label,
            root,
            passive_window_s=passive_window_s,
            extended_window_s=extended_window_s,
        )
        for label, root in cohorts
    ]
    require(rows, "at least one cohort is required")
    return {
        "schema": "blindassist.taro.task_observability_pair_support_audit.v1",
        "mode": "REVERSIBLE_EXPLORATION_SOURCE_ONLY",
        "question": (
            "Do disclosed static TARO cohorts contain pose-valid adjacent frames inside "
            "the passive-history window required by an observability experiment?"
        ),
        "cohorts": rows,
        "decision": (
            "PAIR_SUPPORT_AVAILABLE_IN_AT_LEAST_ONE_COHORT"
            if any(row["pose_valid_pairs_within_passive_window"] > 0 for row in rows)
            else "CURRENT_COHORTS_NOT_EVALUABLE_FOR_PASSIVE_OBSERVABILITY"
        ),
        "read_boundary": {
            "candidate_input_json_only": True,
            "candidate_depth_blob_reads": 0,
            "faro_highres_truth_label_outcome_reads": 0,
            "model_runs": 0,
            "network_requests": 0,
        },
        "claim_ceiling": (
            "Source pair-support capability only; not temporal utility, task accuracy, "
            "Confirmation, product, deployment, or safety evidence."
        ),
    }


def parse_cohort(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    require(bool(separator) and bool(label) and bool(raw_path), "cohort must be LABEL=PATH")
    return label, Path(raw_path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", action="append", required=True, type=parse_cohort)
    parser.add_argument("--passive-window-s", type=float, default=1.0)
    parser.add_argument("--extended-window-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_cohorts(
        args.cohort,
        passive_window_s=args.passive_window_s,
        extended_window_s=args.extended_window_s,
    )
    content = canonical_json_bytes(result)
    if args.output is not None:
        output = args.output.resolve()
        require(not output.exists(), f"output collision: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
    print(content.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

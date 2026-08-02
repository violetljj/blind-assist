#!/usr/bin/env python3
"""Build a public-real manifest from unanimous model-blind SANPO negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RGB_ROLES = (
    "RGB_REVIEWER_A",
    "RGB_REVIEWER_B",
    "RGB_REVIEWER_C",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def completed_reviews(
    paths: list[Path],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for path in paths:
        for row in load_jsonl(path):
            if (
                row.get("dataset_id") != "SANPO-Real"
                or row.get("record_kind") != "COMPLETED_REVIEW"
                or row.get("review_completed") is not True
            ):
                continue
            role = str(row["review_role"])
            if role not in RGB_ROLES:
                continue
            candidate = str(row["candidate_id"])
            role_rows = by_candidate.setdefault(candidate, {})
            if role in role_rows:
                raise ValueError(
                    f"Duplicate completed review: {candidate} {role}"
                )
            role_rows[role] = row
    return by_candidate


def unanimous_negative_candidates(
    review_rows: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    selected = []
    for candidate, roles in sorted(review_rows.items()):
        if set(roles) != set(RGB_ROLES):
            continue
        if any(
            row.get("model_output_visible") is not False
            or row.get("decision") != "REJECT"
            for row in roles.values()
        ):
            continue
        selected.append(candidate)
    return selected


def build_materialization(
    candidate_index: Path,
    review_paths: list[Path],
    staging_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    candidates = {
        str(row["candidate_id"]): row
        for row in load_jsonl(candidate_index)
    }
    reviews = completed_reviews(review_paths)
    selected = unanimous_negative_candidates(reviews)
    if not selected:
        raise ValueError("No unanimous model-blind RGB negative candidates")

    intervals: list[dict[str, Any]] = []
    observations: dict[
        tuple[str, str, str, int], dict[str, Any]
    ] = {}
    observation_count = 0
    for candidate_id in selected:
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Missing candidate index row: {candidate_id}")
        metadata = candidate["source_metadata"]
        raw_session = str(metadata["raw_source_session_id"])
        camera = str(metadata["camera"])
        view = str(metadata["view"])
        start = int(candidate["start_frame_index"])
        end = int(candidate["end_frame_index"])
        temporal_path = (
            staging_root / candidate_id / "temporal_manifest.jsonl"
        )
        temporal = load_jsonl(temporal_path)
        indices = [int(row["frame_index"]) for row in temporal]
        if (
            len(temporal) != int(candidate["frame_count"])
            or min(indices) != start
            or max(indices) != end
            or sorted(indices) != list(range(start, end + 1))
        ):
            raise ValueError(
                f"Temporal frame span mismatch: {candidate_id}"
            )
        interval_reviews = reviews[candidate_id]
        intervals.append(
            {
                "schema": (
                    "blindassist_hftf_stage_c_d6_blind_negative_"
                    "interval_v1"
                ),
                "candidate_id": candidate_id,
                "source_session_id": raw_session,
                "camera": camera,
                "view": view,
                "start_frame_index": start,
                "end_frame_index": end,
                "frame_count": len(temporal),
                "review_roles": list(RGB_ROLES),
                "review_decisions": {
                    role: interval_reviews[role]["decision"]
                    for role in RGB_ROLES
                },
                "review_buckets": {
                    role: interval_reviews[role]["event_bucket"]
                    for role in RGB_ROLES
                },
                "model_output_visible": False,
                "scientific_label": (
                    "UNANIMOUS_MODEL_BLIND_RGB_ACTIONABLE_NEGATIVE"
                ),
                "system_event_truth_authority": False,
            }
        )
        for row in temporal:
            observation_count += 1
            frame_index = int(row["frame_index"])
            rgb_path = Path(row["rgb_path"])
            expected_sha = str(row["rgb_sha256"]).lower()
            if not rgb_path.is_file():
                raise ValueError(f"Missing RGB frame: {rgb_path}")
            actual_sha = sha256(rgb_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"RGB hash mismatch: {candidate_id} {frame_index}"
                )
            key = (raw_session, camera, view, frame_index)
            existing = observations.get(key)
            if existing is not None:
                if existing["rgb_sha256"] != expected_sha:
                    raise ValueError(
                        f"Overlapping RGB mismatch: {key}"
                    )
                existing["blind_review_candidate_ids"].add(
                    candidate_id
                )
                existing["rgb_paths"].add(str(rgb_path.resolve()))
                continue
            observations[key] = {
                "source_session_id": raw_session,
                "camera": camera,
                "view": view,
                "frame_index": frame_index,
                "nominal_time_ns": row.get("nominal_time_ns"),
                "timestamp_ns": None,
                "time_semantics": candidate["timestamp_semantics"],
                "rgb_sha256": expected_sha,
                "rgb_paths": {str(rgb_path.resolve())},
                "blind_review_candidate_ids": {candidate_id},
            }

    media_rows = []
    for key in sorted(observations):
        row = observations[key]
        media_rows.append(
            {
                "schema": (
                    "blindassist_hftf_stage_c_d6_blind_negative_"
                    "media_frame_v1"
                ),
                "source_session_id": row["source_session_id"],
                "camera": row["camera"],
                "view": row["view"],
                "frame_index": row["frame_index"],
                "nominal_time_ns": row["nominal_time_ns"],
                "timestamp_ns": row["timestamp_ns"],
                "time_semantics": row["time_semantics"],
                "rgb_local_path": sorted(row["rgb_paths"])[0],
                "rgb_sha256": row["rgb_sha256"],
                "blind_review_candidate_ids": sorted(
                    row["blind_review_candidate_ids"]
                ),
                "scientific_label": (
                    "UNANIMOUS_MODEL_BLIND_RGB_ACTIONABLE_NEGATIVE"
                ),
                "system_event_truth_authority": False,
            }
        )

    interval_lookup: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = {}
    for interval in intervals:
        key = (
            interval["source_session_id"],
            interval["camera"],
            interval["view"],
        )
        interval_lookup.setdefault(key, []).append(interval)
    by_group: dict[
        tuple[str, str, str], dict[int, dict[str, Any]]
    ] = {}
    for row in media_rows:
        key = (
            row["source_session_id"],
            row["camera"],
            row["view"],
        )
        by_group.setdefault(key, {})[int(row["frame_index"])] = row
    windows = []
    for key, frames in sorted(by_group.items()):
        for anchor in sorted(frames):
            indices = list(range(anchor - 4, anchor + 1))
            if not all(index in frames for index in indices):
                continue
            covering = [
                interval["candidate_id"]
                for interval in interval_lookup[key]
                if interval["start_frame_index"] <= indices[0]
                and interval["end_frame_index"] >= indices[-1]
            ]
            if not covering:
                raise ValueError(
                    "Contiguous output window lacks unanimous review "
                    f"coverage: {key} anchor={anchor}"
                )
            windows.append(
                {
                    "schema": (
                        "blindassist_hftf_stage_c_d6_blind_negative_"
                        "window_evidence_v1"
                    ),
                    "source_session_id": key[0],
                    "camera": key[1],
                    "view": key[2],
                    "history_frame_indices": indices,
                    "anchor_frame_index": anchor,
                    "covering_candidate_ids": sorted(covering),
                    "scientific_label": (
                        "UNANIMOUS_MODEL_BLIND_RGB_ACTIONABLE_NEGATIVE"
                    ),
                    "system_event_truth_authority": False,
                }
            )

    report = {
        "schema": (
            "blindassist_hftf_stage_c_d6_blind_negative_"
            "materialization_v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SANPO_BLIND_REVIEWED_NEGATIVE_MEDIA_READY",
        "candidate_count": len(intervals),
        "source_session_count": len(
            {row["source_session_id"] for row in media_rows}
        ),
        "input_frame_observation_count": observation_count,
        "unique_frame_count": len(media_rows),
        "overlap_observation_count": observation_count - len(media_rows),
        "covered_five_frame_window_count": len(windows),
        "review_rule": (
            "3/3 completed RGB reviewers, model output hidden, "
            "all decisions REJECT"
        ),
        "evidence_boundary": {
            "clip_actionable_negative_support": True,
            "cell_localization_truth": False,
            "positive_event_safety": False,
            "system_event_truth_authority": False,
        },
    }
    return media_rows, intervals, windows, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument(
        "--rgb-review",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError("Refusing to overwrite blind-negative output")

    media, intervals, windows, report = build_materialization(
        args.candidate_index,
        args.rgb_review,
        args.staging_root,
    )
    args.output_root.mkdir(parents=True)
    media_path = args.output_root / "media_manifest.jsonl"
    interval_path = args.output_root / "review_intervals.jsonl"
    window_path = args.output_root / "window_evidence.jsonl"
    write_jsonl(media_path, media)
    write_jsonl(interval_path, intervals)
    write_jsonl(window_path, windows)
    report["inputs"] = {
        "candidate_index": {
            "path": str(args.candidate_index.resolve()),
            "sha256": sha256(args.candidate_index),
        },
        "rgb_reviews": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for path in args.rgb_review
        ],
        "staging_root": str(args.staging_root.resolve()),
    }
    report["outputs"] = {
        "media_manifest": {
            "path": str(media_path.resolve()),
            "sha256": sha256(media_path),
        },
        "review_intervals": {
            "path": str(interval_path.resolve()),
            "sha256": sha256(interval_path),
        },
        "window_evidence": {
            "path": str(window_path.resolve()),
            "sha256": sha256(window_path),
        },
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

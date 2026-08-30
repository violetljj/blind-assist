"""One-shot evaluator-side score for the CARLA C2 X23/X24 comparison.

The scorer intentionally has only two evidence inputs: an already sealed X24
prediction run root and the joined C2 evaluator root.  Both expected seals are
supplied on the command line.  Prediction and freeze identity are checked
first, then an ``O_EXCL`` score-attempt receipt is durably written.  Evaluator
manifests or truth rows are not opened until after that receipt exists.

This is a same-source scripted-CARLA Development score.  It does not claim a
blind score, source-disjoint confirmation, real-world validity, or product
safety.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import statistics
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPERIMENT_ID = "DTR_CARLA_X24_PLAN_ADHERENT_METRIC_TRACK"
PREDICTION_SCHEMA = "blindassist-dtr-carla-x23-x24-predictions-v1"
FREEZE_SCHEMA = "blindassist-dtr-carla-x24-freeze-v1"
SCORE_ATTEMPT_SCHEMA = "blindassist-dtr-carla-c2-x24-score-attempt-v1"
RESULT_SCHEMA = "blindassist-dtr-carla-c2-x24-score-result-v1"

ARM_X23 = "X23_OBSERVED_CV_ROUTE"
ARM_X24 = "X24_ISSUED_PLAN_ADHERENCE"
REQUIRED_EPISODES = ("ep_01", "ep_02", "ep_03", "ep_04")
CONTACT_EPISODE = "ep_01"
SAFE_TWIN_EPISODE = "ep_02"
NO_PLAN_EPISODES = ("ep_03", "ep_04")
OCCLUSION_CONTRACT_ID = "track_then_physical_loss_pair_01"

# Frozen, decision-changing score contract.  Time is always read dynamically;
# these values describe metric-duration windows, never frame or pixel counts.
ELIGIBLE_FULL_OCCLUSION_SECONDS = 0.40
TWIN_FORK_SECONDS = 2.00
SAFE_TAIL_GRACE_SECONDS = 0.70
SAFE_TAIL_START_SECONDS = TWIN_FORK_SECONDS + SAFE_TAIL_GRACE_SECONDS
MAXIMUM_HOLD_EVIDENCE_AGE_SECONDS = 0.60
MINIMUM_X24_CONTACT_LEAD_SECONDS = 0.40
EPSILON = 1e-9
TIME_TOLERANCE_SECONDS = 1e-6

ROUTE_MODE_OBSERVED_CV = "OBSERVED_CV_FALLBACK"
AUTHORITY_NO_PLAN = "NO_PLAN"

PREDICTIONS_NAME = "predictions-x24.json"
FREEZE_NAME = "freeze-x24.json"
ATTEMPT_NAME = "score-attempt-x24.json"
RESULT_NAME = "result-x24.json"
SVG_NAME = "result-x24.svg"
PNG_NAME = "result-x24.png"
EVIDENCE_MANIFEST_NAME = "sealed_evidence_manifest.json"


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def require_digest(value: str, label: str) -> str:
    digest = str(value).upper()
    require(
        len(digest) == 64 and all(character in "0123456789ABCDEF" for character in digest),
        f"{label}:not_sha256",
    )
    return digest


def finite_number(value: Any, label: str) -> float:
    require(not isinstance(value, bool), f"{label}:bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label}:not_numeric") from error
    require(math.isfinite(result), f"{label}:not_finite")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json_value(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label}:unreadable:{path}") from error


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json_value(path, label)
    require(isinstance(value, dict), f"{label}:not_object")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                require(bool(line.strip()), f"{label}:blank_line:{line_number}")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"{label}:invalid_json:{line_number}") from error
                require(isinstance(value, dict), f"{label}:not_object:{line_number}")
                rows.append(value)
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"{label}:unreadable:{path}") from error
    require(bool(rows), f"{label}:empty")
    return rows


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_json_exclusive(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    write_bytes_exclusive(path, payload)


def output_paths(predictions_root: Path) -> dict[str, Path]:
    return {
        "attempt": predictions_root / ATTEMPT_NAME,
        "result": predictions_root / RESULT_NAME,
        "svg": predictions_root / SVG_NAME,
        "png": predictions_root / PNG_NAME,
    }


def predeclared_contract() -> dict[str, Any]:
    return {
        "contact_episode_id": CONTACT_EPISODE,
        "safe_twin_episode_id": SAFE_TWIN_EPISODE,
        "no_plan_episode_ids": list(NO_PLAN_EPISODES),
        "eligible_full_occlusion_seconds": ELIGIBLE_FULL_OCCLUSION_SECONDS,
        "eligible_window_selection": "EARLIEST_0.40S_OF_EVALUATOR_SELECTED_ELIGIBLE_RUN",
        "twin_fork_seconds": TWIN_FORK_SECONDS,
        "safe_tail_grace_seconds": SAFE_TAIL_GRACE_SECONDS,
        "safe_tail_starts_at_seconds_inclusive": SAFE_TAIL_START_SECONDS,
        "safe_false_alert_definition": "X24_ROUTE_RISK_AND_NOT_X23_ROUTE_RISK_AT_OR_AFTER_FORK_PLUS_GRACE",
        "maximum_new_safe_false_alert_segments": 0,
        "maximum_hold_evidence_age_seconds": MAXIMUM_HOLD_EVIDENCE_AGE_SECONDS,
        "minimum_x24_contact_first_alert_lead_seconds": MINIMUM_X24_CONTACT_LEAD_SECONDS,
        "require_x24_full_contact_occlusion_coverage": True,
        "require_positive_x24_contact_benefit_over_x23": True,
        "positive_benefit_definition": "EARLIER_FIRST_ALERT_OR_HIGHER_CONTACT_OCCLUSION_COVERAGE_OR_X23_NO_ALERT",
        "require_no_plan_framewise_route_risk_identity": True,
    }


def contained_existing_file(root: Path, relative_path: str, label: str) -> Path:
    relative = Path(relative_path)
    require(not relative.is_absolute(), f"{label}:absolute")
    resolved = (root / relative).resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), f"{label}:escape")
    return resolved


def validate_prediction_frame(
    episode_id: str,
    ordinal: int,
    frame: Any,
) -> None:
    label = f"prediction_frame:{episode_id}:{ordinal}"
    require(isinstance(frame, Mapping), f"{label}:not_object")
    require(type(frame.get("sample_index")) is int, f"{label}:sample_index")
    finite_number(frame.get("time_s"), f"{label}:time_s")
    arms = frame.get("arms")
    require(isinstance(arms, Mapping), f"{label}:arms")
    for arm in (ARM_X23, ARM_X24):
        value = arms.get(arm)
        require(isinstance(value, Mapping), f"{label}:arm:{arm}")
        require(type(value.get("route_risk")) is bool, f"{label}:route_risk:{arm}")
        require(isinstance(value.get("route_mode"), str), f"{label}:route_mode:{arm}")
        require(isinstance(value.get("authority"), str), f"{label}:authority:{arm}")
    tracks = frame.get("tracks")
    require(isinstance(tracks, list), f"{label}:tracks")
    seen_tracks: set[str] = set()
    for track_number, track in enumerate(tracks):
        track_label = f"{label}:track:{track_number}"
        require(isinstance(track, Mapping), f"{track_label}:not_object")
        track_id = str(track.get("track_id", ""))
        require(bool(track_id) and track_id not in seen_tracks, f"{track_label}:track_id")
        seen_tracks.add(track_id)
        require(track.get("disposition") in {"MEASURED", "HOLD"}, f"{track_label}:disposition")
        age_s = finite_number(track.get("evidence_age_s"), f"{track_label}:evidence_age_s")
        require(age_s >= -EPSILON, f"{track_label}:negative_age")


def validate_predictions_before_attempt(
    predictions_root: Path,
    expected_predictions_sha256: str,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    root = predictions_root.resolve(strict=True)
    require(root.is_dir(), f"predictions_root_not_directory:{root}")
    predictions_path = contained_existing_file(root, PREDICTIONS_NAME, "predictions")
    expected_digest = require_digest(expected_predictions_sha256, "expected_predictions_sha256")
    actual_digest = sha256_file(predictions_path)
    require(actual_digest == expected_digest, "predictions_sha256_drift")
    predictions = read_json_object(predictions_path, "predictions")
    require(predictions.get("schema") == PREDICTION_SCHEMA, "prediction_schema")
    require(
        predictions.get("status") == "SEALED_TRUTH_BLIND_PENDING_SCORE",
        "prediction_status",
    )
    require(predictions.get("experiment_id") == EXPERIMENT_ID, "prediction_experiment")
    require(predictions.get("truth_blind") is True, "prediction_boundary_flag")
    require(predictions.get("arms") == [ARM_X23, ARM_X24], "prediction_arms")
    fixed = predictions.get("fixed_constants")
    require(isinstance(fixed, Mapping), "prediction_fixed_constants")
    require(
        abs(finite_number(fixed.get("hold_window_seconds"), "prediction_hold_window")
            - MAXIMUM_HOLD_EVIDENCE_AGE_SECONDS)
        <= EPSILON,
        "prediction_hold_window_drift",
    )
    episodes = predictions.get("episodes")
    require(isinstance(episodes, Mapping), "prediction_episodes")
    require(set(episodes) == set(REQUIRED_EPISODES), "prediction_episode_set")
    for episode_id in REQUIRED_EPISODES:
        episode = episodes[episode_id]
        require(isinstance(episode, Mapping), f"prediction_episode:{episode_id}")
        require(episode.get("episode_id") == episode_id, f"prediction_episode_identity:{episode_id}")
        frames = episode.get("frames")
        require(isinstance(frames, list) and len(frames) >= 2, f"prediction_frames:{episode_id}")
        previous_index: int | None = None
        previous_time: float | None = None
        for ordinal, frame in enumerate(frames):
            validate_prediction_frame(episode_id, ordinal, frame)
            sample_index = int(frame["sample_index"])
            time_s = finite_number(frame["time_s"], f"prediction_time:{episode_id}:{ordinal}")
            if previous_index is not None:
                require(sample_index > previous_index, f"prediction_index_order:{episode_id}:{ordinal}")
                require(time_s > float(previous_time), f"prediction_time_order:{episode_id}:{ordinal}")
            previous_index = sample_index
            previous_time = time_s

    source = predictions.get("source")
    require(isinstance(source, Mapping), "prediction_source")
    freeze_path = contained_existing_file(root, FREEZE_NAME, "freeze")
    freeze_digest = sha256_file(freeze_path)
    require(
        freeze_digest == require_digest(str(source.get("freeze_sha256", "")), "prediction_freeze_sha256"),
        "prediction_freeze_sha256_drift",
    )
    freeze = read_json_object(freeze_path, "freeze")
    require(freeze.get("schema") == FREEZE_SCHEMA, "freeze_schema")
    require(
        freeze.get("status") == "FROZEN_TRUTH_BLIND_PENDING_PREDICTION",
        "freeze_status",
    )
    require(freeze.get("truth_blind") is True, "freeze_boundary_flag")
    require(freeze.get("experiment_id") == EXPERIMENT_ID, "freeze_experiment")
    manifest = freeze.get("model_manifest")
    require(isinstance(manifest, Mapping), "freeze_model_manifest")
    freeze_model_digest = require_digest(
        str(manifest.get("sha256", "")),
        "freeze_model_manifest_sha256",
    )
    require(
        freeze_model_digest
        == require_digest(str(source.get("model_manifest_sha256", "")), "prediction_model_manifest_sha256"),
        "prediction_model_manifest_identity",
    )
    return predictions_path, predictions, freeze_path, freeze


def evidence_entries(value: Any) -> dict[str, dict[str, Any]]:
    require(isinstance(value, list) and bool(value), "evidence_manifest_not_nonempty_list")
    output: dict[str, dict[str, Any]] = {}
    for ordinal, entry in enumerate(value):
        require(isinstance(entry, Mapping), f"evidence_manifest_entry:{ordinal}")
        raw_path = str(entry.get("path", "")).replace("\\", "/")
        relative = Path(raw_path)
        require(bool(raw_path) and not relative.is_absolute(), f"evidence_manifest_path:{ordinal}")
        require(".." not in relative.parts, f"evidence_manifest_parent:{ordinal}")
        normalized = relative.as_posix()
        require(normalized == raw_path and normalized not in output, f"evidence_manifest_duplicate:{ordinal}")
        size = entry.get("bytes")
        require(type(size) is int and size >= 0, f"evidence_manifest_bytes:{ordinal}")
        digest = require_digest(str(entry.get("sha256", "")), f"evidence_manifest_sha256:{ordinal}")
        output[normalized] = {"path": normalized, "bytes": size, "sha256": digest}
    return output


def verify_used_evaluator_file(
    joined_root: Path,
    evaluator_root: Path,
    entries: Mapping[str, Mapping[str, Any]],
    evaluator_relative: str,
) -> tuple[Path, dict[str, Any]]:
    manifest_relative = f"evaluator/{evaluator_relative}"
    require(manifest_relative in entries, f"evidence_manifest_missing:{manifest_relative}")
    path = contained_existing_file(evaluator_root, evaluator_relative, f"evaluator_file:{evaluator_relative}")
    require(path.is_relative_to(joined_root), f"evaluator_file_outside_joined_root:{evaluator_relative}")
    reference = dict(entries[manifest_relative])
    require(path.stat().st_size == int(reference["bytes"]), f"evaluator_bytes_drift:{evaluator_relative}")
    require(sha256_file(path) == reference["sha256"], f"evaluator_sha256_drift:{evaluator_relative}")
    return path, reference


def validate_evaluator_row(episode_id: str, ordinal: int, row: Any) -> None:
    label = f"evaluator_frame:{episode_id}:{ordinal}"
    require(isinstance(row, Mapping), f"{label}:not_object")
    require(row.get("schema_version") == "dtr-c2-evaluator-frame-v1", f"{label}:schema")
    require(row.get("episode_id") == episode_id, f"{label}:episode")
    require(type(row.get("sample_index")) is int, f"{label}:sample_index")
    finite_number(row.get("time_s"), f"{label}:time_s")
    require(isinstance(row.get("layout_id"), str), f"{label}:layout_id")
    require(isinstance(row.get("instance"), Mapping), f"{label}:instance")
    require(isinstance(row.get("witness"), Mapping), f"{label}:witness")
    require(isinstance(row.get("camera_transform"), Mapping), f"{label}:camera_transform")
    require(isinstance(row.get("actors"), Mapping), f"{label}:actors")
    require(isinstance(row.get("instance_visibility"), Mapping), f"{label}:instance_visibility")
    truth = row.get("truth")
    require(isinstance(truth, Mapping), f"{label}:truth")
    require(isinstance(truth.get("scenario_role"), str), f"{label}:scenario_role")
    require(isinstance(truth.get("twin_role"), str), f"{label}:twin_role")
    require(truth.get("expected_outcome") in {"CONTACT", "SAFE"}, f"{label}:expected_outcome")
    require(isinstance(truth.get("expected_responsible_assets"), list), f"{label}:expected_assets")
    require(type(truth.get("current_contact")) is bool, f"{label}:current_contact")
    finite_number(truth.get("minimum_distance_m"), f"{label}:minimum_distance")
    require(isinstance(truth.get("responsible_assets"), list), f"{label}:responsible_assets")
    require(isinstance(truth.get("collision_polygons_xy"), Mapping), f"{label}:collision_polygons")
    require(type(truth.get("future_contact_within_horizon")) is bool, f"{label}:future_contact")
    realized = truth.get("realized_time_to_contact_seconds")
    if realized is not None:
        require(finite_number(realized, f"{label}:realized_ttc") >= -EPSILON, f"{label}:negative_ttc")


def uniform_sample_period(rows: Sequence[Mapping[str, Any]], label: str) -> float:
    require(len(rows) >= 2, f"{label}:too_short")
    differences = [
        finite_number(rows[index]["time_s"], f"{label}:time:{index}")
        - finite_number(rows[index - 1]["time_s"], f"{label}:time:{index - 1}")
        for index in range(1, len(rows))
    ]
    require(all(value > 0.0 for value in differences), f"{label}:nonpositive_period")
    period = statistics.median(differences)
    require(
        all(abs(value - period) <= TIME_TOLERANCE_SECONDS for value in differences),
        f"{label}:nonuniform_period",
    )
    return period


def load_verified_evaluator(
    evaluator_root_lexical: Path,
    expected_evidence_manifest_sha256: str,
    predictions: Mapping[str, Any],
) -> tuple[
    Path,
    Path,
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    # The first filesystem open below is intentionally after the caller has
    # durably created the score-attempt receipt.
    evaluator_root = evaluator_root_lexical.resolve(strict=True)
    require(evaluator_root.is_dir() and evaluator_root.name == "evaluator", "evaluator_root_contract")
    joined_root = evaluator_root.parent
    evidence_manifest_path = (joined_root / EVIDENCE_MANIFEST_NAME).resolve(strict=True)
    require(evidence_manifest_path.is_file(), "evidence_manifest_missing")
    expected_digest = require_digest(
        expected_evidence_manifest_sha256,
        "expected_evidence_manifest_sha256",
    )
    require(sha256_file(evidence_manifest_path) == expected_digest, "evidence_manifest_sha256_drift")
    entries = evidence_entries(read_json_value(evidence_manifest_path, "evidence_manifest"))

    used_references: dict[str, dict[str, Any]] = {}
    outcome_path, used_references["outcome_summary.json"] = verify_used_evaluator_file(
        joined_root,
        evaluator_root,
        entries,
        "outcome_summary.json",
    )
    occlusion_path, used_references["physical_occlusion_report.json"] = verify_used_evaluator_file(
        joined_root,
        evaluator_root,
        entries,
        "physical_occlusion_report.json",
    )
    rows_by_episode: dict[str, list[dict[str, Any]]] = {}
    for episode_id in REQUIRED_EPISODES:
        relative = f"episodes/{episode_id}/frames.jsonl"
        path, used_references[relative] = verify_used_evaluator_file(
            joined_root,
            evaluator_root,
            entries,
            relative,
        )
        rows = read_jsonl(path, f"evaluator_rows:{episode_id}")
        previous_index: int | None = None
        previous_time: float | None = None
        for ordinal, row in enumerate(rows):
            validate_evaluator_row(episode_id, ordinal, row)
            sample_index = int(row["sample_index"])
            time_s = finite_number(row["time_s"], f"evaluator_time:{episode_id}:{ordinal}")
            if previous_index is not None:
                require(sample_index > previous_index, f"evaluator_index_order:{episode_id}:{ordinal}")
                require(time_s > float(previous_time), f"evaluator_time_order:{episode_id}:{ordinal}")
            previous_index = sample_index
            previous_time = time_s
        rows_by_episode[episode_id] = rows

    outcome_summaries = read_json_value(outcome_path, "outcome_summary")
    require(isinstance(outcome_summaries, list), "outcome_summary_not_list")
    occlusion_reports = read_json_value(occlusion_path, "physical_occlusion_report")
    require(isinstance(occlusion_reports, list), "occlusion_report_not_list")

    for episode_id in REQUIRED_EPISODES:
        prediction_frames = predictions["episodes"][episode_id]["frames"]
        evaluator_frames = rows_by_episode[episode_id]
        require(len(prediction_frames) == len(evaluator_frames), f"prediction_evaluator_frame_count:{episode_id}")
        for ordinal, (prediction, evaluator) in enumerate(zip(prediction_frames, evaluator_frames, strict=True)):
            require(
                int(prediction["sample_index"]) == int(evaluator["sample_index"]),
                f"prediction_evaluator_sample:{episode_id}:{ordinal}",
            )
            require(
                abs(float(prediction["time_s"]) - float(evaluator["time_s"])) <= TIME_TOLERANCE_SECONDS,
                f"prediction_evaluator_time:{episode_id}:{ordinal}",
            )
        prediction_period = uniform_sample_period(prediction_frames, f"prediction_period:{episode_id}")
        evaluator_period = uniform_sample_period(evaluator_frames, f"evaluator_period:{episode_id}")
        require(
            abs(prediction_period - evaluator_period) <= TIME_TOLERANCE_SECONDS,
            f"prediction_evaluator_period:{episode_id}",
        )

    return (
        evaluator_root,
        evidence_manifest_path,
        used_references,
        entries,
        rows_by_episode,
        outcome_summaries,
        occlusion_reports,
    )


def validate_outcomes(
    outcome_summaries: Sequence[Mapping[str, Any]],
    rows_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    require(len(outcome_summaries) == len(REQUIRED_EPISODES), "outcome_summary_count")
    values: dict[str, dict[str, Any]] = {}
    for ordinal, raw in enumerate(outcome_summaries):
        require(isinstance(raw, Mapping), f"outcome_summary_entry:{ordinal}")
        episode_id = str(raw.get("episode_id", ""))
        require(episode_id in REQUIRED_EPISODES and episode_id not in values, f"outcome_episode:{ordinal}")
        require(raw.get("expected_outcome") in {"CONTACT", "SAFE"}, f"outcome_expected:{episode_id}")
        require(raw.get("observed_outcome") in {"CONTACT", "SAFE"}, f"outcome_observed:{episode_id}")
        require(raw.get("expected_outcome") == raw.get("observed_outcome"), f"outcome_mismatch:{episode_id}")
        require(type(raw.get("frames")) is int, f"outcome_frames:{episode_id}")
        require(int(raw["frames"]) == len(rows_by_episode[episode_id]), f"outcome_frame_count:{episode_id}")
        require(isinstance(raw.get("expected_responsible_assets"), list), f"outcome_expected_assets:{episode_id}")
        require(isinstance(raw.get("observed_responsible_assets"), list), f"outcome_observed_assets:{episode_id}")
        require(
            sorted(map(str, raw["expected_responsible_assets"]))
            == sorted(map(str, raw["observed_responsible_assets"])),
            f"outcome_asset_mismatch:{episode_id}",
        )
        truth_contacts = [
            float(row["time_s"])
            for row in rows_by_episode[episode_id]
            if row["truth"]["current_contact"] is True
        ]
        recorded_contact = raw.get("first_contact_time_s")
        if truth_contacts:
            require(recorded_contact is not None, f"outcome_missing_contact_time:{episode_id}")
            require(
                abs(finite_number(recorded_contact, f"outcome_contact_time:{episode_id}") - min(truth_contacts))
                <= TIME_TOLERANCE_SECONDS,
                f"outcome_contact_time_mismatch:{episode_id}",
            )
        else:
            require(recorded_contact is None, f"outcome_unexpected_contact_time:{episode_id}")
        values[episode_id] = dict(raw)

    require(set(values) == set(REQUIRED_EPISODES), "outcome_episode_set")
    require(values[CONTACT_EPISODE]["observed_outcome"] == "CONTACT", "contact_episode_outcome")
    for episode_id in (SAFE_TWIN_EPISODE, *NO_PLAN_EPISODES):
        require(values[episode_id]["observed_outcome"] == "SAFE", f"safe_episode_outcome:{episode_id}")
    return values


def selected_occlusion_windows(
    occlusion_reports: Sequence[Mapping[str, Any]],
    rows_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    matching = [value for value in occlusion_reports if value.get("contract_id") == OCCLUSION_CONTRACT_ID]
    require(len(matching) == 1, "occlusion_contract_identity")
    report = matching[0]
    require(report.get("passed") is True, "occlusion_contract_not_passed")
    require(report.get("pair_occlusion_indices_identical") is True, "occlusion_pair_not_identical")
    episode_reports = report.get("episodes")
    selected_indices = report.get("selected_indices")
    require(isinstance(episode_reports, Mapping), "occlusion_episode_reports")
    require(isinstance(selected_indices, Mapping), "occlusion_selected_indices")

    output: dict[str, dict[str, Any]] = {}
    pair_indices: list[tuple[int, ...]] = []
    for episode_id in (CONTACT_EPISODE, SAFE_TWIN_EPISODE):
        episode_report = episode_reports.get(episode_id)
        require(isinstance(episode_report, Mapping), f"occlusion_episode:{episode_id}")
        require(episode_report.get("passed") is True, f"occlusion_episode_failed:{episode_id}")
        selected = episode_report.get("selected")
        require(isinstance(selected, Mapping) and selected.get("passed") is True, f"occlusion_selected:{episode_id}")
        raw_indices = selected.get("sample_indices")
        require(isinstance(raw_indices, list) and raw_indices, f"occlusion_indices:{episode_id}")
        require(all(type(value) is int for value in raw_indices), f"occlusion_index_type:{episode_id}")
        require(raw_indices == selected_indices.get(episode_id), f"occlusion_selected_identity:{episode_id}")
        pair_indices.append(tuple(raw_indices))

        rows = rows_by_episode[episode_id]
        row_indices = [int(row["sample_index"]) for row in rows]
        positions = [row_indices.index(value) if value in row_indices else -1 for value in raw_indices]
        require(all(value >= 0 for value in positions), f"occlusion_index_missing:{episode_id}")
        require(
            positions == list(range(positions[0], positions[0] + len(positions))),
            f"occlusion_not_contiguous:{episode_id}",
        )
        period_s = uniform_sample_period(rows, f"occlusion_period:{episode_id}")
        recorded_duration = finite_number(selected.get("duration_seconds"), f"occlusion_duration:{episode_id}")
        require(
            abs(recorded_duration - len(raw_indices) * period_s) <= TIME_TOLERANCE_SECONDS,
            f"occlusion_duration_mismatch:{episode_id}",
        )
        window_frames = round(ELIGIBLE_FULL_OCCLUSION_SECONDS / period_s)
        require(window_frames > 0, f"occlusion_window_frames:{episode_id}")
        require(
            abs(window_frames * period_s - ELIGIBLE_FULL_OCCLUSION_SECONDS)
            <= TIME_TOLERANCE_SECONDS,
            f"occlusion_0_40_not_representable:{episode_id}",
        )
        require(len(raw_indices) >= window_frames, f"occlusion_0_40_unavailable:{episode_id}")
        window = list(raw_indices[:window_frames])
        output[episode_id] = {
            "sample_period_seconds": period_s,
            "selected_eligible_run_sample_indices": list(raw_indices),
            "selected_eligible_run_duration_seconds": recorded_duration,
            "coverage_window_sample_indices": window,
            "coverage_window_seconds": window_frames * period_s,
            "coverage_window_selection": "EARLIEST_FIXED_DURATION_SUBWINDOW",
        }
    require(len(set(pair_indices)) == 1, "occlusion_pair_indices_drift")
    return output


def risk(frame: Mapping[str, Any], arm: str) -> bool:
    return frame["arms"][arm]["route_risk"] is True


def first_alert_time(frames: Sequence[Mapping[str, Any]], arm: str) -> float | None:
    return next((float(frame["time_s"]) for frame in frames if risk(frame, arm)), None)


def gap_coverage(
    frames: Sequence[Mapping[str, Any]],
    sample_indices: Sequence[int],
    arm: str,
) -> dict[str, Any]:
    by_index = {int(frame["sample_index"]): frame for frame in frames}
    require(all(index in by_index for index in sample_indices), f"gap_prediction_index_missing:{arm}")
    hits = [index for index in sample_indices if risk(by_index[index], arm)]
    return {
        "hit_sample_indices": hits,
        "hit_frames": len(hits),
        "eligible_frames": len(sample_indices),
        "coverage": len(hits) / len(sample_indices),
    }


def contiguous_new_false_segments(
    frames: Sequence[Mapping[str, Any]],
    sample_period_s: float,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    new_tail_ordinals: list[int] = []
    ignored_before_tail: list[int] = []
    common_tail: list[int] = []
    for ordinal, frame in enumerate(frames):
        time_s = float(frame["time_s"])
        x23 = risk(frame, ARM_X23)
        x24 = risk(frame, ARM_X24)
        if time_s + EPSILON < SAFE_TAIL_START_SECONDS:
            if x24 and not x23:
                ignored_before_tail.append(int(frame["sample_index"]))
            continue
        if x24 and not x23:
            new_tail_ordinals.append(ordinal)
        elif x24 and x23:
            common_tail.append(int(frame["sample_index"]))

    groups: list[list[int]] = []
    for ordinal in new_tail_ordinals:
        if not groups or ordinal != groups[-1][-1] + 1:
            groups.append([ordinal])
        else:
            groups[-1].append(ordinal)
    segments = [
        {
            "start_sample_index": int(frames[group[0]]["sample_index"]),
            "end_sample_index": int(frames[group[-1]]["sample_index"]),
            "start_time_s": float(frames[group[0]]["time_s"]),
            "end_time_s": float(frames[group[-1]]["time_s"]),
            "frames": len(group),
            "duration_seconds": len(group) * sample_period_s,
            "sample_indices": [int(frames[index]["sample_index"]) for index in group],
        }
        for group in groups
    ]
    return segments, ignored_before_tail, common_tail


def hold_metrics(predictions: Mapping[str, Any]) -> dict[str, Any]:
    hold_states = 0
    maximum_age_s: float | None = None
    violations: list[dict[str, Any]] = []
    for episode_id in REQUIRED_EPISODES:
        for frame in predictions["episodes"][episode_id]["frames"]:
            for track in frame["tracks"]:
                if track["disposition"] != "HOLD":
                    continue
                hold_states += 1
                age_s = float(track["evidence_age_s"])
                maximum_age_s = age_s if maximum_age_s is None else max(maximum_age_s, age_s)
                if age_s > MAXIMUM_HOLD_EVIDENCE_AGE_SECONDS + EPSILON:
                    if len(violations) < 100:
                        violations.append(
                            {
                                "episode_id": episode_id,
                                "sample_index": int(frame["sample_index"]),
                                "track_id": str(track["track_id"]),
                                "evidence_age_s": age_s,
                            }
                        )
    return {
        "hold_states": hold_states,
        "maximum_hold_evidence_age_seconds": maximum_age_s,
        "limit_seconds": MAXIMUM_HOLD_EVIDENCE_AGE_SECONDS,
        "violations": violations,
    }


def no_plan_metrics(predictions: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for episode_id in NO_PLAN_EPISODES:
        mismatches: list[int] = []
        authority_or_mode_mismatches: list[int] = []
        frames = predictions["episodes"][episode_id]["frames"]
        for frame in frames:
            if risk(frame, ARM_X23) != risk(frame, ARM_X24):
                mismatches.append(int(frame["sample_index"]))
            x23 = frame["arms"][ARM_X23]
            x24 = frame["arms"][ARM_X24]
            if not (
                x23["route_mode"] == ROUTE_MODE_OBSERVED_CV
                and x24["route_mode"] == ROUTE_MODE_OBSERVED_CV
                and x24["authority"] == AUTHORITY_NO_PLAN
            ):
                authority_or_mode_mismatches.append(int(frame["sample_index"]))
        output[episode_id] = {
            "frames": len(frames),
            "framewise_route_risk_equal": not mismatches,
            "mismatch_sample_indices": mismatches,
            "x24_no_plan_observed_cv_fallback_every_frame": not authority_or_mode_mismatches,
            "authority_or_mode_mismatch_sample_indices": authority_or_mode_mismatches,
        }
    return output


def score_metrics(
    predictions: Mapping[str, Any],
    rows_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
    outcomes: Mapping[str, Mapping[str, Any]],
    occlusion_windows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    contact_frames = predictions["episodes"][CONTACT_EPISODE]["frames"]
    first_contact_s = finite_number(
        outcomes[CONTACT_EPISODE]["first_contact_time_s"],
        "contact_first_time",
    )
    contact: dict[str, Any] = {
        "episode_id": CONTACT_EPISODE,
        "first_contact_time_s": first_contact_s,
        "arms": {},
    }
    for arm in (ARM_X23, ARM_X24):
        alert_s = first_alert_time(contact_frames, arm)
        contact["arms"][arm] = {
            "first_alert_time_s": alert_s,
            "first_alert_lead_seconds": None if alert_s is None else first_contact_s - alert_s,
        }

    coverage: dict[str, Any] = {}
    for episode_id in (CONTACT_EPISODE, SAFE_TWIN_EPISODE):
        window = list(occlusion_windows[episode_id]["coverage_window_sample_indices"])
        frames = predictions["episodes"][episode_id]["frames"]
        coverage[episode_id] = {
            **dict(occlusion_windows[episode_id]),
            "arms": {
                arm: gap_coverage(frames, window, arm)
                for arm in (ARM_X23, ARM_X24)
            },
        }
        coverage[episode_id]["x24_minus_x23_coverage"] = (
            coverage[episode_id]["arms"][ARM_X24]["coverage"]
            - coverage[episode_id]["arms"][ARM_X23]["coverage"]
        )
        coverage[episode_id]["x24_minus_x23_percentage_points"] = (
            coverage[episode_id]["x24_minus_x23_coverage"] * 100.0
        )

    x23_alert_s = contact["arms"][ARM_X23]["first_alert_time_s"]
    x24_alert_s = contact["arms"][ARM_X24]["first_alert_time_s"]
    x23_lead_s = contact["arms"][ARM_X23]["first_alert_lead_seconds"]
    x24_lead_s = contact["arms"][ARM_X24]["first_alert_lead_seconds"]
    contact_coverage_gain = coverage[CONTACT_EPISODE]["x24_minus_x23_coverage"]
    earlier = bool(
        x24_alert_s is not None
        and x23_alert_s is not None
        and x24_alert_s + EPSILON < x23_alert_s
    )
    baseline_no_alert_gain = x24_alert_s is not None and x23_alert_s is None
    positive_benefit = baseline_no_alert_gain or earlier or contact_coverage_gain > EPSILON
    gain = {
        "first_alert_time_gain_seconds": (
            None
            if x23_alert_s is None or x24_alert_s is None
            else x23_alert_s - x24_alert_s
        ),
        "first_alert_lead_gain_seconds": (
            None
            if x23_lead_s is None or x24_lead_s is None
            else x24_lead_s - x23_lead_s
        ),
        "x23_had_no_contact_alert_while_x24_did": baseline_no_alert_gain,
        "x24_first_alert_strictly_earlier": earlier,
        "contact_occlusion_coverage_gain": contact_coverage_gain,
        "contact_occlusion_coverage_gain_percentage_points": contact_coverage_gain * 100.0,
        "positive_contact_benefit": positive_benefit,
    }

    safe_frames = predictions["episodes"][SAFE_TWIN_EPISODE]["frames"]
    safe_period_s = uniform_sample_period(
        rows_by_episode[SAFE_TWIN_EPISODE],
        "safe_tail_period",
    )
    require(
        any(float(frame["time_s"]) + EPSILON >= SAFE_TAIL_START_SECONDS for frame in safe_frames),
        "safe_tail_window_empty",
    )
    segments, ignored_before_tail, common_tail = contiguous_new_false_segments(
        safe_frames,
        safe_period_s,
    )
    safe = {
        "episode_id": SAFE_TWIN_EPISODE,
        "fork_time_s": TWIN_FORK_SECONDS,
        "grace_seconds": SAFE_TAIL_GRACE_SECONDS,
        "scored_tail_start_time_s_inclusive": SAFE_TAIL_START_SECONDS,
        "definition": "ONLY_X24_MINUS_X23_SEGMENTS_IN_POST_GRACE_TAIL",
        "new_false_alert_segments": segments,
        "new_false_alert_segment_count": len(segments),
        "ignored_x24_minus_x23_sample_indices_before_tail": ignored_before_tail,
        "common_x23_x24_alert_sample_indices_in_tail_not_new": common_tail,
        "ep02_prefix_alerts_are_not_false_alerts": True,
    }

    hold = hold_metrics(predictions)
    no_plan = no_plan_metrics(predictions)
    checks = {
        "x24_contact_first_alert_exists": x24_alert_s is not None,
        "x24_contact_first_alert_lead_at_least_0_40s": (
            x24_lead_s is not None
            and x24_lead_s + EPSILON >= MINIMUM_X24_CONTACT_LEAD_SECONDS
        ),
        "x24_covers_full_eligible_0_40s_contact_occlusion_window": (
            coverage[CONTACT_EPISODE]["arms"][ARM_X24]["coverage"] >= 1.0 - EPSILON
        ),
        "x24_has_positive_contact_benefit_over_x23": positive_benefit,
        "safe_twin_has_zero_new_post_fork_plus_0_70s_false_alert_segments": not segments,
        "no_plan_ep03_ep04_x24_equals_x23_framewise": all(
            no_plan[episode_id]["framewise_route_risk_equal"]
            for episode_id in NO_PLAN_EPISODES
        ),
        "no_plan_ep03_ep04_use_observed_cv_fallback": all(
            no_plan[episode_id]["x24_no_plan_observed_cv_fallback_every_frame"]
            for episode_id in NO_PLAN_EPISODES
        ),
        "all_hold_evidence_age_at_most_0_60s": not hold["violations"],
    }
    metrics = {
        "contact_first_alert_and_lead": contact,
        "eligible_full_occlusion_gap_coverage": coverage,
        "x24_relative_to_x23_gain": gain,
        "safe_twin_post_fork_false_alerts": safe,
        "no_plan_framewise_identity": no_plan,
        "hold_evidence_age": hold,
    }
    return metrics, checks


def render_svg(result_status: str, passed: bool, metrics: Mapping[str, Any]) -> bytes:
    contact = metrics["contact_first_alert_and_lead"]
    coverage = metrics["eligible_full_occlusion_gap_coverage"][CONTACT_EPISODE]
    gain = metrics["x24_relative_to_x23_gain"]
    safe = metrics["safe_twin_post_fork_false_alerts"]
    hold = metrics["hold_evidence_age"]
    x23_coverage = float(coverage["arms"][ARM_X23]["coverage"])
    x24_coverage = float(coverage["arms"][ARM_X24]["coverage"])
    x23_lead = contact["arms"][ARM_X23]["first_alert_lead_seconds"]
    x24_lead = contact["arms"][ARM_X24]["first_alert_lead_seconds"]
    maximum_hold = hold["maximum_hold_evidence_age_seconds"]
    status_color = "#22c55e" if passed else "#ef4444"
    status_text = html.escape(result_status)

    def show(value: Any, suffix: str = "") -> str:
        if value is None:
            return "NO ALERT"
        return f"{float(value):.2f}{suffix}"

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="640" viewBox="0 0 1080 640">',
        '<rect width="1080" height="640" fill="#0f172a"/>',
        '<text x="48" y="62" font-family="sans-serif" font-size="30" font-weight="700" fill="#f8fafc">CARLA C2 X24 plan-adherent score</text>',
        f'<circle cx="52" cy="101" r="10" fill="{status_color}"/>',
        f'<text x="72" y="109" font-family="monospace" font-size="17" fill="{status_color}">{status_text}</text>',
        '<text x="48" y="158" font-family="sans-serif" font-size="21" fill="#e2e8f0">CONTACT first-alert lead</text>',
        f'<text x="350" y="158" font-family="monospace" font-size="20" fill="#94a3b8">X23 {show(x23_lead, "s")}</text>',
        f'<text x="590" y="158" font-family="monospace" font-size="20" fill="#38bdf8">X24 {show(x24_lead, "s")}</text>',
        '<text x="48" y="218" font-family="sans-serif" font-size="21" fill="#e2e8f0">Eligible 0.40s CONTACT occlusion coverage</text>',
        '<rect x="48" y="242" width="850" height="34" rx="6" fill="#1e293b"/>',
        f'<rect x="48" y="242" width="{850 * x23_coverage:.1f}" height="34" rx="6" fill="#64748b"/>',
        f'<text x="920" y="267" font-family="monospace" font-size="20" fill="#cbd5e1">X23 {100*x23_coverage:.0f}%</text>',
        '<rect x="48" y="292" width="850" height="34" rx="6" fill="#1e293b"/>',
        f'<rect x="48" y="292" width="{850 * x24_coverage:.1f}" height="34" rx="6" fill="#0ea5e9"/>',
        f'<text x="920" y="317" font-family="monospace" font-size="20" fill="#7dd3fc">X24 {100*x24_coverage:.0f}%</text>',
        f'<text x="48" y="375" font-family="sans-serif" font-size="21" fill="#e2e8f0">X24-X23 coverage gain: {100*float(gain["contact_occlusion_coverage_gain"]):+.0f} pp</text>',
        f'<text x="48" y="419" font-family="sans-serif" font-size="21" fill="#e2e8f0">SAFE new post-(fork+0.70s) false-alert segments: {int(safe["new_false_alert_segment_count"])}</text>',
        f'<text x="48" y="463" font-family="sans-serif" font-size="21" fill="#e2e8f0">Maximum HOLD evidence age: {show(maximum_hold, "s")} / 0.60s</text>',
        '<line x1="48" y1="510" x2="1032" y2="510" stroke="#334155"/>',
        '<text x="48" y="550" font-family="sans-serif" font-size="18" fill="#fbbf24">Same-source scripted-CARLA Development result.</text>',
        '<text x="48" y="584" font-family="sans-serif" font-size="17" fill="#94a3b8">No blind, source-disjoint, real-world, or product-safety claim.</text>',
        '<text x="48" y="616" font-family="sans-serif" font-size="15" fill="#64748b">EP02 prefix and grace-window alerts are excluded; only X24-minus-X23 tail segments count.</text>',
        '</svg>',
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def render_png(passed: bool, metrics: Mapping[str, Any]) -> bytes:
    """Render a dependency-free compact companion chart.

    The labelled SVG is the primary visual.  This PNG carries the same summary
    in standard PNG text metadata and uses fixed color lanes for X23, X24,
    SAFE-tail, and HOLD status.
    """

    width, height = 960, 360
    background = (15, 23, 42)
    pixels = bytearray(background * (width * height))

    def rectangle(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y1)):
            start = (y * width + max(0, x0)) * 3
            end = (y * width + min(width, x1)) * 3
            pixels[start:end] = bytes(color) * ((end - start) // 3)

    coverage = metrics["eligible_full_occlusion_gap_coverage"][CONTACT_EPISODE]
    x23 = float(coverage["arms"][ARM_X23]["coverage"])
    x24 = float(coverage["arms"][ARM_X24]["coverage"])
    safe_segments = int(metrics["safe_twin_post_fork_false_alerts"]["new_false_alert_segment_count"])
    hold_violations = len(metrics["hold_evidence_age"]["violations"])
    rectangle(0, 0, width, 22, (34, 197, 94) if passed else (239, 68, 68))
    rectangle(70, 82, 890, 124, (30, 41, 59))
    rectangle(70, 82, 70 + round(820 * x23), 124, (100, 116, 139))
    rectangle(70, 158, 890, 200, (30, 41, 59))
    rectangle(70, 158, 70 + round(820 * x24), 200, (14, 165, 233))
    rectangle(70, 254, 430, 312, (34, 197, 94) if safe_segments == 0 else (239, 68, 68))
    rectangle(530, 254, 890, 312, (34, 197, 94) if hold_violations == 0 else (239, 68, 68))
    raw = b"".join(
        b"\x00" + bytes(pixels[y * width * 3 : (y + 1) * width * 3])
        for y in range(height)
    )
    description = (
        f"X23 contact occlusion coverage={x23:.3f}; "
        f"X24={x24:.3f}; safe new segments={safe_segments}; "
        f"hold violations={hold_violations}; development gate={'MET' if passed else 'NOT_MET'}"
    ).encode("latin-1")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"tEXt", b"Title\x00CARLA C2 X24 Development score")
        + _png_chunk(b"tEXt", b"Description\x00" + description)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )


def score(
    predictions_root: Path,
    evaluator_root: Path,
    *,
    expected_predictions_sha256: str,
    expected_evidence_manifest_sha256: str,
) -> dict[str, Any]:
    prediction_root_resolved = predictions_root.resolve(strict=True)
    paths = output_paths(prediction_root_resolved)
    require(not paths["attempt"].exists(), f"score_attempt_already_consumed:{paths['attempt']}")
    require(not paths["result"].exists(), f"score_result_exists:{paths['result']}")
    require(not paths["svg"].exists(), f"score_svg_exists:{paths['svg']}")
    require(not paths["png"].exists(), f"score_png_exists:{paths['png']}")
    predictions_path, predictions, freeze_path, freeze = validate_predictions_before_attempt(
        prediction_root_resolved,
        expected_predictions_sha256,
    )
    prediction_digest = sha256_file(predictions_path)
    freeze_digest = sha256_file(freeze_path)
    expected_evidence_digest = require_digest(
        expected_evidence_manifest_sha256,
        "expected_evidence_manifest_sha256",
    )

    # abspath is lexical here: evaluator existence, manifest, and truth remain
    # unopened until after the exclusive receipt has been flushed and fsynced.
    evaluator_root_lexical = Path(os.path.abspath(os.fspath(evaluator_root)))
    evidence_manifest_lexical = evaluator_root_lexical.parent / EVIDENCE_MANIFEST_NAME
    attempt = {
        "schema": SCORE_ATTEMPT_SCHEMA,
        "attempt": 1,
        "status": "CONSUMED_BEFORE_EVALUATOR_MANIFEST_OR_TRUTH_OPEN",
        "experiment_id": EXPERIMENT_ID,
        "predictions": {"path": str(predictions_path), "sha256": prediction_digest},
        "freeze": {"path": str(freeze_path), "sha256": freeze_digest},
        "model_manifest_sha256": freeze["model_manifest"]["sha256"],
        "evaluator_root_expected": str(evaluator_root_lexical),
        "sealed_evidence_manifest_expected": {
            "path": str(evidence_manifest_lexical),
            "sha256": expected_evidence_digest,
        },
        "score_contract": predeclared_contract(),
        "scorer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    write_json_exclusive(paths["attempt"], attempt)

    (
        evaluator_root_resolved,
        evidence_manifest_path,
        used_references,
        _all_evidence_entries,
        rows_by_episode,
        outcome_summaries,
        occlusion_reports,
    ) = load_verified_evaluator(
        evaluator_root_lexical,
        expected_evidence_digest,
        predictions,
    )
    outcomes = validate_outcomes(outcome_summaries, rows_by_episode)
    windows = selected_occlusion_windows(occlusion_reports, rows_by_episode)
    metrics, checks = score_metrics(predictions, rows_by_episode, outcomes, windows)
    checks = {
        **checks,
        "prediction_and_freeze_hash_chain_verified": True,
        "sealed_evidence_manifest_and_used_truth_hashes_verified": True,
        "one_score_attempt_consumed_before_evaluator_open": paths["attempt"].is_file(),
    }
    passed = all(checks.values())
    status = (
        "DTR_CARLA_X24_PLAN_ADHERENT_DEVELOPMENT_GATE_MET"
        if passed
        else "DTR_CARLA_X24_PLAN_ADHERENT_DEVELOPMENT_GATE_NOT_MET"
    )

    svg_payload = render_svg(status, passed, metrics)
    png_payload = render_png(passed, metrics)
    write_bytes_exclusive(paths["svg"], svg_payload)
    write_bytes_exclusive(paths["png"], png_payload)

    result = {
        "schema": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "gate": {"passed": passed, "checks": checks},
        "predeclared_score_contract": predeclared_contract(),
        "metrics": metrics,
        "sources": {
            "predictions": {"path": str(predictions_path), "sha256": prediction_digest},
            "freeze": {"path": str(freeze_path), "sha256": freeze_digest},
            "model_manifest_sha256": freeze["model_manifest"]["sha256"],
            "score_attempt": {"path": str(paths["attempt"]), "sha256": sha256_file(paths["attempt"])},
            "evaluator_root": str(evaluator_root_resolved),
            "sealed_evidence_manifest": {
                "path": str(evidence_manifest_path),
                "sha256": sha256_file(evidence_manifest_path),
            },
            "verified_used_evaluator_files": used_references,
            "visuals": {
                "svg": {"path": str(paths["svg"]), "sha256": sha256_file(paths["svg"])},
                "png": {"path": str(paths["png"]), "sha256": sha256_file(paths["png"])},
            },
        },
        "decision": {
            "next": (
                "RETAIN_X24_AS_C2_DEVELOPMENT_SUCCESSOR_CANDIDATE"
                if passed
                else "CLOSE_FROZEN_X24_C2_ARM_WITHOUT_THRESHOLD_SWEEP"
            )
        },
        "claim_boundary": {
            "tier": "DEVELOPMENT",
            "same_source_scripted_carla": True,
            "blind_score_claimed": False,
            "source_disjoint_claimed": False,
            "real_world_claimed": False,
            "product_safety_claimed": False,
            "note": "Evaluator truth was opened only after the one-shot receipt; that ordering does not make this a blind or source-disjoint score.",
        },
    }
    write_json_exclusive(paths["result"], result)
    return {**result, "result_sha256": sha256_file(paths["result"])}


def status(predictions_root: Path) -> dict[str, Any]:
    root = predictions_root.resolve()
    paths = output_paths(root)
    artifacts = {
        name: (
            None
            if not path.is_file()
            else {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
        for name, path in paths.items()
    }
    if artifacts["result"] is not None:
        result = read_json_object(paths["result"], "score_result")
        state = "GATE_MET" if result.get("gate", {}).get("passed") is True else "GATE_NOT_MET"
    elif artifacts["attempt"] is not None:
        state = "SCORE_ATTEMPT_CONSUMED_RESULT_MISSING"
    elif (root / PREDICTIONS_NAME).is_file():
        state = "SEALED_PREDICTIONS_PENDING_ONE_SCORE"
    else:
        state = "PREDICTIONS_MISSING"
    return {"state": state, "artifacts": artifacts}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--predictions-root", type=Path, required=True)
    score_parser.add_argument("--evaluator-root", type=Path, required=True)
    score_parser.add_argument("--predictions-sha256", required=True)
    score_parser.add_argument("--evidence-manifest-sha256", required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--predictions-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "score":
        value = score(
            args.predictions_root,
            args.evaluator_root,
            expected_predictions_sha256=args.predictions_sha256,
            expected_evidence_manifest_sha256=args.evidence_manifest_sha256,
        )
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        return 0 if value["gate"]["passed"] else 2
    value = status(args.predictions_root)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

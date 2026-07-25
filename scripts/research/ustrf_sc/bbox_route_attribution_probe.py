"""Attribute the frozen bbox signal to matched route versus bbox/source effects.

The probe reuses the exact 15 positive / 15 matched-negative windows and the
exact common-eligible frame subset from the preceding four-arm signal probe.
It changes only the route support applied to one frozen person-bbox confidence
field.  It does not select an alert threshold or run tracker, TTC, lifecycle,
Android, human, or production logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from four_arm_signal_probe import (
    FRAME_COUNT,
    MINIMUM_WINDOW_ELIGIBLE_FRACTION,
    MINIMUM_WINDOW_ELIGIBLE_FRAMES,
    PRIMARY_WINDOW_QUANTILE,
    ROUTE_HALF_WIDTH_FRACTION,
    WINDOW_QUANTILES,
    WINDOWS_SHA256,
    ProbeError,
    bbox_route_score,
    build_pairs,
    canonical_json_bytes,
    compare_delta,
    load_json,
    load_source,
    pretty_json_text,
    quantile,
    quantile_name,
    rectangle_intersection_area,
    require,
    sha256_bytes,
    sha256_file,
    source_specs,
    summarize_deltas,
    validate_windows,
)


SCHEMA = "blindassist_ustrf_bbox_route_attribution_probe_r1"
ARM_MATCHED = "A_bbox_matched_route"
ARM_UNIFORM = "B_bbox_uniform_route"
ARM_SHUFFLED = "C_bbox_shuffled_route"
ARM_ONLY = "D_bbox_only"
ARMS = (ARM_MATCHED, ARM_UNIFORM, ARM_SHUFFLED, ARM_ONLY)
CONTROLS = (ARM_UNIFORM, ARM_SHUFFLED, ARM_ONLY)
REQUIRED_DIRECT_WINS = 12


def person_box_rows(
    detections: Sequence[Mapping[str, Any]],
) -> list[tuple[float, tuple[float, float, float, float]]]:
    rows: list[tuple[float, tuple[float, float, float, float]]] = []
    for detection in detections:
        if detection.get("label") != "person":
            continue
        box = detection.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        confidence = float(detection["confidence"])
        left, top, right, bottom = (float(value) for value in box)
        if not math.isfinite(confidence) or not all(
            math.isfinite(value) for value in (left, top, right, bottom)
        ):
            continue
        rows.append((confidence, (left, top, right, bottom)))
    return rows


def bbox_uniform_route_score(
    detections: Sequence[Mapping[str, Any]],
    *,
    source_width: int,
    source_height: int,
) -> float:
    """Apply the same support-average operator with the full frame as route."""

    frame_area = float(source_width * source_height)
    require(frame_area > 0.0, "source frame must have positive area")
    best = 0.0
    for confidence, box in person_box_rows(detections):
        overlap = rectangle_intersection_area(
            box[0],
            box[1],
            box[2],
            box[3],
            0.0,
            0.0,
            float(source_width),
            float(source_height),
        )
        best = max(best, confidence * min(1.0, overlap / frame_area))
    return best


def bbox_only_score(detections: Sequence[Mapping[str, Any]]) -> float:
    """Return the support-free maximum of the same bbox confidence field."""

    return max((confidence for confidence, _ in person_box_rows(detections)), default=0.0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProbeError(f"invalid JSONL at {path}:{line_number}") from error
        require(isinstance(row, dict), f"JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def make_frame_lines(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def summarize_pair_rows(
    pair_rows: Sequence[Mapping[str, Any]],
    source_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        summaries[arm] = {}
        for q_name in (quantile_name(q) for q in WINDOW_QUANTILES):
            deltas = [float(row["deltas"][arm][q_name]) for row in pair_rows]
            by_source = {}
            for source_id in source_ids:
                source_deltas = [
                    float(row["deltas"][arm][q_name])
                    for row in pair_rows
                    if row["source_id"] == source_id
                ]
                by_source[source_id] = summarize_deltas(source_deltas)
            summaries[arm][q_name] = {
                **summarize_deltas(deltas),
                "by_source": by_source,
            }
    return summaries


def summarize_direct_advantages(
    pair_rows: Sequence[Mapping[str, Any]],
    source_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for control in CONTROLS:
        result[control] = {}
        for q_name in (quantile_name(q) for q in WINDOW_QUANTILES):
            advantages = [
                float(row["deltas"][ARM_MATCHED][q_name])
                - float(row["deltas"][control][q_name])
                for row in pair_rows
            ]
            by_source = {}
            for source_id in source_ids:
                source_advantages = [
                    float(row["deltas"][ARM_MATCHED][q_name])
                    - float(row["deltas"][control][q_name])
                    for row in pair_rows
                    if row["source_id"] == source_id
                ]
                by_source[source_id] = summarize_deltas(source_advantages)
            result[control][q_name] = {
                **summarize_deltas(advantages),
                "by_source": by_source,
            }
    return result


def compute(repo: Path) -> tuple[dict[str, Any], bytes]:
    windows_path = repo / "artifacts.local/evidence/ustrf-tracker-ttc-ablation-v1/windows-v2.json"
    canonical_path = (
        repo
        / "artifacts.local/evidence/ustrf-detector-target-attribution-r1/"
        "canonical-host-ledger-v2.json"
    )
    attribution_path = (
        repo
        / "artifacts.local/evidence/ustrf-detector-target-attribution-r1/"
        "target-attribution-result-r1.json"
    )
    dense_report_path = (
        repo / "artifacts.local/evidence/ustrf-four-arm-signal-probe-r1/report.json"
    )
    dense_frames_path = (
        repo / "artifacts.local/evidence/ustrf-four-arm-signal-probe-r1/frame-scores.jsonl"
    )

    require(sha256_file(windows_path) == WINDOWS_SHA256, "frozen windows SHA drift")
    windows = load_json(windows_path)["windows"]
    canonical = load_json(canonical_path)
    attribution = load_json(attribution_path)
    dense_report = load_json(dense_report_path)
    dense_frame_rows = read_jsonl(dense_frames_path)
    require(canonical["frame_count"] == FRAME_COUNT, "canonical ledger frame count drift")
    require(len(canonical["frames"]) == FRAME_COUNT, "canonical ledger row count drift")
    require(
        canonical["input_tensor_exact_match_count"] == FRAME_COUNT,
        "canonical Android/host input parity is incomplete",
    )
    require(
        attribution["G1b_canonical_semantic_parity"] == "pass"
        and attribution["hard_gate_passed"] is True,
        "target attribution parent gate is not closed",
    )
    require(
        dense_report["decision_gate"]["decision"] == "STOP_CURRENT_DENSE_USTRF_EXPRESSION",
        "dense branch is not formally stopped by its parent receipt",
    )
    require(len(dense_frame_rows) == FRAME_COUNT, "dense parent frame inventory drift")
    require(
        sha256_file(dense_frames_path) == dense_report["frame_scores_sha256"],
        "dense parent frame score SHA drift",
    )

    canonical_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in canonical["frames"]:
        key = (str(row["source_id"]), str(row["frame_id"]))
        require(key not in canonical_by_key, f"duplicate canonical frame {key}")
        canonical_by_key[key] = row

    parent_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in dense_frame_rows:
        key = (str(row["source_id"]), str(row["frame_id"]))
        require(key not in parent_by_key, f"duplicate dense parent frame {key}")
        parent_by_key[key] = row
    require(set(parent_by_key) == set(canonical_by_key), "dense/canonical frame identities differ")

    frames_by_source: dict[str, list[dict[str, Any]]] = {}
    routes_by_source: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    for spec in source_specs(repo):
        _, frames, routes, hashes = load_source(spec)
        frames_by_source[spec.source_id] = frames
        routes_by_source[spec.source_id] = routes
        source_hashes[spec.source_id] = hashes
    validate_windows(
        windows,
        canonical_keys=set(canonical_by_key),
        frames_by_source=frames_by_source,
    )
    pairs = build_pairs(windows)
    require(len(pairs) == 15, "matched pair count is not 15")

    windows_by_source: dict[str, list[Mapping[str, Any]]] = {}
    donors: dict[str, Mapping[str, Any]] = {}
    for source_id in sorted(frames_by_source):
        ordered = sorted(
            (row for row in windows if row["source_id"] == source_id),
            key=lambda row: str(row["window_id"]),
        )
        windows_by_source[source_id] = ordered
        for index, window in enumerate(ordered):
            donors[str(window["window_id"])] = ordered[(index + 1) % len(ordered)]

    frame_rows: list[dict[str, Any]] = []
    window_results: dict[str, dict[str, Any]] = {}
    parent_matched_score_mismatch_count = 0
    for source_id in sorted(frames_by_source):
        frames = frames_by_source[source_id]
        routes = routes_by_source[source_id]
        for window in windows_by_source[source_id]:
            window_id = str(window["window_id"])
            donor = donors[window_id]
            start = int(window["start_frame"])
            end = int(window["end_frame"])
            donor_start = int(donor["start_frame"])
            donor_count = int(donor["end_frame"]) - donor_start + 1
            window_count = end - start + 1
            arm_values: dict[str, list[float]] = {arm: [] for arm in ARMS}
            eligible_count = 0
            for offset, frame_index in enumerate(range(start, end + 1)):
                donor_offset = round(offset * (donor_count - 1) / max(1, window_count - 1))
                donor_frame_index = donor_start + donor_offset
                frame = frames[frame_index]
                key = (source_id, str(frame["frame_id"]))
                canonical_frame = canonical_by_key[key]
                parent_frame = parent_by_key[key]
                matched_route = routes[frame_index]
                shuffled_route = routes[donor_frame_index]
                row: dict[str, Any] = {
                    "source_id": source_id,
                    "window_id": window_id,
                    "window_type": window["window_type"],
                    "frame_id": str(frame["frame_id"]),
                    "frame_index": frame_index,
                    "parent_common_eligible": bool(parent_frame["common_eligible"]),
                    "scores": {arm: None for arm in ARMS},
                }
                if parent_frame["common_eligible"]:
                    require(
                        matched_route["status"] == "known" and shuffled_route["status"] == "known",
                        f"{key}: parent-eligible route became unknown",
                    )
                    source_width, source_height = (
                        int(value) for value in canonical_frame["source_size"]
                    )
                    detections = canonical_frame["post_nms_detections_canonical_320"]
                    matched = bbox_route_score(
                        detections,
                        matched_route["uv"],
                        source_width=source_width,
                        source_height=source_height,
                    )
                    shuffled = bbox_route_score(
                        detections,
                        shuffled_route["uv"],
                        source_width=source_width,
                        source_height=source_height,
                    )
                    uniform = bbox_uniform_route_score(
                        detections,
                        source_width=source_width,
                        source_height=source_height,
                    )
                    only = bbox_only_score(detections)
                    parent_matched = float(parent_frame["scores"][ARM_MATCHED])
                    if abs(matched - parent_matched) > 1e-12:
                        parent_matched_score_mismatch_count += 1
                    scores = {
                        ARM_MATCHED: matched,
                        ARM_UNIFORM: uniform,
                        ARM_SHUFFLED: shuffled,
                        ARM_ONLY: only,
                    }
                    for arm, value in scores.items():
                        arm_values[arm].append(float(value))
                    row.update(
                        {
                            "matched_route_uv": [
                                float(value) for value in matched_route["uv"]
                            ],
                            "shuffled_route_source_window_id": donor["window_id"],
                            "shuffled_route_source_frame_id": str(
                                frames[donor_frame_index]["frame_id"]
                            ),
                            "shuffled_route_uv": [
                                float(value) for value in shuffled_route["uv"]
                            ],
                            "person_bbox_count": len(person_box_rows(detections)),
                            "scores": scores,
                        }
                    )
                    eligible_count += 1
                frame_rows.append(row)
            minimum_eligible = max(
                MINIMUM_WINDOW_ELIGIBLE_FRAMES,
                math.ceil(window_count * MINIMUM_WINDOW_ELIGIBLE_FRACTION),
            )
            require(
                eligible_count >= minimum_eligible,
                f"{window_id}: common eligible frames {eligible_count} < {minimum_eligible}",
            )
            summaries = {
                arm: {
                    quantile_name(q): quantile(arm_values[arm], q)
                    for q in WINDOW_QUANTILES
                }
                for arm in ARMS
            }
            window_results[window_id] = {
                "source_id": source_id,
                "window_type": window["window_type"],
                "frame_count": window_count,
                "common_eligible_frame_count": eligible_count,
                "common_eligible_fraction": eligible_count / window_count,
                "scores": summaries,
            }

    require(
        parent_matched_score_mismatch_count == 0,
        "matched bbox arm does not reproduce parent A frame scores",
    )
    pair_rows: list[dict[str, Any]] = []
    for pair in pairs:
        positive = window_results[pair["positive_window_id"]]
        negative = window_results[pair["negative_window_id"]]
        deltas: dict[str, dict[str, float]] = {}
        outcomes: dict[str, dict[str, str]] = {}
        for arm in ARMS:
            deltas[arm] = {}
            outcomes[arm] = {}
            for q_name in (quantile_name(q) for q in WINDOW_QUANTILES):
                delta = positive["scores"][arm][q_name] - negative["scores"][arm][q_name]
                deltas[arm][q_name] = delta
                outcomes[arm][q_name] = compare_delta(delta)
        pair_rows.append({**pair, "deltas": deltas, "outcomes": outcomes})

    source_ids = sorted(frames_by_source)
    summaries = summarize_pair_rows(pair_rows, source_ids)
    direct = summarize_direct_advantages(pair_rows, source_ids)
    parent_primary = dense_report["summary"][ARM_MATCHED][PRIMARY_WINDOW_QUANTILE]
    matched_primary = summaries[ARM_MATCHED][PRIMARY_WINDOW_QUANTILE]
    require(
        canonical_json_bytes(parent_primary) == canonical_json_bytes(matched_primary),
        "matched bbox pair summary does not reproduce parent A 12/15 result",
    )

    per_control_conditions: dict[str, dict[str, bool]] = {}
    for control in CONTROLS:
        primary = direct[control][PRIMARY_WINDOW_QUANTILE]
        per_control_conditions[control] = {
            "primary_direct_wins_at_least_12_of_15": primary["wins"]
            >= REQUIRED_DIRECT_WINS,
            "primary_direct_wilson_lower_above_half": primary[
                "win_rate_wilson_95"
            ][0]
            > 0.5,
            "primary_direct_median_positive_in_each_source": all(
                row["median_positive_minus_negative"] is not None
                and row["median_positive_minus_negative"] > 0.0
                for row in primary["by_source"].values()
            ),
            "q50_and_q95_direct_median_nonnegative": all(
                direct[control][q_name]["median_positive_minus_negative"] is not None
                and direct[control][q_name]["median_positive_minus_negative"] >= 0.0
                for q_name in ("q50", "q95")
            ),
        }
    stable_exceeds_controls = all(
        all(conditions.values()) for conditions in per_control_conditions.values()
    )
    matched_positive_in_each_source = all(
        row["median_positive_minus_negative"] is not None
        and row["median_positive_minus_negative"] > 0.0
        for row in matched_primary["by_source"].values()
    )
    pass_gate = stable_exceeds_controls and matched_positive_in_each_source
    decision = (
        "MATCHED_BBOX_STABLE_WIN_CAUSAL_LIFECYCLE_NEXT"
        if pass_gate
        else "STOP_ROUTE_CONDITIONED_USTRF_DOWNGRADE_TO_DETECTOR_BASELINE"
    )

    frame_lines = make_frame_lines(frame_rows)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "VALID",
        "authority": {
            "research_signal_attribution_only": True,
            "alert_threshold_selected": False,
            "tracker_ttc_lifecycle_used": False,
            "new_data_used": False,
            "architecture_convergence_authorized": False,
            "android_runtime_authorized": False,
            "human_safety_authorized": False,
            "production_authorized": False,
        },
        "parent_bindings": {
            "windows_path": str(windows_path.relative_to(repo)).replace("\\", "/"),
            "windows_sha256": sha256_file(windows_path),
            "canonical_ledger_path": str(canonical_path.relative_to(repo)).replace(
                "\\", "/"
            ),
            "canonical_ledger_sha256": sha256_file(canonical_path),
            "target_attribution_path": str(attribution_path.relative_to(repo)).replace(
                "\\", "/"
            ),
            "target_attribution_sha256": sha256_file(attribution_path),
            "stopped_dense_report_path": str(dense_report_path.relative_to(repo)).replace(
                "\\", "/"
            ),
            "stopped_dense_report_sha256": sha256_file(dense_report_path),
            "stopped_dense_frame_scores_path": str(
                dense_frames_path.relative_to(repo)
            ).replace("\\", "/"),
            "stopped_dense_frame_scores_sha256": sha256_file(dense_frames_path),
            "source_bindings": source_hashes,
            "implementation_sha256": sha256_file(Path(__file__)),
            "shared_helper_implementation_sha256": sha256_file(
                Path(__file__).with_name("four_arm_signal_probe.py")
            ),
        },
        "method": {
            "arms": list(ARMS),
            "bbox_confidence_field": (
                "each frozen post-NMS person bbox is a constant rectangular field "
                "at its unchanged confidence"
            ),
            "support_operator": (
                "maximum across person bboxes of confidence times the fraction "
                "of route support covered by that bbox"
            ),
            "matched_route": "frozen past_pose_prefix_only route_predictions",
            "uniform_route": "the full source frame as equal route support",
            "shuffled_route": (
                "within-source window_id cyclic shift-one with normalized-time "
                "frame mapping, no seed or labels"
            ),
            "bbox_only": "support-free maximum person confidence from the same field",
            "route_patch_half_width_fraction": ROUTE_HALF_WIDTH_FRACTION,
            "common_eligibility": (
                "exact frame subset used by stopped dense parent so A 12/15 is "
                "reproduced without eligibility drift"
            ),
            "window_score_quantiles": list(WINDOW_QUANTILES),
            "primary_window_score": PRIMARY_WINDOW_QUANTILE,
            "threshold_search_or_alarm_tuning": False,
        },
        "inventory": {
            "source_count": len(source_ids),
            "window_count": len(windows),
            "positive_window_count": 15,
            "negative_window_count": 15,
            "matched_pair_count": len(pairs),
            "selected_frame_count": len(frame_rows),
            "common_eligible_frame_count": sum(
                bool(row["parent_common_eligible"]) for row in frame_rows
            ),
            "parent_matched_score_mismatch_count": parent_matched_score_mismatch_count,
            "all_windows_pass_common_eligibility_floor": True,
        },
        "windows": [
            window_results[key] | {"window_id": key} for key in sorted(window_results)
        ],
        "pairs": pair_rows,
        "summary": summaries,
        "matched_direct_advantage_over_controls": direct,
        "decision_gate": {
            "rule": (
                "matched bbox must stably exceed all three controls and its primary "
                "positive-minus-negative median must be positive in every source"
            ),
            "stable_exceeds_all_three_controls": stable_exceeds_controls,
            "matched_primary_median_positive_in_each_source": matched_positive_in_each_source,
            "per_control_conditions": per_control_conditions,
            "all_conditions_passed": pass_gate,
            "decision": decision,
            "next_authority_if_passed": "causal_lifecycle_research_only",
            "next_authority_if_failed": "ordinary_detector_baseline_only",
        },
        "frame_scores_sha256": sha256_bytes(frame_lines),
    }
    return report, frame_lines


def write_output(output_dir: Path, report: Mapping[str, Any], frame_lines: bytes) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(pretty_json_text(report), encoding="utf-8")
    (output_dir / "frame-scores.jsonl").write_bytes(frame_lines)


def validate_existing(repo: Path, output_dir: Path) -> dict[str, Any]:
    existing_report = load_json(output_dir / "report.json")
    existing_frame_lines = (output_dir / "frame-scores.jsonl").read_bytes()
    report, frame_lines = compute(repo)
    require(canonical_json_bytes(existing_report) == canonical_json_bytes(report), "report replay differs")
    require(existing_frame_lines == frame_lines, "frame score replay differs")
    return {
        "status": "VALID_REPLAY_MATCH",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "frame_scores_sha256": sha256_file(output_dir / "frame-scores.jsonl"),
        "decision": report["decision_gate"]["decision"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts.local/evidence/ustrf-bbox-route-attribution-r1"),
    )
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    if args.validate_existing:
        print(pretty_json_text(validate_existing(repo, output_dir)), end="")
        return 0
    report, frame_lines = compute(repo)
    write_output(output_dir, report, frame_lines)
    print(
        pretty_json_text(
            {
                "status": report["status"],
                "decision": report["decision_gate"]["decision"],
                "report_path": str(output_dir / "report.json"),
                "frame_scores_path": str(output_dir / "frame-scores.jsonl"),
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

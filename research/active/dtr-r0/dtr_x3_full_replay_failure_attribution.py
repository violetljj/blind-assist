"""Attribute sealed X3 full-replay false segments without changing the scorer.

The evidence chain and truth-blind predictions are hash checked before native
JRDB OBB identity, trajectory, or geometry is opened.  This is a post-outcome
diagnostic only: it neither rescores predictions nor changes source, lifecycle,
route geometry, thresholds, or cohort membership.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_x0_motion_source_attribution as x0  # noqa: E402
from dtr_c1_global_obb_cohort_admission import _load_timestamps  # noqa: E402
from dtr_r7_occupancy_flow_canary import _causal_pose  # noqa: E402
from jrdb_rgb_bridge import read_bag_pose_and_rgb, require, sha256_file, write_json  # noqa: E402


SCHEMA = "blindassist-dtr-x3-full-replay-failure-attribution-v1"
STATUS = "DTR_X3_FULL_REPLAY_FAILURE_ATTRIBUTION_COMPLETE"
CHAIN_SCHEMA = "blindassist-dtr-x3-amended-evidence-chain-v1"
RESULT_SCHEMA = "blindassist-dtr-x3-full-lag-floxel-replay-v1"
PREDICTION_SCHEMA = "blindassist-dtr-x3-full-lag-floxel-predictions-v1"

STATIC_PSEUDO_MOTION = "STATIC_PSEUDO_MOTION"
BAD_FLOW_DIRECTION = "BAD_FLOW_DIRECTION"
BAD_FLOW_MAGNITUDE = "BAD_FLOW_MAGNITUDE"
REAL_MOVER_NONCRITICAL = "REAL_MOVER_NONCRITICAL"
ROUTE_GEOMETRY_MISS = "ROUTE_GEOMETRY_MISS"
FRAGMENTATION = "FRAGMENTATION"
WRONG_COMPONENT_BINDING = "WRONG_COMPONENT_BINDING"
NOT_EVALUABLE = "NOT_EVALUABLE"
SOURCE_FAILURES = {STATIC_PSEUDO_MOTION, BAD_FLOW_DIRECTION, BAD_FLOW_MAGNITUDE}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"x3_attr_json_object:{path}")
    return value


def _recorded_path(value: Any, *, label: str) -> Path:
    require(isinstance(value, str) and value, f"x3_attr_path:{label}")
    return Path(value).resolve(strict=True)


def _validate_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = _recorded_path(record.get("path"), label=label)
    require(record.get("sha256") == sha256_file(path), f"x3_attr_hash:{label}")
    return path


def _validate_sealed_chain(root: Path) -> dict[str, Any]:
    chain_path = (root / "evidence-chain.json").resolve(strict=True)
    chain = _load_json(chain_path)
    require(chain.get("schema") == CHAIN_SCHEMA, "x3_attr_chain_schema")
    require(chain.get("status") == "X3_AMENDED_EVIDENCE_CHAIN_SEALED", "x3_attr_chain_status")
    require(chain.get("truth_blind_predictions") is True, "x3_attr_chain_truth")
    freeze_path = _validate_record(chain["freeze"], label="freeze")
    materialization_path = _validate_record(chain["materialization"], label="materialization")
    predictions_path = _validate_record(chain["predictions"], label="predictions")
    result_path = _validate_record(chain["result"], label="result")
    sequence_records = chain.get("sequence_artifacts")
    require(isinstance(sequence_records, list) and len(sequence_records) == 6, "x3_attr_chain_sequences")
    sequence_names = []
    for row in sequence_records:
        sequence = str(row["sequence"])
        sequence_names.append(sequence)
        for kind in ("ledger", "manifest"):
            path = _recorded_path(row.get(kind), label=f"{sequence}:{kind}")
            require(
                row.get(f"{kind}_sha256") == sha256_file(path),
                f"x3_attr_chain_{kind}:{sequence}",
            )
    require(len(set(sequence_names)) == 6, "x3_attr_chain_sequence_duplicates")
    amendments = chain.get("amendments")
    require(isinstance(amendments, list) and len(amendments) == 1, "x3_attr_chain_amendment")
    amendment = amendments[0]
    receipt_path = _recorded_path(amendment.get("receipt"), label="amendment_receipt")
    require(amendment.get("receipt_sha256") == sha256_file(receipt_path), "x3_attr_amendment_receipt")
    script_path = _recorded_path(amendment.get("script"), label="amendment_script")
    require(amendment.get("script_sha256") == sha256_file(script_path), "x3_attr_amendment_script")
    checkpoints = amendment.get("checkpoints")
    require(isinstance(checkpoints, list) and len(checkpoints) == 1, "x3_attr_amendment_checkpoint")
    checkpoint_value = checkpoints[0].get("path")
    require(
        isinstance(checkpoint_value, str) and checkpoint_value,
        "x3_attr_path:amendment_checkpoint",
    )
    checkpoint_path = Path(checkpoint_value).resolve(strict=False)
    checkpoint_available = checkpoint_path.exists()
    if checkpoint_available:
        require(
            checkpoints[0].get("sha256") == sha256_file(checkpoint_path),
            "x3_attr_amendment_checkpoint_hash",
        )
    predictions = _load_json(predictions_path)
    result = _load_json(result_path)
    require(predictions.get("schema") == PREDICTION_SCHEMA, "x3_attr_prediction_schema")
    require(predictions.get("truth_blind") is True, "x3_attr_predictions_truth")
    require(result.get("schema") == RESULT_SCHEMA, "x3_attr_result_schema")
    require(result.get("status") == "DTR_X3_FULL_LAG_FLOXEL_GATE_NOT_MET", "x3_attr_result_status")
    require(
        predictions.get("source", {}).get("materialization_sha256") == sha256_file(materialization_path),
        "x3_attr_predictions_materialization",
    )
    require(
        predictions.get("source", {}).get("freeze_sha256") == sha256_file(freeze_path),
        "x3_attr_predictions_freeze",
    )
    require(
        result.get("source", {}).get("sealed_predictions_sha256") == sha256_file(predictions_path),
        "x3_attr_result_predictions",
    )
    require(
        result.get("source", {}).get("freeze_sha256") == sha256_file(freeze_path),
        "x3_attr_result_freeze",
    )
    return {
        "chain_path": chain_path,
        "chain_sha256": sha256_file(chain_path),
        "freeze_path": freeze_path,
        "materialization_path": materialization_path,
        "predictions_path": predictions_path,
        "result_path": result_path,
        "predictions": predictions,
        "result": result,
        "amendment_checkpoint_available": checkpoint_available,
    }


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return int(left["first_frame"]) <= int(right["last_frame"]) and int(
        right["first_frame"]
    ) <= int(left["last_frame"])


def _refine_bad_flow(row: Mapping[str, Any]) -> str:
    cause = str(row["primary_cause"])
    if cause != x0.BAD_FLOW:
        return cause
    diagnosed = [
        cell for cell in row["diagnostic_cells"] if cell.get("primary_cause") == x0.BAD_FLOW
    ]
    opposite = False
    for cell in diagnosed:
        if "target_velocity_forward_mps" not in cell:
            continue
        flow = np.asarray(
            (cell["velocity_forward_mps"], cell["velocity_left_mps"]), dtype=np.float64
        )
        target = np.asarray(
            (cell["target_velocity_forward_mps"], cell["target_velocity_left_mps"]),
            dtype=np.float64,
        )
        if float(np.dot(flow, target)) <= 0.0:
            opposite = True
            break
    return BAD_FLOW_DIRECTION if opposite else BAD_FLOW_MAGNITUDE


def _diagnose_ranges(
    *,
    arm: str,
    sequence: str,
    ranges: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, np.ndarray],
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    boxes_by_frame: Mapping[int, Sequence[Any]],
    history: Mapping[tuple[int, str], Any],
    poses: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for index, segment in enumerate(ranges, start=1):
        row = x0._diagnose_false_segment(
            segment_id=f"{sequence}:{arm.lower()}-false:{index:03d}",
            segment=segment,
            ledger=ledger,
            frames=frames,
            timestamps=timestamps,
            boxes_by_frame=boxes_by_frame,
            history=history,
            poses=poses,
        )
        output.append(
            {
                "sequence": sequence,
                "arm": arm,
                **row,
                "primary_cause": _refine_bad_flow(row),
                "native_truth_usage": "POST_SEAL_ATTRIBUTION_ONLY",
            }
        )
    return output


def _annotate_comparison(
    pdc_rows: list[dict[str, Any]], x3_rows: list[dict[str, Any]]
) -> None:
    pdc_to_x3: dict[str, list[str]] = defaultdict(list)
    x3_to_pdc: dict[str, list[str]] = defaultdict(list)
    for pdc in pdc_rows:
        for x3 in x3_rows:
            if pdc["sequence"] == x3["sequence"] and _overlap(pdc, x3):
                pdc_to_x3[pdc["segment_id"]].append(x3["segment_id"])
                x3_to_pdc[x3["segment_id"]].append(pdc["segment_id"])
    for row in pdc_rows:
        overlaps = sorted(pdc_to_x3.get(row["segment_id"], []))
        row["comparison_to_other_arm"] = {
            "relation": "OVERLAPS_X3_FALSE" if overlaps else "RESOLVED_BY_X3",
            "overlapping_segment_ids": overlaps,
            "split_into_multiple_x3_segments": len(overlaps) > 1,
        }
    for row in x3_rows:
        overlaps = sorted(x3_to_pdc.get(row["segment_id"], []))
        temporal_fragment = any(
            len(pdc_to_x3[pdc_id]) > 1 for pdc_id in overlaps
        )
        original_cause = str(row["primary_cause"])
        if temporal_fragment and original_cause in {
            REAL_MOVER_NONCRITICAL,
            ROUTE_GEOMETRY_MISS,
            NOT_EVALUABLE,
        }:
            row["primary_cause"] = FRAGMENTATION
        row["comparison_to_other_arm"] = {
            "relation": "OVERLAPS_PDC_FALSE" if overlaps else "X3_INCREMENTAL_FALSE",
            "overlapping_segment_ids": overlaps,
            "temporal_fragmentation": temporal_fragment,
            "pre_temporal_fragmentation_cause": original_cause,
        }


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["primary_cause"]) for row in rows).items()))


def _per_sequence(
    score_rows: Mapping[str, Mapping[str, Any]],
    pdc_rows: Sequence[Mapping[str, Any]],
    x3_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for sequence in sorted(score_rows):
        pdc = [row for row in pdc_rows if row["sequence"] == sequence]
        x3 = [row for row in x3_rows if row["sequence"] == sequence]
        scores = score_rows[sequence]["scores"]
        output.append(
            {
                "sequence": sequence,
                "PDC_false_segments": len(pdc),
                "X3_false_segments": len(x3),
                "false_segment_delta": len(x3) - len(pdc),
                "PDC_primary_causes": _counts(pdc),
                "X3_primary_causes": _counts(x3),
                "X3_incremental_false_segments": sum(
                    row["comparison_to_other_arm"]["relation"] == "X3_INCREMENTAL_FALSE"
                    for row in x3
                ),
                "PDC_resolved_false_segments": sum(
                    row["comparison_to_other_arm"]["relation"] == "RESOLVED_BY_X3"
                    for row in pdc
                ),
                "PDC_contact_recall": int(scores["M1_PDC_GLOBAL"]["bounded_contact_events_recalled"]),
                "X3_contact_recall": int(scores["X3_LAG_FLOXEL"]["bounded_contact_events_recalled"]),
            }
        )
    return output


def _decision(x3_counts: Mapping[str, int], incremental_counts: Mapping[str, int]) -> dict[str, Any]:
    source_total = sum(int(x3_counts.get(name, 0)) for name in SOURCE_FAILURES)
    incremental_source = sum(int(incremental_counts.get(name, 0)) for name in SOURCE_FAILURES)
    total = sum(int(value) for value in x3_counts.values())
    incremental_total = sum(int(value) for value in incremental_counts.values())
    if source_total * 2 >= total or (
        incremental_total and incremental_source * 2 >= incremental_total
    ):
        next_source = "STATIC_AWARE_DIRECTION_CONSISTENT_SCENE_FLOW"
        reason = (
            "Static pseudo-motion and flow/direction errors dominate X3 false segments or "
            "their incremental subset; change the motion information source, not thresholds."
        )
    elif int(x3_counts.get(REAL_MOVER_NONCRITICAL, 0)) >= max(x3_counts.values(), default=0):
        next_source = "SEMANTIC_MOVER_ROUTE_RELEVANCE"
        reason = "Correct real movers dominate but do not realize route entry; source relevance is missing."
    elif int(x3_counts.get(FRAGMENTATION, 0)) >= max(x3_counts.values(), default=0):
        next_source = "TEMPORALLY_COHERENT_SCENE_FLOW_COMPONENTS"
        reason = "Spatial or temporal source fragmentation dominates the sealed false episodes."
    else:
        next_source = "CONTINUOUS_BODY_ROUTE_GEOMETRY_WITH_SOURCE_IDENTITY"
        reason = "Motion is often supported, while route geometry or source binding remains dominant."
    return {
        "close_current_x3": True,
        "next_information_source": next_source,
        "reason": reason,
        "threshold_seed_backbone_or_scorer_sweep": False,
        "source_failure_segments": source_total,
        "source_failure_fraction": source_total / total if total else None,
        "incremental_source_failure_segments": incremental_source,
        "incremental_source_failure_fraction": (
            incremental_source / incremental_total if incremental_total else None
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    sealed = _validate_sealed_chain(root)
    predictions = sealed["predictions"]
    result = sealed["result"]

    roster_path = args.roster.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    roster = _load_json(roster_path)
    require(
        roster["source_authority"]["labels_sha256"] == sha256_file(labels_path),
        "x3_attr_labels_hash",
    )
    require(
        roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path),
        "x3_attr_timestamps_hash",
    )
    require(result["source"]["labels_sha256"] == sha256_file(labels_path), "x3_attr_result_labels")
    require(
        result["source"]["timestamps_sha256"] == sha256_file(timestamps_path),
        "x3_attr_result_timestamps",
    )
    score_rows = {str(row["sequence"]): row for row in result["per_sequence"]}
    prediction_rows = {str(row["sequence"]): row for row in predictions["sequences"]}
    roster_rows = {str(row["sequence"]): row for row in roster["selected_sequences"]}
    require(
        len(score_rows) == 6
        and set(score_rows) == set(prediction_rows) == set(roster_rows),
        "x3_attr_sequence_coverage",
    )

    pdc_false: list[dict[str, Any]] = []
    x3_false: list[dict[str, Any]] = []
    with zipfile.ZipFile(timestamps_path) as timestamp_bundle:
        for sequence in sorted(score_rows):
            timestamps = _load_timestamps(timestamp_bundle, sequence)
            frames = sorted(timestamps)
            prediction_row = prediction_rows[sequence]
            bag_path = _recorded_path(prediction_row["baseline_sources"]["bag"], label=f"bag:{sequence}")
            require(
                prediction_row["baseline_sources"]["bag_sha256"] == sha256_file(bag_path),
                f"x3_attr_bag_hash:{sequence}",
            )
            pose_samples, _rgb, _authority = read_bag_pose_and_rgb(bag_path)
            poses = {
                frame: _causal_pose(pose_samples, round(float(timestamps[frame]) * 1e9))
                for frame in frames
            }
            boxes_by_frame = x0.load_native_boxes(
                labels_path, timestamps, poses, sequence=sequence
            )
            history = x0._box_history(boxes_by_frame)
            pdc_source = prediction_row["baseline_sources"]["ledgers"]["M1_PDC_GLOBAL"]
            x3_source = prediction_row["candidate_ledger"]
            pdc_ledger = x0._load_ledger(
                Path(pdc_source["ledger"]), pdc_source["ledger_sha256"]
            )
            x3_ledger = x0._load_ledger(
                Path(x3_source["ledger"]), x3_source["ledger_sha256"]
            )
            scores = score_rows[sequence]["scores"]
            pdc_false.extend(
                _diagnose_ranges(
                    arm="PDC",
                    sequence=sequence,
                    ranges=scores["M1_PDC_GLOBAL"]["false_alert_ranges"],
                    ledger=pdc_ledger,
                    frames=frames,
                    timestamps=timestamps,
                    boxes_by_frame=boxes_by_frame,
                    history=history,
                    poses=poses,
                )
            )
            x3_false.extend(
                _diagnose_ranges(
                    arm="X3",
                    sequence=sequence,
                    ranges=scores["X3_LAG_FLOXEL"]["false_alert_ranges"],
                    ledger=x3_ledger,
                    frames=frames,
                    timestamps=timestamps,
                    boxes_by_frame=boxes_by_frame,
                    history=history,
                    poses=poses,
                )
            )

    require(len(pdc_false) == 25, "x3_attr_pdc_false_count")
    require(len(x3_false) == 94, "x3_attr_x3_false_count")
    _annotate_comparison(pdc_false, x3_false)
    pdc_counts = _counts(pdc_false)
    x3_counts = _counts(x3_false)
    incremental = [
        row
        for row in x3_false
        if row["comparison_to_other_arm"]["relation"] == "X3_INCREMENTAL_FALSE"
    ]
    resolved = [
        row
        for row in pdc_false
        if row["comparison_to_other_arm"]["relation"] == "RESOLVED_BY_X3"
    ]
    incremental_counts = _counts(incremental)
    per_sequence = _per_sequence(score_rows, pdc_false, x3_false)
    decision = _decision(x3_counts, incremental_counts)
    output = {
        "schema": SCHEMA,
        "status": STATUS,
        "question": (
            "Why did sealed X3 recover all six CONTACT events yet increase false segments "
            "from 25 to 94 relative to PDC?"
        ),
        "frozen_contract": {
            "predictions_or_scorer_changed": False,
            "threshold_seed_backbone_or_cohort_sweep": False,
            "native_truth_opened_after_evidence_chain_validation": True,
            "association_margin_m": x0.ASSOCIATION_MARGIN_M,
            "flow_error_limit_mps": x0.FLOW_ERROR_LIMIT_MPS,
            "dynamic_speed_floor_mps": x0.FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps,
            "route_horizon_s": x0.HORIZON_S,
        },
        "summary": {
            "PDC_false_segments": len(pdc_false),
            "X3_false_segments": len(x3_false),
            "false_segment_delta": len(x3_false) - len(pdc_false),
            "PDC_primary_causes": pdc_counts,
            "X3_primary_causes": x3_counts,
            "X3_incremental_false_segments": len(incremental),
            "X3_incremental_primary_causes": incremental_counts,
            "PDC_resolved_false_segments": len(resolved),
            "X3_temporal_fragment_segments": sum(
                bool(row["comparison_to_other_arm"]["temporal_fragmentation"])
                for row in x3_false
            ),
            "contact_recall": {
                "PDC": int(result["metrics"]["M1_PDC_GLOBAL"]["contact_recall"]),
                "X3": int(result["metrics"]["X3_LAG_FLOXEL"]["contact_recall"]),
            },
            "event_f1": {
                "PDC": float(result["metrics"]["M1_PDC_GLOBAL"]["event_f1"]),
                "X3": float(result["metrics"]["X3_LAG_FLOXEL"]["event_f1"]),
            },
        },
        "per_sequence": per_sequence,
        "decision": decision,
        "PDC_false_segments": pdc_false,
        "X3_false_segments": x3_false,
        "definitions": {
            STATIC_PSEUDO_MOTION: (
                "Risk cell has no current native OBB support or matches an object below "
                "the frozen dynamic-speed floor."
            ),
            BAD_FLOW_DIRECTION: (
                "Associated moving OBB exists, flow error exceeds the frozen limit, and "
                "estimated and native velocities have non-positive dot product."
            ),
            BAD_FLOW_MAGNITUDE: (
                "Associated moving OBB exists and flow error exceeds the frozen limit, "
                "without a full direction reversal."
            ),
            REAL_MOVER_NONCRITICAL: (
                "Flow agrees with a moving native OBB whose realized three-second future "
                "does not enter the body route."
            ),
            ROUTE_GEOMETRY_MISS: (
                "Motion agrees and the native mover realizes route entry, leaving discrete "
                "point/route geometry as the mismatch."
            ),
            FRAGMENTATION: (
                "A native identity is split across compatible local groups, or X3 divides "
                "one overlapping PDC false episode into multiple false segments without a "
                "dominant source-flow error."
            ),
            WRONG_COMPONENT_BINDING: "A compatible local group mixes native identities and unmatched support.",
            NOT_EVALUABLE: "No admissible causal native history or risk cell is available.",
        },
        "source": {
            "evidence_chain": str(sealed["chain_path"]),
            "evidence_chain_sha256": sealed["chain_sha256"],
            "predictions": str(sealed["predictions_path"]),
            "predictions_sha256": sha256_file(sealed["predictions_path"]),
            "result": str(sealed["result_path"]),
            "result_sha256": sha256_file(sealed["result_path"]),
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "amendment_checkpoint_available_after_cleanup": sealed[
                "amendment_checkpoint_available"
            ],
        },
        "claim_limits": [
            "Post-outcome attribution on already opened Development truth; not new performance evidence.",
            "Native OBB identity and trajectory are privileged diagnostics, not deployable inputs.",
            "No-OBB support is consistent with static pseudo-motion but may include unlabeled movers.",
            "Temporal fragmentation is comparative overlap structure, not proof of persistent identity failure.",
            "The amendment checkpoint may be absent after verified cleanup; its pre-cleanup hash remains bound by the sealed evidence chain.",
            "No threshold, seed, backbone, scorer, lifecycle, route geometry, or cohort was changed.",
        ],
    }
    output_path = args.output.resolve()
    if output_path.exists():
        require(_load_json(output_path) == output, f"x3_attr_existing_output_differs:{output_path}")
    else:
        write_json(output_path, output)
    require(_load_json(output_path) == output, "x3_attr_output_verify")
    return output


def parse_args() -> argparse.Namespace:
    root = REPO / "artifacts.local" / "evidence" / "dtr-x3" / "full-lag-floxel-replay-mp"
    source = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--roster",
        type=Path,
        default=REPO
        / "research"
        / "active"
        / "dtr-r0"
        / "dtr_c31_fresh_confirmation_roster.json",
    )
    parser.add_argument("--labels", type=Path, default=source / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=source / "train_timestamps.zip")
    parser.add_argument("--output", type=Path, default=root / "failure-attribution.json")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the candidate-blind replacement holdout amendment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contract import load_json, sha256_file, validate_prereg


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_bound(repo: Path, binding: dict[str, Any], label: str) -> Path:
    path = repo / binding["path"]
    require(path.is_file(), f"{label} missing: {path}")
    require(sha256_file(path) == binding["sha256"], f"{label} hash mismatch")
    return path


def validate_transport_lineage(repo: Path, binding: dict[str, Any]) -> list[str]:
    visited: set[Path] = set()
    hashes: list[str] = []
    current = binding
    while current:
        path = resolve_bound(repo, current, "materialization transport lineage")
        require(path not in visited, "materialization transport lineage contains a cycle")
        visited.add(path)
        hashes.append(current["sha256"])
        payload = load_json(path)
        require(
            payload.get("candidate_outputs_executed") is False,
            "materialization transport lineage exposed candidate outputs",
        )
        require(
            payload.get("scope") == "transport_only_no_candidate_no_truth_no_evaluator",
            "materialization transport lineage scope drifted",
        )
        current = payload.get("predecessor_config")
    require(len(hashes) >= 1, "materialization transport lineage is empty")
    return hashes


def resolve_materializer_amendment(
    repo: Path,
    binding: dict[str, Any],
    amendment: dict[str, Any] | None,
    replacement_sha256: str,
) -> Path:
    path = repo / binding["path"]
    require(path.is_file(), f"replacement implementation missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 == binding["sha256"]:
        return path
    require(amendment is not None, "materializer drifted without materialization amendment")
    require(
        amendment["superseded_materializer_implementation_sha256"] == binding["sha256"],
        "materialization amendment does not supersede frozen materializer",
    )
    require(
        amendment["materializer_implementation_path"] == binding["path"],
        "materialization amendment path mismatch",
    )
    require(
        amendment["materializer_implementation_sha256"] == actual_sha256,
        "materialization amendment implementation hash mismatch",
    )
    require(
        amendment["replacement_preregistration_sha256"] == replacement_sha256,
        "materialization amendment replacement hash mismatch",
    )
    streamer = repo / amendment["streamer_implementation_path"]
    require(streamer.is_file(), "materialization amendment streamer missing")
    require(
        sha256_file(streamer) == amendment["streamer_implementation_sha256"],
        "materialization amendment streamer hash mismatch",
    )
    validate_transport_lineage(repo, amendment["transport_lineage_tip"])
    require(
        amendment["frozen_after_materialization_before_truth_freeze"] is True,
        "materialization amendment timing is not candidate blind",
    )
    require(
        amendment["candidate_outputs_executed_before_amendment"] is False
        and amendment["app_detector_or_event_outputs_exposed_before_amendment"] is False,
        "materialization amendment followed candidate output exposure",
    )
    for key in (
        "candidate_or_truth_algorithm_change",
        "candidate_threshold_nms_tracker_or_scalar_change",
        "selection_gate_change",
        "integrity_contract_change",
    ):
        require(amendment[key] is False, f"materialization amendment changed forbidden scope: {key}")
    require(amendment["android_shadow"] == "closed", "materialization amendment opened Android shadow")
    require(
        amendment["h2_depth_ttc_route_risk_flip"] == "closed",
        "materialization amendment opened H2",
    )
    return path


def source_row(payload: dict[str, Any], source_id: str) -> dict[str, Any]:
    matches = [row for row in payload["sources"] if row["source_id"] == source_id]
    require(len(matches) == 1, f"capacity receipt source mismatch: {source_id}")
    return matches[0]


def sequence_name_from_bag(entry_name: str) -> str:
    return Path(entry_name).stem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacement", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--require-unopened", action="store_true")
    parser.add_argument("--scoring-amendment", type=Path)
    parser.add_argument("--materialization-amendment", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    replacement = load_json(args.replacement)
    require(
        replacement.get("schema")
        == "blindassist_ustrf_route_target_evidence_closure_replacement_holdout_r1",
        "replacement schema mismatch",
    )
    base_path = resolve_bound(repo, replacement["base_preregistration"], "base preregistration")
    base = validate_prereg(load_json(base_path), repo=repo)
    candidate_path = resolve_bound(repo, replacement["candidate_implementation"], "candidate implementation")
    require(candidate_path.name == "candidates.py", "candidate implementation path drifted")
    require(replacement["candidate_implementation"]["candidate_count"] == 3, "candidate count drifted")
    require(replacement["candidate_implementation"]["changed_for_replacement"] is False, "candidate changed")

    failed = replacement["failed_first_holdout"]
    require(failed["candidate_outputs_executed"] is False, "failed holdout exposed candidates")
    failed_fusion = load_json(resolve_bound(repo, failed["fusion"], "failed holdout fusion"))
    failed_truth = load_json(resolve_bound(repo, failed["truth_windows"], "failed holdout truth windows"))
    require(
        failed_fusion.get("candidate_outputs_executed") is False
        and failed_fusion.get("app_detector_or_event_outputs_exposed") is False,
        "failed fusion was not candidate blind",
    )
    require(
        failed_truth.get("candidate_outputs_executed") is False
        and failed_truth.get("app_detector_or_event_outputs_exposed") is False
        and failed_truth.get("admitted_source_count") == 0,
        "failed truth receipt does not prove zero-source admission",
    )
    require(failed["candidate_performance_observed"] is False, "replacement used candidate performance")

    source_ids = [row["source_id"] for row in replacement["replacement_sources"]]
    require(
        source_ids == ["crowdbot_0410_mds", "crowdbot_1203_shared_control"],
        "replacement source order or identities drifted",
    )
    total_sequences = 0
    for source in replacement["replacement_sources"]:
        capacity_payload = load_json(resolve_bound(repo, source["capacity_receipt"], "capacity receipt"))
        require(capacity_payload.get("candidate_outputs_executed") is False, "capacity receipt exposed candidates")
        capacity = source_row(capacity_payload, source["source_id"])
        require(len(capacity["sequence_metrics"]) == source["metadata_sequence_count"], "sequence count drifted")
        total_sequences += source["metadata_sequence_count"]
        for key, expected in source["capacity"].items():
            require(capacity[key] == expected, f"capacity value drifted: {source['source_id']} {key}")
        require(capacity["scorable_route_minutes"] >= 10.0, "route exposure below frozen floor")
        require(capacity["negative_route_minutes_proxy"] >= 10.0, "negative exposure below frozen floor")
        require(capacity["positive_event_capacity_proxy"] >= 10, "positive capacity below frozen floor")
        require(capacity["critical_event_capacity_proxy"] >= 2, "critical capacity below frozen floor")
        require(capacity["cooccurrence_frame_rate_proxy"] > 0.0, "cooccurrence missing")
        resolve_bound(repo, source["processed_inventory"], "processed inventory")
        metadata_sequences = {row["sequence_id"] for row in capacity["sequence_metrics"]}
        raw_sequences: set[str] = set()
        for binding in source["raw_inventories"]:
            inventory = load_json(resolve_bound(repo, binding, "raw inventory"))
            require(inventory.get("sequence_content_decoded") is False, "raw sequence content opened before freeze")
            for entry in inventory["entries"]:
                if entry["name"].endswith(".bag"):
                    raw_sequences.add(sequence_name_from_bag(entry["name"]))
        require(raw_sequences == metadata_sequences, f"raw bag coverage mismatch: {source['source_id']}")
        require(len(raw_sequences) == source["raw_bag_match_count"], "raw bag match count drifted")
    require(total_sequences == 23, "replacement sequence total drifted")
    modality_source_ids = []
    for binding in replacement["modality_admission_receipts"]:
        review = load_json(resolve_bound(repo, binding, "modality admission receipt"))
        modality_source_ids.append(binding["source_id"])
        require(review["source_id"] == binding["source_id"], "modality source binding drifted")
        require(review["decision"] == binding["decision"], "modality decision drifted")
        require(review.get("candidate_outputs_executed") is False, "modality review exposed candidates")
        require(review.get("production_authority") is False, "modality review created production authority")
    require(modality_source_ids == source_ids, "modality coverage differs from replacement sources")

    amendment = replacement["truth_protocol_amendment"]
    require(amendment["uses_candidate_or_app_detector_outputs"] is False, "truth amendment used candidate output")
    require(amendment["uses_failed_holdout_candidate_scores"] is False, "truth amendment used candidate scores")
    positive = amendment["positive_event_truth"]
    require(
        positive["event_source"]
        == "visible_metric_person_rows_after_visual_published_track_fusion_not_raw_lidar_event_onset",
        "positive event source drifted",
    )
    require(positive["missing_occluded_stale_or_unknown_can_clear"] is False, "missing can clear")
    require(positive["new_event_requires_prior_terminal_clear"] is True, "event regeneration contract drifted")
    negative = amendment["negative_truth"]
    require(negative["frame_unit"] == "route_relevant_person_truth_complete", "negative frame unit drifted")
    require(negative["requires_causal_route_known"] is True, "negative route can be unknown")
    require(negative["candidate_specific_denominator"] is False, "candidate-specific denominator enabled")
    require(
        replacement["execution_boundaries"]["candidate_outputs_executed"] is False
        and replacement["execution_boundaries"]["app_detector_or_event_outputs_exposed"] is False,
        "replacement is not sealed",
    )
    require(replacement["execution_boundaries"]["no_threshold_nms_tracker_or_scalar_retuning"] is True, "retuning enabled")
    scoring_amendment = load_json(args.scoring_amendment) if args.scoring_amendment else None
    materialization_amendment = (
        load_json(args.materialization_amendment) if args.materialization_amendment else None
    )
    implementation_paths = []
    for binding in replacement["implementation_bindings"]:
        path = repo / binding["path"]
        if path.name == "materialize_crowdbot_holdout_sources.py":
            implementation_paths.append(
                resolve_materializer_amendment(
                    repo,
                    binding,
                    materialization_amendment,
                    sha256_file(args.replacement),
                ).as_posix()
            )
        elif path.name == "run_crowdbot_holdout_candidates.py" and sha256_file(path) != binding["sha256"]:
            require(scoring_amendment is not None, "candidate runner drifted without scoring amendment")
            require(
                scoring_amendment["superseded_runner_implementation_sha256"] == binding["sha256"],
                "scoring amendment does not supersede frozen runner",
            )
            require(
                scoring_amendment["runner_implementation_sha256"] == sha256_file(path),
                "scoring amendment runner hash mismatch",
            )
            require(
                scoring_amendment["replacement_preregistration_sha256"] == sha256_file(args.replacement),
                "scoring amendment replacement hash mismatch",
            )
            require(
                scoring_amendment["candidate_implementation_sha256"]
                == replacement["candidate_implementation"]["sha256"],
                "scoring amendment candidate hash mismatch",
            )
            require(
                scoring_amendment["candidate_outputs_executed_before_freeze"] is False,
                "scoring amendment followed candidate execution",
            )
            implementation_paths.append(path.as_posix())
        else:
            implementation_paths.append(resolve_bound(repo, binding, "replacement implementation").as_posix())
    require(len(implementation_paths) == 11, "replacement implementation binding count drifted")
    frozen = amendment["frozen_scalars_unchanged"]
    require(frozen["person_confidence_threshold"] == base["frozen_axes"]["person_confidence_threshold"], "person threshold drifted")
    presence = base["sealed_holdout"]["holdout_truth_freeze_contract"]["all_person_presence"]
    require(frozen["visual_consensus_iou_min"] == presence["visual_consensus_iou_min"], "visual IoU drifted")
    require(frozen["route_point_margin_fraction"] == base["frozen_axes"]["route_point_margin_fraction"], "route margin drifted")
    base_holdout = base["sealed_holdout"]
    for key in (
        "minimum_positive_events_each_source",
        "minimum_critical_events_each_source",
        "minimum_matched_negative_windows_each_source",
        "minimum_scorable_negative_exposure_minutes_each_source",
    ):
        require(frozen[key] == base_holdout[key], f"admission scalar drifted: {key}")
    base_window = base_holdout["holdout_truth_freeze_contract"]["window_freeze"]
    require(frozen["positive_pre_context_frames"] == base_window["positive_pre_context_frames"], "pre context drifted")
    require(
        frozen["positive_post_clear_context_frames"] == base_window["positive_post_clear_context_frames"],
        "post context drifted",
    )
    if args.require_unopened:
        for key in ("dataset_root", "evidence_root"):
            require(not (repo / replacement["planned_outputs"][key]).exists(), f"replacement {key} already opened")
    print(
        json.dumps(
            {
                "status": "REPLACEMENT_HOLDOUT_FROZEN_CANDIDATE_BLIND",
                "replacement_sha256": sha256_file(args.replacement),
                "source_ids": source_ids,
                "sequence_count": total_sequences,
                "candidate_count": replacement["candidate_implementation"]["candidate_count"],
                "scoring_amendment_sha256": (
                    sha256_file(args.scoring_amendment) if args.scoring_amendment else None
                ),
                "materialization_amendment_sha256": (
                    sha256_file(args.materialization_amendment)
                    if args.materialization_amendment
                    else None
                ),
                "android_shadow": "closed",
                "h2": "closed",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

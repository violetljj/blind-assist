#!/usr/bin/env python3
"""Frozen TARO R38 preflight, identity lock, and one-shot R32 confirmation.

The preflight may read only final-reference RGB/depth/confidence/intrinsics and
run the frozen model on those reference RGB frames.  Candidate RGB/model output
and candidate sensor depth stay closed until the execution lock is frozen.
During confirmation every R32 score is sealed before candidate sensor depth is
opened for target-only evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.assistive_geometry import arkitscenes_truth_reader as arkit_truth
from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    arkitscenes_balanced_pose_source_frontdoor as arkit,
)
from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    balanced_pose_source_frontdoor as balanced,
)
from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    positive_oracle_canary as bonn,
)
from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    task_evidence_arkitscenes_opportunity_confirmation as opportunity,
)
from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    task_evidence_cross_source_learned_ranker as r21,
)
from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    task_evidence_oracle_canary as oracle,
)
from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    task_evidence_pose_scorer_canary as scorer,
)
from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    task_evidence_rgb_query_interaction_ranker as r25,
)
from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    task_evidence_source_anchored_monocular_geometry as r32,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCK_SCHEMA = "blindassist.taro.r38_arkitscenes_fresh_parent_source_lock.v1"
MATERIALIZATION_SCHEMA = "blindassist.taro.r38_arkitscenes_fresh_parent_materialization.v1"
PREFLIGHT_SCHEMA = "blindassist.taro.r38_r32_reference_input_preflight.v1"
EXECUTION_LOCK_SCHEMA = "blindassist.taro.r38_r32_arkitscenes_confirmation_execution_lock.v1"
RESULT_SCHEMA = "blindassist.taro.r38_r32_arkitscenes_confirmation_result.v1"
SELECTION_SEAL_SCHEMA = "blindassist.taro.r38_r32_selection_seal.v1"

DEFAULT_SOURCE_LOCK = REPO_ROOT / "docs/research/taro/TARO_R38_ARKITSCENES_FRESH_PARENT_SOURCE_LOCK_2026-08-15.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "artifacts.local/datasets/taro-r38-arkitscenes-fresh-parent-r0"
DEFAULT_EVIDENCE_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/r38-r32-arkitscenes-fresh-parent-confirmation-r0"
DEFAULT_PREFLIGHT = DEFAULT_EVIDENCE_ROOT / "preflight.json"
DEFAULT_EXECUTION_LOCK = REPO_ROOT / "docs/research/taro/TARO_R38_R32_ARKITSCENES_FRESH_PARENT_CONFIRMATION_EXECUTION_LOCK_2026-08-15.json"
DEFAULT_RESULT = DEFAULT_EVIDENCE_ROOT / "result.json"
IMPLEMENTATION_PATH = Path(__file__).resolve()
R32_IMPLEMENTATION_PATH = Path(r32.__file__).resolve()
MAX_REFERENCES_PER_PARENT = 5


class R38Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R38Error(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()


def verify_content(value: Mapping[str, Any], label: str) -> None:
    require(value.get("content_sha256") == content_sha256(value), f"{label} content hash drift")


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _entry_stem(entry: Mapping[str, Any]) -> str:
    return Path(str(entry["path"])).stem


def _load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_source_lock(path: Path) -> dict[str, Any]:
    lock = _load_json(path)
    require(lock.get("schema") == SOURCE_LOCK_SCHEMA, "R38 source lock schema drift")
    require(
        lock.get("status") == "FROZEN_AFTER_METADATA_TRAJECTORY_AND_HEAD_ONLY_BEFORE_MEDIA_BODY_OR_TASK_OUTCOME",
        "R38 source lock status drift",
    )
    contract = lock.get("frozen_confirmation_contract", {})
    require(contract.get("algorithm") == "R32_REFERENCE_SENSOR_ANCHORED_MONOCULAR_CANDIDATE_GEOMETRY", "R38 algorithm drift")
    require(contract.get("checkpoint_sha256") == r32.EXPECTED_CHECKPOINT_SHA256, "R38 checkpoint contract drift")
    require(contract.get("candidate_sensor_depth_before_score_seal") is False, "R38 target firewall drift")
    require(contract.get("reference_and_candidate_identity_disjoint") is True, "R38 role firewall drift")
    require(contract.get("trainable_parameter_count") == 0, "R38 training contract drift")
    return lock


def load_roster(
    source_lock_path: Path,
    dataset_root: Path,
) -> tuple[
    list[bonn.Frame],
    dict[str, arkit.FrameAssets],
    dict[str, r25.RgbAsset],
    dict[str, Any],
    dict[str, Any],
]:
    source_lock = _validate_source_lock(source_lock_path)
    manifest_path = dataset_root / "manifest.json"
    manifest = _load_json(manifest_path)
    require(manifest.get("schema") == MATERIALIZATION_SCHEMA, "R38 materialization schema drift")
    verify_content(manifest, "R38 materialization")
    require(manifest.get("source_lock_sha256") == sha256_file(source_lock_path), "R38 materialization source-lock drift")
    require(manifest.get("parent_count") == len(manifest.get("parents", [])) == 12, "R38 materialized parent count drift")
    require(manifest.get("selected_frame_count") == 3600, "R38 materialized frame count drift")
    require(manifest.get("selected_frame_count_per_parent") == 300, "R38 per-parent frame count drift")
    require(manifest.get("reference_input_preflight_opened") is False, "R38 source checkpoint reference boundary drift")
    require(manifest.get("candidate_sensor_depth_pixel_decodes") == 0, "R38 source checkpoint candidate-depth drift")
    require(manifest.get("task_outcome_opened") is False, "R38 source checkpoint outcome drift")
    require(manifest.get("model_outputs_read") is False, "R38 source checkpoint model-read drift")

    lock_parent_ids = {str(row["video_id"]) for row in source_lock["parents"]}
    frames: list[bonn.Frame] = []
    assets: dict[str, arkit.FrameAssets] = {}
    rgb_assets: dict[str, r25.RgbAsset] = {}
    parent_receipts: list[dict[str, Any]] = []
    for parent in sorted(manifest["parents"], key=lambda row: str(row["video_id"])):
        parent_id = str(parent["video_id"])
        require(parent_id in lock_parent_ids, f"R38 parent not source-locked: {parent_id}")
        selected = [str(value) for value in parent["selected_frame_stems"]]
        require(len(selected) == 300, f"R38 selected-frame drift: {parent_id}")
        extracted = parent["extracted"]
        for modality in ("lowres_wide", "lowres_depth", "confidence", "lowres_wide_intrinsics"):
            require(len(extracted[modality]) == len(selected), f"R38 {modality} count drift: {parent_id}")
        trajectory_entry = parent["trajectory"]
        trajectory_path = Path(str(trajectory_entry["path"]))
        require(trajectory_path.is_file(), f"R38 trajectory missing: {parent_id}")
        require(trajectory_path.stat().st_size == int(trajectory_entry["bytes"]), f"R38 trajectory byte drift: {parent_id}")
        require(sha256_file(trajectory_path) == str(trajectory_entry["sha256"]), f"R38 trajectory hash drift: {parent_id}")
        trajectory = arkit_truth.parse_trajectory(trajectory_path)
        admitted = 0
        for index, stem in enumerate(selected):
            rgb_entry = extracted["lowres_wide"][index]
            depth_entry = extracted["lowres_depth"][index]
            confidence_entry = extracted["confidence"][index]
            intrinsics_entry = extracted["lowres_wide_intrinsics"][index]
            require(
                _entry_stem(rgb_entry) == _entry_stem(depth_entry) == _entry_stem(confidence_entry) == _entry_stem(intrinsics_entry) == stem,
                f"R38 frame stem drift: {parent_id}:{stem}",
            )
            timestamp = float(stem.rsplit("_", 1)[1])
            pose, _receipt = arkit_truth.interpolate_camera_to_world(trajectory, timestamp, arkit.MAX_POSE_BRACKET_S)
            canonical = arkit.canonical_landscape_pose(pose)
            require(canonical is not None, f"R38 materialized non-landscape frame: {parent_id}:{stem}")
            canonical_pose, orientation_index = canonical
            rgb_path = Path(str(rgb_entry["path"]))
            depth_path = Path(str(depth_entry["path"]))
            frame = bonn.Frame(parent_id, timestamp, rgb_path, depth_path, canonical_pose)
            require(frame.frame_id not in assets, f"R38 duplicate frame: {frame.frame_id}")
            assets[frame.frame_id] = arkit.FrameAssets(depth_entry, confidence_entry, intrinsics_entry, orientation_index)
            rgb_assets[frame.frame_id] = r25.RgbAsset(
                "file",
                rgb_path.parent,
                str(rgb_path),
                orientation_index,
                int(rgb_entry["bytes"]),
                str(rgb_entry["sha256"]),
            )
            frames.append(frame)
            admitted += 1
        parent_receipts.append(
            {
                "parent_id": parent_id,
                "visit_id": str(parent["visit_id"]),
                "frame_count": admitted,
                "trajectory_sha256": str(trajectory_entry["sha256"]),
            }
        )
    require({frame.parent_id for frame in frames} == lock_parent_ids, "R38 roster/source-lock parent mismatch")
    source = {
        "family": "ARKITSCENES_VALIDATION",
        "analysis_role": "FRESH_PARENT_WITHIN_SOURCE_CONFIRMATION",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_lock_path": str(source_lock_path),
        "source_lock_sha256": sha256_file(source_lock_path),
        "parent_count": len(parent_receipts),
        "frame_count": len(frames),
        "parents": parent_receipts,
        "selection_inputs": ["source-locked identities", "timestamps", "official trajectories"],
        "selection_reads_task_outcome": False,
    }
    return frames, assets, rgb_assets, source, source_lock


def role_disjoint_identity_rows(
    rows: Sequence[bonn.ReferenceSupport],
) -> tuple[list[dict[str, Any]], list[bonn.ReferenceSupport]]:
    """Remove any frame that holds a final-reference role from all candidate roles."""
    active = list(rows)
    while True:
        reference_ids = {row.reference.frame_id for row in active}
        retained: list[bonn.ReferenceSupport] = []
        for row in active:
            candidates = tuple(pair for pair in row.candidates if pair.neighbor.frame_id not in reference_ids)
            micro = tuple(pair for pair in row.micro_candidates if pair.neighbor.frame_id not in reference_ids)
            if candidates and micro:
                retained.append(bonn.ReferenceSupport(row.reference, candidates, micro))
        if len(retained) == len(active):
            active = retained
            break
        active = retained
    reference_ids = {row.reference.frame_id for row in active}
    identity_rows: list[dict[str, Any]] = []
    proposal_rows: list[bonn.ReferenceSupport] = []
    for row in active:
        proposals = oracle.pose_proposal_pairs(row)
        require(all(pair.neighbor.frame_id not in reference_ids for pair in proposals), "R38 role-disjoint filter failed")
        proposal_rows.append(bonn.ReferenceSupport(row.reference, proposals, tuple(pair for pair in proposals if pair in row.micro_candidates)))
        identity_rows.append(
            {
                "parent_id": row.reference.parent_id,
                "reference_frame_id": row.reference.frame_id,
                "candidate_frame_ids": [pair.neighbor.frame_id for pair in proposals],
                "candidate_pose_receipts": [
                    {
                        "frame_id": pair.neighbor.frame_id,
                        "gap_s": pair.gap_s,
                        "translation_m": pair.translation_m,
                        "rotation_deg": pair.rotation_deg,
                    }
                    for pair in proposals
                ],
            }
        )
    candidates = {frame_id for row in identity_rows for frame_id in row["candidate_frame_ids"]}
    require(reference_ids.isdisjoint(candidates), "R38 global role identities overlap")
    return identity_rows, proposal_rows


def confirmation_checks(
    contract: Mapping[str, Any],
    metrics: Mapping[str, Any],
    evaluated_parent_count: int,
    evaluated_reference_count: int,
    role_disjoint: bool,
) -> dict[str, bool]:
    opportunity_parents = int(metrics["opportunity_parent_count"])
    strict_parents = int(metrics["policy_strict_win_opportunity_parent_count"])
    required_strict = max(
        int(contract["minimum_strict_win_parent_count"]),
        math.ceil(float(contract["minimum_strict_win_fraction_of_opportunity_parents"]) * opportunity_parents),
    )
    macro = metrics["parent_macro"]
    return {
        "minimum_evaluated_parent_count": evaluated_parent_count >= int(contract["minimum_evaluated_parent_count"]),
        "minimum_evaluated_reference_count": evaluated_reference_count >= int(contract["minimum_evaluated_reference_count"]),
        "minimum_opportunity_parent_count": opportunity_parents >= int(contract["minimum_opportunity_parent_count"]),
        "opportunity_denominated_strict_win_gate": strict_parents >= required_strict,
        "ranker_parent_macro_beats_generic": float(macro["ranker"]) > float(macro["generic"]),
        "ranker_parent_macro_beats_passive": float(macro["ranker"]) > float(macro["passive"]),
        "same_one_extra_frame_budget": bool(contract["same_one_extra_frame_budget"]),
        "zero_retention_failures": int(contract["retention_failures_allowed"]) == 0,
        "reference_and_candidate_identity_disjoint": role_disjoint,
        "candidate_sensor_depth_opened_only_after_score_seal": True,
    }


def run_preflight(
    source_lock_path: Path,
    dataset_root: Path,
    output_path: Path,
    model_repo: Path,
    checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    require(not output_path.exists(), f"R38 preflight already exists: {output_path}")
    frames, assets, rgb_assets, source, source_lock = load_roster(source_lock_path, dataset_root)
    selected, capability = balanced.select_pose_capable_references(frames, MAX_REFERENCES_PER_PARENT)
    predictor = r32.SourceAnchoredMonocularGeometry(model_repo, checkpoint, device)
    store = r25.RgbStore(rgb_assets, "R38_REFERENCE_INPUT_ONLY_PREFLIGHT")
    evaluable: list[bonn.ReferenceSupport] = []
    reference_receipts: list[dict[str, Any]] = []
    abstentions: list[dict[str, str]] = []
    try:
        for row in selected:
            depth, confidence, intrinsics = arkit.load_observation(assets[row.reference.frame_id])
            low, points, valid = arkit.observation_geometry(depth, confidence, intrinsics)
            queries = oracle._queries(row.reference, low, intrinsics)
            if queries is None:
                abstentions.append({"reference_frame_id": row.reference.frame_id, "reason": "REFERENCE_QUERY_GEOMETRY_NOT_EVALUABLE"})
                continue
            reference_prediction = predictor.low_depth(row.reference, store.rgb(row.reference))
            try:
                _scale, anchor = r32.robust_reference_scale(low, reference_prediction, valid)
            except r32.R32Error as error:
                abstentions.append({"reference_frame_id": row.reference.frame_id, "reason": str(error)})
                continue
            static = oracle.query_evidence_cells(points, valid, queries)
            evaluable.append(row)
            reference_receipts.append(
                {
                    "parent_id": row.reference.parent_id,
                    "reference_frame_id": row.reference.frame_id,
                    "query_count": len(queries),
                    "static_evidence_cell_count": int(np.sum(static)),
                    **anchor,
                }
            )
    finally:
        rgb_receipt = store.receipt()
        store.close()
    identities, final_rows = role_disjoint_identity_rows(evaluable)
    final_reference_ids = {row.reference.frame_id for row in final_rows}
    reference_receipts = [row for row in reference_receipts if row["reference_frame_id"] in final_reference_ids]
    parent_count = len({row["parent_id"] for row in identities})
    reference_count = len(identities)
    candidate_ids = {frame_id for row in identities for frame_id in row["candidate_frame_ids"]}
    checks = {
        "minimum_evaluated_parent_count": parent_count >= int(source_lock["frozen_confirmation_contract"]["minimum_evaluated_parent_count"]),
        "minimum_evaluated_reference_count": reference_count >= int(source_lock["frozen_confirmation_contract"]["minimum_evaluated_reference_count"]),
        "all_final_references_have_proposals": all(row["candidate_frame_ids"] for row in identities),
        "reference_and_candidate_identity_disjoint": final_reference_ids.isdisjoint(candidate_ids),
        "reference_model_checkpoint_hash_valid": predictor.receipt()["checkpoint_sha256"] == source_lock["frozen_confirmation_contract"]["checkpoint_sha256"],
        "candidate_rgb_payload_decodes_zero": True,
        "candidate_model_runs_zero": True,
        "candidate_sensor_depth_pixel_decodes_zero": True,
        "task_outcome_reads_zero": True,
    }
    terminal = "TARO_R38_R32_REFERENCE_INPUT_PREFLIGHT_PASS" if all(checks.values()) else "STOP_TARO_R38_R32_REFERENCE_INPUT_PREFLIGHT_NOT_EVALUABLE"
    result: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "mode": "REFERENCE_INPUT_ONLY_BEFORE_ROLE_IDENTITY_LOCK",
        "source": source,
        "pose_pair_capability_before_reference_input": capability,
        "frozen_model": predictor.receipt(),
        "reference_receipts": reference_receipts,
        "reference_receipt_sha256": hashlib.sha256(canonical_json_bytes(reference_receipts)).hexdigest().upper(),
        "geometry_or_anchor_abstentions": abstentions,
        "final_identity_rows": identities,
        "final_identity_sha256": hashlib.sha256(canonical_json_bytes(identities)).hexdigest().upper(),
        "evaluated_parent_count": parent_count,
        "evaluated_reference_count": reference_count,
        "unique_candidate_identity_count": len(candidate_ids),
        "checks": checks,
        "terminal": terminal,
        "confirmation_lock_authorized": all(checks.values()),
        "read_boundary": {
            "reference_depth_confidence_intrinsics_reads": 3 * len(selected),
            "reference_rgb_payload_decodes": rgb_receipt["unique_rgb_payload_decode_count"],
            "reference_model_runs": predictor.receipt()["model_run_count"],
            "candidate_rgb_payload_decodes": 0,
            "candidate_model_runs": 0,
            "candidate_sensor_depth_pixel_decodes": 0,
            "task_outcome_reads": 0,
            "training_steps": 0,
            "network_requests": 0,
        },
        "claim_ceiling": "Reference-input-only evaluability and role-identity preflight. It is not a TARO result or confirmation.",
    }
    result["content_sha256"] = content_sha256(result)
    write_json_exclusive(output_path, result)
    return result


def freeze_execution_lock(
    source_lock_path: Path,
    dataset_root: Path,
    preflight_path: Path,
    output_path: Path,
    model_repo: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    require(not output_path.exists(), f"R38 execution lock already exists: {output_path}")
    source_lock = _validate_source_lock(source_lock_path)
    preflight = _load_json(preflight_path)
    require(preflight.get("schema") == PREFLIGHT_SCHEMA, "R38 preflight schema drift")
    verify_content(preflight, "R38 preflight")
    require(preflight.get("confirmation_lock_authorized") is True, "R38 preflight did not authorize confirmation")
    require(preflight.get("terminal") == "TARO_R38_R32_REFERENCE_INPUT_PREFLIGHT_PASS", "R38 preflight terminal drift")
    require(preflight["read_boundary"]["candidate_sensor_depth_pixel_decodes"] == 0, "R38 candidate depth opened before lock")
    require(preflight["read_boundary"]["candidate_model_runs"] == 0, "R38 candidate model output opened before lock")
    manifest_path = dataset_root / "manifest.json"
    result: dict[str, Any] = {
        "schema": EXECUTION_LOCK_SCHEMA,
        "lock_id": "TARO_R38_R32_ARKITSCENES_FRESH_PARENT_CONFIRMATION_EXECUTION_LOCK",
        "status": "FROZEN_AFTER_REFERENCE_INPUT_PREFLIGHT_BEFORE_CANDIDATE_RGB_MODEL_OUTPUT_OR_SENSOR_DEPTH",
        "source_lock": {"path": str(source_lock_path), "bytes": source_lock_path.stat().st_size, "sha256": sha256_file(source_lock_path)},
        "materialization_manifest": {"path": str(manifest_path), "bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path)},
        "reference_preflight": {"path": str(preflight_path), "bytes": preflight_path.stat().st_size, "sha256": sha256_file(preflight_path), "content_sha256": preflight["content_sha256"]},
        "implementation": {"path": str(IMPLEMENTATION_PATH), "bytes": IMPLEMENTATION_PATH.stat().st_size, "sha256": sha256_file(IMPLEMENTATION_PATH)},
        "r32_implementation": {"path": str(R32_IMPLEMENTATION_PATH), "bytes": R32_IMPLEMENTATION_PATH.stat().st_size, "sha256": sha256_file(R32_IMPLEMENTATION_PATH)},
        "model": {
            "repository": str(model_repo),
            "dpt_source_path": str(model_repo / "metric_depth/depth_anything_v2/dpt.py"),
            "dpt_source_sha256": sha256_file(model_repo / "metric_depth/depth_anything_v2/dpt.py"),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "device_runtime_not_claim_relevant": True,
            "trainable_parameter_count": 0,
        },
        "frozen_confirmation_contract": source_lock["frozen_confirmation_contract"],
        "identity_rows": preflight["final_identity_rows"],
        "identity_sha256": preflight["final_identity_sha256"],
        "candidate_sensor_depth_before_score_seal": False,
        "candidate_rgb_or_model_output_before_this_lock": False,
        "execution_count": 1,
        "evidence_root": str(DEFAULT_EVIDENCE_ROOT),
        "claim_ceiling": source_lock["claim_ceiling"],
    }
    require(result["model"]["checkpoint_sha256"] == r32.EXPECTED_CHECKPOINT_SHA256, "R38 checkpoint hash drift at freeze")
    require(result["model"]["dpt_source_sha256"] == r32.EXPECTED_DPT_SHA256, "R38 DPT source hash drift at freeze")
    result["content_sha256"] = content_sha256(result)
    write_json_exclusive(output_path, result)
    return result


def _verify_binding(entry: Mapping[str, Any], label: str) -> Path:
    path = Path(str(entry["path"]))
    require(path.is_file(), f"R38 bound {label} missing: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"R38 bound {label} byte drift")
    require(sha256_file(path) == str(entry["sha256"]), f"R38 bound {label} hash drift")
    return path


def _locked_rows(
    frames: Sequence[bonn.Frame],
    identity_rows: Sequence[Mapping[str, Any]],
) -> list[bonn.ReferenceSupport]:
    selected, _capability = balanced.select_pose_capable_references(frames, MAX_REFERENCES_PER_PARENT)
    by_reference = {row.reference.frame_id: row for row in selected}
    output: list[bonn.ReferenceSupport] = []
    for identity in identity_rows:
        reference_id = str(identity["reference_frame_id"])
        require(reference_id in by_reference, f"R38 locked reference unavailable: {reference_id}")
        row = by_reference[reference_id]
        by_candidate = {pair.neighbor.frame_id: pair for pair in row.candidates}
        candidate_ids = [str(value) for value in identity["candidate_frame_ids"]]
        require(all(frame_id in by_candidate for frame_id in candidate_ids), f"R38 locked candidate unavailable: {reference_id}")
        pairs = tuple(by_candidate[frame_id] for frame_id in candidate_ids)
        receipts = [
            {"frame_id": pair.neighbor.frame_id, "gap_s": pair.gap_s, "translation_m": pair.translation_m, "rotation_deg": pair.rotation_deg}
            for pair in pairs
        ]
        require(receipts == identity["candidate_pose_receipts"], f"R38 locked pose receipt drift: {reference_id}")
        output.append(bonn.ReferenceSupport(row.reference, pairs, tuple()))
    references = {row.reference.frame_id for row in output}
    candidates = {pair.neighbor.frame_id for row in output for pair in row.candidates}
    require(references.isdisjoint(candidates), "R38 locked global role overlap")
    return output


def run_confirmation(
    execution_lock_path: Path,
    result_path: Path,
    device: str,
) -> dict[str, Any]:
    execution_lock = _load_json(execution_lock_path)
    require(execution_lock.get("schema") == EXECUTION_LOCK_SCHEMA, "R38 execution lock schema drift")
    verify_content(execution_lock, "R38 execution lock")
    require(execution_lock.get("status") == "FROZEN_AFTER_REFERENCE_INPUT_PREFLIGHT_BEFORE_CANDIDATE_RGB_MODEL_OUTPUT_OR_SENSOR_DEPTH", "R38 execution lock status drift")
    require(int(execution_lock.get("execution_count", 0)) == 1, "R38 execution count drift")
    source_lock_path = _verify_binding(execution_lock["source_lock"], "source lock")
    manifest_path = _verify_binding(execution_lock["materialization_manifest"], "materialization manifest")
    _verify_binding(execution_lock["reference_preflight"], "reference preflight")
    _verify_binding(execution_lock["implementation"], "implementation")
    _verify_binding(execution_lock["r32_implementation"], "R32 implementation")
    model_repo = Path(str(execution_lock["model"]["repository"]))
    checkpoint = Path(str(execution_lock["model"]["checkpoint_path"]))
    require(sha256_file(checkpoint) == execution_lock["model"]["checkpoint_sha256"], "R38 checkpoint drift")
    require(sha256_file(Path(str(execution_lock["model"]["dpt_source_path"]))) == execution_lock["model"]["dpt_source_sha256"], "R38 DPT source drift")
    dataset_root = manifest_path.parent
    frames, assets, rgb_assets, source, _source_lock = load_roster(source_lock_path, dataset_root)
    locked_rows = _locked_rows(frames, execution_lock["identity_rows"])
    require(hashlib.sha256(canonical_json_bytes(execution_lock["identity_rows"])).hexdigest().upper() == execution_lock["identity_sha256"], "R38 identity hash drift")

    evidence_root = result_path.parent
    attempt_path = evidence_root / "attempt.json"
    selection_seal_path = evidence_root / "selection-seal.json"
    require(not result_path.exists(), f"R38 result already exists: {result_path}")
    require(not attempt_path.exists(), f"R38 one-shot attempt already consumed: {attempt_path}")
    require(not selection_seal_path.exists(), f"R38 selection seal already exists: {selection_seal_path}")
    predictor = r32.SourceAnchoredMonocularGeometry(model_repo, checkpoint, device)
    write_json_exclusive(
        attempt_path,
        {
            "schema": "blindassist.taro.r38_r32_confirmation_attempt.v1",
            "execution_lock_path": str(execution_lock_path),
            "execution_lock_sha256": sha256_file(execution_lock_path),
            "identity_sha256": execution_lock["identity_sha256"],
            "status": "ONE_SHOT_CONSUMED_ON_CREATION_BEFORE_CANDIDATE_MODEL_RUN",
            "candidate_sensor_depth_pixel_decodes_at_creation": 0,
        },
    )

    store = r25.RgbStore(rgb_assets, "R38_LOCKED_REFERENCE_AND_CANDIDATE_RGB")
    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    try:
        for row in locked_rows:
            depth, confidence, intrinsics = arkit.load_observation(assets[row.reference.frame_id])
            low, points, valid = arkit.observation_geometry(depth, confidence, intrinsics)
            queries = oracle._queries(row.reference, low, intrinsics)
            require(queries is not None, f"R38 locked reference became non-evaluable: {row.reference.frame_id}")
            static = oracle.query_evidence_cells(points, valid, queries)
            context = scorer.ReferenceContext(row, low, points, valid, intrinsics, queries, static)
            contexts[row.reference.frame_id] = context
            reference_rgb = store.rgb(row.reference)
            for pair in row.candidates:
                features, analytic = predictor.candidate_features(context, pair, reference_rgb, store.rgb(pair.neighbor))
                records.append(
                    scorer.CandidateRecord(
                        row.reference.parent_id,
                        "FRESH_PARENT_WITHIN_SOURCE_CONFIRMATION",
                        row.reference.frame_id,
                        pair,
                        features,
                        analytic,
                    )
                )
    finally:
        rgb_receipt = store.receipt()
        store.close()

    scores = r32.analytic_scores(records)
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    selection_rows: list[dict[str, Any]] = []
    for reference_id, indices in sorted(by_reference.items()):
        chosen = max(indices, key=lambda index: (float(scores[index]), -records[index].pair.translation_m, records[index].pair.neighbor.frame_id))
        selection_rows.append(
            {
                "reference_frame_id": reference_id,
                "candidate_frame_id": records[chosen].pair.neighbor.frame_id,
                "score": float(scores[chosen]),
            }
        )
    selection_seal: dict[str, Any] = {
        "schema": SELECTION_SEAL_SCHEMA,
        "execution_lock_sha256": sha256_file(execution_lock_path),
        "identity_sha256": execution_lock["identity_sha256"],
        "selection_rows": selection_rows,
        "selection_identity_sha256": hashlib.sha256(canonical_json_bytes(selection_rows)).hexdigest().upper(),
        "candidate_sensor_depth_pixel_decodes_before_seal": 0,
        "target_gain_reads_before_seal": 0,
        "model_receipt_before_target": predictor.receipt(),
    }
    selection_seal["content_sha256"] = content_sha256(selection_seal)
    write_json_exclusive(selection_seal_path, selection_seal)

    observation_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    unique_candidate_ids = sorted({record.pair.neighbor.frame_id for record in records})
    for frame_id in unique_candidate_ids:
        depth, confidence, intrinsics = arkit.load_observation(assets[frame_id])
        low, points, valid = arkit.observation_geometry(depth, confidence, intrinsics)
        observation_cache[frame_id] = (low, points, valid, float(np.mean(valid)))
    scorer._attach_targets(records, contexts, observation_cache)
    fold = r21.fold_metrics(records, scores)
    metrics = {
        "parent_macro": fold["parent_macro"],
        "opportunity_parent_count": fold["opportunity_parent_count"],
        "policy_strict_win_opportunity_parent_count": fold["strict_win_parent_count"],
        "strict_win_fraction_of_opportunity_parents": (
            fold["strict_win_parent_count"] / fold["opportunity_parent_count"]
            if fold["opportunity_parent_count"]
            else None
        ),
        "per_parent": fold["per_parent"],
        "opportunity_per_parent": fold["opportunity_per_parent"],
    }
    references = {record.reference_id for record in records}
    candidates = {record.pair.neighbor.frame_id for record in records}
    checks = confirmation_checks(
        execution_lock["frozen_confirmation_contract"],
        metrics,
        len({record.parent_id for record in records}),
        len(references),
        references.isdisjoint(candidates),
    )
    passed = all(checks.values())
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "mode": "ONE_SHOT_PARENT_DISJOINT_WITHIN_ARKITSCENES_VALIDATION_CONFIRMATION",
        "execution_lock": {"path": str(execution_lock_path), "sha256": sha256_file(execution_lock_path), "content_sha256": execution_lock["content_sha256"]},
        "source": source,
        "algorithm": "R32_REFERENCE_SENSOR_ANCHORED_MONOCULAR_CANDIDATE_GEOMETRY",
        "frozen_model": predictor.receipt(),
        "identity_sha256": execution_lock["identity_sha256"],
        "selection_seal": {"path": str(selection_seal_path), "sha256": sha256_file(selection_seal_path), "content_sha256": selection_seal["content_sha256"], "selection_identity_sha256": selection_seal["selection_identity_sha256"]},
        "evaluated_parent_count": len({record.parent_id for record in records}),
        "evaluated_reference_count": len(references),
        "candidate_row_count": len(records),
        "unique_candidate_identity_count": len(candidates),
        "metrics": metrics,
        "checks": checks,
        "terminal": "TARO_R38_R32_ARKITSCENES_FRESH_PARENT_CONFIRMATION_PASS" if passed else "STOP_TARO_R38_R32_ARKITSCENES_FRESH_PARENT_CONFIRMATION_FAIL",
        "r32_parent_disjoint_within_source_confirmation": passed,
        "further_taro_rescue_authorized": False,
        "android_candidate_authorized": False,
        "product_authorized": False,
        "safety_authorized": False,
        "read_boundary": {
            "reference_depth_confidence_intrinsics_reads": 3 * len(references),
            "reference_and_candidate_rgb_payload_decodes": rgb_receipt["unique_rgb_payload_decode_count"],
            "model_runs_before_target": selection_seal["model_receipt_before_target"]["model_run_count"],
            "candidate_sensor_depth_confidence_intrinsics_reads_after_score_seal": 3 * len(unique_candidate_ids),
            "candidate_sensor_depth_before_score_seal": False,
            "training_steps": 0,
            "network_requests": 0,
        },
        "claim_ceiling": "A PASS confirms the frozen R32 selector only on parent-disjoint ARKitScenes Validation parents. It is not fresh-source, broad-domain, Android, product, navigation, collision, deployment, or safety evidence.",
    }
    result["content_sha256"] = content_sha256(result)
    write_json_exclusive(result_path, result)
    return result


def verify_result(execution_lock_path: Path, result_path: Path) -> dict[str, Any]:
    execution_lock = _load_json(execution_lock_path)
    verify_content(execution_lock, "R38 execution lock")
    result = _load_json(result_path)
    require(result.get("schema") == RESULT_SCHEMA, "R38 result schema drift")
    verify_content(result, "R38 result")
    require(result["execution_lock"]["sha256"] == sha256_file(execution_lock_path), "R38 result lock binding drift")
    selection_path = Path(str(result["selection_seal"]["path"]))
    selection = _load_json(selection_path)
    verify_content(selection, "R38 selection seal")
    require(result["selection_seal"]["sha256"] == sha256_file(selection_path), "R38 selection binding drift")
    require(result["identity_sha256"] == execution_lock["identity_sha256"] == selection["identity_sha256"], "R38 identity binding drift")
    passed = all(bool(value) for value in result["checks"].values())
    expected = "TARO_R38_R32_ARKITSCENES_FRESH_PARENT_CONFIRMATION_PASS" if passed else "STOP_TARO_R38_R32_ARKITSCENES_FRESH_PARENT_CONFIRMATION_FAIL"
    require(result["terminal"] == expected, "R38 terminal/check mismatch")
    require(result["r32_parent_disjoint_within_source_confirmation"] is passed, "R38 pass flag drift")
    return {
        "schema": "blindassist.taro.r38_r32_confirmation_verification.v1",
        "passed": True,
        "terminal": "TARO_R38_R32_CONFIRMATION_EVIDENCE_VERIFIED",
        "result_terminal": result["terminal"],
        "result_sha256": sha256_file(result_path),
        "result_content_sha256": result["content_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    preflight.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    preflight.add_argument("--output", type=Path, default=DEFAULT_PREFLIGHT)
    preflight.add_argument("--model-repo", type=Path, default=r32.DEFAULT_MODEL_REPO)
    preflight.add_argument("--checkpoint", type=Path, default=r32.DEFAULT_CHECKPOINT)
    preflight.add_argument("--device", default="cuda")

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    freeze.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    freeze.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    freeze.add_argument("--output", type=Path, default=DEFAULT_EXECUTION_LOCK)
    freeze.add_argument("--model-repo", type=Path, default=r32.DEFAULT_MODEL_REPO)
    freeze.add_argument("--checkpoint", type=Path, default=r32.DEFAULT_CHECKPOINT)

    confirm = subparsers.add_parser("confirm")
    confirm.add_argument("--execution-lock", type=Path, default=DEFAULT_EXECUTION_LOCK)
    confirm.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    confirm.add_argument("--device", default="cuda")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--execution-lock", type=Path, default=DEFAULT_EXECUTION_LOCK)
    verify.add_argument("--result", type=Path, default=DEFAULT_RESULT)

    args = parser.parse_args()
    if args.command == "preflight":
        result = run_preflight(args.source_lock.resolve(), args.dataset_root.resolve(), args.output.resolve(), args.model_repo.resolve(), args.checkpoint.resolve(), args.device)
    elif args.command == "freeze":
        result = freeze_execution_lock(args.source_lock.resolve(), args.dataset_root.resolve(), args.preflight.resolve(), args.output.resolve(), args.model_repo.resolve(), args.checkpoint.resolve())
    elif args.command == "confirm":
        result = run_confirmation(args.execution_lock.resolve(), args.result.resolve(), args.device)
    else:
        result = verify_result(args.execution_lock.resolve(), args.result.resolve())
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the P1-AMRM0 matched consumed-Development canary.

Both arms consume the exact frozen P1-A4 candidate bbox stream. The baseline is
the sealed A4 continuous-correspondence output. AMRM may only abstain from or
commit the same candidate; it cannot add search, boxes, frames, or prompts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from scripts.research.goal_copilot_bridge.p1_verifier_first.core import (
    CandidateEvidence,
    GoalContract,
    VerifierPolicy,
    initialize_ledger,
    update_ledger,
)
from scripts.research.goal_copilot_bridge.p1_verifier_first.memory import (
    MemoryObservation,
    MemoryPolicy,
    initialize_memory,
    record_observation,
)


PROTOCOL_ID = "P1-AMRM0-MATCHED-DEVELOPMENT-CANARY-V1"
MANIFEST_SCHEMA = "blindassist_p1_amrm0_matched_canary_manifest_v1"
TRACE_SCHEMA = "blindassist_p1_amrm0_matched_canary_trace_v1"
RESULT_SCHEMA = "blindassist_p1_amrm0_matched_canary_result_v1"
CLAIM_CEILING = "CONSUMED_ADT_MATCHED_DEVELOPMENT_SIGNAL_ONLY"

# Inherited unchanged from the consumed P1-A2 fixed-reference Development winner.
# AMRM0 performs no threshold sweep or outcome-conditioned fitting.
DENSE_THRESHOLDS = {
    "anchor_match_fraction": 0.1640625,
    "match_confidence": 0.755523741,
    "spatial_consistency": 0.423076923,
    "anchor_coverage": 0.875,
}
CONTEXT_EXPANSION = 2.5
MEMORY_POLICY = MemoryPolicy(max_verified_entries=12, max_tentative_entries=4)
VERIFIER_POLICY = VerifierPolicy(
    max_hypotheses=4,
    reliable_misses_to_stale=1,
    maximum_unconfirmed_age=6,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_backends():
    module_dir = _repo_root() / "scripts" / "research" / "ba_adt_real_evidence"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    import run_p1_a2_dense_identity as a2
    import run_p1_consumed_adt_baseline as r0
    from materialize_p1_temporal_cohort import SourceSpec, load_source, sha256
    return a2, r0, SourceSpec, load_source, sha256


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(value))
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> str:
    return _file_sha256(Path(__file__))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_repo_root(), text=True
    ).strip()


def _episode_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {episode["episode_id"]: episode for episode in payload["episodes"]}


def _public_episodes(public: dict[str, Any]):
    for source in public["sources"]:
        for episode in source["episodes"]:
            yield source, episode


def _validate_candidate_parity(public: dict[str, Any], baseline: dict[str, Any]) -> dict[str, int]:
    baseline_by_id = _episode_map(baseline)
    public_ids = {episode["episode_id"] for _, episode in _public_episodes(public)}
    if public_ids != set(baseline_by_id):
        raise ValueError("public/A4 episode identity drift")
    frames = 0
    available = 0
    for _, episode in _public_episodes(public):
        predicted = baseline_by_id[episode["episode_id"]]
        candidates = predicted["candidate_bboxes"]
        if len(candidates) != len(episode["frames"]):
            raise ValueError("public/A4 frame count drift")
        for frame, candidate in zip(episode["frames"], candidates):
            if candidate["frame_index"] != frame["frame_index"]:
                raise ValueError("public/A4 frame index drift")
            present = candidate["bbox_xyxy"] is not None
            if present != (candidate["candidate_id"] is not None and candidate["source"] is not None):
                raise ValueError("A4 candidate nullability drift")
            frames += 1
            available += int(present)
    return {"episodes": len(public_ids), "frames": frames, "candidate_available_frames": available}


def prepare_manifest(
    *,
    public_path: Path,
    private_path: Path,
    baseline_path: Path,
    model_dir: Path,
    manifest_path: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    a2, r0, _, _, _ = _load_backends()
    public = r0.validate_public_input(_read_json(public_path))
    baseline = _read_json(baseline_path)
    if baseline.get("schema_version") != r0.PREDICTION_SCHEMA:
        raise ValueError("A4 baseline prediction schema drift")
    if baseline.get("public_input_sha256") != r0.object_sha256(public):
        raise ValueError("A4 baseline/public binding drift")
    counts = _validate_candidate_parity(public, baseline)
    model_identity = a2._validate_model(model_dir)
    if implementation_commit != _git_head():
        raise ValueError("implementation commit must equal current HEAD")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "claim_role": "CONSUMED_DEVELOPMENT_ONLY",
        "implementation": {
            "commit": implementation_commit,
            "source_sha256": _source_sha256(),
        },
        "inputs": {
            "public_path": str(public_path.resolve()),
            "public_file_sha256": _file_sha256(public_path),
            "public_object_sha256": r0.object_sha256(public),
            "private_path": str(private_path.resolve()),
            "private_file_sha256": _file_sha256(private_path),
            "baseline_path": str(baseline_path.resolve()),
            "baseline_file_sha256": _file_sha256(baseline_path),
            "model_dir": str(model_dir.resolve()),
            "model_identity": model_identity,
        },
        "matched_surface": {
            **counts,
            "candidate_source": "FROZEN_P1_A4_ONLINE_TAPIR_BBOX_STREAM",
            "baseline_arm": "SEALED_P1_A4_CONTINUOUS_CORRESPONDENCE_OUTPUT",
            "amrm_arm": "SAME_CANDIDATE_OR_NONE_ONLY",
            "added_candidates": 0,
            "added_frames": 0,
            "global_search": 0,
        },
        "policy": {
            "target_dense_thresholds": DENSE_THRESHOLDS,
            "masked_context_dense_thresholds": DENSE_THRESHOLDS,
            "threshold_source": "INHERITED_UNCHANGED_P1_A2_CONSUMED_DEVELOPMENT_WINNER",
            "threshold_search": 0,
            "context_expansion": CONTEXT_EXPANSION,
            "verified_bank_capacity": MEMORY_POLICY.max_verified_entries,
            "tentative_capacity": MEMORY_POLICY.max_tentative_entries,
            "stale_after_reliable_misses": VERIFIER_POLICY.reliable_misses_to_stale,
            "memory_admission": "ORIGINAL_BINDING_TARGET_AND_MASKED_CONTEXT_BOTH_SUPPORTED",
            "viewpoint_semantics": "PUBLIC_2D_BEARING_CHANGE_PROXY_NOT_PHYSICAL_VIEWPOINT",
        },
        "primary_metrics": [
            "identity_precision",
            "identity_coverage",
            "wrong_instance_reacquisition",
            "false_continuity",
            "true_same_instance_reacquisition",
        ],
        "diagnostics": [
            "honest_abstention",
            "timely_stale",
            "verification_friction",
            "memory_poisoning_events",
            "verified_bank_growth_and_coverage",
            "reacquisition_after_scale_change",
            "reacquisition_after_2d_bearing_proxy_change",
            "memory_contribution_source",
        ],
        "adjudication": {
            "hard_gates": [
                "candidate_availability_exact_parity",
                "memory_poisoning_events_zero",
                "post_initialization_gt_reads_zero",
                "added_candidates_zero",
            ],
            "signal_requires": {
                "identity_precision": "strictly_higher_than_baseline",
                "wrong_identity_commitments": "strictly_lower_than_baseline",
                "true_same_instance_reacquisition": "at_least_one_and_not_below_baseline",
                "identity_coverage_absolute_min": 0.20,
                "identity_coverage_relative_to_baseline_min": 0.25,
                "false_reacquisition": "not_above_baseline",
            },
        },
        "truth_firewall": {
            "run_reads_private_truth": False,
            "evaluate_reads_private_truth_after_prediction": True,
            "post_initialization_gt_reads": 0,
        },
        "forbidden": [
            "VIO", "SLAM", "SCENE_GRAPH", "VLM_VERIFIER", "ACTIVE_TRANSLATION",
            "BAYESIAN_BELIEF", "RETRIEVAL_THRESHOLD_SWEEP", "AMRM_SPECIFIC_DATA_REGENERATION",
        ],
    }
    _write_json(manifest_path, manifest)
    return manifest


def _verify_manifest(manifest: dict[str, Any], *, require_commit: bool) -> tuple[Path, Path, Path, Path]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("manifest schema/protocol drift")
    if manifest["implementation"]["source_sha256"] != _source_sha256():
        raise ValueError("runner source drift")
    if require_commit and manifest["implementation"]["commit"] != _git_head():
        raise ValueError("execution commit drift")
    inputs = manifest["inputs"]
    public_path = Path(inputs["public_path"])
    private_path = Path(inputs["private_path"])
    baseline_path = Path(inputs["baseline_path"])
    model_dir = Path(inputs["model_dir"])
    for path, key in (
        (public_path, "public_file_sha256"),
        (private_path, "private_file_sha256"),
        (baseline_path, "baseline_file_sha256"),
    ):
        if _file_sha256(path) != inputs[key]:
            raise ValueError(f"input hash drift: {path}")
    return public_path, private_path, baseline_path, model_dir


def _expand_bbox(bbox: list[float], width: int, height: int, factor: float) -> list[float]:
    x1, y1, x2, y2 = map(float, bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = (x2 - x1) * factor / 2.0
    half_h = (y2 - y1) * factor / 2.0
    return [max(0.0, cx - half_w), max(0.0, cy - half_h), min(float(width), cx + half_w), min(float(height), cy + half_h)]


def _context_tensor(image, bbox: list[float]):
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    expanded = _expand_bbox(bbox, width, height, CONTEXT_EXPANSION)
    ex1, ey1, ex2, ey2 = [int(round(value)) for value in expanded]
    ex2, ey2 = max(ex1 + 2, ex2), max(ey1 + 2, ey2)
    crop = image[ey1:ey2, ex1:ex2].copy()
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    rx1, ry1 = max(0, x1 - ex1), max(0, y1 - ey1)
    rx2, ry2 = min(crop.shape[1], x2 - ex1), min(crop.shape[0], y2 - ey1)
    if rx2 > rx1 and ry2 > ry1:
        crop[ry1:ry2, rx1:rx2] = (123, 116, 103)
    resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
    std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
    return np.transpose((rgb - mean) / std, (2, 0, 1))


def _read_episode_frames(video_path: Path, episode: dict[str, Any], candidates: list[dict[str, Any]]):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"unable to open RGB video: {video_path}")
    result = []
    try:
        for frame, candidate in zip(episode["frames"], candidates):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame["video_frame_index"]))
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"unable to decode video frame {frame['video_frame_index']}")
            result.append((image, candidate))
    finally:
        capture.release()
    return result


def _supported(consensus: dict[str, float]) -> bool:
    return all(float(consensus[key]) >= threshold for key, threshold in DENSE_THRESHOLDS.items())


def _bbox_proxy(bbox: list[float], initial: list[float]) -> tuple[str, str]:
    x1, y1, x2, y2 = map(float, bbox)
    ix1, iy1, ix2, iy2 = map(float, initial)
    area = max(1.0, (x2 - x1) * (y2 - y1))
    initial_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
    ratio = area / initial_area
    scale = "SMALL" if ratio < 0.75 else "LARGE" if ratio > 1.50 else "MEDIUM"
    center_delta = ((x1 + x2) - (ix1 + ix2)) / 2.0
    initial_width = max(1.0, ix2 - ix1)
    bearing = "LEFT" if center_delta < -initial_width else "RIGHT" if center_delta > initial_width else "FRONTAL"
    return scale, bearing


def _asset_ref(array) -> str:
    return "sha256:" + hashlib.sha256(array.tobytes()).hexdigest()


@dataclass
class EncodedView:
    evidence_id: str
    frame_index: int
    target_patches: Any
    context_patches: Any
    scale_band: str
    viewpoint_bin: str


def _candidate_output_frame(
    frame_index: int,
    candidate_id: str | None,
    *,
    committed: bool,
    previous_state: str,
    frames_since_confirmed: int,
    identity_score: float | None,
    context_score: float | None,
) -> tuple[dict[str, Any], str, int]:
    if committed:
        state = "TRACKING"
        event = "REACQUIRED" if previous_state == "LOST" else "NONE"
        frames_since_confirmed = 0
    else:
        state = "LOST"
        event = "LOSS_DETECTED" if previous_state != "LOST" else "NONE"
        frames_since_confirmed += 1
    return ({
        "frame_index": frame_index,
        "state": state,
        "current_candidate_id": candidate_id if committed else None,
        "identity_score": identity_score,
        "stability_score": context_score,
        "oscillation_score": 0.0 if candidate_id is not None else None,
        "frames_since_confirmed": frames_since_confirmed,
        "event": event,
    }, state, frames_since_confirmed)


def _run_episode(a2, encoder, source: dict[str, Any], episode: dict[str, Any], baseline_episode: dict[str, Any]):
    import numpy as np

    candidates = baseline_episode["candidate_bboxes"]
    decoded = _read_episode_frames(Path(source["rgb_video_path"]), episode, candidates)
    initial_bbox = list(map(float, episode["initial_target_bbox_xyxy"]))
    initial_image = decoded[0][0]
    target_tensors = [a2._crop_tensor(initial_image, initial_bbox)]
    context_tensors = [_context_tensor(initial_image, initial_bbox)]
    candidate_positions = []
    for position, (image, candidate_row) in enumerate(decoded):
        bbox = candidate_row["bbox_xyxy"]
        if bbox is None:
            continue
        candidate_positions.append(position)
        target_tensors.append(a2._crop_tensor(image, bbox))
        context_tensors.append(_context_tensor(image, bbox))
    target_features = encoder.encode(target_tensors)
    context_features = encoder.encode(context_tensors)
    feature_by_position = {
        position: (target_features[offset + 1], context_features[offset + 1])
        for offset, position in enumerate(candidate_positions)
    }
    initial_target, initial_context = target_features[0], context_features[0]

    goal = GoalContract(
        goal_id=episode["handoff"]["goal_id"],
        reference_mode="UNIQUE",
        goal_predicate="already-established consumed ADT referent",
        allowed_rebinding=False,
        arrival_predicate="not evaluated in P1-AMRM0",
        safety_constraints=("NO_UNVERIFIED_TRANSLATION",),
    )
    ledger = initialize_ledger(goal, episode["handoff"]["referent_id"], motion_model="STATIC_WORLD")
    memory = initialize_memory(episode["handoff"]["referent_id"])
    encoded_bank: list[EncodedView] = []
    output_frames = []
    trace_frames = []
    previous_state = "LOST"
    frames_since_confirmed = 0

    for position, (public_frame, candidate_row) in enumerate(zip(episode["frames"], candidates)):
        sequence_index = position + 1
        candidate_id = candidate_row["candidate_id"]
        bbox = candidate_row["bbox_xyxy"]
        target_matches: list[str] = []
        context_matches: list[str] = []
        target_consensus = []
        context_consensus = []
        original_target_supported = False
        original_context_supported = False
        target_score = None
        context_score = None
        admitted = False
        matched_by: list[str] = []

        if bbox is not None:
            current_target, current_context = feature_by_position[position]
            for bank_index, view in enumerate(encoded_bank):
                target = a2.dense_consensus(view.target_patches, current_target)
                context = a2.dense_consensus(view.context_patches, current_context)
                target_consensus.append(target)
                context_consensus.append(context)
                if _supported(target):
                    target_matches.append(view.evidence_id)
                if _supported(context):
                    context_matches.append(view.evidence_id)
                if bank_index == 0:
                    original_target_supported = _supported(target)
                    original_context_supported = _supported(context)

            # Frame 0 is the P0-established binding and seeds the verified bank.
            if position == 0 and not encoded_bank:
                original_target_supported = True
                original_context_supported = True
                target_matches = ["binding-kf"]
                context_matches = ["binding-kf"]
                target_score = 1.0
                context_score = 1.0
            else:
                target_score = max(0.0, min(1.0, max(
                    (float(item["global_embedding_cosine"]) for item in target_consensus),
                    default=0.0,
                )))
                context_score = max(0.0, min(1.0, max(
                    (float(item["global_embedding_cosine"]) for item in context_consensus),
                    default=0.0,
                )))

            target_supported = bool(target_matches)
            context_supported = bool(context_matches)
            evidence_id = f"{episode['episode_id']}-f{position}"
            evidence = CandidateEvidence(
                evidence_id=evidence_id,
                candidate_id=candidate_id,
                entity_hypothesis_id=episode["handoff"]["referent_id"],
                proposal_source="TRACKER",
                candidate_region_xyxy=tuple(map(float, bbox)),
                appearance_support=max(0.0, min(1.0, float(target_score))),
                appearance_contradiction=max(0.0, min(1.0, 1.0 - float(target_score))),
                spatial_prediction="SUPPORTED",
                parent_slot="INSUFFICIENT",
                relational_context="SUPPORTED" if context_supported else "REJECTED",
                distractor_exclusion="SUPPORTED" if target_supported else "REJECTED",
                current_goal_validity="SUPPORTED",
            )
            ledger = update_ledger(
                ledger,
                sequence_index=sequence_index,
                observability_reason="IN_VIEW_RELIABLE",
                candidates=(evidence,),
                policy=VERIFIER_POLICY,
            )
            committed = ledger.decision == "CONFIRMED_VISIBLE"
            if committed:
                if "binding-kf" in target_matches or (encoded_bank and encoded_bank[0].evidence_id in target_matches):
                    matched_by.append("ORIGINAL_BINDING_KF")
                if any(item != (encoded_bank[0].evidence_id if encoded_bank else "binding-kf") for item in target_matches):
                    matched_by.append("NEWLY_ACCUMULATED_VERIFIED_KF")
                if context_matches:
                    matched_by.append("CONTEXT_MEMORY")
                if len(set(target_matches)) >= 2:
                    matched_by.append("MULTIPLE_MEMORY_VIEWS_JOINTLY")

            scale_band, viewpoint_bin = _bbox_proxy(bbox, initial_bbox)
            observation = MemoryObservation(
                evidence_id=evidence_id,
                candidate_id=candidate_id,
                referent_id=episode["handoff"]["referent_id"],
                frame_id=f"{episode['episode_id']}-f{position}",
                target_crop_ref=_asset_ref(
                    target_tensors[0] if position == 0 else target_tensors[candidate_positions.index(position) + 1]
                ),
                context_crop_ref=_asset_ref(
                    context_tensors[0] if position == 0 else context_tensors[candidate_positions.index(position) + 1]
                ),
                full_frame_ref="sha256:" + hashlib.sha256(decoded[position][0].tobytes()).hexdigest(),
                orientation_source="UNKNOWN",
                orientation_yaw_deg=None,
                distance_band={"SMALL": "FAR", "MEDIUM": "MID", "LARGE": "NEAR"}[scale_band],
                viewpoint_bin=viewpoint_bin,
                scale_band=scale_band,
                context_anchor_id="MASKED_LOCAL_CONTEXT_2P5X",
            )
            # Anti-poisoning rule: newly verified keyframes must remain directly
            # anchored to both the original target and original masked context.
            if committed and original_target_supported and original_context_supported:
                previous_verified = {entry.observation.evidence_id for entry in memory.verified}
                memory = record_observation(memory, ledger, observation, sequence_index=sequence_index, policy=MEMORY_POLICY)
                current_verified = {entry.observation.evidence_id for entry in memory.verified}
                admitted = evidence_id in current_verified - previous_verified
                if admitted:
                    encoded_bank.append(EncodedView(
                        evidence_id=evidence_id,
                        frame_index=position,
                        target_patches=initial_target if position == 0 else current_target,
                        context_patches=initial_context if position == 0 else current_context,
                        scale_band=scale_band,
                        viewpoint_bin=viewpoint_bin,
                    ))
                    retained_ids = {entry.observation.evidence_id for entry in memory.verified}
                    encoded_bank = [view for view in encoded_bank if view.evidence_id in retained_ids]
        else:
            ledger = update_ledger(
                ledger,
                sequence_index=sequence_index,
                observability_reason="IN_VIEW_RELIABLE",
                candidates=(),
                policy=VERIFIER_POLICY,
            )
            committed = False
            scale_band = None
            viewpoint_bin = None

        output_frame, previous_state, frames_since_confirmed = _candidate_output_frame(
            position,
            candidate_id,
            committed=committed,
            previous_state=previous_state,
            frames_since_confirmed=frames_since_confirmed,
            identity_score=target_score,
            context_score=context_score,
        )
        output_frames.append(output_frame)
        trace_frames.append({
            "frame_index": position,
            "candidate_available": bbox is not None,
            "candidate_id": candidate_id,
            "committed": committed,
            "decision": ledger.decision,
            "target_match_view_ids": target_matches,
            "context_match_view_ids": context_matches,
            "matched_by": matched_by,
            "verified_admitted": admitted,
            "verified_bank_size": len(memory.verified),
            "verified_coverage_cells": [list(cell) for cell in memory.coverage_cells],
            "scale_band": scale_band,
            "viewpoint_bin": viewpoint_bin,
        })

    prediction_episode = {
        **{key: value for key, value in baseline_episode.items() if key not in {"p1_output"}},
        "p1_output": {
            "schema_version": 1,
            "protocol_id": baseline_episode["p1_output"]["protocol_id"],
            "episode_id": episode["episode_id"],
            "referent_id": episode["handoff"]["referent_id"],
            "score_semantics": "ALGORITHMIC_EVIDENCE_NOT_CALIBRATED_PROBABILITY",
            "frames": output_frames,
        },
    }
    trace_episode = {
        "episode_id": episode["episode_id"],
        "frames": trace_frames,
        "final_verified_bank_size": len(memory.verified),
        "final_coverage_cells": [list(cell) for cell in memory.coverage_cells],
        "memory_receipts": [receipt.__dict__ for receipt in memory.receipts],
    }
    return prediction_episode, trace_episode


def run_public(manifest_path: Path, prediction_path: Path, trace_path: Path, device: str) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    public_path, _, baseline_path, model_dir = _verify_manifest(manifest, require_commit=True)
    a2, r0, _, _, sha256 = _load_backends()
    public = r0.validate_public_input(_read_json(public_path))
    baseline = _read_json(baseline_path)
    counts = _validate_candidate_parity(public, baseline)
    if counts != {key: manifest["matched_surface"][key] for key in counts}:
        raise ValueError("matched surface count drift")
    for source in public["sources"]:
        if sha256(Path(source["rgb_video_path"])) != source["rgb_video_sha256"]:
            raise ValueError("RGB source hash drift")
    encoder = a2.DenseEncoder(model_dir, device=device)
    baseline_by_id = _episode_map(baseline)
    prediction_episodes = []
    trace_episodes = []
    for source, episode in _public_episodes(public):
        prediction_episode, trace_episode = _run_episode(
            a2, encoder, source, episode, baseline_by_id[episode["episode_id"]]
        )
        prediction_episodes.append(prediction_episode)
        trace_episodes.append(trace_episode)
    prediction = {
        **{key: value for key, value in baseline.items() if key != "episodes"},
        "tracker": "P1_AMRM0_SAME_A4_CANDIDATE_VERIFIED_MULTI_VIEW_MEMORY",
        "runtime": {
            "device": device,
            "encoded_crops": encoder.encoded_crops,
            "forward_batches": encoder.forward_batches,
            "candidate_available_frames": counts["candidate_available_frames"],
        },
        "episodes": prediction_episodes,
        "truth_access": {"oracle_initializations": 15, "post_initialization_gt_reads": 0},
    }
    trace = {
        "schema_version": TRACE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": _file_sha256(manifest_path),
        "public_input_sha256": manifest["inputs"]["public_object_sha256"],
        "baseline_prediction_sha256": manifest["inputs"]["baseline_file_sha256"],
        "candidate_availability_parity": counts,
        "threshold_search": 0,
        "post_initialization_gt_reads": 0,
        "episodes": trace_episodes,
    }
    _write_json(prediction_path, prediction)
    _write_json(trace_path, trace)
    return {"prediction": prediction, "trace": trace}


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "value": numerator / denominator if denominator else None}


def adjudicate(baseline: dict[str, Any], amrm: dict[str, Any], hard_gates: dict[str, bool]) -> str:
    contract_gates = {
        key: value for key, value in hard_gates.items() if key != "memory_poisoning_events_zero"
    }
    if not all(contract_gates.values()):
        return "P1_AMRM0_NOT_EVALUABLE_MATCHED_CONTRACT"
    if not hard_gates.get("memory_poisoning_events_zero", False):
        return "P1_AMRM0_MEMORY_POISONING_FAIL"
    baseline_coverage = baseline["identity_coverage"]["value"] or 0.0
    amrm_coverage = amrm["identity_coverage"]["value"] or 0.0
    precision_gain = (amrm["identity_precision"]["value"] or 0.0) > (baseline["identity_precision"]["value"] or 0.0)
    wrong_reduction = amrm["wrong_identity_commitments"] < baseline["wrong_identity_commitments"]
    true_reacquisition = (
        amrm["true_same_instance_reacquisition"] >= 1
        and amrm["true_same_instance_reacquisition"] >= baseline["true_same_instance_reacquisition"]
    )
    coverage_ok = amrm_coverage >= 0.20 and amrm_coverage >= 0.25 * baseline_coverage
    false_reacq_ok = amrm["wrong_instance_reacquisition"] <= baseline["wrong_instance_reacquisition"]
    if precision_gain and wrong_reduction and true_reacquisition and coverage_ok and false_reacq_ok:
        if amrm.get("multiview_contributed_true_reacquisition", 0) >= 1:
            return "P1_AMRM0_IDENTITY_VALUE_SIGNAL_ESTABLISHED"
        return "P1_AMRM0_VERIFIER_ONLY_SIGNAL_NO_MULTIVIEW_VALUE"
    if precision_gain and wrong_reduction and not coverage_ok:
        return "P1_AMRM0_ABSTENTION_ONLY_NO_SIGNAL"
    return "P1_AMRM0_NO_IDENTITY_VALUE_SIGNAL"


def _arm_metrics(result: dict[str, Any], candidate_available: int) -> dict[str, Any]:
    attribution = result["post_outcome_descriptive_failure_attribution"]
    correct = int(attribution["correct_target_asserted_frames"])
    wrong = int(attribution["wrong_background_asserted_frames"]) + int(attribution["wrong_other_adt_instance_asserted_frames"])
    aggregate = result["evaluation"]["aggregate"]
    commitments = correct + wrong
    return {
        "correct_identity_commitments": correct,
        "wrong_identity_commitments": wrong,
        "identity_precision": _rate(correct, commitments),
        "identity_coverage": _rate(commitments, candidate_available),
        "wrong_instance_reacquisition": int(aggregate["false_reacquisitions"]),
        "true_same_instance_reacquisition": int(aggregate["reacquisition_precision"]["numerator"]),
        "false_loss_frames": int(aggregate["false_loss_frames"]),
        "correct_identity_coverage": aggregate["correct_identity_coverage"],
    }


def _private_diagnostics(r0, private_path: Path, prediction: dict[str, Any], trace: dict[str, Any]):
    _, _, SourceSpec, load_source, sha256 = _load_backends()
    private = _read_json(private_path)
    predicted_by_id = _episode_map(prediction)
    trace_by_id = _episode_map(trace)
    false_continuity = 0
    honest_abstention = 0
    timely_stale_opportunities = 0
    timely_stale_successes = 0
    poisoning_events = 0
    friction: list[int] = []
    contribution = Counter()
    scale_change_opportunities = 0
    scale_change_successes = 0
    bearing_proxy_opportunities = 0
    bearing_proxy_successes = 0

    for source_row in private["sources"]:
        gt_path = Path(source_row["groundtruth_path"])
        if sha256(gt_path) != source_row["groundtruth_sha256"]:
            raise ValueError("private GT hash drift")
        source = load_source(
            SourceSpec(source_row["source_sequence_id"], gt_path, Path(source_row["rgb_video_path"])),
            probe_video=False,
        )
        for episode_row in source_row["episodes"]:
            raw_episode = _read_json(Path(episode_row["episode_path"]))
            episode = {**raw_episode, "episode_id": episode_row["public_episode_id"]}
            predicted = predicted_by_id[episode["episode_id"]]
            traced = trace_by_id[episode["episode_id"]]
            evaluator_episode = r0._build_evaluator_episode(source, episode, predicted)
            referent = evaluator_episode["truth"]["referent_instance_id"]
            attempts_since_loss = 0
            lost = False
            prior_correct_candidate = False
            initial_bbox = predicted["candidate_bboxes"][0]["bbox_xyxy"] or raw_episode["frames"][0]["target_bbox_xyxy"]
            episode_scale_opportunity = False
            episode_scale_success = False
            episode_bearing_opportunity = False
            episode_bearing_success = False
            for index, (truth_frame, output_frame, bbox_row, trace_frame) in enumerate(zip(
                evaluator_episode["truth"]["frames"],
                predicted["p1_output"]["frames"],
                predicted["candidate_bboxes"],
                traced["frames"],
            )):
                candidate_id = bbox_row["candidate_id"]
                candidate_instance = truth_frame["candidate_instance_map"].get(candidate_id) if candidate_id else None
                correct_candidate = candidate_instance == referent
                wrong_candidate = candidate_instance is not None and not correct_candidate
                committed = output_frame["current_candidate_id"] is not None
                if wrong_candidate and committed and index > 0 and predicted["p1_output"]["frames"][index - 1]["state"] == "TRACKING":
                    false_continuity += 1
                if wrong_candidate and not committed:
                    honest_abstention += 1
                onset = prior_correct_candidate and not correct_candidate
                if onset:
                    timely_stale_opportunities += 1
                    timely_stale_successes += int(not committed)
                prior_correct_candidate = correct_candidate
                if output_frame["event"] == "LOSS_DETECTED":
                    lost = True
                    attempts_since_loss = 0
                if lost and candidate_id is not None:
                    attempts_since_loss += 1
                if output_frame["event"] == "REACQUIRED" and correct_candidate:
                    friction.append(attempts_since_loss)
                    lost = False
                    for item in trace_frame["matched_by"]:
                        contribution[item] += 1
                    bbox = bbox_row["bbox_xyxy"]
                    if bbox is not None and initial_bbox is not None:
                        scale, bearing = _bbox_proxy(bbox, initial_bbox)
                        if scale != "MEDIUM":
                            episode_scale_success = True
                        if bearing != "FRONTAL":
                            episode_bearing_success = True
                if truth_frame["phase"] == "REACQUISITION_WINDOW" and correct_candidate:
                    bbox = bbox_row["bbox_xyxy"]
                    if bbox is not None and initial_bbox is not None:
                        scale, bearing = _bbox_proxy(bbox, initial_bbox)
                        episode_scale_opportunity |= scale != "MEDIUM"
                        episode_bearing_opportunity |= bearing != "FRONTAL"
                if trace_frame["verified_admitted"] and not correct_candidate:
                    poisoning_events += 1
            scale_change_opportunities += int(episode_scale_opportunity)
            scale_change_successes += int(episode_scale_opportunity and episode_scale_success)
            bearing_proxy_opportunities += int(episode_bearing_opportunity)
            bearing_proxy_successes += int(episode_bearing_opportunity and episode_bearing_success)

    final_sizes = [int(item["final_verified_bank_size"]) for item in trace["episodes"]]
    coverage_cells = [len(item["final_coverage_cells"]) for item in trace["episodes"]]
    return {
        "false_continuity": false_continuity,
        "honest_abstention_on_wrong_candidate": honest_abstention,
        "timely_stale": _rate(timely_stale_successes, timely_stale_opportunities),
        "verification_friction_attempts_per_true_reacquisition": {
            "count": len(friction),
            "median": statistics.median(friction) if friction else None,
            "values": friction,
        },
        "memory_poisoning_events": poisoning_events,
        "verified_bank_growth": {
            "final_total": sum(final_sizes),
            "per_episode": final_sizes,
            "median": statistics.median(final_sizes) if final_sizes else None,
        },
        "verified_bank_coverage_cells": {
            "final_total": sum(coverage_cells),
            "per_episode": coverage_cells,
        },
        "memory_contribution_on_true_reacquisition": dict(sorted(contribution.items())),
        "reacquisition_after_scale_change": _rate(scale_change_successes, scale_change_opportunities),
        "reacquisition_after_2d_bearing_proxy_change": _rate(bearing_proxy_successes, bearing_proxy_opportunities),
        "physical_viewpoint_reacquisition": "NOT_EVALUABLE_MISSING_VIEWPOINT_TRUTH",
    }


def evaluate(
    manifest_path: Path,
    prediction_path: Path,
    trace_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    _, private_path, baseline_path, _ = _verify_manifest(manifest, require_commit=False)
    a2, r0, _, _, _ = _load_backends()
    del a2
    if _file_sha256(prediction_path) == "" or _file_sha256(trace_path) == "":
        raise ValueError("missing prediction/trace")
    baseline_prediction = _read_json(baseline_path)
    amrm_prediction = _read_json(prediction_path)
    trace = _read_json(trace_path)
    baseline_eval_path = output_dir / "baseline_evaluation.json"
    amrm_eval_path = output_dir / "amrm_evaluation.json"
    baseline_result = r0.evaluate_predictions(private_path, baseline_path, baseline_eval_path)
    amrm_result = r0.evaluate_predictions(private_path, prediction_path, amrm_eval_path)
    candidate_available = int(manifest["matched_surface"]["candidate_available_frames"])
    baseline_metrics = _arm_metrics(baseline_result, candidate_available)
    amrm_metrics = _arm_metrics(amrm_result, candidate_available)
    diagnostics = _private_diagnostics(r0, private_path, amrm_prediction, trace)
    baseline_diagnostics = _private_diagnostics(
        r0,
        private_path,
        baseline_prediction,
        {
            "episodes": [
                {
                    "episode_id": episode["episode_id"],
                    "frames": [
                        {"matched_by": [], "verified_admitted": False}
                        for _ in episode["p1_output"]["frames"]
                    ],
                    "final_verified_bank_size": 0,
                    "final_coverage_cells": [],
                }
                for episode in baseline_prediction["episodes"]
            ]
        },
    )
    baseline_metrics.update({
        "false_continuity": baseline_diagnostics["false_continuity"],
        "honest_abstention_on_wrong_candidate": baseline_diagnostics["honest_abstention_on_wrong_candidate"],
    })
    amrm_metrics.update({
        "false_continuity": diagnostics["false_continuity"],
        "honest_abstention_on_wrong_candidate": diagnostics["honest_abstention_on_wrong_candidate"],
        "multiview_contributed_true_reacquisition": diagnostics[
            "memory_contribution_on_true_reacquisition"
        ].get("NEWLY_ACCUMULATED_VERIFIED_KF", 0),
    })
    parity = _validate_candidate_parity(_read_json(Path(manifest["inputs"]["public_path"])), amrm_prediction)
    hard_gates = {
        "candidate_availability_exact_parity": parity == {
            key: manifest["matched_surface"][key] for key in parity
        },
        "memory_poisoning_events_zero": diagnostics["memory_poisoning_events"] == 0,
        "post_initialization_gt_reads_zero": trace["post_initialization_gt_reads"] == 0,
        "added_candidates_zero": manifest["matched_surface"]["added_candidates"] == 0,
    }
    terminal = adjudicate(baseline_metrics, amrm_metrics, hard_gates)
    result = {
        "schema_version": RESULT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": terminal,
        "manifest_sha256": _file_sha256(manifest_path),
        "prediction_sha256": _file_sha256(prediction_path),
        "trace_sha256": _file_sha256(trace_path),
        "matched_surface": manifest["matched_surface"],
        "hard_gates": hard_gates,
        "risk_coverage": {
            "baseline": baseline_metrics,
            "amrm": amrm_metrics,
        },
        "amrm_diagnostics": diagnostics,
        "interpretation_rule": "IDENTITY_VALUE_REQUIRES_PRECISION_PLUS_NONTRIVIAL_COVERAGE_AND_TRUE_REACQUISITION",
        "physical_viewpoint_limit": "NO_PHYSICAL_VIEWPOINT_TRUTH_IN_P1_D0_PUBLIC_OR_PRIVATE_ADAPTER",
    }
    _write_json(output_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--public-input", type=Path, required=True)
    prepare.add_argument("--private-input", type=Path, required=True)
    prepare.add_argument("--baseline-prediction", type=Path, required=True)
    prepare.add_argument("--model-dir", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--implementation-commit", required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--prediction", type=Path, required=True)
    run.add_argument("--trace", type=Path, required=True)
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument("--prediction", type=Path, required=True)
    evaluate_parser.add_argument("--trace", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_manifest(
            public_path=args.public_input,
            private_path=args.private_input,
            baseline_path=args.baseline_prediction,
            model_dir=args.model_dir,
            manifest_path=args.manifest,
            implementation_commit=args.implementation_commit,
        )
        print(json.dumps({"manifest": str(args.manifest), "matched_surface": result["matched_surface"]}, indent=2))
    elif args.command == "run":
        result = run_public(args.manifest, args.prediction, args.trace, args.device)
        print(json.dumps({
            "episodes": len(result["prediction"]["episodes"]),
            "runtime": result["prediction"]["runtime"],
            "truth_access": result["prediction"]["truth_access"],
        }, indent=2))
    else:
        result = evaluate(args.manifest, args.prediction, args.trace, args.output_dir)
        print(json.dumps({
            "terminal": result["terminal"],
            "risk_coverage": result["risk_coverage"],
            "hard_gates": result["hard_gates"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

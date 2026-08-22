#!/usr/bin/env python3
"""Read-only first-poison autopsy for the sealed P1-AMRM0 canary.

This script does not rerun either arm or change the frozen terminal. It joins the
sealed public prediction/trace to private Development truth and retains only the
first wrong verified-bank admission in each episode.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p1_verifier_first import run_matched_canary as canary


SCHEMA = "blindassist_p1_amrm0_first_poison_autopsy_v1"
EXPECTED_TERMINAL = "P1_AMRM0_MEMORY_POISONING_FAIL"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_bbox(source, physical_target_id: str, timestamp_ns: int) -> list[float] | None:
    uid = physical_target_id.removeprefix("adt:")
    row = source.boxes.get(uid, {}).get(timestamp_ns)
    if row is None:
        return None
    return [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]


def build_autopsy(
    *,
    result_path: Path,
    prediction_path: Path,
    trace_path: Path,
    private_path: Path,
) -> dict[str, Any]:
    result = _read(result_path)
    prediction = _read(prediction_path)
    trace = _read(trace_path)
    private = _read(private_path)
    if result.get("terminal") != EXPECTED_TERMINAL:
        raise ValueError("autopsy requires the sealed memory-poisoning terminal")
    if result.get("prediction_sha256") != _sha256(prediction_path):
        raise ValueError("prediction hash does not match sealed result")
    if result.get("trace_sha256") != _sha256(trace_path):
        raise ValueError("trace hash does not match sealed result")

    _, r0, SourceSpec, load_source, source_sha256 = canary._load_backends()
    predicted_by_id = canary._episode_map(prediction)
    trace_by_id = canary._episode_map(trace)
    rows: list[dict[str, Any]] = []
    poisoned_admissions = 0
    poisoned_episode_ids: set[str] = set()

    for source_row in private["sources"]:
        gt_path = Path(source_row["groundtruth_path"])
        if source_sha256(gt_path) != source_row["groundtruth_sha256"]:
            raise ValueError("private GT hash drift")
        source = load_source(
            SourceSpec(source_row["source_sequence_id"], gt_path, Path(source_row["rgb_video_path"])),
            probe_video=False,
        )
        for private_episode in source_row["episodes"]:
            raw = _read(Path(private_episode["episode_path"]))
            episode_id = private_episode["public_episode_id"]
            episode = {**raw, "episode_id": episode_id}
            predicted = predicted_by_id[episode_id]
            traced = trace_by_id[episode_id]
            evaluator = r0._build_evaluator_episode(source, episode, predicted)
            referent = evaluator["truth"]["referent_instance_id"]
            first_row = None
            for index, (raw_frame, truth_frame, candidate, output, trace_frame) in enumerate(zip(
                raw["frames"],
                evaluator["truth"]["frames"],
                predicted["candidate_bboxes"],
                predicted["p1_output"]["frames"],
                traced["frames"],
            )):
                candidate_id = candidate["candidate_id"]
                instance = truth_frame["candidate_instance_map"].get(candidate_id) if candidate_id else None
                correct = instance == referent
                poisoned = bool(trace_frame["verified_admitted"] and not correct)
                if not poisoned:
                    continue
                poisoned_admissions += 1
                poisoned_episode_ids.add(episode_id)
                if first_row is not None:
                    continue
                candidate_bbox = candidate["bbox_xyxy"]
                target_bbox = _target_bbox(source, raw["physical_target_id"], int(raw_frame["timestamp_ns"]))
                target_iou = None
                if candidate_bbox is not None and target_bbox is not None:
                    target_iou = r0.iou(candidate_bbox, target_bbox)
                matched_instance, best_object_iou = r0._match_bbox(
                    source, int(raw_frame["timestamp_ns"]), candidate_bbox, episode_id
                )
                candidate_class = "BACKGROUND" if instance.startswith("background:") else "OTHER_ADT_INSTANCE"
                binding_evidence_id = f"{episode_id}-f0"
                first_row = {
                    "episode_id": episode_id,
                    "scenario_class": evaluator["scenario_class"],
                    "frame_index": index,
                    "timestamp_ms": int(raw_frame["timestamp_ns"]) // 1_000_000,
                    "phase": truth_frame["phase"],
                    "target_observable": bool(truth_frame["referent_observable"]),
                    "candidate_id": candidate_id,
                    "candidate_instance": instance,
                    "candidate_class": candidate_class,
                    "candidate_best_object_match": matched_instance,
                    "candidate_best_object_iou": best_object_iou,
                    "candidate_target_iou": target_iou,
                    "candidate_pool_size": 1,
                    "correct_candidate_count_in_frozen_pool": 0,
                    "identity_score": output["identity_score"],
                    "context_score": output["stability_score"],
                    "authority_path": {
                        "tracker_spatial_prediction": "HARDCODED_SUPPORTED",
                        "original_target_dense_match": binding_evidence_id in trace_frame["target_match_view_ids"],
                        "original_context_dense_match": binding_evidence_id in trace_frame["context_match_view_ids"],
                        "bearing_used": False,
                        "local_alternative_contrast_used": False,
                        "negative_veto_used": False,
                    },
                    "causal_bucket": "CORRECT_CANDIDATE_ABSENT_FROM_FROZEN_SINGLE_CANDIDATE_POOL",
                }
            if first_row is not None:
                rows.append(first_row)

    if poisoned_admissions != int(result["amrm_diagnostics"]["memory_poisoning_events"]):
        raise ValueError("poisoning-event count does not reproduce sealed result")

    candidate_classes = Counter(row["candidate_class"] for row in rows)
    scenarios = Counter(row["scenario_class"] for row in rows)
    observable = sum(row["target_observable"] for row in rows)
    target_gate = sum(row["authority_path"]["original_target_dense_match"] for row in rows)
    context_gate = sum(row["authority_path"]["original_context_dense_match"] for row in rows)
    return {
        "schema_version": SCHEMA,
        "role": "POST_OUTCOME_READ_ONLY_FAILURE_AUTOPSY",
        "frozen_terminal_unchanged": EXPECTED_TERMINAL,
        "inputs": {
            "result_sha256": _sha256(result_path),
            "prediction_sha256": _sha256(prediction_path),
            "trace_sha256": _sha256(trace_path),
            "private_input_sha256": _sha256(private_path),
        },
        "counts": {
            "poisoned_admissions": poisoned_admissions,
            "episodes_with_poisoning": len(poisoned_episode_ids),
            "first_poison_events": len(rows),
            "correct_candidate_absent": len(rows),
            "correct_candidate_present_but_lost": 0,
            "multiple_plausible_or_ambiguous": "NOT_EVALUABLE_SINGLE_CANDIDATE_STREAM",
            "explicit_authority_contradiction_bug": 0,
            "target_observable_but_correct_candidate_absent": observable,
            "target_unobservable_and_correct_candidate_absent": len(rows) - observable,
        },
        "first_poison_candidate_class": dict(sorted(candidate_classes.items())),
        "first_poison_scenario_class": dict(sorted(scenarios.items())),
        "authority_path_nonexclusive": {
            "original_target_dense_match": target_gate,
            "original_context_dense_match": context_gate,
            "tracker_spatial_hardcoded_supported": len(rows),
            "bearing": 0,
            "local_alternative_contrast": 0,
            "negative_veto": 0,
            "fallback_or_precedence": 0,
            "exclusive_causal_attribution": "NOT_IDENTIFIABLE_CONJUNCTIVE_AUTHORITY_NO_ABLATION",
        },
        "counterfactual_guardrail": {
            "observed_true_reacquisitions_attributed_to_new_verified_keyframes": 0,
            "claim": "OBSERVED_CONTRIBUTION_ONLY_NOT_A_COUNTERFACTUAL_RERUN",
        },
        "interface_limit": (
            "P1-A4 exposes exactly one candidate per frame; simultaneous correct and wrong alternatives, "
            "discriminative margin, and AMBIGUOUS selection are not evaluable in this cohort."
        ),
        "first_poison_events": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--private-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    autopsy = build_autopsy(
        result_path=args.result,
        prediction_path=args.prediction,
        trace_path=args.trace,
        private_path=args.private_input,
    )
    canary._write_json(args.output, autopsy)
    print(json.dumps({
        "output": str(args.output),
        "counts": autopsy["counts"],
        "first_poison_candidate_class": autopsy["first_poison_candidate_class"],
        "authority_path_nonexclusive": autopsy["authority_path_nonexclusive"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

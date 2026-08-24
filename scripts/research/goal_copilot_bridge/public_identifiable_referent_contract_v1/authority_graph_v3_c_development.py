"""Run SAGE-R V3-C on the consumed V3-A natural-photo cohort.

This is representation-development diagnostics only.  It reuses the sealed
OCR rows and pixels, adds predeclared carrier/cue proposals, and compares V2,
frozen V3-A, V3-C without authority typing, and V3-C full.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import torch

from .authority_typed_sign_destination_graph_v3_c import (
    SCHEMA_VERSION as MODEL_SCHEMA_VERSION,
    SignCarrier,
    _atomic_json,
    _sha256_file,
    _utc_now,
    save_checkpoint,
    score_candidates,
    train_relation_model,
)
from .natural_photo_v3_source_disjoint import (
    _metrics,
    _transform_box,
    _verify_manifest,
    _view_homography,
)
from .semantic_anchor_graph_and_belief_v2 import Box, Candidate, Frame, ReferentBelief, TargetGraph, Token


EXPERIMENT_ID = "SAGE_R_V3_C_AUTHORITY_GRAPH_CONSUMED_NATURAL_DIAGNOSTIC"
SCHEMA_VERSION = "blindassist_sage_r_v3_c_consumed_natural_diagnostic_v1"


def _frame(value: Mapping[str, Any]) -> Frame:
    def box(item: Mapping[str, Any]) -> Box:
        return Box(float(item["x0"]), float(item["y0"]), float(item["x1"]), float(item["y1"]))
    return Frame(
        str(value["episode_id"]), int(value["frame_index"]), str(value["viewpoint"]),
        tuple(Candidate(str(item["candidate_id"]), box(item["box"])) for item in value["candidates"]),
        tuple(Token(str(item["text"]), box(item["box"]), float(item["confidence"])) for item in value["tokens"]),
        float(value["blur"]), float(value["perspective"]), str(value["truth"]),
        str(value["expected_state"]), str(value["note"]),
    )


def _transformed_inputs(
    annotation: Mapping[str, Any], source: Mapping[str, Any], view: Mapping[str, Any]
) -> tuple[tuple[SignCarrier, ...], list[dict[str, Any]]]:
    homography = _view_homography(int(source["width"]), int(source["height"]), view)
    carriers = tuple(
        SignCarrier(
            str(item["carrier_id"]),
            Box(*_transform_box(item["box"], int(source["width"]), int(source["height"]), homography)),
        )
        for item in annotation["carriers"]
    )
    cues = [
        {
            "cue_id": str(item["cue_id"]),
            "carrier_id": str(item["carrier_id"]),
            "box": _transform_box(item["box"], int(source["width"]), int(source["height"]), homography),
        }
        for item in annotation.get("cues", [])
    ]
    return carriers, cues


def run(
    *, cohort_dir: Path, prior_raw_path: Path, annotation_path: Path, run_dir: Path,
    seed: int, examples_per_type: int, epochs: int,
) -> dict[str, Any]:
    if run_dir.exists():
        raise ValueError(f"refusing to overwrite run: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest, seal = _verify_manifest(cohort_dir)
    prior = json.loads(prior_raw_path.read_text(encoding="utf-8"))
    annotations_value = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations = {item["source_id"]: item for item in annotations_value["sources"]}
    sources = {item["source_id"]: item for item in manifest["sources"]}
    observations = {(item["query_id"], int(item["frame_index"])): item for item in manifest["observations"]}
    if set(annotations) != set(sources):
        raise ValueError("carrier annotation source set mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, training = train_relation_model(
        seed=seed, examples_per_type=examples_per_type, epochs=epochs, device=device
    )
    checkpoint = save_checkpoint(run_dir / "authority-relation-v3-c.pt", model, training)
    rows = []
    permutation_errors = {"v3_c_no_authority_typing": [], "v3_c_full": []}
    query_ids = list(dict.fromkeys(str(row["query_id"]) for row in prior["rows"]))
    for query_id in query_ids:
        beliefs = {
            "v3_c_no_authority_typing": ReferentBelief(("A", "B", "C")),
            "v3_c_full": ReferentBelief(("A", "B", "C")),
        }
        for prior_row in [row for row in prior["rows"] if row["query_id"] == query_id]:
            frame = _frame(prior_row["frame"])
            observation = observations[(query_id, frame.frame_index)]
            source = sources[str(prior_row["source_id"])]
            carriers, cues = _transformed_inputs(annotations[str(prior_row["source_id"])], source, observation["view"])
            image = cv2.imread(str(observation["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unreadable observation: {observation['image_path']}")
            target = TargetGraph(tuple(str(value) for value in prior_row["target_tokens"]))
            scores = {}
            diagnostics = {}
            for arm, typing in (("v3_c_no_authority_typing", False), ("v3_c_full", True)):
                arm_scores, arm_diagnostics = score_candidates(
                    model, target, frame, image, carriers, cues,
                    authority_typing=typing, device=device,
                )
                scores[arm], diagnostics[arm] = arm_scores, arm_diagnostics
                reversed_frame = Frame(
                    frame.episode_id, frame.frame_index, frame.viewpoint,
                    tuple(reversed(frame.candidates)), frame.tokens, frame.blur,
                    frame.perspective, frame.truth, frame.expected_state, frame.note,
                )
                permuted, _ = score_candidates(
                    model, target, reversed_frame, image, carriers, cues,
                    authority_typing=typing, device=device,
                )
                permutation_errors[arm].append(max(abs(arm_scores[key]["score"] - permuted[key]["score"]) for key in arm_scores))
            rows.append({
                "query_id": query_id, "source_id": prior_row["source_id"],
                "frame_index": frame.frame_index, "viewpoint": frame.viewpoint,
                "target_tokens": list(target.tokens), "truth": frame.truth,
                "expected_state": frame.expected_state, "scenario": prior_row["scenario"],
                "note": frame.note, "image_path": observation["image_path"],
                "image_sha256": observation["image_sha256"], "frame": asdict(frame),
                "v2_heuristic": prior_row["v2_heuristic"], "v3_a_full": prior_row["v3_full"],
                "v3_c_no_authority_typing": beliefs["v3_c_no_authority_typing"].update(frame, scores["v3_c_no_authority_typing"]),
                "v3_c_full": beliefs["v3_c_full"].update(frame, scores["v3_c_full"]),
                "v3_c_diagnostics": diagnostics,
            })
    metrics = {arm: _metrics(rows, arm) for arm in ("v2_heuristic", "v3_a_full", "v3_c_no_authority_typing", "v3_c_full")}
    for arm, errors in permutation_errors.items():
        metrics[arm]["candidate_permutation_max_abs_error"] = max(errors, default=0.0)
        metrics[arm]["candidate_permutation_invariant_observations"] = sum(value <= 1e-6 for value in errors)
    raw_path = run_dir / "raw-decisions.json"
    _atomic_json(raw_path, {"schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID, "rows": rows})
    full = metrics["v3_c_full"]
    v2 = metrics["v2_heuristic"]
    ablation = metrics["v3_c_no_authority_typing"]
    decision = {
        "v3_c_full_vs_v2_correct_delta": full["correct_terminal_observations"] - v2["correct_terminal_observations"],
        "v3_c_full_vs_v2_wrong_lock_delta": full["wrong_locks"] - v2["wrong_locks"],
        "authority_typing_wrong_lock_delta": full["wrong_locks"] - ablation["wrong_locks"],
        "development_effect_present": full["correct_terminal_observations"] > v2["correct_terminal_observations"] and full["wrong_locks"] <= v2["wrong_locks"],
        "next": "FREEZE_NEW_SOURCE_DISJOINT_COHORT" if full["correct_terminal_observations"] > v2["correct_terminal_observations"] and full["wrong_locks"] <= v2["wrong_locks"] else "REVISE_OR_CLOSE_V3_C_BEFORE_FRESH_COHORT",
    }
    report = {
        "schema_version": SCHEMA_VERSION, "model_schema_version": MODEL_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID, "created_at_utc": _utc_now(),
        "data_role": "CONSUMED_V3_A_NATURAL_COHORT_REPRESENTATION_DEVELOPMENT_DIAGNOSTIC",
        "inputs": {
            "cohort_manifest_sha256": seal["manifest_sha256"],
            "prior_raw_path": str(prior_raw_path.resolve()), "prior_raw_sha256": _sha256_file(prior_raw_path),
            "annotation_path": str(annotation_path.resolve()), "annotation_sha256": _sha256_file(annotation_path),
            "checkpoint": checkpoint,
        },
        "architecture": {
            "node_types": ["TargetToken", "ObservedOCRToken", "SignCarrier", "DirectionCue", "PhysicalCandidate"],
            "relations": ["TOKEN_BELONGS_TO_SIGN", "SIGN_LABELS_CANDIDATE", "SIGN_POINTS_TO_REGION", "SIGN_LISTS_REFERENT", "SIGN_NEAR_CANDIDATE", "UNRELATED"],
            "identity_authority": ["SIGN_LABELS_CANDIDATE", "SIGN_POINTS_TO_REGION"],
            "losses": ["relation_type", "decisive_completeness"],
        },
        "training": training, "metrics": metrics, "decision": decision,
        "interpretation_boundary": [
            "The nine sources and OCR outcomes were already consumed by V3-A and are used only to develop the V3-C representation.",
            "Carrier, cue and physical-candidate boxes are frozen input proposals; this run does not evaluate proposal detection.",
            "Direction orientation is decoded from cue pixels; relation type and decisive completeness come from a synthetic multi-task classifier.",
            "V2 belief and terminal thresholds are reused unchanged. V3-B, V4, Android, navigation, safety and product behavior do not run.",
            "A positive diagnostic only authorizes freezing V3-C and acquiring a new source-disjoint Development cohort.",
        ],
        "raw_decisions_sha256": _sha256_file(raw_path),
    }
    _atomic_json(run_dir / "final-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--prior-raw", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=403)
    parser.add_argument("--examples-per-type", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(
        cohort_dir=args.cohort_dir.resolve(), prior_raw_path=args.prior_raw.resolve(),
        annotation_path=args.annotations.resolve(), run_dir=args.run_dir.resolve(),
        seed=args.seed, examples_per_type=args.examples_per_type, epochs=args.epochs,
    )
    print(json.dumps({"metrics": report["metrics"], "decision": report["decision"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

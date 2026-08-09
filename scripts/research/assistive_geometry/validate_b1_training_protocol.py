#!/usr/bin/env python3
"""Fail-closed validator for the Assistive Geometry B1 training protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_assistive_geometry_b1_training_protocol_v1"
EXPECTED_ARMS = (
    "A0_DEPTH_ONLY",
    "A1_PLUS_GROUND",
    "A2_PLUS_CLEARANCE",
    "A3_PLUS_FALSE_CLEAR",
    "A4_PLUS_CONFIDENCE",
)
EXPECTED_TENSORS = {
    "dense_depth_m": [1, 608, 448],
    "depth_valid": [1, 608, 448],
    "ground_probability": [1, 608, 448],
    "ground_valid": [1, 608, 448],
    "clearance_m": [3],
    "clearance_valid": [3],
    "occupancy_probability": [3, 3],
    "occupancy_valid": [3, 3],
    "band_task_confidence": [3],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate(protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(protocol.get("schema") == SCHEMA, "schema drift")
    check(protocol.get("status") == "B1_PROTOCOL_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED", "status drift")
    authority = protocol.get("authority", {})
    check(authority.get("train_target_materialization") is True, "TRAIN target materialization must be authorized")
    check(authority.get("model_implementation_and_shape_smoke") is True, "model implementation smoke must be authorized")
    check(authority.get("formal_student_training") is False, "formal training must wait for implementation lock")
    check(authority.get("development_outcome_access") is False, "Development outcome access must remain closed")
    check(authority.get("confirmation_payload_or_outcome_access") is False, "Confirmation must remain sealed")

    roles = protocol.get("data_roles", {})
    expected_counts = {"TRAIN": 16, "DEVELOPMENT_CALIBRATION": 4, "DEVELOPMENT_SELECTION": 4, "CONFIRMATION": 8}
    identities: list[tuple[str, str]] = []
    for role, expected_count in expected_counts.items():
        rows = roles.get(role, [])
        check(len(rows) == expected_count, f"{role} count drift")
        for row in rows:
            check(set(row) == {"visit_id", "video_id"}, f"{role} identity schema drift")
            identities.append((str(row.get("visit_id")), str(row.get("video_id"))))
    check(len(identities) == len(set(identities)), "data role identity overlap")
    check(roles.get("consumed_120_frame_cohort_forbidden") is True, "consumed cohort firewall missing")
    check(roles.get("depthart_r2_roster_forbidden") is True, "DepthART R2 firewall missing")

    tensors = protocol.get("target_schema", {}).get("tensors", {})
    for name, shape in EXPECTED_TENSORS.items():
        check(tensors.get(name, {}).get("shape_per_sample") == shape, f"target shape drift: {name}")
    check(protocol.get("target_schema", {}).get("unknown_is_negative") is False, "UNKNOWN cannot become negative")
    check(protocol.get("target_schema", {}).get("truth_reader_hash_bound") is True, "truth reader must be hash-bound")

    model = protocol.get("model", {})
    check(model.get("input_nchw") == [1, 3, 608, 448], "model input shape drift")
    check(model.get("dynamic_k_shape") == [1, 3, 3], "dynamic K shape drift")
    check(model.get("encoder") == "DepthART-S metric indoor initialization", "encoder drift")
    check(model.get("shared_decoder_feature") == {"channels": 48, "stride": 4}, "shared decoder feature drift")
    check(model.get("band_pooling") == "fixed thirds mean pool over stride-4 feature", "band pooling drift")
    check(model.get("deployment_operator_budget") == ["Conv", "Relu", "Resize", "ReduceMean", "MatMul", "Add", "Mul", "Sigmoid", "Softplus"], "operator budget drift")

    arms = protocol.get("ablation_arms", [])
    check(tuple(arm.get("id") for arm in arms) == EXPECTED_ARMS, "ablation order drift")
    previous: set[str] = set()
    for arm in arms:
        losses = set(arm.get("active_losses", []))
        check(previous.issubset(losses), f"arm is not additive: {arm.get('id')}")
        previous = losses

    losses = protocol.get("losses", {})
    expected_weights = {
        "masked_log_depth": 1.0,
        "valid_neighbor_log_gradient": 0.5,
        "ground_bce": 0.5,
        "ground_plane_depth": 0.25,
        "clearance_huber": 1.0,
        "occupancy_bce": 1.0,
        "false_clear_extra": 2.0,
        "confidence_bce": 0.5,
    }
    check({name: value.get("lambda") for name, value in losses.items()} == expected_weights, "loss lambda drift")
    check(losses.get("clearance_huber", {}).get("delta_m") == 0.25, "clearance Huber delta drift")
    check(losses.get("false_clear_extra", {}).get("effective_total_positive_weight") == 3.0, "false-clear effective weight drift")
    check(protocol.get("near_field_weighting") == {"0.25_to_2m": 3.0, "2_to_5m": 2.0, "5_to_6m": 1.0}, "near-field weights drift")

    confidence = protocol.get("confidence", {})
    check(confidence.get("primary_shape") == [3], "confidence primary shape drift")
    check(confidence.get("interface_expansion") == "repeat each band confidence across three horizons", "confidence interface expansion drift")
    check(confidence.get("deterministic_invalid_override") == "FORBIDDEN", "confidence cannot override deterministic invalid")
    check(confidence.get("temperature_fit_role") == "DEVELOPMENT_CALIBRATION", "temperature role drift")
    check(confidence.get("threshold_fit_role") == "DEVELOPMENT_CALIBRATION", "threshold role drift")
    grid = confidence.get("threshold_grid", [])
    check(grid == [round(0.50 + 0.05 * index, 2) for index in range(10)], "confidence threshold grid drift")
    check(confidence.get("checkpoint_selection_role") == "DEVELOPMENT_SELECTION", "checkpoint selection role drift")

    training = protocol.get("training", {})
    check(training.get("seeds") == [17, 29, 43], "seed lock drift")
    check(training.get("epochs") == 20, "epoch lock drift")
    check(training.get("micro_batch_size") == 4 and training.get("gradient_accumulation_steps") == 4, "batch lock drift")
    check(training.get("optimizer") == "AdamW", "optimizer drift")
    check(training.get("encoder_learning_rate") == 2e-5, "encoder learning rate drift")
    check(training.get("head_learning_rate") == 1e-4, "head learning rate drift")
    check(training.get("weight_decay") == 0.01, "weight decay drift")
    check(training.get("warmup_optimizer_steps") == 300, "warmup drift")
    check(training.get("schedule") == "cosine_to_0.05x", "schedule drift")
    check(training.get("gradient_clip_norm") == 1.0, "gradient clip drift")
    check(training.get("checkpoint_epochs") == [5, 10, 15, 20], "checkpoint cadence drift")
    check(training.get("teacher_models_enabled") is False, "teacher models forbidden in B1/B2")

    selection = protocol.get("selection", {})
    check(selection.get("frontdoor_order") == ["finite", "known_coverage", "ground_recovery", "valid_to_unknown"], "selection frontdoor drift")
    check(selection.get("lexicographic_order") == ["false_clear", "clearance_mae", "temporal_clearance_delta_mae"], "selection order drift")
    check(selection.get("absrel_can_override_task_failure") is False, "AbsRel cannot override task failure")
    check(protocol.get("next_successor") == "BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TARGET_MATERIALIZATION_AND_MODEL_IMPLEMENTATION_LOCK", "successor drift")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    path = args.protocol.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    errors = validate(protocol)
    for binding in protocol.get("bindings", {}).values():
        bound = (root / binding["path"]).resolve()
        if root not in bound.parents or not bound.is_file():
            errors.append(f"binding missing or outside repo: {binding['path']}")
        elif sha256_file(bound) != binding["sha256"]:
            errors.append(f"binding hash drift: {binding['path']}")
    if errors:
        print(json.dumps({"terminal": "B1_TRAINING_PROTOCOL_INVALID", "errors": errors}, indent=2))
        return 2
    print(json.dumps({
        "terminal": "B1_CONFIDENCE_THRESHOLD_AND_TRAINING_PROTOCOL_LOCK_PASS",
        "protocol_sha256": sha256_file(path),
        "formal_student_training_authorized": False,
        "next_successor": protocol["next_successor"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed static validator for the Spatial Calibration Head R1 freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = REPO_ROOT / "docs/research/hftf/SPATIAL_CALIBRATION_HEAD_R1_PROTOCOL_2026-08-04.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(protocol: dict[str, Any], repo_root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    require(protocol.get("schema") == "blindassist_spatial_calibration_head_r1_protocol", "unexpected schema", errors)
    require(protocol.get("status") == "FROZEN_BEFORE_COHORT_ROSTER_OR_MEDIA_ACCESS", "protocol is not pre-media frozen", errors)

    authority = protocol.get("authority", {})
    require(authority.get("historical_tum_training_or_selection_for_r1") is False, "consumed TUM must be excluded", errors)
    require(authority.get("metric3d_runtime_allowed") is False, "Metric3D runtime must remain forbidden", errors)
    require(authority.get("sealed_media_access_before_activation_receipt") is False, "sealed media must remain closed", errors)
    require("isolated input-only tool" in authority.get("sealed_rgb_identity_audit_exception", "") and "may not read depth" in authority.get("sealed_rgb_identity_audit_exception", ""), "sealed RGB identity-audit firewall changed", errors)

    predecessor = protocol.get("predecessor_lock", {})
    for name in ("protocol", "result"):
        path = repo_root / str(predecessor.get(f"{name}_path", ""))
        require(path.is_file(), f"missing predecessor {name}", errors)
        if path.is_file():
            require(sha256(path) == predecessor.get(f"{name}_sha256"), f"predecessor {name} hash mismatch", errors)

    data = protocol.get("data", {})
    cohort = data.get("cohort", {})
    source_root = repo_root / "artifacts.local/downloads/ARKitScenes-7283761"
    source_files = {
        "metadata": (data.get("metadata_path"), data.get("metadata_sha256")),
        "data_document": ("DATA.md", data.get("data_document_sha256")),
        "license": ("LICENSE", data.get("license_sha256")),
        "download_script": ("download_data.py", data.get("download_script_sha256")),
    }
    for name, (relative, expected) in source_files.items():
        path = source_root / str(relative)
        require(path.is_file(), f"missing ARKitScenes {name}", errors)
        if path.is_file():
            require(sha256(path) == expected, f"ARKitScenes {name} hash mismatch", errors)
    counts = [cohort.get("train_parent_count"), cohort.get("validation_parent_count"), cohort.get("sealed_parent_count")]
    require(counts == [16, 4, 4], "cohort counts must be 16/4/4", errors)
    require(cohort.get("unique_parent_count") == 24, "cohort total must be 24", errors)
    require(cohort.get("videos_per_parent") == 1, "one video per parent required", errors)
    require(cohort.get("frames_per_video") == 150 and cohort.get("target_frame_count") == 3600, "frame budget changed", errors)
    require(cohort.get("parent_overlap_allowed") is False, "parent overlap forbidden", errors)
    require(data.get("parent_unit") == "visit_id" and data.get("frame_unit_is_independent") is False, "parent unit changed", errors)
    require(data.get("truth", {}).get("metric3d") == "not used as R1 truth, teacher, selector, or runtime input", "Metric3D firewall changed", errors)
    require("visit_id NA" in cohort.get("metadata_exclusions", "") and "381879" in cohort.get("metadata_exclusions", ""), "metadata exclusions changed", errors)
    require(cohort.get("metadata_only_selection", "").startswith("select train first, then validation excluding train selections"), "role selection order changed", errors)

    dav2 = protocol.get("dav2", {})
    require(dav2.get("feature_layer_index_zero_based") == 11, "DA layer changed", errors)
    require(dav2.get("feature_kind") == "patch_tokens_without_cls", "CLS-only input forbidden", errors)
    require(dav2.get("patch_stride_px") == 14 and dav2.get("feature_channels") == 384, "DA feature geometry changed", errors)
    require(dav2.get("backbone_trainable") is False, "DA backbone must stay frozen", errors)
    for name in ("checkpoint", "source"):
        path = repo_root / str(dav2.get(f"{name}_path", ""))
        require(path.is_file(), f"missing DA {name}", errors)
        if path.is_file():
            require(sha256(path) == dav2.get(f"{name}_sha256"), f"DA {name} hash mismatch", errors)

    regions = protocol.get("regions", {})
    require(regions.get("names") == ["left", "center", "right"], "regions changed", errors)
    require(regions.get("input_dimension_per_region") == 781, "regional input must be 781-D", errors)
    require(len(regions.get("feature_scalars_per_region", [])) == 13, "13 scalar features required", errors)

    student = protocol.get("student", {})
    require(student.get("layers") == [781, 12, 3], "student layers changed", errors)
    calculated = 781 * 12 + 12 + 12 * 3 + 3
    require(student.get("trainable_parameters") == calculated == 9423, "parameter count must be 9423", errors)
    require(student.get("parameter_budget_min") <= 9423 <= student.get("parameter_budget_max"), "parameter budget failed", errors)
    require("confidence >= 0.5" in student.get("known_rule", ""), "confidence threshold changed", errors)

    loss = protocol.get("loss", {})
    require(loss.get("clearance_regression", {}).get("delta_m") == 0.25, "Huber delta changed", errors)
    require(loss.get("occupancy", {}).get("horizons_m") == [1.0, 1.5, 2.0], "horizons changed", errors)
    require(loss.get("occupancy", {}).get("false_clear_positive_weight") == 3.0, "false-clear weight changed", errors)

    training = protocol.get("training", {})
    frozen = {"learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 80, "batch_size_frames": 64, "seed": 20260804, "seed_count": 1, "early_stopping": False}
    for key, expected in frozen.items():
        require(training.get(key) == expected, f"training field changed: {key}", errors)

    arms = protocol.get("arms", {})
    require(arms.get("primary") == ["raw_dav2", "train_parent_constant_global_affine", "global_cls_ridge_770", "spatial_shared_mlp_9423_with_confidence_unknown"], "four-arm comparison changed", errors)
    require(arms.get("forbidden_additional_arms") is True, "additional arms must stay forbidden", errors)
    affine = arms.get("global_affine_label_fit", {})
    require(affine.get("sample_stride_px") == 4 and affine.get("minimum_pairs") == 500, "global affine opportunity changed", errors)
    require(affine.get("slope_bounds") == [0.25, 4.0] and affine.get("maximum_inlier_median_absolute_residual_m") == 0.25, "global affine gates changed", errors)

    evaluation = protocol.get("evaluation", {})
    clearance_path = repo_root / str(evaluation.get("clearance_primitive_path", ""))
    require(clearance_path.is_file(), "missing clearance primitive", errors)
    if clearance_path.is_file():
        require(sha256(clearance_path) == evaluation.get("clearance_primitive_sha256"), "clearance primitive hash mismatch", errors)
    gates = evaluation.get("task_gates", {})
    require(gates == {"known_coverage_min": 0.9, "clearance_mae_m_max": 0.25, "envelope_agreement_min": 0.9, "false_clear_rate_max": 0.05, "temporal_delta_mae_m_max": 0.15}, "five task gates changed", errors)
    require(protocol.get("evaluation", {}).get("increment_gates", {}).get("jointly_better_than_constant_cv_folds_min") == 3, "3/4 fold gate changed", errors)
    require("parent-macro" in protocol.get("evaluation", {}).get("aggregation", ""), "parent-macro authority missing", errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    errors = validate(protocol)
    result = {
        "schema": "blindassist_spatial_calibration_head_r1_protocol_validation",
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "errors": errors,
        "terminal": "SPATIAL_CALIBRATION_HEAD_R1_PROTOCOL_VALID" if not errors else "SPATIAL_CALIBRATION_HEAD_R1_PRECHECK_HOLD"
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()

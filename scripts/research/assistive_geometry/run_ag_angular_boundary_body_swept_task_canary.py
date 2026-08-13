#!/usr/bin/env python3
"""Test whether the R21 angular boundary signal reaches the frozen AG task.

The canary deliberately isolates one factor.  ICL RGB/K is passed through the
frozen R20 and R21 boundary heads first.  Only after both predictions exist do
we open the already-consumed ICL source geometry.  Metric depth, support and
obstacle evidence remain source anchored; only boundary probability changes
before the unchanged FactorTensorAdapter and body-swept reducer.

This is a consumed Development diagnostic, not a fresh confirmation or a
product/safety result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from evaluate_ag_st_student_bonn_depth import (  # noqa: E402
    build_students,
    checkpoint_architecture,
    checkpoint_parent_ids,
)
from evaluate_ag_st_unified_student_icl_boundary import (  # noqa: E402
    ICL_OUTPUT_HW,
    extract_icl_feature,
)
from run_ag_st_direct_teacher_to_ag_real_seam import (  # noqa: E402
    ADAPTER_OUTPUT_SCHEMA,
    FACTOR_DOWNSAMPLE,
    REDUCER_OUTPUT_SCHEMA,
    adapt_factor_tensor,
    block_all,
    block_max,
    build_factor_and_receipts,
    reduce_frame,
    reducer_sha256,
    require,
    sha256_file,
    write_json,
)
from train_ag_st_masked_student import load_depthart_backbone  # noqa: E402


DEFAULT_R20_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-depthart-angular-boundary-trisource-r0/masked-factor-head.pt"
)
EXPECTED_R20_CHECKPOINT_SHA256 = (
    "35653B160F57842D63AFBE7B210A7EF5556427C7AECC349F45D7F49EE6B74ADB"
)
DEFAULT_R21_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-depthart-angular-boundary-trisource-massnorm-r1/masked-factor-head.pt"
)
EXPECTED_R21_CHECKPOINT_SHA256 = (
    "72B18AC946CC70CE73E053119C00764768E02B34697DF7DA9ADB635C469D6786"
)
DEFAULT_ICL_LABEL_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-r2-icl-fresh-confirmation-labels-r0/result.json"
)
EXPECTED_ICL_LABEL_RESULT_SHA256 = (
    "E3A8F7FF73BD30AD9701D090F5D8959F4C93F45BB70944C85BA01D0AE3CAFBB1"
)
DEFAULT_DIRECT_SEAM_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-direct-teacher-to-ag-real-seam-r4/result.json"
)
EXPECTED_DIRECT_SEAM_RESULT_SHA256 = (
    "78A2265651948D9C1B26A237308E1B87B11F774275F79C011953C6C26E8A99DC"
)
DEFAULT_DEPTHART_SOURCE = (
    REPO_ROOT / "artifacts.local/models/depthart/source"
)
DEFAULT_DEPTHART_CHECKPOINT = (
    DEFAULT_DEPTHART_SOURCE / "checkpoints/metric/depthart_metric_indoor_s_448.pth"
)
EXPECTED_DEPTHART_CHECKPOINT_SHA256 = (
    "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-angular-boundary-body-swept-task-canary-r0"
)
EXPECTED_PARENT_ID = "icl_living_room_kt1"
EXPECTED_FRAME_COUNT = 12
VARIANTS = ("source_reference", "r20_boundary", "r21_boundary")


def load_rgb_only(label_path: Path, expected_sha256: str) -> tuple[str, np.ndarray]:
    """Read only RGB and identity; source geometry remains unopened."""

    require(label_path.is_file(), f"ICL label payload missing: {label_path}")
    require(sha256_file(label_path) == expected_sha256, "ICL label payload SHA drift")
    with np.load(label_path, allow_pickle=False) as payload:
        sample_id = str(np.asarray(payload["sample_id"]).item())
        rgb = np.asarray(payload["rgb_u8_hwc"], dtype=np.uint8).copy()
    require(rgb.shape == (*ICL_OUTPUT_HW, 3), "ICL RGB shape drift")
    return sample_id, rgb


def load_student(
    checkpoint_path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    require(checkpoint_path.is_file(), f"boundary checkpoint missing: {checkpoint_path}")
    require(sha256_file(checkpoint_path) == expected_sha256, "boundary checkpoint SHA drift")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), "boundary checkpoint invalid")
    architecture = checkpoint_architecture(checkpoint)
    require(
        architecture["objective_profile"] == "angular_boundary_only",
        "checkpoint is not angular-boundary-only",
    )
    require(
        EXPECTED_PARENT_ID not in checkpoint_parent_ids(checkpoint),
        "ICL parent leaked into boundary checkpoint",
    )
    _, student = build_students(checkpoint, architecture, device)
    return student, {"checkpoint": checkpoint, "architecture": architecture}


def source_observed_factor_mask(label_path: Path) -> np.ndarray:
    with np.load(label_path, allow_pickle=False) as payload:
        depth_valid = np.asarray(payload["metric_depth_valid_hw"], dtype=np.bool_)
        evidence_valid = np.asarray(payload["evidence_truth_valid_hw"], dtype=np.bool_)
    require(depth_valid.shape == evidence_valid.shape == ICL_OUTPUT_HW, "ICL validity shape drift")
    return block_all(depth_valid & evidence_valid, FACTOR_DOWNSAMPLE)


def replace_observed_boundary(
    source_prediction: dict[str, Any],
    source_observed: np.ndarray,
    learned_probability: np.ndarray,
    *,
    variant: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Replace only observed boundary blocks and preserve Tier-C completion."""

    expected_hw = (
        source_observed.shape[0] * FACTOR_DOWNSAMPLE,
        source_observed.shape[1] * FACTOR_DOWNSAMPLE,
    )
    require(learned_probability.shape == expected_hw, "learned boundary shape drift")
    require(
        bool(np.isfinite(learned_probability).all())
        and bool(((learned_probability >= 0.0) & (learned_probability <= 1.0)).all()),
        "learned boundary probability invalid",
    )
    output = copy.deepcopy(source_prediction)
    reference = np.asarray(
        output["obstacle_boundary_evidence"]["boundary_probability_hw"],
        dtype=np.float32,
    )
    learned_factor = block_max(learned_probability.astype(np.float32), FACTOR_DOWNSAMPLE)
    require(
        reference.shape == learned_factor.shape == source_observed.shape,
        "boundary factor shape drift",
    )
    replaced = np.where(source_observed, learned_factor, reference).astype(np.float32)
    output["obstacle_boundary_evidence"]["boundary_probability_hw"] = replaced.tolist()
    output["factor_identity"] = {
        **dict(output["factor_identity"]),
        "boundary_source": variant,
        "boundary_checkpoint_sha256": checkpoint_sha256,
        "source_exact_depth_support_obstacle_retained": True,
        "learned_final_task_head": False,
    }
    return output


def reduce_prediction(
    prediction: dict[str, Any],
    geometry: dict[str, Any],
    calibration: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    adapted = adapt_factor_tensor(
        {
            "prediction": prediction,
            "geometry_receipt": geometry,
            "calibration_receipt": calibration,
        }
    )
    require(adapted["schema"] == ADAPTER_OUTPUT_SCHEMA, "adapter output schema drift")
    first = reduce_frame(adapted, profile)
    second = reduce_frame(
        json.loads(json.dumps(adapted)),
        json.loads(json.dumps(profile)),
    )
    require(first["schema"] == REDUCER_OUTPUT_SCHEMA, "reducer output schema drift")
    require(reducer_sha256(first) == reducer_sha256(second), "reducer is nondeterministic")
    return adapted, first


def state_map(reduced: dict[str, Any]) -> dict[tuple[str, float], str]:
    output: dict[tuple[str, float], str] = {}
    for band in reduced["bands"]:
        for cell in band["cells"]:
            key = (str(band["band"]), float(cell["horizon_m"]))
            require(key not in output, "duplicate reducer cell")
            output[key] = str(cell["state"])
    return output


def comparison_metrics(
    references: list[dict[tuple[str, float], str]],
    candidates: list[dict[tuple[str, float], str]],
) -> dict[str, Any]:
    require(len(references) == len(candidates) > 0, "state comparison frame count drift")
    pairs: list[tuple[str, str]] = []
    for reference, candidate in zip(references, candidates):
        require(set(reference) == set(candidate), "reducer cell identity drift")
        pairs.extend((reference[key], candidate[key]) for key in sorted(reference))
    reference_known = sum(reference != "UNKNOWN" for reference, _ in pairs)
    candidate_known = sum(candidate != "UNKNOWN" for _, candidate in pairs)
    exact = sum(reference == candidate for reference, candidate in pairs)
    correct_reference_known = sum(
        reference != "UNKNOWN" and reference == candidate
        for reference, candidate in pairs
    )
    return {
        "cell_count": len(pairs),
        "reference_known_count": reference_known,
        "candidate_known_count": candidate_known,
        "exact_match_count": exact,
        "exact_match_fraction": exact / len(pairs),
        "correct_reference_known_count": correct_reference_known,
        "correct_reference_known_fraction": (
            correct_reference_known / reference_known if reference_known else None
        ),
        "abstain_on_reference_known_count": sum(
            reference != "UNKNOWN" and candidate == "UNKNOWN"
            for reference, candidate in pairs
        ),
        "false_clear_count": sum(
            reference == "OCCUPIED_OBSERVED" and candidate == "CLEAR_OBSERVED"
            for reference, candidate in pairs
        ),
        "false_block_count": sum(
            reference == "CLEAR_OBSERVED" and candidate == "OCCUPIED_OBSERVED"
            for reference, candidate in pairs
        ),
        "unsafe_clear_count": sum(
            reference != "CLEAR_OBSERVED" and candidate == "CLEAR_OBSERVED"
            for reference, candidate in pairs
        ),
        "spurious_definite_count": sum(
            reference == "UNKNOWN" and candidate != "UNKNOWN"
            for reference, candidate in pairs
        ),
        "candidate_state_counts": dict(sorted(Counter(candidate for _, candidate in pairs).items())),
        "reference_state_counts": dict(sorted(Counter(reference for reference, _ in pairs).items())),
    }


def task_gain_gates(r20: dict[str, Any], r21: dict[str, Any]) -> dict[str, bool]:
    strict_gain = (
        r21["exact_match_count"] > r20["exact_match_count"]
        or r21["correct_reference_known_count"] > r20["correct_reference_known_count"]
        or r21["abstain_on_reference_known_count"]
        < r20["abstain_on_reference_known_count"]
    )
    return {
        "R21TASK_C06_NO_UNSAFE_CLEAR": r21["unsafe_clear_count"] == 0,
        "R21TASK_C07_NO_SPURIOUS_DEFINITE_STATE": r21["spurious_definite_count"] == 0,
        "R21TASK_C08_EXACT_MATCH_NO_REGRET_VS_R20": (
            r21["exact_match_count"] >= r20["exact_match_count"]
        ),
        "R21TASK_C09_REFERENCE_KNOWN_CORRECT_NO_REGRET_VS_R20": (
            r21["correct_reference_known_count"]
            >= r20["correct_reference_known_count"]
        ),
        "R21TASK_C10_REFERENCE_KNOWN_ABSTENTION_NO_REGRET_VS_R20": (
            r21["abstain_on_reference_known_count"]
            <= r20["abstain_on_reference_known_count"]
        ),
        "R21TASK_C11_STRICT_BODY_SWEPT_TASK_GAIN_VS_R20": strict_gain,
        "R21TASK_C12_NONZERO_OBSERVED_COVERAGE": r21["candidate_known_count"] > 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"output exists: {output_dir}")
    require(torch.cuda.is_available(), "CUDA required")
    require(
        sha256_file(args.icl_label_result.resolve()) == EXPECTED_ICL_LABEL_RESULT_SHA256,
        "ICL label result SHA drift",
    )
    require(
        sha256_file(args.direct_seam_result.resolve()) == EXPECTED_DIRECT_SEAM_RESULT_SHA256,
        "direct seam result SHA drift",
    )
    require(
        sha256_file(args.depthart_checkpoint.resolve()) == EXPECTED_DEPTHART_CHECKPOINT_SHA256,
        "DepthART checkpoint SHA drift",
    )
    labels = json.loads(args.icl_label_result.read_text(encoding="utf-8"))
    direct = json.loads(args.direct_seam_result.read_text(encoding="utf-8"))
    require(
        labels.get("passed") is True
        and labels.get("status") == "AG_R2_ICL_FRESH_CONFIRMATION_LABELS_PASS",
        "ICL source geometry frontdoor invalid",
    )
    require(
        direct.get("passed") is True
        and direct.get("status") == "AG_ST_DIRECT_TEACHER_TO_AG_REAL_SEAM_PASS",
        "direct SuperTeacher seam prerequisite invalid",
    )
    rows = sorted(labels["frames"], key=lambda row: str(row["sample_id"]))
    require(
        len(rows) == EXPECTED_FRAME_COUNT
        and {str(row["parent_id"]) for row in rows} == {EXPECTED_PARENT_ID},
        "ICL canary roster drift",
    )
    profile = dict(direct["reducer_profile"]["profile"])
    normal_sigma_rad = float(direct["factor_factory"]["support_normal_sigma_rad"])

    device = torch.device(args.device)
    r20_model, r20 = load_student(
        args.r20_checkpoint.resolve(), EXPECTED_R20_CHECKPOINT_SHA256, device
    )
    r21_model, r21 = load_student(
        args.r21_checkpoint.resolve(), EXPECTED_R21_CHECKPOINT_SHA256, device
    )
    require(
        r20["architecture"] == r21["architecture"],
        "R20/R21 architecture drift",
    )
    extractor, scan = load_depthart_backbone(
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        int(r21["checkpoint"]["seed"]),
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    probabilities: dict[str, dict[str, np.ndarray]] = {
        "r20_boundary": {},
        "r21_boundary": {},
    }

    # Prediction phase: source geometry fields are not read here.
    for row in rows:
        label_path = Path(row["output"])
        sample_id, rgb = load_rgb_only(label_path, str(row["output_sha256"]))
        require(sample_id == str(row["sample_id"]), "ICL RGB identity drift")
        feature, base_depth = extract_icl_feature(
            extractor,
            rgb,
            r21["architecture"]["feature_profile"],
            device,
            amp_dtype,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            r20_output = r20_model(feature, base_depth, ICL_OUTPUT_HW)
            r21_output = r21_model(feature, base_depth, ICL_OUTPUT_HW)
        probabilities["r20_boundary"][sample_id] = (
            torch.sigmoid(r20_output["boundary_logits"])[0, 0].float().cpu().numpy()
        )
        probabilities["r21_boundary"][sample_id] = (
            torch.sigmoid(r21_output["boundary_logits"])[0, 0].float().cpu().numpy()
        )
    del extractor, r20_model, r21_model
    torch.cuda.empty_cache()

    state_rows: dict[str, list[dict[tuple[str, float], str]]] = {
        variant: [] for variant in VARIANTS
    }
    frame_receipts: list[dict[str, Any]] = []
    all_deterministic = True
    all_adapter_valid = True
    any_boundary_difference = False
    for row in rows:
        sample_id = str(row["sample_id"])
        label_path = Path(row["output"])
        source_prediction, geometry, calibration, _, receipt = build_factor_and_receipts(
            label_path,
            EXPECTED_ICL_LABEL_RESULT_SHA256,
            normal_sigma_rad,
        )
        observed = source_observed_factor_mask(label_path)
        candidates = {
            "source_reference": source_prediction,
            "r20_boundary": replace_observed_boundary(
                source_prediction,
                observed,
                probabilities["r20_boundary"][sample_id],
                variant="R20_FIXED_LOSS_ANGULAR_BOUNDARY",
                checkpoint_sha256=EXPECTED_R20_CHECKPOINT_SHA256,
            ),
            "r21_boundary": replace_observed_boundary(
                source_prediction,
                observed,
                probabilities["r21_boundary"][sample_id],
                variant="R21_TARGET_MASS_NORMALIZED_ANGULAR_BOUNDARY",
                checkpoint_sha256=EXPECTED_R21_CHECKPOINT_SHA256,
            ),
        }
        variant_receipts: dict[str, Any] = {}
        reference_boundary = np.asarray(
            candidates["source_reference"]["obstacle_boundary_evidence"]["boundary_probability_hw"]
        )
        for variant, prediction in candidates.items():
            adapted, reduced = reduce_prediction(
                prediction, geometry, calibration, profile
            )
            states = state_map(reduced)
            state_rows[variant].append(states)
            candidate_boundary = np.asarray(
                prediction["obstacle_boundary_evidence"]["boundary_probability_hw"]
            )
            if variant != "source_reference":
                any_boundary_difference = any_boundary_difference or bool(
                    np.any(candidate_boundary[observed] != reference_boundary[observed])
                )
            all_adapter_valid = all_adapter_valid and bool(
                adapted["depth_scale"]["valid"]
                and adapted["support"]["valid"]
                and adapted["boundary"]["valid"]
                and adapted["boundary"]["coverage"] >= 0.99
            )
            variant_receipts[variant] = {
                "state_counts": dict(sorted(Counter(states.values()).items())),
                "adapter_boundary_coverage": float(adapted["boundary"]["coverage"]),
                "adapter_obstacle_count": len(adapted["boundary"]["obstacles"]),
                "reducer_canonical_sha256": reducer_sha256(reduced),
            }
        frame_receipts.append(
            {
                "sample_id": sample_id,
                "parent_id": str(row["parent_id"]),
                "source_payload_sha256": str(row["output_sha256"]),
                "source_observed_factor_blocks": int(observed.sum()),
                "factor_receipt": receipt,
                "variants": variant_receipts,
            }
        )

    r20_metrics = comparison_metrics(
        state_rows["source_reference"], state_rows["r20_boundary"]
    )
    r21_metrics = comparison_metrics(
        state_rows["source_reference"], state_rows["r21_boundary"]
    )
    gates = {
        "R21TASK_C01_EXACT_PREREQUISITE_RECEIPTS": True,
        "R21TASK_C02_ICL_EXCLUDED_FROM_CHECKPOINT_FIT_SELECTION": True,
        "R21TASK_C03_RGB_K_PREDICTIONS_BEFORE_SOURCE_GEOMETRY": True,
        "R21TASK_C04_BOUNDARY_ONLY_INTERVENTION_NONZERO": any_boundary_difference,
        "R21TASK_C05_ADAPTER_VALID_AND_REDUCER_DETERMINISTIC_12_OF_12": (
            all_adapter_valid and all_deterministic and len(frame_receipts) == EXPECTED_FRAME_COUNT
        ),
        **task_gain_gates(r20_metrics, r21_metrics),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_angular_boundary_body_swept_task_canary_result_v1",
        "status": (
            "AG_ANGULAR_BOUNDARY_BODY_SWEPT_TASK_GAIN_PASS"
            if passed
            else "AG_ANGULAR_BOUNDARY_BODY_SWEPT_TASK_GAIN_NOT_SUPPORTED"
        ),
        "passed": passed,
        "question": (
            "Does R21 target-mass-normalized angular boundary supervision improve "
            "the unchanged body-swept reducer over R20 when source-exact depth, "
            "support and obstacle evidence are held fixed?"
        ),
        "protocol": {
            "role": "CONSUMED_DEVELOPMENT_BOUNDARY_FACTOR_INTERVENTION",
            "parent_id": EXPECTED_PARENT_ID,
            "frame_count": EXPECTED_FRAME_COUNT,
            "cell_count": EXPECTED_FRAME_COUNT * 9,
            "prediction_inputs": "RGB_PLUS_K_ONLY",
            "source_geometry_opened_after_both_predictions": True,
            "held_fixed_factors": ["metric_depth", "support", "obstacle_evidence"],
            "intervened_factor": "boundary_probability",
            "reference": "SOURCE_EXACT_FACTORS_THROUGH_UNCHANGED_ADAPTER_REDUCER",
            "unknown_policy": "UNKNOWN is abstention and never negative",
            "selection_or_retuning_on_task_outcome": False,
        },
        "inputs": {
            "r20_checkpoint": {
                "path": str(args.r20_checkpoint.resolve()),
                "sha256": EXPECTED_R20_CHECKPOINT_SHA256,
            },
            "r21_checkpoint": {
                "path": str(args.r21_checkpoint.resolve()),
                "sha256": EXPECTED_R21_CHECKPOINT_SHA256,
            },
            "icl_label_result": {
                "path": str(args.icl_label_result.resolve()),
                "sha256": EXPECTED_ICL_LABEL_RESULT_SHA256,
            },
            "direct_seam_result": {
                "path": str(args.direct_seam_result.resolve()),
                "sha256": EXPECTED_DIRECT_SEAM_RESULT_SHA256,
            },
            "depthart_checkpoint_sha256": EXPECTED_DEPTHART_CHECKPOINT_SHA256,
        },
        "frozen_adapter_reducer": {
            "profile": profile,
            "support_normal_sigma_rad": normal_sigma_rad,
        },
        "metrics": {
            "r20_boundary_vs_source_reference": r20_metrics,
            "r21_boundary_vs_source_reference": r21_metrics,
        },
        "gates": gates,
        "frames": frame_receipts,
        "execution": {
            "device": str(torch.cuda.get_device_name(device)),
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "scan_backend": scan,
            "elapsed_seconds": time.perf_counter() - started,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "decision": {
            "r21_boundary_component_retained": True,
            "r21_boundary_task_gain_supported": passed,
            "correction_router_reopened": False,
            "fresh3_tum_opened": False,
            "complete_task_superiority_claim": False,
            "default_app_changed": False,
            "next_action_if_pass": (
                "Freeze the R21 boundary component in the research factor recipe; "
                "next test its runtime/export mechanics without reopening depth correction."
            ),
            "next_action_if_fail": (
                "Retain R21 only as boundary-localization evidence; do not claim task landing."
            ),
        },
        "claim_boundary": (
            "Consumed one-parent synthetic-exact Development evidence for an isolated "
            "boundary-factor intervention through the frozen adapter/reducer. Not fresh "
            "cross-sensor confirmation, complete task superiority, deployment, product, "
            "default-App, navigation or assistive-safety evidence."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r20-checkpoint", type=Path, default=DEFAULT_R20_CHECKPOINT)
    parser.add_argument("--r21-checkpoint", type=Path, default=DEFAULT_R21_CHECKPOINT)
    parser.add_argument("--icl-label-result", type=Path, default=DEFAULT_ICL_LABEL_RESULT)
    parser.add_argument("--direct-seam-result", type=Path, default=DEFAULT_DIRECT_SEAM_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "metrics": result["metrics"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

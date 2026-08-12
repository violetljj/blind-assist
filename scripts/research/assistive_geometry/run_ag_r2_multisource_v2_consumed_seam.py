#!/usr/bin/env python3
"""Run the v2 uncertainty decomposition on consumed ICL and real TUM labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from calibrate_ag_r2_metric_scale_residual_bank import (  # noqa: E402
    load_residual_bank,
    pooled_uncertainty_feature,
    predict_scale_sigma,
)
from factor_tensor_adapter_v2 import (  # noqa: E402
    CALIBRATION_SCHEMA,
    FACTOR_SCHEMA_SHA256,
    PREDICTION_SCHEMA,
    adapt_factor_tensor,
    canonical_sha256 as adapter_sha256,
)
import run_ag_r2_hybrid_factor_student_to_ag_seam as hybrid  # noqa: E402
from train_ag_r2_metric_depth_student import MetricDepthStudentHead  # noqa: E402


FROZEN_R10_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-hybrid-factor-student-to-ag-seam-r10/result.json"
)
EXPECTED_FROZEN_R10_SHA256 = (
    "5077213EA3B5B0CF0186E755BF1D0FDDF8C25DBB65D415E3A05263A6780A9720"
)
ICL_LABEL_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-icl-fresh-confirmation-labels-r0/result.json"
)
EXPECTED_ICL_LABEL_SHA256 = (
    "E3A8F7FF73BD30AD9701D090F5D8959F4C93F45BB70944C85BA01D0AE3CAFBB1"
)
TUM_LABEL_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-tum-real-fresh-confirmation-labels-r0/result.json"
)
EXPECTED_TUM_LABEL_SHA256 = (
    "04C10167E2C94010D4680510A30F0F05B284D822EF4858ED772E83B0390F4ABB"
)
METRIC_STUDENT_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-multisource-metric-depth-student-r0/result.json"
)
EXPECTED_METRIC_STUDENT_RESULT_SHA256 = (
    "F0703357B0F25C7ABF209EE53DE9B04E588BEDE3629C1B1F5273D9E31D41BFF3"
)
EXPECTED_METRIC_STUDENT_CHECKPOINT_SHA256 = (
    "980B26D16659BF1AAF47C64C5CBAC63A5E91D60573E93AD23190DFF3BB67E4B7"
)
SCALE_BANK_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-metric-scale-residual-bank-r2/result.json"
)
EXPECTED_SCALE_BANK_RESULT_SHA256 = (
    "68188B00FB4951771443B4706526F6BAAED1AA85E77C4AE4D73958600AB4C0E5"
)
EXPECTED_SCALE_BANK_SHA256 = (
    "D6B8BE961A9576309097842D264791CD74F74B52B33CD97B0842DDBA47EEC88E"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-r2-multisource-v2-consumed-seam-r0"
)
LOCAL_SHAPE_RELATIVE_SIGMA = 0.17132260198626506


def load_metric_model(
    result: dict[str, Any], device: torch.device
) -> tuple[MetricDepthStudentHead, Path]:
    checkpoint = Path(result["checkpoint"]["path"])
    hybrid.require(
        hybrid.sha256_file(checkpoint) == EXPECTED_METRIC_STUDENT_CHECKPOINT_SHA256,
        "multi-source metric checkpoint drift",
    )
    model = MetricDepthStudentHead(
        hidden=int(result["architecture"]["hidden_channels"]),
        global_hidden=int(result["architecture"]["global_hidden_channels"]),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, checkpoint


def load_prediction_v2(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as value:
        return {
            "schema": str(np.asarray(value["schema"]).item()),
            "sample_id": str(np.asarray(value["sample_id"]).item()),
            "factor_identity": json.loads(
                str(np.asarray(value["factor_identity_json"]).item())
            ),
            "camera_geometry_receipt_sha256": str(
                np.asarray(value["camera_geometry_receipt_sha256"]).item()
            ),
            "depth_scale": {
                "depth_shape_positive_hw": np.asarray(
                    value["depth_shape_positive_hw"]
                ).tolist(),
                "log_metric_scale_m_scalar": float(
                    np.asarray(value["log_metric_scale_m_scalar"]).item()
                ),
                "metric_scale_log_sigma_scalar": float(
                    np.asarray(value["metric_scale_log_sigma_scalar"]).item()
                ),
                "depth_shape_log_sigma_hw": np.asarray(
                    value["depth_shape_log_sigma_hw"]
                ).tolist(),
                "depth_valid_probability_hw": np.asarray(
                    value["depth_valid_probability_hw"]
                ).tolist(),
                "metric_scale_valid": bool(
                    np.asarray(value["metric_scale_valid"]).item()
                ),
            },
            "support_surface": {
                "support_probability_hw": np.asarray(
                    value["support_probability_hw"]
                ).tolist(),
                "support_plane_normal_camera_xyz": np.asarray(
                    value["support_plane_normal_camera_xyz"]
                ).tolist(),
                "camera_height_m": float(
                    np.asarray(value["camera_height_m"]).item()
                ),
                "support_residual_sigma_m": float(
                    np.asarray(value["support_residual_sigma_m"]).item()
                ),
                "support_valid": bool(np.asarray(value["support_valid"]).item()),
            },
            "obstacle_boundary_evidence": {
                "obstacle_evidence_probability_hw": np.asarray(
                    value["obstacle_evidence_probability_hw"]
                ).tolist(),
                "boundary_probability_hw": np.asarray(
                    value["boundary_probability_hw"]
                ).tolist(),
                "boundary_localization_sigma_px_hw": np.asarray(
                    value["boundary_localization_sigma_px_hw"]
                ).tolist(),
                "evidence_valid_hw": np.asarray(value["evidence_valid_hw"]).tolist(),
            },
        }


def serialize_v2(
    samples: list[Any],
    outputs: dict[str, dict[str, torch.Tensor]],
    scale_sigmas: dict[str, float],
    identity: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    destination = output_dir / "factor_tensors"
    destination.mkdir(parents=True, exist_ok=False)
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    receipts: list[dict[str, Any]] = []
    for sample in samples:
        factors = outputs[sample.sample_id]
        log_depth = factors["predicted_log_depth"][0, 0]
        log_scale = log_depth.mean()
        with np.load(sample.label_path, allow_pickle=False) as source:
            camera_receipt = str(
                np.asarray(source["camera_geometry_receipt_sha256"]).item()
            )
        evidence = factors["evidence_valid_probability"][0, 0] >= 0.5
        scale_sigma = float(scale_sigmas[sample.sample_id])
        payload = {
            "schema": np.asarray(PREDICTION_SCHEMA),
            "sample_id": np.asarray(sample.sample_id),
            "factor_identity_json": np.asarray(identity_json),
            "camera_geometry_receipt_sha256": np.asarray(camera_receipt),
            "depth_shape_positive_hw": torch.exp(log_depth - log_scale)
            .cpu()
            .numpy()
            .astype(np.float32),
            "log_metric_scale_m_scalar": np.asarray(float(log_scale), dtype=np.float32),
            "metric_scale_log_sigma_scalar": np.asarray(
                math.log(scale_sigma), dtype=np.float32
            ),
            "depth_shape_log_sigma_hw": factors["depth_log_sigma"][0, 0]
            .cpu()
            .numpy()
            .astype(np.float32),
            "depth_valid_probability_hw": factors["depth_valid_probability"][0, 0]
            .cpu()
            .numpy()
            .astype(np.float32),
            "metric_scale_valid": np.asarray(
                bool(factors["depth_valid_probability"].mean() >= 0.5)
            ),
            "support_probability_hw": factors["support_probability"][0, 0]
            .cpu()
            .numpy()
            .astype(np.float32),
            "support_plane_normal_camera_xyz": factors[
                "support_plane_normal_camera_xyz"
            ][0]
            .cpu()
            .numpy()
            .astype(np.float32),
            "camera_height_m": np.asarray(
                float(factors["camera_height_m"][0]), dtype=np.float32
            ),
            "support_residual_sigma_m": np.asarray(
                float(factors["support_residual_sigma_m"][0]), dtype=np.float32
            ),
            "support_valid": np.asarray(
                bool(factors["support_valid_probability"][0] >= 0.5)
            ),
            "obstacle_evidence_probability_hw": factors["obstacle_probability"][0, 0]
            .cpu()
            .numpy()
            .astype(np.float32),
            "boundary_probability_hw": factors["boundary_probability"][0, 0]
            .cpu()
            .numpy()
            .astype(np.float32),
            "boundary_localization_sigma_px_hw": factors["boundary_sigma_px"][0, 0]
            .cpu()
            .numpy()
            .astype(np.float32),
            "evidence_valid_hw": evidence.cpu().numpy().astype(np.bool_),
        }
        path = destination / f"{sample.sample_id}.npz"
        np.savez_compressed(path, **payload)
        with np.load(path, allow_pickle=False) as written:
            hybrid.require(set(written.files) == set(payload), "v2 payload field drift")
            hybrid.require(
                str(np.asarray(written["schema"]).item()) == PREDICTION_SCHEMA,
                "v2 payload schema drift",
            )
        receipts.append(
            {
                "sample_id": sample.sample_id,
                "parent_id": sample.parent_id,
                "path": str(path.resolve()),
                "sha256": hybrid.sha256_file(path),
                "bytes": path.stat().st_size,
                "metric_scale_relative_sigma": scale_sigma,
                "camera_geometry_receipt_sha256": camera_receipt,
            }
        )
    return receipts


def calibration_receipt_v2(calibration: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": CALIBRATION_SCHEMA,
        "calibration_id": "AG_R2_MULTISOURCE_V2_CONSUMED_CALIBRATION_R0",
        "factor_schema_sha256": FACTOR_SCHEMA_SHA256,
        "source_role": "FIT_ONLY_CALIBRATION",
        "task_outcome_used": False,
        "scale_relative_sigma_floor": 0.02,
        "scale_relative_sigma_cap": 1.0,
        "support_normal_sigma_rad": float(calibration["support_normal_sigma_rad"]),
        "support_height_sigma_m": float(calibration["support_height_sigma_m"]),
        "boundary_sigma_floor_px": hybrid.BOUNDARY_SIGMA_FLOOR_FACTOR_PX,
        "evidence_sigma_floor": hybrid.EVIDENCE_SIGMA_FLOOR,
    }


def state_reason_counts(reduced: dict[str, Any]) -> tuple[Counter[str], Counter[str]]:
    states: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for band in reduced["bands"]:
        for cell in band["cells"]:
            states[str(cell["state"])] += 1
            reasons.update(str(value) for value in cell["reason_codes"])
    return states, reasons


def run(args: argparse.Namespace) -> dict[str, Any]:
    hybrid.require(
        torch.cuda.is_available() and str(args.device).startswith("cuda"),
        "CUDA required",
    )
    hybrid.require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    exact_receipts = {
        args.frozen_r10_result: EXPECTED_FROZEN_R10_SHA256,
        args.icl_label_result: EXPECTED_ICL_LABEL_SHA256,
        args.tum_label_result: EXPECTED_TUM_LABEL_SHA256,
        args.metric_student_result: EXPECTED_METRIC_STUDENT_RESULT_SHA256,
        args.scale_bank_result: EXPECTED_SCALE_BANK_RESULT_SHA256,
    }
    for path, expected in exact_receipts.items():
        hybrid.require(hybrid.sha256_file(path) == expected, f"receipt drift: {path}")
    hybrid.require(
        hybrid.sha256_file(args.baseline_result) == hybrid.EXPECTED_BASELINE_RESULT_SHA256,
        "baseline result drift",
    )
    hybrid.require(
        hybrid.sha256_file(args.depthart_checkpoint) == hybrid.EXPECTED_DEPTHART_SHA256,
        "DepthART checkpoint drift",
    )

    frozen = json.loads(args.frozen_r10_result.read_text(encoding="utf-8"))
    icl = json.loads(args.icl_label_result.read_text(encoding="utf-8"))
    tum = json.loads(args.tum_label_result.read_text(encoding="utf-8"))
    metric_result = json.loads(args.metric_student_result.read_text(encoding="utf-8"))
    bank_result = json.loads(args.scale_bank_result.read_text(encoding="utf-8"))
    hybrid.require(
        frozen["passed"] and icl["passed"] and tum["passed"]
        and metric_result["passed"] and bank_result["passed"],
        "consumed seam prerequisite failed",
    )
    rows = sorted(
        [
            *[{**dict(row), "role": "CONSUMED_ICL"} for row in icl["frames"]],
            *[{**dict(row), "role": "CONSUMED_TUM_REAL"} for row in tum["frames"]],
        ],
        key=lambda row: str(row["sample_id"]),
    )
    hybrid.require(
        len(rows) == 24 and len({row["sample_id"] for row in rows}) == 24,
        "consumed v2 roster drift",
    )
    bank_path = Path(bank_result["bank"]["path"])
    hybrid.require(
        hybrid.sha256_file(bank_path) == EXPECTED_SCALE_BANK_SHA256,
        "scale residual bank drift",
    )
    bank = load_residual_bank(bank_path)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    samples, feature_receipt = hybrid.extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))[
        "baseline_parameters"
    ]
    factor_model, factor_checkpoint = hybrid.load_factor_model(baseline, device)
    raw_by_sample = hybrid.raw_model_outputs(samples, factor_model, device)
    del factor_model
    metric_model, metric_checkpoint = load_metric_model(metric_result, device)
    hybrid.attach_metric_depth_student_outputs(
        samples, raw_by_sample, metric_model, device
    )
    del metric_model
    torch.cuda.empty_cache()

    height_estimator = dict(frozen["height_estimator"])
    calibration = dict(frozen["uncertainty_calibration"])
    calibration["depth_shape_relative_sigma"] = LOCAL_SHAPE_RELATIVE_SIGMA
    calibration["depth_shape_relative_sigma_rms"] = LOCAL_SHAPE_RELATIVE_SIGMA
    reducer_profile = dict(frozen["reducer_profile"]["profile"])
    session_profiles = hybrid.session_height_profiles(samples)
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    geometry_rows: dict[str, dict[str, Any]] = {}
    scale_sigmas: dict[str, float] = {}
    scale_diagnostics: dict[str, dict[str, float]] = {}
    for sample in samples:
        raw = raw_by_sample[sample.sample_id]
        pooled = pooled_uncertainty_feature(
            sample.feature[None],
            sample.base_depth_feature[None],
            raw["metric_depth_student_log_depth"],
        )[0]
        scale_sigma, diagnostic = predict_scale_sigma(pooled, bank)
        scale_sigmas[sample.sample_id] = scale_sigma
        scale_diagnostics[sample.sample_id] = diagnostic
        output, receipt = hybrid.hybrid_output(
            sample,
            raw,
            height_estimator,
            calibration,
            session_profiles.get(sample.parent_id),
        )
        outputs[sample.sample_id] = output
        geometry_rows[sample.sample_id] = receipt

    identity = {
        **dict(frozen["factor_identity"]),
        "model_id": "AG_R2_MULTISOURCE_FACTOR_STUDENT_V2",
        "metric_depth_student_checkpoint_sha256": EXPECTED_METRIC_STUDENT_CHECKPOINT_SHA256,
        "metric_scale_uncertainty_source": "PCA_WHITENED_KNN_RESIDUAL_BANK_WITH_OOD_DISTANCE_INFLATION",
        "local_shape_uncertainty_source": "PARENT_BALANCED_CONSUMED_FACTOR_RESIDUAL_Q68",
        "factor_schema_sha256": FACTOR_SCHEMA_SHA256,
    }
    factors = serialize_v2(samples, outputs, scale_sigmas, identity, args.output_dir)
    factor_by_sample = {row["sample_id"]: row for row in factors}
    adapter_dir = args.output_dir / "adapter_frames"
    reducer_dir = args.output_dir / "reducer_outputs"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    reducer_dir.mkdir(parents=True, exist_ok=False)
    calibration_payload = calibration_receipt_v2(calibration)
    aggregate_states: Counter[str] = Counter()
    aggregate_reasons: Counter[str] = Counter()
    by_role_states: dict[str, Counter[str]] = {
        "CONSUMED_ICL": Counter(),
        "CONSUMED_TUM_REAL": Counter(),
    }
    seam_rows: list[dict[str, Any]] = []
    for sample in samples:
        factor = factor_by_sample[sample.sample_id]
        prediction = load_prediction_v2(Path(factor["path"]))
        geometry = hybrid.geometry_receipt(sample, prediction)
        adapted = adapt_factor_tensor(
            {
                "prediction": prediction,
                "geometry_receipt": geometry,
                "calibration_receipt": calibration_payload,
            }
        )
        reduced_first = hybrid.reduce_frame(adapted, reducer_profile)
        reduced_second = hybrid.reduce_frame(
            json.loads(json.dumps(adapted)),
            json.loads(json.dumps(reducer_profile)),
        )
        deterministic = (
            hybrid.reducer_sha256(reduced_first)
            == hybrid.reducer_sha256(reduced_second)
        )
        state_counts, reason_counts = state_reason_counts(reduced_first)
        aggregate_states.update(state_counts)
        aggregate_reasons.update(reason_counts)
        by_role_states[sample.role].update(state_counts)
        adapter_path = adapter_dir / f"{sample.sample_id}.json"
        reducer_path = reducer_dir / f"{sample.sample_id}.json"
        hybrid.write_json(adapter_path, adapted)
        hybrid.write_json(reducer_path, reduced_first)
        seam_rows.append(
            {
                **geometry_rows[sample.sample_id],
                "role": sample.role,
                "metric_scale_relative_sigma": scale_sigmas[sample.sample_id],
                "scale_uncertainty_diagnostic": scale_diagnostics[sample.sample_id],
                "factor_tensor": factor,
                "adapter_frame": {
                    "path": str(adapter_path.resolve()),
                    "sha256": hybrid.sha256_file(adapter_path),
                    "canonical_sha256": adapter_sha256(adapted),
                },
                "reducer_output": {
                    "path": str(reducer_path.resolve()),
                    "sha256": hybrid.sha256_file(reducer_path),
                    "canonical_sha256": hybrid.reducer_sha256(reduced_first),
                },
                "adapter_depth_valid": bool(adapted["depth_scale"]["valid"]),
                "adapter_support_valid": bool(adapted["support"]["valid"]),
                "adapter_boundary_valid": bool(adapted["boundary"]["valid"]),
                "adapter_boundary_coverage": float(adapted["boundary"]["coverage"]),
                "state_counts": dict(sorted(state_counts.items())),
                "reason_counts": dict(sorted(reason_counts.items())),
                "deterministic_repeat_equal": deterministic,
            }
        )

    structurally_valid = [
        row
        for row in seam_rows
        if row["adapter_depth_valid"]
        and row["adapter_support_valid"]
        and row["adapter_boundary_valid"]
        and row["adapter_boundary_coverage"] > 0.0
    ]
    icl_states = by_role_states["CONSUMED_ICL"]
    tum_states = by_role_states["CONSUMED_TUM_REAL"]
    gates = {
        "V2SEAM_C01_EXACT_CONSUMED_RECEIPTS": True,
        "V2SEAM_C02_SCALE_AND_SHAPE_UNCERTAINTY_SEPARATE": True,
        "V2SEAM_C03_24_FACTOR_TENSORS_ROUNDTRIP": len(factors) == 24,
        "V2SEAM_C04_ADAPTER_STRUCTURALLY_VALID_AT_LEAST_17": len(structurally_valid)
        >= 17,
        "V2SEAM_C05_REDUCER_DETERMINISTIC_24_OF_24": all(
            row["deterministic_repeat_equal"] for row in seam_rows
        ),
        "V2SEAM_C06_GOOD_REAL_TUM_HAS_NONUNKNOWN_STATE": bool(
            set(tum_states) - {"UNKNOWN"}
        ),
        "V2SEAM_C07_BAD_ICL_FAILS_CLOSED_MORE_THAN_TUM": bool(
            icl_states["UNKNOWN"] > tum_states["UNKNOWN"]
        ),
        "V2SEAM_C08_FACTOR_ONLY_NO_TASK_OR_REDUCER_FIT": bool(
            identity["learned_final_task_head"] is False
            and identity["task_outcome_used"] is False
        ),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_multisource_v2_consumed_seam_result_v1",
        "status": "AG_R2_MULTISOURCE_V2_CONSUMED_SEAM_PASS_READY_FOR_FRESH_REAL"
        if passed
        else "AG_R2_MULTISOURCE_V2_CONSUMED_SEAM_FAIL",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "receipts": {
            "frozen_r10_result_sha256": EXPECTED_FROZEN_R10_SHA256,
            "icl_labels_sha256": EXPECTED_ICL_LABEL_SHA256,
            "tum_real_labels_sha256": EXPECTED_TUM_LABEL_SHA256,
            "metric_student_result_sha256": EXPECTED_METRIC_STUDENT_RESULT_SHA256,
            "metric_student_checkpoint_sha256": EXPECTED_METRIC_STUDENT_CHECKPOINT_SHA256,
            "scale_bank_result_sha256": EXPECTED_SCALE_BANK_RESULT_SHA256,
            "scale_bank_sha256": EXPECTED_SCALE_BANK_SHA256,
        },
        "feature_receipt": feature_receipt,
        "factor_checkpoint": factor_checkpoint,
        "factor_identity": identity,
        "uncertainty": {
            "local_shape_relative_sigma": LOCAL_SHAPE_RELATIVE_SIGMA,
            "metric_scale_relative_sigma_by_frame": scale_sigmas,
        },
        "aggregate_state_counts": dict(sorted(aggregate_states.items())),
        "aggregate_reason_counts": dict(sorted(aggregate_reasons.items())),
        "state_counts_by_role": {
            role: dict(sorted(values.items()))
            for role, values in by_role_states.items()
        },
        "valid_adapter_frame_count": len(structurally_valid),
        "gates": gates,
        "frames": seam_rows,
        "decision": {
            "v2_uncertainty_decomposition_mechanics_complete": passed,
            "fresh_source_claim": False,
            "third_real_source_opened": False,
            "mobile_or_product_claim": False,
            "next_action_if_pass": "Freeze this v2 recipe and open exactly one third checkpoint-unseen real parent once.",
            "next_action_if_fail": "Do not open the third source; freeze the negative and revisit the factor task.",
        },
    }
    hybrid.write_json(args.output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-r10-result", type=Path, default=FROZEN_R10_RESULT)
    parser.add_argument("--icl-label-result", type=Path, default=ICL_LABEL_RESULT)
    parser.add_argument("--tum-label-result", type=Path, default=TUM_LABEL_RESULT)
    parser.add_argument(
        "--metric-student-result", type=Path, default=METRIC_STUDENT_RESULT
    )
    parser.add_argument("--scale-bank-result", type=Path, default=SCALE_BANK_RESULT)
    parser.add_argument(
        "--baseline-result", type=Path, default=hybrid.DEFAULT_BASELINE_RESULT
    )
    parser.add_argument(
        "--depthart-source", type=Path, default=hybrid.DEFAULT_DEPTHART_SOURCE
    )
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=hybrid.DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument(
        "--depthart-extension", type=Path, default=hybrid.DEFAULT_DEPTHART_EXTENSION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in (
        "frozen_r10_result",
        "icl_label_result",
        "tum_label_result",
        "metric_student_result",
        "scale_bank_result",
        "baseline_result",
        "depthart_source",
        "depthart_checkpoint",
        "depthart_extension",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "valid_adapter_frame_count": result["valid_adapter_frame_count"],
                "aggregate_state_counts": result["aggregate_state_counts"],
                "state_counts_by_role": result["state_counts_by_role"],
                "aggregate_reason_counts": result["aggregate_reason_counts"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

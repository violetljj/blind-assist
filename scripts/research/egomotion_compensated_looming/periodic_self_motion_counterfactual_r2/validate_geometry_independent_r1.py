"""Independent R1 full G01-G14 geometry validator.

The validator imports only the already-independent R0 geometry math, never the
R0/R1 generator and never RCLE. It revalidates every gate on R1 evidence,
recomputes the revised rendered-target G13, and proves R0 immutability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import cv2
import numpy as np
import scipy

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_geometry_independent as base,
)


IMPLEMENTATION_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_R1"
)
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EVIDENCE = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r1"
)
R0_EVIDENCE = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r0"
)
DEFAULT_LOCK = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R1_2026-07-28.json"
)
AMENDMENT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GEOMETRY_SPEC_REPAIR_R1_2026-07-28.json"
)
R0_LOCK_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R0_2026-07-28.json"
)
EXPECTED_R0_RECEIPT_SHA256 = (
    "72e0b8e042be9eb6208389eb8d83e9e9e4ad28e54ec82f7064b5387cc1abd279"
)
LOG_1P20 = 0.1823215567939546


def _validate_r1_lock(
    lock: dict[str, Any], evidence: Path, errors: list[str]
) -> None:
    if lock.get("implementation_id") != IMPLEMENTATION_ID:
        errors.append("R1_LOCK_IMPLEMENTATION_ID")
    for field in (
        "formal_execution_authorized",
        "quality_calibration_authorized",
        "automatic_p2_authority",
    ):
        if lock.get(field) is not False:
            errors.append(f"R1_LOCK_AUTHORITY:{field}")
    for relative, expected in lock.get("source_sha256", {}).items():
        path = REPO_ROOT / relative
        if not path.is_file() or base.sha256_file(path) != expected:
            errors.append(f"R1_LOCK_SOURCE_HASH:{relative}")
    for name, expected in lock.get("evidence_sha256", {}).items():
        path = evidence / name
        if not path.is_file() or base.sha256_file(path) != expected:
            errors.append(f"R1_LOCK_EVIDENCE_HASH:{name}")
    expected_environment = lock.get("environment", {})
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
    }
    for field, value in actual.items():
        if expected_environment.get(field) != value:
            errors.append(f"R1_LOCK_ENVIRONMENT:{field}")
    predecessor = lock.get("immutable_r0", {})
    if predecessor.get("receipt_sha256") != EXPECTED_R0_RECEIPT_SHA256:
        errors.append("R1_LOCK_R0_RECEIPT_IDENTITY")
    if predecessor.get("terminal") != "INTERVENTION_NOT_EVALUABLE / HOLD_P1":
        errors.append("R1_LOCK_R0_TERMINAL")


def _validate_r0_immutability(
    r1_records: list[dict[str, Any]], errors: list[str]
) -> None:
    r0_receipt_path = R0_EVIDENCE / "independent_geometry_validation_receipt.json"
    if base.sha256_file(r0_receipt_path) != EXPECTED_R0_RECEIPT_SHA256:
        errors.append("R0_RECEIPT_HASH")
    else:
        receipt = base.load_json(r0_receipt_path)
        if (
            receipt.get("terminal") != "INTERVENTION_NOT_EVALUABLE"
            or receipt.get("state") != "HOLD_P1"
            or receipt.get("gate_pass_count") != 13
        ):
            errors.append("R0_TERMINAL_CONTENT")
    r0_records = base.load_jsonl(R0_EVIDENCE / "all_seed_geometry_manifest.jsonl")
    r0_main = {
        item["cluster_id"]: item
        for item in r0_records
        if item.get("record_type") == "main_cluster"
    }
    r1_main = {
        item["cluster_id"]: item
        for item in r1_records
        if item.get("record_type") == "main_cluster"
    }
    if set(r0_main) != set(r1_main) or len(r1_main) != 80:
        errors.append("R1_MAIN_IDENTITY_SET")
    else:
        for cluster_id in sorted(r0_main):
            if base.canonical_bytes(r0_main[cluster_id]) != base.canonical_bytes(
                r1_main[cluster_id]
            ):
                errors.append(f"R1_MAIN_RECORD_DRIFT:{cluster_id}")
    r0_guard = {
        item["cluster_id"]: item
        for item in r0_records
        if item.get("record_type") == "guardrail_cluster"
    }
    r1_guard = {
        item["cluster_id"]: item
        for item in r1_records
        if item.get("record_type") == "guardrail_cluster"
    }
    if set(r0_guard) != set(r1_guard) or len(r1_guard) != 8:
        errors.append("R1_GUARD_IDENTITY_SET")
        return
    for cluster_id in sorted(r0_guard):
        before = r0_guard[cluster_id]
        after = r1_guard[cluster_id]
        if int(before["numeric_seed_uint64"]) != int(after["numeric_seed_uint64"]):
            errors.append(f"R1_GUARD_SEED_DRIFT:{cluster_id}")
        before_arms = {item["arm_id"]: item for item in before["arms"]}
        after_arms = {item["arm_id"]: item for item in after["arms"]}
        if set(before_arms) != set(after_arms):
            errors.append(f"R1_GUARD_ARM_SET:{cluster_id}")
            continue
        for arm_id in sorted(before_arms):
            for field in ("trajectory_sha256", "trajectory"):
                if base.canonical_bytes(before_arms[arm_id][field]) != base.canonical_bytes(
                    after_arms[arm_id][field]
                ):
                    errors.append(
                        f"R1_GUARD_TRAJECTORY_DRIFT:{cluster_id}:{arm_id}:{field}"
                    )


def _target_visibility_and_measurements(
    scene: dict[str, Any], arm: dict[str, Any]
) -> dict[str, Any]:
    target = scene["designated_target"]
    target_id = int(target["object_id"])
    point = np.asarray([target["world_point_m"]], dtype=np.float64)
    objects = {
        int(item["object_id"]): item for item in scene["world"]["objects"]
    }
    if target_id not in objects:
        raise ValueError("TARGET_OBJECT_MISSING")
    target_object = objects[target_id]
    x0, x1, y0, y1 = target_object["bounds_xy_m"]
    on_mesh = bool(
        abs(float(point[0, 2]) - float(target_object["plane_z_m"])) <= 1e-12
        and x0 <= point[0, 0] <= x1
        and y0 <= point[0, 1] <= y1
    )
    depths = []
    radii = []
    visible = []
    translations = []
    rotations = []
    timestamps = []
    for pose in arm["trajectory"]:
        rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
        translation = np.asarray(pose["translation_m"], dtype=np.float64)
        uv, projected_depth = base.project(point, rotation, translation)
        rendered_depth, rendered_object, _ = base.raycast(
            scene, rotation, translation, uv
        )
        is_visible = bool(
            0.0 <= uv[0, 0] < base.WIDTH
            and 0.0 <= uv[0, 1] < base.HEIGHT
            and np.isfinite(rendered_depth[0])
            and int(rendered_object[0]) == target_id
            and abs(float(rendered_depth[0] - projected_depth[0])) <= 1e-7
        )
        visible.append(is_visible)
        depths.append(float(projected_depth[0]))
        radii.append(float(np.linalg.norm(uv[0] - base.K[:2, 2])))
        translations.append(translation)
        rotations.append(rotation)
        timestamps.append(float(pose["timestamp_s"]))
    depth_array = np.asarray(depths, dtype=np.float64)
    radius_array = np.asarray(radii, dtype=np.float64)
    inverse_increase = float(
        (1.0 / depth_array[-1]) / (1.0 / depth_array[0]) - 1.0
    )
    integrated_log_radial = float(
        math.log(radius_array[-1] / radius_array[0])
    )
    return {
        "on_mesh": on_mesh,
        "visible": np.asarray(visible, dtype=bool),
        "depth": depth_array,
        "radius": radius_array,
        "inverse_increase": inverse_increase,
        "integrated_log_radial": integrated_log_radial,
        "translation": np.asarray(translations, dtype=np.float64),
        "rotation": np.asarray(rotations, dtype=np.float64),
        "timestamp": np.asarray(timestamps, dtype=np.float64),
        "point": point,
    }


def gate_g13_r1(
    guards: list[dict[str, Any]], trajectories: dict[str, Any]
) -> dict[str, Any]:
    summaries = []
    failures = []
    for record in guards:
        block = str(record["block"])
        _, periodic_translation, periodic_rotation = base.pose_arrays(
            trajectories[block]
        )
        for arm in record["arms"]:
            arm_id = str(arm["arm_id"])
            try:
                measured = _target_visibility_and_measurements(
                    record["scene"], arm
                )
            except (KeyError, ValueError, ZeroDivisionError) as error:
                failures.append(
                    f"{record['cluster_id']}:{arm_id}:TARGET:{type(error).__name__}"
                )
                continue
            timestamps = measured["timestamp"]
            phase = (
                (timestamps - timestamps[0])
                / (timestamps[-1] - timestamps[0])
            )
            expected_approach = np.column_stack(
                (
                    np.zeros(base.FRAME_COUNT),
                    np.zeros(base.FRAME_COUNT),
                    0.8 * phase,
                )
            )
            if arm_id == "MONOTONIC_APPROACH":
                deperiodized_translation = measured["translation"]
                deperiodized_rotation = measured["rotation"]
            elif arm_id == "MONOTONIC_APPROACH_PLUS_PERIODIC":
                deperiodized_translation = (
                    measured["translation"] - periodic_translation
                )
                deperiodized_rotation = np.einsum(
                    "nij,njk->nik",
                    np.transpose(periodic_rotation, (0, 2, 1)),
                    measured["rotation"],
                )
            else:
                failures.append(
                    f"{record['cluster_id']}:{arm_id}:UNEXPECTED_ARM"
                )
                continue
            translation_error = float(
                np.max(
                    np.abs(deperiodized_translation - expected_approach)
                )
            )
            rotation_error = float(
                max(base.rotation_angle(item) for item in deperiodized_rotation)
            )
            deperiodized_depth = np.asarray(
                [
                    (
                        deperiodized_rotation[index].T
                        @ (
                            measured["point"]
                            - deperiodized_translation[index]
                        ).T
                    ).T[0, 2]
                    for index in range(base.FRAME_COUNT)
                ],
                dtype=np.float64,
            )
            monotonic = bool(np.all(np.diff(deperiodized_depth) <= 1e-12))
            passed = bool(
                measured["on_mesh"]
                and np.all(measured["visible"])
                and measured["inverse_increase"] >= 0.20
                and measured["integrated_log_radial"] >= LOG_1P20
                and translation_error <= 1e-12
                and rotation_error <= 1e-12
                and monotonic
            )
            if not passed:
                failures.append(f"{record['cluster_id']}:{arm_id}")
            summaries.append(
                {
                    "cluster_id": record["cluster_id"],
                    "arm_id": arm_id,
                    "target_on_rendered_mesh": measured["on_mesh"],
                    "persistent_visible_frame_count": int(
                        np.count_nonzero(measured["visible"])
                    ),
                    "inverse_depth_endpoint_increase": measured[
                        "inverse_increase"
                    ],
                    "integrated_endpoint_log_radial_expansion": measured[
                        "integrated_log_radial"
                    ],
                    "deperiodized_translation_max_error_m": translation_error,
                    "deperiodized_rotation_max_error_rad": rotation_error,
                    "deperiodized_depth_monotonic": monotonic,
                }
            )
    return {
        "id": "G13_MONOTONIC_APPROACH_TRUTH",
        "status": (
            "PASS" if len(summaries) == 16 and not failures else "FAIL"
        ),
        "sequence_count": len(summaries),
        "integrated_log_radial_gate": LOG_1P20,
        "failures": failures,
        "summaries": summaries,
    }


def validate(evidence: Path, lock_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    lock = base.load_json(lock_path)
    amendment = base.load_json(AMENDMENT_PATH)
    geometry_spec = base.load_json(base.GEOMETRY_PATH)
    runtime = base.load_json(evidence / "runtime_manifest.json")
    trajectories = base.load_json(evidence / "trajectory_manifest.json")
    fixtures = base.load_json(evidence / "analytic_fixture_ledger.json")
    replay = base.load_json(evidence / "deterministic_replay_ledger.json")
    samples = base.load_json(evidence / "projective_sample_ledger.json")
    producer = base.load_json(evidence / "generator_r1_receipt.json")
    records = base.load_jsonl(evidence / "all_seed_geometry_manifest.jsonl")
    _validate_r1_lock(lock, evidence, errors)
    _validate_r0_immutability(records, errors)
    if amendment.get("status") != "FROZEN_BEFORE_R1_IMPLEMENTATION_VALIDATION":
        errors.append("R1_AMENDMENT_STATUS")
    for field in (
        "formal_execution_authorized",
        "quality_calibration_authorized",
        "automatic_p2_authority",
    ):
        if amendment.get(field) is not False:
            errors.append(f"R1_AMENDMENT_AUTHORITY:{field}")
    if runtime.get("rcle_imported_or_executed") is not False:
        errors.append("R1_RUNTIME_RCLE_FIREWALL")
    for field in (
        "rcle_output_accessed_or_executed",
        "quality_strength_calibrated",
        "performance_preflight_run",
        "formal_sequences_run",
        "formal_execution_authorized",
    ):
        if producer.get(field) is not False:
            errors.append(f"R1_PRODUCER_BOUNDARY:{field}")
    main = [item for item in records if item.get("record_type") == "main_cluster"]
    guards = [
        item for item in records if item.get("record_type") == "guardrail_cluster"
    ]
    main_by_key = {(item["block"], int(item["ordinal"])): item for item in main}
    fixture_by_id = {item["id"]: item for item in fixtures["fixtures"]}
    gates: list[dict[str, Any]] = []
    gates.extend(base.gate_g01_g02(main, guards))
    gates.append(base.gate_g03(fixtures, samples, trajectories, main_by_key))
    gates.append(base.gate_g04(fixture_by_id, main))
    gates.append(
        base.gate_g05(
            fixture_by_id["PURE_TRANSLATION_LATERAL_MULTI_DEPTH"]
        )
    )
    gates.append(
        base.gate_g06(
            fixture_by_id["PURE_TRANSLATION_LATERAL_MULTI_DEPTH"],
            trajectories,
            main_by_key,
        )
    )
    gates.append(
        base.gate_g07(
            fixture_by_id["PURE_ROTATION_SHARED_BEARINGS_MULTI_DEPTH"]
        )
    )
    gates.append(base.gate_g08(main, trajectories))
    gates.extend(base.gates_g09_g10(trajectories))
    gates.append(base.gate_g11(main))
    gates.append(base.gate_g12(main))
    gates.append(gate_g13_r1(guards, trajectories))
    gates.append(base.gate_g14(replay))
    required = [gate["id"] for gate in geometry_spec["required_gates"]]
    if [gate["id"] for gate in gates] != required:
        errors.append("R1_GATE_ORDER_OR_IDENTITY")
    failed = [gate["id"] for gate in gates if gate["status"] != "PASS"]
    if errors or failed:
        status = "INVALID" if errors else "VALID_FAIL_CLOSED"
        terminal = "INTERVENTION_NOT_EVALUABLE"
        state = "HOLD_P1"
    else:
        status = "VALID"
        terminal = "GENERATOR_GEOMETRY_PASS"
        state = "EXECUTION_NOT_AUTHORIZED"
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p1_independent_geometry_r1_receipt.v1"
        ),
        "protocol_id": base.PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": status,
        "terminal": terminal,
        "state": state,
        "gate_pass_count": sum(
            gate["status"] == "PASS" for gate in gates
        ),
        "gate_required_count": 14,
        "failed_gates": failed,
        "errors": sorted(errors),
        "gates": gates,
        "immutable_r0": {
            "receipt_sha256": EXPECTED_R0_RECEIPT_SHA256,
            "terminal": "INTERVENTION_NOT_EVALUABLE / HOLD_P1",
        },
        "implementation_lock_sha256": base.sha256_file(lock_path),
        "amendment_sha256": base.sha256_file(AMENDMENT_PATH),
        "evidence_sha256": {
            path.name: base.sha256_file(path)
            for path in sorted(evidence.iterdir())
            if path.is_file()
            and path.name != "independent_geometry_validation_receipt.json"
        },
        "validator_source_sha256": base.sha256_file(Path(__file__)),
        "generator_imported": False,
        "rcle_output_accessed_or_executed": False,
        "quality_strength_calibrated": False,
        "performance_preflight_run": False,
        "formal_sequences_run": False,
        "formal_execution_authorized": False,
        "automatic_p2_authority": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = validate(args.evidence.resolve(), args.lock.resolve())
    except Exception as error:
        receipt = {
            "status": "INVALID",
            "terminal": "INTERVENTION_NOT_EVALUABLE",
            "state": "HOLD_P1",
            "errors": [f"{type(error).__name__}:{error}"],
            "formal_execution_authorized": False,
        }
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(base.canonical_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt.get("status") == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())

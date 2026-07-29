"""Independent, response-blind validator for the R3 leakage-localization preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0"
)
DOCS = Path("docs/research/rcle")
CONTRACT = DOCS / f"{TASK_ID}_CONTRACT_2026-07-29.json"
LOCK = DOCS / f"{TASK_ID}_IDENTITY_INPUT_LOCK_2026-07-29.json"
RECEIPT = DOCS / f"{TASK_ID}_INDEPENDENT_RECEIPT_2026-07-29.json"
DECISION = DOCS / f"{TASK_ID}_EXECUTION_ACTIVATION_DECISION_2026-07-29.json"
VALIDATOR = Path(
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "validate_r3_rotation_leakage_source_localization_contract_preflight_r0.py"
)
EXPECTED_LAYERS = [
    "INPUT_GEOMETRY",
    "ROTATION_WARP",
    "MASK_BOUNDARY",
    "SPARSE_LK_AND_TRACK_FILTERING",
    "LOCAL_AFFINE_AND_FINAL_AGGREGATION",
]
EXPECTED_ROUTES = {
    "LEAKAGE_ALREADY_PRESENT_IN_INPUT_GEOMETRY",
    "LEAKAGE_FIRST_VISIBLE_AT_WARP",
    "LEAKAGE_FIRST_VISIBLE_AT_MASK_BOUNDARY",
    "LEAKAGE_FIRST_VISIBLE_AT_FLOW",
    "LEAKAGE_FIRST_VISIBLE_AT_LOCAL_FIT",
    "MULTIPLE_SOURCES_NOT_SEPARABLE",
    "NOT_EVALUABLE",
}


class InvalidPreflight(ValueError):
    """Raised when a frozen preflight condition fails."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidPreflight(f"DUPLICATE_JSON_KEY:{key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            InvalidPreflight(f"NONFINITE_JSON:{token}")
        ),
    )
    if not isinstance(value, dict):
        raise InvalidPreflight(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def require(condition: bool, code: str) -> None:
    if not condition:
        raise InvalidPreflight(code)


def _verify_hash_binding(root: Path, binding: dict[str, Any], code: str) -> None:
    path = root / binding["path"]
    require(path.is_file(), f"{code}_MISSING")
    require(sha256_file(path) == binding["sha256"], f"{code}_HASH")


def validate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = root / CONTRACT
    lock_path = root / LOCK
    contract = load_json(contract_path)
    lock = load_json(lock_path)
    checks: dict[str, str] = {}

    require(contract["task_id"] == TASK_ID, "CONTRACT_TASK")
    require(lock["task_id"] == TASK_ID, "LOCK_TASK")
    require(
        contract["predecessor"]["scientific_status"] == "B_ORACLE_NOT_EVALUABLE",
        "PREDECESSOR_TERMINAL",
    )
    checks["predecessor_terminal"] = "PASS"

    require(
        contract["estimand"]["ordered_layers"] == EXPECTED_LAYERS,
        "LAYER_ORDER",
    )
    require(
        [item["layer"] for item in contract["observation_primitives"]]
        == EXPECTED_LAYERS,
        "PRIMITIVE_LAYER_ORDER",
    )
    require(
        set(contract["per_cluster_routing"]["routes"]) == EXPECTED_ROUTES,
        "ROUTE_SET",
    )
    checks["estimand_layers_routes"] = "PASS"

    counts = lock["counts"]
    require(
        counts
        == {
            "clusters": 8,
            "sequences": 8,
            "frames_per_sequence": 602,
            "pairs_per_sequence": 601,
            "pair_records_are_longitudinal_repeats": True,
            "analysis_unit": "cluster",
        },
        "LOCK_COUNTS",
    )
    clusters = lock["clusters"]
    require(len(clusters) == 8, "CLUSTER_COUNT")
    require(len({item["cluster_id"] for item in clusters}) == 8, "CLUSTER_COLLISION")
    require(
        len({item["sequence_id"] for item in clusters}) == 8,
        "SEQUENCE_COLLISION",
    )
    require(
        all(
            item["sequence_id"].endswith(
                "__EGO_ROTATION_STATIC_SCENE__CLEAN"
            )
            for item in clusters
        ),
        "NON_ROTATION_IDENTITY",
    )
    checks["sealed_identity_input_lock"] = "PASS"

    for name, binding in lock["bindings"].items():
        _verify_hash_binding(root, binding, f"LOCK_BINDING_{name.upper()}")
    closeout = load_json(root / lock["bindings"]["stage_b_closeout"]["path"])
    require(
        closeout["scientific_status"] == "B_ORACLE_NOT_EVALUABLE",
        "CLOSEOUT_STATUS",
    )
    require(
        closeout["routing"]["retry_replacement_or_reseed_authorized"] is False,
        "CLOSEOUT_RETRY_AUTHORITY",
    )
    checks["stage_b_closeout_binding"] = "PASS"

    geometry = load_json(
        root / lock["bindings"]["stage_b_geometry_manifest"]["path"]
    )
    identity = load_json(root / lock["bindings"]["stage_b_identity_lock"]["path"])
    geometry_by_cluster = {
        item["cluster_id"]: item for item in geometry["clusters"]
    }
    identity_by_cluster = {
        item["cluster_id"]: item for item in identity["clusters"]
    }
    for sealed in clusters:
        cluster_id = sealed["cluster_id"]
        require(cluster_id in geometry_by_cluster, f"GEOMETRY_CLUSTER:{cluster_id}")
        require(cluster_id in identity_by_cluster, f"IDENTITY_CLUSTER:{cluster_id}")
        geometry_cluster = geometry_by_cluster[cluster_id]
        arm = next(
            (
                item
                for item in geometry_cluster["arms"]
                if item["arm"] == "EGO_ROTATION_STATIC_SCENE"
            ),
            None,
        )
        require(arm is not None, f"ROTATION_ARM:{cluster_id}")
        require(
            sealed["sequence_id"]
            in identity_by_cluster[cluster_id]["sequence_ids"],
            f"SEQUENCE_ID:{cluster_id}",
        )
        require(
            sealed["scene_geometry_sha256"]
            == geometry_cluster["base_scene_sha256"],
            f"SCENE_HASH:{cluster_id}",
        )
        for field in ("pose_sha256", "render_input_sha256"):
            require(
                sealed[field] == arm[field],
                f"{field.upper()}:{cluster_id}",
            )
        require(arm["frame_count"] == 602, f"FRAME_COUNT:{cluster_id}")
    checks["geometry_identity_crosscheck"] = "PASS"

    for name, binding in contract["frozen_bindings"].items():
        if name == "identity_input_lock":
            continue
        _verify_hash_binding(root, binding, f"CONTRACT_BINDING_{name.upper()}")
    checks["r3_and_contract_bindings"] = "PASS"

    numeric = contract["coordinate_numeric_contract"]
    require(numeric["image"]["axes"] == "+x right, +y down", "IMAGE_AXES")
    require(numeric["pose"]["translation_unit"] == "metre", "POSE_UNIT")
    require(
        numeric["warp"]["image_interpolation"] == "INTER_LINEAR"
        and numeric["warp"]["mask_interpolation"] == "INTER_NEAREST"
        and numeric["warp"]["border_mode"] == "BORDER_CONSTANT",
        "WARP_CONTRACT",
    )
    require(
        numeric["flow"]["point_dtype"] == "float32"
        and numeric["audit"]["geometry_and_recomputation_dtype"] == "float64",
        "NUMERIC_REPRESENTATION",
    )
    checks["coordinates_units_warp_numeric"] = "PASS"

    final = contract["unchanged_r3_parameters"]["final"]
    require(final["rotation_absolute_leakage_boundary_per_s"] == 0.01, "BOUNDARY")
    require(final["fixed_pair_denominator"] == 601, "PAIR_DENOMINATOR")
    require(final["minimum_evaluable_pair_fraction"] == 0.75, "COVERAGE")
    require(
        contract["aggregation_and_coverage"]["pair_absolute"]
        == "median(abs(evaluable common-cell signed expansion))",
        "ABSOLUTE_REDUCTION",
    )
    require(
        "abs(median(signed)) is forbidden"
        in contract["estimand"]["signed_absolute_separation"],
        "SIGNED_ABSOLUTE_SEPARATION",
    )
    checks["boundary_aggregation_coverage"] = "PASS"

    resource = contract["resource_gate"]
    require(
        resource["launch_and_refill_minimum_available_ram_bytes"] == 6 * 1024**3,
        "MEMORY_6GIB",
    )
    require(
        resource["in_flight_emergency_floor_bytes"] == 4 * 1024**3,
        "MEMORY_4GIB_FLOOR",
    )
    require(resource["workers"] == 4, "WORKERS")
    checks["resource_gate_6gib"] = "PASS"

    require(
        lock["stage_b_response_payload_read_during_preflight"] is False
        and lock["source_localization_workload_run"] is False
        and lock["execution_authorized"] is False,
        "LOCK_AUTHORITY",
    )
    require(
        all(value is False for value in contract["forbidden"].values()),
        "FORBIDDEN_AUTHORITY",
    )
    require(
        contract["activation"]["preflight_execution_authorized"] is False
        and contract["activation"]["future_single_variable_repair_authorized"]
        is False,
        "ACTIVATION_AUTHORITY",
    )
    checks["firewall_and_authority"] = "PASS"

    validator_path = root / VALIDATOR
    receipt = {
        "schema": "rcle.r3_rotation_leakage_source_localization.independent_preflight_receipt.v1",
        "protocol_id": contract["protocol_id"],
        "task_id": TASK_ID,
        "date": "2026-07-29",
        "contract_sha256": sha256_file(contract_path),
        "identity_input_lock_sha256": sha256_file(lock_path),
        "validator_source_path": VALIDATOR.as_posix(),
        "validator_source_sha256": sha256_file(validator_path),
        "checks": checks,
        "check_count": len(checks),
        "stage_b_response_payload_read": False,
        "source_localization_workload_run": False,
        "formal_480_plus_16_consumed": False,
        "protocol_status": "VALID",
        "execution_authorized": False,
        "terminal": "CONTRACT_PREFLIGHT_VALID / EXECUTION_NOT_ACTIVATED",
    }
    decision = {
        "schema": "rcle.r3_rotation_leakage_source_localization.execution_activation_decision.v1",
        "protocol_id": contract["protocol_id"],
        "task_id": TASK_ID,
        "date": "2026-07-29",
        "contract_sha256": receipt["contract_sha256"],
        "identity_input_lock_sha256": receipt["identity_input_lock_sha256"],
        "independent_validator_source_sha256": receipt["validator_source_sha256"],
        "preflight_status": receipt["protocol_status"],
        "decision": "HOLD_ROTATION_LEAKAGE_LOCALIZATION_EXECUTION_PENDING_SEPARATE_ACTIVATION",
        "execution_authorized": False,
        "stage_b_rerun_authorized": False,
        "r3_modification_authorized": False,
        "single_variable_repair_authorized": False,
        "formal_480_plus_16_authorized": False,
        "terminal": "PREFLIGHT_VALID / EXECUTION_NOT_ACTIVATED / FUTURE_REPAIR_NOT_AUTHORIZED",
    }
    return receipt, decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--decision", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    receipt, decision = validate(root)
    if (args.receipt is None) != (args.decision is None):
        raise InvalidPreflight("RECEIPT_AND_DECISION_MUST_BE_PAIRED")
    if args.receipt is not None:
        write_exclusive(args.receipt.resolve(), receipt)
        decision["independent_receipt_sha256"] = sha256_file(
            args.receipt.resolve()
        )
        write_exclusive(args.decision.resolve(), decision)
    print(
        json.dumps(
            {
                "terminal": receipt["terminal"],
                "check_count": receipt["check_count"],
                "execution_authorized": receipt["execution_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

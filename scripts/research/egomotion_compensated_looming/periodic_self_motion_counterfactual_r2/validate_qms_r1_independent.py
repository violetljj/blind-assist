"""Independent validator for the response-blind QMS-R1 new-CAL evidence.

This module deliberately uses only the Python standard library.  It does not
import the QMS producer, its rendering operator, P4, or any R3 implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any


QMS_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_MANIPULATION_SUCCESSOR_R1"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
FRAME_POSITIONS = (
    0, 40, 80, 120, 160, 200, 240, 280,
    320, 360, 400, 440, 480, 520, 560, 601,
)
RATIO_RANGE = (0.10, 0.20)
SHA256_KEYS = (
    "clean_rgb_sha256",
    "low_texture_rgb_sha256",
    "valid_mask_sha256",
    "object_id_sha256",
)
IDENTITY_KEYS = {
    "block",
    "cluster_id",
    "cohort",
    "motion",
    "numeric_seed_uint64",
    "ordinal",
    "scene_geometry_sha256",
}
LEDGER_KEYS = IDENTITY_KEYS | {
    "frame_positions",
    "frame_rows",
    "sequence_medians",
    "sequence_pass",
}
FRAME_KEYS = {
    "clean_rgb_sha256",
    "descriptive_full_frame_gradient_density_ratio",
    "frame_index",
    "low_texture_rgb_sha256",
    "material_count",
    "material_residual_ratio",
    "object_id_sha256",
    "prequantization_residual_max_abs_error",
    "scene_geometry_sha256",
    "structure_contrast_ratio",
    "structure_edge_count",
    "valid_mask_sha256",
}
MEDIAN_KEYS = {
    "descriptive_full_frame_gradient_density_ratio",
    "material_residual_ratio",
    "structure_contrast_ratio",
}
EXPECTED_OPERATOR = {
    "alpha": 0.15,
    "clean_modulation_formula": "0.65+0.35*checker",
    "domain": "PREQUANTIZATION_LINEAR_RGB",
    "low_modulation_formula": "0.825+0.15*(clean-0.825)",
    "material_mean_modulation": 0.825,
    "operator_id": "QMS_R1_MATERIAL_RESIDUAL_CONTRACTION",
    "pairing": "ONE_RAYCAST_SHARED_GEOMETRY",
    "psf_none": True,
}
EXPECTED_GATES = {
    "full_frame_gradient_density_role": "DESCRIPTIVE_ONLY",
    "geometry_identity": "EXACT_SHARED_RAYCAST_PER_PAIRED_FRAME",
    "material_mean_identity": "EXACT_PREQUANTIZATION_BY_OPERATOR",
    "material_residual_ratio_inclusive": [0.1, 0.2],
    "structure_contrast_ratio_role": "DESCRIPTIVE_ONLY",
}
EXPECTED_FIREWALL = {
    "predecessor_outcome_read": False,
    "r3_imported_or_executed": False,
    "r3_outcome_read": False,
    "sequence16_android_realtime": False,
}
EXPECTED_BINDING_PATHS = {
    "operator_sha256": (
        "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/"
        "material_residual_contraction_r1.py"
    ),
    "qualification_sha256": (
        "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/qms_r1_qualification.py"
    ),
    "generator_sha256": (
        "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/generator_geometry.py"
    ),
    "quality_metrics_sha256": (
        "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/quality_interventions_r0.py"
    ),
    "source_manifest_sha256": (
        "artifacts.local/evidence/"
        "rcle_periodic_self_motion_counterfactual_r2/"
        "p1_geometry_r2_keyset_repair_r0/all_seed_geometry_manifest.jsonl"
    ),
    "trajectory_manifest_sha256": (
        "artifacts.local/evidence/"
        "rcle_periodic_self_motion_counterfactual_r2/"
        "p1_geometry_r2_keyset_repair_r0/trajectory_manifest.json"
    ),
}


class InvalidQmsR1Independent(ValueError):
    """Raised when any independent QMS-R1 invariant fails."""


def _fail(code: str) -> None:
    raise InvalidQmsR1Independent(code)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise InvalidQmsR1Independent(f"BOUND_FILE:{path.name}") from error
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("NON_NUMERIC_METRIC")
    result = float(value)
    if not math.isfinite(result):
        _fail("NONFINITE_METRIC")
    return result


def _median(values: list[float]) -> float:
    if len(values) != len(FRAME_POSITIONS):
        _fail("FRAME_METRIC_CARDINALITY")
    return float(statistics.median(values))


def _seed(block: str, ordinal: int) -> tuple[str, str, int]:
    token = f"{QMS_ID}|CAL|{block}|{ordinal:02d}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    numeric = int.from_bytes(bytes.fromhex(digest)[:8], "big")
    return token, digest, numeric


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidQmsR1Independent(f"JSON:{path.name}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if any(not line.strip() for line in lines):
            _fail("LEDGER_BLANK_LINE")
        rows = [json.loads(line) for line in lines]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidQmsR1Independent(f"JSONL:{path.name}") from error
    if not all(isinstance(row, dict) for row in rows):
        _fail("LEDGER_ROW_TYPE")
    return rows


def _expected_identity(block: str, ordinal: int, motion: str) -> dict[str, Any]:
    _, _, numeric_seed = _seed(block, ordinal)
    return {
        "block": block,
        "cluster_id": f"QMS_R1_{block}_CAL_{ordinal:02d}",
        "cohort": "QMS_R1_DISJOINT_CAL",
        "motion": motion,
        "numeric_seed_uint64": numeric_seed,
        "ordinal": ordinal,
    }


def _validate_identity_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 32:
        _fail("IDENTITY_CARDINALITY")
    expected_order = [
        (block, ordinal, motion)
        for block in BLOCKS
        for ordinal in range(4)
        for motion in MOTIONS
    ]
    scene_hashes: dict[tuple[str, int], str] = {}
    numeric_seeds: dict[tuple[str, int], int] = {}
    for row, (block, ordinal, motion) in zip(rows, expected_order, strict=True):
        if not isinstance(row, dict) or set(row) != IDENTITY_KEYS:
            _fail("IDENTITY_FIELDS")
        expected = _expected_identity(block, ordinal, motion)
        if any(row[key] != value for key, value in expected.items()):
            _fail("IDENTITY_TOKEN_OR_SEED")
        if not _is_sha256(row["scene_geometry_sha256"]):
            _fail("IDENTITY_SCENE_HASH")
        key = (block, ordinal)
        if key in scene_hashes and scene_hashes[key] != row["scene_geometry_sha256"]:
            _fail("PAIRED_SCENE_IDENTITY")
        if key in numeric_seeds and numeric_seeds[key] != row["numeric_seed_uint64"]:
            _fail("PAIRED_SEED_IDENTITY")
        scene_hashes[key] = row["scene_geometry_sha256"]
        numeric_seeds[key] = row["numeric_seed_uint64"]
    if len(set(numeric_seeds.values())) != 16:
        _fail("SEED_UNIQUENESS")
    return rows


def _validate_frame(
    frame: Any,
    frame_index: int,
    scene_hash: str,
) -> tuple[float, float, float]:
    if not isinstance(frame, dict) or set(frame) != FRAME_KEYS:
        _fail("FRAME_FIELDS")
    if frame["frame_index"] != frame_index:
        _fail("FRAME_ORDER")
    for key in SHA256_KEYS:
        if not _is_sha256(frame[key]):
            _fail("FRAME_HASH_FIELD")
    if frame["clean_rgb_sha256"] == frame["low_texture_rgb_sha256"]:
        _fail("CLEAN_LOW_HASH_IDENTITY")
    if frame["scene_geometry_sha256"] != scene_hash:
        _fail("FRAME_SCENE_IDENTITY")
    if (
        isinstance(frame["material_count"], bool)
        or not isinstance(frame["material_count"], int)
        or frame["material_count"] < 9
    ):
        _fail("MATERIAL_COUNT")
    if (
        isinstance(frame["structure_edge_count"], bool)
        or not isinstance(frame["structure_edge_count"], int)
        or frame["structure_edge_count"] < 32
    ):
        _fail("STRUCTURE_EDGE_COUNT")
    prequant = _finite_number(frame["prequantization_residual_max_abs_error"])
    if prequant != 0.0:
        _fail("PREQUANTIZATION_IDENTITY")
    residual = _finite_number(frame["material_residual_ratio"])
    structure = _finite_number(frame["structure_contrast_ratio"])
    descriptive = _finite_number(
        frame["descriptive_full_frame_gradient_density_ratio"]
    )
    if residual < 0.0 or structure < 0.0 or descriptive < 0.0:
        _fail("NEGATIVE_METRIC")
    return residual, structure, descriptive


def _validate_ledger(
    rows: list[dict[str, Any]],
    identities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(rows) != 32:
        _fail("SEQUENCE_CARDINALITY")
    validated = []
    for row, identity in zip(rows, identities, strict=True):
        if set(row) != LEDGER_KEYS:
            _fail("SEQUENCE_FIELDS")
        for key in IDENTITY_KEYS:
            if row[key] != identity[key]:
                _fail("LEDGER_IDENTITY")
        if row["frame_positions"] != list(FRAME_POSITIONS):
            _fail("FRAME_POSITIONS")
        frame_rows = row["frame_rows"]
        if not isinstance(frame_rows, list) or len(frame_rows) != 16:
            _fail("FRAME_CARDINALITY")
        metrics = [
            _validate_frame(frame, position, row["scene_geometry_sha256"])
            for frame, position in zip(frame_rows, FRAME_POSITIONS, strict=True)
        ]
        medians = row["sequence_medians"]
        if not isinstance(medians, dict) or set(medians) != MEDIAN_KEYS:
            _fail("MEDIAN_FIELDS")
        recomputed = {
            "material_residual_ratio": _median([item[0] for item in metrics]),
            "structure_contrast_ratio": _median([item[1] for item in metrics]),
            "descriptive_full_frame_gradient_density_ratio": _median(
                [item[2] for item in metrics]
            ),
        }
        if medians != recomputed:
            _fail("SEQUENCE_MEDIANS")
        residual = recomputed["material_residual_ratio"]
        if not RATIO_RANGE[0] <= residual <= RATIO_RANGE[1]:
            _fail("MATERIAL_RESIDUAL_GATE")
        if row["sequence_pass"] is not True:
            _fail("SEQUENCE_PASS")
        validated.append(row)
    return validated


def _subgroups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for block in BLOCKS:
        for motion in MOTIONS:
            selected = [
                row for row in rows
                if row["block"] == block and row["motion"] == motion
            ]
            if len(selected) != 4:
                _fail("SUBGROUP_CARDINALITY")
            pass_count = sum(row["sequence_pass"] is True for row in selected)
            if pass_count != 4:
                _fail("SUBGROUP_NOT_FOUR_OF_FOUR")
            result.append(
                {
                    "block": block,
                    "motion": motion,
                    "pass_count": 4,
                    "sequence_count": 4,
                    "subgroup_pass": True,
                }
            )
    return result


def _validate_receipt_header(receipt: Any) -> None:
    if not isinstance(receipt, dict):
        _fail("RECEIPT_TYPE")
    expected = {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1.v1",
        "protocol_id": QMS_ID,
        "mode": "new-cal",
        "terminal": "QMS_R1_MATERIAL_OPERATOR_CAL_QUALIFIED",
        "qualified": True,
        "estimand": (
            "effect_of_fixed_global_material_albedo_texture_residual_"
            "contraction_alpha_0p15"
        ),
        "operator": EXPECTED_OPERATOR,
        "gates": EXPECTED_GATES,
        "counts": {"frame_states": 512, "sequences": 32},
        "firewall": EXPECTED_FIREWALL,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            _fail(f"RECEIPT_{key.upper()}")
    try:
        created = datetime.fromisoformat(receipt["created_at_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidQmsR1Independent("RECEIPT_TIMESTAMP") from error
    if created.tzinfo is None or created.utcoffset() is None:
        _fail("RECEIPT_TIMESTAMP")


def _validate_bindings(
    repo_root: Path,
    evidence_dir: Path,
    receipt: dict[str, Any],
) -> dict[str, str]:
    bindings = receipt.get("bindings")
    expected_keys = set(EXPECTED_BINDING_PATHS) | {
        "identity_manifest_sha256",
        "response_blind_ledger_sha256",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_keys:
        _fail("BINDING_FIELDS")
    actual = {
        "identity_manifest_sha256": _sha256_file(
            evidence_dir / "identity_manifest.json"
        ),
        "response_blind_ledger_sha256": _sha256_file(
            evidence_dir / "response_blind_ledger.jsonl"
        ),
    }
    actual.update(
        {
            key: _sha256_file(repo_root / relative)
            for key, relative in EXPECTED_BINDING_PATHS.items()
        }
    )
    for key, digest in actual.items():
        if bindings.get(key) != digest:
            _fail(f"BINDING_{key.upper()}")
    return actual


def validate(repo_root: Path, evidence_dir: Path) -> dict[str, Any]:
    """Validate a QMS-R1 new-CAL directory without importing its implementation."""

    root = repo_root.resolve()
    source = evidence_dir.resolve()
    receipt_path = source / "receipt.json"
    identity_path = source / "identity_manifest.json"
    ledger_path = source / "response_blind_ledger.jsonl"
    receipt = _read_json(receipt_path)
    _validate_receipt_header(receipt)
    bindings = _validate_bindings(root, source, receipt)
    identities = _validate_identity_rows(_read_json(identity_path))
    rows = _validate_ledger(_read_jsonl(ledger_path), identities)
    summaries = _subgroups(rows)
    if receipt.get("subgroups") != summaries:
        _fail("SUBGROUP_RECEIPT")
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "qms_r1_independent_validation.v1"
        ),
        "protocol_id": QMS_ID,
        "validation": "VALID",
        "terminal": "QMS_R1_INDEPENDENT_VALIDATION_VALID",
        "source_terminal": "QMS_R1_MATERIAL_OPERATOR_CAL_QUALIFIED",
        "counts": {"sequences": 32, "frame_states": 512, "subgroups": 8},
        "identity_manifest_sha256": bindings["identity_manifest_sha256"],
        "response_blind_ledger_sha256": bindings[
            "response_blind_ledger_sha256"
        ],
        "source_receipt_sha256": _sha256_file(receipt_path),
        "validator_sha256": _sha256_file(Path(__file__)),
        "firewall": {
            "producer_imported": False,
            "operator_imported": False,
            "p4_imported": False,
            "r3_imported_or_executed": False,
            "r3_outcome_read": False,
        },
    }


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Write one canonical JSON receipt and refuse to replace any existing file."""

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise InvalidQmsR1Independent(
            f"OUTPUT_EXISTS:{destination.name}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    arguments = parser.parse_args()
    result = validate(arguments.repo_root, arguments.evidence_dir)
    write_exclusive(arguments.receipt_out, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

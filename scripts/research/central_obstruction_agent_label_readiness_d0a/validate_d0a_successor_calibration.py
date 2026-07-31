"""Validate the D0-A successor fixed-clip calibration and emit its terminal."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .fixed_clip_units import FixedClipUnitError, compare_observation_reviews, validate_review_envelope
from .freeze_input_universe import canonical_bytes, sha256_file


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
DEFAULT_PROTOCOL = REPO_ROOT / "docs/research/dual-loop/CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_SUCCESSOR_PROTOCOL_2026-07-31.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a-successor-r0"
RESULT_NAME = "calibration-result.json"
VALIDATION_NAME = "calibration-validation.json"


class SuccessorValidationError(ValueError):
    """Raised when the successor input or review envelope is invalid."""


def load_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SuccessorValidationError(f"{where}: cannot read JSON: {error}") from error
    if not isinstance(value, dict):
        raise SuccessorValidationError(f"{where}: expected JSON object")
    return value


def relative_posix(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise SuccessorValidationError(f"path escapes repository: {path}") from error


def validate_input_manifest(
    *,
    repo_root: Path,
    protocol_path: Path,
    output_root: Path,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise SuccessorValidationError("successor manifest protocol id mismatch")
    if manifest.get("evidence_instance") != protocol["evidence_instance"]:
        raise SuccessorValidationError("successor manifest evidence instance mismatch")
    if manifest.get("candidate_output_access") is not False or manifest.get("labels_generated") is not False:
        raise SuccessorValidationError("successor input firewall is open")
    if manifest.get("natural_event_grouping_used") is not False:
        raise SuccessorValidationError("successor manifest permits natural-event grouping")
    if manifest.get("analysis_unit", {}).get("kind") != "FIXED_CLIP":
        raise SuccessorValidationError("successor analysis unit is not FIXED_CLIP")
    observations = manifest.get("observations")
    units = manifest.get("fixed_units")
    if not isinstance(observations, list) or len(observations) != 24:
        raise SuccessorValidationError("successor observation count is not 24")
    if not isinstance(units, list) or len(units) != 6:
        raise SuccessorValidationError("successor fixed-clip count is not 6")
    if manifest.get("calibration_source_count") != 3:
        raise SuccessorValidationError("successor source count is not 3")
    if manifest.get("freshness", {}).get("burned_d0a1_source_overlap_count") != 0:
        raise SuccessorValidationError("successor input overlaps burned D0-A1 sources")
    if manifest.get("freshness", {}).get("production_frame_overlap_count") != 0:
        raise SuccessorValidationError("successor input overlaps a production frame")
    protocol_sha = sha256_file(protocol_path)
    if manifest.get("protocol", {}).get("sha256") != protocol_sha:
        raise SuccessorValidationError("successor protocol binding drifted")
    if receipt.get("protocol_sha256") != protocol_sha:
        raise SuccessorValidationError("successor receipt protocol binding drifted")
    manifest_path = output_root / "calibration-input-manifest.json"
    if receipt.get("output_manifest", {}).get("sha256") != sha256_file(manifest_path):
        raise SuccessorValidationError("successor input manifest receipt hash mismatch")

    expected_keys: set[tuple[str, int]] = set()
    for row in observations:
        if not isinstance(row, dict):
            raise SuccessorValidationError("successor observation row is invalid")
        key = (row.get("unit_id"), row.get("slot_ordinal"))
        if key in expected_keys:
            raise SuccessorValidationError(f"successor observation key duplicated: {key}")
        expected_keys.add(key)
        if row.get("claim_critical") is not True:
            raise SuccessorValidationError(f"successor observation is not claim-critical: {key}")
        if "label" in row:
            raise SuccessorValidationError("labels must not be present in the frozen input manifest")
        review_image = row.get("review_image_path")
        if not isinstance(review_image, str):
            raise SuccessorValidationError(f"review image path is missing: {key}")
        image_path = repo_root / review_image
        if not image_path.is_file() or sha256_file(image_path) != row.get("review_image_sha256"):
            raise SuccessorValidationError(f"review image hash mismatch: {key}")

    flattened_keys = {
        (observation.get("unit_id"), observation.get("slot_ordinal"))
        for unit in units
        for observation in unit.get("observations", [])
    }
    if flattened_keys != expected_keys:
        raise SuccessorValidationError("fixed-unit observations do not match manifest observations")


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise SuccessorValidationError(f"refusing to overwrite formal output: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_bytes(canonical_bytes(payload))
    os.replace(temporary, path)


def validate_calibration(
    *,
    repo_root: Path,
    protocol_path: Path,
    output_root: Path,
    primary_path: Path,
    isolated_path: Path,
) -> tuple[Path, Path]:
    protocol = load_json(protocol_path, where="successor protocol")
    manifest_path = output_root / "calibration-input-manifest.json"
    receipt_path = output_root / "calibration-input-receipt.json"
    manifest = load_json(manifest_path, where="successor input manifest")
    receipt = load_json(receipt_path, where="successor input receipt")
    validate_input_manifest(
        repo_root=repo_root,
        protocol_path=protocol_path,
        output_root=output_root,
        protocol=protocol,
        manifest=manifest,
        receipt=receipt,
    )
    primary = load_json(primary_path, where="primary review")
    isolated = load_json(isolated_path, where="isolated review")
    validate_review_envelope(primary, expected_protocol_id=protocol["protocol_id"], where="primary review")
    validate_review_envelope(isolated, expected_protocol_id=protocol["protocol_id"], where="isolated review")
    if primary.get("evidence_instance") != protocol["evidence_instance"]:
        raise SuccessorValidationError("primary review evidence instance mismatch")
    if isolated.get("evidence_instance") != protocol["evidence_instance"]:
        raise SuccessorValidationError("isolated review evidence instance mismatch")
    if primary.get("review_context") == isolated.get("review_context"):
        raise SuccessorValidationError("primary and isolated reviews must use distinct contexts")
    try:
        result = compare_observation_reviews(
            manifest,
            primary,
            isolated,
            gates=protocol["gates"],
        )
    except FixedClipUnitError as error:
        raise SuccessorValidationError(str(error)) from error
    result.update(
        {
            "protocol_sha256": sha256_file(protocol_path),
            "input_manifest_sha256": sha256_file(manifest_path),
            "primary_review_sha256": sha256_file(primary_path),
            "isolated_review_sha256": sha256_file(isolated_path),
        }
    )
    validation = {
        "schema_version": "blindassist.central_obstruction_d0a_successor_calibration_validation.v1",
        "protocol_id": protocol["protocol_id"],
        "evidence_instance": protocol["evidence_instance"],
        "status": "VALID",
        "terminal": result["decision"]["terminal"],
        "input_manifest_sha256": sha256_file(manifest_path),
        "primary_review_sha256": sha256_file(primary_path),
        "isolated_review_sha256": sha256_file(isolated_path),
        "natural_event_grouping_used": False,
        "third_agent_adjudication_used": False,
        "d0a2_authorized": False,
        "d0a3_authorized": False,
        "d0a4_authorized": False,
        "central_obstruction_auxiliary_feature_only": result["decision"]["central_obstruction_role"]
        == "AUXILIARY_FEATURE_ONLY",
        "gate_results": result["gate_results"],
    }
    result_path = output_root / RESULT_NAME
    validation_path = output_root / VALIDATION_NAME
    write_json_once(result_path, result)
    write_json_once(validation_path, validation)
    return result_path, validation_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--primary-review", type=Path)
    parser.add_argument("--isolated-review", type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    primary_path = (args.primary_review or output_root / "primary-review.json").resolve()
    isolated_path = (args.isolated_review or output_root / "isolated-review.json").resolve()
    result_path, validation_path = validate_calibration(
        repo_root=args.repo_root.resolve(),
        protocol_path=args.protocol.resolve(),
        output_root=output_root,
        primary_path=primary_path,
        isolated_path=isolated_path,
    )
    print(
        json.dumps(
            {
                "status": "VALID",
                "result": relative_posix(result_path, args.repo_root.resolve()),
                "result_sha256": sha256_file(result_path),
                "validation": relative_posix(validation_path, args.repo_root.resolve()),
                "validation_sha256": sha256_file(validation_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

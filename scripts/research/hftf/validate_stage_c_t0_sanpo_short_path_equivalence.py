#!/usr/bin/env python3
"""Offline equivalence validator for the future T0 short-path transport.

The validator proves that a newly transported, already-consumed Development
package is byte-equivalent to the frozen canonical package at every selected
frame.  It has no network code and grants no acquisition, fresh-evaluation,
training, mainline, or application authority.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_hftf_stage_c_t0_sanpo_short_path_equivalence"
READY = "T0_SANPO_SHORT_PATH_CONSUMED_PACKAGE_EQUIVALENT"
NOT_EVALUABLE = (
    "T0_SANPO_SHORT_PATH_TRANSPORT_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT"
)
CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_t0_consumed_development_transport_contract"
)
CONTRACT_STATUS = "FROZEN_BEFORE_T0_CONSUMED_DEVELOPMENT_SOURCE_OPEN"
EXECUTION_ROLE = "outcome_open_development_transport_canary"
TRANSPORT_SCHEMA = "blindassist_hftf_stage_c_t0_sanpo_short_path_transport"
TRANSPORT_READY = "T0_SANPO_SHORT_PATH_TRANSPORT_READY"
CANARY_SCHEMA = "blindassist_hftf_stage_c_t0_sanpo_short_path_canary"
CANARY_READY = "T0_SANPO_SHORT_PATH_FILESYSTEM_CANARY_READY"
PREFLIGHT_READY = "T0_SANPO_SHORT_PATH_PREFLIGHT_READY"
REPLAY_SCHEMA = "blindassist_sanpo_synthetic_replay_v1"
MAX_CONTENT_PATH_EXCLUSIVE = 240

IMPLEMENTATIONS = {
    "acquirer": (
        "scripts/research/hftf/"
        "acquire_stage_c_t0_sanpo_short_path_transport.py"
    ),
    "filesystem_canary": (
        "scripts/research/hftf/"
        "run_stage_c_t0_sanpo_short_path_canary.py"
    ),
    "equivalence_validator": (
        "scripts/research/hftf/"
        "validate_stage_c_t0_sanpo_short_path_equivalence.py"
    ),
}
METADATA_PATHS = {
    "description": "source_metadata/source_session_description.json",
    "labelmap": "source_metadata/source_labelmap.json",
    "annotation_types": "source_metadata/source_annotation_types.json",
    "camera_poses": "source_metadata/camera_poses.csv",
}
FRAME_INVENTORY_KEYS = {
    "rgb": ("image_path", "image_sha256", "rgb"),
    "masks": ("source_mask_path", "source_mask_sha256", "panoptic_mask"),
    "depth": ("source_depth_path", "source_depth_sha256", "metric_depth"),
}
OBJECT_IDENTITY_KEYS = (
    "name",
    "generation",
    "size",
    "md5_base64",
    "crc32c_base64",
)


class EquivalenceError(ValueError):
    """A frozen input or local package differs from the T0 contract."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EquivalenceError(f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise EquivalenceError(
                f"JSON object required at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def _resolve_repo_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise EquivalenceError("non-empty contract path required")
    raw = Path(value)
    return raw.resolve() if raw.is_absolute() else (_repo_root() / raw).resolve()


def _safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise EquivalenceError("non-empty relative content path required")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise EquivalenceError(f"path escapes package root: {relative}") from error
    if not path.is_file():
        raise EquivalenceError(f"package content missing: {relative}")
    return path


def _require_hash(path: Path, expected: Any, label: str) -> None:
    if not path.is_file() or expected != _sha256(path):
        raise EquivalenceError(f"{label} SHA-256 mismatch")


def _implementation_bindings(contract: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    implementations = contract.get("implementations", {})
    if not isinstance(implementations, dict):
        raise EquivalenceError("implementation receipts missing")
    for key, expected_relative in IMPLEMENTATIONS.items():
        receipt = implementations.get(key)
        if not isinstance(receipt, dict):
            raise EquivalenceError(f"{key} implementation receipt missing")
        if Path(str(receipt.get("path", ""))).as_posix() != expected_relative:
            raise EquivalenceError(f"{key} implementation path mismatch")
        implementation_path = (_repo_root() / expected_relative).resolve()
        implementation_sha = _sha256(implementation_path)
        if receipt.get("sha256") != implementation_sha:
            raise EquivalenceError(f"{key} implementation SHA-256 mismatch")
        observed[key] = implementation_sha
    return observed


def _load_bound_report(
    contract: dict[str, Any], key: str
) -> tuple[Path, dict[str, Any]]:
    binding = contract.get("evidence", {}).get(key)
    if not isinstance(binding, dict):
        raise EquivalenceError(f"{key} evidence binding missing")
    path = _resolve_repo_path(binding.get("path"))
    _require_hash(path, binding.get("sha256"), f"{key} report")
    return path, _load_json(path)


def _validate_path_report(value: Any, label: str) -> int:
    if not isinstance(value, dict):
        raise EquivalenceError(f"{label} path report missing")
    maximum = value.get("maximum_content_path_length")
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or maximum < 0
        or maximum >= MAX_CONTENT_PATH_EXCLUSIVE
        or value.get("limit_exclusive") != MAX_CONTENT_PATH_EXCLUSIVE
        or value.get("all_content_paths_under_limit") is not True
        or value.get("session_id_present_in_any_content_path") is not False
    ):
        raise EquivalenceError(f"{label} path report is not admissible")
    return maximum


def _validate_evidence_reports(
    contract: dict[str, Any], candidate_root: Path
) -> dict[str, Any]:
    canary_path, canary = _load_bound_report(
        contract, "filesystem_canary"
    )
    if (
        canary.get("schema") != CANARY_SCHEMA
        or canary.get("terminal") != CANARY_READY
        or canary.get("generation_bound_url_observed") is not True
        or canary.get("size_and_md5_verified") is not True
        or canary.get("long_session_id_excluded_from_content_paths") is not True
        or canary.get("network_opened") is not False
        or canary.get("source_opened") is not False
        or canary.get("fresh_or_reserved_source_opened") is not False
    ):
        raise EquivalenceError("filesystem canary report mismatch")
    canary_maximum = _validate_path_report(
        canary.get("path_preflight"), "filesystem canary"
    )

    preflight_path, preflight = _load_bound_report(
        contract, "path_preflight"
    )
    if (
        preflight.get("schema") != TRANSPORT_SCHEMA
        or preflight.get("terminal") != PREFLIGHT_READY
        or preflight.get("network_opened") is not False
        or preflight.get("source_opened") is not False
        or preflight.get("output_created") is not False
        or Path(str(preflight.get("planned_output_root", ""))).resolve()
        != candidate_root.resolve()
    ):
        raise EquivalenceError("short-path preflight report mismatch")
    preflight_maximum = _validate_path_report(
        preflight.get("path_preflight"), "short-path preflight"
    )
    return {
        "filesystem_canary_path": str(canary_path),
        "filesystem_canary_sha256": _sha256(canary_path),
        "filesystem_canary_maximum_path_length": canary_maximum,
        "path_preflight_path": str(preflight_path),
        "path_preflight_sha256": _sha256(preflight_path),
        "preflight_maximum_path_length": preflight_maximum,
        "preflight_acquisition_token": preflight.get("acquisition_token"),
    }


def _object_identity(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, dict):
        raise EquivalenceError(f"{label} object inventory missing")
    identity = tuple(value.get(key) for key in OBJECT_IDENTITY_KEYS)
    name, generation, size, md5_value, _crc = identity
    if (
        not isinstance(name, str)
        or not name
        or generation is None
        or not str(generation)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(md5_value, str)
        or not md5_value
    ):
        raise EquivalenceError(
            f"{label} lacks generation, size, or MD5 identity"
        )
    return identity


def _verify_local_object(
    root: Path, relative: str, inventory: Any, label: str
) -> None:
    path = _safe_file(root, relative)
    identity = _object_identity(inventory, label)
    if path.stat().st_size != identity[2] or _md5_base64(path) != identity[3]:
        raise EquivalenceError(f"{label} local size or MD5 mismatch")


def _load_package(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    manifest_path = root / "manifest.replay.jsonl"
    spec_path = root / "dataset_spec.json"
    _require_hash(manifest_path, binding.get("manifest_sha256"), f"{label} manifest")
    _require_hash(spec_path, binding.get("dataset_spec_sha256"), f"{label} spec")
    spec = _load_json(spec_path)
    rows = _load_jsonl(manifest_path)
    if spec.get("schema") != REPLAY_SCHEMA or not rows:
        raise EquivalenceError(f"{label} replay schema or rows invalid")
    return {
        "root": root,
        "manifest_path": manifest_path,
        "spec_path": spec_path,
        "manifest_sha256": _sha256(manifest_path),
        "dataset_spec_sha256": _sha256(spec_path),
        "spec": spec,
        "rows": rows,
    }


def _verify_package_local_bindings(package: dict[str, Any], label: str) -> None:
    root = package["root"]
    spec = package["spec"]
    rows = package["rows"]
    inventory = spec.get("source_inventory")
    if not isinstance(inventory, dict):
        raise EquivalenceError(f"{label} source inventory missing")
    for key, relative in METADATA_PATHS.items():
        _verify_local_object(root, relative, inventory.get(key), f"{label}:{key}")
    split_path = root / "source_metadata/official_split_session_ids.txt"
    if label == "candidate":
        _verify_local_object(
            root,
            "source_metadata/official_split_session_ids.txt",
            inventory.get("official_split_receipt"),
            f"{label}:official_split_receipt",
        )
    elif split_path.exists():
        _verify_local_object(
            root,
            "source_metadata/official_split_session_ids.txt",
            inventory.get("official_split_receipt"),
            f"{label}:official_split_receipt",
        )
    selected = spec.get("sampling", {}).get("selected_source_frames")
    if (
        not isinstance(selected, list)
        or [row.get("source_frame_index") for row in rows] != selected
        or [row.get("frame_index") for row in rows] != list(range(len(rows)))
        or len({row.get("id") for row in rows}) != len(rows)
    ):
        raise EquivalenceError(f"{label} manifest timeline mismatch")
    for inventory_key, (
        path_key,
        sha_key,
        modality_key,
    ) in FRAME_INVENTORY_KEYS.items():
        objects = inventory.get(inventory_key)
        if not isinstance(objects, list) or len(objects) != len(rows):
            raise EquivalenceError(f"{label} {inventory_key} inventory mismatch")
        for index, (row, object_receipt) in enumerate(zip(rows, objects)):
            path = _safe_file(root, row.get(path_key))
            if _sha256(path) != row.get(sha_key):
                raise EquivalenceError(
                    f"{label}:frame-{index}:{path_key} SHA-256 mismatch"
                )
            modality = row.get("modalities", {}).get(modality_key)
            if _object_identity(modality, f"{label}:frame-{index}:{modality_key}") != (
                _object_identity(
                    object_receipt,
                    f"{label}:inventory-{index}:{inventory_key}",
                )
            ):
                raise EquivalenceError(
                    f"{label}:frame-{index}:{modality_key} inventory mismatch"
                )
            if (
                path.stat().st_size != int(modality["size"])
                or _md5_base64(path) != modality["md5_base64"]
            ):
                raise EquivalenceError(
                    f"{label}:frame-{index}:{modality_key} local MD5 mismatch"
                )


def _frame_equivalence_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "session_id": row.get("session_id"),
        "sequence_id": row.get("sequence_id"),
        "frame_index": row.get("frame_index"),
        "source_frame_index": row.get("source_frame_index"),
        "source_timestamp_ms": row.get("source_timestamp_ms"),
        "width": row.get("width"),
        "height": row.get("height"),
        "label_authority": row.get("label_authority"),
        "event_truth": row.get("event_truth"),
        "source": row.get("source"),
        "image_sha256": row.get("image_sha256"),
        "source_mask_sha256": row.get("source_mask_sha256"),
        "source_depth_sha256": row.get("source_depth_sha256"),
        "rgb_object": _object_identity(row.get("modalities", {}).get("rgb"), "rgb"),
        "mask_object": _object_identity(
            row.get("modalities", {}).get("panoptic_mask"), "mask"
        ),
        "depth_object": _object_identity(
            row.get("modalities", {}).get("metric_depth"), "depth"
        ),
    }


def _compare_packages(
    canonical: dict[str, Any], candidate: dict[str, Any]
) -> None:
    canonical_spec = canonical["spec"]
    candidate_spec = candidate["spec"]
    for key in ("source", "sampling", "camera"):
        if canonical_spec.get(key) != candidate_spec.get(key):
            raise EquivalenceError(f"package {key} differs from canonical")
    for key in (
        "description",
        "labelmap",
        "annotation_types",
        "camera_poses",
        "official_split_receipt",
    ):
        if _object_identity(
            canonical_spec.get("source_inventory", {}).get(key),
            f"canonical:{key}",
        ) != _object_identity(
            candidate_spec.get("source_inventory", {}).get(key),
            f"candidate:{key}",
        ):
            raise EquivalenceError(f"metadata object identity differs: {key}")
    canonical_rows = canonical["rows"]
    candidate_rows = candidate["rows"]
    if len(canonical_rows) != len(candidate_rows):
        raise EquivalenceError("candidate frame count differs from canonical")
    for index, (canonical_row, candidate_row) in enumerate(
        zip(canonical_rows, candidate_rows)
    ):
        if _frame_equivalence_identity(canonical_row) != (
            _frame_equivalence_identity(candidate_row)
        ):
            raise EquivalenceError(
                f"frame {index} source identity or content differs"
            )


def _validate_transport_receipt(
    candidate: dict[str, Any],
    contract_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    root = candidate["root"]
    receipt_path = root / "qa/transport_receipt.json"
    replay_validation_path = root / "qa/replay_validation.json"
    receipt = _load_json(receipt_path)
    source = contract.get("source", {})
    selected = candidate["spec"].get("sampling", {}).get(
        "selected_source_frames"
    )
    if (
        receipt.get("schema") != TRANSPORT_SCHEMA
        or receipt.get("terminal") != TRANSPORT_READY
        or Path(str(receipt.get("output_root", ""))).resolve() != root.resolve()
        or receipt.get("session_id") != source.get("session_id")
        or receipt.get("official_split") != "train"
        or receipt.get("selected_source_frames") != selected
        or Path(str(receipt.get("execution_contract_path", ""))).resolve()
        != contract_path.resolve()
        or receipt.get("execution_contract_sha256") != _sha256(contract_path)
        or receipt.get("dataset_spec_sha256")
        != candidate["dataset_spec_sha256"]
        or receipt.get("manifest_sha256") != candidate["manifest_sha256"]
        or receipt.get("replay_validation_sha256")
        != _sha256(replay_validation_path)
        or receipt.get("all_downloads_generation_bound") is not True
        or receipt.get("all_downloads_size_and_md5_verified") is not True
    ):
        raise EquivalenceError("candidate transport receipt mismatch")
    receipt_maximum = _validate_path_report(
        receipt.get("path_preflight"), "candidate transport receipt"
    )
    authorization = receipt.get("authorization", {})
    if (
        authorization.get("fresh_or_reserved_evaluation_authorized")
        is not False
        or authorization.get("teacher_label_or_corpus_authorized")
        is not False
        or authorization.get("student_training_authorized") is not False
        or authorization.get("research_mainline_changed") is not False
        or authorization.get("default_app_changed") is not False
    ):
        raise EquivalenceError("candidate transport authorization drift")
    actual_files = [path for path in root.rglob("*") if path.is_file()]
    if any(path.name.endswith(".tmp") for path in actual_files):
        raise EquivalenceError("candidate contains incomplete temporary content")
    maximum_actual = max(
        (
            len(str(path.resolve()))
            for path in actual_files
        ),
        default=0,
    )
    maximum_with_tmp = max(
        (
            len(str(path.with_suffix(path.suffix + ".tmp").resolve()))
            for path in actual_files
        ),
        default=0,
    )
    if max(maximum_actual, maximum_with_tmp) >= MAX_CONTENT_PATH_EXCLUSIVE:
        raise EquivalenceError("candidate actual or temporary path exceeds budget")
    return {
        "transport_receipt_path": str(receipt_path),
        "transport_receipt_sha256": _sha256(receipt_path),
        "acquisition_token": receipt.get("acquisition_token"),
        "receipt_maximum_content_path_length": receipt_maximum,
        "actual_maximum_content_path_length": maximum_actual,
        "actual_maximum_content_path_with_tmp_length": maximum_with_tmp,
    }


def validate(
    execution_contract_path: Path,
    candidate_root: Path,
) -> dict[str, Any]:
    contract_path = execution_contract_path.resolve()
    docs_root = (_repo_root() / "docs/research/hftf").resolve()
    try:
        contract_path.relative_to(docs_root)
    except ValueError as error:
        raise EquivalenceError(
            f"T0 execution contract must stay under {docs_root}"
        ) from error
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
    ):
        raise EquivalenceError("T0 execution contract is not frozen")
    source = contract.get("source", {})
    config = contract.get("acquisition_config", {})
    if (
        source.get("role") != EXECUTION_ROLE
        or source.get("outcome_open_before_t0") is not True
        or source.get("fresh_evidence_credit") is not False
        or source.get("reserved_source") is not False
        or source.get("official_split") != "train"
        or config.get("official_split") != source.get("official_split")
        or config.get("session_id") != source.get("session_id")
    ):
        raise EquivalenceError("contract source is not exact consumed Development")
    authorization = contract.get("authorization", {})
    if (
        authorization.get("consumed_development_transport_execution")
        is not True
        or authorization.get("fresh_or_reserved_source_open") is not False
        or authorization.get("teacher_or_student_execution") is not False
    ):
        raise EquivalenceError("contract authorization boundary mismatch")
    implementation_hashes = _implementation_bindings(contract)

    candidate_contract = contract.get("short_path_package")
    if (
        not isinstance(candidate_contract, dict)
        or candidate_contract.get(
            "post_open_local_hashes_are_transport_receipts_only"
        )
        is not True
        or "manifest_sha256" in candidate_contract
        or "dataset_spec_sha256" in candidate_contract
    ):
        raise EquivalenceError(
            "candidate contract must freeze only root and post-open receipt policy"
        )
    frozen_candidate_root = _resolve_repo_path(
        candidate_contract.get("root")
    )
    if candidate_root.resolve() != frozen_candidate_root:
        raise EquivalenceError("candidate root differs from frozen contract")
    evidence = _validate_evidence_reports(contract, frozen_candidate_root)

    canonical_binding = source.get("canonical_consumed_package")
    if not isinstance(canonical_binding, dict):
        raise EquivalenceError("canonical consumed package binding missing")
    canonical_root = _resolve_repo_path(canonical_binding.get("root"))
    candidate_binding = _load_json(
        frozen_candidate_root / "qa/transport_receipt.json"
    )

    canonical = _load_package(
        canonical_root, canonical_binding, "canonical consumed package"
    )
    candidate = _load_package(
        frozen_candidate_root, candidate_binding, "candidate short-path package"
    )
    _verify_package_local_bindings(canonical, "canonical")
    _verify_package_local_bindings(candidate, "candidate")
    _compare_packages(canonical, candidate)
    receipt = _validate_transport_receipt(
        candidate, contract_path, contract
    )
    if (
        evidence["preflight_acquisition_token"]
        != receipt["acquisition_token"]
        or evidence["preflight_maximum_path_length"]
        != receipt["receipt_maximum_content_path_length"]
    ):
        raise EquivalenceError(
            "preflight and transport receipt path plan differ"
        )
    return {
        "schema": SCHEMA,
        "terminal": READY,
        "execution_contract_path": str(contract_path),
        "execution_contract_sha256": _sha256(contract_path),
        "implementation_hashes": implementation_hashes,
        "evidence": evidence,
        "canonical_package": {
            "root": str(canonical_root),
            "manifest_sha256": canonical["manifest_sha256"],
            "dataset_spec_sha256": canonical["dataset_spec_sha256"],
        },
        "short_path_package": {
            "root": str(frozen_candidate_root),
            "manifest_sha256": candidate["manifest_sha256"],
            "dataset_spec_sha256": candidate["dataset_spec_sha256"],
        },
        "frame_count": len(candidate["rows"]),
        "selected_source_frames": candidate["spec"]["sampling"][
            "selected_source_frames"
        ],
        "gates": {
            "frozen_contract_and_implementations_bound": True,
            "offline_canary_and_preflight_bound": True,
            "source_identity_and_selected_indices_equivalent": True,
            "per_frame_local_sha256_equivalent": True,
            "metadata_object_identities_equivalent": True,
            "generation_and_md5_receipts_valid": True,
            "content_and_temporary_paths_under_240": True,
        },
        "transport_receipt": receipt,
        "authorization": {
            "consumed_development_equivalence_established": True,
            "acquisition_authorized": False,
            "fresh_or_reserved_source_open_authorized": False,
            "teacher_or_student_execution_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
        },
        "network_opened": False,
    }


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (_repo_root() / "artifacts.local").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise EquivalenceError(
            f"output must stay under {artifacts_root}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise EquivalenceError(f"refusing to overwrite output: {output}")
        result = validate(
            args.execution_contract, args.candidate_root
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": result["terminal"],
                    "frame_count": result["frame_count"],
                    "output": str(output),
                }
            )
        )
        return 0
    except (
        EquivalenceError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "terminal": NOT_EVALUABLE,
                    "raw_failure_class": (
                        "CONSUMED_PACKAGE_EQUIVALENCE_FAILURE"
                    ),
                    "error": str(error),
                    "network_opened": False,
                    "fresh_or_reserved_source_open_authorized": False,
                },
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

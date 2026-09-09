from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.io_utils import (
    read_json,
    read_jsonl,
    sha256_file,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.validator import (
    validate_dataset,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_stress_r1.builder import (
    MODULE_ROOT,
    REPO_ROOT,
    _validate_frozen_parent_chain,
)


SEQUENCE_FIELDS = {
    "schema",
    "protocol_id",
    "sequence_id",
    "base_sequence_id",
    "condition_id",
    "condition_family",
    "condition_parameters",
    "scene_id",
    "motion_id",
    "role",
    "frame_count",
    "pair_count",
}
ALGORITHM_FIELDS = {
    "schema",
    "protocol_id",
    "split",
    "sequence_id",
    "frame_index",
    "timestamp",
    "rgb_path",
    "rgb_sha256",
    "valid_mask_path",
    "valid_mask_sha256",
    "intrinsics",
    "rotation_current_from_previous",
    "algorithm_seed",
}
OBSERVED_DEPTH_FIELDS = {
    "schema",
    "sequence_id",
    "base_sequence_id",
    "frame_index",
    "depth_npy_path",
    "depth_npy_sha256",
    "relative_sigma",
}


def _exact_fields(row: dict[str, Any], expected: set[str], label: str) -> None:
    if set(row) != expected:
        raise ValueError(f"{label}_SCHEMA_FIELDS")


def _validate_receipt(
    root: Path,
    spec: dict[str, Any],
    receipt: dict[str, Any],
    sequence_count: int,
    algorithm_count: int,
    observed_count: int,
) -> None:
    if receipt.get("schema") != "rcle.synthetic_stress.dataset_receipt.v1":
        raise ValueError("RECEIPT_SCHEMA")
    if receipt.get("protocol_id") != spec["protocol_id"]:
        raise ValueError("RECEIPT_PROTOCOL")
    if (
        receipt.get("status")
        != "CANDIDATE_STRESS_DATASET_VALIDATION_REQUIRED"
    ):
        raise ValueError("RECEIPT_STATUS")
    expected_counts = {
        "sequence_count": sequence_count,
        "algorithm_frame_count": algorithm_count,
        "observed_depth_frame_count": observed_count,
    }
    for field, expected in expected_counts.items():
        if receipt.get(field) != expected:
            raise ValueError(f"RECEIPT_COUNT:{field}")
    bindings = {
        "dataset_spec_sha256": "dataset_spec.json",
        "stress_sequence_manifest_sha256": "stress_sequence_manifest.jsonl",
        "algorithm_input_manifest_sha256": "algorithm_input_manifest.jsonl",
        "observed_depth_manifest_sha256": "observed_depth_manifest.jsonl",
        "base_r0_spec_sha256": "base_r0_spec.json",
        "base_dataset_spec_sha256": "base/dataset_spec.json",
        "base_dataset_receipt_sha256": "base/receipt.json",
    }
    for field, name in bindings.items():
        if receipt.get(field) != sha256_file(root / name):
            raise ValueError(f"RECEIPT_HASH_MISMATCH:{name}")
    parent_path = (REPO_ROOT / spec["parent_r0"]["spec_path"]).resolve()
    if receipt.get("parent_r0_spec_sha256") != sha256_file(parent_path):
        raise ValueError("RECEIPT_PARENT_R0_SPEC")
    if (
        receipt["base_r0_spec_sha256"]
        != receipt["base_dataset_spec_sha256"]
    ):
        raise ValueError("BASE_SPEC_COPY_DRIFT")
    parent_chain = _validate_frozen_parent_chain(spec)
    if receipt.get("parent_chain") != parent_chain:
        raise ValueError("RECEIPT_PARENT_CHAIN")
    expected_code = {
        name: sha256_file(MODULE_ROOT / name)
        for name in ("builder.py", "evaluator.py", "validator.py", "qa.py")
    }
    if receipt.get("stress_code_sha256") != expected_code:
        raise ValueError("RECEIPT_STRESS_CODE_BINDING")
    budget = spec["resource_budget"]
    if receipt.get("free_bytes_before_start", -1) < int(
        budget["minimum_free_bytes_before_start"]
    ):
        raise ValueError("RECEIPT_FREE_SPACE")
    if receipt.get("dataset_bytes", -1) > int(
        budget["disk_ceiling_bytes"]
    ):
        raise ValueError("RECEIPT_DISK_BUDGET")
    if receipt.get("elapsed_seconds", -1) > float(
        budget["runtime_ceiling_seconds"]
    ):
        raise ValueError("RECEIPT_RUNTIME_BUDGET")
    if receipt.get("disk_ceiling_bytes") != budget["disk_ceiling_bytes"]:
        raise ValueError("RECEIPT_DISK_CEILING_BINDING")
    if (
        receipt.get("runtime_ceiling_seconds")
        != budget["runtime_ceiling_seconds"]
    ):
        raise ValueError("RECEIPT_RUNTIME_CEILING_BINDING")
    if receipt.get("authority") != spec["authority"]:
        raise ValueError("RECEIPT_AUTHORITY")


def validate_stress_dataset(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    try:
        spec = read_json(root / "dataset_spec.json")
        receipt = read_json(root / "receipt.json")
        sequences = read_jsonl(root / "stress_sequence_manifest.jsonl")
        algorithm = read_jsonl(root / "algorithm_input_manifest.jsonl")
        observed = read_jsonl(root / "observed_depth_manifest.jsonl")
        if spec.get("status") != "PREREGISTERED_BEFORE_STRESS_OUTCOME_ACCESS":
            raise ValueError("STRESS_SPEC_STATUS")
        allocation = spec["allocation"]
        if len(sequences) != allocation["expected_stress_sequences"]:
            raise ValueError("STRESS_SEQUENCE_COUNT")
        if len(algorithm) != allocation["expected_algorithm_frames"]:
            raise ValueError("STRESS_ALGORITHM_FRAME_COUNT")
        expected_observed = (
            allocation["expected_base_sequences"]
            * allocation["frame_count_per_sequence"]
        )
        if len(observed) != expected_observed:
            raise ValueError("STRESS_OBSERVED_DEPTH_FRAME_COUNT")

        conditions = {
            condition["condition_id"]: condition
            for condition in spec["conditions"]
        }
        expected_base = {
            f"{scene['scene_id']}__{motion_id}"
            for scene in allocation["scene_families"]
            for motion_id in allocation["motion_ids"]
        }
        expected_sequences = {
            f"{base_id}__{condition_id}"
            for base_id in expected_base
            for condition_id in conditions
        }
        sequence_ids: set[str] = set()
        for row in sequences:
            _exact_fields(row, SEQUENCE_FIELDS, "STRESS_SEQUENCE")
            if row["schema"] != "rcle.synthetic_stress.sequence.v1":
                raise ValueError("STRESS_SEQUENCE_SCHEMA")
            if row["protocol_id"] != spec["protocol_id"]:
                raise ValueError("STRESS_SEQUENCE_PROTOCOL")
            expected_id = (
                f"{row['base_sequence_id']}__{row['condition_id']}"
            )
            if row["sequence_id"] != expected_id:
                raise ValueError("STRESS_SEQUENCE_ID")
            if row["base_sequence_id"] not in expected_base:
                raise ValueError("STRESS_BASE_CARTESIAN")
            condition = conditions.get(row["condition_id"])
            if condition is None:
                raise ValueError("STRESS_CONDITION_ID")
            if (
                row["condition_family"] != condition["family"]
                or row["condition_parameters"] != condition
            ):
                raise ValueError("STRESS_CONDITION_BINDING")
            if (
                row["frame_count"] != allocation["frame_count_per_sequence"]
                or row["pair_count"] != allocation["pair_count_per_sequence"]
            ):
                raise ValueError("STRESS_SEQUENCE_DENOMINATOR")
            sequence_ids.add(row["sequence_id"])
        if len(sequence_ids) != len(sequences):
            raise ValueError("DUPLICATE_STRESS_SEQUENCE")
        if sequence_ids != expected_sequences:
            raise ValueError("STRESS_CARTESIAN_PRODUCT")

        counts = Counter(row["sequence_id"] for row in algorithm)
        frame_indices: dict[str, list[int]] = defaultdict(list)
        for row in algorithm:
            _exact_fields(row, ALGORITHM_FIELDS, "STRESS_ALGORITHM")
            if row["schema"] != "rcle.synthetic_scene.algorithm_input.v1":
                raise ValueError("STRESS_ALGORITHM_SCHEMA")
            if (
                row["protocol_id"]
                != "RCLE_SYNTHETIC_SCENE_MECHANISM_VALIDATION_R0"
                or row["split"] != "development"
            ):
                raise ValueError("STRESS_ALGORITHM_PARENT_PROTOCOL")
            if row["sequence_id"] not in expected_sequences:
                raise ValueError("STRESS_ALGORITHM_SEQUENCE")
            frame_indices[row["sequence_id"]].append(row["frame_index"])
            for field, hash_field in (
                ("rgb_path", "rgb_sha256"),
                ("valid_mask_path", "valid_mask_sha256"),
            ):
                path = (root / row[field]).resolve()
                if not path.is_relative_to(
                    (root / "algorithm_staging").resolve()
                ):
                    raise ValueError("STRESS_STAGING_PATH_ESCAPE")
                if sha256_file(path) != row[hash_field]:
                    raise ValueError("STRESS_STAGING_HASH_MISMATCH")
        if set(counts) != expected_sequences or set(counts.values()) != {
            allocation["frame_count_per_sequence"]
        }:
            raise ValueError("STRESS_SEQUENCE_FRAME_DENOMINATOR")
        expected_indices = list(
            range(allocation["frame_count_per_sequence"])
        )
        if any(
            sorted(indices) != expected_indices
            for indices in frame_indices.values()
        ):
            raise ValueError("STRESS_FRAME_INDEX_SEQUENCE")

        depth_condition = next(
            condition
            for condition in spec["conditions"]
            if condition["family"] == "depth_noise"
        )
        expected_observed_keys = {
            (
                f"{base_id}__{depth_condition['condition_id']}",
                frame_index,
            )
            for base_id in expected_base
            for frame_index in expected_indices
        }
        observed_keys: set[tuple[str, int]] = set()
        for row in observed:
            _exact_fields(row, OBSERVED_DEPTH_FIELDS, "OBSERVED_DEPTH")
            if row["schema"] != "rcle.synthetic_stress.observed_depth.v1":
                raise ValueError("OBSERVED_DEPTH_SCHEMA")
            expected_id = (
                f"{row['base_sequence_id']}__"
                f"{depth_condition['condition_id']}"
            )
            if row["sequence_id"] != expected_id:
                raise ValueError("OBSERVED_DEPTH_SEQUENCE")
            if row["relative_sigma"] != depth_condition["relative_sigma"]:
                raise ValueError("OBSERVED_DEPTH_SIGMA")
            key = (row["sequence_id"], row["frame_index"])
            if key in observed_keys:
                raise ValueError("OBSERVED_DEPTH_DUPLICATE")
            observed_keys.add(key)
            path = (root / row["depth_npy_path"]).resolve()
            if not path.is_relative_to(
                (root / "observed_depth").resolve()
            ):
                raise ValueError("OBSERVED_DEPTH_PATH_ESCAPE")
            if sha256_file(path) != row["depth_npy_sha256"]:
                raise ValueError("OBSERVED_DEPTH_HASH_MISMATCH")
        if observed_keys != expected_observed_keys:
            raise ValueError("OBSERVED_DEPTH_CARTESIAN")

        base_validation = validate_dataset(
            root / "base", check_algorithm=False
        )
        if base_validation["status"] != "PASS":
            raise ValueError("BASE_R0_VALIDATION_FAILED")
        _validate_receipt(
            root,
            spec,
            receipt,
            len(sequences),
            len(algorithm),
            len(observed),
        )

        summary_path = root / "scientific_summary.json"
        summary_checked = False
        if summary_path.is_file():
            summary = read_json(summary_path)
            if summary["pair_count"] != allocation[
                "expected_algorithm_pairs"
            ]:
                raise ValueError("STRESS_SUMMARY_PAIR_COUNT")
            ledger_names = {
                "algorithm": "algorithm_pair_ledger.jsonl",
                "latent_truth": "latent_truth_pair_ledger.jsonl",
                "old_r1_comparison": "old_r1_comparison.jsonl",
                "observed_depth_truth": (
                    "observed_depth_truth_pair_ledger.jsonl"
                ),
            }
            for key, name in ledger_names.items():
                if summary["ledger_sha256"][key] != sha256_file(
                    root / name
                ):
                    raise ValueError(f"STRESS_LEDGER_HASH:{name}")
            summary_checked = True
        return {
            "schema": "rcle.synthetic_stress.validation.v1",
            "protocol_id": spec["protocol_id"],
            "status": "PASS",
            "errors": [],
            "counts": {
                "sequences": len(sequences),
                "algorithm_frames": len(algorithm),
                "algorithm_pairs": sum(value - 1 for value in counts.values()),
                "observed_depth_frames": len(observed),
            },
            "base_r0_validation": base_validation["status"],
            "scientific_summary_hashes_checked": summary_checked,
        }
    except Exception as exc:
        errors.append(str(exc))
        return {
            "schema": "rcle.synthetic_stress.validation.v1",
            "status": "FAIL",
            "errors": errors,
        }

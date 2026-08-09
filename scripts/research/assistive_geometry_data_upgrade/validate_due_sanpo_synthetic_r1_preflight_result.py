"""Validate the tracked AG-DUE R1 SANPO-Synthetic preflight terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.assistive_geometry_data_upgrade import (
    run_due_sanpo_synthetic_r1_metadata_preflight as runner,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry-data-upgrade/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_METADATA_AND_"
    "OBJECT_INVENTORY_PREFLIGHT_RESULT_2026-08-10.json"
)
RESULT_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_METADATA_AND_"
    "OBJECT_INVENTORY_PREFLIGHT_RESULT_2026-08-10"
)


class ResultError(ValueError):
    """The governed preflight terminal drifted or gained authority."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_result(result: dict[str, Any], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    require(
        set(result)
        == {
            "schema",
            "result_id",
            "status",
            "execution_lock",
            "source",
            "observed_failure",
            "artifact_receipts",
            "decision",
            "terminal",
            "inventory_counts",
            "capability_counts",
            "execution_disclosure",
            "authority",
            "implementation",
            "unique_successor",
            "reentry_condition",
            "claim_ceiling",
        },
        "result field set drift",
    )
    require(
        result["schema"]
        == "blindassist.assistive_geometry_due.sanpo_synthetic_r1_metadata_preflight_governed_result.v1",
        "result schema drift",
    )
    require(result["result_id"] == RESULT_ID, "result identity drift")
    require(result["status"] == "NOT_EVALUABLE_EXACT_METADATA_OBJECT_MISSING_ROUTE_CLOSED", "status drift")
    lock_path = runner.LOCK_PATH.relative_to(repo_root).as_posix()
    require(
        result["execution_lock"]
        == {"path": lock_path, "sha256": sha256_file(runner.LOCK_PATH)},
        "execution lock binding drift",
    )
    runner.validate_execution_lock(runner.load_json(runner.LOCK_PATH), repo_root)
    require(
        result["source"]
        == {
            "source_id": runner.SOURCE_ID,
            "source_family": "SANPO_SYNTHETIC",
            "official_split": "train",
            "session_id": runner.SESSION_ID,
            "camera": runner.CAMERA,
            "lens": runner.LENS,
            "parent_count": 1,
            "fallback_or_substitution_attempted": False,
        },
        "source identity drift",
    )
    expected_missing = runner._expected_metadata_objects()[2]["path"]
    require(
        result["observed_failure"]
        == {
            "kind": "EXACT_REQUIRED_METADATA_OBJECT_MISSING",
            "object_name": expected_missing,
            "http_status": 404,
            "attempt_count": 2,
            "requests_per_attempt": 6,
            "total_actual_network_requests": 12,
            "request_count_basis": "DERIVED_FROM_HASH_LOCKED_SEQUENTIAL_CONTROL_FLOW",
            "retry_receipt_prior_request_count_field_correct": False,
            "frame_prefix_listing_started": False,
            "frame_body_request_count": 0,
        },
        "observed failure drift",
    )
    root = runner.OUTPUT_ROOT.relative_to(repo_root).as_posix()
    expected_artifacts = [
        ("attempt_receipt.json", 2680, "A48993BC086646EBE68EB051AEF2DD5A6BC56F00724162B950C70B6C458823D2"),
        ("retry_attempt_receipt.json", 1512, "04BC5955DF93303DCDBDE308B25ABB232719DA77A5D19866642FAA6ED50A2970"),
        ("source_object_inventory.json", 1464, "7A8B86F3C8EBE7218879BFACA624119728FD56C99B52C2F4562C0E27140EEE7C"),
        ("metadata_schema_receipt.json", 1056, "1712D4AA6D580875AE29F567F84094A0C46A191D73E0BB05B4EB2CAB38760E35"),
        ("preflight_result.json", 3628, "3ACE4B3AD2C433FE150440F97148AA5885AEF46F116146AF9D61C8D57E5A8FC0"),
    ]
    require(
        result["artifact_receipts"]
        == [
            {
                "path": f"{root}/{name}",
                "bytes": size,
                "sha256": digest,
                "tracked": False,
            }
            for name, size, digest in expected_artifacts
        ],
        "artifact receipt drift",
    )
    require(result["decision"] == "NOT_EVALUABLE", "decision drift")
    require(result["terminal"] == "STOP_SOURCE_OBJECT_INVENTORY_OR_SCHEMA_INCOMPLETE", "terminal drift")
    require(
        result["inventory_counts"]
        == {
            "rgb_object_inventory_count": 0,
            "panoptic_object_inventory_count": 0,
            "metric_depth_object_inventory_count": 0,
            "numeric_index_intersection_count": 0,
            "pose_table_row_count": 0,
            "selected_lowest_25_count": 0,
        },
        "inventory count drift",
    )
    require(
        result["capability_counts"]
        == {
            "oracle_depth_factor_frames": 0,
            "oracle_support_factor_frames": 0,
            "boundary_truth_frames": 0,
            "explicit_timestamp_frames": 0,
            "pose_transform_frames": 0,
            "portrait_frames": 0,
            "landscape_frames": 0,
        },
        "capability count drift",
    )
    require(
        result["execution_disclosure"]
        == {
            "network_used": True,
            "description_and_labelmap_bodies_read_in_memory": True,
            "raw_metadata_bytes_persisted": False,
            "annotation_type_body_read": False,
            "pose_table_body_read": False,
            "frame_prefix_listing_performed": False,
            "frame_body_requested_or_read": False,
            "frame_body_bytes_read": 0,
            "rgb_visual_access": False,
            "mask_or_depth_decoded": False,
            "local_existing_payload_opened": False,
            "teacher_model_derivation_or_training": False,
            "failure_finalization_network_used": False,
        },
        "execution disclosure drift",
    )
    require(
        result["authority"]
        == {
            "inventory_is_capability_truth": False,
            "source_data_support_established": False,
            "dca_pass": False,
            "r2_f1_parent_gate_pass": False,
            "body_canary_execution_authorized": False,
            "pose_or_timestamp_admitted": False,
            "support_truth_established": False,
            "boundary_truth_materialized": False,
            "derivation_label_materialization_or_training": False,
            "development_confirmation_android_product_safety": False,
        },
        "authority drift",
    )
    expected_impl = {
        "scripts/research/assistive_geometry_data_upgrade/validate_due_sanpo_synthetic_r1_preflight_result.py",
        "scripts/research/assistive_geometry_data_upgrade/test_validate_due_sanpo_synthetic_r1_preflight_result.py",
    }
    require(set(result["implementation"]) == expected_impl, "implementation set drift")
    for logical_path, digest in result["implementation"].items():
        require(sha256_file(repo_root / logical_path) == digest, f"implementation SHA drift: {logical_path}")
    require(result["unique_successor"] == runner.STOP_SUCCESSOR, "successor drift")
    require(
        result["reentry_condition"]
        == "No active AG-DUE R1 successor. A new source/session/path may enter only through a separately versioned R0 source manifest and source-specific protocol; this exact R1 session cannot be rescued by fallback or path guessing.",
        "reentry condition drift",
    )
    require(result["claim_ceiling"] == runner.load_json(runner.LOCK_PATH)["claim_ceiling"], "claim ceiling drift")
    return {"result_id": RESULT_ID, "status": "VALID", "decision": "NOT_EVALUABLE", "unique_successor": runner.STOP_SUCCESSOR}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(validate_result(load_json(RESULT_PATH)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

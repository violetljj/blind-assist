#!/usr/bin/env python3
"""Independent validator for a completed TARO R11 source-only Phase A root."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import platform
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_factor_headroom_runtime import candidate_inputs, depthart_runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_reducer_integration_runtime import reducer_integration as reducer
from scripts.research.taro_o1r_r7_canary_runtime import positive_occupancy_factor as r7_positive
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r11_abstention_runtime import abstention_candidate
from scripts.research.taro_o1r_r11_abstention_runtime import fresh_pool
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_inventory


REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-phase-a-r0"
LOCK_RELATIVE = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json"
INVENTORY_PATH = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-inventory-r0/exact-frame-plan.json"
PASS_TERMINAL = "TARO_O1R_R11_FRESH_POOL_PHASE_A_SOURCE_ONLY_SEALED_PASS"
PARENT_COUNT = 48
FRAME_COUNT = 1043
QUERY_COUNT = 9387
PRE_TERMINAL_FILE_COUNT = 5218
FINAL_FILE_COUNT = 5219
FROZEN_FRAME_COUNTS = [
    20, 14, 23, 24, 29, 7, 12, 14, 10, 21, 28, 15, 11, 28, 29, 72,
    36, 14, 18, 4, 54, 32, 83, 17, 15, 16, 29, 10, 12, 34, 7, 14,
    11, 6, 9, 1, 46, 6, 27, 26, 50, 9, 11, 27, 12, 9, 28, 13,
]
INVENTORY_CONTENT_SHA256 = "35156C2901A4CBEEDB6D611A56ABE3D711CEB68EF932480C21428BA4FF741600"
INVENTORY_RESULT_CONTENT_SHA256 = "C4F15A3EA4DC1C51463860B9510658620BA49086116F63EB9514FF89F9A494B1"
INVENTORY_MANIFEST_CONTENT_SHA256 = "59A1B3180E467266E16330D87C256F5D57B8D3C9BC2111DA9CD060DC043C01B8"
INVENTORY_FORMAL_CONTENT_SHA256 = "8961A155DF4FF23F882D1F1587C3516FD21ED2D137B026D5D8C6721E6A74D4AC"
PROTOCOL_CONTENT_SHA256 = "2A2854364E41CE2E94FE2D1DBF1F5EF068E18335DE1171A8486BC41CBAECF756"
AUTHORIZATION_CONTENT_SHA256 = "CF7814D52532FAB6A5EE8A4CA8EA29E9A7EF1017E075CF8FE597EEBE0834FF5F"
POOL_CONTENT_SHA256 = "9F1EE94980C9B2EB0C8D7A6503A25E11587760247C5A30F656DB28E60A27FFAF"
REQUEST_PLAN_SHA256 = "FE3578E4F8403F9F57DA767B21DC5EFBCAF6BBF6514DF776A7B3124B966BD521"
AUTHORITY_SCOPE = (
    "Exact frozen R11 48-parent Training pool and 144-URL plan: zero-body HEAD, bounded source download and "
    "integrity validation, all-48 source-only Phase A, source-only top-24 selection, then FARO only for the "
    "sealed top 24; no training, device, deployment, product, safety, or redistribution authority."
)
EXPECTED_BINDING_PATHS = {
    "R11_PROTOCOL": "docs/research/taro/TARO_O1R_R11_POSITIVE_OCCUPANCY_ABSTENTION_AND_FRESH_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json",
    "R11_DATA_USE_AUTHORIZATION": "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-12.json",
    "R11_PROTOCOL_VALIDATOR": "scripts/research/taro_o1r_r11_abstention_runtime/validate_protocol_lock.py",
    "R11_POOL_PLANNER": "scripts/research/taro_o1r_r11_abstention_runtime/fresh_pool.py",
    "R11_HEAD_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_head.py",
    "R11_DOWNLOAD_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_download.py",
    "R11_INVENTORY_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_INVENTORY_IMPLEMENTATION_LOCK_2026-08-12.md",
    "R11_INVENTORY_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_inventory.py",
    "R11_INVENTORY_PLAN": INVENTORY_PATH,
    "R11_INVENTORY_RESULT": "artifacts.local/evidence/taro/o1r-r11-fresh-pool-inventory-r0/result.json",
    "R11_INVENTORY_MANIFEST": "artifacts.local/evidence/taro/o1r-r11-fresh-pool-inventory-r0/manifest.json",
    "R11_INVENTORY_FORMAL_RESULT": "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_INVENTORY_RESULT_2026-08-12.json",
    "DEPTHART_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/depthart_runner.py",
    "CANDIDATE_INPUT_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/candidate_inputs.py",
    "SOURCE_ADAPTER_RUNTIME": "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py",
    "TRUTH_MATERIALIZER_RUNTIME": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "CANDIDATE_SCALE_PACKAGE": "scripts/research/taro_o0r_candidate_scale_runtime/__init__.py",
    "PROSPECTIVE_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/prospective_factor_runtime.py",
    "SOURCE_FACTOR_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/source_factor.py",
    "APPLE_SCALE_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/apple_scale.py",
    "DIRECT_APPLE_SUPPORT_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/direct_apple_support.py",
    "R5_CONFIRMATION_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/r5_confirmation.py",
    "R6_CONFIRMATION_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/r6_confirmation.py",
    "R6_CONFIRMATION_IO_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/r6_confirmation_io.py",
    "R6_FACTOR_SPLIT_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/r6_factor_split.py",
    "R6_UNTOUCHED_COHORT_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/r6_untouched_cohort.py",
    "R6_UNTOUCHED_INVENTORY_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/run_r6_untouched_inventory.py",
    "FACTOR_HEADROOM_PACKAGE": "scripts/research/taro_o0r_factor_headroom_runtime/__init__.py",
    "CANDIDATE_PHASE_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/candidate_phase.py",
    "FACTOR_CANARY_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/factor_canary.py",
    "FACTOR_HEADROOM_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/factor_headroom.py",
    "R6_REDUCER_RUNTIME": "scripts/research/taro_o1r_reducer_integration_runtime/reducer_integration.py",
    "R6_REDUCER_PACKAGE": "scripts/research/taro_o1r_reducer_integration_runtime/__init__.py",
    "LOCKED_UNCERTAINTY_LOADER": "scripts/research/taro_o1r_reducer_integration_runtime/locked_uncertainty.py",
    "UNCERTAINTY_LOADER_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/uncertainty_loader.py",
    "LOCKED_UNCERTAINTY_ARTIFACT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-artifact.json.gz",
    "LOCKED_UNCERTAINTY_RECEIPT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-receipt.json",
    "R7_SOURCE_FEATURE_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "R7_POSITIVE_FACTOR_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/positive_occupancy_factor.py",
    "R7_CANARY_PACKAGE": "scripts/research/taro_o1r_r7_canary_runtime/__init__.py",
    "R11_ABSTENTION_PACKAGE": "scripts/research/taro_o1r_r11_abstention_runtime/__init__.py",
    "R11_ABSTENTION_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/abstention_candidate.py",
    "R9_CLEAR_PACKAGE": "scripts/research/taro_o1r_r9_clear_runtime/__init__.py",
    "R9_SELECTOR_RUNTIME": "scripts/research/taro_o1r_r9_clear_runtime/clear_enrichment_fit.py",
    "R9_SELECTOR_ARTIFACT": "artifacts.local/evidence/taro/o1r-r9-clear-enrichment-development-r0/selector.json",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "R11_PHASE_A_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_IMPLEMENTATION_LOCK_2026-08-12.md",
    "R11_PHASE_A_RUNNER": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_phase_a.py",
    "R11_PHASE_A_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_run_pool_phase_a.py",
    "R11_PHASE_A_INDEPENDENT_VALIDATOR": "scripts/research/taro_o1r_r11_abstention_runtime/validate_pool_phase_a.py",
    "R11_PHASE_A_VALIDATOR_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_validate_pool_phase_a.py",
    "R7_FRESH_COHORT_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/fresh_confirmation_cohort.py",
    "R10_CLEAR_PACKAGE": "scripts/research/taro_o1r_r10_clear_runtime/__init__.py",
    "R10_FRESH_POOL_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/fresh_pool.py",
    "R10_PHASE_B_METRICS_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/phase_b_metrics.py",
    "R10_DOWNLOAD_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_download.py",
    "R10_HEAD_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_head.py",
    "R10_INVENTORY_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_inventory.py",
    "R10_PHASE_A_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_phase_a.py",
    "R10_PHASE_A_R1_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_phase_a_r1.py",
    "R10_SELECTED_PHASE_B_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_selected_phase_b.py",
    "R10_TOP8_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_top8_selection.py",
    "R10_TOP8_R1_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_top8_selection_r1.py",
    "R11_DEVELOPMENT_REPLAY_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/development_replay.py",
}
ARTIFACT_BINDING_ROLES = {
    "R11_INVENTORY_PLAN",
    "R11_INVENTORY_RESULT",
    "R11_INVENTORY_MANIFEST",
    "LOCKED_UNCERTAINTY_ARTIFACT",
    "LOCKED_UNCERTAINTY_RECEIPT",
    "R9_SELECTOR_ARTIFACT",
}
EXPECTED_AUTHORITY = {
    "source_container_read": True,
    "source_frame_decode": True,
    "depthart_inference": True,
    "candidate_inference_count": FRAME_COUNT,
    "source_only_phase_a": True,
    "r7_base_factor_materialization": True,
    "r11_candidate_materialization": True,
    "source_only_parent_scoring": False,
    "top24_selection": False,
    "highres_depth_member_payload_read": False,
    "faro_payload_read": False,
    "truth_scoring": False,
    "threshold_fit": False,
    "training": False,
    "network": False,
    "device": False,
    "deployment": False,
    "product": False,
    "safety": False,
    "redistribution": False,
}
EXPECTED_RESOURCE_BUDGET = {
    "maximum_wall_seconds": 57_600,
    "maximum_peak_rss_bytes": 17_179_869_184,
    "maximum_cuda_allocated_bytes": 12_884_901_888,
    "maximum_evidence_bytes": 2_147_483_648,
}
EXPECTED_RUNTIME_ENVIRONMENT = {
    "python_executable": "E:/codex-tools/tools/venvs/blindassist-venv-export312/Scripts/python.exe",
    "python_version": "3.11.9",
    "torch_version": "2.11.0+cu128",
    "timm_version": "1.0.28",
    "numpy_version": "2.1.3",
    "opencv_version": "4.10.0",
    "cuda_available": True,
    "cuda_version": "12.8",
    "cuda_device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
}
EXPECTED_CANDIDATE_IDENTITY = {
    "model_id": "depthart-s-metric-indoor-448-official-fp32",
    "source_root": "F:/ba-data/blindassist-artifacts-20260805/models/depthart/source",
    "source_commit": "0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c",
    "checkpoint_path": "F:/ba-data/blindassist-artifacts-20260805/models/depthart/source/checkpoints/metric/depthart_metric_indoor_s_448.pth",
    "checkpoint_bytes": 32_871_942,
    "checkpoint_sha256": "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65",
    "preprocess_id": "DEPTHART_OFFICIAL_LOWER_BOUND_448_RGB_CUBIC_IMAGENET_V1",
    "postprocess_id": "TARO_TORCH_CPU_BILINEAR_ALIGN_CORNERS_TRUE_FLOAT32_448X608_TO_1440X1920_V1",
    "inference_seed": 0,
    "device": "cuda",
    "output_dtype": "float32",
}
EXPECTED_DEPTHART_RUNTIME_IDENTITY = {
    "source_git_commit": "0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c",
    "source_tree_clean": True,
    "checkpoint_bytes": 32_871_942,
    "torch_version": "2.11.0+cu128",
    "cuda_version": "12.8",
    "opencv_version": "4.10.0",
    "numpy_version": "2.1.3",
    "device": "cuda",
    "cuda_device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "tf32_matmul": False,
    "tf32_cudnn": False,
    "cudnn_benchmark": False,
    "autocast": False,
    "inference_dtype": "float32",
    "seed": 0,
    "timm_compat_shim": True,
    "selective_scan_backend": "depthart_selective_scan.cross_selective_scan",
    "selective_scan_replaced": "network.tvimblock.cross_selective_scan",
}
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "授权",
    "scope": AUTHORITY_SCOPE,
}
EXPECTED_NEXT_STAGE_SELECTOR = {
    "selector_id": "TARO_R9_SOURCE_ONLY_CLEAR_ENRICHMENT_GRID_SEARCH_V1",
    "selector_content_sha256": "67FD8430418E23E4C974EBA4D7F49DCBD4DE66164A16491DE76F05AC974796CC",
    "rule_id": "02CE016D6B0011F0",
    "use": "PARENT_RANKING_ONLY_IN_SEPARATE_SUCCESSOR",
    "phase_a_scoring_performed": False,
}
EXPECTED_PHASE_CONTRACT = {
    "parent_count": PARENT_COUNT,
    "frame_count": FRAME_COUNT,
    "query_count": QUERY_COUNT,
    "pre_terminal_file_count": PRE_TERMINAL_FILE_COUNT,
    "final_file_count": FINAL_FILE_COUNT,
    "atomic_terminal_bundle": True,
    "allowed_member_payload_roles_by_phase": {
        "candidate": ["color", "intrinsics"],
        "source_feature": ["lowres_depth", "confidence"],
    },
    "forbidden_member_payload_roles": ["highres_depth"],
    "source_payload_read_attempts_on_success": {
        "color": FRAME_COUNT,
        "intrinsics": FRAME_COUNT,
        "lowres_depth": FRAME_COUNT,
        "confidence": FRAME_COUNT,
        "highres_depth": 0,
    },
    "source_payload_reads_completed_on_success": {
        "color": FRAME_COUNT,
        "intrinsics": FRAME_COUNT,
        "lowres_depth": FRAME_COUNT,
        "confidence": FRAME_COUNT,
        "highres_depth": 0,
    },
    "all_candidates_sealed_before_source_features": True,
    "all_source_r7_r11_records_sealed_before_parent_scoring": True,
    "r9_parent_scoring_performed": False,
    "top24_selection_performed": False,
    "faro_reads": 0,
    "truth_reads": 0,
}


class PhaseAValidationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PhaseAValidationError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhaseAValidationError("R11_PHASE_A_VALIDATION_JSON", f"JSON record cannot be read: {path}") from error
    require(isinstance(value, dict), "R11_PHASE_A_VALIDATION_JSON", f"JSON object required: {path}")
    return value


def _load_json_gzip(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseAValidationError("R11_PHASE_A_VALIDATION_JSON", f"gzip JSON record cannot be read: {path}") from error
    require(isinstance(value, dict), "R11_PHASE_A_VALIDATION_JSON", f"gzip JSON object required: {path}")
    return value


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R11_PHASE_A_VALIDATION_RECORD", "sealed record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(
        record.get("schema") == schema
        and isinstance(observed, str)
        and adapter.canonical_sha256(record) == observed,
        "R11_PHASE_A_VALIDATION_SEAL",
        f"record seal/schema drift: {schema}",
    )
    record["content_sha256"] = observed
    return record


def _relative_paths(parent_id: str, video_id: str, token: str) -> dict[str, str]:
    return {
        "candidate_input": f"candidate-inputs/{parent_id}/{video_id}/{token}.json",
        "candidate_blob": f"candidates/{parent_id}/{video_id}/{token}.depth.npy.gz",
        "candidate_record": f"candidates/{parent_id}/{video_id}/{token}.json",
        "source_receipt": f"phase-a-sources/{parent_id}/{video_id}/{token}.json",
        "lineage": f"phase-a-lineage/{parent_id}/{video_id}/{token}.json.gz",
    }


def _frame_rows(inventory: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    validated = run_pool_inventory.validate_inventory(inventory)
    rows = [
        (str(parent["visit_id"]), str(parent["video_id"]), str(token))
        for parent in validated["parents"]
        for token in parent["frame_plan"]["exact_timestamp_tokens"]
    ]
    require(len(rows) == len(set(rows)) == FRAME_COUNT, "R11_PHASE_A_VALIDATION_COHORT", "frame cohort drift")
    return rows


def _member_index_sha256(value: Mapping[str, Mapping[str, Any]]) -> str:
    return adapter.canonical_sha256(
        [
            {
                "role": role,
                "timestamp_token": token,
                "source_member_path": binding.source_member_path,
                "canonical_member_path": binding.canonical_member_path,
                "bytes": binding.bytes,
                "declared_crc32": binding.declared_crc32,
            }
            for role, members in sorted(value.items())
            for token, binding in sorted(members.items())
        ]
    )


def _intrinsics_index_sha256(value: Mapping[str, Any]) -> str:
    return adapter.canonical_sha256(
        [
            {
                "role": binding.role,
                "timestamp_token": token,
                "source_member_path": binding.source_member_path,
                "canonical_member_path": binding.canonical_member_path,
                "bytes": binding.bytes,
                "declared_crc32": binding.declared_crc32,
            }
            for token, binding in sorted(value.items())
        ]
    )


def _inventory_member_bindings(inventory: Mapping[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for parent in inventory["parents"]:
        visit_id, video_id = str(parent["visit_id"]), str(parent["video_id"])
        containers = parent["container_bindings"]
        paths = {role: _repo_path(binding["path"]) for role, binding in containers.items()}
        for role, path in paths.items():
            binding = containers[role]
            require(
                path.is_file()
                and path.stat().st_size == binding["bytes"]
                and materializer.sha256_file(path) == binding["sha256"],
                "R11_PHASE_A_VALIDATION_SOURCE_CONTAINER",
                f"source container differs from frozen inventory: {binding['path']}",
            )
        up_index, up_declared = run_pool_inventory.index_upsampling_archive_metadata_only(
            paths["upsampling"],
            video_id,
            maximum_declared_uncompressed_bytes=containers["upsampling"]["declared_uncompressed_bytes"],
        )
        intr_index, intr_declared = run_pool_inventory.index_intrinsics_archive_metadata_only(
            paths["intrinsics"],
            video_id,
            maximum_declared_uncompressed_bytes=containers["intrinsics"]["declared_uncompressed_bytes"],
        )
        require(
            up_declared == containers["upsampling"]["declared_uncompressed_bytes"]
            and intr_declared == containers["intrinsics"]["declared_uncompressed_bytes"]
            and _member_index_sha256(up_index) == containers["upsampling"]["recognized_member_index_sha256"]
            and _intrinsics_index_sha256(intr_index) == containers["intrinsics"]["recognized_member_index_sha256"],
            "R11_PHASE_A_VALIDATION_SOURCE_INDEX",
            "source central-directory index differs from frozen inventory",
        )
        trajectory_payload = paths["trajectory"].read_bytes()
        trajectory_rows = tuple(materializer.parse_trajectory_payload(trajectory_payload))
        with zipfile.ZipFile(paths["upsampling"]) as up_bundle, zipfile.ZipFile(paths["intrinsics"]) as intr_bundle:
            for token in parent["frame_plan"]["exact_timestamp_tokens"]:
                for role in ("color", "lowres_depth", "confidence"):
                    member = up_index[role][token]
                    payload = up_bundle.read(member.source_member_path)
                    require(
                        len(payload) == member.bytes and materializer.crc32_bytes(payload) == member.declared_crc32,
                        "R11_PHASE_A_VALIDATION_SOURCE_PAYLOAD",
                        "source payload differs from frozen member metadata",
                    )
                    result[(visit_id, video_id, token, role)] = {
                        "container_sha256": containers["upsampling"]["sha256"],
                        "member_path": member.source_member_path,
                        "bytes": member.bytes,
                        "crc32": member.declared_crc32,
                        "sha256": materializer.sha256_bytes(payload),
                        "decoded_sha256": adapter.canonical_sha256(
                            candidate_inputs._decode_color(payload)
                            if role == "color"
                            else materializer._decode_png(payload, role)
                        ),
                    }
                member = intr_index[token]
                payload = intr_bundle.read(member.source_member_path)
                require(
                    len(payload) == member.bytes and materializer.crc32_bytes(payload) == member.declared_crc32,
                    "R11_PHASE_A_VALIDATION_SOURCE_PAYLOAD",
                    "intrinsics payload differs from frozen member metadata",
                )
                result[(visit_id, video_id, token, "intrinsics")] = {
                    "container_sha256": containers["intrinsics"]["sha256"],
                    "member_path": member.source_member_path,
                    "bytes": member.bytes,
                    "crc32": member.declared_crc32,
                    "sha256": materializer.sha256_bytes(payload),
                    "parsed_intrinsics": materializer.parse_pincam_payload(payload),
                }
                transform, pose = adapter.interpolate_camera_to_world_exact(trajectory_rows, token)
                result[(visit_id, video_id, token, "trajectory")] = {
                    "binding": dict(containers["trajectory"]),
                    "camera_to_world_4x4": transform.tolist(),
                    "gravity_up_camera_xyz": adapter._normalize_vector(
                        transform[2, :3], "R11_VALIDATION_GRAVITY_INVALID"
                    ).tolist(),
                    "sensor_timestamp_ns": pose["frame_timestamp_ns"],
                    "max_source_timestamp_ns": pose["max_source_timestamp_ns"],
                }
    require(len(result) == FRAME_COUNT * 5, "R11_PHASE_A_VALIDATION_SOURCE_INDEX", "source binding index count drift")
    return result


def _receipt_matches_member(receipt: Any, expected: Mapping[str, Any]) -> bool:
    return (
        isinstance(receipt, dict)
        and receipt.get("container_sha256") == expected["container_sha256"]
        and receipt.get("member_path") == expected["member_path"]
        and receipt.get("bytes") == expected["bytes"]
        and receipt.get("crc32") == expected["crc32"]
        and receipt.get("sha256") == expected["sha256"]
    )


def _validate_factor_pair(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[Counter[str], Counter[str], int]:
    r7 = r7_positive.validate_positive_occupancy_factor(dict(base))
    r11 = abstention_candidate.validate_abstention_bundle(dict(candidate))
    require(
        r7["physical_frame_id"] == r11["physical_frame_id"]
        and r7["source_frame_record_sha256"] == r11["source_frame_record_sha256"],
        "R11_PHASE_A_VALIDATION_FACTOR_LINEAGE",
        "R7/R11 factor lineage drift",
    )
    r7_counts: Counter[str] = Counter()
    r11_counts: Counter[str] = Counter()
    for left, right in zip(r7["query_results"], r11["query_results"], strict=True):
        require(
            left["query_id"] == right["query_id"]
            and left["grid_index"] == right["grid_index"]
            and (right["state"] != "OCCUPIED_OBSERVED" or left["state"] == "OCCUPIED_OBSERVED"),
            "R11_PHASE_A_VALIDATION_FACTOR_SUBSET",
            "R11 positive is not an ordered subset of R7 positive",
        )
        r7_counts[left["state"]] += 1
        r11_counts[right["state"]] += 1
    abstained = r7_counts["OCCUPIED_OBSERVED"] - r11_counts["OCCUPIED_OBSERVED"]
    require(
        sum(r7_counts.values()) == sum(r11_counts.values()) == 9
        and r7_counts["CLEAR_OBSERVED"] == r11_counts["CLEAR_OBSERVED"] == 0
        and abstained >= 0
        and r11["base_positive_count"] == r7_counts["OCCUPIED_OBSERVED"]
        and r11["candidate_positive_count"] == r11_counts["OCCUPIED_OBSERVED"]
        and r11["abstained_base_positive_count"] == abstained,
        "R11_PHASE_A_VALIDATION_FACTOR_COUNTS",
        "R7/R11 factor count identity drift",
    )
    return r7_counts, r11_counts, abstained


def _validate_terminal(root: Path, terminal: Mapping[str, Any], expected_files: set[str]) -> dict[str, Any]:
    sealed = _validate_seal(terminal, "blindassist.taro.o1r.r11_fresh_pool_phase_a_terminal.v1")
    files = sealed.get("files")
    require(
        sealed.get("terminal") == PASS_TERMINAL
        and sealed.get("passed") is True
        and sealed.get("execution_valid") is True
        and sealed.get("one_shot_consumed") is True
        and isinstance(files, dict)
        and set(files) == expected_files
        and sealed.get("file_count_before_terminal") == len(files) == PRE_TERMINAL_FILE_COUNT,
        "R11_PHASE_A_VALIDATION_TERMINAL",
        "terminal identity/file set drift",
    )
    total = 0
    for relative, receipt in files.items():
        target = materializer.safe_join(root, relative)
        require(
            isinstance(receipt, dict)
            and receipt.get("path") == relative
            and target.is_file()
            and target.stat().st_size == receipt.get("bytes")
            and materializer.sha256_file(target) == receipt.get("sha256"),
            "R11_PHASE_A_VALIDATION_TERMINAL_FILE",
            f"terminal file binding drift: {relative}",
        )
        total += int(receipt["bytes"])
    require(total == sealed.get("bytes_before_terminal"), "R11_PHASE_A_VALIDATION_TERMINAL", "terminal byte sum drift")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    require(actual == expected_files | {"terminal.json"}, "R11_PHASE_A_VALIDATION_ROOT_SET", "evidence root file set drift")
    return sealed


def _git_bytes(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    require(completed.returncode == 0, "R11_PHASE_A_VALIDATION_BINDING", f"implementation commit lacks binding: {relative}")
    return completed.stdout


def _runtime_environment() -> dict[str, Any]:
    import cv2
    import timm
    import torch

    require(torch.cuda.is_available(), "R11_PHASE_A_VALIDATION_CUDA", "CUDA is unavailable to the independent validator")
    return {
        "python_executable": Path(sys.executable).resolve().as_posix(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "timm_version": str(timm.__version__),
        "numpy_version": str(np.__version__),
        "opencv_version": str(cv2.__version__),
        "cuda_available": True,
        "cuda_version": str(torch.version.cuda),
        "cuda_device_name": str(torch.cuda.get_device_name(torch.cuda.current_device())),
    }


def _validate_execution_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(lock)
    require(
        value.get("lock_id") == "TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK"
        and value.get("status") == "AUTHORIZED_UNCONSUMED"
        and value.get("consumed") is False
        and value.get("argv")
        == [
            "-m",
            "scripts.research.taro_o1r_r11_abstention_runtime.run_pool_phase_a",
            "--execution-lock",
            LOCK_RELATIVE,
        ]
        and value.get("inventory_path") == INVENTORY_PATH
        and value.get("output_root") == EVIDENCE_ROOT
        and value.get("overwrite") is False
        and value.get("rerun") is False,
        "R11_PHASE_A_VALIDATION_EXECUTION_LOCK",
        "execution lock identity/argv/root policy drift",
    )
    implementation_commit = value.get("implementation_commit")
    require(
        isinstance(implementation_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is not None,
        "R11_PHASE_A_VALIDATION_IMPLEMENTATION_COMMIT",
        "implementation commit malformed",
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    require(ancestor.returncode == 0, "R11_PHASE_A_VALIDATION_IMPLEMENTATION_COMMIT", "implementation commit is not an ancestor")
    bindings = value.get("bindings")
    require(
        isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDING_PATHS),
        "R11_PHASE_A_VALIDATION_BINDING",
        "execution lock binding count drift",
    )
    seen: set[str] = set()
    for row in bindings:
        require(isinstance(row, dict), "R11_PHASE_A_VALIDATION_BINDING", "binding row is not an object")
        role, relative = row.get("role"), row.get("path")
        require(
            set(row) == {"role", "path", "bytes", "sha256"}
            and isinstance(role, str)
            and role not in seen
            and EXPECTED_BINDING_PATHS.get(role) == relative,
            "R11_PHASE_A_VALIDATION_BINDING",
            "execution lock binding role/path drift",
        )
        target = _repo_path(str(relative))
        payload = target.read_bytes() if target.is_file() else b""
        require(
            len(payload) == row.get("bytes") and materializer.sha256_bytes(payload) == row.get("sha256"),
            "R11_PHASE_A_VALIDATION_BINDING",
            f"execution lock binding bytes drift: {relative}",
        )
        if role not in ARTIFACT_BINDING_ROLES:
            require(
                payload == _git_bytes(implementation_commit, str(relative)),
                "R11_PHASE_A_VALIDATION_BINDING",
                f"binding differs from implementation commit: {relative}",
            )
        seen.add(role)
    protocol = _validate_seal(
        _load_json(_repo_path(EXPECTED_BINDING_PATHS["R11_PROTOCOL"])),
        "blindassist.taro.o1r.r11_positive_occupancy_abstention_protocol_lock.v1",
    )
    authorization = _validate_seal(
        _load_json(_repo_path(EXPECTED_BINDING_PATHS["R11_DATA_USE_AUTHORIZATION"])),
        "blindassist.taro.o1r.r11_data_use_authorization_receipt.v1",
    )
    selector = _validate_seal(
        _load_json(_repo_path(EXPECTED_BINDING_PATHS["R9_SELECTOR_ARTIFACT"])),
        "blindassist.taro.o1r.r9_source_only_clear_enrichment_selector.v1",
    )
    user = value.get("user_authority")
    candidate = value.get("candidate_identity")
    runtime = value.get("expected_depthart_runtime_identity")
    environment = value.get("runtime_environment")
    require(
        protocol["content_sha256"] == value.get("protocol_content_sha256") == PROTOCOL_CONTENT_SHA256
        and authorization["content_sha256"] == value.get("authorization_receipt_content_sha256") == AUTHORIZATION_CONTENT_SHA256
        and value.get("pool_content_sha256") == POOL_CONTENT_SHA256
        and value.get("request_plan_sha256") == REQUEST_PLAN_SHA256
        and value.get("inventory_content_sha256") == INVENTORY_CONTENT_SHA256
        and value.get("inventory_result_content_sha256") == INVENTORY_RESULT_CONTENT_SHA256
        and value.get("inventory_manifest_content_sha256") == INVENTORY_MANIFEST_CONTENT_SHA256
        and value.get("inventory_formal_result_content_sha256") == INVENTORY_FORMAL_CONTENT_SHA256
        and value.get("execution_authority") == EXPECTED_AUTHORITY
        and value.get("resource_budget") == EXPECTED_RESOURCE_BUDGET
        and value.get("next_stage_selector") == EXPECTED_NEXT_STAGE_SELECTOR
        and value.get("phase_contract") == EXPECTED_PHASE_CONTRACT
        and user == EXPECTED_USER_AUTHORITY
        and selector["content_sha256"] == EXPECTED_NEXT_STAGE_SELECTOR["selector_content_sha256"]
        and selector.get("selector_id") == EXPECTED_NEXT_STAGE_SELECTOR["selector_id"]
        and selector.get("chosen_rule", {}).get("rule_id") == EXPECTED_NEXT_STAGE_SELECTOR["rule_id"]
        and selector.get("selection_uses_only_source_features") is True
        and selector.get("confirmation_authority") is False,
        "R11_PHASE_A_VALIDATION_LOCK_AUTHORITY",
        "execution lock predecessor/authority/selector drift",
    )
    require(
        candidate == EXPECTED_CANDIDATE_IDENTITY
        and environment == EXPECTED_RUNTIME_ENVIRONMENT
        and _runtime_environment() == EXPECTED_RUNTIME_ENVIRONMENT
        and runtime == EXPECTED_DEPTHART_RUNTIME_IDENTITY
        and candidate["model_id"] == adapter.BASELINE_MODEL_ID
        and candidate["source_commit"] == depthart_runner.EXPECTED_SOURCE_GIT_COMMIT
        and candidate["checkpoint_sha256"] == adapter.BASELINE_CHECKPOINT_SHA256
        and candidate["preprocess_id"] == depthart_runner.PREPROCESS_ID
        and candidate["postprocess_id"] == depthart_runner.POSTPROCESS_ID,
        "R11_PHASE_A_VALIDATION_RUNTIME_IDENTITY",
        "candidate/runtime identity drift",
    )
    source = Path(candidate["source_root"]).resolve()
    checkpoint = Path(candidate["checkpoint_path"]).resolve()
    require(
        source.is_dir()
        and checkpoint.is_file()
        and checkpoint.stat().st_size == candidate["checkpoint_bytes"]
        and materializer.sha256_file(checkpoint) == candidate["checkpoint_sha256"],
        "R11_PHASE_A_VALIDATION_CANDIDATE_ASSET",
        "candidate source/checkpoint asset drift",
    )
    source_commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    source_dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--short"], capture_output=True, text=True, check=True
    ).stdout.strip()
    require(
        source_commit == candidate["source_commit"] and not source_dirty,
        "R11_PHASE_A_VALIDATION_CANDIDATE_ASSET",
        "candidate source commit/cleanliness drift",
    )
    return value


def validate_evidence(root: Path | None = None, lock_path: Path | None = None) -> dict[str, Any]:
    evidence_root = (root or _repo_path(EVIDENCE_ROOT)).resolve()
    require(evidence_root.is_dir(), "R11_PHASE_A_VALIDATION_ROOT", "Phase A evidence root missing")
    inventory = run_pool_inventory.validate_inventory(_load_json(_repo_path(INVENTORY_PATH)))
    require(
        inventory["content_sha256"] == INVENTORY_CONTENT_SHA256
        and inventory["parent_count"] == PARENT_COUNT
        and inventory["exact_pose_bounded_frame_count"] == FRAME_COUNT,
        "R11_PHASE_A_VALIDATION_INVENTORY",
        "sealed inventory drift",
    )
    rows = _frame_rows(inventory)
    expected_files = {
        "execution-receipt.json",
        "candidate-completion.json",
        "phase-a-completion.json",
    }
    for parent_id, video_id, token in rows:
        expected_files.update(_relative_paths(parent_id, video_id, token).values())
    terminal = _validate_terminal(evidence_root, _load_json(evidence_root / "terminal.json"), expected_files)
    inventory_members = _inventory_member_bindings(inventory)

    execution = _validate_seal(
        _load_json(evidence_root / "execution-receipt.json"),
        "blindassist.taro.o1r.r11_fresh_pool_phase_a_execution_receipt.v1",
    )
    candidate_completion = _validate_seal(
        _load_json(evidence_root / "candidate-completion.json"),
        "blindassist.taro.o1r.r11_fresh_pool_candidate_completion.v1",
    )
    completion = _validate_seal(
        _load_json(evidence_root / "phase-a-completion.json"),
        "blindassist.taro.o1r.r11_fresh_pool_phase_a_completion.v1",
    )
    result = _validate_seal(
        terminal.get("result"),
        "blindassist.taro.o1r.r11_fresh_pool_phase_a_result.v1",
    )
    execution_lock_path = (lock_path or _repo_path(LOCK_RELATIVE)).resolve()
    require(
        execution_lock_path == _repo_path(LOCK_RELATIVE),
        "R11_PHASE_A_VALIDATION_EXECUTION_LOCK_PATH",
        "execution lock path drift",
    )
    execution_lock = _validate_execution_lock(
        _validate_seal(
            _load_json(execution_lock_path),
            "blindassist.taro.o1r.r11_fresh_pool_phase_a_execution_lock.v1",
        )
    )
    require(
        execution.get("execution_lock_sha256") == materializer.sha256_file(execution_lock_path)
        and execution.get("execution_lock_content_sha256") == execution_lock["content_sha256"]
        and execution_lock.get("lock_id") == "TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK"
        and execution_lock.get("status") == "AUTHORIZED_UNCONSUMED"
        and execution_lock.get("consumed") is False
        and execution_lock.get("inventory_path") == INVENTORY_PATH
        and execution_lock.get("output_root") == EVIDENCE_ROOT
        and execution_lock.get("overwrite") is False
        and execution_lock.get("rerun") is False
        and execution_lock.get("inventory_content_sha256") == INVENTORY_CONTENT_SHA256
        and execution_lock.get("phase_contract") == EXPECTED_PHASE_CONTRACT
        and execution_lock.get("execution_authority", {}).get("candidate_inference_count") == FRAME_COUNT
        and execution_lock.get("execution_authority", {}).get("highres_depth_member_payload_read") is False
        and execution_lock.get("execution_authority", {}).get("faro_payload_read") is False
        and execution_lock.get("execution_authority", {}).get("truth_scoring") is False
        and execution_lock.get("execution_authority", {}).get("training") is False
        and execution_lock.get("execution_authority", {}).get("network") is False
        and execution_lock.get("resource_budget", {}).get("maximum_wall_seconds") == 57_600
        and execution_lock.get("resource_budget", {}).get("maximum_peak_rss_bytes") == 17_179_869_184
        and execution_lock.get("resource_budget", {}).get("maximum_cuda_allocated_bytes") == 12_884_901_888
        and execution_lock.get("resource_budget", {}).get("maximum_evidence_bytes") == 2_147_483_648,
        "R11_PHASE_A_VALIDATION_EXECUTION_LOCK",
        "execution receipt/lock binding drift",
    )

    candidate_input_hashes: list[str] = []
    candidate_record_hashes: list[str] = []
    source_receipt_hashes: list[str] = []
    source_frame_hashes: list[str] = []
    prospective_hashes: list[str] = []
    reducer_hashes: list[str] = []
    r7_hashes: list[str] = []
    r11_hashes: list[str] = []
    r7_counts: Counter[str] = Counter()
    r11_counts: Counter[str] = Counter()
    abstained_total = 0
    runtime_identity_hashes: set[str] = set()
    parent_counts: dict[tuple[str, str], dict[str, Any]] = {
        identity: {"r7": Counter(), "r11": Counter(), "abstained": 0}
        for identity in [(visit, video) for visit, video, _rank in fresh_pool.EXPECTED_POOL]
    }

    for parent_id, video_id, token in rows:
        relative = _relative_paths(parent_id, video_id, token)
        physical_frame_id = f"{video_id}:{token}"
        candidate_input = _validate_seal(
            _load_json(evidence_root / relative["candidate_input"]),
            "blindassist.taro.o1r.r11_fresh_pool_candidate_input.v1",
        )
        candidate_record = _validate_seal(
            _load_json(evidence_root / relative["candidate_record"]),
            "blindassist.taro.o1r.r11_fresh_pool_candidate_frame.v1",
        )
        require(
            candidate_input["parent_id"] == candidate_record["parent_id"] == parent_id
            and candidate_input["video_id"] == candidate_record["video_id"] == video_id
            and candidate_input["timestamp_token"] == candidate_record["timestamp_token"] == token
            and candidate_input["physical_frame_id"] == candidate_record["physical_frame_id"] == physical_frame_id
            and candidate_record["candidate_input_sha256"] == candidate_input["content_sha256"]
            and candidate_input.get("allowed_model_inputs") == ["REGISTERED_RGB", "BOUND_EFFECTIVE_K"]
            and candidate_input.get("highres_depth_member_payload_read") is False
            and candidate_record.get("highres_depth_member_payload_read") is False
            and candidate_record.get("faro_payload_read") is False,
            "R11_PHASE_A_VALIDATION_CANDIDATE_LINEAGE",
            "candidate identity/authority drift",
        )
        expected_color = inventory_members[(parent_id, video_id, token, "color")]
        expected_intrinsics = inventory_members[(parent_id, video_id, token, "intrinsics")]
        expected_trajectory = inventory_members[(parent_id, video_id, token, "trajectory")]
        expected_lowres_intrinsics = expected_intrinsics["parsed_intrinsics"]
        require(
            _receipt_matches_member(candidate_input.get("color_binding"), expected_color)
            and _receipt_matches_member(candidate_input.get("intrinsics_binding"), expected_intrinsics)
            and candidate_input.get("color_decoded_sha256") == expected_color["decoded_sha256"]
            and candidate_input.get("lowres_intrinsics") == expected_lowres_intrinsics
            and candidate_input.get("intrinsics_highres")
            == adapter.scale_lowres_intrinsics(expected_lowres_intrinsics)
            and candidate_input.get("trajectory_binding") == expected_trajectory["binding"]
            and candidate_input.get("camera_to_world_4x4") == expected_trajectory["camera_to_world_4x4"]
            and candidate_input.get("gravity_up_camera_xyz") == expected_trajectory["gravity_up_camera_xyz"]
            and candidate_input.get("sensor_timestamp_ns") == expected_trajectory["sensor_timestamp_ns"]
            and candidate_input.get("max_source_timestamp_ns") == expected_trajectory["max_source_timestamp_ns"],
            "R11_PHASE_A_VALIDATION_SOURCE_BINDING",
            "candidate source receipt differs from frozen inventory metadata",
        )
        inference = _validate_seal(
            candidate_record["inference_receipt"],
            "blindassist.taro.o1r.r11_fresh_pool_depthart_inference.v1",
        )
        require(
            inference.get("model_id") == adapter.BASELINE_MODEL_ID
            and inference.get("checkpoint_sha256") == adapter.BASELINE_CHECKPOINT_SHA256
            and inference.get("preprocess_id") == depthart_runner.PREPROCESS_ID
            and inference.get("postprocess_id") == depthart_runner.POSTPROCESS_ID
            and inference.get("candidate_input_sha256") == candidate_input["content_sha256"]
            and inference.get("physical_frame_id") == physical_frame_id
            and inference.get("truth_alignment_used") is False
            and inference.get("highres_depth_member_payload_read") is False
            and inference.get("faro_payload_read") is False
            and isinstance(inference.get("runtime_identity"), dict),
            "R11_PHASE_A_VALIDATION_INFERENCE",
            "DepthART inference identity/firewall drift",
        )
        require(
            inference["runtime_identity"] == execution_lock.get("expected_depthart_runtime_identity"),
            "R11_PHASE_A_VALIDATION_RUNTIME_IDENTITY",
            "DepthART inference runtime identity differs from execution lock",
        )
        runtime_identity_hashes.add(adapter.canonical_sha256(inference["runtime_identity"]))
        blob = candidate_record["native_depth_blob"]
        blob_path = evidence_root / relative["candidate_blob"]
        payload = blob_path.read_bytes()
        require(
            blob.get("path") == relative["candidate_blob"]
            and len(payload) == blob.get("bytes")
            and materializer.sha256_bytes(payload) == blob.get("sha256"),
            "R11_PHASE_A_VALIDATION_CANDIDATE_BLOB",
            "candidate blob binding drift",
        )
        native = np.ascontiguousarray(depthart_runner.decode_npy_gzip_bytes(payload), dtype=np.float32)
        require(
            native.shape == depthart_runner.NATIVE_SHAPE_HW
            and bool(np.all(np.isfinite(native)))
            and adapter.canonical_sha256(native) == blob.get("array_sha256"),
            "R11_PHASE_A_VALIDATION_CANDIDATE_ARRAY",
            "candidate native array drift",
        )
        require(
            blob.get("shape_hw") == list(depthart_runner.NATIVE_SHAPE_HW)
            and blob.get("dtype") == "float32"
            and blob.get("encoding") == "DETERMINISTIC_GZIP_NPY_MTIME_0"
            and inference.get("native_depth_sha256") == adapter.canonical_sha256(native),
            "R11_PHASE_A_VALIDATION_CANDIDATE_ARRAY",
            "candidate blob metadata/inference hash drift",
        )
        candidate_depth_highres_m = depthart_runner.upsample_native_depth(native)
        require(
            adapter.canonical_sha256(candidate_depth_highres_m)
            == candidate_record["inference_receipt"].get("candidate_depth_highres_sha256"),
            "R11_PHASE_A_VALIDATION_CANDIDATE_ARRAY",
            "candidate high-resolution replay drift",
        )

        source = _validate_seal(
            _load_json(evidence_root / relative["source_receipt"]),
            "blindassist.taro.o1r.r11_fresh_pool_source_frame_receipt.v1",
        )
        lineage = _validate_seal(
            _load_json_gzip(evidence_root / relative["lineage"]),
            "blindassist.taro.o1r.r11_fresh_pool_phase_a_lineage.v1",
        )
        require(
            source["physical_frame_id"] == lineage["physical_frame_id"] == physical_frame_id
            and source["candidate_input_sha256"] == candidate_input["content_sha256"]
            and source["candidate_frame_record_sha256"] == candidate_record["content_sha256"]
            and lineage["source_frame_receipt_sha256"] == source["content_sha256"]
            and source.get("highres_depth_member_payload_read") is False
            and source.get("faro_payload_read") is False
            and source.get("truth_payload_read") is False
            and lineage.get("highres_depth_member_payload_read") is False
            and lineage.get("faro_payload_read") is False
            and lineage.get("truth_inputs") == 0,
            "R11_PHASE_A_VALIDATION_SOURCE_LINEAGE",
            "source/lineage identity or firewall drift",
        )
        expected_apple = inventory_members[(parent_id, video_id, token, "lowres_depth")]
        expected_confidence = inventory_members[(parent_id, video_id, token, "confidence")]
        require(
            _receipt_matches_member(source.get("apple_depth_binding"), expected_apple)
            and _receipt_matches_member(source.get("confidence_binding"), expected_confidence)
            and source.get("apple_depth_sha256") == expected_apple["decoded_sha256"]
            and source.get("confidence_sha256") == expected_confidence["decoded_sha256"],
            "R11_PHASE_A_VALIDATION_SOURCE_BINDING",
            "source feature receipt differs from frozen inventory metadata",
        )
        prospective_bundle = prospective.validate_prospective_factor_bundle(
            lineage["prospective_bundle"], candidate_highres_depth_m=candidate_depth_highres_m
        )
        reducer_bundle = reducer.validate_reducer_bundle(lineage["r6_reducer_bundle"])
        source_frame = r7_canary.validate_source_frame_record(lineage["r7_source_frame_record"])
        require(
            prospective_bundle["source_frame_receipt_sha256"] == source["content_sha256"]
            and prospective_bundle["candidate_frame_record_sha256"] == candidate_record["content_sha256"]
            and reducer_bundle["prospective_bundle_sha256"] == prospective_bundle["content_sha256"]
            and source_frame["physical_frame_id"] == physical_frame_id,
            "R11_PHASE_A_VALIDATION_NESTED_LINEAGE",
            "nested prospective/reducer/source lineage drift",
        )
        frame_r7, frame_r11, abstained = _validate_factor_pair(
            lineage["r7_positive_factor_bundle"], lineage["r11_abstention_bundle"]
        )
        require(
            lineage["r7_positive_factor_bundle"]["source_frame_record_sha256"] == source_frame["content_sha256"]
            and lineage["r11_abstention_bundle"]["source_frame_record_sha256"] == source_frame["content_sha256"],
            "R11_PHASE_A_VALIDATION_NESTED_LINEAGE",
            "nested factor/source hash drift",
        )
        candidate_input_hashes.append(candidate_input["content_sha256"])
        candidate_record_hashes.append(candidate_record["content_sha256"])
        source_receipt_hashes.append(source["content_sha256"])
        source_frame_hashes.append(source_frame["content_sha256"])
        prospective_hashes.append(prospective_bundle["content_sha256"])
        reducer_hashes.append(reducer_bundle["content_sha256"])
        r7_hashes.append(lineage["r7_positive_factor_bundle"]["content_sha256"])
        r11_hashes.append(lineage["r11_abstention_bundle"]["content_sha256"])
        r7_counts.update(frame_r7)
        r11_counts.update(frame_r11)
        abstained_total += abstained
        parent = parent_counts[(parent_id, video_id)]
        parent["r7"].update(frame_r7)
        parent["r11"].update(frame_r11)
        parent["abstained"] += abstained

    states = ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
    expected_r7 = {state: int(r7_counts[state]) for state in states}
    expected_r11 = {state: int(r11_counts[state]) for state in states}
    expected_parent_counts = [
        {
            "visit_id": identity[0],
            "video_id": identity[1],
            "frame_count": FROZEN_FRAME_COUNTS[index],
            "query_count": FROZEN_FRAME_COUNTS[index] * 9,
            "r7_state_counts": {state: int(parent_counts[identity]["r7"][state]) for state in states},
            "r11_state_counts": {state: int(parent_counts[identity]["r11"][state]) for state in states},
            "r11_abstained_base_positive_count": int(parent_counts[identity]["abstained"]),
        }
        for index, identity in enumerate(parent_counts)
    ]
    expected_ledger = {
        "attempts_by_role": {
            "color": FRAME_COUNT,
            "intrinsics": FRAME_COUNT,
            "lowres_depth": FRAME_COUNT,
            "confidence": FRAME_COUNT,
            "highres_depth": 0,
        },
        "completed_by_role": {
            "color": FRAME_COUNT,
            "intrinsics": FRAME_COUNT,
            "lowres_depth": FRAME_COUNT,
            "confidence": FRAME_COUNT,
            "highres_depth": 0,
        },
    }
    ledger = completion.get("source_payload_read_accounting")
    expected_roles = {"color", "intrinsics", "lowres_depth", "confidence", "highres_depth"}
    require(
        isinstance(ledger, dict)
        and ledger.get("attempts_by_role") == expected_ledger["attempts_by_role"]
        and ledger.get("completed_by_role") == expected_ledger["completed_by_role"]
        and isinstance(ledger.get("bytes_by_role"), dict)
        and set(ledger["bytes_by_role"]) == expected_roles
        and all(int(ledger["bytes_by_role"][role]) > 0 for role in expected_roles - {"highres_depth"})
        and ledger["bytes_by_role"]["highres_depth"] == 0
        and ledger.get("total_zip_member_payload_reads") == 4 * FRAME_COUNT
        and ledger.get("trajectory_payload_reads") == PARENT_COUNT
        and ledger.get("candidate_blob_reloads") == FRAME_COUNT
        and ledger.get("depthart_inferences") == FRAME_COUNT
        and all(ledger.get(key) == 0 for key in ("faro_values_interpreted", "truth_reads", "label_reads", "outcome_reads", "network_requests", "training_steps")),
        "R11_PHASE_A_VALIDATION_READ_LEDGER",
        "source payload read ledger drift",
    )
    sequence_fields = {
        "candidate_input_hash_sequence_sha256": candidate_input_hashes,
        "candidate_record_hash_sequence_sha256": candidate_record_hashes,
        "source_receipt_hash_sequence_sha256": source_receipt_hashes,
        "source_frame_hash_sequence_sha256": source_frame_hashes,
        "prospective_bundle_hash_sequence_sha256": prospective_hashes,
        "r6_reducer_hash_sequence_sha256": reducer_hashes,
        "r7_factor_hash_sequence_sha256": r7_hashes,
        "r11_factor_hash_sequence_sha256": r11_hashes,
    }
    require(
        all(completion.get(field) == adapter.canonical_sha256(values) for field, values in sequence_fields.items())
        and completion.get("parent_count") == PARENT_COUNT
        and completion.get("frame_count") == FRAME_COUNT
        and completion.get("query_count") == QUERY_COUNT
        and completion.get("inventory_content_sha256") == INVENTORY_CONTENT_SHA256
        and completion.get("r7_base_state_counts") == expected_r7
        and completion.get("r11_candidate_state_counts") == expected_r11
        and completion.get("r11_abstained_base_positive_count") == abstained_total
        and completion.get("per_parent_factor_counts") == expected_parent_counts
        and completion.get("highres_depth_member_payload_reads") == 0
        and completion.get("faro_reads") == completion.get("truth_reads") == 0
        and completion.get("clear_output_allowed") is False
        and completion.get("unknown_is_negative") is False
        and completion.get("all_candidate_records_sealed_before_source_features") is True
        and completion.get("all_r7_and_r11_records_sealed_before_parent_scoring") is True
        and completion.get("all_source_records_sealed_before_faro") is True
        and completion.get("r9_parent_scoring_performed") is False
        and completion.get("top24_selection_performed") is False
        and completion.get("training_steps") == completion.get("network_requests") == 0,
        "R11_PHASE_A_VALIDATION_COMPLETION",
        "Phase A completion aggregation/barrier drift",
    )
    require(
        candidate_completion.get("frame_count") == FRAME_COUNT
        and candidate_completion.get("candidate_input_hash_sequence_sha256") == adapter.canonical_sha256(candidate_input_hashes)
        and candidate_completion.get("candidate_record_hash_sequence_sha256") == adapter.canonical_sha256(candidate_record_hashes)
        and candidate_completion.get("payload_read_attempts") == {"color": FRAME_COUNT, "intrinsics": FRAME_COUNT}
        and candidate_completion.get("payload_reads_completed") == {"color": FRAME_COUNT, "intrinsics": FRAME_COUNT}
        and candidate_completion.get("candidate_inference_count") == FRAME_COUNT
        and candidate_completion.get("highres_depth_member_payload_reads") == 0
        and candidate_completion.get("faro_reads") == candidate_completion.get("truth_reads") == 0
        and candidate_completion.get("all_candidates_sealed_before_source_features") is True,
        "R11_PHASE_A_VALIDATION_CANDIDATE_COMPLETION",
        "candidate completion drift",
    )
    require(
        execution.get("expected_parent_count") == PARENT_COUNT
        and execution.get("expected_frame_count") == FRAME_COUNT
        and execution.get("expected_query_count") == QUERY_COUNT
        and execution.get("source_integrity_verification_begins_after_root_creation") is True
        and execution.get("highres_depth_member_payload_read") is False
        and execution.get("faro_payload_read") is False
        and execution.get("truth_scoring") is False
        and execution.get("parent_scoring") is False
        and execution.get("top24_selection") is False
        and execution.get("training_steps") == execution.get("network_requests") == 0
        and execution.get("one_shot_consumed_on_root_creation") is True,
        "R11_PHASE_A_VALIDATION_EXECUTION_RECEIPT",
        "execution receipt drift",
    )
    actual_evidence_bytes = sum(path.stat().st_size for path in evidence_root.rglob("*") if path.is_file())
    resource_budget = execution_lock["resource_budget"]
    require(
        result.get("terminal") == PASS_TERMINAL
        and result.get("passed") is True
        and result.get("execution_valid") is True
        and result.get("parent_count") == PARENT_COUNT
        and result.get("frame_count") == FRAME_COUNT
        and result.get("query_count") == QUERY_COUNT
        and result.get("candidate_inference_count") == FRAME_COUNT
        and result.get("r7_base_state_counts") == expected_r7
        and result.get("r11_candidate_state_counts") == expected_r11
        and result.get("r11_abstained_base_positive_count") == abstained_total
        and result.get("phase_a_completion_sha256") == completion["content_sha256"]
        and len(runtime_identity_hashes) == 1
        and result.get("runtime_identity_sha256") == next(iter(runtime_identity_hashes))
        and result.get("resource_budget") == resource_budget
        and isinstance(result.get("elapsed_seconds_before_terminal"), (int, float))
        and 0 <= float(result["elapsed_seconds_before_terminal"])
        <= resource_budget["maximum_wall_seconds"] - 60
        and isinstance(result.get("peak_rss_bytes"), int)
        and 0 < result["peak_rss_bytes"] <= resource_budget["maximum_peak_rss_bytes"]
        and isinstance(result.get("cuda_peak_allocated_bytes"), int)
        and 0 <= result["cuda_peak_allocated_bytes"] <= resource_budget["maximum_cuda_allocated_bytes"]
        and result.get("evidence_bytes_before_terminal") == terminal.get("bytes_before_terminal")
        and actual_evidence_bytes
        == terminal["bytes_before_terminal"] + (evidence_root / "terminal.json").stat().st_size
        and actual_evidence_bytes <= resource_budget["maximum_evidence_bytes"]
        and result.get("highres_depth_member_payload_reads") == result.get("faro_reads") == 0
        and result.get("truth_scoring") is False
        and result.get("r9_parent_scoring_performed") is False
        and result.get("top24_selection_performed") is False
        and result.get("clear_output_allowed") is False
        and result.get("unknown_is_negative") is False
        and result.get("training_steps") == result.get("network_requests") == 0
        and result.get("one_shot_consumed") is True,
        "R11_PHASE_A_VALIDATION_RESULT",
        "Phase A result drift",
    )
    return {
        "passed": True,
        "terminal": PASS_TERMINAL,
        "producer_module_imported": False,
        "parent_count": PARENT_COUNT,
        "frame_count": FRAME_COUNT,
        "query_count": QUERY_COUNT,
        "root_file_count": FINAL_FILE_COUNT,
        "r7_base_state_counts": expected_r7,
        "r11_candidate_state_counts": expected_r11,
        "r11_abstained_base_positive_count": abstained_total,
        "highres_depth_member_payload_reads": 0,
        "faro_reads": 0,
        "truth_reads": 0,
        "validator_allowed_source_payload_reads": 4 * FRAME_COUNT,
        "validator_highres_depth_member_payload_reads": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=None)
    parser.add_argument("--execution-lock", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = validate_evidence(args.evidence_root, args.execution_lock)
    except Exception as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "failure_code": str(getattr(error, "code", type(error).__name__)),
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

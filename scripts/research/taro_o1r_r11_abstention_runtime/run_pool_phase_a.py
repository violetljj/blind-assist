#!/usr/bin/env python3
"""Run sealed TARO R11 all-48 DepthART and source-only Phase A."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import json
import platform
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_factor_headroom_runtime import candidate_inputs, depthart_runner
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_reducer_integration_runtime import reducer_integration as r6_reducer
from scripts.research.taro_o1r_reducer_integration_runtime.locked_uncertainty import load_locked_uncertainty_model
from scripts.research.taro_o1r_r7_canary_runtime import positive_occupancy_factor as r7_positive
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r11_abstention_runtime import abstention_candidate
from scripts.research.taro_o1r_r11_abstention_runtime import fresh_pool
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_head
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_inventory


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_phase_a_execution_lock.v1"
LOCK_ID = "TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK"
LOCK_RELATIVE = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json"
INVENTORY_PATH = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-inventory-r0/exact-frame-plan.json"
INVENTORY_RESULT_PATH = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-inventory-r0/result.json"
INVENTORY_MANIFEST_PATH = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-inventory-r0/manifest.json"
INVENTORY_FORMAL_RESULT = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_INVENTORY_RESULT_2026-08-12.json"
INVENTORY_IMPLEMENTATION_LOCK = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_INVENTORY_IMPLEMENTATION_LOCK_2026-08-12.md"
PHASE_A_IMPLEMENTATION_LOCK = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_IMPLEMENTATION_LOCK_2026-08-12.md"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-phase-a-r0"
PASS_TERMINAL = "TARO_O1R_R11_FRESH_POOL_PHASE_A_SOURCE_ONLY_SEALED_PASS"
FAIL_TERMINAL = "TARO_O1R_R11_FRESH_POOL_PHASE_A_EXECUTION_INVALID"
ANALYSIS_ROLE = "R11_FRESH_CONFIRMATION_SOURCE_ONLY"
PARENT_COUNT = 48
FRAME_COUNT = 1043
QUERY_COUNT = FRAME_COUNT * 9
PRE_MANIFEST_FILE_COUNT = 5 * FRAME_COUNT + 4
FROZEN_FRAME_COUNTS = [
    20, 14, 23, 24, 29, 7, 12, 14, 10, 21, 28, 15, 11, 28, 29, 72,
    36, 14, 18, 4, 54, 32, 83, 17, 15, 16, 29, 10, 12, 34, 7, 14,
    11, 6, 9, 1, 46, 6, 27, 26, 50, 9, 11, 27, 12, 9, 28, 13,
]
FROZEN_MATERIALIZED_BYTES = 3_540_113_101
EXPECTED_PARENT_IDENTITIES = fresh_pool.EXPECTED_POOL
PROTOCOL_CONTENT_SHA256 = run_pool_inventory.PROTOCOL_CONTENT_SHA256
AUTHORIZATION_CONTENT_SHA256 = run_pool_inventory.AUTHORIZATION_CONTENT_SHA256
POOL_CONTENT_SHA256 = run_pool_inventory.POOL_CONTENT_SHA256
REQUEST_PLAN_SHA256 = run_pool_inventory.REQUEST_PLAN_SHA256
INVENTORY_CONTENT_SHA256 = "35156C2901A4CBEEDB6D611A56ABE3D711CEB68EF932480C21428BA4FF741600"
INVENTORY_RESULT_CONTENT_SHA256 = "C4F15A3EA4DC1C51463860B9510658620BA49086116F63EB9514FF89F9A494B1"
INVENTORY_MANIFEST_CONTENT_SHA256 = "59A1B3180E467266E16330D87C256F5D57B8D3C9BC2111DA9CD060DC043C01B8"
INVENTORY_FORMAL_CONTENT_SHA256 = "8961A155DF4FF23F882D1F1587C3516FD21ED2D137B026D5D8C6721E6A74D4AC"
AUTHORITY_SCOPE = run_pool_head.EXPECTED_USER_SCOPE

EXPECTED_ARGV = [
    "-m",
    "scripts.research.taro_o1r_r11_abstention_runtime.run_pool_phase_a",
    "--execution-lock",
    LOCK_RELATIVE,
]
EXPECTED_BINDINGS = {
    "R11_PROTOCOL": run_pool_head.PROTOCOL_RELATIVE,
    "R11_DATA_USE_AUTHORIZATION": run_pool_head.AUTHORIZATION_RELATIVE,
    "R11_POOL_PLANNER": "scripts/research/taro_o1r_r11_abstention_runtime/fresh_pool.py",
    "R11_INVENTORY_IMPLEMENTATION_LOCK": INVENTORY_IMPLEMENTATION_LOCK,
    "R11_INVENTORY_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_inventory.py",
    "R11_INVENTORY_PLAN": INVENTORY_PATH,
    "R11_INVENTORY_RESULT": INVENTORY_RESULT_PATH,
    "R11_INVENTORY_MANIFEST": INVENTORY_MANIFEST_PATH,
    "R11_INVENTORY_FORMAL_RESULT": INVENTORY_FORMAL_RESULT,
    "DEPTHART_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/depthart_runner.py",
    "CANDIDATE_INPUT_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/candidate_inputs.py",
    "PROSPECTIVE_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/prospective_factor_runtime.py",
    "R6_REDUCER_RUNTIME": "scripts/research/taro_o1r_reducer_integration_runtime/reducer_integration.py",
    "LOCKED_UNCERTAINTY_LOADER": "scripts/research/taro_o1r_reducer_integration_runtime/locked_uncertainty.py",
    "LOCKED_UNCERTAINTY_ARTIFACT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-artifact.json.gz",
    "LOCKED_UNCERTAINTY_RECEIPT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-receipt.json",
    "R7_SOURCE_FEATURE_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "R7_POSITIVE_FACTOR_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/positive_occupancy_factor.py",
    "R11_ABSTENTION_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/abstention_candidate.py",
    "R9_SELECTOR_RUNTIME": "scripts/research/taro_o1r_r9_clear_runtime/clear_enrichment_fit.py",
    "R9_SELECTOR_ARTIFACT": "artifacts.local/evidence/taro/o1r-r9-clear-enrichment-development-r0/selector.json",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "R11_PHASE_A_IMPLEMENTATION_LOCK": PHASE_A_IMPLEMENTATION_LOCK,
    "R11_PHASE_A_RUNNER": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_phase_a.py",
    "R11_PHASE_A_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_run_pool_phase_a.py",
    "R11_PHASE_A_INDEPENDENT_VALIDATOR": "scripts/research/taro_o1r_r11_abstention_runtime/validate_pool_phase_a.py",
    "R11_PHASE_A_VALIDATOR_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_validate_pool_phase_a.py",
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
EXPECTED_NEXT_STAGE_SELECTOR = {
    "selector_id": "TARO_R9_SOURCE_ONLY_CLEAR_ENRICHMENT_GRID_SEARCH_V1",
    "selector_content_sha256": "67FD8430418E23E4C974EBA4D7F49DCBD4DE66164A16491DE76F05AC974796CC",
    "rule_id": "02CE016D6B0011F0",
    "use": "PARENT_RANKING_ONLY_IN_SEPARATE_SUCCESSOR",
    "phase_a_scoring_performed": False,
}


class FreshPhaseAError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FreshPhaseAError(code, message, **context)


@dataclass(frozen=True)
class PhaseAMemberRef:
    role: str
    path: str
    bytes: int
    crc32: str


@dataclass(frozen=True)
class PhaseAFrameRef:
    parent_id: str
    video_id: str
    timestamp_token: str
    physical_frame_id: str
    upsampling_archive: Path
    intrinsics_archive: Path
    trajectory_path: Path
    container_bindings: dict[str, Any]
    trajectory_rows: tuple[dict[str, Any], ...]
    payload_members: dict[str, PhaseAMemberRef]

    def __post_init__(self) -> None:
        require(
            set(self.payload_members) == {"color", "intrinsics", "lowres_depth", "confidence"},
            "R11_PHASE_A_FRAME_CAPABILITY",
            "Phase A frame retained a forbidden or missing payload capability",
        )


@dataclass
class PayloadReadLedger:
    attempts_by_role: Counter[str] = field(default_factory=Counter)
    completed_by_role: Counter[str] = field(default_factory=Counter)
    bytes_by_role: Counter[str] = field(default_factory=Counter)

    def record_attempt(self, role: str) -> None:
        self.attempts_by_role[role] += 1

    def record_completed(self, role: str, payload_bytes: int) -> None:
        self.completed_by_role[role] += 1
        self.bytes_by_role[role] += int(payload_bytes)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R11_PHASE_A_SEAL_COLLISION", "caller supplied a content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R11_PHASE_A_RECORD_INVALID", "sealed record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(
        record.get("schema") == schema
        and isinstance(observed, str)
        and re.fullmatch(r"[0-9A-F]{64}", observed) is not None
        and adapter.canonical_sha256(record) == observed,
        "R11_PHASE_A_RECORD_HASH_DRIFT",
        "sealed record hash/schema drift",
        schema=schema,
    )
    record["content_sha256"] = observed
    return record


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "R11_PHASE_A_JSON_INVALID", "JSON record must be an object", path=str(path))
    return value


def _load_json_gzip(path: Path) -> dict[str, Any]:
    value = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    require(isinstance(value, dict), "R11_PHASE_A_JSON_INVALID", "gzip JSON record must be an object", path=str(path))
    return value


def _candidate_input_relative(frame: PhaseAFrameRef) -> str:
    return f"candidate-inputs/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def _candidate_blob_relative(frame: PhaseAFrameRef) -> str:
    return f"candidates/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.depth.npy.gz"


def _candidate_record_relative(frame: PhaseAFrameRef) -> str:
    return f"candidates/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def _source_receipt_relative(frame: PhaseAFrameRef) -> str:
    return f"phase-a-sources/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def _lineage_relative(frame: PhaseAFrameRef) -> str:
    return f"phase-a-lineage/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def _inventory_frame_keys(inventory: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    validated = run_pool_inventory.validate_inventory(inventory)
    keys = [
        (str(parent["visit_id"]), str(parent["video_id"]), str(token))
        for parent in validated["parents"]
        for token in parent["frame_plan"]["exact_timestamp_tokens"]
    ]
    require(len(keys) == FRAME_COUNT and len(set(keys)) == FRAME_COUNT, "R11_PHASE_A_COHORT_DRIFT", "R11 cohort is not exact and unique 48/1043")
    return keys


def _verify_inventory_evidence() -> dict[str, Any]:
    root = _repo_path(run_pool_inventory.OUTPUT_ROOT)
    manifest = _validate_seal(_load_json(_repo_path(INVENTORY_MANIFEST_PATH)), "blindassist.taro.o1r.r11_fresh_pool_inventory_manifest.v1")
    require(
        manifest["content_sha256"] == INVENTORY_MANIFEST_CONTENT_SHA256
        and manifest.get("terminal") == run_pool_inventory.PASS_TERMINAL
        and manifest.get("one_shot_consumed") is True,
        "R11_PHASE_A_INVENTORY_MANIFEST",
        "R11 inventory manifest identity drift",
    )
    files = manifest.get("files")
    require(
        isinstance(files, dict)
        and set(files) == {"start-receipt.json", "exact-frame-plan.json", "result.json"},
        "R11_PHASE_A_INVENTORY_MANIFEST",
        "R11 inventory manifest file set drift",
    )
    for relative, receipt in files.items():
        target = materializer.safe_join(root, relative)
        require(
            isinstance(receipt, dict)
            and target.is_file()
            and target.stat().st_size == receipt.get("bytes")
            and materializer.sha256_file(target) == receipt.get("sha256"),
            "R11_PHASE_A_INVENTORY_FILE",
            f"R11 inventory artifact drift: {relative}",
        )
    inventory = run_pool_inventory.validate_inventory(_load_json(_repo_path(INVENTORY_PATH)))
    result = _validate_seal(_load_json(_repo_path(INVENTORY_RESULT_PATH)), "blindassist.taro.o1r.r11_fresh_pool_inventory_result.v1")
    formal = _validate_seal(_load_json(_repo_path(INVENTORY_FORMAL_RESULT)), "blindassist.taro.o1r.r11_fresh_pool_inventory_formal_result.v1")
    require(
        inventory["content_sha256"] == INVENTORY_CONTENT_SHA256
        and result["content_sha256"] == INVENTORY_RESULT_CONTENT_SHA256
        and formal["content_sha256"] == INVENTORY_FORMAL_CONTENT_SHA256,
        "R11_PHASE_A_INVENTORY_CONTENT",
        "R11 inventory content seal drift",
    )
    require(
        result.get("passed") is True
        and result.get("execution_valid") is True
        and result.get("phase_a_ready") is True
        and result.get("parent_count") == PARENT_COUNT
        and result.get("exact_pose_bounded_frame_count") == FRAME_COUNT
        and result.get("per_parent_frame_counts") == FROZEN_FRAME_COUNTS
        and result.get("declared_materialized_bytes") == FROZEN_MATERIALIZED_BYTES
        and result.get("inventory_content_sha256") == inventory["content_sha256"]
        and result.get("zip_member_payload_reads") == 0
        and result.get("highres_depth_member_payload_reads") == 0
        and result.get("pixel_arrays_decoded") is False
        and result.get("faro_values_interpreted") is False
        and result.get("truth_values_interpreted") is False
        and result.get("model_outputs_read") is False
        and result.get("training") is False,
        "R11_PHASE_A_INVENTORY_RESULT",
        "R11 inventory result is not admitted",
    )
    evidence = formal.get("evidence")
    summary = formal.get("inventory_summary")
    firewall = formal.get("phase_firewall")
    require(
        formal.get("passed") is True
        and formal.get("status") == run_pool_inventory.PASS_TERMINAL
        and formal.get("unique_successor") == "TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_IMPLEMENTATION_LOCK"
        and isinstance(evidence, dict)
        and evidence.get("inventory", {}).get("content_sha256") == inventory["content_sha256"]
        and evidence.get("result", {}).get("content_sha256") == result["content_sha256"]
        and evidence.get("manifest", {}).get("content_sha256") == manifest["content_sha256"]
        and isinstance(summary, dict)
        and summary.get("parent_count") == PARENT_COUNT
        and summary.get("exact_pose_bounded_frame_count") == FRAME_COUNT
        and summary.get("per_parent_frame_counts") == FROZEN_FRAME_COUNTS
        and isinstance(firewall, dict)
        and firewall.get("zip_member_payload_reads") == 0
        and firewall.get("highres_depth_member_payload_reads") == 0,
        "R11_PHASE_A_INVENTORY_FORMAL_RESULT",
        "R11 formal inventory result drift",
    )
    _inventory_frame_keys(inventory)
    return inventory


def _verify_container(path: Path, binding: Mapping[str, Any]) -> None:
    require(
        path.is_file()
        and path.stat().st_size == int(binding["bytes"])
        and materializer.sha256_file(path) == binding["sha256"],
        "R11_PHASE_A_CONTAINER_BINDING_DRIFT",
        "R11 source container differs from inventory",
        path=str(path),
    )


def _member_index_sha256(value: Mapping[str, Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for role, members in sorted(value.items()):
        for token, binding in sorted(members.items()):
            rows.append(
                {
                    "role": role,
                    "timestamp_token": token,
                    "source_member_path": binding.source_member_path,
                    "canonical_member_path": binding.canonical_member_path,
                    "bytes": binding.bytes,
                    "declared_crc32": binding.declared_crc32,
                }
            )
    return adapter.canonical_sha256(rows)


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


def _phase_member(binding: Any) -> PhaseAMemberRef:
    return PhaseAMemberRef(
        role=str(binding.role),
        path=str(binding.source_member_path),
        bytes=int(binding.bytes),
        crc32=str(binding.declared_crc32),
    )


def _load_frames(inventory: Mapping[str, Any]) -> list[PhaseAFrameRef]:
    """Verify source containers and build frame refs only after root activation."""

    frames: list[PhaseAFrameRef] = []
    for parent, expected, expected_count in zip(
        inventory["parents"], EXPECTED_PARENT_IDENTITIES, FROZEN_FRAME_COUNTS, strict=True
    ):
        identity = (str(parent["visit_id"]), str(parent["video_id"]))
        require(identity == expected[:2], "R11_PHASE_A_ROSTER_DRIFT", "R11 inventory roster drift")
        bindings = parent["container_bindings"]
        up_path = _repo_path(bindings["upsampling"]["path"])
        intr_path = _repo_path(bindings["intrinsics"]["path"])
        traj_path = _repo_path(bindings["trajectory"]["path"])
        _verify_container(up_path, bindings["upsampling"])
        _verify_container(intr_path, bindings["intrinsics"])
        _verify_container(traj_path, bindings["trajectory"])
        up_index, up_declared = run_pool_inventory.index_upsampling_archive_metadata_only(
            up_path,
            identity[1],
            maximum_declared_uncompressed_bytes=int(bindings["upsampling"]["declared_uncompressed_bytes"]),
        )
        intr_index, intr_declared = run_pool_inventory.index_intrinsics_archive_metadata_only(
            intr_path,
            identity[1],
            maximum_declared_uncompressed_bytes=int(bindings["intrinsics"]["declared_uncompressed_bytes"]),
        )
        require(
            up_declared == bindings["upsampling"]["declared_uncompressed_bytes"]
            and intr_declared == bindings["intrinsics"]["declared_uncompressed_bytes"]
            and _member_index_sha256(up_index) == bindings["upsampling"]["recognized_member_index_sha256"]
            and _intrinsics_index_sha256(intr_index) == bindings["intrinsics"]["recognized_member_index_sha256"],
            "R11_PHASE_A_MEMBER_INDEX_DRIFT",
            "R11 source member index differs from sealed inventory",
        )
        trajectory = tuple(materializer.parse_trajectory_payload(traj_path.read_bytes()))
        tokens = parent["frame_plan"]["exact_timestamp_tokens"]
        require(len(tokens) == expected_count, "R11_PHASE_A_FRAME_COUNT_DRIFT", "R11 parent frame count drift")
        for token in tokens:
            require(
                all(token in up_index[role] for role in ("color", "highres_depth", "lowres_depth", "confidence"))
                and token in intr_index,
                "R11_PHASE_A_MEMBER_MISSING",
                "R11 exact source member is absent",
                token=token,
            )
            # Validate highres/FARO metadata existence, then deliberately discard
            # that capability from the Phase-A frame object.
            members = {
                "color": _phase_member(up_index["color"][token]),
                "lowres_depth": _phase_member(up_index["lowres_depth"][token]),
                "confidence": _phase_member(up_index["confidence"][token]),
                "intrinsics": _phase_member(intr_index[token]),
            }
            frames.append(
                PhaseAFrameRef(
                    identity[0], identity[1], token, f"{identity[1]}:{token}", up_path,
                    intr_path, traj_path, {key: dict(value) for key, value in bindings.items()},
                    trajectory, members,
                )
            )
    require(len(frames) == FRAME_COUNT, "R11_PHASE_A_COHORT_DRIFT", "R11 cohort is not exact 48/1043")
    require(
        [(row.parent_id, row.video_id, row.timestamp_token) for row in frames] == _inventory_frame_keys(inventory),
        "R11_PHASE_A_COHORT_ORDER_DRIFT",
        "R11 frame order drift",
    )
    return frames


def _frames_by_parent(frames: Sequence[PhaseAFrameRef]) -> list[tuple[tuple[str, str], list[PhaseAFrameRef]]]:
    grouped: dict[tuple[str, str], list[PhaseAFrameRef]] = {}
    for frame in frames:
        grouped.setdefault((frame.parent_id, frame.video_id), []).append(frame)
    expected = [(visit, video) for visit, video, _rank in EXPECTED_PARENT_IDENTITIES]
    require(set(grouped) == set(expected), "R11_PHASE_A_PARENT_GROUP_DRIFT", "R11 parent group identity drift")
    result = [(identity, grouped[identity]) for identity in expected]
    require(
        [len(rows) for _identity, rows in result] == FROZEN_FRAME_COUNTS,
        "R11_PHASE_A_PARENT_GROUP_DRIFT",
        "R11 parent group frame count drift",
    )
    return result


def _member_binding(frame: PhaseAFrameRef, role: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    container_role = "intrinsics" if role == "intrinsics" else "upsampling"
    return {"container_sha256": frame.container_bindings[container_role]["sha256"], **dict(binding)}


def _read_allowed_member(
    bundle: zipfile.ZipFile,
    frame: PhaseAFrameRef,
    role: str,
    phase: str,
    ledger: PayloadReadLedger,
) -> tuple[bytes, dict[str, Any]]:
    ledger.record_attempt(role)
    allowed = {
        "CANDIDATE": {"color", "intrinsics"},
        "SOURCE_FEATURE": {"lowres_depth", "confidence"},
    }
    require(phase in allowed, "R11_PHASE_A_READER_PHASE", "unknown Phase A reader phase", phase=phase)
    require(
        role in allowed[phase],
        "R11_PHASE_A_PAYLOAD_FIREWALL",
        f"{phase} attempted a forbidden source payload",
        phase=phase,
        role=role,
    )
    require(
        role in frame.payload_members,
        "R11_PHASE_A_FRAME_CAPABILITY",
        "Phase A frame lacks requested payload capability",
        role=role,
    )
    member = frame.payload_members[role]
    require(member.role == role, "R11_PHASE_A_MEMBER_ROLE_DRIFT", "Phase A member role/path binding drift")
    try:
        info = bundle.getinfo(member.path)
    except KeyError as error:
        raise FreshPhaseAError("R11_PHASE_A_MEMBER_MISSING", "bound source member is absent", role=role) from error
    require(
        not info.is_dir()
        and int(info.file_size) == member.bytes
        and f"{info.CRC:08X}" == member.crc32,
        "R11_PHASE_A_MEMBER_BINDING_DRIFT",
        "source member central-directory binding drift",
        role=role,
    )
    with bundle.open(info, "r") as stream:
        payload = stream.read(member.bytes + 1)
    require(
        len(payload) == member.bytes
        and materializer.crc32_bytes(payload) == member.crc32,
        "R11_PHASE_A_MEMBER_BINDING_DRIFT",
        "source member payload bytes/CRC drift",
        role=role,
    )
    ledger.record_completed(role, len(payload))
    return payload, {
        "member_path": member.path,
        "bytes": len(payload),
        "sha256": materializer.sha256_bytes(payload),
        "crc32": member.crc32,
    }


def _read_candidate_input(
    frame: PhaseAFrameRef,
    up_bundle: zipfile.ZipFile,
    intr_bundle: zipfile.ZipFile,
    ledger: PayloadReadLedger,
) -> tuple[dict[str, Any], np.ndarray]:
    color_payload, color_binding = _read_allowed_member(up_bundle, frame, "color", "CANDIDATE", ledger)
    intr_payload, intr_binding = _read_allowed_member(intr_bundle, frame, "intrinsics", "CANDIDATE", ledger)
    color = np.ascontiguousarray(candidate_inputs._decode_color(color_payload))
    low = materializer.parse_pincam_payload(intr_payload)
    high = adapter.scale_lowres_intrinsics(low)
    transform, pose = adapter.interpolate_camera_to_world_exact(frame.trajectory_rows, frame.timestamp_token)
    gravity = adapter._normalize_vector(transform[2, :3], "R11_FRESH_GRAVITY_INVALID")
    record = _seal(
        {
            "schema": "blindassist.taro.o1r.r11_fresh_pool_candidate_input.v1",
            "analysis_role": ANALYSIS_ROLE,
            "parent_id": frame.parent_id,
            "video_id": frame.video_id,
            "timestamp_token": frame.timestamp_token,
            "physical_frame_id": frame.physical_frame_id,
            "color_binding": _member_binding(frame, "color", color_binding),
            "intrinsics_binding": _member_binding(frame, "intrinsics", intr_binding),
            "trajectory_binding": dict(frame.container_bindings["trajectory"]),
            "color_decoded_sha256": adapter.canonical_sha256(color),
            "lowres_intrinsics": low,
            "intrinsics_highres": high,
            "camera_to_world_4x4": transform.tolist(),
            "gravity_up_camera_xyz": gravity.tolist(),
            "sensor_timestamp_ns": pose["frame_timestamp_ns"],
            "max_source_timestamp_ns": pose["max_source_timestamp_ns"],
            "allowed_model_inputs": ["REGISTERED_RGB", "BOUND_EFFECTIVE_K"],
            "highres_depth_member_payload_read": False,
            "faro_payload_read": False,
            "truth_payload_read": False,
            "prior_outcome_read": False,
        }
    )
    return record, color


def _run_candidate(
    model: Any,
    runtime_identity: Mapping[str, Any],
    candidate_input: Mapping[str, Any],
    color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    matrix = np.asarray(candidate_input["intrinsics_highres"]["matrix_3x3"], dtype=np.float32)
    tensor, resized_k = depthart_runner.preprocess_depthart_input(color, matrix)
    with torch.inference_mode():
        prediction = model(torch.from_numpy(tensor).to("cuda"), torch.from_numpy(resized_k).to("cuda"))
    native_batch = prediction.detach().float().cpu().numpy()
    require(native_batch.shape == (1, *depthart_runner.NATIVE_SHAPE_HW), "R11_NATIVE_DEPTH_INVALID", "DepthART native shape drift")
    native = np.ascontiguousarray(native_batch[0], dtype=np.float32)
    require(bool(np.all(np.isfinite(native))), "R11_NATIVE_DEPTH_INVALID", "DepthART output contains non-finite values")
    candidate_depth_highres_m = depthart_runner.upsample_native_depth(native)
    inference = _seal(
        {
            "schema": "blindassist.taro.o1r.r11_fresh_pool_depthart_inference.v1",
            "analysis_role": ANALYSIS_ROLE,
            "model_id": adapter.BASELINE_MODEL_ID,
            "checkpoint_sha256": adapter.BASELINE_CHECKPOINT_SHA256,
            "preprocess_id": depthart_runner.PREPROCESS_ID,
            "postprocess_id": depthart_runner.POSTPROCESS_ID,
            "candidate_input_sha256": candidate_input["content_sha256"],
            "parent_id": candidate_input["parent_id"],
            "video_id": candidate_input["video_id"],
            "timestamp_token": candidate_input["timestamp_token"],
            "physical_frame_id": candidate_input["physical_frame_id"],
            "input_tensor_sha256": adapter.canonical_sha256(tensor),
            "resized_intrinsics_sha256": adapter.canonical_sha256(resized_k),
            "native_depth_sha256": adapter.canonical_sha256(native),
            "candidate_depth_highres_sha256": adapter.canonical_sha256(candidate_depth_highres_m),
            "runtime_identity": dict(runtime_identity),
            "truth_alignment_used": False,
            "highres_depth_member_payload_read": False,
            "faro_payload_read": False,
        }
    )
    return native, candidate_depth_highres_m, inference


def _load_candidate(root: Path, frame: PhaseAFrameRef) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    candidate_input = _validate_seal(
        _load_json(root / _candidate_input_relative(frame)),
        "blindassist.taro.o1r.r11_fresh_pool_candidate_input.v1",
    )
    record = _validate_seal(
        _load_json(root / _candidate_record_relative(frame)),
        "blindassist.taro.o1r.r11_fresh_pool_candidate_frame.v1",
    )
    require(
        record["candidate_input_sha256"] == candidate_input["content_sha256"]
        and record["physical_frame_id"] == frame.physical_frame_id,
        "R11_CANDIDATE_LINEAGE_DRIFT",
        "candidate lineage drift",
    )
    blob = record["native_depth_blob"]
    require(blob.get("path") == _candidate_blob_relative(frame), "R11_CANDIDATE_BLOB_DRIFT", "candidate blob path drift")
    payload = (root / blob["path"]).read_bytes()
    require(
        len(payload) == blob["bytes"] and materializer.sha256_bytes(payload) == blob["sha256"],
        "R11_CANDIDATE_BLOB_DRIFT",
        "candidate blob hash drift",
    )
    native = np.ascontiguousarray(depthart_runner.decode_npy_gzip_bytes(payload), dtype=np.float32)
    require(
        native.shape == depthart_runner.NATIVE_SHAPE_HW
        and adapter.canonical_sha256(native) == blob["array_sha256"],
        "R11_CANDIDATE_ARRAY_DRIFT",
        "candidate native array drift",
    )
    candidate_depth_highres_m = depthart_runner.upsample_native_depth(native)
    require(
        adapter.canonical_sha256(candidate_depth_highres_m)
        == record["inference_receipt"]["candidate_depth_highres_sha256"],
        "R11_CANDIDATE_HIGHRES_DRIFT",
        "candidate high-resolution replay drift",
    )
    return candidate_input, record, candidate_depth_highres_m


def _runtime_environment() -> dict[str, Any]:
    import cv2
    import timm
    import torch

    require(torch.cuda.is_available(), "R11_PHASE_A_CUDA_UNAVAILABLE", "R11 Phase A requires CUDA")
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


def _git_bytes(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    require(completed.returncode == 0, "R11_PHASE_A_IMPLEMENTATION_BINDING", f"implementation commit lacks binding: {relative}")
    return completed.stdout


def _validate_implementation_ancestor(value: Any) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None, "R11_PHASE_A_IMPLEMENTATION_COMMIT", "implementation commit invalid")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", value, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    require(completed.returncode == 0, "R11_PHASE_A_IMPLEMENTATION_COMMIT", "implementation commit is not an ancestor of HEAD")
    return value


def validate_execution_lock(path: Path, *, require_output_absent: bool = True) -> dict[str, Any]:
    lock_path = path.resolve()
    require(lock_path == _repo_path(LOCK_RELATIVE), "R11_PHASE_A_LOCK_PATH", "Phase A lock path drift")
    lock = _validate_seal(_load_json(lock_path), LOCK_SCHEMA)
    require(
        lock.get("lock_id") == LOCK_ID
        and lock.get("status") == "AUTHORIZED_UNCONSUMED"
        and lock.get("consumed") is False,
        "R11_PHASE_A_LOCK_IDENTITY",
        "R11 Phase A lock identity drift",
    )
    require(
        lock.get("argv") == EXPECTED_ARGV
        and lock.get("inventory_path") == INVENTORY_PATH
        and lock.get("output_root") == OUTPUT_ROOT
        and lock.get("overwrite") is False
        and lock.get("rerun") is False,
        "R11_PHASE_A_ROOT_DRIFT",
        "R11 Phase A argv/root policy drift",
    )
    original_argv = [str(value) for value in getattr(sys, "orig_argv", [])]
    require("-m" in original_argv, "R11_PHASE_A_ARGV_DRIFT", "R11 Phase A must use the frozen module-form argv")
    module_index = original_argv.index("-m")
    require(
        original_argv[module_index:] == EXPECTED_ARGV,
        "R11_PHASE_A_ARGV_DRIFT",
        "actual R11 Phase A module argv drift",
    )
    implementation_commit = _validate_implementation_ancestor(lock.get("implementation_commit"))
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R11_PHASE_A_BINDINGS", "R11 Phase A binding count drift")
    verified: dict[str, dict[str, Any]] = {}
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(
            set(row) == {"role", "path", "bytes", "sha256"}
            and role not in verified
            and EXPECTED_BINDINGS.get(role) == relative,
            "R11_PHASE_A_BINDING_ROW",
            "R11 Phase A binding row drift",
        )
        target = _repo_path(relative)
        payload = target.read_bytes() if target.is_file() else b""
        require(
            len(payload) == row.get("bytes") and materializer.sha256_bytes(payload) == row.get("sha256"),
            "R11_PHASE_A_BINDING_HASH",
            f"R11 Phase A binding drift: {relative}",
        )
        if role not in ARTIFACT_BINDING_ROLES:
            require(
                payload == _git_bytes(implementation_commit, relative),
                "R11_PHASE_A_BINDING_HASH",
                f"implementation-commit binding drift: {relative}",
            )
        verified[str(role)] = dict(row)
    inventory = _verify_inventory_evidence()
    require(
        lock.get("protocol_content_sha256") == PROTOCOL_CONTENT_SHA256
        and lock.get("authorization_receipt_content_sha256") == AUTHORIZATION_CONTENT_SHA256
        and lock.get("pool_content_sha256") == POOL_CONTENT_SHA256
        and lock.get("request_plan_sha256") == REQUEST_PLAN_SHA256
        and lock.get("inventory_content_sha256") == INVENTORY_CONTENT_SHA256
        and lock.get("inventory_result_content_sha256") == INVENTORY_RESULT_CONTENT_SHA256
        and lock.get("inventory_manifest_content_sha256") == INVENTORY_MANIFEST_CONTENT_SHA256
        and lock.get("inventory_formal_result_content_sha256") == INVENTORY_FORMAL_CONTENT_SHA256,
        "R11_PHASE_A_PREDECESSOR_DRIFT",
        "R11 Phase A predecessor binding drift",
    )
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "R11_PHASE_A_AUTHORITY_DRIFT", "R11 Phase A authority drift")
    require(lock.get("resource_budget") == EXPECTED_RESOURCE_BUDGET, "R11_PHASE_A_BUDGET_DRIFT", "R11 Phase A resource budget drift")
    require(
        lock.get("next_stage_selector") == EXPECTED_NEXT_STAGE_SELECTOR,
        "R11_PHASE_A_NEXT_STAGE_SELECTOR_DRIFT",
        "frozen next-stage R9 selector identity drift",
    )
    selector = _validate_seal(
        _load_json(_repo_path(EXPECTED_BINDINGS["R9_SELECTOR_ARTIFACT"])),
        "blindassist.taro.o1r.r9_source_only_clear_enrichment_selector.v1",
    )
    require(
        selector.get("selector_id") == EXPECTED_NEXT_STAGE_SELECTOR["selector_id"]
        and selector["content_sha256"] == EXPECTED_NEXT_STAGE_SELECTOR["selector_content_sha256"]
        and selector.get("chosen_rule", {}).get("rule_id") == EXPECTED_NEXT_STAGE_SELECTOR["rule_id"]
        and selector.get("selection_uses_only_source_features") is True
        and selector.get("confirmation_authority") is False,
        "R11_PHASE_A_NEXT_STAGE_SELECTOR_DRIFT",
        "frozen next-stage R9 selector artifact drift",
    )
    user = lock.get("user_authority")
    require(
        isinstance(user, dict)
        and user.get("confirmed_by") == "user"
        and user.get("confirmed_at") == "2026-08-12"
        and user.get("confirmation_verbatim") == "授权"
        and user.get("scope") == AUTHORITY_SCOPE,
        "R11_PHASE_A_USER_AUTHORITY",
        "R11 Phase A user authority drift",
    )
    require(lock.get("runtime_environment") == _runtime_environment(), "R11_PHASE_A_RUNTIME_ENVIRONMENT_DRIFT", "R11 Phase A runtime environment drift")
    identity = lock.get("candidate_identity")
    require(isinstance(identity, dict), "R11_PHASE_A_CANDIDATE_IDENTITY", "R11 candidate identity missing")
    source = Path(str(identity.get("source_root", ""))).resolve()
    checkpoint = Path(str(identity.get("checkpoint_path", ""))).resolve()
    require(
        identity.get("model_id") == adapter.BASELINE_MODEL_ID
        and identity.get("source_commit") == depthart_runner.EXPECTED_SOURCE_GIT_COMMIT
        and identity.get("checkpoint_sha256") == adapter.BASELINE_CHECKPOINT_SHA256
        and identity.get("preprocess_id") == depthart_runner.PREPROCESS_ID
        and identity.get("postprocess_id") == depthart_runner.POSTPROCESS_ID
        and identity.get("inference_seed") == 0
        and identity.get("device") == "cuda"
        and identity.get("output_dtype") == "float32",
        "R11_PHASE_A_CANDIDATE_IDENTITY",
        "R11 candidate identity drift",
    )
    require(
        source.is_dir()
        and checkpoint.is_file()
        and checkpoint.stat().st_size == identity.get("checkpoint_bytes")
        and materializer.sha256_file(checkpoint) == identity.get("checkpoint_sha256"),
        "R11_PHASE_A_CANDIDATE_ASSET",
        "R11 candidate source/checkpoint drift",
    )
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(source), "status", "--short"], capture_output=True, text=True, check=True).stdout.strip()
    require(commit == identity["source_commit"] and not dirty, "R11_PHASE_A_CANDIDATE_SOURCE_DRIFT", "R11 candidate source tree drift")
    require(inventory["exact_pose_bounded_frame_count"] == FRAME_COUNT, "R11_PHASE_A_INVENTORY_INVALID", "R11 inventory count drift")
    if require_output_absent:
        require(not _repo_path(OUTPUT_ROOT).exists(), "R11_PHASE_A_ROOT_COLLISION", "R11 Phase A evidence root exists")
    lock["_lock_path"] = lock_path
    lock["_source_root"] = source
    lock["_checkpoint_path"] = checkpoint
    lock["_inventory"] = inventory
    lock["_verified_bindings"] = verified
    return lock


def _validate_factor_pair(
    r7_bundle: Mapping[str, Any], r11_bundle: Mapping[str, Any]
) -> tuple[Counter[str], Counter[str], int]:
    base = r7_positive.validate_positive_occupancy_factor(dict(r7_bundle))
    candidate = abstention_candidate.validate_abstention_bundle(dict(r11_bundle))
    require(
        base["physical_frame_id"] == candidate["physical_frame_id"]
        and base["source_frame_record_sha256"] == candidate["source_frame_record_sha256"],
        "R11_PHASE_A_FACTOR_LINEAGE",
        "R7/R11 factor lineage drift",
    )
    base_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    for base_row, candidate_row in zip(base["query_results"], candidate["query_results"], strict=True):
        require(
            base_row["query_id"] == candidate_row["query_id"]
            and base_row["grid_index"] == candidate_row["grid_index"],
            "R11_PHASE_A_FACTOR_LINEAGE",
            "R7/R11 query order drift",
        )
        require(
            candidate_row["state"] != "OCCUPIED_OBSERVED"
            or base_row["state"] == "OCCUPIED_OBSERVED",
            "R11_PHASE_A_FACTOR_SUBSET",
            "R11 positive is not a subset of R7 positive",
        )
        base_counts[base_row["state"]] += 1
        candidate_counts[candidate_row["state"]] += 1
    abstained = base_counts["OCCUPIED_OBSERVED"] - candidate_counts["OCCUPIED_OBSERVED"]
    require(
        base_counts["CLEAR_OBSERVED"] == candidate_counts["CLEAR_OBSERVED"] == 0
        and sum(base_counts.values()) == sum(candidate_counts.values()) == 9
        and abstained >= 0
        and candidate["base_positive_count"] == base_counts["OCCUPIED_OBSERVED"]
        and candidate["candidate_positive_count"] == candidate_counts["OCCUPIED_OBSERVED"]
        and candidate["abstained_base_positive_count"] == abstained,
        "R11_PHASE_A_FACTOR_COUNTS",
        "R7/R11 factor state count drift",
    )
    return base_counts, candidate_counts, abstained


def _write_failure(writer: FactorEvidenceWriter, error: BaseException) -> None:
    if not writer.activated:
        return
    writer.write_json(
        "failure.json",
        _seal(
            {
                "schema": "blindassist.taro.o1r.r11_fresh_pool_phase_a_failure.v1",
                "terminal": FAIL_TERMINAL,
                "execution_valid": False,
                "failure_code": str(getattr(error, "code", type(error).__name__)),
                "message": str(error),
                "one_shot_consumed": True,
            }
        ),
    )
    writer.write_json(
        "manifest.json",
        _seal(
            {
                "schema": "blindassist.taro.o1r.r11_fresh_pool_phase_a_failure_manifest.v1",
                "terminal": FAIL_TERMINAL,
                "files": dict(sorted(writer.file_receipts.items())),
                "file_count_before_manifest": len(writer.file_receipts),
                "bytes_before_manifest": writer.bytes_written,
                "one_shot_consumed": True,
            }
        ),
    )


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    output = _repo_path(OUTPUT_ROOT)
    budget = lock["resource_budget"]
    writer = FactorEvidenceWriter(output, int(budget["maximum_evidence_bytes"]))
    started = time.monotonic()
    process = psutil.Process()

    def guard() -> None:
        require(
            time.monotonic() - started <= float(budget["maximum_wall_seconds"]),
            "R11_PHASE_A_TIMEOUT",
            "R11 Phase A wall budget exceeded",
        )
        require(
            process.memory_info().rss <= int(budget["maximum_peak_rss_bytes"]),
            "R11_PHASE_A_RSS_EXCEEDED",
            "R11 Phase A RSS budget exceeded",
        )

    try:
        writer.activate(
            _seal(
                {
                    "schema": "blindassist.taro.o1r.r11_fresh_pool_phase_a_execution_receipt.v1",
                    "terminal_on_success": PASS_TERMINAL,
                    "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]),
                    "execution_lock_content_sha256": lock["content_sha256"],
                    "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "analysis_role": ANALYSIS_ROLE,
                    "expected_parent_count": PARENT_COUNT,
                    "expected_frame_count": FRAME_COUNT,
                    "expected_query_count": QUERY_COUNT,
                    "source_integrity_verification_begins_after_root_creation": True,
                    "highres_depth_member_payload_read": False,
                    "faro_payload_read": False,
                    "truth_scoring": False,
                    "parent_scoring": False,
                    "top24_selection": False,
                    "training_steps": 0,
                    "network_requests": 0,
                    "one_shot_consumed_on_root_creation": True,
                }
            )
        )

        # This is the first source-container access in formal execution. The root
        # already exists, so any source/hash/index failure consumes the one shot.
        frames = _load_frames(lock["_inventory"])
        guard()
        import torch

        torch.cuda.reset_peak_memory_stats()
        model, runtime_identity = depthart_runner.load_official_depthart(
            lock["_source_root"], lock["_checkpoint_path"], device="cuda", seed=0
        )
        candidate_reads = PayloadReadLedger()
        candidate_input_hashes: list[str] = []
        candidate_hashes: list[str] = []
        completed = 0
        for _parent_key, parent_frames in _frames_by_parent(frames):
            with zipfile.ZipFile(parent_frames[0].upsampling_archive) as up_bundle, zipfile.ZipFile(
                parent_frames[0].intrinsics_archive
            ) as intr_bundle:
                for frame in parent_frames:
                    candidate_input, color = _read_candidate_input(frame, up_bundle, intr_bundle, candidate_reads)
                    writer.write_json(_candidate_input_relative(frame), candidate_input)
                    native, _highres, inference = _run_candidate(model, runtime_identity, candidate_input, color)
                    blob_payload = depthart_runner.deterministic_npy_gzip_bytes(native)
                    blob_receipt = writer.write_bytes(_candidate_blob_relative(frame), blob_payload)
                    blob = {
                        **blob_receipt,
                        "array_sha256": adapter.canonical_sha256(native),
                        "shape_hw": list(native.shape),
                        "dtype": "float32",
                        "encoding": "DETERMINISTIC_GZIP_NPY_MTIME_0",
                    }
                    candidate = _seal(
                        {
                            "schema": "blindassist.taro.o1r.r11_fresh_pool_candidate_frame.v1",
                            "analysis_role": ANALYSIS_ROLE,
                            "parent_id": frame.parent_id,
                            "video_id": frame.video_id,
                            "timestamp_token": frame.timestamp_token,
                            "physical_frame_id": frame.physical_frame_id,
                            "candidate_input_sha256": candidate_input["content_sha256"],
                            "inference_receipt": inference,
                            "native_depth_blob": blob,
                            "highres_depth_member_payload_read": False,
                            "faro_payload_read": False,
                            "truth_alignment_used": False,
                        }
                    )
                    writer.write_json(_candidate_record_relative(frame), candidate)
                    candidate_input_hashes.append(candidate_input["content_sha256"])
                    candidate_hashes.append(candidate["content_sha256"])
                    completed += 1
                    guard()
                    if completed % 10 == 0 or completed == FRAME_COUNT:
                        print(
                            json.dumps(
                                {
                                    "phase": "R11_FRESH_CANDIDATE",
                                    "completed": completed,
                                    "total": FRAME_COUNT,
                                    "physical_frame_id": frame.physical_frame_id,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
        require(
            candidate_reads.attempts_by_role == Counter({"color": FRAME_COUNT, "intrinsics": FRAME_COUNT})
            and candidate_reads.completed_by_role == Counter({"color": FRAME_COUNT, "intrinsics": FRAME_COUNT}),
            "R11_CANDIDATE_READ_COUNT_DRIFT",
            "candidate payload read counts drift",
            attempts=dict(candidate_reads.attempts_by_role),
            completed=dict(candidate_reads.completed_by_role),
        )
        writer.write_json(
            "candidate-completion.json",
            _seal(
                {
                    "schema": "blindassist.taro.o1r.r11_fresh_pool_candidate_completion.v1",
                    "frame_count": FRAME_COUNT,
                    "candidate_input_hash_sequence_sha256": adapter.canonical_sha256(candidate_input_hashes),
                    "candidate_record_hash_sequence_sha256": adapter.canonical_sha256(candidate_hashes),
                    "payload_read_attempts": dict(sorted(candidate_reads.attempts_by_role.items())),
                    "payload_reads_completed": dict(sorted(candidate_reads.completed_by_role.items())),
                    "payload_bytes": dict(sorted(candidate_reads.bytes_by_role.items())),
                    "candidate_inference_count": FRAME_COUNT,
                    "highres_depth_member_payload_reads": 0,
                    "faro_reads": 0,
                    "truth_reads": 0,
                    "all_candidates_sealed_before_source_features": True,
                }
            ),
        )
        cuda_peak = int(torch.cuda.max_memory_allocated())
        del model
        torch.cuda.empty_cache()
        require(
            cuda_peak <= int(budget["maximum_cuda_allocated_bytes"]),
            "R11_PHASE_A_CUDA_EXCEEDED",
            "candidate CUDA budget exceeded",
        )

        uncertainty_model = load_locked_uncertainty_model()
        source_reads = PayloadReadLedger()
        source_receipt_hashes: list[str] = []
        source_frame_hashes: list[str] = []
        prospective_hashes: list[str] = []
        reducer_hashes: list[str] = []
        r7_factor_hashes: list[str] = []
        r11_factor_hashes: list[str] = []
        r7_counts: Counter[str] = Counter()
        r11_counts: Counter[str] = Counter()
        parent_identities = [(parent, video) for parent, video, _rank in EXPECTED_PARENT_IDENTITIES]
        per_parent_r7: dict[tuple[str, str], Counter[str]] = {identity: Counter() for identity in parent_identities}
        per_parent_r11: dict[tuple[str, str], Counter[str]] = {identity: Counter() for identity in parent_identities}
        per_parent_abstained: Counter[tuple[str, str]] = Counter()
        abstained_total = 0
        completed = 0

        for _parent_key, parent_frames in _frames_by_parent(frames):
            with zipfile.ZipFile(parent_frames[0].upsampling_archive) as up_bundle:
                for frame in parent_frames:
                    candidate_input, candidate, candidate_depth_highres_m = _load_candidate(output, frame)
                    apple_payload, apple_binding = _read_allowed_member(
                        up_bundle, frame, "lowres_depth", "SOURCE_FEATURE", source_reads
                    )
                    confidence_payload, confidence_binding = _read_allowed_member(
                        up_bundle, frame, "confidence", "SOURCE_FEATURE", source_reads
                    )
                    apple = np.ascontiguousarray(materializer._decode_png(apple_payload, "lowres_depth"))
                    confidence = np.ascontiguousarray(materializer._decode_png(confidence_payload, "confidence"))
                    source = _seal(
                        {
                            "schema": "blindassist.taro.o1r.r11_fresh_pool_source_frame_receipt.v1",
                            "analysis_role": ANALYSIS_ROLE,
                            "parent_id": frame.parent_id,
                            "video_id": frame.video_id,
                            "timestamp_token": frame.timestamp_token,
                            "physical_frame_id": frame.physical_frame_id,
                            "candidate_input_sha256": candidate_input["content_sha256"],
                            "candidate_frame_record_sha256": candidate["content_sha256"],
                            "apple_depth_binding": _member_binding(frame, "lowres_depth", apple_binding),
                            "confidence_binding": _member_binding(frame, "confidence", confidence_binding),
                            "apple_depth_sha256": adapter.canonical_sha256(apple),
                            "confidence_sha256": adapter.canonical_sha256(confidence),
                            "intrinsics_highres": candidate_input["intrinsics_highres"],
                            "lowres_intrinsics": candidate_input["lowres_intrinsics"],
                            "gravity_up_camera_xyz": candidate_input["gravity_up_camera_xyz"],
                            "max_source_timestamp_ns": candidate_input["max_source_timestamp_ns"],
                            "highres_depth_member_payload_read": False,
                            "faro_payload_read": False,
                            "truth_payload_read": False,
                        }
                    )
                    writer.write_json(_source_receipt_relative(frame), source)
                    require(
                        _validate_seal(
                            _load_json(output / _source_receipt_relative(frame)),
                            "blindassist.taro.o1r.r11_fresh_pool_source_frame_receipt.v1",
                        )
                        == source,
                        "R11_PHASE_A_SOURCE_RELOAD_DRIFT",
                        "source receipt replay drift",
                    )
                    low = candidate_input["lowres_intrinsics"]
                    low_matrix = [
                        [float(low["fx"]), 0.0, float(low["cx"])],
                        [0.0, float(low["fy"]), float(low["cy"])],
                        [0.0, 0.0, 1.0],
                    ]
                    prospective_bundle = prospective.build_prospective_factor_bundle(
                        parent_id=frame.parent_id,
                        video_id=frame.video_id,
                        timestamp_token=frame.timestamp_token,
                        source_frame_receipt_sha256=source["content_sha256"],
                        candidate_frame_record_sha256=candidate["content_sha256"],
                        max_source_timestamp_ns=int(source["max_source_timestamp_ns"]),
                        candidate_highres_depth_m=candidate_depth_highres_m,
                        apple_depth_mm=apple,
                        confidence=confidence,
                        intrinsics_highres_3x3=source["intrinsics_highres"]["matrix_3x3"],
                        intrinsics_apple_3x3=low_matrix,
                        gravity_up_camera_xyz=source["gravity_up_camera_xyz"],
                    )
                    reducer_bundle = r6_reducer.integrate_prospective_factor_bundle(
                        prospective_bundle=prospective_bundle,
                        candidate_highres_depth_m=candidate_depth_highres_m,
                        confidence=confidence,
                        intrinsics_apple_3x3=low_matrix,
                        uncertainty_model=uncertainty_model,
                    )
                    source_features = r7_canary.build_source_frame_record(
                        prospective_bundle,
                        candidate_depth_highres_m,
                        apple,
                        confidence,
                        low_matrix,
                        source["intrinsics_highres"]["matrix_3x3"],
                        reducer_bundle,
                    )
                    r7_bundle = r7_positive.build_positive_occupancy_factor(source_features)
                    r11_bundle = abstention_candidate.build_abstention_bundle(source_features)
                    base_frame_counts, candidate_frame_counts, abstained = _validate_factor_pair(
                        r7_bundle, r11_bundle
                    )
                    lineage = _seal(
                        {
                            "schema": "blindassist.taro.o1r.r11_fresh_pool_phase_a_lineage.v1",
                            "physical_frame_id": frame.physical_frame_id,
                            "source_frame_receipt_sha256": source["content_sha256"],
                            "prospective_bundle": prospective_bundle,
                            "r6_reducer_bundle": reducer_bundle,
                            "r7_source_frame_record": source_features,
                            "r7_positive_factor_bundle": r7_bundle,
                            "r11_abstention_bundle": r11_bundle,
                            "highres_depth_member_payload_read": False,
                            "faro_payload_read": False,
                            "truth_inputs": 0,
                        }
                    )
                    writer.write_json_gzip(_lineage_relative(frame), lineage)
                    reloaded_lineage = _validate_seal(
                        _load_json_gzip(output / _lineage_relative(frame)),
                        "blindassist.taro.o1r.r11_fresh_pool_phase_a_lineage.v1",
                    )
                    require(reloaded_lineage == lineage, "R11_PHASE_A_LINEAGE_RELOAD_DRIFT", "Phase A lineage replay drift")
                    require(
                        prospective.validate_prospective_factor_bundle(
                            reloaded_lineage["prospective_bundle"],
                            candidate_highres_depth_m=candidate_depth_highres_m,
                        )["content_sha256"]
                        == prospective_bundle["content_sha256"]
                        and r6_reducer.validate_reducer_bundle(reloaded_lineage["r6_reducer_bundle"])["content_sha256"]
                        == reducer_bundle["content_sha256"]
                        and r7_canary.validate_source_frame_record(reloaded_lineage["r7_source_frame_record"])["content_sha256"]
                        == source_features["content_sha256"]
                        and r7_positive.validate_positive_occupancy_factor(reloaded_lineage["r7_positive_factor_bundle"])["content_sha256"]
                        == r7_bundle["content_sha256"]
                        and abstention_candidate.validate_abstention_bundle(reloaded_lineage["r11_abstention_bundle"])["content_sha256"]
                        == r11_bundle["content_sha256"],
                        "R11_PHASE_A_LINEAGE_RELOAD_DRIFT",
                        "nested Phase A lineage replay drift",
                    )
                    source_receipt_hashes.append(source["content_sha256"])
                    source_frame_hashes.append(source_features["content_sha256"])
                    prospective_hashes.append(prospective_bundle["content_sha256"])
                    reducer_hashes.append(reducer_bundle["content_sha256"])
                    r7_factor_hashes.append(r7_bundle["content_sha256"])
                    r11_factor_hashes.append(r11_bundle["content_sha256"])
                    r7_counts.update(base_frame_counts)
                    r11_counts.update(candidate_frame_counts)
                    identity = (frame.parent_id, frame.video_id)
                    per_parent_r7[identity].update(base_frame_counts)
                    per_parent_r11[identity].update(candidate_frame_counts)
                    per_parent_abstained[identity] += abstained
                    abstained_total += abstained
                    completed += 1
                    guard()
                    if completed % 10 == 0 or completed == FRAME_COUNT:
                        print(
                            json.dumps(
                                {
                                    "phase": "R11_FRESH_SOURCE_FEATURES",
                                    "completed": completed,
                                    "total": FRAME_COUNT,
                                    "physical_frame_id": frame.physical_frame_id,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
        require(
            source_reads.attempts_by_role == Counter({"lowres_depth": FRAME_COUNT, "confidence": FRAME_COUNT})
            and source_reads.completed_by_role == Counter({"lowres_depth": FRAME_COUNT, "confidence": FRAME_COUNT}),
            "R11_SOURCE_READ_COUNT_DRIFT",
            "source feature payload read counts drift",
            attempts=dict(source_reads.attempts_by_role),
            completed=dict(source_reads.completed_by_role),
        )
        require(
            sum(r7_counts.values()) == sum(r11_counts.values()) == QUERY_COUNT
            and r7_counts["CLEAR_OBSERVED"] == r11_counts["CLEAR_OBSERVED"] == 0
            and r11_counts["OCCUPIED_OBSERVED"] <= r7_counts["OCCUPIED_OBSERVED"]
            and abstained_total == r7_counts["OCCUPIED_OBSERVED"] - r11_counts["OCCUPIED_OBSERVED"],
            "R11_SOURCE_STATE_DRIFT",
            "R7/R11 source factors emitted CLEAR, lost queries, or violated subset accounting",
        )
        for index, identity in enumerate(parent_identities):
            expected_queries = FROZEN_FRAME_COUNTS[index] * 9
            require(
                sum(per_parent_r7[identity].values()) == expected_queries
                and sum(per_parent_r11[identity].values()) == expected_queries
                and per_parent_r11[identity]["OCCUPIED_OBSERVED"]
                <= per_parent_r7[identity]["OCCUPIED_OBSERVED"]
                and per_parent_abstained[identity]
                == per_parent_r7[identity]["OCCUPIED_OBSERVED"]
                - per_parent_r11[identity]["OCCUPIED_OBSERVED"],
                "R11_PARENT_FACTOR_COUNT_DRIFT",
                "per-parent R7/R11 factor accounting drift",
                identity=identity,
            )

        states = ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
        completion = _seal(
            {
                "schema": "blindassist.taro.o1r.r11_fresh_pool_phase_a_completion.v1",
                "parent_count": PARENT_COUNT,
                "frame_count": FRAME_COUNT,
                "query_count": QUERY_COUNT,
                "inventory_content_sha256": INVENTORY_CONTENT_SHA256,
                "candidate_input_hash_sequence_sha256": adapter.canonical_sha256(candidate_input_hashes),
                "candidate_record_hash_sequence_sha256": adapter.canonical_sha256(candidate_hashes),
                "source_receipt_hash_sequence_sha256": adapter.canonical_sha256(source_receipt_hashes),
                "source_frame_hash_sequence_sha256": adapter.canonical_sha256(source_frame_hashes),
                "prospective_bundle_hash_sequence_sha256": adapter.canonical_sha256(prospective_hashes),
                "r6_reducer_hash_sequence_sha256": adapter.canonical_sha256(reducer_hashes),
                "r7_factor_hash_sequence_sha256": adapter.canonical_sha256(r7_factor_hashes),
                "r11_factor_hash_sequence_sha256": adapter.canonical_sha256(r11_factor_hashes),
                "r7_base_state_counts": {state: int(r7_counts[state]) for state in states},
                "r11_candidate_state_counts": {state: int(r11_counts[state]) for state in states},
                "r11_abstained_base_positive_count": int(abstained_total),
                "per_parent_factor_counts": [
                    {
                        "visit_id": identity[0],
                        "video_id": identity[1],
                        "frame_count": FROZEN_FRAME_COUNTS[index],
                        "query_count": FROZEN_FRAME_COUNTS[index] * 9,
                        "r7_state_counts": {state: int(per_parent_r7[identity][state]) for state in states},
                        "r11_state_counts": {state: int(per_parent_r11[identity][state]) for state in states},
                        "r11_abstained_base_positive_count": int(per_parent_abstained[identity]),
                    }
                    for index, identity in enumerate(parent_identities)
                ],
                "source_payload_read_accounting": {
                    "attempts_by_role": {
                        role: int(candidate_reads.attempts_by_role[role] + source_reads.attempts_by_role[role])
                        for role in ("color", "intrinsics", "lowres_depth", "confidence", "highres_depth")
                    },
                    "completed_by_role": {
                        role: int(candidate_reads.completed_by_role[role] + source_reads.completed_by_role[role])
                        for role in ("color", "intrinsics", "lowres_depth", "confidence", "highres_depth")
                    },
                    "bytes_by_role": {
                        role: int(candidate_reads.bytes_by_role[role] + source_reads.bytes_by_role[role])
                        for role in ("color", "intrinsics", "lowres_depth", "confidence", "highres_depth")
                    },
                    "total_zip_member_payload_reads": 4 * FRAME_COUNT,
                    "trajectory_payload_reads": PARENT_COUNT,
                    "candidate_blob_reloads": FRAME_COUNT,
                    "depthart_inferences": FRAME_COUNT,
                    "faro_values_interpreted": 0,
                    "truth_reads": 0,
                    "label_reads": 0,
                    "outcome_reads": 0,
                    "network_requests": 0,
                    "training_steps": 0,
                },
                "source_container_integrity_validations": PARENT_COUNT * 3,
                "zip_central_directory_indexes": PARENT_COUNT * 2,
                "trajectory_payload_reads": PARENT_COUNT,
                "highres_depth_member_payload_reads": 0,
                "faro_reads": 0,
                "truth_reads": 0,
                "clear_output_allowed": False,
                "unknown_is_negative": False,
                "all_candidate_records_sealed_before_source_features": True,
                "all_r7_and_r11_records_sealed_before_parent_scoring": True,
                "all_source_records_sealed_before_faro": True,
                "r9_parent_scoring_performed": False,
                "top24_selection_performed": False,
                "training_steps": 0,
                "network_requests": 0,
            }
        )
        writer.write_json("phase-a-completion.json", completion)
        reloaded = _validate_seal(
            _load_json(output / "phase-a-completion.json"),
            "blindassist.taro.o1r.r11_fresh_pool_phase_a_completion.v1",
        )
        require(
            reloaded == completion
            and reloaded["highres_depth_member_payload_reads"] == 0
            and reloaded["faro_reads"] == 0
            and reloaded["r9_parent_scoring_performed"] is False,
            "R11_PHASE_A_COMPLETION_RELOAD_DRIFT",
            "R11 Phase A completion reload drift",
        )
        result = _seal(
            {
                "schema": "blindassist.taro.o1r.r11_fresh_pool_phase_a_result.v1",
                "terminal": PASS_TERMINAL,
                "passed": True,
                "execution_valid": True,
                "parent_count": PARENT_COUNT,
                "frame_count": FRAME_COUNT,
                "query_count": QUERY_COUNT,
                "candidate_inference_count": FRAME_COUNT,
                "r7_base_state_counts": completion["r7_base_state_counts"],
                "r11_candidate_state_counts": completion["r11_candidate_state_counts"],
                "r11_abstained_base_positive_count": abstained_total,
                "phase_a_completion_sha256": completion["content_sha256"],
                "runtime_identity_sha256": adapter.canonical_sha256(runtime_identity),
                "highres_depth_member_payload_reads": 0,
                "faro_reads": 0,
                "truth_scoring": False,
                "r9_parent_scoring_performed": False,
                "top24_selection_performed": False,
                "clear_output_allowed": False,
                "unknown_is_negative": False,
                "training_steps": 0,
                "network_requests": 0,
                "cuda_peak_allocated_bytes": cuda_peak,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "one_shot_consumed": True,
                "unique_successor": "TARO_O1R_R11_SEAL_R7_BASE_R11_CANDIDATE_AND_R9_PARENT_SCORES_IMPLEMENTATION_LOCK",
                "claim_ceiling": "Sealed R11 all-48 source-only R7/R11 factors ready for frozen R9 parent scoring; no FARO label, task effectiveness, training, deployment, product, or safety evidence.",
            }
        )
        require(
            len(writer.file_receipts) == PRE_MANIFEST_FILE_COUNT - 1,
            "R11_PHASE_A_MANIFEST_COUNT_DRIFT",
            "R11 Phase A file count before result drift",
            file_count=len(writer.file_receipts),
        )
        writer.write_json("result.json", result)
        require(
            len(writer.file_receipts) == PRE_MANIFEST_FILE_COUNT,
            "R11_PHASE_A_MANIFEST_COUNT_DRIFT",
            "R11 Phase A file count before manifest drift",
            file_count=len(writer.file_receipts),
        )
        writer.write_json(
            "manifest.json",
            _seal(
                {
                    "schema": "blindassist.taro.o1r.r11_fresh_pool_phase_a_manifest.v1",
                    "terminal": PASS_TERMINAL,
                    "files": dict(sorted(writer.file_receipts.items())),
                    "file_count_before_manifest": len(writer.file_receipts),
                    "bytes_before_manifest": writer.bytes_written,
                    "one_shot_consumed": True,
                }
            ),
        )
        return result
    except Exception as error:
        try:
            _write_failure(writer, error)
        except Exception as failure_error:
            raise FreshPhaseAError(
                "R11_PHASE_A_FAILURE_SEAL_FAILED",
                f"Phase A failed and failure evidence could not be sealed: {failure_error}",
                original_failure_code=str(getattr(error, "code", type(error).__name__)),
            ) from error
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        print(
            json.dumps(
                {
                    "terminal": FAIL_TERMINAL,
                    "failure_code": str(getattr(error, "code", type(error).__name__)),
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "passed": result["passed"],
                "execution_valid": result["execution_valid"],
                "frame_count": result["frame_count"],
                "query_count": result["query_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

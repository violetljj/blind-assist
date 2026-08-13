#!/usr/bin/env python3
"""Read FARO only for the sealed R11 top 24 and reduce frozen confirmation gates."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import inspect
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import numpy as np

from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import positive_occupancy_factor as r7_positive
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r11_abstention_runtime import abstention_candidate
from scripts.research.taro_o1r_r11_abstention_runtime import phase_b_metrics
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_inventory
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_phase_a as phase_a
from scripts.research.taro_o1r_r11_abstention_runtime import validate_protocol_lock


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELATIVE = validate_protocol_lock.PROTOCOL_RELATIVE
AUTHORIZATION_RELATIVE = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-12.json"
INVENTORY_RELATIVE = phase_a.INVENTORY_PATH
PHASE_A_ROOT = phase_a.OUTPUT_ROOT
PHASE_A_AUDIT_RELATIVE = (
    "artifacts.local/evidence/taro/"
    "o1r-r11-fresh-pool-phase-a-validator-round12-repair-r0/post-result-audit.json"
)
TOP24_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-top24-selection-r0"
TOP24_TERMINAL_RELATIVE = f"{TOP24_ROOT}/terminal.json"
TOP24_SCORES_RELATIVE = f"{TOP24_ROOT}/parent-scores.json"
TOP24_SELECTION_RELATIVE = f"{TOP24_ROOT}/selection.json"
TOP24_RESULT_RELATIVE = (
    "docs/research/taro/"
    "TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_RESULT_2026-08-13.json"
)
LOCK_RELATIVE = (
    "docs/research/taro/"
    "TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_ONE_SHOT_EXECUTION_LOCK_2026-08-13.json"
)
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r11-selected-top24-faro-phase-b-r0"

LOCK_SCHEMA = "blindassist.taro.o1r.r11_selected_top24_faro_phase_b_execution_lock.v1"
LOCK_ID = "TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_ONE_SHOT_EXECUTION_LOCK"
TERMINAL_SCHEMA = "blindassist.taro.o1r.r11_selected_top24_faro_phase_b_terminal.v1"
RESULT_SCHEMA = "blindassist.taro.o1r.r11_selected_top24_faro_phase_b_result.v1"
COMPLETION_SCHEMA = "blindassist.taro.o1r.r11_selected_top24_faro_label_completion.v1"
FAILURE_SCHEMA = "blindassist.taro.o1r.r11_selected_top24_faro_phase_b_failure.v1"
INVALID_TERMINAL = "EXECUTION_INVALID"

SELECTED_PARENT_COUNT = 24
SELECTED_FRAME_COUNT = 674
SELECTED_QUERY_COUNT = 6066
SUCCESS_PRE_TERMINAL_FILE_COUNT = SELECTED_FRAME_COUNT + 3
SUCCESS_FINAL_FILE_COUNT = SUCCESS_PRE_TERMINAL_FILE_COUNT + 1
TERMINAL_RESERVE_BYTES = 4_194_304
TERMINAL_WALL_RESERVE_SECONDS = 60
TERMINAL_RESERVE_NAME = ".terminal-reserve.bin"
TOP24_TERMINAL_CONTENT_SHA256 = "1278843404CCF43896A5AB7EA028C8E2C0D3DDF13579A2E4B14BE4A7932145BB"
TOP24_SCORES_CONTENT_SHA256 = "A870517F05DD99C12AC8D901623BCB718526912A24D63644DB0A33516C47D2BE"
TOP24_SELECTION_CONTENT_SHA256 = "629ECF7069EE5942EAEF7946059CAD03D20D0F66CBD4DAF95E06A5315211A7B7"
TOP24_FORMAL_RESULT_CONTENT_SHA256 = "D2445F8ABCD2D8AD96F921AA6C1427DC4E3D5AB39AD674288F94720CAABCA7CD"

EXPECTED_ARGV = [
    "-m",
    "scripts.research.taro_o1r_r11_abstention_runtime.run_selected_phase_b",
    "--execution-lock",
    LOCK_RELATIVE,
]
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "授权",
    "scope": (
        "Exact frozen R11 48-parent Training pool and 144-URL plan: zero-body HEAD, bounded source download "
        "and integrity validation, all-48 source-only Phase A, source-only top-24 selection, then FARO only "
        "for the sealed top 24; no training, device, deployment, product, safety, or redistribution authority."
    ),
}
EXPECTED_RESOURCE_BUDGET = {
    "maximum_wall_seconds": 14_400,
    "maximum_peak_rss_bytes": 17_179_869_184,
    "maximum_evidence_bytes": 536_870_912,
}
EXPECTED_AUTHORITY = {
    "sealed_phase_a_reload": True,
    "sealed_top24_reload": True,
    "faro_payload_read": True,
    "selected_faro_frame_count": SELECTED_FRAME_COUNT,
    "unselected_faro_frame_count": 0,
    "truth_label_construction": True,
    "fixed_gate_evaluation": True,
    "source_reselection": False,
    "parent_reselection": False,
    "selector_fit": False,
    "candidate_reselection": False,
    "threshold_fit": False,
    "model_execution": False,
    "training": False,
    "network": False,
    "device": False,
    "deployment": False,
    "product": False,
    "safety": False,
    "redistribution": False,
}
EXPECTED_ONE_SHOT_POLICY = {
    "consumed_on_partial_root_creation": True,
    "failure_does_not_restore_authority": True,
    "atomic_directory_publish": True,
    "success_pre_terminal_file_count": SUCCESS_PRE_TERMINAL_FILE_COUNT,
    "success_final_file_count": SUCCESS_FINAL_FILE_COUNT,
    "terminal_reserve_bytes": TERMINAL_RESERVE_BYTES,
    "terminal_wall_reserve_seconds": TERMINAL_WALL_RESERVE_SECONDS,
}
EXPECTED_BINDINGS = {
    "R11_PROTOCOL": PROTOCOL_RELATIVE,
    "R11_DATA_USE_AUTHORIZATION": AUTHORIZATION_RELATIVE,
    "R11_INVENTORY": INVENTORY_RELATIVE,
    "R11_PHASE_A_COMPLETION": f"{PHASE_A_ROOT}/phase-a-completion.json",
    "R11_PHASE_A_TERMINAL": f"{PHASE_A_ROOT}/terminal.json",
    "R11_PHASE_A_REPAIRED_AUDIT": PHASE_A_AUDIT_RELATIVE,
    "R11_TOP24_TERMINAL": TOP24_TERMINAL_RELATIVE,
    "R11_TOP24_PARENT_SCORES": TOP24_SCORES_RELATIVE,
    "R11_TOP24_SELECTION": TOP24_SELECTION_RELATIVE,
    "R11_TOP24_FORMAL_RESULT": TOP24_RESULT_RELATIVE,
    "SOURCE_ADAPTER_RUNTIME": "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py",
    "TRUTH_MATERIALIZER_RUNTIME": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "CANDIDATE_SCALE_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/prospective_factor_runtime.py",
    "REDUCER_INTEGRATION_RUNTIME": "scripts/research/taro_o1r_reducer_integration_runtime/reducer_integration.py",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "R7_CANARY_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "R7_POSITIVE_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/positive_occupancy_factor.py",
    "R11_ABSTENTION_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/abstention_candidate.py",
    "R11_INVENTORY_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_inventory.py",
    "R11_PHASE_A_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_phase_a.py",
    "R11_PROTOCOL_VALIDATOR": "scripts/research/taro_o1r_r11_abstention_runtime/validate_protocol_lock.py",
    "R11_PHASE_B_METRICS": "scripts/research/taro_o1r_r11_abstention_runtime/phase_b_metrics.py",
    "R11_PHASE_B_METRICS_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_phase_b_metrics.py",
    "R11_PHASE_B_RUNNER": "scripts/research/taro_o1r_r11_abstention_runtime/run_selected_phase_b.py",
    "R11_PHASE_B_RUNNER_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_run_selected_phase_b.py",
    "R11_PHASE_B_INDEPENDENT_VALIDATOR": "scripts/research/taro_o1r_r11_abstention_runtime/validate_selected_phase_b.py",
    "R11_PHASE_B_VALIDATOR_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_validate_selected_phase_b.py",
    "R11_PHASE_B_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK_2026-08-13.md",
}
ARTIFACT_BINDING_ROLES = {
    "R11_INVENTORY",
    "R11_PHASE_A_COMPLETION",
    "R11_PHASE_A_TERMINAL",
    "R11_PHASE_A_REPAIRED_AUDIT",
    "R11_TOP24_TERMINAL",
    "R11_TOP24_PARENT_SCORES",
    "R11_TOP24_SELECTION",
}


class R11PhaseBError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise R11PhaseBError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "R11_PHASE_B_JSON", f"JSON object required: {path}")
    return value


def _load_json_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), "R11_PHASE_B_JSON", f"gzip JSON object required: {path}")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R11_PHASE_B_SEAL_COLLISION", "caller supplied content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), "R11_PHASE_B_SEAL", "sealed record must be an object")
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(
        record.get("schema") == schema
        and isinstance(observed, str)
        and adapter.canonical_sha256(record) == observed,
        "R11_PHASE_B_SEAL",
        f"record seal/schema drift: {schema}",
    )
    record["content_sha256"] = observed
    return record


def _git_bytes(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    require(completed.returncode == 0, "R11_PHASE_B_IMPLEMENTATION_BINDING", f"implementation commit lacks {relative}")
    return completed.stdout


def _commit_is_on_master(commit: Any) -> bool:
    if not isinstance(commit, str) or len(commit) != 40:
        return False
    return all(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, ancestor],
            capture_output=True,
            check=False,
        ).returncode
        == 0
        for ancestor in ("HEAD", "refs/remotes/origin/master")
    )


def _validate_actual_argv() -> None:
    original = [str(value) for value in getattr(sys, "orig_argv", [])]
    require("-m" in original, "R11_PHASE_B_ARGV", "Phase B must use module-form argv")
    index = original.index("-m")
    require(original[index:] == EXPECTED_ARGV, "R11_PHASE_B_ARGV", "actual Phase B argv drift")


def _validate_bindings(lock: Mapping[str, Any]) -> None:
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R11_PHASE_B_BINDINGS", "binding count drift")
    seen: set[str] = set()
    for row in bindings:
        require(isinstance(row, Mapping), "R11_PHASE_B_BINDING", "binding row must be an object")
        role, relative = row.get("role"), row.get("path")
        require(
            set(row) == {"role", "path", "bytes", "sha256"}
            and isinstance(role, str)
            and role not in seen
            and EXPECTED_BINDINGS.get(role) == relative,
            "R11_PHASE_B_BINDING",
            "binding role/path drift",
        )
        path = _repo_path(str(relative))
        payload = path.read_bytes() if path.is_file() else b""
        require(
            len(payload) == row.get("bytes") and materializer.sha256_bytes(payload) == row.get("sha256"),
            "R11_PHASE_B_BINDING_HASH",
            f"binding drift: {relative}",
        )
        if role not in ARTIFACT_BINDING_ROLES:
            require(
                payload == _git_bytes(str(lock.get("implementation_commit")), str(relative)),
                "R11_PHASE_B_IMPLEMENTATION_BINDING",
                f"implementation-commit binding drift: {relative}",
            )
        seen.add(role)
    require(seen == set(EXPECTED_BINDINGS), "R11_PHASE_B_BINDINGS", "binding role set drift")


@dataclass(frozen=True)
class SelectedFrame:
    parent_id: str
    video_id: str
    timestamp_token: str
    physical_frame_id: str
    upsampling_archive: Path
    highres_member: phase_a.PhaseAMemberRef
    container_binding: dict[str, Any]


def derive_selected_frames(inventory: Mapping[str, Any], selection: Mapping[str, Any]) -> list[SelectedFrame]:
    validated = run_pool_inventory.validate_inventory(inventory)
    selected_scores = selection.get("selected_parent_scores")
    selected_identities = selection.get("selected_parent_identities")
    require(
        isinstance(selected_scores, list)
        and isinstance(selected_identities, list)
        and len(selected_scores) == len(selected_identities) == SELECTED_PARENT_COUNT,
        "R11_PHASE_B_SELECTION_COUNT",
        "sealed top24 identity/score count drift",
    )
    identity_order = [(str(row[0]), str(row[1])) for row in selected_identities]
    score_counts = {
        (str(row["parent_id"]), str(row["video_id"])): int(row["frame_count"])
        for row in selected_scores
    }
    require(identity_order == list(score_counts), "R11_PHASE_B_SELECTION_SCORE", "selected scores do not preserve identity order")
    parent_map = {(str(row["visit_id"]), str(row["video_id"])): row for row in validated["parents"]}
    require(set(identity_order).issubset(parent_map), "R11_PHASE_B_SELECTED_PARENT", "selected parent missing from inventory")
    frames: list[SelectedFrame] = []
    for identity in identity_order:
        parent = parent_map[identity]
        tokens = parent["frame_plan"]["exact_timestamp_tokens"]
        require(len(tokens) == score_counts[identity], "R11_PHASE_B_SELECTED_FRAME_COUNT", "selected parent frame count drift")
        binding = parent["container_bindings"]["upsampling"]
        archive = _repo_path(binding["path"])
        phase_a._verify_container(archive, binding)
        index, declared = run_pool_inventory.index_upsampling_archive_metadata_only(
            archive,
            identity[1],
            maximum_declared_uncompressed_bytes=int(binding["declared_uncompressed_bytes"]),
        )
        require(
            declared == binding["declared_uncompressed_bytes"]
            and phase_a._member_index_sha256(index) == binding["recognized_member_index_sha256"],
            "R11_PHASE_B_MEMBER_INDEX",
            "selected parent member index drift",
        )
        for token in tokens:
            member = phase_a._phase_member(index["highres_depth"][token])
            frames.append(
                SelectedFrame(
                    identity[0], identity[1], str(token), f"{identity[1]}:{token}", archive,
                    member, dict(binding),
                )
            )
    require(
        len(frames) == SELECTED_FRAME_COUNT
        and len({frame.physical_frame_id for frame in frames}) == SELECTED_FRAME_COUNT,
        "R11_PHASE_B_SELECTED_COHORT",
        "selected cohort is not exact 24/674",
    )
    return frames


def _load_top24() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    terminal = _validate_seal(
        _load_json(_repo_path(TOP24_TERMINAL_RELATIVE)),
        "blindassist.taro.o1r.r11_fresh_pool_top24_selection_terminal.v1",
    )
    scores = _validate_seal(
        _load_json(_repo_path(TOP24_SCORES_RELATIVE)),
        "blindassist.taro.o1r.r11_fresh_pool_source_only_parent_scores.v1",
    )
    selection = _validate_seal(
        _load_json(_repo_path(TOP24_SELECTION_RELATIVE)),
        "blindassist.taro.o1r.r11_fresh_pool_top24_source_only_selection.v1",
    )
    formal = _validate_seal(
        _load_json(_repo_path(TOP24_RESULT_RELATIVE)),
        "blindassist.taro.o1r.r11_fresh_pool_top24_selection_formal_result.v1",
    )
    require(
        terminal["content_sha256"] == TOP24_TERMINAL_CONTENT_SHA256
        and scores["content_sha256"] == TOP24_SCORES_CONTENT_SHA256
        and selection["content_sha256"] == TOP24_SELECTION_CONTENT_SHA256
        and formal["content_sha256"] == TOP24_FORMAL_RESULT_CONTENT_SHA256
        and terminal.get("passed") is terminal.get("execution_valid") is True
        and terminal.get("result", {}).get("selection_sha256") == selection["content_sha256"]
        and selection.get("parent_scores_sha256") == scores["content_sha256"]
        and selection.get("selected_parent_count") == SELECTED_PARENT_COUNT
        and sum(int(row["frame_count"]) for row in selection["selected_parent_scores"]) == SELECTED_FRAME_COUNT
        and formal.get("status") == "TARO_O1R_R11_SOURCE_ONLY_TOP24_INDEPENDENT_VALIDATION_PASS"
        and formal.get("firewall", {}).get("faro_reads") == formal.get("firewall", {}).get("truth_reads") == 0,
        "R11_PHASE_B_TOP24",
        "sealed and independently validated top24 is not admitted",
    )
    return scores, selection, formal


def _load_selected_lineages(frames: Sequence[SelectedFrame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    root = _repo_path(PHASE_A_ROOT)
    for frame in frames:
        source_path = root / f"phase-a-sources/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"
        lineage_path = root / f"phase-a-lineage/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"
        receipt = phase_a._validate_seal(
            _load_json(source_path),
            "blindassist.taro.o1r.r11_fresh_pool_source_frame_receipt.v1",
        )
        lineage = phase_a._validate_seal(
            _load_json_gzip(lineage_path),
            "blindassist.taro.o1r.r11_fresh_pool_phase_a_lineage.v1",
        )
        source = r7_canary.validate_source_frame_record(lineage["r7_source_frame_record"])
        baseline = r7_positive.validate_positive_occupancy_factor(lineage["r7_positive_factor_bundle"])
        candidate = abstention_candidate.validate_abstention_bundle(lineage["r11_abstention_bundle"])
        require(
            receipt["physical_frame_id"] == source["physical_frame_id"] == baseline["physical_frame_id"]
            == candidate["physical_frame_id"] == frame.physical_frame_id
            and lineage["source_frame_receipt_sha256"] == receipt["content_sha256"]
            and source["source_frame_receipt_sha256"] == receipt["content_sha256"]
            and baseline["source_frame_record_sha256"] == candidate["source_frame_record_sha256"] == source["content_sha256"]
            and receipt["highres_depth_member_payload_read"] is receipt["faro_payload_read"] is receipt["truth_payload_read"] is False
            and lineage["highres_depth_member_payload_read"] is lineage["faro_payload_read"] is False
            and lineage["truth_inputs"] == 0,
            "R11_PHASE_B_LINEAGE",
            "selected Phase A lineage drift",
        )
        sources.append(source)
        baselines.append(baseline)
        candidates.append(candidate)
        receipts.append(receipt)
    require(len(sources) == SELECTED_FRAME_COUNT, "R11_PHASE_B_LINEAGE_COUNT", "selected lineage count drift")
    return sources, baselines, candidates, receipts


def _read_faro_member(bundle: zipfile.ZipFile, frame: SelectedFrame, ledger: Counter[str]) -> bytes:
    require(frame.highres_member.role == "highres_depth", "R11_PHASE_B_PAYLOAD_ROLE", "non-FARO member capability reached reader")
    ledger["attempt:highres_depth"] += 1
    try:
        info = bundle.getinfo(frame.highres_member.path)
    except KeyError as error:
        raise R11PhaseBError("R11_PHASE_B_FARO_MEMBER", "bound highres member is absent") from error
    require(
        int(info.file_size) == frame.highres_member.bytes
        and f"{info.CRC:08X}" == frame.highres_member.crc32,
        "R11_PHASE_B_FARO_MEMBER",
        "bound highres member metadata drift",
    )
    payload = bundle.read(info)
    require(len(payload) == frame.highres_member.bytes, "R11_PHASE_B_FARO_BYTES", "FARO payload byte count drift")
    ledger["completed:highres_depth"] += 1
    ledger["bytes:highres_depth"] += len(payload)
    return payload


def _label_relative(frame: SelectedFrame) -> str:
    return f"labels/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def build_label_frame_record(
    source_frame_record: Mapping[str, Any],
    highres_faro_depth_mm: Any,
    intrinsics_highres_3x3: Any,
    gravity_up_camera_xyz: Any,
) -> dict[str, Any]:
    """Build the frozen R7 FARO label while reusing geometry once per physical frame."""

    source = r7_canary.validate_source_frame_record(dict(source_frame_record))
    faro = np.asarray(highres_faro_depth_mm)
    require(
        faro.shape == adapter.HIGHRES_SHAPE_HW and faro.dtype == np.uint16,
        "R11_PHASE_B_FARO_DEPTH",
        "FARO label depth must be uint16 1440x1920",
    )
    matrix = r7_canary._matrix(intrinsics_highres_3x3, "intrinsics_highres_3x3")
    gravity = adapter._normalize_vector(gravity_up_camera_xyz, "R11_PHASE_B_GRAVITY")
    require(
        adapter.canonical_sha256(matrix) == source["input_bindings"]["intrinsics_highres_sha256"]
        and adapter.canonical_sha256(gravity) == source["input_bindings"]["gravity_up_camera_xyz_sha256"],
        "R11_PHASE_B_LABEL_LINEAGE",
        "FARO label intrinsics/gravity lineage drift",
    )
    faro_m = np.ascontiguousarray(faro.astype(np.float64) / 1000.0, dtype=np.float64)
    plane = prospective._fit_depth_plane(faro_m, matrix, gravity)
    geometry = prospective._build_geometry(faro_m, adapter.canonical_sha256(faro_m), matrix) if plane["evaluable"] else None
    labels = []
    for query in source["query_features"]:
        if query["query_receipt"] is None:
            label = {
                "state": "UNKNOWN", "obstacle_pixel_count": 0,
                "minimum_truth_obstacle_pixels": r7_canary.MINIMUM_TRUTH_OBSTACLE_PIXELS,
                "query_support_points": 0, "observed_forward_m": None, "local_valid_fraction": 0.0,
                "reason_codes": ["SOURCE_QUERY_FRAME_UNAVAILABLE"],
            }
        elif geometry is None:
            label = {
                "state": "UNKNOWN", "obstacle_pixel_count": 0,
                "minimum_truth_obstacle_pixels": r7_canary.MINIMUM_TRUTH_OBSTACLE_PIXELS,
                "query_support_points": 0, "observed_forward_m": None, "local_valid_fraction": 0.0,
                "reason_codes": list(plane["reason_codes"]),
            }
        else:
            label = r7_canary._truth_query_label(geometry, plane, matrix, query["query_receipt"])
        labels.append({"grid_index": query["grid_index"], "query_id": query["query_id"], **label})
    return r7_canary.validate_label_frame_record(
        _seal(
            {
                "schema": r7_canary.LABEL_FRAME_SCHEMA,
                "reducer_id": r7_canary.REDUCER_ID,
                "parent_id": source["parent_id"], "video_id": source["video_id"],
                "timestamp_token": source["timestamp_token"], "physical_frame_id": source["physical_frame_id"],
                "source_frame_record_sha256": source["content_sha256"],
                "highres_faro_depth_sha256": adapter.canonical_sha256(faro),
                "truth_plane": plane, "query_labels": labels,
                "source_phase_reselection": False, "unknown_is_negative": False,
            }
        ),
        source,
    )


def _resource_snapshot(process: psutil.Process, started: float, reserve: int = 0) -> dict[str, Any]:
    elapsed = time.monotonic() - started
    peak = getattr(process.memory_info(), "peak_wset", None)
    require(isinstance(peak, int) and peak > 0, "R11_PHASE_B_PEAK_RSS", "OS peak RSS unavailable")
    require(elapsed + reserve <= EXPECTED_RESOURCE_BUDGET["maximum_wall_seconds"], "R11_PHASE_B_TIMEOUT", "wall budget exceeded")
    require(peak <= EXPECTED_RESOURCE_BUDGET["maximum_peak_rss_bytes"], "R11_PHASE_B_RSS", "peak RSS budget exceeded")
    return {"elapsed_seconds": round(float(elapsed), 6), "peak_rss_bytes": peak}


def _allocate_reserve(writer: FactorEvidenceWriter) -> None:
    path = writer.root / TERMINAL_RESERVE_NAME
    with path.open("xb") as stream:
        stream.write(bytes(TERMINAL_RESERVE_BYTES))
        stream.flush()
        os.fsync(stream.fileno())


def _release_reserve(writer: FactorEvidenceWriter) -> None:
    (writer.root / TERMINAL_RESERVE_NAME).unlink(missing_ok=True)


def validate_execution_lock(path: Path, *, require_argv: bool = True) -> dict[str, Any]:
    lock_path = path.resolve()
    require(lock_path == _repo_path(LOCK_RELATIVE), "R11_PHASE_B_LOCK_PATH", "execution lock path drift")
    lock = _validate_seal(_load_json(lock_path), LOCK_SCHEMA)
    require(
        lock.get("lock_id") == LOCK_ID
        and lock.get("status") == "AUTHORIZED_UNCONSUMED"
        and lock.get("consumed") is False
        and lock.get("argv") == EXPECTED_ARGV
        and lock.get("inventory_path") == INVENTORY_RELATIVE
        and lock.get("phase_a_root") == PHASE_A_ROOT
        and lock.get("top24_root") == TOP24_ROOT
        and lock.get("output_root") == OUTPUT_ROOT
        and lock.get("overwrite") is lock.get("rerun") is False
        and lock.get("user_authority") == EXPECTED_USER_AUTHORITY
        and lock.get("execution_authority") == EXPECTED_AUTHORITY
        and lock.get("resource_budget") == EXPECTED_RESOURCE_BUDGET
        and lock.get("one_shot_policy") == EXPECTED_ONE_SHOT_POLICY
        and lock.get("evaluability_gates") == phase_b_metrics.EVALUABILITY_GATES
        and lock.get("confirmation_gates") == phase_b_metrics.CONFIRMATION_GATES
        and lock.get("top24_selection_content_sha256") == TOP24_SELECTION_CONTENT_SHA256
        and lock.get("implementation_on_origin_master") is True
        and _commit_is_on_master(lock.get("implementation_commit")),
        "R11_PHASE_B_LOCK",
        "execution lock identity/authority/gate drift",
    )
    if require_argv:
        _validate_actual_argv()
    _validate_bindings(lock)
    validate_protocol_lock.validate_protocol(_load_json(_repo_path(PROTOCOL_RELATIVE)), repo_root=REPO_ROOT, recompute_pool=False)
    _scores, selection, _formal = _load_top24()
    inventory = phase_a._verify_inventory_evidence()
    frames = derive_selected_frames(inventory, selection)
    require(
        lock.get("selected_parent_identities") == selection["selected_parent_identities"]
        and lock.get("selected_frame_count") == len(frames)
        and lock.get("selected_query_count") == len(frames) * 9,
        "R11_PHASE_B_SELECTED_COHORT",
        "execution lock selected cohort drift",
    )
    require(not _repo_path(OUTPUT_ROOT).exists(), "R11_PHASE_B_ROOT_COLLISION", "formal Phase B root exists")
    lock["_path"] = lock_path
    lock["_frames"] = frames
    lock["_selection"] = selection
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    target = _repo_path(OUTPUT_ROOT)
    require(not target.exists(), "R11_PHASE_B_ROOT_COLLISION", "formal Phase B root exists")
    partial = target.parent / f"{target.name}.partial-{uuid.uuid4().hex}"
    require(not partial.exists(), "R11_PHASE_B_PARTIAL_COLLISION", "partial evidence root collision")
    writer = FactorEvidenceWriter(partial, EXPECTED_RESOURCE_BUDGET["maximum_evidence_bytes"] - TERMINAL_RESERVE_BYTES)
    started = time.monotonic()
    process = psutil.Process(os.getpid())
    try:
        writer.activate(
            _seal(
                {
                    "schema": "blindassist.taro.o1r.r11_selected_top24_faro_phase_b_execution_receipt.v1",
                    "execution_lock_sha256": materializer.sha256_file(lock["_path"]),
                    "execution_lock_content_sha256": lock["content_sha256"],
                    "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "selected_parent_count": SELECTED_PARENT_COUNT,
                    "expected_frame_count": SELECTED_FRAME_COUNT,
                    "only_payload_role_read": "highres_depth",
                    "read_unselected_parent_faro": False,
                    "source_reselection": False,
                    "parent_reselection": False,
                    "selector_fit": False,
                    "candidate_reselection": False,
                    "threshold_fit": False,
                    "model_executions": 0,
                    "training_steps": 0,
                    "network_requests": 0,
                    "one_shot_consumed_on_partial_root_creation": True,
                }
            )
        )
        _allocate_reserve(writer)
        frames = lock["_frames"]
        selection = lock["_selection"]
        sources, baselines, candidates, receipts = _load_selected_lineages(frames)
        ledger: Counter[str] = Counter()
        per_parent: Counter[tuple[str, str]] = Counter()
        labels: list[dict[str, Any]] = []
        label_hashes: list[str] = []
        grouped: dict[tuple[str, str], list[tuple[SelectedFrame, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for row in zip(frames, sources, receipts, strict=True):
            grouped[(row[0].parent_id, row[0].video_id)].append(row)
        selected_order = [tuple(row) for row in selection["selected_parent_identities"]]
        require(list(grouped) == selected_order, "R11_PHASE_B_PARENT_ORDER", "selected frame parent order drift")
        for identity in selected_order:
            rows = grouped[identity]
            with zipfile.ZipFile(rows[0][0].upsampling_archive) as bundle:
                for frame, source, receipt in rows:
                    payload = _read_faro_member(bundle, frame, ledger)
                    per_parent[identity] += 1
                    faro = materializer._decode_png(payload, "highres_depth")
                    label = build_label_frame_record(
                        source,
                        faro,
                        receipt["intrinsics_highres"]["matrix_3x3"],
                        receipt["gravity_up_camera_xyz"],
                    )
                    writer.write_json_gzip(_label_relative(frame), label)
                    labels.append(label)
                    label_hashes.append(label["content_sha256"])
                    if len(labels) % 50 == 0 or len(labels) == SELECTED_FRAME_COUNT:
                        _resource_snapshot(process, started)
                        print(json.dumps({"phase": "R11_SELECTED_FARO_LABEL", "completed": len(labels), "total": SELECTED_FRAME_COUNT}), flush=True)
        expected_parent_counts = {
            (str(row["parent_id"]), str(row["video_id"])): int(row["frame_count"])
            for row in selection["selected_parent_scores"]
        }
        require(
            ledger["attempt:highres_depth"] == ledger["completed:highres_depth"] == SELECTED_FRAME_COUNT
            and Counter(per_parent) == Counter(expected_parent_counts)
            and set(per_parent) == set(selected_order),
            "R11_PHASE_B_FARO_LEDGER",
            "selected-only FARO read ledger drift",
        )
        summary = phase_b_metrics.summarize(selected_order, baselines, candidates, labels)
        completion = _seal(
            {
                "schema": COMPLETION_SCHEMA,
                "selected_parent_count": SELECTED_PARENT_COUNT,
                "frame_count": SELECTED_FRAME_COUNT,
                "query_count": SELECTED_QUERY_COUNT,
                "label_hash_sequence_sha256": adapter.canonical_sha256(label_hashes),
                "top24_selection_sha256": TOP24_SELECTION_CONTENT_SHA256,
                "faro_read_attempts": ledger["attempt:highres_depth"],
                "faro_read_completed": ledger["completed:highres_depth"],
                "faro_read_bytes": ledger["bytes:highres_depth"],
                "per_parent_faro_reads": [
                    {"parent_id": identity[0], "video_id": identity[1], "reads": per_parent[identity]}
                    for identity in selected_order
                ],
                "unselected_faro_reads": 0,
                "only_payload_role_read": "highres_depth",
                "source_reselection": False,
                "parent_reselection": False,
                "selector_fit": False,
                "candidate_reselection": False,
                "threshold_fit": False,
                "model_executions": 0,
                "training_steps": 0,
                "network_requests": 0,
                "unknown_is_negative": False,
            }
        )
        writer.write_json("label-completion.json", completion)
        resource = _resource_snapshot(process, started, TERMINAL_WALL_RESERVE_SECONDS)
        result = _seal(
            {
                "schema": RESULT_SCHEMA,
                **summary,
                "execution_valid": True,
                "selected_parent_count": SELECTED_PARENT_COUNT,
                "frame_count": SELECTED_FRAME_COUNT,
                "query_count": SELECTED_QUERY_COUNT,
                "selected_parent_identities": selection["selected_parent_identities"],
                "top24_selection_sha256": TOP24_SELECTION_CONTENT_SHA256,
                "label_completion_sha256": completion["content_sha256"],
                "faro_frame_count": SELECTED_FRAME_COUNT,
                "unselected_faro_frame_count": 0,
                "source_reselection": False,
                "parent_reselection": False,
                "selector_fit": False,
                "candidate_reselection": False,
                "threshold_fit": False,
                "model_executions": 0,
                "training_steps": 0,
                "network_requests": 0,
                **resource,
                "resource_budget": EXPECTED_RESOURCE_BUDGET,
                "one_shot_consumed": True,
                "claim_ceiling": (
                    "Fresh-parent WILD_LAB factor confirmation only; no clear-output, deployment, device, "
                    "product, default-App, or safety claim."
                ),
            }
        )
        writer.write_json("result.json", result)
        require(len(writer.file_receipts) == SUCCESS_PRE_TERMINAL_FILE_COUNT, "R11_PHASE_B_FILE_COUNT", "pre-terminal file count drift")
        terminal = _seal(
            {
                "schema": TERMINAL_SCHEMA,
                "terminal": result["terminal"],
                "passed": result["passed"],
                "execution_valid": True,
                "result": result,
                "files": dict(sorted(writer.file_receipts.items())),
                "file_count_before_terminal": len(writer.file_receipts),
                "bytes_before_terminal": writer.bytes_written,
                "one_shot_consumed": True,
            }
        )
        terminal_bytes = len(adapter.canonical_json_bytes(terminal)) + 1
        require(
            terminal_bytes <= TERMINAL_RESERVE_BYTES
            and writer.bytes_written + terminal_bytes <= EXPECTED_RESOURCE_BUDGET["maximum_evidence_bytes"],
            "R11_PHASE_B_TERMINAL_RESERVE",
            "terminal exceeds reserved budget",
        )
        _resource_snapshot(process, started, TERMINAL_WALL_RESERVE_SECONDS)
        _release_reserve(writer)
        writer.maximum_bytes = EXPECTED_RESOURCE_BUDGET["maximum_evidence_bytes"]
        writer.write_json("terminal.json", terminal)
        actual = {path.relative_to(partial).as_posix() for path in partial.rglob("*") if path.is_file()}
        require(
            actual == set(writer.file_receipts)
            and len(actual) == SUCCESS_FINAL_FILE_COUNT
            and "terminal.json" in actual,
            "R11_PHASE_B_FINAL_SET",
            "final partial root exact file set drift",
        )
        os.replace(partial, target)
        return result
    except Exception as error:
        if writer.activated and partial.exists():
            try:
                _release_reserve(writer)
                writer.maximum_bytes = EXPECTED_RESOURCE_BUDGET["maximum_evidence_bytes"]
                if not (partial / "terminal.json").exists():
                    writer.write_json(
                        "terminal.json",
                        _seal(
                            {
                                "schema": TERMINAL_SCHEMA,
                                "terminal": INVALID_TERMINAL,
                                "passed": False,
                                "execution_valid": False,
                                "result": _seal(
                                    {
                                        "schema": FAILURE_SCHEMA,
                                        "terminal": INVALID_TERMINAL,
                                        "failure_code": str(getattr(error, "code", type(error).__name__))[:256],
                                        "message": str(error)[:4096],
                                        "unselected_faro_frame_count": 0,
                                        "one_shot_consumed": True,
                                    }
                                ),
                                "files": dict(sorted(writer.file_receipts.items())),
                                "file_count_before_terminal": len(writer.file_receipts),
                                "bytes_before_terminal": writer.bytes_written,
                                "one_shot_consumed": True,
                            }
                        ),
                    )
                os.replace(partial, target)
            except Exception:
                pass
        raise


def assert_public_api_selected_only() -> None:
    names = inspect.signature(derive_selected_frames).parameters
    require("selection" in names and "inventory" in names, "R11_PHASE_B_PUBLIC_API", "selected cohort API drift")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        print(json.dumps({"terminal": INVALID_TERMINAL, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "scientifically_evaluable": result["scientifically_evaluable"], "label_state_counts": result["label_state_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

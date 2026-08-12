#!/usr/bin/env python3
"""Read FARO only for the sealed R10 top eight and evaluate frozen gates."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any, Mapping, Sequence

import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation_io as r6io
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r10_clear_runtime import phase_b_metrics
from scripts.research.taro_o1r_r10_clear_runtime import run_pool_phase_a as phase_a
from scripts.research.taro_o1r_r10_clear_runtime import run_pool_phase_a_r1 as phase_a_r1
from scripts.research.taro_o1r_r10_clear_runtime import run_top8_selection as top8


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r10_fresh_pool_selected_phase_b_execution_lock.v1"
LOCK_ID = "TARO_O1R_R10_FRESH_POOL_SELECTED_TOP8_PHASE_B_FARO_ONE_SHOT_EXECUTION_LOCK"
PHASE_A_ROOT = phase_a_r1.OUTPUT_ROOT
SELECTION_ROOT = top8.OUTPUT_ROOT
INVENTORY_PATH = phase_a.INVENTORY_PATH
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r10-fresh-pool-selected-phase-b-r0"
PASS_TERMINAL = phase_b_metrics.PASS_TERMINAL
FAIL_TERMINAL = phase_b_metrics.FAIL_TERMINAL
NOT_EVALUABLE_TERMINAL = phase_b_metrics.NOT_EVALUABLE_TERMINAL
INVALID_TERMINAL = "TARO_O1R_R10_FRESH_CLEAR_ENRICHED_CONFIRMATION_EXECUTION_INVALID"
SELECTED_PARENT_COUNT = top8.SELECTED_PARENT_COUNT

EXPECTED_BINDINGS = {
    "R10_PROTOCOL": "docs/research/taro/TARO_O1R_R10_FRESH_PARENT_SOURCE_ONLY_CLEAR_ENRICHED_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json",
    "R10_INVENTORY_PLAN": INVENTORY_PATH,
    "R10_PHASE_A_R1_LOCK": "docs/research/taro/TARO_O1R_R10_FRESH_POOL_SOURCE_ONLY_PHASE_A_R1_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json",
    "R10_PHASE_A_R1_COMPLETION": f"{PHASE_A_ROOT}/phase-a-completion.json",
    "R10_PHASE_A_R1_RESULT": f"{PHASE_A_ROOT}/result.json",
    "R10_PHASE_A_R1_MANIFEST": f"{PHASE_A_ROOT}/manifest.json",
    "R10_PHASE_A_BASE_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_phase_a.py",
    "R10_PHASE_A_R1_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_phase_a_r1.py",
    "R10_TOP8_LOCK": "docs/research/taro/TARO_O1R_R10_FRESH_POOL_TOP8_SOURCE_ONLY_SELECTION_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json",
    "R10_TOP8_PARENT_SCORES": f"{SELECTION_ROOT}/parent-scores.json",
    "R10_TOP8_SELECTION": f"{SELECTION_ROOT}/selection.json",
    "R10_TOP8_RESULT": f"{SELECTION_ROOT}/result.json",
    "R10_TOP8_MANIFEST": f"{SELECTION_ROOT}/manifest.json",
    "R10_TOP8_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_top8_selection.py",
    "R10_PHASE_B_METRICS": "scripts/research/taro_o1r_r10_clear_runtime/phase_b_metrics.py",
    "R7_LABEL_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "R6_CONTAINER_IO": "scripts/research/taro_o0r_candidate_scale_runtime/r6_confirmation_io.py",
    "TRUTH_MATERIALIZER": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "R10_PHASE_B_RUNNER": "scripts/research/taro_o1r_r10_clear_runtime/run_selected_phase_b.py",
}
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "先推动，我授权",
    "scope": phase_a.AUTHORITY_SCOPE,
}
EXPECTED_BUDGET = {
    "maximum_wall_seconds": 14_400,
    "maximum_peak_rss_bytes": 17_179_869_184,
    "maximum_evidence_bytes": 536_870_912,
}


class SelectedPhaseBError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise SelectedPhaseBError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R10_PHASE_B_SEAL_COLLISION", "Phase-B caller supplied a seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _label_relative(frame: r6io.R6FrameRef) -> str:
    return f"labels/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def derive_selected_cohort(selection: Mapping[str, Any]) -> dict[str, Any]:
    identities = selection.get("selected_parent_identities")
    scores = selection.get("selected_parent_scores")
    require(
        isinstance(identities, list)
        and len(identities) == SELECTED_PARENT_COUNT
        and len({tuple(row) for row in identities if isinstance(row, list) and len(row) == 2})
        == SELECTED_PARENT_COUNT
        and isinstance(scores, list)
        and len(scores) == SELECTED_PARENT_COUNT,
        "R10_PHASE_B_SELECTION_COUNT",
        "R10 Phase-B requires eight unique sealed parent identities and scores",
    )
    normalized = [[str(row[0]), str(row[1])] for row in identities]
    require(
        normalized == [[str(row.get("parent_id")), str(row.get("video_id"))] for row in scores]
        and all(isinstance(row.get("frame_count"), int) and row["frame_count"] > 0 for row in scores),
        "R10_PHASE_B_SELECTION_SCORE_DRIFT",
        "sealed selected identities and frame counts drift",
    )
    frame_count = sum(int(row["frame_count"]) for row in scores)
    return {
        "parent_count": SELECTED_PARENT_COUNT,
        "physical_frame_count": frame_count,
        "query_count": frame_count * 9,
        "selected_parent_identities": normalized,
        "selection_sha256": selection.get("content_sha256"),
        "parent_scores_sha256": selection.get("parent_scores_sha256"),
    }


def select_frames(
    all_frames: Sequence[r6io.R6FrameRef],
    selected_identities: Sequence[Sequence[str]],
) -> list[r6io.R6FrameRef]:
    selected = {(str(row[0]), str(row[1])) for row in selected_identities}
    require(len(selected) == SELECTED_PARENT_COUNT, "R10_PHASE_B_SELECTION_COUNT", "selected identity set drift")
    frames = [frame for frame in all_frames if (frame.parent_id, frame.video_id) in selected]
    require(
        {(frame.parent_id, frame.video_id) for frame in frames} == selected,
        "R10_PHASE_B_SELECTED_PARENT_MISSING",
        "one or more sealed selected parents have no inventory frames",
    )
    return frames


def validate_faro_read_counts(
    faro_reads: Mapping[str, int],
    per_parent_reads: Mapping[tuple[str, str], int],
    frames: Sequence[r6io.R6FrameRef],
    selected_identities: Sequence[Sequence[str]],
) -> dict[str, Any]:
    selected = {(str(row[0]), str(row[1])) for row in selected_identities}
    expected: Counter[tuple[str, str]] = Counter((frame.parent_id, frame.video_id) for frame in frames)
    require(
        dict(faro_reads) == {"highres_depth": len(frames)}
        and Counter(per_parent_reads) == expected
        and set(per_parent_reads).issubset(selected),
        "R10_PHASE_B_FARO_COUNT",
        "FARO reads are not exactly one highres_depth payload per selected frame",
    )
    return {
        "selected_highres_depth_reads": len(frames),
        "unselected_highres_depth_reads": 0,
        "only_payload_role_read": "highres_depth",
    }


def _verify_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    schema: str,
    terminal: str,
    expected_paths: set[str],
) -> None:
    files = manifest.get("files")
    require(
        manifest.get("schema") == schema
        and manifest.get("terminal") == terminal
        and isinstance(files, dict)
        and len(files) == manifest.get("file_count_before_manifest") == len(expected_paths)
        and set(files) == expected_paths,
        "R10_PHASE_B_MANIFEST_DRIFT",
        "bound manifest identity, terminal, cardinality, or file set drift",
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
            "R10_PHASE_B_MANIFEST_FILE_DRIFT",
            f"bound artifact drift: {relative}",
        )
        total += target.stat().st_size
    require(total == manifest.get("bytes_before_manifest"), "R10_PHASE_B_MANIFEST_BYTE_DRIFT", "bound manifest byte total drift")


def load_selection_bundle() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _repo_path(SELECTION_ROOT)
    parent_scores = top8.validate_parent_scores(_read_json(root / "parent-scores.json"))
    selection = top8.validate_selection(_read_json(root / "selection.json"), parent_scores)
    result = _read_json(root / "result.json")
    require(
        result.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_top8_selection_result.v1"
        and result.get("terminal") == top8.PASS_TERMINAL
        and result.get("passed") is True
        and result.get("execution_valid") is True
        and result.get("parent_scores_sha256") == parent_scores["content_sha256"]
        and result.get("selection_sha256") == selection["content_sha256"]
        and result.get("selected_parent_identities") == selection["selected_parent_identities"]
        and result.get("faro_reads") == result.get("truth_reads") == result.get("label_reads") == result.get("outcome_reads") == 0,
        "R10_PHASE_B_SELECTION_RESULT",
        "R10 sealed top-eight result is not admitted",
    )
    _verify_manifest(
        root,
        _read_json(root / "manifest.json"),
        "blindassist.taro.o1r.r10_fresh_pool_top8_selection_manifest.v1",
        top8.PASS_TERMINAL,
        {"execution-receipt.json", "parent-scores.json", "selection.json", "result.json"},
    )
    return parent_scores, selection, result


def load_selected_rows() -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[r6io.R6FrameRef],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    parent_scores, selection, _result = load_selection_bundle()
    cohort = derive_selected_cohort(selection)
    all_frames = phase_a._load_frames(_repo_path(INVENTORY_PATH))
    frames = select_frames(all_frames, cohort["selected_parent_identities"])
    require(
        len(frames) == cohort["physical_frame_count"],
        "R10_PHASE_B_SELECTED_FRAME_COUNT",
        "selected frame count differs from the sealed source-only scores",
    )
    completion = phase_a._validate_seal(
        _read_json(_repo_path(PHASE_A_ROOT) / "phase-a-completion.json"),
        "blindassist.taro.o1r.r10_fresh_pool_phase_a_completion.v1",
    )
    require(
        parent_scores["phase_a_completion_sha256"] == completion["content_sha256"]
        and parent_scores["source_frame_hash_sequence_sha256"]
        == completion["source_frame_hash_sequence_sha256"]
        and completion["faro_reads"] == completion["truth_reads"] == 0
        and completion["all_source_records_sealed_before_faro"] is True,
        "R10_PHASE_B_PHASE_A_LINEAGE",
        "sealed selection does not bind the admitted source-only Phase A completion",
    )

    sources: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for frame in frames:
        lineage = _read_gzip_json(_repo_path(PHASE_A_ROOT) / phase_a._lineage_relative(frame))
        require(
            isinstance(lineage, dict)
            and set(lineage) == {"prospective_bundle", "r6_reducer_bundle", "r7_source_frame_record"},
            "R10_PHASE_B_SOURCE_LINEAGE_SCHEMA",
            "selected source lineage schema drift",
        )
        source = r7_canary.validate_source_frame_record(lineage["r7_source_frame_record"])
        receipt = phase_a._validate_seal(
            _read_json(_repo_path(PHASE_A_ROOT) / phase_a._source_receipt_relative(frame)),
            "blindassist.taro.o1r.r10_fresh_pool_source_frame_receipt.v1",
        )
        require(
            source["physical_frame_id"] == frame.physical_frame_id
            and source["parent_id"] == frame.parent_id
            and source["video_id"] == frame.video_id
            and source["timestamp_token"] == frame.timestamp_token
            and source["source_frame_receipt_sha256"] == receipt["content_sha256"]
            and receipt["faro_payload_read"] is False
            and receipt["truth_payload_read"] is False,
            "R10_PHASE_B_SOURCE_LINEAGE",
            "selected source receipt/record lineage drift",
        )
        sources.append(source)
        receipts.append(receipt)

    by_parent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        by_parent[(source["parent_id"], source["video_id"])].append(source)
    sealed_scores = {
        (row["parent_id"], row["video_id"]): row for row in selection["selected_parent_scores"]
    }
    for identity in map(tuple, cohort["selected_parent_identities"]):
        require(
            top8.score_parent(by_parent[identity], top8.FROZEN_RULE) == sealed_scores[identity],
            "R10_PHASE_B_SELECTED_SOURCE_SCORE_DRIFT",
            "selected source lineage no longer reproduces its sealed parent score",
        )
    return parent_scores, selection, frames, sources, receipts


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = _read_json(lock_path)
    require(
        lock.get("schema") == LOCK_SCHEMA
        and lock.get("lock_id") == LOCK_ID
        and lock.get("status") == "AUTHORIZED_UNCONSUMED"
        and lock.get("consumed") is False,
        "R10_PHASE_B_LOCK_IDENTITY",
        "R10 Phase-B lock identity drift",
    )
    require(lock.get("user_authority") == EXPECTED_USER_AUTHORITY, "R10_PHASE_B_USER_AUTHORITY", "R10 Phase-B user authority drift")
    actual_argv = [
        Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(),
        "--execution-lock",
        lock_path.relative_to(REPO_ROOT).as_posix(),
    ]
    require(
        lock.get("argv") == actual_argv
        and lock.get("phase_a_root") == PHASE_A_ROOT
        and lock.get("selection_root") == SELECTION_ROOT
        and lock.get("inventory_path") == INVENTORY_PATH
        and lock.get("output_root") == OUTPUT_ROOT
        and lock.get("overwrite") is False
        and lock.get("rerun") is False,
        "R10_PHASE_B_LOCK_POLICY",
        "R10 Phase-B argv/root policy drift",
    )
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R10_PHASE_B_BINDINGS", "R10 Phase-B binding count drift")
    seen: set[str] = set()
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(
            set(row) == {"role", "path", "bytes", "sha256"}
            and role not in seen
            and EXPECTED_BINDINGS.get(role) == relative,
            "R10_PHASE_B_BINDING_ROW",
            "R10 Phase-B binding row drift",
        )
        seen.add(role)
        target = _repo_path(relative)
        require(
            target.is_file()
            and target.stat().st_size == row["bytes"]
            and materializer.sha256_file(target) == row["sha256"],
            "R10_PHASE_B_BINDING_HASH",
            f"R10 Phase-B binding drift: {relative}",
        )
    require(seen == set(EXPECTED_BINDINGS), "R10_PHASE_B_BINDINGS", "R10 Phase-B binding roles drift")
    top8.validate_protocol(_read_json(_repo_path(EXPECTED_BINDINGS["R10_PROTOCOL"])))

    phase_result = _read_json(_repo_path(EXPECTED_BINDINGS["R10_PHASE_A_R1_RESULT"]))
    phase_manifest = _read_json(_repo_path(EXPECTED_BINDINGS["R10_PHASE_A_R1_MANIFEST"]))
    require(
        phase_result.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_phase_a_result.v1"
        and phase_result.get("terminal") == phase_a_r1.PASS_TERMINAL
        and phase_result.get("passed") is True
        and phase_result.get("execution_valid") is True
        and phase_result.get("parent_count") == phase_a.PARENT_COUNT
        and phase_result.get("frame_count") == phase_a.FRAME_COUNT
        and phase_result.get("query_count") == phase_a.QUERY_COUNT
        and phase_result.get("faro_reads") == 0
        and phase_result.get("truth_scoring") is False
        and phase_manifest.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_phase_a_manifest.v1"
        and phase_manifest.get("terminal") == phase_a_r1.PASS_TERMINAL
        and phase_manifest.get("file_count_before_manifest") == top8.PHASE_A_FILE_COUNT
        and len(phase_manifest.get("files", {})) == top8.PHASE_A_FILE_COUNT,
        "R10_PHASE_B_PHASE_A_NOT_ADMITTED",
        "R10 Phase-A R1 result/manifest is not admitted",
    )

    _scores, selection, _selection_result = load_selection_bundle()
    cohort = derive_selected_cohort(selection)
    expected_authority = {
        "phase_a_r1_reload": True,
        "sealed_top8_reload": True,
        "faro_payload_read": True,
        "selected_faro_frame_count": cohort["physical_frame_count"],
        "unselected_faro_frame_count": 0,
        "truth_label_construction": True,
        "fixed_gate_evaluation": True,
        "source_reselection": False,
        "selector_fit": False,
        "threshold_fit": False,
        "training": False,
        "network": False,
        "device": False,
        "deployment": False,
        "product": False,
        "safety": False,
    }
    require(lock.get("execution_authority") == expected_authority, "R10_PHASE_B_AUTHORITY", "R10 Phase-B authority drift")
    require(lock.get("selected_cohort") == cohort, "R10_PHASE_B_SELECTED_COHORT", "R10 Phase-B selected cohort drift")
    require(lock.get("unchanged_gates") == phase_b_metrics.EXPECTED_GATES, "R10_PHASE_B_GATE_DRIFT", "R10 Phase-B gates drift")
    require(
        lock.get("phase_firewall")
        == {
            "selection_sha256": selection["content_sha256"],
            "source_reselection": False,
            "selector_fit": False,
            "threshold_reselection": False,
            "only_payload_role_read": "highres_depth",
            "read_unselected_parent_faro": False,
            "unknown_is_negative": False,
        },
        "R10_PHASE_B_FIREWALL",
        "R10 Phase-B firewall drift",
    )
    require(
        lock.get("resource_budget") == EXPECTED_BUDGET
        and lock.get("one_shot_policy")
        == {
            "consumed_on_output_root_creation": True,
            "failure_does_not_restore_authority": True,
            "expected_file_count_before_manifest": cohort["physical_frame_count"] + 3,
        },
        "R10_PHASE_B_BUDGET",
        "R10 Phase-B budget or one-shot policy drift",
    )
    require(not _repo_path(OUTPUT_ROOT).exists(), "R10_PHASE_B_ROOT_COLLISION", "R10 Phase-B output root exists")
    lock["_lock_path"] = lock_path
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    budget = lock["resource_budget"]
    writer = FactorEvidenceWriter(_repo_path(OUTPUT_ROOT), int(budget["maximum_evidence_bytes"]))
    started = time.monotonic()
    process = psutil.Process()
    cohort = lock["selected_cohort"]
    writer.activate(
        {
            "schema": "blindassist.taro.o1r.r10_fresh_pool_selected_phase_b_execution_receipt.v1",
            "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]),
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selected_parent_count": SELECTED_PARENT_COUNT,
            "expected_frame_count": cohort["physical_frame_count"],
            "phase_a_and_top8_reloaded_before_faro": True,
            "only_payload_role_read": "highres_depth",
            "read_unselected_parent_faro": False,
            "source_reselection": False,
            "selector_fit": False,
            "threshold_reselection": False,
            "unknown_is_negative": False,
            "training_steps": 0,
            "network_requests": 0,
            "one_shot_consumed_on_root_creation": True,
        }
    )
    try:
        parent_scores, selection, frames, sources, receipts = load_selected_rows()
        require(len(frames) == cohort["physical_frame_count"], "R10_PHASE_B_SELECTED_FRAME_COUNT", "selected frame count changed after lock validation")
        faro_reads: Counter[str] = Counter()
        per_parent_reads: Counter[tuple[str, str]] = Counter()
        labels: list[dict[str, Any]] = []
        label_hashes: list[str] = []

        def observed(role: str, _: str) -> None:
            require(role == "highres_depth", "R10_PHASE_B_PAYLOAD_FIREWALL", "Phase-B attempted a non-FARO payload read")
            faro_reads[role] += 1

        packed = list(zip(frames, sources, receipts, strict=True))
        for _identity, parent_rows_iter in groupby(packed, key=lambda row: (row[0].parent_id, row[0].video_id)):
            parent_rows = list(parent_rows_iter)
            with zipfile.ZipFile(parent_rows[0][0].upsampling_archive) as bundle:
                for frame, source, receipt in parent_rows:
                    faro_payload, _binding = r6io._read_member(
                        bundle,
                        frame.members["highres_depth"],
                        observer=observed,
                    )
                    per_parent_reads[(frame.parent_id, frame.video_id)] += 1
                    faro = materializer._decode_png(faro_payload, "highres_depth")
                    label = r7_canary.build_label_frame_record(
                        source,
                        faro,
                        receipt["intrinsics_highres"]["matrix_3x3"],
                        receipt["gravity_up_camera_xyz"],
                    )
                    writer.write_json_gzip(_label_relative(frame), label)
                    labels.append(label)
                    label_hashes.append(label["content_sha256"])
                    require(
                        time.monotonic() - started <= budget["maximum_wall_seconds"]
                        and process.memory_info().rss <= budget["maximum_peak_rss_bytes"],
                        "R10_PHASE_B_RESOURCE",
                        "R10 Phase-B resource budget exceeded",
                    )
                    if len(labels) % 10 == 0 or len(labels) == len(frames):
                        print(
                            json.dumps(
                                {
                                    "phase": "R10_SELECTED_FARO_LABEL",
                                    "completed": len(labels),
                                    "total": len(frames),
                                    "physical_frame_id": frame.physical_frame_id,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

        read_summary = validate_faro_read_counts(
            faro_reads,
            per_parent_reads,
            frames,
            selection["selected_parent_identities"],
        )
        identities = [tuple(row) for row in selection["selected_parent_identities"]]
        summary = phase_b_metrics.summarize(identities, sources, labels)
        require(summary.get("unknown_is_negative") is False, "R10_PHASE_B_UNKNOWN_SEMANTICS", "UNKNOWN entered the negative class")
        completion = _seal(
            {
                "schema": "blindassist.taro.o1r.r10_fresh_pool_selected_phase_b_label_completion.v1",
                "selected_parent_count": SELECTED_PARENT_COUNT,
                "frame_count": len(frames),
                "query_count": len(frames) * 9,
                "label_hash_sequence_sha256": adapter.canonical_sha256(label_hashes),
                "phase_a_completion_sha256": parent_scores["phase_a_completion_sha256"],
                "parent_scores_sha256": parent_scores["content_sha256"],
                "selection_sha256": selection["content_sha256"],
                "faro_payload_reads": dict(faro_reads),
                **read_summary,
                "source_reselection": False,
                "selector_fit": False,
                "threshold_reselection": False,
                "unknown_is_negative": False,
            }
        )
        writer.write_json("label-completion.json", completion)
        result = {
            "schema": "blindassist.taro.o1r.r10_fresh_clear_enriched_confirmation_result.v1",
            **summary,
            "execution_valid": True,
            "selected_parent_count": SELECTED_PARENT_COUNT,
            "frame_count": len(frames),
            "query_count": len(frames) * 9,
            "parent_scores_sha256": parent_scores["content_sha256"],
            "selection_sha256": selection["content_sha256"],
            "phase_a_completion_sha256": parent_scores["phase_a_completion_sha256"],
            "label_completion_sha256": completion["content_sha256"],
            "faro_frame_count": len(frames),
            "unselected_faro_frame_count": 0,
            "source_reselection": False,
            "selector_fit": False,
            "threshold_reselection": False,
            "training_steps": 0,
            "network_requests": 0,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "one_shot_consumed": True,
            "promotion_scope": "RESEARCH_ROUTE_POSITIVE_OCCUPANCY_FACTOR_ONLY" if summary["passed"] else None,
            "clear_branch_promotion": False,
            "claim_ceiling": "Fresh-parent WILD_LAB confirmation of the positive-occupancy factor and its definite-clear negative control only; no clear-output, deployment, device, product, or safety claim.",
        }
        writer.write_json("result.json", result)
        expected_files = len(frames) + 3
        require(
            len(writer.file_receipts) == expected_files,
            "R10_PHASE_B_MANIFEST_COUNT",
            "R10 Phase-B file count before manifest drift",
        )
        writer.write_json(
            "manifest.json",
            {
                "schema": "blindassist.taro.o1r.r10_fresh_pool_selected_phase_b_manifest.v1",
                "terminal": result["terminal"],
                "files": dict(sorted(writer.file_receipts.items())),
                "file_count_before_manifest": len(writer.file_receipts),
                "bytes_before_manifest": writer.bytes_written,
            },
        )
        return result
    except Exception as error:
        try:
            writer.write_json(
                "failure.json",
                {
                    "schema": "blindassist.taro.o1r.r10_fresh_pool_selected_phase_b_failure.v1",
                    "terminal": INVALID_TERMINAL,
                    "execution_valid": False,
                    "failure_code": str(getattr(error, "code", type(error).__name__)),
                    "message": str(error),
                    "unselected_faro_frame_count": 0,
                    "one_shot_consumed": True,
                },
            )
            writer.write_json(
                "manifest.json",
                {
                    "schema": "blindassist.taro.o1r.r10_fresh_pool_selected_phase_b_manifest.v1",
                    "terminal": INVALID_TERMINAL,
                    "files": dict(sorted(writer.file_receipts.items())),
                    "file_count_before_manifest": len(writer.file_receipts),
                    "bytes_before_manifest": writer.bytes_written,
                },
            )
        except Exception:
            pass
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
                    "terminal": INVALID_TERMINAL,
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
                "scientifically_evaluable": result["scientifically_evaluable"],
                "label_state_counts": result["label_state_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

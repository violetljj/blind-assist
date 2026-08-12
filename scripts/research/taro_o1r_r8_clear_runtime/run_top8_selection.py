#!/usr/bin/env python3
"""Seal the R8 top-eight parent selection using only completed Phase-A records."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_phase_a as shared_phase_a
from scripts.research.taro_o1r_r8_clear_runtime import clear_enrichment
from scripts.research.taro_o1r_r8_clear_runtime import pool_cohort
from scripts.research.taro_o1r_r8_clear_runtime import run_pool_phase_a
from scripts.research.taro_o1r_r8_clear_runtime import run_pool_phase_a_recovery


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r8_clear_top8_selection_execution_lock.v1"
LOCK_ID = "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_TOP8_SOURCE_ONLY_SELECTION_ONE_SHOT_EXECUTION_LOCK"
PHASE_A_ROOT = run_pool_phase_a_recovery.OUTPUT_ROOT
INVENTORY_PATH = run_pool_phase_a.INVENTORY_PATH
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r8-clear-top8-selection-r0"
PASS_TERMINAL = "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_TOP8_SOURCE_ONLY_SELECTION_SEALED_PASS"
FAIL_TERMINAL = "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_TOP8_SELECTION_EXECUTION_INVALID"
PARENT_COUNT = 24
FRAME_COUNT = 402
QUERY_COUNT = FRAME_COUNT * 9
PHASE_A_FILE_COUNT = FRAME_COUNT * 2 + 4

EXPECTED_BINDINGS = {
    "R8_PROTOCOL": "docs/research/taro/TARO_O1R_R8_SOURCE_ONLY_CLEAR_NEGATIVE_CONTROL_COHORT_ENRICHMENT_PROTOCOL_LOCK_2026-08-12.json",
    "R8_INVENTORY_PLAN": INVENTORY_PATH,
    "R8_PHASE_A_RESULT": f"{PHASE_A_ROOT}/result.json",
    "R8_PHASE_A_COMPLETION": f"{PHASE_A_ROOT}/phase-a-completion.json",
    "R8_PHASE_A_MANIFEST": f"{PHASE_A_ROOT}/manifest.json",
    "R8_PHASE_A_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_pool_phase_a.py",
    "R8_PHASE_A_RECOVERY_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_pool_phase_a_recovery.py",
    "SHARED_PHASE_A_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_phase_a.py",
    "SOURCE_RECORD_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "SOURCE_ONLY_SELECTOR": "scripts/research/taro_o1r_r8_clear_runtime/clear_enrichment.py",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "R8_SELECTION_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_top8_selection.py",
}
EXPECTED_AUTHORITY = {
    "sealed_phase_a_reload": True,
    "source_only_parent_scoring": True,
    "top8_selection": True,
    "faro_read": False,
    "truth_read": False,
    "label_read": False,
    "outcome_read": False,
    "candidate_rerun": False,
    "threshold_fit": False,
    "training": False,
    "network": False,
}
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "推进taro成功落地",
    "scope": "Reload the exact sealed R8 24-parent source-only recovery Phase A, score all parents with the frozen truth-blind selector, and irreversibly seal the final top eight before any FARO read.",
}


class Top8SelectionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise Top8SelectionError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R8_SELECTION_SEAL_COLLISION", "selection caller supplied a seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def validate_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(record.get("schema") == "blindassist.taro.o1r.r8_clear_top8_source_only_selection.v1" and isinstance(observed, str) and adapter.canonical_sha256(record) == observed, "R8_SELECTION_HASH_DRIFT", "top-eight selection seal drift")
    scores = record.get("parent_scores")
    selected = record.get("selected_parents")
    require(isinstance(scores, list) and len(scores) == PARENT_COUNT and isinstance(selected, list) and len(selected) == 8, "R8_SELECTION_CARDINALITY", "selection cardinality drift")
    recomputed = clear_enrichment.select_final_parents(scores)
    require([(row["parent_id"], row["video_id"]) for row in selected] == [(row["parent_id"], row["video_id"]) for row in recomputed], "R8_SELECTION_TOP8_DRIFT", "sealed top eight differs from frozen selector")
    require(record.get("faro_reads") == record.get("truth_reads") == record.get("label_reads") == record.get("outcome_reads") == 0 and record.get("clear_output_emitted") is False and record.get("training_steps") == 0, "R8_SELECTION_FIREWALL", "selection result-side firewall drift")
    record["content_sha256"] = observed
    return record


def verify_phase_a_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    require(manifest.get("schema") == "blindassist.taro.o1r.r8_clear_pool_phase_a_recovery_manifest.v1" and manifest.get("terminal") == run_pool_phase_a_recovery.PASS_TERMINAL, "R8_SELECTION_PHASE_A_MANIFEST", "R8 Phase-A manifest terminal drift")
    files = manifest.get("files")
    require(isinstance(files, dict) and len(files) == manifest.get("file_count_before_manifest") == PHASE_A_FILE_COUNT, "R8_SELECTION_PHASE_A_MANIFEST", "R8 Phase-A manifest cardinality drift")
    total = 0
    for relative, receipt in files.items():
        target = materializer.safe_join(root, relative)
        require(target.is_file() and target.stat().st_size == receipt.get("bytes") and materializer.sha256_file(target) == receipt.get("sha256") and receipt.get("path") == relative, "R8_SELECTION_PHASE_A_FILE_DRIFT", f"R8 Phase-A artifact drift: {relative}")
        total += target.stat().st_size
    require(total == manifest.get("bytes_before_manifest"), "R8_SELECTION_PHASE_A_BYTE_DRIFT", "R8 Phase-A byte total drift")
    return manifest


def load_phase_a_sources(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = run_pool_phase_a.load_frames(_repo_path(INVENTORY_PATH))
    completion = run_pool_phase_a_recovery.validate_completion(_read_json(root / "phase-a-completion.json"))
    require(completion["parent_count"] == PARENT_COUNT and completion["frame_count"] == FRAME_COUNT and completion["query_count"] == QUERY_COUNT and completion["faro_reads"] == completion["truth_reads"] == 0 and completion["clear_output_allowed"] is False, "R8_SELECTION_PHASE_A_COMPLETION", "R8 Phase-A completion not admitted")
    sources: list[dict[str, Any]] = []
    for frame in frames:
        lineage = _read_gzip_json(root / shared_phase_a._lineage_relative(frame))
        source = r7_canary.validate_source_frame_record(lineage["r7_source_frame_record"])
        require(source["physical_frame_id"] == frame.physical_frame_id, "R8_SELECTION_SOURCE_IDENTITY", "R8 source record identity drift")
        receipt = shared_phase_a._validate_seal(_read_json(root / shared_phase_a._source_receipt_relative(frame)), "blindassist.taro.o1r.r7_fresh_source_frame_receipt.v1")
        require(source["source_frame_receipt_sha256"] == receipt["content_sha256"], "R8_SELECTION_SOURCE_LINEAGE", "R8 source receipt lineage drift")
        sources.append(source)
    require(adapter.canonical_sha256([row["content_sha256"] for row in sources]) == completion["source_frame_hash_sequence_sha256"], "R8_SELECTION_SOURCE_SEQUENCE", "R8 source hash sequence drift")
    return completion, sources


def build_selection(completion: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(sources) == FRAME_COUNT, "R8_SELECTION_SOURCE_COUNT", "R8 source record count drift")
    by_parent: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for source in sources:
        by_parent[(str(source["parent_id"]), str(source["video_id"]))].append(source)
    expected_identities = [(visit, video) for visit, video, _ in pool_cohort.EXPECTED_POOL]
    require(list(by_parent) == expected_identities, "R8_SELECTION_PARENT_ORDER", "R8 source parent order drift")
    scores = [clear_enrichment.score_parent(by_parent[identity]) for identity in expected_identities]
    selected = clear_enrichment.select_final_parents(scores)
    ranked = sorted(scores, key=lambda row: (-int(row["eligible_query_count"]), -float(row["eligible_fraction_of_available"]), str(row["tie_break_sha256"])))
    require(ranked[:8] == selected, "R8_SELECTION_RANKING_DRIFT", "R8 selector ranking implementation drift")
    return validate_selection(_seal({
        "schema": "blindassist.taro.o1r.r8_clear_top8_source_only_selection.v1",
        "selector_id": clear_enrichment.SELECTOR_ID,
        "parent_count": PARENT_COUNT,
        "frame_count": FRAME_COUNT,
        "query_count": QUERY_COUNT,
        "phase_a_completion_sha256": completion["content_sha256"],
        "source_frame_hash_sequence_sha256": completion["source_frame_hash_sequence_sha256"],
        "parent_scores": scores,
        "ranked_parent_identities": [[row["parent_id"], row["video_id"]] for row in ranked],
        "selected_parents": selected,
        "selected_parent_count": 8,
        "faro_reads": 0,
        "truth_reads": 0,
        "label_reads": 0,
        "outcome_reads": 0,
        "clear_output_emitted": False,
        "candidate_rerun": False,
        "threshold_fit": False,
        "training_steps": 0,
        "selection_sealed_before_faro": True,
    }))


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = _read_json(lock_path)
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R8_SELECTION_LOCK_IDENTITY", "selection lock identity drift")
    require(lock.get("user_authority") == EXPECTED_USER_AUTHORITY, "R8_SELECTION_USER_AUTHORITY", "selection user authority drift")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv and lock.get("phase_a_root") == PHASE_A_ROOT and lock.get("inventory_path") == INVENTORY_PATH and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R8_SELECTION_LOCK_POLICY", "selection argv/root policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R8_SELECTION_BINDINGS", "selection binding count drift")
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and EXPECTED_BINDINGS.get(role) == relative, "R8_SELECTION_BINDING_ROW", "selection binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R8_SELECTION_BINDING_HASH", f"selection binding drift: {relative}")
    result = _read_json(_repo_path(EXPECTED_BINDINGS["R8_PHASE_A_RESULT"]))
    require(result.get("execution_valid") is True and result.get("passed") is True and result.get("terminal") == run_pool_phase_a_recovery.PASS_TERMINAL and result.get("parent_count") == PARENT_COUNT and result.get("frame_count") == FRAME_COUNT and result.get("candidate_inference_count") == 0 and result.get("faro_reads") == 0, "R8_SELECTION_PHASE_A_NOT_ADMITTED", "R8 Phase-A recovery result not admitted")
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY and lock.get("resource_budget") == {"maximum_evidence_bytes": 16777216}, "R8_SELECTION_AUTHORITY", "selection authority/budget drift")
    require(not _repo_path(OUTPUT_ROOT).exists(), "R8_SELECTION_ROOT_COLLISION", "selection output root exists")
    lock["_lock_path"] = lock_path
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    writer = FactorEvidenceWriter(_repo_path(OUTPUT_ROOT), int(lock["resource_budget"]["maximum_evidence_bytes"]))
    writer.activate({"schema": "blindassist.taro.o1r.r8_clear_top8_selection_execution_receipt.v1", "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "phase_a_reloaded": True, "faro_reads": 0, "truth_reads": 0, "one_shot_consumed_on_root_creation": True})
    try:
        phase_root = _repo_path(PHASE_A_ROOT)
        verify_phase_a_manifest(phase_root)
        completion, sources = load_phase_a_sources(phase_root)
        selection = build_selection(completion, sources)
        writer.write_json("selection.json", selection)
        result = {"schema": "blindassist.taro.o1r.r8_clear_top8_selection_result.v1", "terminal": PASS_TERMINAL, "passed": True, "execution_valid": True, "parent_count": PARENT_COUNT, "selected_parent_count": 8, "selected_parent_identities": [[row["parent_id"], row["video_id"]] for row in selection["selected_parents"]], "selection_sha256": selection["content_sha256"], "faro_reads": 0, "truth_reads": 0, "label_reads": 0, "outcome_reads": 0, "training_steps": 0, "one_shot_consumed": True, "claim_ceiling": "Source-only parent enrichment and sealed top-eight identity; no FARO label, effectiveness, deployment, product, or safety evidence."}
        writer.write_json("result.json", result)
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o1r.r8_clear_top8_selection_manifest.v1", "terminal": PASS_TERMINAL, "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written})
        return result
    except Exception as error:
        try:
            writer.write_json("failure.json", {"schema": "blindassist.taro.o1r.r8_clear_top8_selection_failure.v1", "terminal": FAIL_TERMINAL, "execution_valid": False, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error), "faro_reads": 0, "one_shot_consumed": True})
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
        print(json.dumps({"terminal": FAIL_TERMINAL, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

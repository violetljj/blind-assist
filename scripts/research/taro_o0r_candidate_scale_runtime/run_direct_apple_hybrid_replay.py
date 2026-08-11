#!/usr/bin/env python3
"""Materialize the sealed TARO R4A direct-Apple/baseline hybrid replay."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_full_cohort as r4
from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_hybrid as hybrid
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


R4_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-support-r4-full-cohort"
EVIDENCE_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-r4a"
MAXIMUM_EVIDENCE_BYTES = 64 * 1024 * 1024
EXPECTED_FRAMES = 171
EXPECTED_QUERIES = 1539
EXPECTED_PARENTS = 16
EXPECTED_R4_BINDINGS = {
    "r4_manifest_sha256": "EB87FC6141723D2B44DCB384DF594FE3BFD65436262B07BCF2C0E3809709F760",
    "r4_result_sha256": "F8BDDCB58534D5C6436A40C4509F72E21B5CB7BBFEFC29522DBAFEBE12483C3C",
    "r4_summary_sha256": "26A29C90137A1114CEABFC35F104D85785B04383F80D8853EC33E578694FBD25",
    "r4_query_blob_sha256": "323FFB7F456517C4EAEAED301A18FC96A1B2813526D963230B3EB43B8D58F2A7",
    "r4_query_record_sequence_sha256": "DA64D061734BEB384FA0DC80854775C835C73D5F43D8F830F8810D5FEB347DD0",
}


class HybridReplayError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise HybridReplayError(code, message, **context)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HybridReplayError("R4A_INPUT_INVALID", "JSON input cannot be decoded", path=str(path)) from error


def _verify_r4() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = R4_ROOT / "manifest.json"
    require(manifest_path.is_file(), "R4A_R4_MISSING", "sealed R4 manifest is missing")
    manifest = _load_json(manifest_path)
    require(
        isinstance(manifest, dict)
        and manifest.get("schema") == "blindassist.taro.o0r.direct_apple_support_r4_full_cohort_manifest.v1"
        and manifest.get("one_shot_root_consumed") is True
        and isinstance(manifest.get("files"), dict),
        "R4A_R4_MANIFEST_INVALID",
        "R4 manifest fields drift",
    )
    files = manifest["files"]
    expected_paths = set(files) | {"manifest.json"}
    actual_paths = {path.relative_to(R4_ROOT).as_posix() for path in R4_ROOT.rglob("*") if path.is_file()}
    require(actual_paths == expected_paths, "R4A_R4_MANIFEST_COVERAGE", "R4 manifest does not cover its root")
    verified_bytes = 0
    for relative, receipt in files.items():
        require(isinstance(receipt, dict) and receipt.get("path") == relative, "R4A_R4_MANIFEST_INVALID", "R4 file receipt path drift")
        path = R4_ROOT / relative
        payload = path.read_bytes()
        require(
            receipt.get("bytes") == len(payload)
            and receipt.get("sha256") == hashlib.sha256(payload).hexdigest().upper(),
            "R4A_R4_FILE_DRIFT",
            "R4 file differs from manifest",
            path=relative,
        )
        verified_bytes += len(payload)
    require(
        len(files) == manifest.get("file_count_before_manifest")
        and verified_bytes == manifest.get("bytes_before_manifest"),
        "R4A_R4_MANIFEST_INVALID",
        "R4 manifest totals drift",
    )
    result = _load_json(R4_ROOT / "result.json")
    summary = _load_json(R4_ROOT / "summary.json")
    require(
        result.get("terminal") == "TARO_O0R_DIRECT_APPLE_SUPPORT_R4_FULL_COHORT_COMPLETE"
        and result.get("execution_valid") is True
        and result.get("summary_sha256") == summary.get("content_sha256"),
        "R4A_R4_TERMINAL_INVALID",
        "R4 is not the exact sealed COMPLETE result",
    )
    query_path = R4_ROOT / "full-cohort-query-records.json.gz"
    try:
        with gzip.open(query_path, "rt", encoding="utf-8") as stream:
            raw_rows = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise HybridReplayError("R4A_R4_QUERY_INVALID", "R4 query records cannot be decoded") from error
    require(isinstance(raw_rows, list), "R4A_R4_QUERY_INVALID", "R4 query payload must be a list")
    rows = [r4.validate_full_cohort_query_record(row) for row in raw_rows]
    require(
        len(rows) == EXPECTED_QUERIES
        and len({(row["physical_frame_id"], row["query_id"]) for row in rows}) == EXPECTED_QUERIES
        and len({row["physical_frame_id"] for row in rows}) == EXPECTED_FRAMES
        and len({row["parent_id"] for row in rows}) == EXPECTED_PARENTS,
        "R4A_R4_QUERY_COHORT_DRIFT",
        "R4 query cohort cardinality drift",
    )
    bindings = {
        "r4_manifest_sha256": materializer.sha256_file(manifest_path),
        "r4_result_sha256": materializer.sha256_file(R4_ROOT / "result.json"),
        "r4_summary_sha256": summary["content_sha256"],
        "r4_query_blob_sha256": materializer.sha256_file(query_path),
        "r4_query_record_sequence_sha256": materializer.canonical_sha256([row["content_sha256"] for row in rows]),
    }
    require(
        bindings == EXPECTED_R4_BINDINGS,
        "R4A_R4_FROZEN_BINDING_DRIFT",
        "R4 root does not match the exact frozen formal R4 execution",
        observed=bindings,
    )
    return rows, bindings


def _write_consumed_failure(writer: FactorEvidenceWriter, error: Exception) -> None:
    if not writer.activated or not writer.root.exists() or "manifest.json" in writer.file_receipts:
        return
    failure = {
        "schema": "blindassist.taro.o0r.direct_apple_hybrid_r4a_failure.v1",
        "terminal": "TARO_O0R_DIRECT_APPLE_HYBRID_R4A_EXECUTION_INVALID",
        "execution_valid": False,
        "error_code": str(getattr(error, "code", type(error).__name__)),
        "message": str(error),
        "one_shot_consumed": True,
    }
    try:
        if "failure.json" not in writer.file_receipts:
            writer.write_json("failure.json", failure)
        files = dict(sorted(writer.file_receipts.items()))
        if "manifest.json" not in writer.file_receipts:
            writer.write_json(
                "manifest.json",
                {
                    "schema": "blindassist.taro.o0r.direct_apple_hybrid_r4a_manifest.v1",
                    "files": files,
                    "file_count_before_manifest": len(files),
                    "bytes_before_manifest": sum(int(item["bytes"]) for item in files.values()),
                    "one_shot_root_consumed": True,
                },
            )
    except Exception:
        pass


def _activate(writer: FactorEvidenceWriter, receipt: Mapping[str, Any]) -> None:
    try:
        writer.activate(receipt)
    except Exception as error:
        _write_consumed_failure(writer, error)
        raise


def _best_effort_terminal_print(payload: Mapping[str, Any]) -> None:
    try:
        print(json.dumps(dict(payload), sort_keys=True), flush=True)
    except Exception:
        pass


def execute() -> dict[str, Any]:
    rows, input_bindings = _verify_r4()
    require(not EVIDENCE_ROOT.exists(), "R4A_ONE_SHOT_ROOT_COLLISION", "R4A evidence root already exists", root=str(EVIDENCE_ROOT))
    writer = FactorEvidenceWriter(EVIDENCE_ROOT, MAXIMUM_EVIDENCE_BYTES)
    _activate(
        writer,
        {
            "schema": "blindassist.taro.o0r.direct_apple_hybrid_r4a_execution_start.v1",
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "analysis_kind": hybrid.ANALYSIS_KIND,
            "claim_ceiling": hybrid.CLAIM_CEILING,
            "policy_id": hybrid.POLICY_ID,
            "input_bindings": input_bindings,
            "code_bindings": {
                "hybrid": {
                    "path": Path(hybrid.__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                    "sha256": materializer.sha256_file(Path(hybrid.__file__).resolve()),
                },
                "runner": {
                    "path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                    "sha256": materializer.sha256_file(Path(__file__).resolve()),
                },
                "r4": {
                    "path": Path(r4.__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                    "sha256": materializer.sha256_file(Path(r4.__file__).resolve()),
                },
            },
            "runtime": {"python": platform.python_version(), "device": "cpu"},
            "training": False,
            "threshold_count": 0,
            "network": False,
            "one_shot_consumed_on_root_creation": True,
        },
    )
    try:
        records = [hybrid.build_hybrid_query_record(row) for row in rows]
        summary = hybrid.summarize_hybrid(records, rows)
        writer.write_json_gzip("hybrid-query-records.json.gz", records)
        writer.write_json("summary.json", summary)
        result = {
            "schema": "blindassist.taro.o0r.direct_apple_hybrid_r4a_result.v1",
            "terminal": "TARO_O0R_DIRECT_APPLE_HYBRID_R4A_COMPLETE",
            "execution_valid": True,
            "scientific_status": "POST_HOC_RETROSPECTIVE_ZERO_PARAMETER_HYBRID_MAP_ONLY",
            "claim_ceiling": hybrid.CLAIM_CEILING,
            "policy_id": hybrid.POLICY_ID,
            "summary_sha256": summary["content_sha256"],
            "physical_frame_count": EXPECTED_FRAMES,
            "query_record_count": EXPECTED_QUERIES,
            "parent_count": EXPECTED_PARENTS,
            "training_steps": 0,
            "threshold_count": 0,
            "network_requests": 0,
            "fresh_confirmation": False,
        }
        writer.write_json("result.json", result)
        files = dict(sorted(writer.file_receipts.items()))
        writer.write_json(
            "manifest.json",
            {
                "schema": "blindassist.taro.o0r.direct_apple_hybrid_r4a_manifest.v1",
                "files": files,
                "file_count_before_manifest": len(files),
                "bytes_before_manifest": sum(int(item["bytes"]) for item in files.values()),
                "one_shot_root_consumed": True,
            },
        )
        _best_effort_terminal_print(
            {
                "terminal": result["terminal"],
                "hybrid_evaluable_queries": summary["hybrid_extraction_evaluable_query_count"],
                "parents_jointly_positive": summary["parents_jointly_positive_height_and_normal"],
                "summary_sha256": summary["content_sha256"],
            }
        )
        return result
    except Exception as error:
        _write_consumed_failure(writer, error)
        raise


def main() -> int:
    execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

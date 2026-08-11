#!/usr/bin/env python3
"""Independently verify the consumed TARO R6 confirmation evidence root."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation as r6
from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation_io as r6io
from scripts.research.taro_o0r_candidate_scale_runtime import r6_factor_split
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-r6-untouched-confirmation-r0"
EXECUTION_LOCK = REPO_ROOT / "docs/research/taro/TARO_O0R_R6_UNTOUCHED_CONFIRMATION_ONE_SHOT_EXECUTION_LOCK_2026-08-11.json"


class R6EvidenceError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R6EvidenceError(code, message, **context)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_gzip(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _semantic_delta(left: Any, right: Any, path: str = "summary") -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        require(set(left) == set(right), "R6_EVIDENCE_SUMMARY_FIELD_DRIFT", "R6 summary fields drift", path=path)
        return max((_semantic_delta(left[key], right[key], f"{path}.{key}") for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), "R6_EVIDENCE_SUMMARY_SEQUENCE_DRIFT", "R6 summary sequence length drift", path=path)
        return max((_semantic_delta(a, b, f"{path}[{index}]") for index, (a, b) in enumerate(zip(left, right))), default=0.0)
    if isinstance(left, bool) or isinstance(right, bool) or left is None or right is None or isinstance(left, str) or isinstance(right, str):
        require(left == right, "R6_EVIDENCE_SUMMARY_VALUE_DRIFT", "R6 summary non-numeric value drift", path=path)
        return 0.0
    if isinstance(left, int) and isinstance(right, int):
        require(left == right, "R6_EVIDENCE_SUMMARY_VALUE_DRIFT", "R6 summary integer value drift", path=path)
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        delta = abs(float(left) - float(right))
        require(math.isfinite(delta) and delta <= 1e-12, "R6_EVIDENCE_SUMMARY_NUMERIC_DRIFT", "R6 summary numeric replay exceeds canonical precision", path=path, delta=delta)
        return delta
    require(left == right, "R6_EVIDENCE_SUMMARY_TYPE_DRIFT", "R6 summary value/type drift", path=path)
    return 0.0


def verify(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    evidence = root.resolve()
    require(evidence.is_dir(), "R6_EVIDENCE_ROOT_MISSING", "R6 evidence root is missing")
    execution_lock = _json(EXECUTION_LOCK)
    inventory_path = Path(execution_lock["roots"]["inventory_path"]).resolve()
    require(
        inventory_path.is_file()
        and inventory_path.stat().st_size == int(execution_lock["inventory_binding"]["bytes"])
        and _sha(inventory_path) == execution_lock["inventory_binding"]["sha256"],
        "R6_EVIDENCE_INVENTORY_BINDING_DRIFT",
        "R6 execution lock no longer binds the exact cohort inventory",
    )
    frames = r6io.load_exact_cohort(inventory_path, Path(execution_lock["roots"]["repo_root"]))
    manifest_path = materializer.safe_join(evidence, "manifest.json")
    manifest = _json(manifest_path)
    files = manifest.get("files")
    require(isinstance(files, dict) and manifest.get("file_count_before_manifest") == len(files) == 725, "R6_EVIDENCE_MANIFEST_COUNT_DRIFT", "R6 manifest file count drift")
    require(manifest.get("bytes_before_manifest") == sum(int(row["bytes"]) for row in files.values()) == 109_980_064, "R6_EVIDENCE_MANIFEST_BYTE_DRIFT", "R6 manifest byte accounting drift")
    require(manifest.get("evidence_root_consumed") is True and manifest.get("terminal") == "TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_CONFIRMATION_PASS", "R6_EVIDENCE_MANIFEST_TERMINAL_DRIFT", "R6 manifest terminal drift")
    for relative, receipt in files.items():
        path = materializer.safe_join(evidence, relative)
        require(isinstance(receipt, dict) and set(receipt) == {"path", "bytes", "sha256"} and receipt["path"] == relative, "R6_EVIDENCE_MANIFEST_ROW_DRIFT", "R6 manifest row fields drift", path=relative)
        require(path.is_file() and path.stat().st_size == int(receipt["bytes"]) and _sha(path) == receipt["sha256"], "R6_EVIDENCE_FILE_DRIFT", "R6 evidence file differs from manifest", path=relative)

    candidate_completion = r6.validate_candidate_completion(_json(materializer.safe_join(evidence, "candidate-completion.json")))
    phase_a = r6.validate_phase_a_completion(_json(materializer.safe_join(evidence, "phase-a-completion.json")))
    require(phase_a["candidate_completion_sha256"] == candidate_completion["content_sha256"], "R6_EVIDENCE_PHASE_A_LINEAGE_DRIFT", "R6 Phase-A/candidate completion lineage drift")

    source_paths = sorted(evidence.glob("phase-a-sources/*/*/*.json"))
    candidate_paths = sorted(evidence.glob("candidates/*/*/*.json"))
    decision_paths = sorted(evidence.glob("phase-a-decisions/*/*/*.json"))
    truth_paths = sorted(evidence.glob("phase-b-truth-bindings/*/*/*.json"))
    query_paths = sorted(evidence.glob("phase-b-query-pairs/*/*/*.json.gz"))
    require([len(source_paths), len(candidate_paths), len(decision_paths), len(truth_paths), len(query_paths)] == [120] * 5, "R6_EVIDENCE_FRAME_ARTIFACT_COUNT_DRIFT", "R6 per-frame artifact counts drift")

    sources = [r6.validate_phase_a_source_receipt(_json(materializer.safe_join(evidence, r6io.source_receipt_relative(frame)))) for frame in frames]
    candidates = [r6.validate_candidate_frame(_json(materializer.safe_join(evidence, r6io.candidate_record_relative(frame)))) for frame in frames]
    decisions = [r6.validate_source_decision(_json(materializer.safe_join(evidence, r6io.source_decision_relative(frame)))) for frame in frames]
    truths = [r6.validate_truth_binding(_json(materializer.safe_join(evidence, r6io.truth_binding_relative(frame)))) for frame in frames]
    source_by_frame = {row["physical_frame_id"]: row for row in sources}
    candidate_by_frame = {row["physical_frame_id"]: row for row in candidates}
    decision_by_frame = {row["physical_frame_id"]: row for row in decisions}
    truth_by_frame = {row["physical_frame_id"]: row for row in truths}
    require(all(len(index) == 120 for index in (source_by_frame, candidate_by_frame, decision_by_frame, truth_by_frame)), "R6_EVIDENCE_FRAME_IDENTITY_DUPLICATE", "R6 per-frame identity is duplicated")
    require(adapter.canonical_sha256([row["content_sha256"] for row in decisions]) == phase_a["decision_hash_sequence_sha256"], "R6_EVIDENCE_DECISION_SEQUENCE_DRIFT", "R6 decision hash sequence drift")

    for candidate in candidates:
        frame_id = candidate["physical_frame_id"]
        source = source_by_frame[frame_id]
        decision = decision_by_frame[frame_id]
        truth = truth_by_frame[frame_id]
        require(candidate["candidate_input"]["phase_a_source_receipt_sha256"] == source["content_sha256"] and decision["candidate_frame_sha256"] == candidate["content_sha256"] and decision["phase_a_source_receipt_sha256"] == source["content_sha256"], "R6_EVIDENCE_PHASE_A_FRAME_LINEAGE_DRIFT", "R6 per-frame Phase-A lineage drift", frame=frame_id)
        require(truth["phase_a_source_receipt_sha256"] == source["content_sha256"] and truth["phase_a_completion_sha256"] == phase_a["content_sha256"], "R6_EVIDENCE_TRUTH_LINEAGE_DRIFT", "R6 per-frame truth lineage drift", frame=frame_id)
        blob = candidate["native_depth_blob"]
        blob_path = materializer.safe_join(evidence, blob["path"])
        native = depthart_runner.decode_npy_gzip_bytes(blob_path.read_bytes())
        require(adapter.canonical_sha256(native) == blob["array_sha256"] == candidate["inference_receipt"]["native_depth_sha256"], "R6_EVIDENCE_CANDIDATE_ARRAY_DRIFT", "R6 candidate array lineage drift", frame=frame_id)

    records: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for frame in frames:
        path = materializer.safe_join(evidence, r6io.query_pairs_relative(frame))
        rows = _json_gzip(path)
        require(isinstance(rows, list) and len(rows) == 9, "R6_EVIDENCE_QUERY_FRAME_COUNT_DRIFT", "R6 query frame does not contain nine slots", path=str(path))
        for row in rows:
            require(isinstance(row, dict) and set(row) == {"truth_scoring_record", "factor_components", "composite_query"}, "R6_EVIDENCE_QUERY_ROW_DRIFT", "R6 query evidence row fields drift")
            scoring = r6._validate_seal(row["truth_scoring_record"], r6.TRUTH_SCORING_SCHEMA)
            components = r6_factor_split.validate_factor_components(row["factor_components"])
            composite = r6_factor_split.validate_composite_query(row["composite_query"], factor_components=components)
            frame_id = composite["physical_frame_id"]
            require(scoring["content_sha256"] == components["truth_scoring_record_sha256"] and scoring["physical_frame_id"] == frame_id and scoring["query_id"] == composite["query_id"], "R6_EVIDENCE_QUERY_TRUTH_LINEAGE_DRIFT", "R6 query truth/component lineage drift")
            require(components["source_frame_receipt_sha256"] == source_by_frame[frame_id]["content_sha256"] and components["candidate_frame_record_sha256"] == candidate_by_frame[frame_id]["content_sha256"] and components["r6_phase_a_policy_seal_sha256"] == decision_by_frame[frame_id]["content_sha256"] and scoring["truth_binding_sha256"] == truth_by_frame[frame_id]["content_sha256"], "R6_EVIDENCE_QUERY_FRAME_LINEAGE_DRIFT", "R6 query does not bind its frame evidence")
            records.append((scoring, components, composite))
    require(len(records) == r6.EXPECTED_QUERY_COUNT, "R6_EVIDENCE_QUERY_COUNT_DRIFT", "R6 query record count drift")
    recomputed = r6.summarize(records)
    stored_summary = _json(materializer.safe_join(evidence, "summary.json"))
    stored_payload = dict(stored_summary)
    stored_seal = stored_payload.pop("content_sha256", None)
    require(
        isinstance(stored_seal, str) and adapter.canonical_sha256(stored_payload) == stored_seal,
        "R6_EVIDENCE_SUMMARY_SEAL_DRIFT",
        "R6 stored summary seal drift",
    )
    recomputed_payload = dict(recomputed)
    recomputed_payload.pop("content_sha256")
    maximum_summary_replay_delta = _semantic_delta(recomputed_payload, stored_payload)
    result = _json(materializer.safe_join(evidence, "result.json"))
    require(result.get("execution_valid") is True and result.get("passed") is True and result.get("terminal") == stored_summary["terminal"] and result.get("summary_sha256") == stored_summary["content_sha256"], "R6_EVIDENCE_RESULT_DRIFT", "R6 result/summary terminal drift")
    require(result.get("candidate_inference_count") == result.get("phase_b_faro_frame_count") == 120 and result.get("query_record_count") == 1080 and result.get("training_steps") == result.get("network_requests") == 0, "R6_EVIDENCE_RESULT_COUNT_DRIFT", "R6 result counts/side effects drift")
    require(EXECUTION_LOCK.is_file() and _sha(EXECUTION_LOCK) == result["execution_lock_sha256"], "R6_EVIDENCE_EXECUTION_LOCK_DRIFT", "R6 result execution-lock binding drift")
    return {
        "schema": "blindassist.taro.o0r.r6_untouched_confirmation_evidence_verification.v1",
        "passed": True,
        "terminal": "TARO_O0R_R6_UNTOUCHED_CONFIRMATION_EVIDENCE_VERIFIED",
        "scientific_terminal": result["terminal"],
        "manifest_file_count": len(files),
        "manifest_bytes": manifest["bytes_before_manifest"],
        "parent_count": stored_summary["parent_count"],
        "physical_frame_count": stored_summary["physical_frame_count"],
        "query_record_count": stored_summary["query_record_count"],
        "all_gate_count": len(stored_summary["gates"]),
        "all_gates_passed": all(row["passed"] for row in stored_summary["gates"]),
        "summary_exact_seal_replay": recomputed["content_sha256"] == stored_summary["content_sha256"],
        "maximum_summary_replay_delta": maximum_summary_replay_delta,
        "result_file_sha256": _sha(materializer.safe_join(evidence, "result.json")),
        "summary_file_sha256": _sha(materializer.safe_join(evidence, "summary.json")),
        "manifest_file_sha256": _sha(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = verify(args.root)
    except Exception as error:
        print(json.dumps({"passed": False, "error_code": str(getattr(error, "code", type(error).__name__)), "message": str(error), "context": getattr(error, "context", {})}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

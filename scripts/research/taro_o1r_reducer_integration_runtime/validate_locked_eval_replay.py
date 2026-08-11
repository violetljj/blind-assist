#!/usr/bin/env python3
"""Replay-validate every sealed file in the consumed TARO O1R eval run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_reducer_integration_runtime import reducer_integration as runtime
from scripts.research.taro_o1r_reducer_integration_runtime.run_locked_eval_replay import (
    MANIFEST_SCHEMA,
    RESULT_SCHEMA,
    SUMMARY_SCHEMA,
    TERMINAL,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o1r-r6-reducer-integration-r0"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_gzip(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _seal_valid(value: Any, schema: str) -> bool:
    if not isinstance(value, dict) or value.get("schema") != schema:
        return False
    payload = dict(value)
    observed = payload.pop("content_sha256", None)
    return observed == adapter.canonical_sha256(payload)


def validate(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    root = root.resolve()
    try:
        manifest = _load_json(root / "manifest.json")
        result = _load_json(root / "result.json")
        summary = _load_json(root / "summary.json")
    except Exception as error:
        return {"passed": False, "errors": [f"ROOT_READ_ERROR:{type(error).__name__}"]}
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("terminal") != TERMINAL:
        errors.append("MANIFEST_IDENTITY_DRIFT")
    if not _seal_valid(result, RESULT_SCHEMA) or result.get("terminal") != TERMINAL:
        errors.append("RESULT_SEAL_OR_TERMINAL_DRIFT")
    if not _seal_valid(summary, SUMMARY_SCHEMA) or summary.get("terminal") != TERMINAL:
        errors.append("SUMMARY_SEAL_OR_TERMINAL_DRIFT")
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("file_count_before_manifest") != len(files):
        errors.append("MANIFEST_FILE_COUNT_DRIFT")
        files = {}
    observed_paths = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "manifest.json")
    if observed_paths != sorted(files):
        errors.append("MANIFEST_PATH_SET_DRIFT")
    for relative, receipt in files.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            errors.append(f"MANIFEST_FILE_MISSING:{relative}")
            continue
        payload = path.read_bytes()
        if len(payload) != receipt.get("bytes") or hashlib.sha256(payload).hexdigest().upper() != receipt.get("sha256"):
            errors.append(f"MANIFEST_FILE_BINDING_DRIFT:{relative}")
    frame_paths = sorted(root.glob("frame-results/*/*/*.json.gz"))
    if len(frame_paths) != 239:
        errors.append("FRAME_RESULT_COUNT_DRIFT")
    states: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    frame_ids: set[str] = set()
    parent_ids: set[str] = set()
    for path in frame_paths:
        try:
            frame = runtime.validate_reducer_bundle(_load_json_gzip(path))
        except Exception as error:
            errors.append(f"FRAME_RESULT_INVALID:{path.relative_to(root).as_posix()}:{getattr(error, 'code', type(error).__name__)}")
            continue
        frame_ids.add(frame["physical_frame_id"])
        parent_ids.add(path.parts[-3])
        states.update(frame["state_counts"])
        for query in frame["query_results"]:
            reasons.update(query["reason_codes"])
    expected_states = {state: int(states[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")}
    if len(frame_ids) != 239 or len(parent_ids) != 16:
        errors.append("FRAME_OR_PARENT_IDENTITY_DRIFT")
    if summary.get("state_counts") != expected_states or summary.get("reason_counts") != dict(sorted(reasons.items())):
        errors.append("SUMMARY_AGGREGATION_DRIFT")
    if result.get("state_counts") != expected_states or result.get("summary_sha256") != summary.get("content_sha256"):
        errors.append("RESULT_AGGREGATION_DRIFT")
    if expected_states != {"CLEAR_OBSERVED": 0, "OCCUPIED_OBSERVED": 0, "UNKNOWN": 2151}:
        errors.append("TERMINAL_STATE_COUNTS_DRIFT")
    if result.get("faro_payload_reads") != 0 or result.get("source_payload_reads") != {"confidence": 239} or result.get("training_steps") != 0 or result.get("network_requests") != 0:
        errors.append("EXECUTION_FIREWALL_RECEIPT_DRIFT")
    return {
        "schema": "blindassist.taro.o1r.r6_reducer_integration_eval_replay_validation.v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "frame_count": len(frame_ids),
        "parent_count": len(parent_ids),
        "query_count": sum(expected_states.values()),
        "state_counts": expected_states,
        "terminal": "TARO_O1R_R6_REDUCER_INTEGRATION_EVIDENCE_VALID" if not errors else "TARO_O1R_R6_REDUCER_INTEGRATION_EVIDENCE_INVALID",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = validate(args.root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

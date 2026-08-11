from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary


DEFAULT_ROOT = Path("artifacts.local/evidence/taro/o1r-r7-positive-occupancy-clear-coverage-fit-canary-r0")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    _require(isinstance(value, dict), f"record is not an object: {path}")
    return value


def _validate_seal(value: dict[str, Any], schema: str) -> dict[str, Any]:
    record = dict(value)
    observed = record.pop("content_sha256", None)
    _require(record.get("schema") == schema and observed == adapter.canonical_sha256(record), f"seal mismatch: {schema}")
    record["content_sha256"] = observed
    return record


def validate(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    evidence_root = root.resolve()
    _require(evidence_root.is_dir(), "R7 evidence root is missing")
    manifest_path = evidence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == "blindassist.taro.o1r.r7_fit_lopo_canary_manifest.v1", "R7 manifest schema drift")
    files = manifest.get("files")
    _require(isinstance(files, dict) and len(files) == manifest.get("file_count_before_manifest") == 425, "R7 manifest cardinality drift")
    byte_total = 0
    for relative, receipt in files.items():
        path = (evidence_root / relative).resolve()
        _require(evidence_root in path.parents and path.is_file(), f"manifest file missing: {relative}")
        raw = path.read_bytes()
        _require(receipt == {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest().upper()}, f"manifest receipt mismatch: {relative}")
        byte_total += len(raw)
    _require(byte_total == manifest.get("bytes_before_manifest") == 706732, "R7 manifest byte total drift")
    actual_files = [path for path in evidence_root.rglob("*") if path.is_file()]
    _require(len(actual_files) == 426 and {path.relative_to(evidence_root).as_posix() for path in actual_files} == set(files) | {"manifest.json"}, "R7 evidence file set drift")

    source_paths = sorted((evidence_root / "source-features").rglob("*.json.gz"))
    label_paths = sorted((evidence_root / "label-frames").rglob("*.json.gz"))
    _require(len(source_paths) == len(label_paths) == 211, "R7 source/label frame count drift")
    sources = [r7_canary.validate_source_frame_record(_read_gzip_json(path)) for path in source_paths]
    roster_index = {identity: index for index, identity in enumerate(adapter.ADAPTER_FIT_ROSTER)}
    sources.sort(key=lambda row: (roster_index[(row["parent_id"], row["video_id"])], adapter.decimal_timestamp_ns(row["timestamp_token"]), row["timestamp_token"]))
    source_by_identity = {(row["parent_id"], row["video_id"], row["timestamp_token"]): row for row in sources}
    _require(len(source_by_identity) == 211 and {row["parent_id"] for row in sources} == {parent for parent, _ in adapter.ADAPTER_FIT_ROSTER}, "R7 source cohort identity drift")
    labels = []
    for path in label_paths:
        relative = path.relative_to(evidence_root / "label-frames")
        identity = (relative.parts[0], relative.parts[1], relative.name[:-8])
        _require(identity in source_by_identity, f"label has no source frame: {identity}")
        labels.append(r7_canary.validate_label_frame_record(_read_gzip_json(path), source_by_identity[identity]))

    completion = _validate_seal(json.loads((evidence_root / "source-phase-completion.json").read_text(encoding="utf-8")), "blindassist.taro.o1r.r7_source_phase_completion.v1")
    _require(completion["parent_count"] == 8 and completion["frame_count"] == 211 and completion["query_count"] == 1899, "R7 source completion cohort drift")
    _require(completion["source_payload_reads"] == {"confidence": 211, "lowres_depth": 211} and completion["faro_reads"] == completion["label_reads"] == 0, "R7 source completion firewall drift")
    _require(completion["source_frame_hash_sequence_sha256"] == adapter.canonical_sha256([row["content_sha256"] for row in sources]), "R7 source hash sequence drift")

    result = _validate_seal(json.loads((evidence_root / "result.json").read_text(encoding="utf-8")), r7_canary.CANARY_RESULT_SCHEMA)
    replay = r7_canary.run_lopo_canary(sources, labels)
    expected = dict(replay)
    expected.pop("content_sha256")
    observed = dict(result)
    observed.pop("content_sha256")
    receipts = observed.pop("execution_receipts")
    _require(observed == expected, "R7 LOPO result does not replay from sealed source/label records")
    _require(receipts["source_payload_reads"] == {"confidence": 211, "lowres_depth": 211} and receipts["label_payload_reads"] == {"highres_depth": 211}, "R7 execution read receipt drift")
    _require(receipts["source_phase_completion_sha256"] == completion["content_sha256"] and receipts["source_phase_reloaded_before_label_join"] is True and receipts["source_phase_reselection_after_labels"] is False, "R7 phase join receipt drift")
    _require(result["terminal"] == manifest["terminal"] == "TARO_O1R_R7_FIT_LOPO_CANARY_PASS" and result["passed"] is True and result["promotion_authorized"] is False, "R7 terminal/authority drift")
    return {
        "schema": "blindassist.taro.o1r.r7_fit_lopo_canary_evidence_validation.v1",
        "passed": True,
        "terminal": "TARO_O1R_R7_FIT_LOPO_CANARY_EVIDENCE_VALID",
        "parent_count": 8,
        "frame_count": 211,
        "query_count": 1899,
        "result_terminal": result["terminal"],
        "state_counts": result["r7_state_counts"],
        "label_state_counts": result["label_state_counts"],
        "false_clear_count": result["false_clear_count"],
        "promotion_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = validate(args.root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

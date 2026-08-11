#!/usr/bin/env python3
"""Replay consumed R5 formation evidence through the frozen R6 compositor."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import r6_factor_split as r6
from scripts.research.taro_o0r_candidate_scale_runtime.run_factor_split_canary import _verify_manifest
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


RESULT_SCHEMA = "blindassist.taro.o0r.r6_factor_split_implementation_replay_result.v1"
TERMINAL = "TARO_O0R_R6_FACTOR_SPLIT_IMPLEMENTATION_REPLAY_PASS"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "phase-b-query-records").rglob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        require(isinstance(payload, list), f"R5 query payload is not a list: {path}")
        records.extend(r5.validate_query_record(row) for row in payload)
    require(bool(records), "R5 formation replay contains no query records")
    return records


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(adapter.canonical_json_bytes(payload))
    temporary.replace(path)


def execute(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    require(root.is_dir() and not output.exists(), "input root must exist and output must be absent")
    verified = _verify_manifest(root)
    records = _load_records(root)
    expected_counts = dict(zip((parent for parent, _ in r5.R5_ROSTER), r5.EXPECTED_PARENT_FRAME_COUNTS))

    pairs = []
    input_hashes = []
    component_hashes = []
    composite_hashes = []
    record_keys = []
    for record in records:
        components = r6.factor_components_from_r5_query_record(record)
        composite = r6.build_composite_query(components)
        pairs.append((components, composite))
        input_hashes.append(record["content_sha256"])
        component_hashes.append(components["content_sha256"])
        composite_hashes.append(composite["content_sha256"])
        record_keys.append([record["parent_id"], record["physical_frame_id"], record["query_id"], record["grid_index"]])

    require(len(record_keys) == len({tuple(key) for key in record_keys}), "R5 formation replay contains duplicate query keys")
    summary = r6.summarize_factor_split_pairs(
        pairs,
        analysis_role=r6.FORMATION_REPLAY,
        expected_parent_frame_counts=expected_counts,
    )
    require(summary["all_gate_landscape_would_pass"] is True and summary["pass_fail_terminal_absent"] is True, "R6 implementation replay landscape differs")
    require(summary["query_record_count"] == len(records) == 1899, "R6 implementation replay query count differs")

    module_path = Path(r6.__file__).resolve()
    protocol_path = Path("docs/research/taro/TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_PARENT_CONFIRMATION_PROTOCOL_LOCK_2026-08-11.json").resolve()
    require(materializer.sha256_file(protocol_path) == r6.PROTOCOL_LOCK_SHA256, "R6 protocol lock file hash differs from implementation")
    payload = {
        "schema": RESULT_SCHEMA,
        "terminal": TERMINAL,
        "execution_valid": True,
        "analysis_role": r6.FORMATION_REPLAY,
        "confirmation_eligible": False,
        "promotion_allowed": False,
        "input_r5_r3_manifest_sha256": verified["manifest_sha256"],
        "input_r5_r3_result_sha256": verified["result_sha256"],
        "input_r5_r3_terminal": verified["result"]["terminal"],
        "protocol_lock_sha256": r6.PROTOCOL_LOCK_SHA256,
        "implementation_module_sha256": materializer.sha256_file(module_path),
        "input_query_key_sequence_sha256": adapter.canonical_sha256(record_keys),
        "input_query_record_hash_sequence_sha256": adapter.canonical_sha256(input_hashes),
        "factor_component_hash_sequence_sha256": adapter.canonical_sha256(component_hashes),
        "composite_query_hash_sequence_sha256": adapter.canonical_sha256(composite_hashes),
        "factor_component_count": len(component_hashes),
        "composite_query_count": len(composite_hashes),
        "summary": summary,
        "training_steps": 0,
        "model_inference_calls": 0,
        "network_requests": 0,
        "claim_ceiling": "IMPLEMENTATION_REPLAY_ON_CONSUMED_FORMATION_EVIDENCE_ONLY",
        "requires_untouched_confirmation": True,
    }
    payload["content_sha256"] = adapter.canonical_sha256(payload)
    output.mkdir(parents=True, exist_ok=False)
    _write_atomic(output / "result.json", payload)
    manifest = {
        "schema": "blindassist.taro.o0r.r6_factor_split_implementation_replay_manifest.v1",
        "files": {
            "result.json": {
                "bytes": (output / "result.json").stat().st_size,
                "sha256": materializer.sha256_file(output / "result.json"),
            }
        },
    }
    manifest["content_sha256"] = adapter.canonical_sha256(manifest)
    _write_atomic(output / "manifest.json", manifest)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r5-r3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = execute(args.r5_r3_root, args.output)
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "composite_query_count": result["composite_query_count"],
                "all_gate_landscape_would_pass": result["summary"]["all_gate_landscape_would_pass"],
                "content_sha256": result["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

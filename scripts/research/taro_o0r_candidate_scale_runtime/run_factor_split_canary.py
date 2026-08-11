#!/usr/bin/env python3
"""Materialize the post-hoc R6 factor-split landscape from sealed R5 R3 records."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import factor_split_canary
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _verify_manifest(root: Path) -> dict[str, Any]:
    manifest_path = materializer.safe_join(root, "manifest.json")
    result_path = materializer.safe_join(root, "result.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(result.get("execution_valid") is True and result.get("terminal") == "TARO_O0R_DIRECT_APPLE_HYBRID_R5_TASK_METRIC_CONFIRMATION_FAIL", "R5 R3 diagnostic terminal differs")
    files = manifest.get("files")
    require(isinstance(files, dict) and len(files) == manifest.get("file_count_before_manifest"), "R5 R3 manifest count differs")
    for relative, receipt in files.items():
        path = materializer.safe_join(root, relative)
        require(path.is_file() and path.stat().st_size == receipt["bytes"] and materializer.sha256_file(path) == receipt["sha256"], f"R5 R3 file differs: {relative}")
    return {"manifest": manifest, "result": result, "manifest_sha256": materializer.sha256_file(manifest_path), "result_sha256": materializer.sha256_file(result_path)}


def execute(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    require(root.is_dir() and not output.exists(), "input root must exist and output must be absent")
    verified = _verify_manifest(root)
    records = []
    for path in sorted((root / "phase-b-query-records").rglob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            records.extend(json.load(stream))
    canary = factor_split_canary.evaluate_factor_split_landscape(records)
    payload = {
        "schema": "blindassist.taro.o0r.r6_factor_split_posthoc_canary_result.v1",
        "input_r5_r3_manifest_sha256": verified["manifest_sha256"],
        "input_r5_r3_result_sha256": verified["result_sha256"],
        "input_r5_r3_terminal": verified["result"]["terminal"],
        "canary": canary,
        "training_steps": 0,
        "network_requests": 0,
        "pass_fail_terminal_absent": True,
    }
    payload["content_sha256"] = adapter.canonical_sha256(payload)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "result.json"
    temporary = output / "result.json.tmp"
    temporary.write_bytes(adapter.canonical_json_bytes(payload))
    temporary.replace(target)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r5-r3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = execute(args.r5_r3_root, args.output)
    print(json.dumps({"post_hoc": True, "all_gate_landscape_would_pass": result["canary"]["all_gate_landscape_would_pass"], "content_sha256": result["content_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

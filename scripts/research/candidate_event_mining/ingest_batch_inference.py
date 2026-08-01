#!/usr/bin/env python3
"""Validate and materialize adapter-produced canonical frame traces.

This command is the stable hand-off after model-specific batch inference.  It
does not run a model itself; YOLO/segmentation/depth/HFTF adapters write the
canonical frame JSONL and this command binds it to the source index and run
identity before candidate mining.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.research.candidate_event_mining.pipeline import (
    ContractError,
    load_contract,
    normalize_frames,
    read_json,
    read_jsonl,
    refuse_overwrite,
    sha256_file,
    validate_project_index,
    write_json,
    write_jsonl,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    contract, contract_meta = load_contract(args.contract.resolve())
    project_index = validate_project_index(read_json(args.project_index.resolve()))
    if not project_index["sources"]:
        raise ContractError("project index has no registered sources")
    allowed = {
        (source["source_id"], source["session_id"])
        for source in project_index["sources"]
    }
    raw_rows = []
    for input_path in args.input:
        raw_rows.extend(read_jsonl(input_path.resolve()))
    rows = normalize_frames(raw_rows)
    observed = {(row["source_id"], row["session_id"]) for row in rows}
    unknown = sorted(observed - allowed)
    if unknown:
        raise ContractError(f"frame trace contains unregistered source/session: {unknown}")

    refuse_overwrite(args.output.resolve())
    manifest_path = args.output.resolve().with_name(args.output.stem + ".manifest.json")
    refuse_overwrite(manifest_path)
    output_rows = [
        {
            **row,
            "batch_run_id": args.run_id,
            "inference_adapter": args.adapter_id,
        }
        for row in rows
    ]
    write_jsonl(args.output.resolve(), output_rows)
    manifest = {
        "schema": "blindassist_candidate_event_mining_batch_trace_manifest_v1",
        "run_id": args.run_id,
        "inference_adapter": args.adapter_id,
        "contract": contract_meta,
        "project_index": {
            "path": str(args.project_index.resolve()),
            "sha256": sha256_file(args.project_index.resolve()),
        },
        "input_paths": [str(path.resolve()) for path in args.input],
        "input_sha256": [sha256_file(path.resolve()) for path in args.input],
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
        "frame_count": len(rows),
        "source_session_count": len(observed),
        "model_bundle_sha256": args.model_bundle_sha256,
        "execution_boundary": "adapter_trace_ingest_only; no event truth or production authority",
        "authorization": {
            "event_truth": False,
            "training": False,
            "confirmation": False,
            "production": False,
            "default_app": False,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--project-index", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--model-bundle-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        manifest = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "run_id": manifest["run_id"], "frame_count": manifest["frame_count"], "output": manifest["output_path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Extract, deduplicate and cluster discovery candidates from a frame trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.research.candidate_event_mining.pipeline import (
    ContractError,
    build_candidate_report,
    load_contract,
    normalize_frames,
    read_json,
    read_jsonl,
    refuse_overwrite,
    sha256_file,
    validate_project_index,
    write_json,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    contract, contract_meta = load_contract(args.contract.resolve())
    project_index = validate_project_index(read_json(args.project_index.resolve()))
    raw_rows = read_jsonl(args.input_trace.resolve())
    rows = normalize_frames(raw_rows)
    allowed = {
        (source["source_id"], source["session_id"])
        for source in project_index["sources"]
    }
    if not allowed:
        raise ContractError("project index has no registered sources")
    observed = {(row["source_id"], row["session_id"]) for row in rows}
    unknown = sorted(observed - allowed)
    if unknown:
        raise ContractError(f"frame trace contains unregistered source/session: {unknown}")
    report = build_candidate_report(
        rows,
        contract_meta,
        {"path": str(args.project_index.resolve()), "sha256": sha256_file(args.project_index.resolve())},
        {"path": str(args.input_trace.resolve()), "sha256": sha256_file(args.input_trace.resolve())},
        args.run_id,
        contract,
    )
    output = args.output.resolve()
    refuse_overwrite(output)
    refuse_overwrite(Path(str(output) + ".sha256"))
    write_json(output, report)
    Path(str(output) + ".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--project-index", type=Path, required=True)
    parser.add_argument("--input-trace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

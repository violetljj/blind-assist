#!/usr/bin/env python3
"""Build a candidate-blind review bundle for the Luna discovery pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.research.candidate_event_mining.pipeline import (
    CANDIDATE_REPORT_SCHEMA,
    ContractError,
    load_contract,
    make_review_bundle,
    read_json,
    sha256_file,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    contract, contract_meta = load_contract(args.contract.resolve())
    report = read_json(args.candidate_report.resolve())
    if report.get("schema") != CANDIDATE_REPORT_SCHEMA:
        raise ContractError("unexpected candidate report schema")
    if report.get("contract", {}).get("sha256") != contract_meta["sha256"]:
        raise ContractError("candidate report was not produced under the supplied contract")
    manifest = make_review_bundle(
        report,
        args.candidate_report.resolve(),
        contract_meta,
        contract,
        args.output.resolve(),
    )
    manifest_path = args.output.resolve() / "review_bundle_manifest.json"
    (args.output.resolve() / "review_bundle_manifest.json.sha256").write_text(
        sha256_file(manifest_path) + "\n", encoding="ascii"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        manifest = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "candidate_count": len(manifest["candidate_ids"]), "output": manifest["review_inputs_path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

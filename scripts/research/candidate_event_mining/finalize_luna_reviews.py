#!/usr/bin/env python3
"""Validate isolated Luna receipts and publish a discovery candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.research.candidate_event_mining.pipeline import (
    BUNDLE_SCHEMA,
    CANDIDATE_REPORT_SCHEMA,
    ContractError,
    finalize_candidate_pool,
    load_contract,
    read_json,
    read_jsonl,
    refuse_overwrite,
    sha256_file,
    write_json,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    contract, _contract_meta = load_contract(args.contract.resolve())
    candidate_report = read_json(args.candidate_report.resolve())
    if candidate_report.get("schema") != CANDIDATE_REPORT_SCHEMA:
        raise ContractError("unexpected candidate report schema")
    bundle_path = args.review_bundle.resolve() / "review_bundle_manifest.json"
    bundle_manifest = read_json(bundle_path)
    if bundle_manifest.get("schema") != BUNDLE_SCHEMA:
        raise ContractError("unexpected review bundle schema")
    if bundle_manifest.get("candidate_report_sha256") != sha256_file(args.candidate_report.resolve()):
        raise ContractError("review bundle is not bound to the supplied candidate report")
    reviews = read_jsonl(args.reviews.resolve())
    pool = finalize_candidate_pool(
        candidate_report,
        args.candidate_report.resolve(),
        bundle_manifest,
        bundle_path,
        reviews,
        args.reviews.resolve(),
        contract,
    )
    output = args.output.resolve()
    refuse_overwrite(output)
    refuse_overwrite(Path(str(output) + ".sha256"))
    write_json(output, pool)
    Path(str(output) + ".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    return pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--review-bundle", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        pool = run(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **pool["summary"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

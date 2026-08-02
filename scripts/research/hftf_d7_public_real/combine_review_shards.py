#!/usr/bin/env python3
"""Combine immutable per-role review shards into the ingest file.

Review agents may write disjoint ``completed_review_part*.jsonl`` files.  This
utility validates the union against the immutable batch manifest and writes
the canonical ``completed_review.jsonl`` consumed by the append-only ingest
step.  It never changes a review row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    batch_root = root / "reviews" / "input_bundles" / args.batch_id
    bundle = load_json(batch_root / "bundle_manifest.json")
    if not isinstance(bundle, dict) or bundle.get("status") != "READY_FOR_ISOLATED_REVIEW":
        raise ContractError("review bundle is not ready for isolated review")
    expected_ids = {str(value) for value in bundle.get("candidate_ids", [])}
    if not expected_ids:
        raise ContractError("review bundle has no candidate IDs")
    roles = [value.strip() for value in args.roles.split(",") if value.strip()]
    if not roles or len(set(roles)) != len(roles):
        raise ContractError("roles must be non-empty and unique")
    output_hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for role in roles:
        role_root = batch_root / role
        shard_paths = sorted(role_root.glob("completed_review_part*.jsonl"))
        if not shard_paths:
            if (role_root / "completed_review.jsonl").is_file():
                continue
            raise ContractError(f"no review shards found for {role}")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for path in shard_paths:
            for row in load_jsonl(path):
                candidate_id = str(row.get("candidate_id") or "")
                if not candidate_id or candidate_id in seen:
                    raise ContractError(f"duplicate or missing candidate_id in {role}: {candidate_id}")
                seen.add(candidate_id)
                rows.append(row)
        if seen != expected_ids:
            raise ContractError(f"{role} shard union mismatch: expected {len(expected_ids)}, got {len(seen)}")
        output_path = role_root / "completed_review.jsonl"
        if output_path.exists():
            raise ContractError(f"refusing to overwrite existing canonical review output: {output_path}")
        rows.sort(key=lambda row: str(row.get("candidate_id")))
        write_jsonl(output_path, rows)
        output_hashes[role] = sha256_file(output_path)
        row_counts[role] = len(rows)
    receipt = {
        "schema": "hftf_d7_public_real_review_shard_combine_receipt_v1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": utc_now(),
        "status": "REVIEW_SHARDS_COMBINED",
        "candidate_count": len(expected_ids),
        "review_roles": roles,
        "row_counts": row_counts,
        "output_hashes": output_hashes,
        "authority": {"event_truth": False, "training": False, "confirmation": False, "production": False},
    }
    write_json(root / "receipts" / f"review_shard_combine_receipt_{args.batch_id}.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--roles", default="RGB_REVIEWER_A,RGB_REVIEWER_B,RGB_REVIEWER_C")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))

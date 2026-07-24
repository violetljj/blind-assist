#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exploratory_profiles_r2_l1 import atomic_write_json, load_json, sha256_file
from finalize_candidate_replay_r2_a3 import TERMINAL_SCHEMA, load_context
from validate_exploratory_profiles_r2_l1 import validate_json_schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        config, bindings, _, _ = load_context(repo, args.config.resolve())
        receipt_path = (
            args.receipt.resolve()
            if args.receipt
            else repo / config["output_root"] / "terminal-receipt-a3.json"
        )
        receipt = load_json(receipt_path)
        schema = load_json(
            repo / "schemas/ustrf_route_target_l1_candidate_replay_a3.schema.json"
        )
        validate_json_schema(receipt, schema, schema)
        if (
            receipt.get("schema") != TERMINAL_SCHEMA
            or receipt.get("bindings") != bindings
            or receipt["verified_scope"]["authoritative_traces"] != 123
            or receipt["verified_scope"]["authoritative_trace_frames"] != 186687
            or receipt["verified_scope"]["authoritative_trace_resets"] != 45
            or receipt["profiles"]
            != {"generated": False, "count": 0, "authority": False}
            or any(receipt["claim_boundary"].values())
        ):
            raise RuntimeError("finalization_a3_receipt_contract_invalid")
        result = {
            "schema": "blindassist_ustrf_route_target_l1_candidate_replay_final_validation_a3",
            "status": "VALID",
            "terminal_state": receipt["terminal_state"],
            "authoritative_trace_count": 123,
            "verified_trace_frames": 186687,
            "verified_trace_resets": 45,
            "minimum_available_memory_bytes": 4 * 1024**3,
            "profile_authority": False,
            "candidate_comparison_authority": False,
            "terminal_receipt_sha256": sha256_file(receipt_path),
        }
        output = repo / config["output_root"] / "validation-receipt-a3.json"
        atomic_write_json(output, result)
        print(json.dumps(result))
        return 0
    except Exception as error:
        print(json.dumps({"status": "INVALID", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

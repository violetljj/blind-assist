#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import candidate_replay_r2 as parent
from candidate_replay_r2_continuation_a2 import (
    activate_short_trace_paths,
    collect_a2_receipts,
    load_a2_context,
)
from exploratory_profiles_r2_l1 import (
    atomic_write_json,
    canonical_bytes,
    identity,
    load_json,
    load_route_map,
    replay_candidate_ledger,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        config, base_config, bindings, groups, resets, _ = load_a2_context(
            repo, args.config
        )
        output_root = (repo / config["output_root"]).resolve()
        terminal_path = (
            args.receipt.resolve()
            if args.receipt
            else output_root / "terminal-receipt-a2.json"
        )
        input_pairs, inventory = parent.resolve_input_ledgers(repo, config, groups)
        activate_short_trace_paths()
        traces = collect_a2_receipts(
            repo, config, bindings, groups, input_pairs, output_root
        )
        terminal = load_json(terminal_path)
        if (
            terminal.get("terminal_state") != "CANDIDATE_REPLAY_COMPLETE"
            or terminal.get("bindings") != bindings
            or terminal["verified_input"]["inventory"] != inventory
            or terminal["candidate_execution"]["trace_inventory"] != traces
            or len(traces) != 123
            or sum(row["frame_count"] for row in traces) != 186687
            or sum(row["discontinuity_reset_count"] for row in traces) != 45
            or terminal["profiles"]
            != {"generated": False, "count": 0, "authority": False}
        ):
            raise RuntimeError("continuation_a2_terminal_or_trace_contract_invalid")
        adopted = 0
        for row in traces:
            marker = load_json(repo / row["authoritative_receipt_path"])
            if marker.get("adopted_without_candidate_rerun") is True:
                adopted += 1
        if adopted != config["parent_trace_adoption"][
            "expected_parent_authoritative_trace_count"
        ]:
            raise RuntimeError("continuation_a2_adopted_trace_count_invalid")
        route_map = load_route_map(base_config, repo)
        tracker_config = load_json(
            repo / base_config["input_contract"]["association"]["config_path"]
        )
        reset_next_identities = {
            (
                row["source_id"],
                row["sequence_id"],
                int(row["next_frame_id"]),
                int(row["next_timestamp_ns"]),
            )
            for row in resets
        }
        groups_by_key = {
            (descriptor["source_id"], descriptor["sequence_id"]): rows
            for descriptor, rows in groups
        }
        for row in traces:
            key = (row["source_id"], row["sequence_id"])
            marker = load_json(repo / row["authoritative_receipt_path"])
            trace = load_json(repo / marker["trace_path"])
            ledger = load_json(input_pairs[key][0])
            expected_frames, _ = replay_candidate_ledger(
                row["candidate_id"],
                ledger,
                route_map,
                reset_next_identities,
                tracker_config,
                base_config,
            )
            for frame in expected_frames:
                frame.pop("candidate_consume_timestamp_ns", None)
            if canonical_bytes(expected_frames) != canonical_bytes(trace["frames"]):
                raise RuntimeError(
                    "continuation_a2_independent_state_replay_mismatch:"
                    f"{row['candidate_id']}:{row['source_id']}/{row['sequence_id']}"
                )
            if [identity(frame) for frame in trace["frames"]] != [
                identity(frame) for frame in groups_by_key[key]
            ]:
                raise RuntimeError("continuation_a2_trace_identity_order_mismatch")
        result = {
            "schema": "blindassist_ustrf_route_target_l1_candidate_replay_validation_a2",
            "status": "VALID",
            "terminal_state": terminal["terminal_state"],
            "authoritative_trace_count": len(traces),
            "verified_trace_frames": sum(row["frame_count"] for row in traces),
            "verified_discontinuity_resets": sum(
                row["discontinuity_reset_count"] for row in traces
            ),
            "parent_traces_adopted_without_candidate_rerun": adopted,
            "minimum_available_memory_bytes": 4 * 1024**3,
            "profile_authority": False,
            "candidate_comparison_authority": False,
            "terminal_receipt_sha256": sha256_file(terminal_path),
        }
        atomic_write_json(output_root / "validation-receipt-a2.json", result)
        print(json.dumps(result))
        return 0
    except Exception as error:
        print(json.dumps({"status": "INVALID", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import candidate_replay_r2 as parent
from candidate_replay_r2_continuation_a2 import (
    activate_short_trace_paths,
    adopt_parent_traces,
    bind_a2_marker,
    collect_a2_receipts,
    load_a2_context,
    short_trace_paths,
)
from exploratory_profiles_r2_l1 import atomic_write_json, load_json, load_route_map


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config: dict = {}
    bindings: dict[str, str] = {}
    groups = []
    input_pairs = {}
    inventory = []
    output_root: Path | None = None
    failure_error: Exception | None = None
    try:
        (
            config,
            base_config,
            bindings,
            groups,
            resets,
            _,
        ) = load_a2_context(repo, args.config)
        output_root = (repo / config["output_root"]).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        input_pairs, inventory = parent.resolve_input_ledgers(repo, config, groups)
        adopted = adopt_parent_traces(
            repo, config, bindings, groups, input_pairs, output_root
        )
        activate_short_trace_paths()
        preflight = parent.preflight_receipt(config, bindings, inventory)
        preflight["parent_authoritative_traces_adopted_without_rerun"] = adopted
        preflight["minimum_available_memory_bytes"] = base_config["resource_guards"][
            "minimum_system_available_physical_memory_bytes"
        ]
        atomic_write_json(output_root / "preflight-receipt-a2.json", preflight)
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": "INPUT_PREFLIGHT_PASS",
                        "sequence_ledgers": len(inventory),
                        "frames": sum(row["frame_count"] for row in inventory),
                        "parent_traces_adopted": adopted,
                    }
                )
            )
            return 0
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
        for candidate_id in config["candidate_roster"]:
            for descriptor, rows in groups:
                key = (descriptor["source_id"], descriptor["sequence_id"])
                _, marker = short_trace_paths(output_root, candidate_id, descriptor)
                existing = parent.validate_authoritative_trace(
                    repo,
                    config,
                    bindings,
                    candidate_id,
                    descriptor,
                    rows,
                    input_pairs[key],
                    marker,
                )
                if existing is not None:
                    continue
                parent.materialize_candidate_trace(
                    repo,
                    config,
                    base_config,
                    bindings,
                    candidate_id,
                    descriptor,
                    rows,
                    input_pairs[key],
                    route_map,
                    reset_next_identities,
                    tracker_config,
                    output_root,
                )
                bind_a2_marker(repo, marker, bindings)
        trace_inventory = collect_a2_receipts(
            repo, config, bindings, groups, input_pairs, output_root
        )
        expected = config["expected_scope"]["candidate_ledger_traces"]
        if len(trace_inventory) != expected:
            raise parent.ExecutionAborted(
                f"authoritative_trace_count_mismatch:expected={expected}:"
                f"observed={len(trace_inventory)}"
            )
        receipt = parent.base_terminal_receipt(
            "CANDIDATE_REPLAY_COMPLETE",
            config,
            bindings,
            inventory,
            trace_inventory,
            None,
        )
        terminal = output_root / "terminal-receipt-a2.json"
        atomic_write_json(terminal, receipt)
        print(
            json.dumps(
                {
                    "terminal_state": receipt["terminal_state"],
                    "authoritative_trace_count": len(trace_inventory),
                    "parent_traces_adopted": adopted,
                    "receipt": str(terminal),
                }
            )
        )
        return 0
    except parent.InputBlocked as error:
        state = "FAIL_CLOSED_INPUT_BLOCKED"
        code = 2
        failure_error = error
    except Exception as error:
        state = "FAIL_CLOSED_EXECUTION_ABORTED"
        code = 3
        failure_error = error
    assert failure_error is not None
    if config and output_root is not None:
        traces = (
            collect_a2_receipts(
                repo, config, bindings, groups, input_pairs, output_root
            )
            if groups and input_pairs
            else []
        )
        receipt = parent.base_terminal_receipt(
            state, config, bindings, inventory, traces, str(failure_error)
        )
        receipt["execution_error"] = {
            "type": failure_error.__class__.__name__,
            "message": str(failure_error),
        }
        atomic_write_json(output_root / "terminal-receipt-a2.json", receipt)
    print(json.dumps({"terminal_state": state, "error": str(failure_error)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

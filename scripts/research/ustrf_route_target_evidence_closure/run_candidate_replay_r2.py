#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from candidate_replay_r2 import (
    ExecutionAborted,
    InputBlocked,
    atomic_write_json,
    base_terminal_receipt,
    collect_authoritative_receipts,
    load_replay_config,
    materialize_candidate_trace,
    preflight_receipt,
    resolve_input_ledgers,
    trace_paths,
)
from exploratory_profiles_r2_l1 import load_json, load_route_map


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--maximum-traces", type=int)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config: dict = {}
    bindings: dict[str, str] = {}
    groups = []
    input_pairs = {}
    input_inventory = []
    output_root: Path | None = None
    try:
        (
            config,
            base_config,
            bindings,
            groups,
            resets,
            _,
        ) = load_replay_config(repo, args.config)
        output_root = (repo / config["output_root"]).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        input_pairs, input_inventory = resolve_input_ledgers(repo, config, groups)
        preflight_path = output_root / "preflight-receipt-r2.json"
        atomic_write_json(
            preflight_path, preflight_receipt(config, bindings, input_inventory)
        )
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": "INPUT_PREFLIGHT_PASS",
                        "sequence_ledgers": len(input_inventory),
                        "frames": sum(row["frame_count"] for row in input_inventory),
                        "receipt": str(preflight_path),
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
        completed_this_run = 0
        for candidate_id in config["candidate_roster"]:
            for descriptor, rows in groups:
                _, authoritative_path = trace_paths(
                    output_root, candidate_id, descriptor
                )
                existed_before = authoritative_path.is_file()
                if (
                    args.maximum_traces is not None
                    and completed_this_run >= args.maximum_traces
                ):
                    trace_inventory = collect_authoritative_receipts(
                        repo, config, bindings, groups, input_pairs, output_root
                    )
                    print(
                        json.dumps(
                            {
                                "status": "BOUNDED_CHECKPOINT",
                                "authoritative_trace_count": len(trace_inventory),
                            }
                        )
                    )
                    return 0
                materialize_candidate_trace(
                    repo,
                    config,
                    base_config,
                    bindings,
                    candidate_id,
                    descriptor,
                    rows,
                    input_pairs[(descriptor["source_id"], descriptor["sequence_id"])],
                    route_map,
                    reset_next_identities,
                    tracker_config,
                    output_root,
                )
                if not existed_before:
                    completed_this_run += 1
        trace_inventory = collect_authoritative_receipts(
            repo, config, bindings, groups, input_pairs, output_root
        )
        expected = config["expected_scope"]["candidate_ledger_traces"]
        if len(trace_inventory) != expected:
            raise ExecutionAborted(
                f"authoritative_trace_count_mismatch:expected={expected}:"
                f"observed={len(trace_inventory)}"
            )
        receipt = base_terminal_receipt(
            "CANDIDATE_REPLAY_COMPLETE",
            config,
            bindings,
            input_inventory,
            trace_inventory,
            None,
        )
        terminal = output_root / "terminal-receipt-r2.json"
        atomic_write_json(terminal, receipt)
        print(
            json.dumps(
                {
                    "terminal_state": receipt["terminal_state"],
                    "authoritative_trace_count": len(trace_inventory),
                    "receipt": str(terminal),
                }
            )
        )
        return 0
    except InputBlocked as error:
        if config and output_root is not None:
            trace_inventory = (
                collect_authoritative_receipts(
                    repo, config, bindings, groups, input_pairs, output_root
                )
                if groups and input_pairs
                else []
            )
            receipt = base_terminal_receipt(
                "FAIL_CLOSED_INPUT_BLOCKED",
                config,
                bindings,
                input_inventory,
                trace_inventory,
                str(error),
            )
            atomic_write_json(output_root / "terminal-receipt-r2.json", receipt)
        print(
            json.dumps(
                {"terminal_state": "FAIL_CLOSED_INPUT_BLOCKED", "error": str(error)}
            )
        )
        return 2
    except Exception as error:
        if config and output_root is not None:
            trace_inventory = (
                collect_authoritative_receipts(
                    repo, config, bindings, groups, input_pairs, output_root
                )
                if groups and input_pairs
                else []
            )
            receipt = base_terminal_receipt(
                "FAIL_CLOSED_EXECUTION_ABORTED",
                config,
                bindings,
                input_inventory,
                trace_inventory,
                str(error),
            )
            receipt["execution_error"] = {
                "type": error.__class__.__name__,
                "message": str(error),
            }
            atomic_write_json(output_root / "terminal-receipt-r2.json", receipt)
        print(
            json.dumps(
                {
                    "terminal_state": "FAIL_CLOSED_EXECUTION_ABORTED",
                    "error": str(error),
                }
            )
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

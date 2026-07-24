#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from candidate_replay_r2 import (
    TERMINAL_SCHEMA,
    TERMINAL_STATES,
    InputBlocked,
    atomic_write_json,
    collect_authoritative_receipts,
    load_replay_config,
    load_json,
    resolve_input_ledgers,
    sha256_file,
)
from exploratory_profiles_r2_l1 import (
    canonical_bytes,
    identity,
    load_route_map,
    replay_candidate_ledger,
)


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def forbidden_fragments(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    forbidden = ("winner", "ranking", "best_candidate", "tie_break", "promotion")
    if isinstance(value, dict):
        for key, child in value.items():
            lower = str(key).lower()
            if any(fragment in lower for fragment in forbidden) and child is not False:
                findings.append(f"{path}.{key}")
            findings.extend(forbidden_fragments(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(forbidden_fragments(child, f"{path}[{index}]"))
    return findings


def validate(
    repo: Path, config_path: Path, terminal_path: Path
) -> dict[str, Any]:
    (
        config,
        base_config,
        bindings,
        groups,
        resets,
        _,
    ) = load_replay_config(repo, config_path)
    input_pairs, input_inventory = resolve_input_ledgers(repo, config, groups)
    receipt = load_json(terminal_path)
    require(receipt.get("schema") == TERMINAL_SCHEMA, "terminal schema mismatch")
    require(
        receipt.get("terminal_state") in TERMINAL_STATES, "illegal terminal state"
    )
    require(receipt.get("bindings") == bindings, "terminal bindings drift")
    require(not forbidden_fragments(receipt), "forbidden comparison field present")
    require(
        receipt["verified_input"]["sequence_ledgers"] == 41,
        "verified ledger count mismatch",
    )
    require(receipt["verified_input"]["frames"] == 62229, "verified frame count mismatch")
    require(
        receipt["verified_input"]["discontinuity_resets"] == len(resets) == 15,
        "reset count mismatch",
    )
    require(
        receipt["verified_input"]["inventory"] == input_inventory,
        "input inventory drift",
    )
    output_root = (repo / config["output_root"]).resolve()
    observed = collect_authoritative_receipts(
        repo, config, bindings, groups, input_pairs, output_root
    )
    expected_authoritative_paths = {
        row["authoritative_receipt_path"] for row in observed
    }
    actual_authoritative_paths = {
        str(path.relative_to(repo)).replace("\\", "/")
        for path in output_root.glob(
            "candidate-traces/*/*/authoritative-receipt.json"
        )
    }
    require(
        actual_authoritative_paths == expected_authoritative_paths,
        "unexpected or invalid authoritative receipt present",
    )
    execution = receipt["candidate_execution"]
    require(execution["trace_inventory"] == observed, "trace inventory drift")
    require(
        execution["authoritative_trace_count"] == len(observed),
        "trace count drift",
    )
    require(
        execution["partial_trace_profile_authority"] is False,
        "partial trace gained profile authority",
    )
    require(
        receipt["profiles"]
        == {"generated": False, "count": 0, "authority": False},
        "profile authority drift",
    )
    require(not (output_root / "profiles").exists(), "profile output exists")
    complete = receipt["terminal_state"] == "CANDIDATE_REPLAY_COMPLETE"
    if complete:
        require(receipt["blocker"] is None, "complete receipt has blocker")
        require(len(observed) == 123, "complete trace count mismatch")
        require(
            sum(row["frame_count"] for row in observed) == 3 * 62229,
            "complete trace frame coverage mismatch",
        )
        per_candidate = {
            candidate: sum(1 for row in observed if row["candidate_id"] == candidate)
            for candidate in config["candidate_roster"]
        }
        require(
            all(count == 41 for count in per_candidate.values()),
            "candidate trace coverage mismatch",
        )
        require(
            sum(row["discontinuity_reset_count"] for row in observed) == 3 * 15,
            "candidate reset coverage mismatch",
        )
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
            (descriptor["source_id"], descriptor["sequence_id"]): (descriptor, rows)
            for descriptor, rows in groups
        }
        for item in observed:
            key = (item["source_id"], item["sequence_id"])
            descriptor, rows = groups_by_key[key]
            marker = load_json(repo / item["authoritative_receipt_path"])
            trace = load_json(repo / marker["trace_path"])
            ledger = load_json(input_pairs[key][0])
            expected_frames, _ = replay_candidate_ledger(
                item["candidate_id"],
                ledger,
                route_map,
                reset_next_identities,
                tracker_config,
                base_config,
            )
            for frame in expected_frames:
                frame.pop("candidate_consume_timestamp_ns", None)
            require(
                canonical_bytes(expected_frames) == canonical_bytes(trace["frames"]),
                "independent state replay mismatch:"
                f"{item['candidate_id']}:{item['source_id']}/{item['sequence_id']}",
            )
            require(
                [identity(frame) for frame in trace["frames"]]
                == [identity(row) for row in rows],
                "trace identity/order mismatch",
            )
    else:
        require(isinstance(receipt["blocker"], str), "failure receipt lacks blocker")
        require(len(observed) < 123, "failure receipt has complete trace set")
    return {
        "schema": "blindassist_ustrf_route_target_l1_candidate_replay_validation_r2",
        "status": "VALID",
        "terminal_state": receipt["terminal_state"],
        "authoritative_trace_count": len(observed),
        "candidate_counts": {
            candidate: sum(1 for row in observed if row["candidate_id"] == candidate)
            for candidate in config["candidate_roster"]
        },
        "verified_trace_frames": sum(row["frame_count"] for row in observed),
        "verified_discontinuity_resets": sum(
            row["discontinuity_reset_count"] for row in observed
        ),
        "terminal_receipt_sha256": sha256_file(terminal_path),
        "profile_authority": False,
        "candidate_comparison_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = load_json(args.config.resolve())
    receipt = (
        args.receipt.resolve()
        if args.receipt
        else (repo / config["output_root"] / "terminal-receipt-r2.json")
    )
    try:
        result = validate(repo, args.config.resolve(), receipt)
        output = repo / config["output_root"] / "validation-receipt-r2.json"
        atomic_write_json(output, result)
        print(json.dumps(result))
        return 0
    except (InputBlocked, ValidationError, OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"status": "INVALID", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

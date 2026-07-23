#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from exploratory_profiles_r2_l1 import (
    ExecutionAborted,
    InputBlocked,
    atomic_write_json,
    base_terminal_receipt,
    collect_verified_input_artifacts,
    gap_matrix,
    implementation_bindings,
    load_and_verify_config,
    load_json,
    load_route_map,
    materialize_existing_lilocbench,
    materialize_one_crowdbot,
    run_all_candidate_profiles,
    sha256_file,
    validate_exhausted_resource_guard_receipt,
    validate_mask_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--maximum-crowdbot-shards", type=int)
    parser.add_argument("--finalize-existing-abort", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = repo / "artifacts.local/evidence/ustrf-route-target-l1-exploratory-profile-r1"
    output_root.mkdir(parents=True, exist_ok=True)
    terminal_path = output_root / "terminal-receipt-r1.json"
    bindings: dict[str, str] = {}
    groups = []
    resets = []
    config = {}
    try:
        config, bindings = load_and_verify_config(repo, args.config)
        bindings.update(implementation_bindings(repo))
        mask = load_json(repo / config["parent_bindings"]["eligibility_mask"]["path"])
        groups, resets = validate_mask_contract(config, mask)
        gc.collect()
        guard_path = output_root / "resource-guard-attempts-r1.json"
        guard_preview = load_json(guard_path) if guard_path.exists() else None
        if guard_preview is not None and not isinstance(
            guard_preview.get("automatic_retry_allowed_after_receipt"), bool
        ):
            raise ExecutionAborted("resource_guard_receipt_retry_flag_invalid")
        if (
            guard_preview is not None
            and guard_preview["automatic_retry_allowed_after_receipt"] is False
        ):
            current_implementations = {
                key: value
                for key, value in bindings.items()
                if key.endswith("_implementation_sha256")
                or key == "terminal_schema_sha256"
            }
            guard = validate_exhausted_resource_guard_receipt(
                guard_path,
                bindings["config_sha256"],
                current_implementations,
                int(
                    config["resource_guards"][
                        "minimum_system_available_physical_memory_bytes"
                    ]
                ),
                1
                + int(
                    config["resource_guards"][
                        "maximum_retry_count_after_initial_attempt"
                    ]
                ),
            )
            route_map = load_route_map(config, repo)
            gaps = gap_matrix(groups, output_root, route_map)
            final_attempt = guard["attempts"][-1]
            blocker = (
                "available_memory_guard:"
                f"observed={final_attempt['observed_available_bytes']}:"
                f"required={guard['required_minimum_bytes']}"
            )
            verified_artifacts = collect_verified_input_artifacts(
                repo, groups, output_root
            )
            receipt = base_terminal_receipt(
                "FAIL_CLOSED_EXECUTION_ABORTED",
                bindings,
                groups,
                resets,
                gaps,
                config,
                blocker,
                verified_artifacts,
            )
            receipt["guards"] = {
                **receipt["guards"],
                "attempt_receipt_path": str(guard_path.relative_to(repo)).replace("\\", "/"),
                "attempt_receipt_sha256": __import__(
                    "exploratory_profiles_r2_l1"
                ).sha256_file(guard_path),
                "attempt_count": 3,
                "retry_limit_exhausted": True,
            }
            receipt["execution_error"] = {
                "type": "ExecutionAborted",
                "message": blocker,
                "stage": "pre_device_resource_guard",
                "device_attempt_created": False,
                "last_safe_checkpoint": final_attempt["last_safe_checkpoint"],
            }
            atomic_write_json(terminal_path, receipt)
            print(
                json.dumps(
                    {
                        "terminal_state": receipt["terminal_state"],
                        "first_blocker": blocker,
                        "receipt": str(terminal_path),
                    }
                )
            )
            return 3
        if args.finalize_existing_abort:
            raise ExecutionAborted("no_exhausted_resource_guard_receipt_to_finalize")
        materialize_existing_lilocbench(config, repo, groups, output_root)
        if not args.preflight_only:
            completed = 0
            for descriptor, rows in groups:
                if descriptor["source_id"].startswith("lilocbench_"):
                    continue
                if args.maximum_crowdbot_shards is not None and completed >= args.maximum_crowdbot_shards:
                    break
                materialize_one_crowdbot(
                    config, bindings, repo, descriptor, rows, output_root
                )
                completed += 1
        route_map = load_route_map(config, repo)
        gaps = gap_matrix(groups, output_root, route_map)
        first_gap = next((row for row in gaps if row["missing_fields"]), None)
        if first_gap is not None:
            blocker = (
                f"{first_gap['source_id']}/{first_gap['sequence_id']}:"
                f"{','.join(first_gap['missing_fields'])}"
            )
            receipt = base_terminal_receipt(
                "FAIL_CLOSED_INPUT_BLOCKED",
                bindings,
                groups,
                resets,
                gaps,
                config,
                blocker,
                collect_verified_input_artifacts(repo, groups, output_root),
            )
            atomic_write_json(terminal_path, receipt)
            print(
                json.dumps(
                    {
                        "terminal_state": receipt["terminal_state"],
                        "first_blocker": blocker,
                        "receipt": str(terminal_path),
                    }
                )
            )
            return 2
        profiles, trace_receipts = run_all_candidate_profiles(
            config,
            repo,
            groups,
            resets,
            route_map,
            mask,
            bindings,
            output_root,
        )
        receipt = base_terminal_receipt(
            "EXPLORATORY_PROFILES_COMPLETE",
            bindings,
            groups,
            resets,
            gaps,
            config,
            None,
            collect_verified_input_artifacts(repo, groups, output_root),
        )
        receipt["candidate_execution"] = {
            "started": True,
            "candidate_order": config["candidate_roster"],
            "authoritative_trace_count": len(trace_receipts),
            "partial_trace_evaluation_authority": False,
            "trace_receipts": trace_receipts,
        }
        receipt["profiles"] = profiles
        atomic_write_json(terminal_path, receipt)
        print(
            json.dumps(
                {
                    "terminal_state": receipt["terminal_state"],
                    "profile_count": len(profiles),
                    "receipt": str(terminal_path),
                }
            )
        )
        return 0
    except InputBlocked as error:
        if config and groups:
            gaps = gap_matrix(groups, output_root, load_route_map(config, repo))
            receipt = base_terminal_receipt(
                "FAIL_CLOSED_INPUT_BLOCKED",
                bindings,
                groups,
                resets,
                gaps,
                config,
                str(error),
                collect_verified_input_artifacts(repo, groups, output_root),
            )
            atomic_write_json(terminal_path, receipt)
        print(json.dumps({"terminal_state": "FAIL_CLOSED_INPUT_BLOCKED", "error": str(error)}))
        return 2
    except (ExecutionAborted, Exception) as error:
        if config and groups:
            route_map = load_route_map(config, repo)
            gaps = gap_matrix(groups, output_root, route_map)
            receipt = base_terminal_receipt(
                "FAIL_CLOSED_EXECUTION_ABORTED",
                bindings,
                groups,
                resets,
                gaps,
                config,
                str(error),
                collect_verified_input_artifacts(repo, groups, output_root),
            )
            receipt["execution_error"] = {
                "type": error.__class__.__name__,
                "message": str(error),
                "stage": "runtime_contract",
                "device_attempt_created": (
                    output_root.joinpath("attempts").exists()
                    and any(output_root.joinpath("attempts").rglob("attempt-*"))
                ),
                "last_safe_checkpoint": {
                    "verified_sequence_ledgers": receipt["verified_scope"][
                        "fully_input_verified_sequence_ledgers"
                    ],
                    "verified_frames": receipt["verified_scope"][
                        "fully_input_verified_frames"
                    ],
                    "candidate_execution_started": False,
                },
            }
            atomic_write_json(terminal_path, receipt)
        print(json.dumps({"terminal_state": "FAIL_CLOSED_EXECUTION_ABORTED", "error": str(error)}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

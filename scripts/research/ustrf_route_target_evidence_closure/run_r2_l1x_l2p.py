#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import exploratory_profiles_r2_l1 as r1
from r2_l1x_l2p import (
    READY_STATE,
    build_context,
    build_mechanism_gap_audit,
    build_terminal,
    materialize_one_crowdbot_r2,
    output_root,
    write_progress,
)


def _write_failure(
    repo: Path,
    root: Path,
    prereg: dict,
    context: dict,
    state: str,
    error: Exception,
) -> None:
    receipt = build_terminal(state, prereg, context, root, str(error))
    receipt["execution_error"] = {
        "type": error.__class__.__name__,
        "message": str(error),
        "candidate_partial_trace_has_profile_authority": False,
    }
    r1.atomic_write_json(root / "terminal-receipt-r2-l1x-l2p.json", receipt)


def _materialize_with_bounded_retries(
    prereg: dict,
    context: dict,
    repo: Path,
    descriptor: dict,
    rows: list[dict],
    root: Path,
) -> None:
    ledger_path, successor_path = r1.compact_paths(
        root, descriptor["source_id"], descriptor["sequence_id"]
    )
    if r1.validate_compact_ledger(ledger_path, successor_path, descriptor, rows):
        return
    maximum_attempts = (
        int(prereg["execution_recovery"]["initial_attempts"])
        + int(prereg["execution_recovery"]["bounded_retries"])
    )
    errors = []
    for _ in range(maximum_attempts):
        try:
            gc.collect()
            materialize_one_crowdbot_r2(
                prereg,
                context["base_config"],
                context["bindings"],
                repo,
                descriptor,
                rows,
                root,
            )
            if not r1.validate_compact_ledger(
                ledger_path, successor_path, descriptor, rows
            ):
                raise r1.ExecutionAborted("successor_not_valid_after_materialization")
            return
        except r1.ExecutionAborted as error:
            errors.append(str(error))
            guard_path = root / "resource-guard-attempts-r2.json"
            if guard_path.exists():
                guard = r1.load_json(guard_path)
                if guard.get("automatic_retry_allowed_after_receipt") is False:
                    break
    raise r1.ExecutionAborted(
        f"bounded_retries_exhausted:{descriptor['source_id']}/"
        f"{descriptor['sequence_id']}:{'|'.join(errors)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--maximum-crowdbot-shards", type=int)
    args = parser.parse_args()
    repo = args.repo.resolve()
    prereg: dict = {}
    context: dict = {}
    root = repo / "artifacts.local/evidence/ustrf-route-target-r2-l1x-l2p-r1"
    try:
        prereg, context = build_context(repo, args.config)
        root = output_root(repo, prereg)
        root.mkdir(parents=True, exist_ok=True)
        if (root / "terminal-receipt-r2-l1x-l2p.json").exists():
            terminal = r1.load_json(root / "terminal-receipt-r2-l1x-l2p.json")
            print(
                json.dumps(
                    {
                        "terminal_state": terminal["terminal_state"],
                        "receipt": str(root / "terminal-receipt-r2-l1x-l2p.json"),
                        "rerun_refused": True,
                    }
                )
            )
            return 0 if terminal["terminal_state"] == READY_STATE else 3
        r1.materialize_existing_lilocbench(
            context["base_config"], repo, context["groups"], root
        )
        progress = write_progress(repo, root, prereg, context, "PREFLIGHT_VALID")
        if args.preflight_only:
            print(json.dumps(progress))
            return 0
        completed = 0
        for descriptor, rows in context["groups"]:
            if descriptor["source_id"].startswith("lilocbench_"):
                continue
            ledger_path, successor_path = r1.compact_paths(
                root, descriptor["source_id"], descriptor["sequence_id"]
            )
            if r1.validate_compact_ledger(
                ledger_path, successor_path, descriptor, rows
            ):
                continue
            if (
                args.maximum_crowdbot_shards is not None
                and completed >= args.maximum_crowdbot_shards
            ):
                progress = write_progress(
                    repo, root, prereg, context, "BOUNDED_MATERIALIZATION_CHECKPOINT"
                )
                print(json.dumps(progress))
                return 4
            _materialize_with_bounded_retries(
                prereg, context, repo, descriptor, rows, root
            )
            completed += 1
            write_progress(
                repo,
                root,
                prereg,
                context,
                f"MATERIALIZED_LEDGER_{completed}",
            )
        gaps, verified_ledgers, verified_frames = __import__(
            "r2_l1x_l2p"
        ).verified_scope(context["groups"], root, context["route_map"])
        if verified_ledgers != 41 or verified_frames != 62229:
            first = next(row for row in gaps if row["missing_fields"])
            raise r1.InputBlocked(
                f"{first['source_id']}/{first['sequence_id']}:"
                f"{','.join(first['missing_fields'])}"
            )
        if args.materialize_only:
            progress = write_progress(
                repo, root, prereg, context, "FULL_INPUT_READY_CANDIDATES_UNRUN"
            )
            print(json.dumps(progress))
            return 0
        profiles, trace_receipts = r1.run_all_candidate_profiles(
            context["base_config"],
            repo,
            context["groups"],
            context["resets"],
            context["route_map"],
            context["mask"],
            context["bindings"],
            root,
        )
        if len(trace_receipts) != 123 or len(profiles) != 3:
            raise r1.ExecutionAborted("candidate_trace_or_profile_coverage_incomplete")
        exploratory = r1.base_terminal_receipt(
            "EXPLORATORY_PROFILES_COMPLETE",
            context["bindings"],
            context["groups"],
            context["resets"],
            gaps,
            context["base_config"],
            None,
            r1.collect_verified_input_artifacts(repo, context["groups"], root),
        )
        exploratory["recovery_stage"] = "R2-L1E-R2"
        exploratory["attempt_namespace"] = prereg["execution_recovery"][
            "attempt_namespace"
        ]
        exploratory["parent_r1_attempts_count_toward_this_stage"] = False
        exploratory["candidate_execution"] = {
            "started": True,
            "candidate_order": prereg["l1_profile"]["candidate_order"],
            "authoritative_trace_count": len(trace_receipts),
            "partial_trace_evaluation_authority": False,
            "trace_receipts": trace_receipts,
        }
        exploratory["profiles"] = profiles
        exploratory_path = root / "exploratory-terminal-receipt-r2.json"
        r1.atomic_write_json(exploratory_path, exploratory)
        audit = build_mechanism_gap_audit(profiles, prereg, context)
        audit_path = root / "mechanism-gap-audit-r1.json"
        r1.atomic_write_json(audit_path, audit)
        progress = write_progress(
            repo, root, prereg, context, "AWAITING_INDEPENDENT_READ_ONLY_REVIEW"
        )
        progress["exploratory_terminal_receipt_sha256"] = r1.sha256_file(
            exploratory_path
        )
        progress["mechanism_gap_audit_sha256"] = r1.sha256_file(audit_path)
        r1.atomic_write_json(root / "progress-receipt-r2.json", progress)
        print(json.dumps(progress))
        return 0
    except r1.InputBlocked as error:
        if prereg and context:
            root.mkdir(parents=True, exist_ok=True)
            _write_failure(
                repo, root, prereg, context, "FAIL_CLOSED_INPUT_BLOCKED", error
            )
        print(json.dumps({"terminal_state": "FAIL_CLOSED_INPUT_BLOCKED", "error": str(error)}))
        return 2
    except Exception as error:
        if prereg and context:
            root.mkdir(parents=True, exist_ok=True)
            _write_failure(
                repo,
                root,
                prereg,
                context,
                "FAIL_CLOSED_EXECUTION_ABORTED",
                error,
            )
        print(
            json.dumps(
                {"terminal_state": "FAIL_CLOSED_EXECUTION_ABORTED", "error": str(error)}
            )
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

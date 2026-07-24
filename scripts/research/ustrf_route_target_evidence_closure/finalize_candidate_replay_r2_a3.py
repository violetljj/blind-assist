#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exploratory_profiles_r2_l1 import atomic_write_json, load_json, sha256_file

CONFIG_SCHEMA = "blindassist_ustrf_route_target_l1_candidate_replay_finalization_a3"
TERMINAL_SCHEMA = "blindassist_ustrf_route_target_l1_candidate_replay_terminal_a3"


def verify(repo: Path, binding: dict[str, Any], label: str) -> Path:
    path = (repo / binding["path"]).resolve()
    if not path.is_file():
        raise RuntimeError(f"{label}_missing")
    observed = sha256_file(path)
    if observed != binding["sha256"]:
        raise RuntimeError(
            f"{label}_sha256_mismatch:expected={binding['sha256']}:observed={observed}"
        )
    return path


def implementation_paths(repo: Path) -> dict[str, Path]:
    module = repo / "scripts/research/ustrf_route_target_evidence_closure"
    return {
        "finalizer_sha256": module / "finalize_candidate_replay_r2_a3.py",
        "validator_sha256": module / "validate_candidate_replay_r2_a3.py",
        "tests_sha256": module / "test_candidate_replay_r2_a3.py",
        "terminal_schema_sha256": (
            repo / "schemas/ustrf_route_target_l1_candidate_replay_a3.schema.json"
        ),
    }


def load_context(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("unexpected_finalization_a3_config_schema")
    closed = config.get("authority", {})
    if any(value is not False for value in closed.values()):
        raise RuntimeError("finalization_a3_authority_must_remain_closed")
    bindings = {"config_sha256": sha256_file(config_path)}
    parent_config = verify(repo, config["parent"]["config"], "parent_config")
    parent_terminal = verify(repo, config["parent"]["terminal"], "parent_terminal")
    parent_validation = verify(
        repo, config["parent"]["validation"], "parent_validation"
    )
    bindings.update(
        {
            "parent_config_sha256": sha256_file(parent_config),
            "parent_terminal_sha256": sha256_file(parent_terminal),
            "parent_validation_sha256": sha256_file(parent_validation),
        }
    )
    for name, path in implementation_paths(repo).items():
        if sha256_file(path) != config["implementation_bindings"][name]:
            raise RuntimeError(f"{name}_mismatch")
        bindings[name] = sha256_file(path)
    terminal = load_json(parent_terminal)
    validation = load_json(parent_validation)
    if (
        terminal.get("terminal_state") != "CANDIDATE_REPLAY_COMPLETE"
        or terminal["candidate_execution"]["authoritative_trace_count"] != 123
        or terminal["verified_input"]["sequence_ledgers"] != 41
        or terminal["verified_input"]["frames"] != 62229
        or terminal["verified_input"]["discontinuity_resets"] != 15
        or terminal["profiles"]
        != {"generated": False, "count": 0, "authority": False}
        or validation.get("status") != "VALID"
        or validation.get("terminal_state") != "CANDIDATE_REPLAY_COMPLETE"
        or validation.get("authoritative_trace_count") != 123
        or validation.get("verified_trace_frames") != 186687
        or validation.get("verified_discontinuity_resets") != 45
        or validation.get("parent_traces_adopted_without_candidate_rerun") != 10
        or validation.get("minimum_available_memory_bytes") != 4 * 1024**3
        or validation.get("profile_authority") is not False
        or validation.get("candidate_comparison_authority") is not False
    ):
        raise RuntimeError("parent_a2_completion_or_validation_contract_invalid")
    return config, bindings, terminal, validation


def build_terminal(
    config: dict[str, Any],
    bindings: dict[str, str],
    parent_terminal: dict[str, Any],
    parent_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": TERMINAL_SCHEMA,
        "stage": "R2-L1-CANDIDATE-REPLAY-R2-FINALIZATION-A3",
        "terminal_state": "CANDIDATE_REPLAY_COMPLETE",
        "authority": "candidate_trace_only_no_profile_comparison_selection_or_promotion",
        "bindings": bindings,
        "verified_scope": {
            "candidate_count": 3,
            "sequence_ledgers": 41,
            "input_frames": 62229,
            "discontinuity_resets_per_candidate": 15,
            "authoritative_traces": 123,
            "authoritative_trace_frames": 186687,
            "authoritative_trace_resets": 45,
            "parent_traces_adopted_without_candidate_rerun": 10,
            "new_a2_traces": 113,
            "minimum_available_memory_bytes": 4 * 1024**3,
        },
        "parent_evidence": {
            "a2_terminal_state": parent_terminal["terminal_state"],
            "a2_terminal_sha256": bindings["parent_terminal_sha256"],
            "a2_validation_status": parent_validation["status"],
            "a2_validation_sha256": bindings["parent_validation_sha256"],
        },
        "profiles": {
            "generated": False,
            "count": 0,
            "authority": False,
        },
        "claim_boundary": {
            "truth_joined": False,
            "metric_profile_generated": False,
            "candidate_comparison_allowed": False,
            "winner_or_ranking_allowed": False,
            "selection_allowed": False,
            "l2_or_l3_allowed": False,
            "android_shadow_allowed": False,
            "h2_allowed": False,
            "human_outcome_allowed": False,
            "independent_walking_safety_allowed": False,
            "production_allowed": False,
            "new_data_added": False,
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        config, bindings, terminal, validation = load_context(
            repo, args.config.resolve()
        )
        receipt = build_terminal(config, bindings, terminal, validation)
        output = (repo / config["output_root"]).resolve()
        output.mkdir(parents=True, exist_ok=True)
        path = output / "terminal-receipt-a3.json"
        atomic_write_json(path, receipt)
        print(
            json.dumps(
                {
                    "terminal_state": receipt["terminal_state"],
                    "authoritative_trace_count": receipt["verified_scope"][
                        "authoritative_traces"
                    ],
                    "receipt": str(path),
                }
            )
        )
        return 0
    except Exception as error:
        print(json.dumps({"terminal_state": "FAIL_CLOSED", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

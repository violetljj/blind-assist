#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import exploratory_profiles_r2_l1 as r1
import run_r2_l1x_l2p as base_runner
from r2_l1x_l2p import load_json, verify_bound_file


AMENDMENT_ID = "R2-L1E-R2-TRANSPORT-A1"
SAFE_REMOTE = re.compile(
    r"r2l1e-r2/[A-Za-z0-9._-]+/attempt-(001|002|003)"
)


def verify_amendment(repo: Path, config_path: Path) -> dict:
    config = load_json(config_path)
    if (
        config.get("schema")
        != "blindassist_ustrf_route_target_r2_l1x_l2p_transport_amendment_a1"
    ):
        raise r1.InputBlocked("transport_amendment_config_schema_drift")
    parent_path = verify_bound_file(
        repo, config["parent_prereg"], "transport_amendment_parent_prereg"
    )
    parent = load_json(parent_path)
    amendment = config.get("transport_amendment")
    if not isinstance(amendment, dict) or amendment.get("id") != AMENDMENT_ID:
        raise r1.InputBlocked("transport_amendment_missing")
    if amendment.get("outcome_unseen") is not True:
        raise r1.InputBlocked("transport_amendment_not_outcome_unseen")
    if amendment.get("only_change") != "remote_cleanup_path_validation":
        raise r1.InputBlocked("transport_amendment_scope_drift")
    if amendment.get("parent_attempts_count_toward_amendment") is not False:
        raise r1.InputBlocked("failed_r2_attempts_reused")
    for name, binding in amendment["failed_r2_parent_bindings"].items():
        verify_bound_file(repo, binding, f"failed_r2_{name}")
    terminal = load_json(
        repo
        / amendment["failed_r2_parent_bindings"]["terminal_receipt"]["path"]
    )
    if terminal.get("terminal_state") != "FAIL_CLOSED_EXECUTION_ABORTED":
        raise r1.InputBlocked("failed_r2_terminal_rewritten")
    if terminal["verified_scope"].get("verified_frames") != 4594:
        raise r1.InputBlocked("failed_r2_verified_scope_rewritten")
    if terminal.get("claim_boundary", {}).get("candidate_selected") is not False:
        raise r1.InputBlocked("failed_r2_authority_rewritten")
    return {"overlay": config, "parent": parent}


def materialize_amended_prereg(
    repo: Path, overlay_path: Path, verified: dict
) -> Path:
    overlay = verified["overlay"]
    amended = verified["parent"]
    amended["status"] = "FROZEN_TRANSPORT_AMENDMENT_A1_BEFORE_DEVICE_OUTPUT"
    amended["execution_recovery"]["output_root"] = overlay["output_root"]
    amended["execution_recovery"]["resource_guards"]["output_root"] = overlay[
        "output_root"
    ]
    amended["transport_amendment"] = {
        **overlay["transport_amendment"],
        "overlay_path": str(overlay_path.relative_to(repo)).replace("\\", "/"),
        "overlay_sha256": r1.sha256_file(overlay_path),
    }
    amended["implementation_bindings"].update(
        overlay["amended_implementation_bindings"]
    )
    path = repo / overlay["output_root"] / "frozen-merged-prereg-a1.json"
    r1.atomic_write_json(path, amended)
    return path


def remote_cleanup_a1(adb: Path, remote_relative: str, retry_count: int) -> None:
    if not SAFE_REMOTE.fullmatch(remote_relative):
        raise r1.ExecutionAborted(
            f"unsafe_a1_remote_cleanup_target:{remote_relative}"
        )
    absolute = (
        "/sdcard/Android/data/com.linnan.blindassist/files/" + remote_relative
    )
    errors = []
    for _ in range(1 + retry_count):
        process = r1.run_command(
            [str(adb), "shell", "rm", "-rf", absolute], timeout=120
        )
        if process.returncode == 0:
            return
        errors.append(process.stdout)
    raise r1.ExecutionAborted(
        f"a1_remote_cleanup_failed:{remote_relative}:{' | '.join(errors)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    known, _ = parser.parse_known_args()
    repo = known.repo.resolve()
    overlay_path = known.config.resolve()
    verified = verify_amendment(repo, overlay_path)
    amended_path = materialize_amended_prereg(repo, overlay_path, verified)
    r1.bounded_remote_cleanup = remote_cleanup_a1
    for index, value in enumerate(sys.argv):
        if value == "--config" and index + 1 < len(sys.argv):
            sys.argv[index + 1] = str(amended_path)
            break
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import candidate_replay_r2 as parent
from candidate_replay_r2_continuation_a2 import (
    MINIMUM_AVAILABLE_MEMORY_BYTES_A2,
    activate_short_trace_paths,
    collect_a2_receipts,
    load_a2_context,
)
from exploratory_profiles_r2_l1 import (
    atomic_write_json,
    available_memory_bytes,
    canonical_bytes,
    load_json,
    load_route_map,
    replay_candidate_ledger,
    sha256_file,
)

CONFIG_SCHEMA = (
    "blindassist_ustrf_route_target_l1_candidate_replay_memory_guard_validation_a4"
)


def require_memory(observed: int, minimum: int) -> None:
    if observed < minimum:
        raise RuntimeError(
            f"available_memory_guard:observed={observed}:minimum={minimum}"
        )


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        config_path = args.config.resolve()
        config = load_json(config_path)
        if config.get("schema") != CONFIG_SCHEMA:
            raise RuntimeError("unexpected_memory_guard_a4_config_schema")
        if any(value is not False for value in config["authority"].values()):
            raise RuntimeError("memory_guard_a4_authority_must_remain_closed")
        if (
            config.get("minimum_available_memory_bytes")
            != MINIMUM_AVAILABLE_MEMORY_BYTES_A2
        ):
            raise RuntimeError("memory_guard_a4_minimum_must_equal_four_gib")
        parent_config_path = verify(
            repo, config["parent"]["a2_config"], "a2_config"
        )
        a2_terminal = verify(repo, config["parent"]["a2_terminal"], "a2_terminal")
        a2_validation = verify(
            repo, config["parent"]["a2_validation"], "a2_validation"
        )
        a3_terminal = verify(repo, config["parent"]["a3_terminal"], "a3_terminal")
        a3_validation = verify(
            repo, config["parent"]["a3_validation"], "a3_validation"
        )
        implementation = Path(__file__).resolve()
        if sha256_file(implementation) != config["implementation_sha256"]:
            raise RuntimeError("memory_guard_a4_implementation_sha256_mismatch")
        test_path = Path(__file__).with_name(
            "test_candidate_replay_r2_memory_guard_a4.py"
        )
        if sha256_file(test_path) != config["test_sha256"]:
            raise RuntimeError("memory_guard_a4_test_sha256_mismatch")
        (
            a2_config,
            base_config,
            bindings,
            groups,
            resets,
            _,
        ) = load_a2_context(repo, parent_config_path)
        input_pairs, _ = parent.resolve_input_ledgers(repo, a2_config, groups)
        activate_short_trace_paths()
        output_root = (repo / a2_config["output_root"]).resolve()
        traces = collect_a2_receipts(
            repo, a2_config, bindings, groups, input_pairs, output_root
        )
        if len(traces) != 123:
            raise RuntimeError("memory_guard_a4_trace_count_mismatch")
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
        samples = []
        verified_frames = 0
        for index, row in enumerate(traces, start=1):
            observed = available_memory_bytes()
            require_memory(observed, MINIMUM_AVAILABLE_MEMORY_BYTES_A2)
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
                    "memory_guard_a4_state_replay_mismatch:"
                    f"{row['candidate_id']}:{row['source_id']}/{row['sequence_id']}"
                )
            verified_frames += len(expected_frames)
            samples.append(
                {
                    "index": index,
                    "candidate_id": row["candidate_id"],
                    "source_id": row["source_id"],
                    "sequence_id": row["sequence_id"],
                    "observed_available_memory_bytes_before_replay": observed,
                    "trace_sha256": row["trace_sha256"],
                }
            )
        if verified_frames != 186687:
            raise RuntimeError("memory_guard_a4_verified_frame_count_mismatch")
        receipt = {
            "schema": "blindassist_ustrf_route_target_l1_candidate_replay_memory_guard_receipt_a4",
            "stage": "R2-L1-CANDIDATE-REPLAY-R2-MEMORY-GUARD-A4",
            "status": "PASS",
            "authority": "independent_validation_only_no_new_authoritative_candidate_trace",
            "bindings": {
                "config_sha256": sha256_file(config_path),
                "implementation_sha256": sha256_file(implementation),
                "test_sha256": sha256_file(test_path),
                "a2_config_sha256": sha256_file(parent_config_path),
                "a2_terminal_sha256": sha256_file(a2_terminal),
                "a2_validation_sha256": sha256_file(a2_validation),
                "a3_terminal_sha256": sha256_file(a3_terminal),
                "a3_validation_sha256": sha256_file(a3_validation),
            },
            "minimum_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES_A2,
            "sample_count": len(samples),
            "minimum_observed_available_memory_bytes": min(
                row["observed_available_memory_bytes_before_replay"]
                for row in samples
            ),
            "maximum_observed_available_memory_bytes": max(
                row["observed_available_memory_bytes_before_replay"]
                for row in samples
            ),
            "authoritative_trace_count_verified": len(traces),
            "deterministic_trace_frames_replayed": verified_frames,
            "samples": samples,
            "new_authoritative_candidate_traces_created": 0,
            "profile_authority": False,
            "candidate_comparison_authority": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        output = (repo / config["output_root"]).resolve()
        output.mkdir(parents=True, exist_ok=True)
        receipt_path = output / "memory-guard-validation-a4.json"
        atomic_write_json(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "sample_count": receipt["sample_count"],
                    "minimum_observed_available_memory_bytes": receipt[
                        "minimum_observed_available_memory_bytes"
                    ],
                    "authoritative_trace_count_verified": len(traces),
                    "deterministic_trace_frames_replayed": verified_frames,
                    "receipt": str(receipt_path),
                }
            )
        )
        return 0
    except Exception as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

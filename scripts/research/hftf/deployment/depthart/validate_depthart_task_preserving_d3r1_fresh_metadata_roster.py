#!/usr/bin/env python3
"""Independently replay and validate the D3R1 metadata-only roster."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.hftf.deployment.depthart.plan_depthart_task_preserving_d3r1_fresh_metadata_roster import (
    ROSTER_SCHEMA,
    load_json,
    plan,
    require,
    sha256_file,
    validate_bindings,
    write_bytes_exclusive,
)


RESULT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_fresh_metadata_roster_validation_v1"


def assert_exact_roster(generated: dict[str, Any], recomputed: dict[str, Any]) -> None:
    require(generated.get("schema") == ROSTER_SCHEMA, "roster schema drift")
    require(
        generated.get("status")
        == "D3R1_FRESH_METADATA_POOL_127_LOCKED_MEDIA_UNOPENED",
        "roster status drift",
    )
    require(generated == recomputed, "roster does not match independent replay")
    source = generated["source"]
    require(
        source["repository_commit"]
        == "7283761bf26c27570ec59a5dc0f8686fbff07726"
        and source["metadata_bytes"] == 127263
        and source["metadata_sha256"]
        == "06A0686866F186764ED0B92DE1A943529CEBD78B4AF5B671907C40BB2DCD13E1"
        and source["official_rows"] == 5047,
        "roster source drift",
    )
    invariants = generated["invariants"]
    require(
        invariants["pool_count"] == 127
        and invariants["unique_parent_count"] == 127
        and invariants["unique_session_count"] == 127,
        "roster cardinality drift",
    )
    require(
        invariants["workspace_excluded_identity_count"] == 490
        and invariants["concurrent_excluded_identity_count"] == 64
        and invariants["effective_excluded_identity_count"] == 554,
        "roster exclusion count drift",
    )
    require(
        invariants["selection_overlap_with_workspace_snapshot"] == 0
        and invariants["selection_overlap_with_concurrent_identity_firewalls"] == 0
        and invariants["selection_overlap_with_d3_predecessor_pool"] == 0,
        "roster identity firewall drift",
    )
    require(
        invariants["media_head_requests"] == 0
        and invariants["media_body_bytes_read"] == 0
        and invariants["truth_read"] is False
        and invariants["model_outputs_read"] is False
        and invariants["training"] is False
        and invariants["source_scope_registered"] is False
        and invariants["download_authorized"] is False,
        "roster authority drift",
    )
    pool = generated["selection"]["pool"]
    require(
        {
            key: generated["selection"][key]
            for key in (
                "training_row_count",
                "eligible_row_count",
                "eligible_unique_visit_count",
                "eligible_unique_session_count",
            )
        }
        == {
            "training_row_count": 4498,
            "eligible_row_count": 3724,
            "eligible_unique_visit_count": 1233,
            "eligible_unique_session_count": 3724,
        },
        "roster capacity drift",
    )
    require(
        [row["pool_order"] for row in pool] == list(range(1, 128)),
        "pool order drift",
    )
    require(
        all(
            set(row) == {
                "pool_order",
                "visit_id",
                "video_id",
                "fold",
                "role",
                "selection_sha256",
            }
            and row["fold"] == "Training"
            and row["role"] == "D3R1_METADATA_CANDIDATE_POOL_ONLY"
            and row["selection_sha256"]
            == hashlib.sha256(
                f"{row['visit_id']}:{row['video_id']}".encode("ascii")
            ).hexdigest().upper()
            for row in pool
        ),
        "pool row drift",
    )
    require(
        len({row["visit_id"] for row in pool}) == 127
        and len({row["video_id"] for row in pool}) == 127,
        "pool parent/session uniqueness drift",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--activation", required=True, type=Path)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol, _ = validate_bindings(args.protocol, args.activation)
    require(
        protocol["validator"]["sha256"] == sha256_file(Path(__file__)),
        "validator self SHA drift",
    )
    generated = load_json(args.roster)
    recomputed = plan(
        args.metadata,
        args.repo,
        args.protocol,
        args.activation,
    )
    assert_exact_roster(generated, recomputed)
    result = {
        "schema": RESULT_SCHEMA,
        "status": "D3R1_FRESH_METADATA_ROSTER_INDEPENDENT_REPLAY_PASS",
        "protocol_sha256": sha256_file(args.protocol),
        "activation_sha256": sha256_file(args.activation),
        "roster": {
            "path": str(args.roster.resolve()),
            "bytes": args.roster.stat().st_size,
            "sha256": sha256_file(args.roster),
        },
        "pool_count": 127,
        "unique_parent_count": 127,
        "unique_session_count": 127,
        "workspace_excluded_identity_count": generated["workspace_snapshot"][
            "matched_official_identity_count"
        ],
        "concurrent_excluded_identity_count": generated["invariants"][
            "concurrent_excluded_identity_count"
        ],
        "effective_excluded_identity_count": generated["invariants"][
            "effective_excluded_identity_count"
        ],
        "d3_predecessor_overlap_count": 0,
        "media_head_requests": 0,
        "media_body_bytes_read": 0,
        "truth_or_model_output_read": False,
        "training": False,
        "development_outcome_access": "NONE",
        "r2_cohort_access": "NONE",
        "source_scope_registered": False,
        "next_gate": "EXPLICIT_D3R1_SOURCE_SCOPE_REGISTRATION_FOR_EXACT_127_METADATA_ROSTER",
        "authority": "Independent metadata-only replay; no source-use, media, source truth, selection, training, Development, R2, performance, production or safety authority.",
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes_exclusive(args.output, encoded)
    print(
        json.dumps(
            {
                "status": result["status"],
                "pool_count": result["pool_count"],
                "result_bytes": len(encoded),
                "result_sha256": hashlib.sha256(encoded).hexdigest().upper(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

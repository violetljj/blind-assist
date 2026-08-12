#!/usr/bin/env python3
"""Materialize the four identity-locked fresh Attempt-03 held parents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(MODULE_DIR))

from materialize_ag_r2_f1_source_native_labels import (  # noqa: E402
    FORBIDDEN_TASK_FIELD_TOKENS,
    arrays_equal,
    build_payload,
    load_parent_frames,
    require,
    sha256_file,
    sha256_json,
    support_identity,
)

DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT03_FRESH_HELD_SOURCE_LOCK_R1_2026-08-11.json"
DEFAULT_SOURCE_CONTRACT = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK_2026-08-11.json"
DEFAULT_SOURCE_DIR = REPO_ROOT / "artifacts.local/downloads/ag-r2-f1-attempt03-fresh-held-r1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt03-fresh-held-labels-r1"


def run(lock_path: Path, source_contract_path: Path, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), f"output directory exists: {output_dir}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(
        lock["status"] == "ATTEMPT03_FRESH_HELD_R1_DOWNLOAD_AND_LABEL_MATERIALIZATION_AUTHORIZED",
        "Attempt-03 fresh-held lock status invalid",
    )
    source_contract_sha = sha256_file(source_contract_path)
    require(
        source_contract_sha == lock["source_and_label_contract"]["sha256"],
        "Attempt-03 source contract binding drift",
    )
    base_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source = base_contract["source_contract"]
    token = str(lock["assignment"]["token"])
    rows: list[dict[str, Any]] = []
    for role in ("CHECKPOINT_SELECTION", "TRAIN_CANARY"):
        for row in lock["roles"][role]:
            require(
                hashlib.sha256(f"{row.get('assignment_token', token)}:{row['parent_id']}".encode("utf-8")).hexdigest().upper()
                == row["assignment_sha256"],
                "Attempt-03 assignment SHA drift",
            )
            rows.append({**row, "role": role})
    require(
        len(rows) == 4
        and len({row["parent_id"] for row in rows}) == 4
        and {row["role"] for row in rows} == {"CHECKPOINT_SELECTION", "TRAIN_CANARY"},
        "Attempt-03 role roster drift",
    )

    all_frames = []
    source_receipts = []
    identities: dict[str, Any] = {}
    identity_receipts: dict[str, Any] = {}
    for row in rows:
        frames, receipt = load_parent_frames(row, source_dir, source, token)
        require(len(frames) == 3, "Attempt-03 selected frame count drift")
        identity, identity_receipt = support_identity(frames)
        all_frames.extend(frames)
        source_receipts.append(receipt)
        identities[row["parent_id"]] = identity
        identity_receipts[row["parent_id"]] = identity_receipt
    require(len(all_frames) == 12, "Attempt-03 total frame count drift")

    output_dir.mkdir(parents=True, exist_ok=False)
    frame_receipts = []
    for frame in all_frames:
        payload, report = build_payload(
            frame,
            identities[frame.parent_id],
            sha256_json(identity_receipts[frame.parent_id]),
            sha256_file(lock_path),
        )
        require(
            not any(token in key.lower() for key in payload for token in FORBIDDEN_TASK_FIELD_TOKENS),
            "task field leaked into Attempt-03 payload",
        )
        output_path = output_dir / f"{frame.frame_id}.npz"
        np.savez_compressed(output_path, **payload)
        with np.load(output_path, allow_pickle=False) as written:
            require(set(written.files) == set(payload), "Attempt-03 output field set drift")
            require(
                all(arrays_equal(np.asarray(written[key]), value) for key, value in payload.items()),
                "Attempt-03 output roundtrip drift",
            )
        frame_receipts.append(
            {
                "sample_id": frame.frame_id,
                "parent_id": frame.parent_id,
                "role": frame.role,
                "orientation": frame.orientation,
                "output": str(output_path.resolve()),
                "output_bytes": output_path.stat().st_size,
                "output_sha256": sha256_file(output_path),
                "source_archive_sha256": frame.source_archive_sha256,
                "accelerometer_sample_count": frame.accelerometer_sample_count,
                **report,
            }
        )

    parent_joint: dict[str, bool] = {}
    for parent in identities:
        parent_rows = [row for row in frame_receipts if row["parent_id"] == parent]
        parent_joint[parent] = (
            sum(int(row["metric_depth_valid_pixels"]) for row in parent_rows) > 0
            and any(bool(row["support_plane_valid"]) for row in parent_rows)
            and sum(int(row["support_valid_pixels"]) for row in parent_rows) > 0
            and sum(int(row["support_positive_pixels_ge_0_5"]) for row in parent_rows) > 0
            and sum(int(row["evidence_valid_pixels"]) for row in parent_rows) > 0
            and sum(int(row["boundary_seed_pixels"]) for row in parent_rows) > 0
        )
    role_orientations = {
        role: {row["orientation"] for row in frame_receipts if row["role"] == role}
        for role in ("CHECKPOINT_SELECTION", "TRAIN_CANARY")
    }
    gates = {
        "A3_S01_SOURCE_ARCHIVES_EXACT": (
            len(source_receipts) == 4
            and sum(int(row["source_archive_bytes"]) for row in source_receipts)
            == int(lock["assignment"]["expected_archive_bytes"])
        ),
        "A3_S02_ROLE_FRAME_AND_ORIENTATION_EXACT": (
            len(frame_receipts) == 12
            and all(len([row for row in frame_receipts if row["role"] == role]) == 6 for role in role_orientations)
            and all(value == {"LANDSCAPE_IDENTITY", "PORTRAIT_ROT90_CLOCKWISE"} for value in role_orientations.values())
        ),
        "A3_S03_ALL_FOUR_PARENTS_JOINT_PARENT_LEVEL": all(parent_joint.values()),
        "A3_S04_TASK_FIREWALL": True,
        "A3_S05_MODEL_METRICS_UNOPENED": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_attempt03_fresh_held_labels_v1",
        "status": (
            "ATTEMPT03_FRESH_HELD_LABEL_FRONTDOOR_PASS_EXECUTION_LOCK_REQUIRED"
            if passed
            else "ATTEMPT03_FRESH_HELD_LABEL_FRONTDOOR_FAIL_NO_MODEL"
        ),
        "passed": passed,
        "lock": str(lock_path.resolve()),
        "lock_sha256": sha256_file(lock_path),
        "source_contract_sha256": source_contract_sha,
        "source_receipts": source_receipts,
        "support_identity_receipts": identity_receipts,
        "frame_count": len(frame_receipts),
        "parent_count": len(identities),
        "joint_parent_count": sum(parent_joint.values()),
        "parent_joint": parent_joint,
        "role_orientations": {key: sorted(value) for key, value in role_orientations.items()},
        "gates": gates,
        "frames": frame_receipts,
        "decision": {
            "attempt03_model_or_optimizer_authorized": False,
            "selection_or_canary_model_metrics_opened": False,
            "next_action_if_pass": "Freeze the gravity/support-geometry Attempt-03 execution lock before evaluating any model or baseline metric on these labels.",
        },
    }
    result_path = output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--source-contract", type=Path, default=DEFAULT_SOURCE_CONTRACT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(
        args.lock.resolve(),
        args.source_contract.resolve(),
        args.source_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps({"status": result["status"], "passed": result["passed"], "parent_joint": result["parent_joint"], "gates": result["gates"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

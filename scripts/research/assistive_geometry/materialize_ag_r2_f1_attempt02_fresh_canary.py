#!/usr/bin/env python3
"""Materialize the two prelocked fresh Attempt-02 F1 canary parents."""

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

DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT02_ROLE_CORRECTION_AND_FRESH_CANARY_LOCK_2026-08-11.json"
DEFAULT_SOURCE_CONTRACT = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK_2026-08-11.json"
DEFAULT_SOURCE_DIR = REPO_ROOT / "artifacts.local/downloads/ag-r2-f1-attempt02-corrected-canary-r0"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt02-corrected-canary-labels-r0"
EXPECTED_SUPERSEDED_LOCK_SHA256 = "19ADAA5CD5F5B8922CC4EF6EEDA97AD68BFD6B2F55A553493B82FC3753371CBE"
EXPECTED_SELECTION_LABEL_RESULT_SHA256 = "7AFB581F30779A53CF1A54C15B06CFE3176D45D3CD5B92F52544283174188CCD"
TOKEN = "AG_R2_F1_ATTEMPT02_CORRECTED_CANARY_2026-08-11"


def run(lock_path: Path, source_contract_path: Path, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), f"output directory exists: {output_dir}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(
        lock["status"]
        == "ATTEMPT02_ROLE_CORRECTED_NEW_CANARY_DOWNLOAD_AND_LABEL_MATERIALIZATION_AUTHORIZED",
        "Attempt-02 corrected lock status invalid",
    )
    superseded = REPO_ROOT / lock["supersedes"]["path"]
    require(
        sha256_file(superseded) == EXPECTED_SUPERSEDED_LOCK_SHA256,
        "superseded Attempt-02 lock binding drift",
    )
    selection_binding = lock["corrected_roles"]["ATTEMPT02_CHECKPOINT_SELECTION"]["label_result"]
    selection_result = REPO_ROOT / selection_binding["path"]
    require(
        sha256_file(selection_result) == EXPECTED_SELECTION_LABEL_RESULT_SHA256
        and selection_binding["sha256"] == EXPECTED_SELECTION_LABEL_RESULT_SHA256,
        "corrected selection label binding drift",
    )
    base_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source = base_contract["source_contract"]
    fresh_rows = []
    canary = lock["corrected_roles"]["ATTEMPT02_TRAIN_CANARY"]
    require(canary["assignment_token"] == TOKEN, "fresh assignment token drift")
    for row in canary["parents"]:
        fresh_rows.append({**row, "role": "TRAIN_CANARY"})
        require(
            hashlib.sha256(f"{TOKEN}:{row['parent_id']}".encode("utf-8")).hexdigest().upper()
            == row["assignment_sha256"],
            "fresh assignment SHA drift",
        )

    all_frames = []
    source_receipts = []
    identities = {}
    identity_receipts = {}
    for row in fresh_rows:
        frames, receipt = load_parent_frames(row, source_dir, source, TOKEN)
        require(len(frames) == 3, "fresh selected frame count drift")
        identity, identity_receipt = support_identity(frames)
        all_frames.extend(frames)
        source_receipts.append(receipt)
        identities[row["parent_id"]] = identity
        identity_receipts[row["parent_id"]] = identity_receipt
    require(len(all_frames) == 6, "fresh total frame count drift")

    output_dir.mkdir(parents=True, exist_ok=False)
    frame_receipts = []
    joint_parents = set()
    for frame in all_frames:
        identity_receipt = identity_receipts[frame.parent_id]
        payload, report = build_payload(
            frame,
            identities[frame.parent_id],
            sha256_json(identity_receipt),
            sha256_file(lock_path),
        )
        require(
            not any(token in key.lower() for key in payload for token in FORBIDDEN_TASK_FIELD_TOKENS),
            "task field leaked into fresh payload",
        )
        output_path = output_dir / f"{frame.frame_id}.npz"
        np.savez_compressed(output_path, **payload)
        with np.load(output_path, allow_pickle=False) as written:
            require(set(written.files) == set(payload), "fresh output field set drift")
            require(
                all(arrays_equal(np.asarray(written[key]), value) for key, value in payload.items()),
                "fresh output roundtrip drift",
            )
        if (
            report["metric_depth_valid_pixels"] > 0
            and report["support_plane_valid"]
            and report["support_valid_pixels"] > 0
            and report["support_positive_pixels_ge_0_5"] > 0
            and report["evidence_valid_pixels"] > 0
            and report["boundary_seed_pixels"] > 0
        ):
            joint_parents.add(frame.parent_id)
        frame_receipts.append(
            {
                "sample_id": frame.frame_id,
                "parent_id": frame.parent_id,
                "role": "TRAIN_CANARY",
                "orientation": frame.orientation,
                "output": str(output_path.resolve()),
                "output_bytes": output_path.stat().st_size,
                "output_sha256": sha256_file(output_path),
                "source_archive_sha256": frame.source_archive_sha256,
                "accelerometer_sample_count": frame.accelerometer_sample_count,
                **report,
            }
        )
    parent_joint = {}
    for parent in identities:
        rows = [row for row in frame_receipts if row["parent_id"] == parent]
        parent_joint[parent] = (
            sum(int(row["metric_depth_valid_pixels"]) for row in rows) > 0
            and any(bool(row["support_plane_valid"]) for row in rows)
            and sum(int(row["support_valid_pixels"]) for row in rows) > 0
            and sum(int(row["support_positive_pixels_ge_0_5"]) for row in rows) > 0
            and sum(int(row["evidence_valid_pixels"]) for row in rows) > 0
            and sum(int(row["boundary_seed_pixels"]) for row in rows) > 0
        )
    gates = {
        "A2_S01_SOURCE_ARCHIVES_EXACT": (
            len(source_receipts) == 2
            and sum(int(row["source_archive_bytes"]) for row in source_receipts)
            == int(canary["expected_archive_bytes"])
        ),
        "A2_S02_FRAME_AND_ORIENTATION_EXACT": (
            len(frame_receipts) == 6
            and {row["orientation"] for row in frame_receipts}
            == {"LANDSCAPE_IDENTITY", "PORTRAIT_ROT90_CLOCKWISE"}
        ),
        "A2_S03_BOTH_PARENTS_JOINT_PARENT_LEVEL": all(parent_joint.values()),
        "A2_S04_TASK_FIREWALL": True,
        "A2_S05_CORRECTED_CANARY_IDENTITY_FRESH": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_attempt02_fresh_canary_labels_v1",
        "status": (
            "ATTEMPT02_FRESH_CANARY_LABEL_FRONTDOOR_PASS_MODEL_LOCK_REQUIRED"
            if passed
            else "ATTEMPT02_FRESH_CANARY_LABEL_FRONTDOOR_FAIL_NO_MODEL"
        ),
        "passed": passed,
        "lock": str(lock_path.resolve()),
        "lock_sha256": sha256_file(lock_path),
        "source_contract_sha256": sha256_file(source_contract_path),
        "source_receipts": source_receipts,
        "support_identity_receipts": identity_receipts,
        "frame_count": len(frame_receipts),
        "parent_count": len(identities),
        "joint_parent_count": sum(parent_joint.values()),
        "parent_joint": parent_joint,
        "gates": gates,
        "frames": frame_receipts,
        "decision": {
            "attempt02_model_or_optimizer_authorized": False,
            "fresh_canary_opened_for_metrics": False,
            "next_action_if_pass": "Freeze the component-split Attempt-02 model and execution lock using only FIT and the still-unopened Attempt-02 selection role.",
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "parent_joint": result["parent_joint"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

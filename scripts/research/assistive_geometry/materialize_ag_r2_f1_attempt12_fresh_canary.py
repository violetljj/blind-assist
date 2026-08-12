#!/usr/bin/env python3
"""Materialize the four identity-frozen Attempt-12 fresh canary parents."""

from __future__ import annotations

import argparse
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


DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT12_FRESH_CANARY_SOURCE_LOCK_2026-08-11.json"
SOURCE_CONTRACT = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK_2026-08-11.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt12-fresh-canary-labels-r0"


def locked_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in lock["parents"]:
        path = (REPO_ROOT / row["archive_path"]).resolve()
        require(path.is_file(), f"Attempt12 archive missing: {row['parent_id']}")
        require(path.stat().st_size == int(row["archive_bytes"]), "Attempt12 archive length drift")
        require(sha256_file(path) == row["archive_sha256"], "Attempt12 archive SHA drift")
        rows.append(
            {
                "parent_id": row["parent_id"],
                "family": "freiburg3",
                "role": "TRAIN_CANARY",
                "orientation": row["orientation"],
                "content_length": int(row["archive_bytes"]),
                "source_sha256": row["archive_sha256"],
                "_source_path": path,
            }
        )
    require(len(rows) == 4 and len({row["parent_id"] for row in rows}) == 4, "Attempt12 roster drift")
    return rows


def run(lock_path: Path, output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), f"output exists: {output_dir}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(lock["status"] == "ATTEMPT12_FRESH_CANARY_LABEL_MATERIALIZATION_AUTHORIZED_MODEL_EVALUATION_FORBIDDEN", "Attempt12 lock invalid")
    source = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))["source_contract"]
    rows = locked_rows(lock)
    all_frames = []
    source_receipts = []
    identities: dict[str, Any] = {}
    identity_receipts: dict[str, Any] = {}
    for row in rows:
        source_path = row.pop("_source_path")
        frames, receipt = load_parent_frames(row, source_path.parent, source, str(lock["selection_token"]))
        require(len(frames) == 3, "Attempt12 selected frame count drift")
        require(receipt["source_archive_sha256"] == row["source_sha256"], "Attempt12 source receipt drift")
        identity, identity_receipt = support_identity(frames)
        all_frames.extend(frames)
        source_receipts.append(receipt)
        identities[row["parent_id"]] = identity
        identity_receipts[row["parent_id"]] = identity_receipt
        print(json.dumps({"loaded_parent": row["parent_id"], "support_identity": identity_receipt["status"]}), flush=True)
    require(len(all_frames) == 12, "Attempt12 total frame count drift")
    output_dir.mkdir(parents=True, exist_ok=False)
    frame_receipts = []
    for frame in all_frames:
        payload, report = build_payload(
            frame,
            identities[frame.parent_id],
            sha256_json(identity_receipts[frame.parent_id]),
            sha256_file(lock_path),
        )
        require(not any(token in key.lower() for key in payload for token in FORBIDDEN_TASK_FIELD_TOKENS), "task field leaked into Attempt12 payload")
        output_path = output_dir / f"{frame.frame_id}.npz"
        np.savez_compressed(output_path, **payload)
        with np.load(output_path, allow_pickle=False) as written:
            require(set(written.files) == set(payload), "Attempt12 output field drift")
            require(all(arrays_equal(np.asarray(written[key]), value) for key, value in payload.items()), "Attempt12 output roundtrip drift")
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
        members = [row for row in frame_receipts if row["parent_id"] == parent]
        parent_joint[parent] = (
            sum(int(row["metric_depth_valid_pixels"]) for row in members) > 0
            and any(bool(row["support_plane_valid"]) for row in members)
            and sum(int(row["support_valid_pixels"]) for row in members) > 0
            and sum(int(row["support_positive_pixels_ge_0_5"]) for row in members) > 0
            and sum(int(row["evidence_valid_pixels"]) for row in members) > 0
            and sum(int(row["boundary_seed_pixels"]) for row in members) > 0
        )
    orientations = {row["orientation"] for row in frame_receipts}
    gates = {
        "A12_S01_ARCHIVES_EXACT": len(source_receipts) == 4 and all(len(row["source_archive_sha256"]) == 64 for row in source_receipts),
        "A12_S02_FRAME_PARENT_EXACT": len(frame_receipts) == 12 and all(sum(row["parent_id"] == parent for row in frame_receipts) == 3 for parent in identities),
        "A12_S03_BOTH_ORIENTATIONS": orientations == {"LANDSCAPE_IDENTITY", "PORTRAIT_ROT90_CLOCKWISE"},
        "A12_S04_ALL_PARENTS_JOINT": all(parent_joint.values()),
        "A12_S05_MODEL_METRICS_UNOPENED": True,
        "A12_S06_TASK_FIREWALL": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_f1_attempt12_fresh_canary_labels_result_v1",
        "status": "ATTEMPT12_FRESH_CANARY_LABELS_PASS_FULL_FACTOR_RETRAIN_REQUIRED" if passed else "ATTEMPT12_FRESH_CANARY_LABELS_FAIL_NO_MODEL_EVALUATION",
        "passed": passed,
        "lock": {"path": str(lock_path.resolve()), "sha256": sha256_file(lock_path)},
        "source_contract": {"path": str(SOURCE_CONTRACT.resolve()), "sha256": sha256_file(SOURCE_CONTRACT)},
        "source_receipts": source_receipts,
        "support_identity_receipts": identity_receipts,
        "frame_count": len(frame_receipts),
        "parent_count": len(identities),
        "parent_joint": parent_joint,
        "gates": gates,
        "frames": frame_receipts,
        "decision": {"attempt12_model_metrics_opened": False, "attempt12_optimizer_use_authorized": False, "next_action_if_pass": "Retrain complete factors on consumed parents only, freeze exact execution, then open Attempt12 once."},
    }
    with (output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.lock.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": result["status"], "passed": result["passed"], "parent_joint": result["parent_joint"], "gates": result["gates"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

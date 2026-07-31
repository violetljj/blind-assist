#!/usr/bin/env python3
"""Transcribe unchanged R1 calibration labels into R2 with honest timing metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import uuid
from pathlib import Path

from .freeze_d0a1_pilot import DEFAULT_LOCK, load_json, load_lock, repo_path
from .freeze_input_universe import canonical_bytes, sha256_file


PREDECESSOR_REVIEW = Path(
    "artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a1-r1/primary-review.json"
)
PRIMARY_REVIEW_NAME = "primary-review.json"


class PrimaryReviewTranscriptionError(ValueError):
    """Fail-closed timestamp-repair transcription error."""


def write_transcription(
    *,
    repo_root: Path,
    lock_path: Path,
    output_root: Path,
    predecessor_path: Path,
) -> Path:
    destination = output_root / PRIMARY_REVIEW_NAME
    if destination.exists():
        raise PrimaryReviewTranscriptionError("current primary review already exists; refusing overwrite")
    lock = load_lock(repo_root, lock_path)
    manifest_path = output_root / "pilot-input-manifest.json"
    input_validation_path = output_root / "pilot-input-validation.json"
    if not manifest_path.is_file() or not input_validation_path.is_file():
        raise PrimaryReviewTranscriptionError("current pilot inputs are not validated")
    input_validation = load_json(input_validation_path, where="pilot input validation")
    if input_validation.get("status") != "VALID":
        raise PrimaryReviewTranscriptionError("current pilot input validation is not VALID")
    predecessor = load_json(predecessor_path, where="predecessor primary review")
    if (
        predecessor.get("evidence_instance") != "CENTRAL_OBSTRUCTION_D0_A1_LABELABILITY_PILOT_R1"
        or predecessor.get("candidate_output_visible") is not False
        or predecessor.get("source_only_view") is not True
        or not isinstance(predecessor.get("clip_reviews"), list)
    ):
        raise PrimaryReviewTranscriptionError("predecessor primary review contract mismatch")
    prompt_path = repo_path(
        repo_root,
        lock["bindings"]["review_prompt"]["path"],
        where="review prompt",
    )
    now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    review = {
        "schema_version": "blindassist.central_obstruction_d0a1_primary_review.v2",
        "protocol_id": lock["protocol_id"],
        "phase": "D0-A1",
        "evidence_instance": lock["evidence_instance"],
        "review_id": "D0-A1-R2-PRIMARY-TIMESTAMP-REPAIR-TRANSCRIPTION",
        "reviewer_id": "codex-current-task-primary-r2",
        "reviewer_type": "ai_model",
        "review_context": "PRIMARY_CURRENT_TASK_NON_ISOLATED_TIMESTAMP_REPAIR_TRANSCRIPTION",
        "isolated_context": False,
        "source_only_view": True,
        "candidate_output_visible": False,
        "prior_review_visible": True,
        "other_review_visible_before_submission": True,
        "labels_generated_before_r1_lock": False,
        "pilot_input_manifest_sha256": sha256_file(manifest_path),
        "prompt_sha256": sha256_file(prompt_path),
        "submitted_at_utc": now,
        "predecessor_review": {
            "path": predecessor_path.resolve().relative_to(repo_root.resolve()).as_posix(),
            "sha256": sha256_file(predecessor_path),
            "status": "INVALID_REVIEW_TIMESTAMP_ORDER",
        },
        "transcription_reason": "R1 labels are copied unchanged because only the declared submission timestamp was invalid.",
        "label_changes_from_predecessor": 0,
        "clip_reviews": predecessor["clip_reviews"],
        "claim_ceiling": "NON_ISOLATED_PRIMARY_CALIBRATION_TRANSCRIPTION_ONLY; no agreement, consensus, readiness, D0-A2, or D0-B authority.",
    }
    temporary = output_root / f".{PRIMARY_REVIEW_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(canonical_bytes(review))
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--predecessor-review", type=Path, default=PREDECESSOR_REVIEW)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = load_lock(repo_root, lock_path)
    output_root = args.output_root or repo_path(repo_root, lock["output_root"], where="output_root")
    predecessor_path = args.predecessor_review
    if not predecessor_path.is_absolute():
        predecessor_path = repo_root / predecessor_path
    path = write_transcription(
        repo_root=repo_root,
        lock_path=lock_path,
        output_root=output_root.resolve(),
        predecessor_path=predecessor_path.resolve(),
    )
    print(
        json.dumps(
            {
                "status": "PRIMARY_TIMESTAMP_REPAIR_TRANSCRIBED",
                "review": str(path),
                "sha256": sha256_file(path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

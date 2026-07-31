#!/usr/bin/env python3
"""Recompute and validate the stored D0-A1 final readiness evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .finalize_d0a1_adjudication import (
    ADJUDICATION_REVIEW_NAME,
    CANONICAL_EVENTS_NAME,
    CANONICAL_LABELS_NAME,
    FINAL_READINESS_NAME,
    AdjudicationValidationError,
    finalize_adjudication,
)
from .freeze_d0a1_pilot import DEFAULT_LOCK, load_json, load_lock, repo_path
from .freeze_input_universe import canonical_bytes, sha256_file


VALIDATION_NAME = "d0a1-final-validation.json"


class FinalReadinessValidationError(ValueError):
    """Fail-closed final-readiness validation error."""


def _without_dynamic_fields(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for key in [
        "validated_at_utc",
        "adjudication_review_sha256",
        "canonical_labels_sha256",
        "canonical_events_sha256",
    ]:
        result.pop(key, None)
    return result


def validate_final(
    *, repo_root: Path, lock_path: Path, output_root: Path
) -> dict[str, Any]:
    review_path = output_root / ADJUDICATION_REVIEW_NAME
    labels_path = output_root / CANONICAL_LABELS_NAME
    events_path = output_root / CANONICAL_EVENTS_NAME
    readiness_path = output_root / FINAL_READINESS_NAME
    if any(not path.is_file() for path in [review_path, labels_path, events_path, readiness_path]):
        raise FinalReadinessValidationError("stored final D0-A1 outputs are missing")
    stored = load_json(readiness_path, where="stored final readiness")
    try:
        recomputed, labels, events = finalize_adjudication(
            repo_root=repo_root,
            lock_path=lock_path,
            output_root=output_root,
            review_path=review_path,
        )
    except AdjudicationValidationError as error:
        raise FinalReadinessValidationError("adjudication recomputation failed") from error
    expected_labels = b"".join(canonical_bytes(row) for row in labels)
    expected_events = b"".join(canonical_bytes(row) for row in events)
    if labels_path.read_bytes() != expected_labels or events_path.read_bytes() != expected_events:
        raise FinalReadinessValidationError("canonical labels or events differ from recomputation")
    if _without_dynamic_fields(stored) != _without_dynamic_fields(recomputed):
        raise FinalReadinessValidationError("stored final readiness differs from recomputation")
    if (
        stored.get("adjudication_review_sha256") != sha256_file(review_path)
        or stored.get("canonical_labels_sha256") != sha256_file(labels_path)
        or stored.get("canonical_events_sha256") != sha256_file(events_path)
        or stored.get("status") != "VALID"
        or stored.get("readiness_evaluated") is not True
        or stored.get("d0a2_production_labeling_authorized") is not False
        or stored.get("d0b_authorized") is not False
        or stored.get("terminal") != "AGENT_LABEL_PROTOCOL_NOT_RELIABLE"
    ):
        raise FinalReadinessValidationError("stored hash, authority, or terminal binding mismatch")
    return {
        "schema_version": "blindassist.central_obstruction_d0a1_final_validation.v1",
        "protocol_id": stored["protocol_id"],
        "phase": "D0-A1",
        "evidence_instance": stored["evidence_instance"],
        "status": "VALID",
        "terminal": stored["terminal"],
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "lock_sha256": sha256_file(lock_path),
        "adjudication_review_sha256": sha256_file(review_path),
        "canonical_labels_sha256": sha256_file(labels_path),
        "canonical_events_sha256": sha256_file(events_path),
        "final_readiness_sha256": sha256_file(readiness_path),
        "recomputed_observation_count": len(labels),
        "recomputed_parent_event_count": len(events),
        "recomputed_threshold_checks": stored["threshold_checks"],
        "d0a2_production_labeling_authorized": False,
        "d0b_authorized": False,
        "errors": [],
    }


def write_validation(*, repo_root: Path, lock_path: Path, output_root: Path) -> Path:
    validation_path = output_root / VALIDATION_NAME
    if validation_path.exists():
        raise FinalReadinessValidationError("final validation already exists; refusing overwrite")
    result = validate_final(repo_root=repo_root, lock_path=lock_path, output_root=output_root)
    tmp = output_root / f".{VALIDATION_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_bytes(canonical_bytes(result))
        os.replace(tmp, validation_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return validation_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = load_lock(repo_root, lock_path)
    output_root = args.output_root or repo_path(repo_root, lock["output_root"], where="output_root")
    validation_path = write_validation(
        repo_root=repo_root,
        lock_path=lock_path,
        output_root=output_root.resolve(),
    )
    result = load_json(validation_path, where="final validation")
    print(
        json.dumps(
            {
                "status": result["status"],
                "terminal": result["terminal"],
                "validation": str(validation_path),
                "validation_sha256": sha256_file(validation_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "EVAL_VALIDITY_R0"
COHORT_SCHEMA = "blindassist.eval_validity_r0.cohort.v1"
EXCLUSION_SCHEMA = "blindassist.eval_validity_r0.exclusion_registry.v1"
ACTION_REVIEW_SCHEMA = "blindassist.eval_validity_r0.action_review.v2"
P0_ANCHOR_AGREEMENT_SCHEMA = "blindassist.eval_validity_r0.p0_anchor_agreement.v1"
P1_ACTION_REVIEW_SCHEMA = "blindassist.eval_validity_r0.p1_action_review.v1"
P1_ACTION_FACTS_SCHEMA = "blindassist.eval_validity_r0.p1_action_facts.v1"
FULL_EVENT_FACTS_SCHEMA = "blindassist.eval_validity_r0.full_event_facts.v1"
SCENE_FRAME_SCHEMA = "blindassist.eval_validity_r0.scene_frame.v1"
TRACE_SCHEMA = "blindassist.eval_validity_r0.feedback_trace.v1"
TRACE_MANIFEST_SCHEMA = "blindassist.eval_validity_r0.trace_manifest.v1"
PHASH_REVIEW_PACKET_SCHEMA = "blindassist.eval_validity_r0.phash_rgb_packet.v1"
PHASH_PRIVATE_MAP_SCHEMA = "blindassist.eval_validity_r0.phash_private_review_map.v1"
PHASH_REVIEW_SUBMISSION_SCHEMA = "blindassist.eval_validity_r0.phash_manual_review.v1"
PHASH_RESOLUTION_SCHEMA = "blindassist.eval_validity_r0.phash_manual_resolution.v1"
ADMISSION_RECONCILIATION_SCHEMA = "blindassist.eval_validity_r0.data_admission_reconciliation.v1"

# The second status is available only from the RGB-only, two-reviewer pHash
# resolution route.  It does not weaken any other data-admission check.
ADMISSION_PASSED_STATUSES = {
    "EVAL_VALIDITY_DATA_ADMISSION_PASSED",
    "EVAL_VALIDITY_DATA_ADMISSION_PASSED_AFTER_PHASH_MANUAL_REVIEW",
}

ARMS = ("current_yolo", "truth_box", "truth_mask", "synthetic_oracle")
REPRESENTATION_ARMS = ARMS[:3]
POSITIVE_BUCKETS = {"blocking_obstacle_positive", "boundary_level_change_positive"}
NEGATIVE_BUCKETS = {"parallel_curb_negative", "normal_walkable_negative"}
BUCKETS = POSITIVE_BUCKETS | NEGATIVE_BUCKETS
THREE_STATE = {"YES", "NO", "UNKNOWN"}
KNOWNNESS = {"KNOWN", "UNKNOWN"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: JSONL is empty")
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_finite_nonnegative(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{where}: expected number")
    number = float(value)
    if number < 0 or number == float("inf") or number != number:
        raise ValueError(f"{where}: expected finite non-negative number")
    return number

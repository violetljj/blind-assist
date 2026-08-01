from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


PROTOCOL_ID = "RISKSEG_ACT_A0"
CONDITIONS = ("CURRENT_ONLY", "CAUSAL_HISTORY", "HINDSIGHT_REFERENCE")
PASS_IDS = (
    "CURRENT_A",
    "CURRENT_B",
    "CAUSAL_A",
    "CAUSAL_B",
    "HINDSIGHT_A",
    "HINDSIGHT_B",
)
PASS_CONDITION = {
    "CURRENT_A": "CURRENT_ONLY",
    "CURRENT_B": "CURRENT_ONLY",
    "CAUSAL_A": "CAUSAL_HISTORY",
    "CAUSAL_B": "CAUSAL_HISTORY",
    "HINDSIGHT_A": "HINDSIGHT_REFERENCE",
    "HINDSIGHT_B": "HINDSIGHT_REFERENCE",
}
CONDITION_PASSES = {
    condition: tuple(pass_id for pass_id in PASS_IDS if PASS_CONDITION[pass_id] == condition)
    for condition in CONDITIONS
}

CELL_IDS = (
    "FAR_LEFT",
    "FAR_CENTER",
    "FAR_RIGHT",
    "NEAR_LEFT",
    "NEAR_CENTER",
    "NEAR_RIGHT",
)
INTRUSION_LABELS = {"INTRUDING", "NON_INTRUDING", "UNKNOWN"}
BOUNDARY_LABELS = {
    "TRANSVERSE_CROSSING",
    "PARALLEL_BOUNDARY",
    "AMBIGUOUS",
    "NOT_APPLICABLE",
}
TERNARY_LABELS = {"YES", "NO", "UNKNOWN"}
KNOWNNESS_LABELS = {"KNOWN", "UNKNOWN"}
HAZARD_LABELS = {
    "BLOCKING_OBSTACLE",
    "BOUNDARY_LEVEL_CHANGE",
    "NONE_IN_SCOPE",
    "UNRESOLVED",
}
QUALITY_STATES = {
    "STABLE",
    "TURNING",
    "BLURRED",
    "DARK",
    "OCCLUDED",
    "MULTIPLE_PLAUSIBLE_ROUTES",
    "OTHER_NOT_EVALUABLE",
}
NON_ACTIONABLE_REASONS = {
    "CLEAR_FORWARD",
    "LATERAL_OR_PARALLEL",
    "TOO_EARLY_OR_NOT_CURRENTLY_ALERTABLE",
    "PASSED_CLEAR",
    "NO_HAZARD_IN_SCOPE",
}
DERIVED_LABELS = (
    "ACTIONABLE_NOW",
    "NON_ACTIONABLE_NOW",
    "ABSTAIN_NOT_EVALUABLE",
)

FORBIDDEN_PUBLIC_KEYS = {
    "parent_event_id",
    "source_session_id",
    "bucket",
    "positive",
    "alertable_interval_frames",
    "passed_interval_frames",
    "oracle_mask",
    "oracle_mask_path",
    "truth",
    "truth_ledger",
    "yolo",
    "segmentation",
    "model_output",
    "p0_score",
}


class A0Error(ValueError):
    """Raised when an A0 contract or review invariant is violated."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise A0Error(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(canonical_bytes(row))


def round_half_up(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def anchor_indices(frame_count: int) -> tuple[int, int, int, int]:
    if frame_count < 21:
        raise A0Error("an event must contain at least 21 frames")
    span = Decimal(frame_count - 21)
    anchors = tuple(
        round_half_up(Decimal(10) + Decimal(k) * span / Decimal(3))
        for k in range(4)
    )
    if len(set(anchors)) != 4:
        raise A0Error(f"anchor formula produced duplicate anchors for N={frame_count}")
    if anchors[0] < 10 or anchors[-1] > frame_count - 11:
        raise A0Error(f"anchor window is incomplete for N={frame_count}: {anchors}")
    return anchors  # type: ignore[return-value]


def condition_offsets(condition: str) -> tuple[int, ...]:
    if condition == "CURRENT_ONLY":
        return (0,)
    if condition == "CAUSAL_HISTORY":
        return tuple(range(-10, 1))
    if condition == "HINDSIGHT_REFERENCE":
        return tuple(range(-10, 11))
    raise A0Error(f"unknown information condition: {condition}")


def derive_actionability(row: dict[str, Any]) -> str:
    cells = row["intrusion_cells"]
    if (
        row["knownness"] == "UNKNOWN"
        or row["alertable"] == "UNKNOWN"
        or row["passed"] == "UNKNOWN"
        or all(cells[cell] == "UNKNOWN" for cell in CELL_IDS)
        or row["hazard_aux"] == "UNRESOLVED"
    ):
        return "ABSTAIN_NOT_EVALUABLE"
    if (
        row["knownness"] == "KNOWN"
        and row["alertable"] == "YES"
        and row["passed"] == "NO"
        and any(cells[cell] == "INTRUDING" for cell in CELL_IDS)
    ):
        return "ACTIONABLE_NOW"
    return "NON_ACTIONABLE_NOW"


def unknown_to_non_actionable_violation(row: dict[str, Any]) -> bool:
    """Detect a raw UNKNOWN-to-negative conversion hidden by a supplied label."""
    required_unknown = (
        row["knownness"] == "UNKNOWN"
        or row["alertable"] == "UNKNOWN"
        or row["passed"] == "UNKNOWN"
        or all(row["intrusion_cells"][cell] == "UNKNOWN" for cell in CELL_IDS)
        or row["hazard_aux"] == "UNRESOLVED"
    )
    supplied = row.get("derived_actionability")
    return required_unknown and supplied == "NON_ACTIONABLE_NOW"


def validate_review_row(row: dict[str, Any], *, where: str) -> dict[str, Any]:
    required = {
        "review_item_id",
        "hazard_aux",
        "intrusion_cells",
        "boundary_relation",
        "alertable",
        "passed",
        "knownness",
        "non_actionable_reason",
        "quality_state",
        "rationale_code",
    }
    missing = required - row.keys()
    if missing:
        raise A0Error(f"{where}: missing fields {sorted(missing)}")
    if not isinstance(row["review_item_id"], str) or not row["review_item_id"]:
        raise A0Error(f"{where}: invalid review_item_id")
    if row["hazard_aux"] not in HAZARD_LABELS:
        raise A0Error(f"{where}: invalid hazard_aux")
    cells = row["intrusion_cells"]
    if not isinstance(cells, dict) or set(cells) != set(CELL_IDS):
        raise A0Error(f"{where}: intrusion_cells must contain exactly the six frozen cells")
    if any(label not in INTRUSION_LABELS for label in cells.values()):
        raise A0Error(f"{where}: invalid intrusion cell label")
    if row["boundary_relation"] not in BOUNDARY_LABELS:
        raise A0Error(f"{where}: invalid boundary_relation")
    if row["alertable"] not in TERNARY_LABELS or row["passed"] not in TERNARY_LABELS:
        raise A0Error(f"{where}: invalid alertable/passed label")
    if row["knownness"] not in KNOWNNESS_LABELS:
        raise A0Error(f"{where}: invalid knownness")
    if row["quality_state"] not in QUALITY_STATES:
        raise A0Error(f"{where}: invalid quality_state")
    if not isinstance(row["rationale_code"], str) or not row["rationale_code"].strip():
        raise A0Error(f"{where}: rationale_code must be non-empty")
    derived = derive_actionability(row)
    reason = row["non_actionable_reason"]
    if derived == "NON_ACTIONABLE_NOW":
        if reason not in NON_ACTIONABLE_REASONS:
            raise A0Error(f"{where}: a valid non_actionable_reason is required")
    elif reason is not None:
        raise A0Error(f"{where}: non_actionable_reason must be null unless derived NON_ACTIONABLE")
    if "derived_actionability" in row and row["derived_actionability"] != derived:
        raise A0Error(f"{where}: supplied derived_actionability contradicts frozen derivation")
    normalized = dict(row)
    normalized["derived_actionability"] = derived
    return normalized


def assert_no_forbidden_public_fields(value: Any, *, where: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            absence_declaration = (
                isinstance(child, bool)
                and child is True
                and (
                    lowered.endswith("_absent")
                    or lowered.endswith("_not_seen")
                    or lowered.endswith("_hidden")
                )
            )
            if (
                lowered in FORBIDDEN_PUBLIC_KEYS
                or (
                    ("oracle" in lowered or "model_output" in lowered)
                    and not absence_declaration
                )
            ):
                raise A0Error(f"{where}: forbidden public key {key}")
            assert_no_forbidden_public_fields(child, where=f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_forbidden_public_fields(child, where=f"{where}[{index}]")

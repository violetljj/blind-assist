from __future__ import annotations

"""Freeze an output-blind source-screening cohort for EVAL-VALIDITY R0.

This consumes source-mask discovery only.  Its strata are *not* event labels:
the downstream P0 reviews receive opaque RGB-only packets and are the first
authority for actionability, reminder and clearance facts.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from .common import EXCLUSION_SCHEMA, PROTOCOL_ID, read_json, sha256_file


SCHEMA = "blindassist.eval_validity_r0.screening_cohort.v1"
WINDOW_FRAME_COUNT = 60
ANCHOR_OFFSETS = [8, 20, 36, 52]
STRATA = (
    ("strict_normal_candidate", "strict_normal_walkable_source_mask_only", 12, "normal"),
    ("center_obstacle_candidate", "center_obstacle", 12, "hazard"),
    ("boundary_candidate", "step_curb", 24, "hazard"),
)


class ScreeningCohortError(ValueError):
    """Raised when the source-only screening pool cannot be frozen safely."""


def _candidate_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["session_id"]),
        int(row["recommended_start_frame"]),
        str(row["camera"]),
        str(row["lens"]),
    )


def _validate_candidate(row: dict[str, Any], *, profile: str, excluded: set[str]) -> None:
    required_text = ("session_id", "official_split", "camera", "lens", "selection_profile")
    if any(not isinstance(row.get(field), str) or not row[field] for field in required_text):
        raise ScreeningCohortError("source candidate has missing identity")
    if row["official_split"] != "train":
        raise ScreeningCohortError("source candidate must remain in the declared official train split")
    if row["selection_profile"] != profile:
        raise ScreeningCohortError("source candidate profile mismatch")
    if row["session_id"] in excluded:
        raise ScreeningCohortError("source candidate belongs to the frozen exclusion registry")
    start = row.get("recommended_start_frame")
    if not isinstance(start, int) or start < 0:
        raise ScreeningCohortError("source candidate has invalid recommended start")
    matching_frames = row.get("geometry_matching_source_frames")
    if (
        not isinstance(matching_frames, list)
        or not matching_frames
        or any(not isinstance(value, int) or value < 0 for value in matching_frames)
    ):
        raise ScreeningCohortError("source candidate has no usable source-screening reference frame")


def freeze_screening_cohort(
    hazard_selection: dict[str, Any], normal_discovery: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    if registry.get("schema_version") != EXCLUSION_SCHEMA or registry.get("protocol_id") != PROTOCOL_ID:
        raise ScreeningCohortError("exclusion registry schema/protocol mismatch")
    excluded_rows = registry.get("excluded_source_sessions")
    if not isinstance(excluded_rows, list) or not all(isinstance(value, str) and value for value in excluded_rows):
        raise ScreeningCohortError("exclusion registry is invalid")
    if hazard_selection.get("protocol_id") != PROTOCOL_ID or hazard_selection.get("status") != "SOURCE_MASK_DISCOVERY_ONLY_NOT_EVENT_TRUTH":
        raise ScreeningCohortError("hazard discovery has wrong protocol/status")
    if normal_discovery.get("protocol_id") != PROTOCOL_ID or normal_discovery.get("status") != "SOURCE_MASK_DISCOVERY_ONLY_NOT_EVENT_TRUTH":
        raise ScreeningCohortError("normal discovery has wrong protocol/status")
    hazard_rows = hazard_selection.get("eligible_candidates")
    normal_rows = normal_discovery.get("candidates")
    if not isinstance(hazard_rows, list) or not isinstance(normal_rows, list):
        raise ScreeningCohortError("discovery candidates are missing")

    source_rows = {"hazard": hazard_rows, "normal": normal_rows}
    excluded = set(excluded_rows)
    used_sessions: set[str] = set()
    items: list[dict[str, Any]] = []
    stratum_counts: dict[str, int] = {}
    for stratum, profile, required_count, source_kind in STRATA:
        candidates = [row for row in source_rows[source_kind] if isinstance(row, dict) and row.get("selection_profile") == profile]
        for row in candidates:
            _validate_candidate(row, profile=profile, excluded=excluded)
        candidates.sort(key=_candidate_key)
        selected: list[dict[str, Any]] = []
        for row in candidates:
            if row["session_id"] in used_sessions:
                continue
            selected.append(row)
            used_sessions.add(row["session_id"])
            if len(selected) == required_count:
                break
        if len(selected) != required_count:
            raise ScreeningCohortError(f"{stratum}: requires {required_count} disjoint source sessions, found {len(selected)}")
        stratum_counts[stratum] = len(selected)
        for row in selected:
            ordinal = len(items) + 1
            matching_frames = row["geometry_matching_source_frames"]
            items.append({
                "screening_event_id": f"evr0-screen-{ordinal:03d}",
                "source_session_id": row["session_id"],
                "official_split": row["official_split"],
                "camera": row["camera"],
                "lens": row["lens"],
                "source_selection_profile": profile,
                "screening_stratum": stratum,
                "source_window": {
                    "start_frame": row["recommended_start_frame"],
                    "frame_count": WINDOW_FRAME_COUNT,
                    "required_contiguous_native_frames": WINDOW_FRAME_COUNT,
                    "p0_anchor_offsets": ANCHOR_OFFSETS,
                    "source_screening_reference_frame": matching_frames[len(matching_frames) // 2],
                },
                "event_bucket": None,
                "event_facts": None,
                "candidate_outputs_opened": False,
                "source_screening_limit": "Source-mask screening only; this is not a positive/negative, reminder, clearance or safety label.",
            })
    if len(items) != len(used_sessions) or len(items) != 48:
        raise ScreeningCohortError("screening cohort must contain exactly one item per 48 source sessions")
    return {
        "schema_version": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "OUTPUT_BLIND_SCREENING_COHORT_FROZEN",
        "purpose": "Freeze a source-only input universe for continuous native RGB/mask admission and two isolated P0 reviews. It is not the final event-truth cohort.",
        "candidate_outputs_opened": False,
        "final_event_facts_frozen": False,
        "reviewer_visible_fields": ["opaque review item ID", "causal RGB frame order", "frame ordinal"],
        "reviewer_forbidden_fields": ["source session", "screening stratum", "source mask", "event bucket", "YOLO/model/oracle output", "other review"],
        "screening_stratum_counts": stratum_counts,
        "source_session_count": len(used_sessions),
        "items": items,
        "next_required_gate": "Materialize exactly the declared continuous native RGB and source-mask frames, complete full data contamination audit, then submit two isolated opaque P0 reviews before any output trace exists.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazard-selection", type=Path, required=True)
    parser.add_argument("--normal-discovery", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = freeze_screening_cohort(
        read_json(args.hazard_selection), read_json(args.normal_discovery), read_json(args.exclusion_registry)
    )
    result["input_sha256"] = {
        "hazard_selection": sha256_file(args.hazard_selection),
        "normal_discovery": sha256_file(args.normal_discovery),
        "exclusion_registry": sha256_file(args.exclusion_registry),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"screening_events={len(result['items'])} source_sessions={result['source_session_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

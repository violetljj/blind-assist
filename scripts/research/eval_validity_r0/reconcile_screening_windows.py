from __future__ import annotations

"""Freeze feasible continuous source windows before any pixel or output access.

This is an output-blind *metadata admission* correction: it can only move a
window within the same source session, must preserve the source-mask screening
reference frame, and writes a new immutable cohort rather than mutating a
failed preflight cohort.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .common import PROTOCOL_ID, read_json, sha256_file, sha256_json
from .freeze_screening_cohort import SCHEMA as SCREENING_COHORT_SCHEMA


SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_sanpo_sequence_evalset import GCS_PREFIX, frame_number, list_gcs_objects  # noqa: E402


class WindowReconciliationError(ValueError):
    """Raised when metadata cannot support a continuous output-blind window."""


def _png_frames(objects: list[dict[str, Any]], *, where: str) -> set[int]:
    values: set[int] = set()
    for item in objects:
        name = item.get("name")
        if not isinstance(name, str) or not name.endswith(".png"):
            continue
        value = frame_number(name)
        if value in values:
            raise WindowReconciliationError(f"{where}: duplicate source frame")
        values.add(value)
    return values


def _best_start(
    available: set[int], *, count: int, reference: int, preferred_start: int, target_anchor_offset: int
) -> int | None:
    """Keep the frozen start if viable; otherwise choose the smallest metadata-only repair."""
    if count < 20 or reference not in available:
        return None
    preferred_frames = range(preferred_start, preferred_start + count)
    if reference in preferred_frames and all(frame in available for frame in preferred_frames):
        return preferred_start
    choices: list[int] = []
    for start in range(max(0, reference - count + 1), reference + 1):
        if all(frame in available for frame in range(start, start + count)):
            choices.append(start)
    if not choices:
        return None
    return min(choices, key=lambda start: (abs((start + target_anchor_offset) - reference), start))


def reconcile_screening_windows(
    cohort: dict[str, Any], *, list_objects: Callable[[str], list[dict[str, Any]]]
) -> dict[str, Any]:
    if cohort.get("schema_version") != SCREENING_COHORT_SCHEMA or cohort.get("protocol_id") != PROTOCOL_ID:
        raise WindowReconciliationError("screening cohort schema/protocol mismatch")
    if cohort.get("status") != "OUTPUT_BLIND_SCREENING_COHORT_FROZEN":
        raise WindowReconciliationError("only the original output-blind screening cohort may be reconciled")
    if cohort.get("candidate_outputs_opened") is not False or cohort.get("final_event_facts_frozen") is not False:
        raise WindowReconciliationError("cohort output/event-fact state is invalid")
    items = cohort.get("items")
    if not isinstance(items, list) or len(items) != 48:
        raise WindowReconciliationError("screening cohort coverage is invalid")
    result = copy.deepcopy(cohort)
    resolved: list[dict[str, Any]] = []
    for item in result["items"]:
        if not isinstance(item, dict):
            raise WindowReconciliationError("screening cohort item is invalid")
        window = item.get("source_window")
        if not isinstance(window, dict):
            raise WindowReconciliationError("screening cohort source window is invalid")
        count, old_start, reference = window.get("frame_count"), window.get("start_frame"), window.get("source_screening_reference_frame")
        anchors = window.get("p0_anchor_offsets")
        if (
            not isinstance(count, int) or count < 20 or not isinstance(old_start, int) or old_start < 0
            or not isinstance(reference, int) or reference < 0 or not isinstance(anchors, list) or len(anchors) != 4
        ):
            raise WindowReconciliationError(f"{item.get('screening_event_id')}: invalid frozen window/reference")
        session, camera, lens = item.get("source_session_id"), item.get("camera"), item.get("lens")
        if not all(isinstance(value, str) and value for value in (session, camera, lens)):
            raise WindowReconciliationError("screening cohort source identity is invalid")
        rgb_prefix = f"{GCS_PREFIX}/sanpo-real/{session}/{camera}/{lens}/video_frames/"
        mask_prefix = f"{GCS_PREFIX}/sanpo-real/{session}/{camera}/{lens}/segmentation_masks/"
        available = _png_frames(list_objects(rgb_prefix), where=f"{item['screening_event_id']} RGB") & _png_frames(list_objects(mask_prefix), where=f"{item['screening_event_id']} mask")
        anchor_offset = int(anchors[1])
        start = _best_start(
            available, count=count, reference=reference, preferred_start=old_start,
            target_anchor_offset=anchor_offset,
        )
        if start is None:
            raise WindowReconciliationError(f"{item['screening_event_id']}: no contiguous native RGB/mask window can preserve screening reference")
        window["start_frame"] = start
        window["metadata_admission"] = {
            "predecessor_start_frame": old_start,
            "source_screening_reference_frame": reference,
            "reference_anchor_offset": anchor_offset,
            "contiguous_frame_count_verified": count,
            "pixel_payload_read": False,
            "model_or_oracle_output_read": False,
        }
        resolved.append({
            "screening_event_id": item["screening_event_id"],
            "predecessor_start_frame": old_start,
            "resolved_start_frame": start,
            "source_screening_reference_frame": reference,
        })
    result["status"] = "OUTPUT_BLIND_SCREENING_COHORT_CONTINUOUS_WINDOWS_FROZEN"
    result["predecessor_screening_cohort_sha256"] = sha256_json(cohort)
    result["candidate_outputs_opened"] = False
    result["final_event_facts_frozen"] = False
    result["metadata_window_resolution"] = resolved
    result["next_required_gate"] = "Fetch exactly these metadata-verified continuous native RGB/mask windows, then complete contamination audit before any opaque reviewer packet or output trace is generated."
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.retries <= 0:
        raise SystemExit("retries must be positive")
    cohort = read_json(args.screening_cohort)
    result = reconcile_screening_windows(
        cohort, list_objects=lambda prefix: list_gcs_objects(prefix, retries=args.retries)
    )
    result["input_sha256"] = {"screening_cohort": sha256_file(args.screening_cohort)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    moved = sum(row["predecessor_start_frame"] != row["resolved_start_frame"] for row in result["metadata_window_resolution"])
    print(f"screening_events={len(result['items'])} metadata_window_moves={moved} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

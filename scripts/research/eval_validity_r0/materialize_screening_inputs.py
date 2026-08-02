from __future__ import annotations

"""Plan or materialize continuous native SANPO inputs for output-blind P0 review.

The metadata plan performs no pixel download.  Fetch mode uses only that frozen
plan and writes raw RGB/masks under an ignored evidence root.  Neither mode
loads model, oracle, trace or reviewer output.
"""

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

from .common import PROTOCOL_ID, read_json, sha256_file, sha256_json
from .freeze_screening_cohort import SCHEMA as SCREENING_COHORT_SCHEMA


PLAN_SCHEMA = "blindassist.eval_validity_r0.continuous_native_input_plan.v1"
MATERIALIZED_SCHEMA = "blindassist.eval_validity_r0.continuous_native_inputs.v1"
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_sanpo_sequence_evalset import (  # noqa: E402
    GCS_PREFIX,
    download,
    frame_number,
    list_gcs_objects,
    media_url,
    object_inventory,
    verify_gcs_md5,
)
from screen_sanpo_mask_windows import source_fps_for  # noqa: E402


class MaterializationError(ValueError):
    """Raised when a continuous source window is incomplete or unbound."""


def _cohort_items(cohort: dict[str, Any]) -> list[dict[str, Any]]:
    if cohort.get("schema_version") != SCREENING_COHORT_SCHEMA or cohort.get("protocol_id") != PROTOCOL_ID:
        raise MaterializationError("screening cohort schema/protocol mismatch")
    if cohort.get("status") not in {
        "OUTPUT_BLIND_SCREENING_COHORT_FROZEN",
        "OUTPUT_BLIND_SCREENING_COHORT_CONTINUOUS_WINDOWS_FROZEN",
    }:
        raise MaterializationError("screening cohort is not output-blind frozen")
    if cohort.get("candidate_outputs_opened") is not False or cohort.get("final_event_facts_frozen") is not False:
        raise MaterializationError("screening cohort output/event-fact state is invalid")
    items = cohort.get("items")
    if not isinstance(items, list) or len(items) != 48:
        raise MaterializationError("screening cohort must have exactly 48 items")
    identities = [item.get("screening_event_id") for item in items if isinstance(item, dict)]
    sessions = [item.get("source_session_id") for item in items if isinstance(item, dict)]
    if len(identities) != 48 or len(set(identities)) != 48 or len(sessions) != 48 or len(set(sessions)) != 48:
        raise MaterializationError("screening cohort identities/session disjointness is invalid")
    return items


def _object_index(objects: list[dict[str, Any]], *, where: str) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for item in objects:
        name = item.get("name")
        if not isinstance(name, str) or not name.endswith(".png"):
            continue
        number = frame_number(name)
        if number in index:
            raise MaterializationError(f"{where}: duplicate source frame {number}")
        index[number] = item
    return index


def build_plan(
    cohort: dict[str, Any],
    *,
    list_objects: Callable[[str], list[dict[str, Any]]],
    get_source_fps: Callable[[str, str], float],
) -> dict[str, Any]:
    """Create a source-metadata plan; callers inject GCS access for testability."""
    items = _cohort_items(cohort)
    result_items: list[dict[str, Any]] = []
    total_rgb_bytes = total_mask_bytes = 0
    for item in items:
        session = item["source_session_id"]
        camera, lens = item["camera"], item["lens"]
        window = item.get("source_window")
        if not isinstance(window, dict):
            raise MaterializationError(f"{item['screening_event_id']}: source window missing")
        start, count, anchors = window.get("start_frame"), window.get("frame_count"), window.get("p0_anchor_offsets")
        if not isinstance(start, int) or start < 0 or not isinstance(count, int) or count < 20:
            raise MaterializationError(f"{item['screening_event_id']}: invalid continuous source window")
        if not isinstance(anchors, list) or len(anchors) != 4 or any(not isinstance(value, int) or value < 0 or value >= count for value in anchors):
            raise MaterializationError(f"{item['screening_event_id']}: invalid P0 anchors")
        rgb_prefix = f"{GCS_PREFIX}/sanpo-real/{session}/{camera}/{lens}/video_frames/"
        mask_prefix = f"{GCS_PREFIX}/sanpo-real/{session}/{camera}/{lens}/segmentation_masks/"
        rgb_index = _object_index(list_objects(rgb_prefix), where=f"{item['screening_event_id']} RGB")
        mask_index = _object_index(list_objects(mask_prefix), where=f"{item['screening_event_id']} mask")
        source_frames = list(range(start, start + count))
        missing_rgb = [frame for frame in source_frames if frame not in rgb_index]
        missing_mask = [frame for frame in source_frames if frame not in mask_index]
        if missing_rgb or missing_mask:
            raise MaterializationError(
                f"{item['screening_event_id']}: incomplete continuous window "
                f"missing_rgb={missing_rgb[:3]} missing_mask={missing_mask[:3]}"
            )
        frame_rows = []
        for ordinal, frame in enumerate(source_frames):
            rgb, mask = rgb_index[frame], mask_index[frame]
            rgb_inventory, mask_inventory = object_inventory(rgb), object_inventory(mask)
            if rgb_inventory["size"] is None or mask_inventory["size"] is None:
                raise MaterializationError(f"{item['screening_event_id']}: object size missing")
            total_rgb_bytes += int(rgb_inventory["size"])
            total_mask_bytes += int(mask_inventory["size"])
            frame_rows.append({
                "ordinal": ordinal,
                "source_frame_index": frame,
                "rgb": rgb_inventory,
                "source_mask": mask_inventory,
            })
        source_fps = float(get_source_fps(session, camera))
        if source_fps <= 0:
            raise MaterializationError(f"{item['screening_event_id']}: invalid source fps")
        result_items.append({
            "screening_event_id": item["screening_event_id"],
            "source_session_id": session,
            "camera": camera,
            "lens": lens,
            "screening_stratum": item["screening_stratum"],
            "source_selection_profile": item["source_selection_profile"],
            "source_fps": source_fps,
            "anchor_ordinals": anchors,
            "frames": frame_rows,
        })
    return {
        "schema_version": PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "CONTINUOUS_NATIVE_ASSET_PLAN_FROZEN",
        "purpose": "Source metadata only; no pixel payload, reviewer response, model, oracle or feedback trace has been read.",
        "screening_cohort_sha256": sha256_json(cohort),
        "screening_event_count": len(result_items),
        "native_source_session_count": len({item["source_session_id"] for item in result_items}),
        "frame_count": sum(len(item["frames"]) for item in result_items),
        "total_rgb_bytes": total_rgb_bytes,
        "total_source_mask_bytes": total_mask_bytes,
        "candidate_outputs_opened": False,
        "items": result_items,
        "next_required_gate": "Fetch exactly this plan to an ignored root, run contamination audit, then render opaque RGB-only reviewer packets. No model trace may be materialized first.",
    }


def _validate_plan(plan: dict[str, Any], cohort: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("protocol_id") != PROTOCOL_ID:
        raise MaterializationError("native asset plan schema/protocol mismatch")
    if plan.get("status") != "CONTINUOUS_NATIVE_ASSET_PLAN_FROZEN":
        raise MaterializationError("native asset plan is not frozen")
    if plan.get("screening_cohort_sha256") != sha256_json(cohort):
        raise MaterializationError("native asset plan screening cohort binding mismatch")
    if plan.get("candidate_outputs_opened") is not False:
        raise MaterializationError("native asset plan output state is invalid")
    items = plan.get("items")
    if not isinstance(items, list) or len(items) != 48:
        raise MaterializationError("native asset plan item coverage mismatch")
    return items


def materialize_plan(plan: dict[str, Any], cohort: dict[str, Any], output_root: Path, *, retries: int) -> dict[str, Any]:
    items = _validate_plan(plan, cohort)
    if output_root.exists():
        raise MaterializationError(f"refusing to overwrite materialization root: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        records: list[dict[str, Any]] = []
        for item in items:
            event_root = staging / "events" / item["screening_event_id"]
            rows = []
            for frame in item["frames"]:
                ordinal = int(frame["ordinal"])
                rgb_item, mask_item = frame["rgb"], frame["source_mask"]
                rgb_target = event_root / "rgb" / f"{ordinal:03d}.png"
                mask_target = event_root / "source_masks" / f"{ordinal:03d}.png"
                download(media_url(str(rgb_item["name"]), rgb_item.get("generation")), rgb_target, retries=retries)
                download(media_url(str(mask_item["name"]), mask_item.get("generation")), mask_target, retries=retries)
                verify_gcs_md5(rgb_target, {"md5Hash": rgb_item.get("md5_base64")})
                verify_gcs_md5(mask_target, {"md5Hash": mask_item.get("md5_base64")})
                rows.append({
                    "ordinal": ordinal,
                    "source_frame_index": frame["source_frame_index"],
                    "rgb_path": rgb_target.relative_to(staging).as_posix(),
                    "rgb_sha256": sha256_file(rgb_target),
                    "source_mask_path": mask_target.relative_to(staging).as_posix(),
                    "source_mask_sha256": sha256_file(mask_target),
                })
            records.append({
                "screening_event_id": item["screening_event_id"],
                "source_session_id": item["source_session_id"],
                "camera": item["camera"], "lens": item["lens"], "source_fps": item["source_fps"],
                "anchor_ordinals": item["anchor_ordinals"], "frames": rows,
            })
        result = {
            "schema_version": MATERIALIZED_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "CONTINUOUS_NATIVE_RGB_AND_MASKS_MATERIALIZED_OUTPUT_BLIND",
            "screening_cohort_sha256": sha256_json(cohort),
            "native_asset_plan_sha256": sha256_json(plan),
            "candidate_outputs_opened": False,
            "items": records,
            "next_required_gate": "Run the full exact/RGB/session/ancestry/parent/pHash contamination audit before reviewer-packet generation; do not render model or oracle output.",
        }
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, output_root)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _gcs_list(prefix: str, retries: int) -> list[dict[str, Any]]:
    return list_gcs_objects(prefix, retries=retries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "fetch"), required=True)
    parser.add_argument("--screening-cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Required in fetch mode; must be the immutable metadata plan.")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.retries <= 0:
        raise SystemExit("retries must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    cohort = read_json(args.screening_cohort)
    if args.mode == "plan":
        if args.plan is not None:
            raise SystemExit("--plan is only valid in fetch mode")
        result = build_plan(
            cohort,
            list_objects=lambda prefix: _gcs_list(prefix, args.retries),
            get_source_fps=lambda session, camera: source_fps_for(session, camera, args.retries),
        )
        result["input_sha256"] = {"screening_cohort": sha256_file(args.screening_cohort)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"planned_events={result['screening_event_count']} frames={result['frame_count']} total_bytes={result['total_rgb_bytes'] + result['total_source_mask_bytes']} output={args.output}")
        return 0
    if args.plan is None:
        raise SystemExit("fetch mode requires --plan")
    result = materialize_plan(read_json(args.plan), cohort, args.output, retries=args.retries)
    print(f"materialized_events={len(result['items'])} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

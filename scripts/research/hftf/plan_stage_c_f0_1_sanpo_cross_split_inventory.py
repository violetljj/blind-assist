#!/usr/bin/env python3
"""Build the fixed cross-split SANPO source plan for HFTF Stage C F0.1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acquire_sanpo_synthetic_replay import (  # noqa: E402
    camera_metadata,
    indexed_objects,
)
from build_sanpo_sequence_evalset import (  # noqa: E402
    GCS_PREFIX,
    fetch_json,
    fetch_text,
    get_gcs_object,
    list_gcs_objects,
    media_url,
    object_inventory,
)
from plan_stage_c_f0_sanpo_inventory import (  # noqa: E402
    LEDGER_SCHEMA,
    READY as F0_READY,
    SCHEMA as F0_PLAN_SCHEMA,
    _load_json,
    _select_timeline,
    _sha256,
    _sha256_text,
    _validate_intrinsics,
)


PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_cross_split_body_head_"
    "temporal_student_canary_f0_1"
)
SCHEMA = "blindassist_hftf_stage_c_f0_1_sanpo_cross_split_inventory_plan"
READY = "F0_1_SANPO_CROSS_SPLIT_SOURCE_INVENTORY_READY"
NOT_EVALUABLE = (
    "F0_1_SANPO_CROSS_SPLIT_SOURCE_INVENTORY_NOT_EVALUABLE"
)


def _validate_protocol_and_f0_plan(
    protocol_path: Path,
    ledger_path: Path,
    f0_plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    protocol = _load_json(protocol_path)
    ledger = _load_json(ledger_path)
    f0_plan = _load_json(f0_plan_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status")
        != "FROZEN_BEFORE_F0_1_SOURCE_OUTCOME"
    ):
        raise ValueError("F0.1 protocol is not frozen")
    parent = protocol["parent_f0_protocol"]
    parent_path = (protocol_path.parent / str(parent["path"])).resolve()
    if _sha256(parent_path) != str(parent["sha256"]):
        raise ValueError("F0.1 parent F0 protocol hash mismatch")
    ledger_binding = protocol["source_pool_burn_ledger"]
    if (
        ledger_path.name != str(ledger_binding["path"])
        or _sha256(ledger_path) != str(ledger_binding["sha256"])
        or ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("status") != "FROZEN_BEFORE_F0_SOURCE_OUTCOME"
    ):
        raise ValueError("F0.1 burn-ledger binding mismatch")
    parent_ledger_path = (
        ledger_path.parent
        / str(ledger["parent_r4_burn_ledger"]["path"])
    ).resolve()
    parent_ledger = _load_json(parent_ledger_path)
    parent_ids = [
        str(value) for value in parent_ledger["burned_session_ids"]
    ]
    additional = [
        str(value)
        for value in ledger["additional_r4_outcome_open_session_ids"]
    ]
    burned = parent_ids + additional
    if (
        len(burned)
        != int(ledger_binding["effective_train_burned_session_count"])
        or len(burned) != len(set(burned))
    ):
        raise ValueError("F0.1 effective burn union mismatch")
    if (
        f0_plan.get("schema") != F0_PLAN_SCHEMA
        or f0_plan.get("terminal") != F0_READY
        or f0_plan.get("protocol_sha256") != str(parent["sha256"])
        or f0_plan.get("burn_ledger_sha256")
        != str(ledger_binding["sha256"])
        or f0_plan.get("geometry_outcome_read") is not False
        or f0_plan.get("teacher_outcome_read") is not False
        or f0_plan.get("student_outcome_read") is not False
    ):
        raise ValueError("F0 metadata plan binding or firewall mismatch")
    candidates = f0_plan.get("inventory_candidates", [])
    ranks = [int(item["inventory_eligible_rank"]) for item in candidates]
    if len(candidates) != 12 or ranks != list(range(1, 13)):
        raise ValueError("F0 metadata plan is not the exact 12-source plan")
    expected_roles = ["train"] * 6 + ["dev"] * 3
    if [str(item["role"]) for item in candidates[:9]] != expected_roles:
        raise ValueError("F0 first-nine train/dev roles drifted")
    return protocol, f0_plan, set(burned)


def _validate_test_split(
    selection: dict[str, Any],
    generation: str,
    split_text: str,
) -> None:
    if str(selection["split_object_generation"]) != str(generation):
        raise ValueError("Official test split generation drift")
    if str(selection["split_text_sha256"]) != _sha256_text(split_text):
        raise ValueError("Official test split text hash drift")
    ids = [
        line.strip() for line in split_text.splitlines() if line.strip()
    ]
    if len(ids) != int(selection["split_session_count"]):
        raise ValueError("Official test split session count drift")
    if len(ids) != len(set(ids)):
        raise ValueError("Official test split contains duplicate sessions")


def _test_candidate(
    session_id: str,
    rank: int,
    retries: int,
) -> dict[str, Any]:
    prefix = f"{GCS_PREFIX}/sanpo-synthetic/{session_id}"
    description_name = f"{prefix}/description.json"
    poses_name = f"{prefix}/camera_chest/camera_poses.csv"
    description_object = get_gcs_object(description_name, retries)
    poses_object = get_gcs_object(poses_name, retries)
    description = fetch_json(
        media_url(
            description_name, description_object.get("generation")
        ),
        retries,
    )
    if description.get("session_type") != "synthetic":
        raise ValueError("Source description is not synthetic")
    source_fps, dimensions = camera_metadata(
        description, "camera_chest", "left"
    )
    _validate_intrinsics(dimensions)
    rgb = indexed_objects(
        list_gcs_objects(
            f"{prefix}/camera_chest/left/video_frames/", retries
        ),
        ".png",
    )
    masks = indexed_objects(
        list_gcs_objects(
            f"{prefix}/camera_chest/left/segmentation_masks/", retries
        ),
        ".png",
    )
    depth = indexed_objects(
        list_gcs_objects(
            f"{prefix}/camera_chest/left/depth_maps/", retries
        ),
        ".float16.gz",
    )
    target_fps, selected = _select_timeline(
        rgb, masks, depth, source_fps
    )
    return {
        "session_id": session_id,
        "inventory_eligible": True,
        "inventory_eligible_rank": rank,
        "role": "heldout",
        "official_split": "test",
        "source_fps": source_fps,
        "target_fps": target_fps,
        "aligned_source_frame_count": len(
            set(rgb) & set(masks) & set(depth)
        ),
        "selected_source_frames": selected,
        "description_object": object_inventory(description_object),
        "camera_poses_object": object_inventory(poses_object),
        "camera": {
            key: dimensions[key]
            for key in (
                "fx",
                "fy",
                "cx",
                "cy",
                "image_width",
                "image_height",
            )
        },
    }


def plan(
    protocol_path: Path,
    ledger_path: Path,
    f0_plan_path: Path,
    retries: int,
) -> dict[str, Any]:
    if retries <= 0:
        raise ValueError("Retries must be positive")
    protocol, f0_plan, burned = _validate_protocol_and_f0_plan(
        protocol_path, ledger_path, f0_plan_path
    )
    train_dev: list[dict[str, Any]] = []
    for item in f0_plan["inventory_candidates"][:9]:
        copied = dict(item)
        copied["official_split"] = "train"
        copied["source_plan_origin"] = "f0_metadata_plan_first_nine"
        train_dev.append(copied)
    selected_ids = {str(item["session_id"]) for item in train_dev}
    heldout_selection = protocol["heldout_selection"]
    split_name = (
        f"{GCS_PREFIX}/sanpo-synthetic/splits/test_session_ids.txt"
    )
    split_object = get_gcs_object(split_name, retries)
    split_text = fetch_text(
        media_url(split_name, split_object.get("generation")), retries
    )
    _validate_test_split(
        heldout_selection,
        str(split_object.get("generation")),
        split_text,
    )
    scanned: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for session_id in (
        line.strip() for line in split_text.splitlines() if line.strip()
    ):
        if session_id in burned or session_id in selected_ids:
            scanned.append(
                {
                    "session_id": session_id,
                    "inventory_eligible": False,
                    "reason": "burned_or_selected_train_dev_session",
                }
            )
            continue
        try:
            item = _test_candidate(
                session_id, len(heldout) + 1, retries
            )
            heldout.append(item)
            scanned.append(item)
            if len(heldout) == int(
                heldout_selection["fixed_inventory_eligible_count"]
            ):
                break
        except (KeyError, OSError, TypeError, ValueError) as error:
            scanned.append(
                {
                    "session_id": session_id,
                    "inventory_eligible": False,
                    "reason": str(error),
                }
            )
    expected_heldout = int(
        heldout_selection["fixed_inventory_eligible_count"]
    )
    terminal = READY if len(heldout) == expected_heldout else NOT_EVALUABLE
    sources = train_dev + heldout
    role_counts = {
        role: sum(str(item["role"]) == role for item in sources)
        for role in ("train", "dev", "heldout")
    }
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": protocol["workflow_profile"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "parent_f0_plan_path": str(f0_plan_path.resolve()),
        "parent_f0_plan_sha256": _sha256(f0_plan_path),
        "burn_ledger_path": str(ledger_path.resolve()),
        "burn_ledger_sha256": _sha256(ledger_path),
        "test_split_object": object_inventory(split_object),
        "test_split_text_sha256": _sha256_text(split_text),
        "role_counts": role_counts,
        "source_count": len(sources),
        "sources": sources,
        "test_scan_ledger": scanned,
        "parent_session_disjoint": (
            len({str(item["session_id"]) for item in sources})
            == len(sources)
        ),
        "geometry_outcome_read": False,
        "teacher_outcome_read": False,
        "student_outcome_read": False,
        "source_acquisition_authorized": terminal == READY,
        "teacher_corpus_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--burn-ledger", type=Path, required=True)
    parser.add_argument("--f0-plan", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = plan(
            args.protocol.resolve(),
            args.burn_ledger.resolve(),
            args.f0_plan.resolve(),
            args.retries,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "role_counts": report["role_counts"],
                    "output": str(output),
                }
            )
        )
        return 0
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

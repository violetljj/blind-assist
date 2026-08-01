#!/usr/bin/env python3
"""Plan the bounded HFTF R4 obstacle-only SANPO source screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acquire_sanpo_synthetic_replay import (  # noqa: E402
    camera_metadata,
    indexed_objects,
    select_aligned_indices,
)
from build_sanpo_sequence_evalset import (  # noqa: E402
    GCS_PREFIX,
    fetch_json,
    fetch_text,
    get_gcs_object,
    list_gcs_objects,
    media_url,
)
from verify_sanpo_pose_geometry_authority import _load_json  # noqa: E402


SCHEMA = "blindassist_hftf_r4_obstacle_inventory_candidate_plan"
PROTOCOL_SCHEMA = "blindassist_hftf_stage_b_split_source_validation_r4"
LEDGER_SCHEMA = "blindassist_hftf_r4_source_pool_burn_ledger"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_frozen_inputs(
    protocol: dict[str, Any],
    protocol_path: Path,
    ledger: dict[str, Any],
    ledger_path: Path,
    repo_root: Path | None = None,
) -> set[str]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "FROZEN_BEFORE_R4_OUTCOME"
    ):
        raise ValueError("R4 protocol is not frozen")
    parent_result = (
        protocol_path.parent / str(protocol["parent_result_path"])
    )
    if _sha256(parent_result) != protocol["parent_result_sha256"]:
        raise ValueError("R4 parent result hash mismatch")
    if (
        ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("status") != "FROZEN_BEFORE_R4_OUTCOME"
    ):
        raise ValueError("R4 burn ledger is not frozen")
    old = ledger["parent_r3_1_burn_ledger"]
    if _sha256(ledger_path.parent / str(old["path"])) != old["sha256"]:
        raise ValueError("R4 parent burn-ledger hash mismatch")
    cohort = ledger["parent_r3_1_cohort_report"]
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    if _sha256(repo_root / str(cohort["path"])) != cohort["sha256"]:
        raise ValueError("R4 parent cohort-report hash mismatch")
    burned = [str(value) for value in ledger["burned_session_ids"]]
    if (
        len(burned) != int(ledger["burned_session_count"])
        or len(burned) != len(set(burned))
    ):
        raise ValueError("R4 burn ledger count or uniqueness mismatch")
    return set(burned)


def _validate_split(
    source: dict[str, Any], split_generation: str, split_text: str
) -> None:
    if str(split_generation) != str(source["split_object_generation"]):
        raise ValueError("Official split generation drift")
    if _sha256_text(split_text) != source["split_text_sha256"]:
        raise ValueError("Official split text hash drift")


def plan(
    protocol_path: Path,
    ledger_path: Path,
    retries: int,
) -> dict[str, Any]:
    if retries <= 0:
        raise ValueError("Retries must be positive")
    protocol = _load_json(protocol_path)
    ledger = _load_json(ledger_path)
    burned = _validate_frozen_inputs(
        protocol, protocol_path, ledger, ledger_path
    )
    source = protocol["obstacle_source_role"]
    split_name = (
        f"{GCS_PREFIX}/sanpo-synthetic/splits/train_session_ids.txt"
    )
    split_object = get_gcs_object(split_name, retries)
    split_text = fetch_text(
        media_url(split_name, split_object.get("generation")), retries
    )
    _validate_split(
        source, str(split_object.get("generation")), split_text
    )
    limit = int(source["maximum_inventory_eligible_sessions_to_screen"])
    replay = source["replay"]
    frame_count = int(replay["frame_count"])
    start_frame = int(replay["start_frame"])
    scanned: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for session_id in sorted(
        line.strip() for line in split_text.splitlines() if line.strip()
    ):
        if session_id in burned:
            scanned.append(
                {
                    "session_id": session_id,
                    "inventory_eligible": False,
                    "reason": "burned_session",
                }
            )
            continue
        prefix = f"{GCS_PREFIX}/sanpo-synthetic/{session_id}"
        try:
            description_name = f"{prefix}/description.json"
            description_object = get_gcs_object(description_name, retries)
            description = fetch_json(
                media_url(
                    description_name,
                    description_object.get("generation"),
                ),
                retries,
            )
            source_fps, _ = camera_metadata(
                description, "camera_chest", "left"
            )
            target_fps = min(10.0, source_fps)
            rgb = indexed_objects(
                list_gcs_objects(
                    f"{prefix}/camera_chest/left/video_frames/", retries
                ),
                ".png",
            )
            masks = indexed_objects(
                list_gcs_objects(
                    f"{prefix}/camera_chest/left/segmentation_masks/",
                    retries,
                ),
                ".png",
            )
            depth = indexed_objects(
                list_gcs_objects(
                    f"{prefix}/camera_chest/left/depth_maps/", retries
                ),
                ".float16.gz",
            )
            frames = select_aligned_indices(
                rgb,
                masks,
                depth,
                source_fps=source_fps,
                target_fps=target_fps,
                start_frame=start_frame,
                frame_count=frame_count,
            )
            item = {
                "session_id": session_id,
                "inventory_eligible": True,
                "inventory_eligible_rank": len(eligible) + 1,
                "source_fps": source_fps,
                "target_fps": target_fps,
                "aligned_available": len(
                    set(rgb) & set(masks) & set(depth)
                ),
                "selected_source_frames": frames,
                "description_generation": str(
                    description_object.get("generation")
                ),
            }
            eligible.append(item)
            scanned.append(item)
            if len(eligible) == limit:
                break
        except (KeyError, OSError, TypeError, ValueError) as error:
            scanned.append(
                {
                    "session_id": session_id,
                    "inventory_eligible": False,
                    "reason": str(error),
                }
            )
    return {
        "schema": SCHEMA,
        "terminal": (
            "R4_OBSTACLE_INVENTORY_CANDIDATE_PLAN_READY"
            if len(eligible) == limit
            else "R4_OBSTACLE_INVENTORY_CANDIDATE_PLAN_NOT_EVALUABLE"
        ),
        "workflow_profile": protocol["workflow_profile"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "burn_ledger_path": str(ledger_path.resolve()),
        "burn_ledger_sha256": _sha256(ledger_path),
        "split_object_generation": str(split_object.get("generation")),
        "split_text_sha256": _sha256_text(split_text),
        "requested_inventory_eligible_count": limit,
        "inventory_eligible_count": len(eligible),
        "scanned_session_count_including_burned_and_ineligible": len(
            scanned
        ),
        "inventory_candidates": eligible,
        "scan_ledger": scanned,
        "reference_outcome_read": False,
        "ground_outcome_read": False,
        "candidate_outcome_read": False,
        "baseline_outcome_read": False,
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
                    "inventory_eligible_count": report[
                        "inventory_eligible_count"
                    ],
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

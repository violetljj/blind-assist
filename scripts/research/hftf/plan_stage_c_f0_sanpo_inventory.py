#!/usr/bin/env python3
"""Plan the fixed, outcome-blind SANPO source set for HFTF Stage C F0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    object_inventory,
)


PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_body_head_temporal_student_canary_f0"
)
LEDGER_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_body_head_source_pool_burn_ledger_f0"
)
PARENT_LEDGER_SCHEMA = "blindassist_hftf_r4_source_pool_burn_ledger"
COHORT_LOCK_SCHEMA = (
    "blindassist_hftf_r4_obstacle_opportunity_cohort_lock"
)
SCHEMA = "blindassist_hftf_stage_c_f0_sanpo_inventory_plan"
READY = "F0_SANPO_FIXED_SOURCE_INVENTORY_READY"
NOT_EVALUABLE = "F0_SANPO_FIXED_SOURCE_INVENTORY_NOT_EVALUABLE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve_repo_path(
    repo_root: Path, docs_root: Path, value: str
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    docs_candidate = (docs_root / path).resolve()
    if docs_candidate.exists():
        return docs_candidate
    return (repo_root / path).resolve()


def _validate_frozen_inputs(
    protocol_path: Path,
    ledger_path: Path,
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], set[str]]:
    protocol = _load_json(protocol_path)
    ledger = _load_json(ledger_path)
    repo_root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    docs_root = protocol_path.parent.resolve()
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "FROZEN_BEFORE_F0_SOURCE_OUTCOME"
    ):
        raise ValueError("F0 protocol is not frozen")
    if (
        ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("status") != "FROZEN_BEFORE_F0_SOURCE_OUTCOME"
    ):
        raise ValueError("F0 burn ledger is not frozen")
    ledger_parent = protocol["parents"]["source_pool_burn_ledger"]
    if str(ledger_parent["path"]) != ledger_path.name:
        raise ValueError("F0 protocol burn-ledger path mismatch")
    for name in (
        "stage_b_r4_result",
        "egowalk_route_closure",
        "swept_envelope_mechanics",
    ):
        binding = protocol["parents"][name]
        path = _resolve_repo_path(
            repo_root, docs_root, str(binding["path"])
        )
        if _sha256(path) != str(binding["sha256"]):
            raise ValueError(f"F0 parent hash mismatch: {name}")
    parent_binding = ledger["parent_r4_burn_ledger"]
    parent_path = _resolve_repo_path(
        repo_root, ledger_path.parent, str(parent_binding["path"])
    )
    if _sha256(parent_path) != str(parent_binding["sha256"]):
        raise ValueError("F0 parent burn-ledger hash mismatch")
    parent = _load_json(parent_path)
    parent_ids = [str(value) for value in parent["burned_session_ids"]]
    if (
        parent.get("schema") != PARENT_LEDGER_SCHEMA
        or parent.get("status") != "FROZEN_BEFORE_R4_OUTCOME"
        or len(parent_ids)
        != int(parent_binding["burned_session_count"])
        or len(parent_ids) != len(set(parent_ids))
    ):
        raise ValueError("F0 parent burn-ledger structure mismatch")
    cohort_binding = ledger["r4_obstacle_cohort_lock"]
    cohort_path = _resolve_repo_path(
        repo_root, ledger_path.parent, str(cohort_binding["path"])
    )
    if _sha256(cohort_path) != str(cohort_binding["sha256"]):
        raise ValueError("F0 R4 cohort-lock hash mismatch")
    cohort = _load_json(cohort_path)
    cohort_ids = [
        str(item["source_session_id"])
        for item in cohort.get("required_sessions", [])
    ]
    additional = [
        str(value)
        for value in ledger["additional_r4_outcome_open_session_ids"]
    ]
    if (
        cohort.get("schema") != COHORT_LOCK_SCHEMA
        or cohort.get("terminal")
        != "R4_OBSTACLE_OPPORTUNITY_COHORT_QUALIFIED"
        or additional != cohort_ids
        or len(additional) != len(set(additional))
    ):
        raise ValueError("F0 additional burn set does not match R4 cohort")
    effective = parent_ids + additional
    if (
        len(effective) != int(ledger["effective_burned_session_count"])
        or len(effective) != len(set(effective))
        or int(protocol["source_selection"][
            "exclude_effective_burned_session_count"
        ])
        != len(effective)
    ):
        raise ValueError("F0 effective burn set count or uniqueness mismatch")
    return protocol, set(effective)


def _validate_split(
    source: dict[str, Any],
    generation: str,
    split_text: str,
) -> None:
    if str(source["split_object_generation"]) != str(generation):
        raise ValueError("Official split generation drift")
    if str(source["split_text_sha256"]) != _sha256_text(split_text):
        raise ValueError("Official split text hash drift")


def _validate_intrinsics(dimensions: dict[str, Any]) -> None:
    for key in ("fx", "fy", "cx", "cy"):
        value = float(dimensions[key])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite camera intrinsic: {key}")
    if float(dimensions["fx"]) <= 0 or float(dimensions["fy"]) <= 0:
        raise ValueError("Camera focal lengths must be positive")
    if (
        int(dimensions["image_width"]) <= 0
        or int(dimensions["image_height"]) <= 0
    ):
        raise ValueError("Camera dimensions must be positive")


def _select_timeline(
    rgb: dict[int, dict[str, Any]],
    masks: dict[int, dict[str, Any]],
    depth: dict[int, dict[str, Any]],
    source_fps: float,
) -> tuple[float, list[int]]:
    if source_fps not in (5.0, 20.0):
        raise ValueError("F0 source fps must be exactly 5 or 20")
    aligned = set(rgb) & set(masks) & set(depth)
    required = set(range(50))
    if not required.issubset(aligned):
        raise ValueError(
            "F0 requires aligned RGB/mask/depth source frames 0..49"
        )
    target_fps = min(10.0, source_fps)
    selected = select_aligned_indices(
        rgb,
        masks,
        depth,
        source_fps=source_fps,
        target_fps=target_fps,
        start_frame=0,
        frame_count=25,
    )
    return target_fps, selected


def _role(rank: int) -> str:
    if 1 <= rank <= 6:
        return "train"
    if 7 <= rank <= 9:
        return "dev"
    if 10 <= rank <= 12:
        return "heldout"
    raise ValueError(f"F0 rank outside fixed role contract: {rank}")


def plan(
    protocol_path: Path,
    ledger_path: Path,
    retries: int,
) -> dict[str, Any]:
    if retries <= 0:
        raise ValueError("Retries must be positive")
    protocol, burned = _validate_frozen_inputs(
        protocol_path, ledger_path
    )
    source = protocol["source_selection"]
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
    required_count = int(source["fixed_inventory_eligible_count"])
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
            poses_name = f"{prefix}/camera_chest/camera_poses.csv"
            description_object = get_gcs_object(description_name, retries)
            poses_object = get_gcs_object(poses_name, retries)
            description = fetch_json(
                media_url(
                    description_name,
                    description_object.get("generation"),
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
                    f"{prefix}/camera_chest/left/video_frames/",
                    retries,
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
                    f"{prefix}/camera_chest/left/depth_maps/",
                    retries,
                ),
                ".float16.gz",
            )
            target_fps, selected = _select_timeline(
                rgb, masks, depth, source_fps
            )
            rank = len(eligible) + 1
            item = {
                "session_id": session_id,
                "inventory_eligible": True,
                "inventory_eligible_rank": rank,
                "role": _role(rank),
                "source_fps": source_fps,
                "target_fps": target_fps,
                "aligned_source_frame_count": len(
                    set(rgb) & set(masks) & set(depth)
                ),
                "selected_source_frames": selected,
                "description_object": object_inventory(
                    description_object
                ),
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
            eligible.append(item)
            scanned.append(item)
            if len(eligible) == required_count:
                break
        except (KeyError, OSError, TypeError, ValueError) as error:
            scanned.append(
                {
                    "session_id": session_id,
                    "inventory_eligible": False,
                    "reason": str(error),
                }
            )
    terminal = READY if len(eligible) == required_count else NOT_EVALUABLE
    role_counts = {
        name: sum(item["role"] == name for item in eligible)
        for name in ("train", "dev", "heldout")
    }
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": protocol["workflow_profile"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "burn_ledger_path": str(ledger_path.resolve()),
        "burn_ledger_sha256": _sha256(ledger_path),
        "split_object_generation": str(
            split_object.get("generation")
        ),
        "split_text_sha256": _sha256_text(split_text),
        "effective_burned_session_count": len(burned),
        "requested_inventory_eligible_count": required_count,
        "inventory_eligible_count": len(eligible),
        "role_counts": role_counts,
        "scanned_session_count_including_burned_and_ineligible": len(
            scanned
        ),
        "inventory_candidates": eligible,
        "scan_ledger": scanned,
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

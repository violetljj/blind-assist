#!/usr/bin/env python3
"""Plan the bounded HFTF R3.1 inventory-eligible source scan."""

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


SCHEMA = "blindassist_hftf_r3_1_inventory_candidate_plan"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_b_reference_only_opportunity_qualification_r3_1"
)
LEDGER_SCHEMA = "blindassist_hftf_r3_1_source_pool_burn_ledger"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_source_pool(
    protocol: dict[str, Any],
    ledger: dict[str, Any],
    split_generation: str,
    split_text: str,
) -> set[str]:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unexpected R3.1 protocol schema")
    if ledger.get("schema") != LEDGER_SCHEMA:
        raise ValueError("Unexpected R3.1 burn ledger schema")
    source_pool = protocol["source_pool"]
    if str(split_generation) != str(
        source_pool["split_object_generation"]
    ):
        raise ValueError("Official split generation drift")
    if _sha256_text(split_text) != source_pool["split_text_sha256"]:
        raise ValueError("Official split text hash drift")
    burned = [str(value) for value in ledger["burned_session_ids"]]
    if (
        len(burned) != int(ledger["burned_session_count"])
        or len(burned) != len(set(burned))
    ):
        raise ValueError("Burn ledger count or uniqueness mismatch")
    return set(burned)


def plan(
    protocol_path: Path,
    ledger_path: Path,
    retries: int,
) -> dict[str, Any]:
    if retries <= 0:
        raise ValueError("Retries must be positive")
    protocol = _load_json(protocol_path)
    ledger = _load_json(ledger_path)
    split_name = (
        f"{GCS_PREFIX}/sanpo-synthetic/splits/train_session_ids.txt"
    )
    split_object = get_gcs_object(split_name, retries)
    split_text = fetch_text(
        media_url(split_name, split_object.get("generation")),
        retries,
    )
    burned = _validate_source_pool(
        protocol,
        ledger,
        str(split_object.get("generation")),
        split_text,
    )
    session_ids = sorted(
        line.strip()
        for line in split_text.splitlines()
        if line.strip()
    )
    limit = int(
        protocol["source_pool"][
            "maximum_inventory_eligible_sessions_to_screen"
        ]
    )
    frame_count = int(protocol["replay_and_authority"]["frame_count"])
    start_frame = int(protocol["replay_and_authority"]["start_frame"])
    scanned: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for session_id in session_ids:
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
            description_object = get_gcs_object(
                description_name, retries
            )
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
            "R3_1_INVENTORY_CANDIDATE_PLAN_READY"
            if len(eligible) == limit
            else "R3_1_INVENTORY_CANDIDATE_PLAN_NOT_EVALUABLE"
        ),
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest(),
        "burn_ledger_path": str(ledger_path.resolve()),
        "burn_ledger_sha256": hashlib.sha256(
            ledger_path.read_bytes()
        ).hexdigest(),
        "split_object_generation": str(
            split_object.get("generation")
        ),
        "split_text_sha256": _sha256_text(split_text),
        "requested_inventory_eligible_count": limit,
        "inventory_eligible_count": len(eligible),
        "scanned_session_count_including_burned_and_ineligible": len(
            scanned
        ),
        "inventory_candidates": eligible,
        "scan_ledger": scanned,
        "reference_outcome_read": False,
        "candidate_outcome_read": False,
        "baseline_outcome_read": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _require_artifacts_output(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_root = (repo_root / "artifacts.local").resolve()
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
        payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "inventory_eligible_count": report[
                        "inventory_eligible_count"
                    ],
                    "output": str(output),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

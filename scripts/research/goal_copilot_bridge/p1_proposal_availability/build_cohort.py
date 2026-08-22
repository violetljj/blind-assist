#!/usr/bin/env python3
"""Materialize the seven visible first-poison frames without exposing GT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROTOCOL_ID = "P1-PA0-TARGET-CANDIDATE-AVAILABILITY-V1"
PUBLIC_SCHEMA = "blindassist_p1_pa0_public_input_v1"
PRIVATE_SCHEMA = "blindassist_p1_pa0_private_eval_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(autopsy_path: Path, public_path: Path, private_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    autopsy = load(autopsy_path)
    source_public = load(public_path)
    source_private = load(private_path)
    if source_public.get("claim_role") != "CONSUMED_DEVELOPMENT_ONLY":
        raise ValueError("source public input is not consumed Development")
    if source_private.get("public_input_sha256") != sha256(public_path):
        raise ValueError("source public/private identity mismatch")

    public_episodes: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for source in source_public["sources"]:
        for episode in source["episodes"]:
            public_episodes[episode["episode_id"]] = (source, episode)
    private_episodes: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for source in source_private["sources"]:
        for episode in source["episodes"]:
            private_episodes[episode["public_episode_id"]] = (source, episode)

    visible = [row for row in autopsy["first_poison_events"] if row["target_observable"]]
    if len(visible) != 7 or any(row["candidate_target_iou"] != 0.0 for row in visible):
        raise ValueError("expected the frozen seven visible zero-IoU first-poison events")

    public_cases: list[dict[str, Any]] = []
    private_cases: list[dict[str, Any]] = []
    for row in sorted(visible, key=lambda value: (value["episode_id"], value["frame_index"])):
        episode_id = row["episode_id"]
        source, public_episode = public_episodes[episode_id]
        private_source, private_ref = private_episodes[episode_id]
        truth_episode_path = Path(private_ref["episode_path"])
        truth_episode = load(truth_episode_path)
        frame_index = int(row["frame_index"])
        public_frame = public_episode["frames"][frame_index]
        truth_frame = truth_episode["frames"][frame_index]
        if int(public_frame["video_frame_index"]) != int(truth_frame["source_frame_index"]):
            raise ValueError(f"frame identity mismatch: {episode_id}")
        case_id = f"{episode_id}-frame-{frame_index:03d}"
        video_path = Path(source["rgb_video_path"])
        if video_path != Path(private_source["rgb_video_path"]):
            raise ValueError(f"video identity mismatch: {episode_id}")
        public_cases.append({
            "case_id": case_id,
            "episode_id": episode_id,
            "frame_index": frame_index,
            "timestamp_ms": int(public_frame["timestamp_ms"]),
            "query": {
                "rgb_video_path": str(video_path.resolve()),
                "rgb_video_sha256": source["rgb_video_sha256"],
                "video_frame_index": int(public_frame["video_frame_index"]),
            },
            "target_specification": {
                "authority": public_episode["handoff"]["grounding_provenance"]["authority"],
                "referent_id": public_episode["handoff"]["referent_id"],
                "exemplar_rgb_video_path": str(video_path.resolve()),
                "exemplar_video_frame_index": int(public_episode["frames"][0]["video_frame_index"]),
                "exemplar_bbox_xyxy": [float(value) for value in public_episode["initial_target_bbox_xyxy"]],
            },
        })
        bbox = [float(value) for value in truth_frame["target_bbox_xyxy"]]
        private_cases.append({
            "case_id": case_id,
            "target_visible": bool(truth_frame["target_visible"]),
            "target_bbox_xyxy": bbox,
            "target_visibility_ratio": float(truth_frame["target_visibility_ratio"]),
            "target_shortest_side_px": min(bbox[2] - bbox[0], bbox[3] - bbox[1]),
            "diagnostic_target_metadata": truth_episode["target_metadata"],
            "truth_authority": truth_episode["truth_authority"],
        })

    public = {
        "schema_version": PUBLIC_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "claim_role": "POST_OUTCOME_SELECTED_CONSUMED_DEVELOPMENT_MECHANISM_COHORT",
        "selection": "SEVEN_TARGET_VISIBLE_FIRST_POISON_EVENTS_FROM_SEALED_P1_AMRM0_AUTOPSY",
        "provider_contract": {
            "input": "CURRENT_FRAME_PLUS_FROZEN_INITIAL_TARGET_EXEMPLAR",
            "maximum_candidates": 10,
            "ordered_candidate_pool": True,
            "downstream_selection_memory_verifier_reacquisition": "FORBIDDEN",
            "private_truth_access": "FORBIDDEN",
        },
        "cases": public_cases,
    }
    return public, {
        "schema_version": PRIVATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "public_input_sha256": None,
        "primary_correct_iou_threshold": 0.10,
        "diagnostic_correct_iou_thresholds": [0.30, 0.50],
        "recall_at_k": [1, 3, 5, 10],
        "cases": private_cases,
        "claim_ceiling": "FAILURE_COHORT_MECHANISM_DIAGNOSTIC_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autopsy", type=Path, required=True)
    parser.add_argument("--source-public", type=Path, required=True)
    parser.add_argument("--source-private", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    public, private = build(args.autopsy, args.source_public, args.source_private)
    public_output = args.output_dir / "public_input.json"
    atomic_json(public_output, public)
    private["public_input_sha256"] = sha256(public_output)
    atomic_json(args.output_dir / "private_eval_input.json", private)
    atomic_json(args.output_dir / "cohort_receipt.json", {
        "protocol_id": PROTOCOL_ID,
        "public_input_sha256": sha256(public_output),
        "private_eval_input_sha256": sha256(args.output_dir / "private_eval_input.json"),
        "source_sha256": {
            "first_poison_autopsy": sha256(args.autopsy),
            "source_public": sha256(args.source_public),
            "source_private": sha256(args.source_private),
        },
        "case_count": len(public["cases"]),
        "target_visible_cases": len(public["cases"]),
        "claim_ceiling": private["claim_ceiling"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

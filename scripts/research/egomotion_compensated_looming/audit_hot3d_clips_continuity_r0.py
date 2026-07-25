#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFINITIONS_SHA256 = "b9b50b30934c548e55a0400ef076eeb25f937b574fe2b205c5dcaccb4e4ab610"
SPLITS_SHA256 = "54067720be3645aff6e9d79dcba2bd8d1c6d3f91e1dfba2cfae42da9ec5f497c"
HF_COMMIT = "6ff007d45ffb6ec2cfdab7ff3f9475d24b6ad28a"
TOOLKIT_COMMIT = "146b34afef8c1a32adeef7e981c070109f225c87"
SELECTION_SALT = "hot3d-clips-r0-source-prescreen-v1"
MAX_CONTIGUOUS_GAP_NS = 40_000_000
SEQUENCES_PER_PARTICIPANT = 3


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    material = "\t".join(str(part) for part in parts)
    return hashlib.sha256(f"{SELECTION_SALT}\t{material}".encode()).hexdigest()


def stream_timestamps_ns(clip: dict[str, Any], stream_id: str = "214-1") -> list[int]:
    frames = clip.get("per_frame_timestamps_ns")
    if not isinstance(frames, list) or len(frames) != 150:
        raise ValueError("every HOT3D-Clip must contain exactly 150 timestamp rows")
    result: list[int] = []
    for frame in frames:
        if not isinstance(frame, dict) or stream_id not in frame:
            raise ValueError(f"missing {stream_id} timestamp")
        value = frame[stream_id]
        if not isinstance(value, int):
            raise ValueError("timestamps must be integers")
        result.append(value)
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("clip timestamps must be strictly increasing")
    return result


def build_receipt(
    definitions: dict[str, Any],
    splits: dict[str, Any],
    definitions_sha256: str,
    splits_sha256: str,
) -> dict[str, Any]:
    if definitions_sha256 != DEFINITIONS_SHA256:
        raise ValueError("clip_definitions.json SHA-256 mismatch")
    if splits_sha256 != SPLITS_SHA256:
        raise ValueError("clip_splits.json SHA-256 mismatch")
    if len(definitions) != 3832:
        raise ValueError("unexpected HOT3D-Clip count")

    train = splits.get("train")
    if not isinstance(train, dict):
        raise ValueError("missing train split")
    train_aria_ids = {str(value) for value in train.get("Aria", [])}
    train_quest_ids = {str(value) for value in train.get("Quest3", [])}
    if train_aria_ids & train_quest_ids:
        raise ValueError("train device split overlap")
    if len(train_aria_ids) != 1516 or len(train_quest_ids) != 1288:
        raise ValueError("unexpected HOT3D train split counts")

    all_devices = Counter()
    sequence_clips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip_id, clip in definitions.items():
        if not isinstance(clip, dict):
            raise ValueError("clip definition must be an object")
        device = clip.get("device")
        sequence_id = clip.get("sequence_id")
        if not isinstance(device, str) or not isinstance(sequence_id, str):
            raise ValueError("clip device and sequence_id are required")
        all_devices[device] += 1
        if clip_id not in train_aria_ids:
            continue
        if device != "Aria":
            raise ValueError("train Aria split contains a non-Aria clip")
        timestamps = stream_timestamps_ns(clip)
        sequence_clips[sequence_id].append(
            {
                "clip_id": int(clip_id),
                "start_timestamp_ns": timestamps[0],
                "end_timestamp_ns": timestamps[-1],
                "frame_count": len(timestamps),
            }
        )

    candidates_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence_id, clips in sequence_clips.items():
        clips.sort(key=lambda item: (item["start_timestamp_ns"], item["clip_id"]))
        for left, right in zip(clips, clips[1:]):
            gap_ns = right["start_timestamp_ns"] - left["end_timestamp_ns"]
            if 0 < gap_ns <= MAX_CONTIGUOUS_GAP_NS:
                candidates_by_sequence[sequence_id].append(
                    {
                        "sequence_id": sequence_id,
                        "participant_id": sequence_id.split("_", 1)[0],
                        "device": "Aria",
                        "left_clip_id": left["clip_id"],
                        "right_clip_id": right["clip_id"],
                        "first_timestamp_ns": left["start_timestamp_ns"],
                        "last_timestamp_ns": right["end_timestamp_ns"],
                        "boundary_gap_ns": gap_ns,
                        "frame_count": left["frame_count"] + right["frame_count"],
                    }
                )

    sequences_by_participant: dict[str, list[str]] = defaultdict(list)
    for sequence_id in candidates_by_sequence:
        participant_id = sequence_id.split("_", 1)[0]
        sequences_by_participant[participant_id].append(sequence_id)
    if len(sequences_by_participant) != 9:
        raise ValueError("expected nine train-Aria participants with contiguous pairs")

    selected_pairs: list[dict[str, Any]] = []
    for participant_id in sorted(sequences_by_participant):
        ranked_sequences = sorted(
            sequences_by_participant[participant_id],
            key=lambda sequence_id: stable_hash("sequence", participant_id, sequence_id),
        )
        if len(ranked_sequences) < SEQUENCES_PER_PARTICIPANT:
            raise ValueError(f"insufficient contiguous sequences for {participant_id}")
        for sequence_id in ranked_sequences[:SEQUENCES_PER_PARTICIPANT]:
            pair = min(
                candidates_by_sequence[sequence_id],
                key=lambda item: stable_hash(
                    "pair",
                    sequence_id,
                    item["left_clip_id"],
                    item["right_clip_id"],
                    item["first_timestamp_ns"],
                ),
            )
            selected_pairs.append(pair)

    selected_pairs.sort(
        key=lambda item: (item["participant_id"], item["sequence_id"])
    )
    identity_lines = [
        "\t".join(
            [
                item["participant_id"],
                item["sequence_id"],
                str(item["left_clip_id"]),
                str(item["right_clip_id"]),
                str(item["first_timestamp_ns"]),
                str(item["last_timestamp_ns"]),
            ]
        )
        for item in selected_pairs
    ]
    cohort_sha256 = hashlib.sha256(
        ("\n".join(identity_lines) + "\n").encode()
    ).hexdigest()

    participant_counts = Counter(
        item["participant_id"] for item in selected_pairs
    )
    return {
        "schema_version": "hot3d_clips_continuity_and_authority_audit_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0",
        "source_id": "HOT3D_CLIPS",
        "authority": {
            "huggingface_repository": "bop-benchmark/hot3d",
            "huggingface_commit": HF_COMMIT,
            "official_toolkit_commit": TOOLKIT_COMMIT,
            "clip_definitions_sha256": definitions_sha256,
            "clip_splits_sha256": splits_sha256,
        },
        "observed_inventory": {
            "clip_count": len(definitions),
            "device_counts": dict(sorted(all_devices.items())),
            "train_aria_clip_count": len(train_aria_ids),
            "train_quest3_clip_count": len(train_quest_ids),
            "train_aria_sequence_count": len(sequence_clips),
            "train_aria_participant_count": len(sequences_by_participant),
            "train_aria_contiguous_pair_count": sum(
                len(value) for value in candidates_by_sequence.values()
            ),
            "train_aria_sequence_count_with_contiguous_pair": len(
                candidates_by_sequence
            ),
        },
        "continuity_contract": {
            "same_sequence_required": True,
            "same_device_required": True,
            "train_split_only": True,
            "stream_id": "214-1",
            "clip_frame_count": 150,
            "pair_frame_count": 300,
            "maximum_boundary_gap_ns": MAX_CONTIGUOUS_GAP_NS,
            "positive_gap_required": True,
        },
        "source_prescreen_freeze": {
            "selection_salt": SELECTION_SALT,
            "selection_rule": (
                "Within each train-Aria participant, rank sequences having at "
                "least one contiguous clip pair by SHA256(salt, participant, "
                "sequence); take three sequences. Within each selected sequence, "
                "take the lowest-hash contiguous pair."
            ),
            "permanent_role": "SOURCE_PRESCREEN_ONLY",
            "participant_count": len(participant_counts),
            "sequence_and_pair_count": len(selected_pairs),
            "clip_count": 2 * len(selected_pairs),
            "frame_count": sum(item["frame_count"] for item in selected_pairs),
            "participant_pair_counts": dict(sorted(participant_counts.items())),
            "cohort_identity_sha256": cohort_sha256,
            "selected_pairs": selected_pairs,
        },
        "authority_interpretation": {
            "participant_id_is_capture_cluster": True,
            "sequence_id_is_session_id": True,
            "public_train_clips_have_camera_pose_and_object_hand_ground_truth": True,
            "single_five_second_clip_meets_ten_second_unit": False,
            "contiguous_pair_meets_twenty_half_second_epoch_mechanics": True,
            "counterfactual_cell_truth_proven": False,
            "source_admitted": False,
        },
        "read_firewall": {
            "old_window_identity_or_outcome_read_count": 0,
            "clip_tar_read_count": 0,
            "candidate_image_byte_read_count": 0,
            "candidate_signal_computed": False,
            "role_split_frozen": False,
        },
        "terminal": "HOT3D_PUBLIC_CLIPS_ANNOTATION_ONLY_PRESCREEN_READY",
        "status": "VALID",
        "next_boundary": (
            "Freeze an annotation-only HTTP Range extractor before touching "
            "selected tar members; read tar headers plus cameras/objects/info "
            "JSON only, never image members, then prescreen the four cells."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definitions", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    definitions_sha256 = sha256_path(args.definitions)
    splits_sha256 = sha256_path(args.splits)
    with args.definitions.open("r", encoding="utf-8") as stream:
        definitions = json.load(stream)
    with args.splits.open("r", encoding="utf-8") as stream:
        splits = json.load(stream)
    receipt = build_receipt(
        definitions,
        splits,
        definitions_sha256,
        splits_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "terminal": receipt["terminal"],
                "status": receipt["status"],
                "selected_pairs": receipt["source_prescreen_freeze"][
                    "sequence_and_pair_count"
                ],
                "cohort_identity_sha256": receipt["source_prescreen_freeze"][
                    "cohort_identity_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

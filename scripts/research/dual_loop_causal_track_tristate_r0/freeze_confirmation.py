from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from .common import (
    EXCLUDED_OUTCOME_OPEN_SEQUENCES,
    PROTOCOL_ID,
    SELECTED_SEQUENCES,
    SELECTION_SEED,
    WINDOW_FRAMES,
    sha256_file,
    write_exclusive,
)


EXPECTED_LABELS_INVENTORY_SHA256 = (
    "90894486eaedcde6342f0df1285492ddb80ff1db8a4d17c0ba952e5d00a422f3"
)
EXPECTED_TIMESTAMPS_SHA256 = (
    "60b440f4fd69b93a96b84ae07b957c169f8af304106a61b5f297286ed04b2e30"
)


def _sequence_members(
    inventory: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for row in inventory["entries"]:
        name = str(row["name"])
        role = None
        if name.startswith("labels/labels_2d_stitched/") and name.endswith(".json"):
            role = "source_2d"
        elif name.startswith("labels/labels_3d/") and name.endswith(".json"):
            role = "truth_3d"
        if role is None:
            continue
        sequence = Path(name).stem
        output.setdefault(sequence, {})[role] = {
            key: row[key]
            for key in (
                "name",
                "compression_method",
                "flags",
                "crc32",
                "compressed_size",
                "uncompressed_size",
                "local_header_offset",
            )
        }
    return output


def _timestamp_rows(archive: zipfile.ZipFile, sequence: str) -> list[dict[str, Any]]:
    member = f"timestamps/{sequence}/frames_img.json"
    payload = json.loads(archive.read(member))
    rows: list[dict[str, Any]] = []
    for position, frame in enumerate(payload["data"]):
        cameras = {
            str(row["name"]): row
            for row in frame["cameras"]
        }
        stitched = cameras.get("stitched_image0")
        if stitched is None:
            raise ValueError(f"stitched timestamp absent: {sequence}/{position}")
        stem = Path(str(stitched["url"])).stem
        timestamp_s = float(stitched["timestamp"])
        rows.append(
            {
                "position": position,
                "frame_stem": stem,
                "timestamp_ns": round(timestamp_s * 1_000_000_000),
            }
        )
    return rows


def run(
    labels_inventory: Path,
    timestamps_zip: Path,
    output: Path,
) -> dict[str, Any]:
    if sha256_file(labels_inventory) != EXPECTED_LABELS_INVENTORY_SHA256:
        raise ValueError("labels inventory drift")
    if sha256_file(timestamps_zip) != EXPECTED_TIMESTAMPS_SHA256:
        raise ValueError("timestamps archive drift")
    inventory = json.loads(labels_inventory.read_text(encoding="utf-8"))
    members = _sequence_members(inventory)
    candidates: list[tuple[str, str, list[dict[str, Any]]]] = []
    with zipfile.ZipFile(timestamps_zip) as archive:
        timestamp_sequences = {
            Path(name).parent.name
            for name in archive.namelist()
            if name.endswith("/frames_img.json")
        }
        for sequence in sorted(set(members) & timestamp_sequences):
            if sequence in EXCLUDED_OUTCOME_OPEN_SEQUENCES:
                continue
            if set(members[sequence]) != {"source_2d", "truth_3d"}:
                continue
            rows = _timestamp_rows(archive, sequence)
            if len(rows) < WINDOW_FRAMES:
                continue
            if any(
                right["timestamp_ns"] <= left["timestamp_ns"]
                for left, right in zip(rows, rows[1:])
            ):
                continue
            rank_hash = hashlib.sha256(
                f"{SELECTION_SEED}|sequence|{sequence}".encode("utf-8")
            ).hexdigest()
            candidates.append((rank_hash, sequence, rows))
    candidates.sort(key=lambda value: (value[0], value[1]))
    if len(candidates) < SELECTED_SEQUENCES:
        raise ValueError("insufficient metadata-only confirmation candidates")
    selected = []
    for rank_hash, sequence, rows in candidates[:SELECTED_SEQUENCES]:
        window_hash = hashlib.sha256(
            f"{SELECTION_SEED}|window|{sequence}".encode("utf-8")
        ).hexdigest()
        start = int(window_hash, 16) % (len(rows) - WINDOW_FRAMES + 1)
        window = rows[start : start + WINDOW_FRAMES]
        stems = [int(row["frame_stem"]) for row in window]
        if stems != list(range(stems[0], stems[0] + WINDOW_FRAMES)):
            raise ValueError(f"non-contiguous frozen stems: {sequence}")
        selected.append(
            {
                "sequence": sequence,
                "rank_hash": rank_hash,
                "window_hash": window_hash,
                "window_first_position": start,
                "window_last_position": start + WINDOW_FRAMES - 1,
                "frame_count": WINDOW_FRAMES,
                "frame_first_stem": window[0]["frame_stem"],
                "frame_last_stem": window[-1]["frame_stem"],
                "frames": window,
                "members": members[sequence],
            }
        )
    freeze = {
        "schema": "blindassist.dual_loop_causal_track_tristate_confirmation_freeze.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_BEFORE_SELECTED_LABEL_PAYLOAD_ACCESS",
        "selection_seed": SELECTION_SEED,
        "selection_rule": (
            "Exclude every outcome-open sequence, intersect stitched-2D/3D label "
            "members with strictly increasing stitched timestamps, rank by "
            "sha256(seed|sequence|id), then freeze a hash-selected 360-frame "
            "contiguous window for the first three sequences."
        ),
        "selected_sequence_count": SELECTED_SEQUENCES,
        "window_frames": WINDOW_FRAMES,
        "excluded_outcome_open_sequences": sorted(
            EXCLUDED_OUTCOME_OPEN_SEQUENCES
        ),
        "labels_inventory_sha256": sha256_file(labels_inventory),
        "timestamps_sha256": sha256_file(timestamps_zip),
        "source_payload_opened": False,
        "truth_payload_opened": False,
        "selected": selected,
        "claim_ceiling": "ANNOTATION_TRACK_MECHANISM_CONFIRMATION_ONLY",
    }
    write_exclusive(output, freeze)
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-inventory", type=Path, required=True)
    parser.add_argument("--timestamps-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.labels_inventory, args.timestamps_zip, args.output),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

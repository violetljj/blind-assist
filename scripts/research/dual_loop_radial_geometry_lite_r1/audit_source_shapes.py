#!/usr/bin/env python3
"""Audit decoded RGB shapes without reading candidate outcomes or truth."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_key(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def summarize(
    rows: list[dict[str, Any]],
    shapes_by_image: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    row_shapes = Counter(
        shape_key(shapes_by_image[str(row["image_relative_path"])])
        for row in rows
    )
    image_shapes = Counter(shape_key(shape) for shape in shapes_by_image.values())
    previous_by_target: dict[str, dict[str, Any]] = {}
    immediate_pairs = 0
    eligible_pairs = 0
    pair_count_by_target: Counter[str] = Counter()
    pair_count_by_region: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=1):
        target_id = str(row["target_id"])
        current_shape = shapes_by_image[str(row["image_relative_path"])]
        previous = previous_by_target.get(target_id)
        if previous is not None:
            immediate_pairs += 1
        if (
            previous is not None
            and not bool(row["history_reset"])
            and str(previous["track_epoch"]) == str(row["track_epoch"])
        ):
            eligible_pairs += 1
            pair_count_by_target[target_id] += 1
            pair_count_by_region[str(row["region"])] += 1
            previous_shape = shapes_by_image[str(previous["image_relative_path"])]
            if previous_shape != current_shape:
                mismatches.append({
                    "replay_line": line_number,
                    "target_id": target_id,
                    "track_epoch": str(row["track_epoch"]),
                    "previous_source_frame_id": str(previous["source_frame_id"]),
                    "current_source_frame_id": str(row["source_frame_id"]),
                    "previous_region": str(previous["region"]),
                    "current_region": str(row["region"]),
                    "previous_shape": shape_key(previous_shape),
                    "current_shape": shape_key(current_shape),
                    "previous_image_relative_path": str(previous["image_relative_path"]),
                    "current_image_relative_path": str(row["image_relative_path"]),
                    "delta_t_ns": (
                        int(row["captured_at_ns"])
                        - int(previous["captured_at_ns"])
                    ),
                })
        previous_by_target[target_id] = row
    canonical_shape = image_shapes.most_common(1)[0][0] if image_shapes else None
    unique_source_rows: dict[int, dict[str, Any]] = {}
    for row in rows:
        source_index = int(row["source_frame_index"])
        if source_index not in unique_source_rows:
            unique_source_rows[source_index] = row
    noncanonical = [
        (
            source_index,
            shape_key(shapes_by_image[str(row["image_relative_path"])]),
        )
        for source_index, row in sorted(unique_source_rows.items())
        if shape_key(shapes_by_image[str(row["image_relative_path"])])
        != canonical_shape
    ]
    segments: list[dict[str, Any]] = []
    for source_index, shape in noncanonical:
        if (
            not segments
            or source_index != int(segments[-1]["end_source_frame_index"]) + 1
            or shape != str(segments[-1]["shape"])
        ):
            segments.append({
                "start_source_frame_index": source_index,
                "end_source_frame_index": source_index,
                "shape": shape,
                "source_frame_count": 1,
            })
        else:
            segments[-1]["end_source_frame_index"] = source_index
            segments[-1]["source_frame_count"] = (
                int(segments[-1]["source_frame_count"]) + 1
            )
    transition_counts = Counter(
        f"{row['previous_shape']}->{row['current_shape']}"
        for row in mismatches
    )
    return {
        "replay_rows": len(rows),
        "unique_replay_images": len(shapes_by_image),
        "canonical_decoded_shape": canonical_shape,
        "decoded_image_shape_counts": dict(sorted(image_shapes.items())),
        "decoded_row_shape_counts": dict(sorted(row_shapes.items())),
        "same_target_immediate_pair_count": immediate_pairs,
        "same_target_same_epoch_pair_count": eligible_pairs,
        "same_target_same_epoch_pair_count_by_target": dict(
            sorted(pair_count_by_target.items())
        ),
        "same_target_same_epoch_pair_count_by_current_region": dict(
            sorted(pair_count_by_region.items())
        ),
        "shape_mismatch_pair_count": len(mismatches),
        "shape_mismatch_fraction": (
            len(mismatches) / eligible_pairs if eligible_pairs else 0.0
        ),
        "shape_mismatch_transition_counts": dict(sorted(transition_counts.items())),
        "shape_mismatch_by_target": dict(sorted(Counter(
            row["target_id"] for row in mismatches
        ).items())),
        "shape_mismatch_by_current_region": dict(sorted(Counter(
            row["current_region"] for row in mismatches
        ).items())),
        "shape_mismatch_target_epoch_count": len({
            (row["target_id"], row["track_epoch"]) for row in mismatches
        }),
        "shape_mismatch_same_region_count": sum(
            row["previous_region"] == row["current_region"]
            for row in mismatches
        ),
        "shape_mismatch_delta_t_ns_min": min(
            (row["delta_t_ns"] for row in mismatches),
            default=None,
        ),
        "shape_mismatch_delta_t_ns_max": max(
            (row["delta_t_ns"] for row in mismatches),
            default=None,
        ),
        "noncanonical_source_segment_count": len(segments),
        "noncanonical_source_segments": segments,
        "expected_common_shape_abstention_arm_rows": len(mismatches) * 2,
        "shape_mismatches": mismatches,
    }


def audit(
    replay_input: Path,
    image_root: Path,
    output: Path,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in replay_input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    image_root = image_root.resolve()
    shapes_by_image: dict[str, tuple[int, int]] = {}
    image_hashes: dict[str, str] = {}
    for row in rows:
        relative = Path(str(row["image_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("image path escapes root")
        name = relative.as_posix()
        if name in shapes_by_image:
            continue
        image_path = (image_root / relative).resolve()
        image_path.relative_to(image_root)
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"cannot decode {name}")
        shapes_by_image[name] = (int(gray.shape[0]), int(gray.shape[1]))
        image_hashes[name] = sha256_file(image_path)
    summary = summarize(rows, shapes_by_image)
    mismatch_images = sorted({
        str(row[field])
        for row in summary["shape_mismatches"]
        for field in (
            "previous_image_relative_path",
            "current_image_relative_path",
        )
    })
    result = {
        "schema": "blindassist_dual_loop_radial_geometry_lite_r1_shape_audit_v1",
        "status": "SOURCE_SHAPE_AUDIT_COMPLETE",
        "replay_input_path": replay_input.as_posix(),
        "replay_input_sha256": sha256_file(replay_input),
        "image_root": image_root.as_posix(),
        **summary,
        "mismatch_image_sha256": {
            name: image_hashes[name] for name in mismatch_images
        },
        "candidate_output_accessed": False,
        "truth_or_event_accessed": False,
        "old_f1b_decision_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.replay_input, args.image_root, args.output)
    print(json.dumps({
        "status": result["status"],
        "pairs": result["same_target_same_epoch_pair_count"],
        "shape_mismatches": result["shape_mismatch_pair_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

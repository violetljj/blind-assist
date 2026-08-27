from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import artvideo_ocr_replay as replay
import artvideo_semantic_visual_replay as sc0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen binding artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _canonical(rows: list[dict[str, Any]]) -> tuple[str, str] | None:
    observed = [
        str(row["Transcription"])
        for row in rows
        if replay.normalize(str(row["Transcription"]))
        and replay.normalize(str(row["Transcription"])) != replay.normalize("###")
    ]
    if not observed:
        return None
    normalized_counts = Counter(replay.normalize(value) for value in observed)
    normalized = min(
        normalized_counts,
        key=lambda value: (-normalized_counts[value], value),
    )
    spellings = Counter(value for value in observed if replay.normalize(value) == normalized)
    display = min(spellings, key=lambda value: (-spellings[value], value))
    return normalized, display


def freeze_bindings(
    dataset: Path,
    videos: list[str],
    output_dir: Path,
    minimum_frames: int,
    minimum_overlap_frames: int,
    max_bindings: int,
) -> None:
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    source_annotations: list[dict[str, Any]] = []
    candidates: list[tuple[Any, ...]] = []
    for video in videos:
        annotation_path = dataset / "Test/json" / f"{video}.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        source_annotations.append(
            {
                "video": video,
                "path": str(annotation_path),
                "sha256": _sha256(annotation_path),
            }
        )
        by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in annotation["annotations"]:
            by_id[int(row["obj_id"])].append(row)
        identity: dict[int, tuple[str, str]] = {}
        frames_by_id: dict[int, set[int]] = {}
        row_by_id_frame: dict[tuple[int, int], dict[str, Any]] = {}
        for object_id, rows in by_id.items():
            canonical = _canonical(rows)
            if canonical is None:
                continue
            identity[object_id] = canonical
            frames_by_id[object_id] = {int(row["frame_id"]) for row in rows}
            for row in rows:
                row_by_id_frame[(object_id, int(row["frame_id"]))] = row
        by_goal: dict[str, list[int]] = defaultdict(list)
        for object_id, (normalized, _display) in identity.items():
            by_goal[normalized].append(object_id)
        for normalized, object_ids in by_goal.items():
            if len(object_ids) < 2:
                continue
            for target_id in sorted(object_ids):
                target_frames = frames_by_id[target_id]
                if len(target_frames) < minimum_frames:
                    continue
                distractors = [object_id for object_id in object_ids if object_id != target_id]
                overlap_frames = sorted(
                    frame_id
                    for frame_id in target_frames
                    if any(frame_id in frames_by_id[object_id] for object_id in distractors)
                )
                if len(overlap_frames) < minimum_overlap_frames:
                    continue
                readable_overlap = [
                    frame_id
                    for frame_id in overlap_frames
                    if replay.normalize(
                        str(row_by_id_frame[(target_id, frame_id)]["Transcription"])
                    )
                    == normalized
                ]
                if not readable_overlap:
                    continue
                anchor_frame_id = readable_overlap[0]
                anchor_row = row_by_id_frame[(target_id, anchor_frame_id)]
                box = np.asarray(anchor_row["point"], dtype=np.float32)
                candidates.append(
                    (
                        video,
                        normalized,
                        identity[target_id][1],
                        target_id,
                        sorted(distractors),
                        sorted(target_frames),
                        overlap_frames,
                        anchor_frame_id,
                        box.astype(float).tolist(),
                    )
                )
    candidates.sort(key=lambda row: (row[0], row[1], row[3]))
    selected = candidates[:max_bindings]
    for index, row in enumerate(selected, start=1):
        (
            video,
            normalized,
            goal_text,
            target_id,
            distractor_ids,
            target_frames,
            overlap_frames,
            anchor_frame_id,
            anchor_box,
        ) = row
        binding_id = f"sc6-ref-{index:03d}"
        public_rows.append(
            {
                "binding_id": binding_id,
                "binding_type": "REFERENCE_IMAGE_INSTANCE",
                "task_cardinality": "UNIQUE",
                "goal_text": goal_text,
                "goal_normalized": normalized,
                "video": video,
                "anchor_frame_id": anchor_frame_id,
                "anchor_frame_key": f"Test/frame/{video}/{anchor_frame_id}.jpg",
                "anchor_box_quadrilateral": anchor_box,
                "anchor_crop_context_fraction_per_axis": 0.25,
                "anchor_crop_path": f"public-anchor-crops/{binding_id}.png",
                "opaque_anchor": hashlib.sha256(
                    f"{binding_id}|{video}|{normalized}|{anchor_frame_id}|{anchor_box}".encode("utf-8")
                ).hexdigest(),
            }
        )
        private_rows.append(
            {
                "binding_id": binding_id,
                "native_physical_id": target_id,
                "same_goal_distractor_ids": distractor_ids,
                "target_frame_ids": target_frames,
                "co_visible_same_goal_frame_ids": overlap_frames,
            }
        )
    frozen_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    public = {
        "schema": "l10_sc6_public_instance_binding_contract_v0",
        "status": "FROZEN_BEFORE_ANCHOR_PIXEL_DECODE_OCR_EMBEDDING_OR_OUTCOME_ACCESS",
        "frozen_at_utc": frozen_at,
        "selection_rule": {
            "minimum_target_frames": minimum_frames,
            "minimum_co_visible_same_goal_frames": minimum_overlap_frames,
            "maximum_bindings": max_bindings,
            "order": "video, normalized goal, native physical ID; take deterministic prefix",
            "anchor": "earliest co-visible frame where the target's public transcription equals its canonical normalized goal",
        },
        "source_annotations": source_annotations,
        "bindings": public_rows,
        "public_input_firewall": (
            "Contains goal text, public anchor frame/box/crop plan, and an opaque anchor only. "
            "It contains no native physical ID, distractor ID, later truth box, or evaluator outcome."
        ),
    }
    private = {
        "schema": "l10_sc6_private_instance_binding_truth_v0",
        "status": "EVALUATOR_PRIVATE",
        "frozen_at_utc": frozen_at,
        "bindings": private_rows,
    }
    _write_new(output_dir / "public-binding-contract.json", public)
    _write_new(output_dir / "private-binding-truth.json", private)
    print(
        json.dumps(
            {
                "status": "PUBLIC_INSTANCE_BINDINGS_FROZEN",
                "eligible_candidates": len(candidates),
                "bindings": len(public_rows),
            },
            ensure_ascii=False,
        )
    )


def materialize_crops(dataset: Path, output_dir: Path) -> None:
    contract_path = output_dir / "public-binding-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    materialized_rows: list[dict[str, Any]] = []
    for binding in contract["bindings"]:
        frame_path = dataset / Path(binding["anchor_frame_key"])
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(frame_path)
        height, width = image.shape[:2]
        box = np.asarray(binding["anchor_box_quadrilateral"], dtype=np.float32)
        x0, y0, x1, y1 = sc0._crop_bounds(box, width, height, context=0.25)
        crop = image[y0:y1, x0:x1]
        crop_path = output_dir / binding["anchor_crop_path"]
        if crop_path.exists():
            raise FileExistsError(f"refusing to overwrite public anchor crop: {crop_path}")
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(f"failed to write anchor crop: {crop_path}")
        materialized_rows.append(
            {
                **binding,
                "anchor_frame_sha256": _sha256(frame_path),
                "anchor_crop_sha256": _sha256(crop_path),
                "anchor_crop_xyxy": [x0, y0, x1, y1],
                "anchor_crop_shape": list(crop.shape),
            }
        )
    result = {
        "schema": "l10_sc6_public_instance_bindings_v0",
        "status": "PUBLIC_REFERENCE_CROPS_MATERIALIZED",
        "binding_contract_path": str(contract_path),
        "binding_contract_sha256": _sha256(contract_path),
        "materialized_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "bindings": materialized_rows,
        "public_input_firewall": contract["public_input_firewall"],
    }
    _write_new(output_dir / "public-bindings.json", result)
    print(json.dumps({"status": result["status"], "bindings": len(materialized_rows)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze or materialize L10-SC6 public instance bindings.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--videos", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "materialize"), required=True)
    parser.add_argument("--minimum-frames", type=int, default=12)
    parser.add_argument("--minimum-overlap-frames", type=int, default=8)
    parser.add_argument("--max-bindings", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "freeze":
        if not args.videos:
            parser.error("--videos is required for --phase freeze")
        freeze_bindings(
            args.dataset,
            args.videos,
            args.output_dir,
            args.minimum_frames,
            args.minimum_overlap_frames,
            args.max_bindings,
        )
    else:
        materialize_crops(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()

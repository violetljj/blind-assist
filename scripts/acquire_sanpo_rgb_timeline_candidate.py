#!/usr/bin/env python3
"""Download a sparse public SANPO RGB timeline for an explicit research role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_sanpo_sequence_evalset import (
    GCS_PREFIX,
    download,
    frame_number,
    list_gcs_objects,
    media_url,
    object_inventory,
    resample_indices,
    sha256_file,
    verify_gcs_md5,
)
from screen_sanpo_mask_windows import source_fps_for


class CandidateError(ValueError):
    pass


def build_candidate_spec(
    *,
    session_id: str,
    start_frame: int,
    target_fps: float,
    frame_count: int,
    purpose: str,
    camera: str = "camera_chest",
    lens: str = "left",
    official_split: str = "train",
    source_fps: float = 15.0,
) -> dict[str, Any]:
    source = {
        "dataset": "SANPO-Real v0",
        "official_split": official_split,
        "session_id": session_id,
        "camera": camera,
        "lens": lens,
        "license": "CC-BY-4.0",
    }
    sampling = {
        "source_fps": source_fps,
        "target_fps": target_fps,
        "frame_count": frame_count,
        "start_frame": start_frame,
    }
    if purpose == "event-eval-screening":
        return {
            "format": "blindassist_sanpo_rgb_timeline_candidate_v3",
            "source": source,
            "sampling": sampling,
            "status": "output_blind_event_eval_rgb_screening_only",
            "pixel_masks_downloaded": False,
            "human_event_truth_present": False,
            "training_authorized": False,
            "candidate_output_accessed": False,
            "important_limit": (
                "Sparse RGB is only a cost-control screen for a future full event "
                "window. It is not frozen event truth, training data, calibration "
                "input, or promotion evidence."
            ),
        }
    return {
        "format": "blindassist_sanpo_rgb_timeline_candidate_v2",
        "source": source,
        "sampling": sampling,
        "status": "provisional_model_supervision_candidate",
        "pixel_masks_downloaded": False,
        "human_event_truth_present": False,
        "provisional_training_authorized": True,
        "production_model_replacement_authorized": False,
        "important_limit": (
            "Public RGB plus provisional model labels may support provisional "
            "training only; it is not independent GPT/Codex consensus truth, a "
            "calibration input, blind-evaluation truth, or production-replacement "
            "evidence."
        ),
    }


def select_rgb_window(objects: list[dict[str, Any]], *, source_fps: float, target_fps: float, start_frame: int, frame_count: int) -> list[tuple[int, dict[str, Any]]]:
    by_index = {frame_number(str(item["name"])): item for item in objects if str(item.get("name", "")).endswith(".png")}
    selected = resample_indices(by_index, source_fps, target_fps, start_frame, frame_count)
    if len(selected) != frame_count:
        raise CandidateError(f"only {len(selected)} RGB frames available; requested {frame_count}")
    return [(index, by_index[index]) for index in selected]


def assert_output_can_resume(output_root: Path) -> None:
    if any((output_root / name).exists() for name in ("manifest.rgb_timeline.jsonl", "candidate_spec.json")):
        raise CandidateError(f"refusing to overwrite completed output root: {output_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--frame-count", type=int, default=10)
    parser.add_argument("--camera", choices=("camera_chest", "camera_head"), default="camera_chest")
    parser.add_argument("--lens", choices=("left", "right"), default="left")
    parser.add_argument("--official-split", choices=("train", "test"), default="train")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--purpose",
        choices=("provisional-supervision", "event-eval-screening"),
        default="provisional-supervision",
    )
    args = parser.parse_args()
    try:
        if args.start_frame < 0 or args.target_fps <= 0 or args.target_fps > 10 or args.frame_count <= 0 or args.retries <= 0:
            raise CandidateError("invalid window arguments")
        assert_output_can_resume(args.output_root)
        prefix = (
            f"{GCS_PREFIX}/sanpo-real/{args.session_id}/"
            f"{args.camera}/{args.lens}/video_frames/"
        )
        source_fps = source_fps_for(args.session_id, args.camera, args.retries)
        selected = select_rgb_window(
            list_gcs_objects(prefix, retries=args.retries),
            source_fps=source_fps,
            target_fps=args.target_fps,
            start_frame=args.start_frame,
            frame_count=args.frame_count,
        )
        images = args.output_root / "images"
        rows: list[dict[str, Any]] = []
        for timeline_index, (source_index, item) in enumerate(selected):
            path = images / f"{timeline_index:04d}_{source_index:06d}.png"
            download(media_url(str(item["name"]), item.get("generation")), path, retries=args.retries)
            verify_gcs_md5(path, item)
            row = {
                "timeline_index": timeline_index,
                "source_frame_index": source_index,
                "timeline_timestamp_ms": int(round(timeline_index * 1000 / args.target_fps)),
                "image_path": path.relative_to(args.output_root).as_posix(),
                "image_sha256": sha256_file(path),
                "source": object_inventory(item),
                "event_truth": None,
            }
            if args.purpose == "event-eval-screening":
                row.update({
                    "role": "output_blind_rgb_cost_control_screen_only",
                    "model_output_accessed": False,
                    "training_authorized": False,
                })
            else:
                row["model_assisted_candidate_screening_only"] = True
            rows.append(row)
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "manifest.rgb_timeline.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        spec = build_candidate_spec(
            session_id=args.session_id,
            start_frame=args.start_frame,
            target_fps=args.target_fps,
            frame_count=args.frame_count,
            purpose=args.purpose,
            camera=args.camera,
            lens=args.lens,
            official_split=args.official_split,
            source_fps=source_fps,
        )
        (args.output_root / "candidate_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "ok": True,
            "output_root": str(args.output_root.resolve()),
            "frame_count": len(rows),
            "purpose": args.purpose,
        }, ensure_ascii=False))
    except (CandidateError, OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate post-goal physical captures and materialize outcome-blind PA3 frames."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import content_sha256


C0_SCHEMA = "blindassist_p1_pa3_c0_public_goal_cohort_v1"
CAPTURE_RECEIPT_SCHEMA = "blindassist_p1_pa3_prospective_first_person_capture_v1"
CAPTURE_PLAN_SCHEMA = "blindassist_p1_pa3_prospective_first_person_capture_plan_v1"
CAPTURE_MANIFEST_SCHEMA = "blindassist_p1_pa3_capture_manifest_v1"
SOURCE_ROLE = "PROSPECTIVE_FIRST_PERSON_PHYSICAL_CAPTURE_AFTER_GOAL"
CAPTURE_INSTRUCTION = "APPROACH_NAMED_BUILDING_AND_STOP_AFTER_ENTRANCE_IS_IN_VIEW_V1"
FRAME_OFFSETS_FROM_END_SECONDS = (-2.5, -1.5, -0.5)
MINIMUM_EPISODES = 5
MINIMUM_VISIBLE_FRAMES_FOR_LATER_AUTHORIZATION = 8
MINIMUM_DURATION_SECONDS = 3.0
MAXIMUM_DURATION_SECONDS = 45.0
FORBIDDEN_CAPTURE_KEYS = {
    "bbox", "bbox_xyxy", "mask", "target_bbox_xyxy", "target_mask", "target_category",
    "instance_name", "object_uid", "referent_id", "truth", "evaluator", "visibility",
    "selected_frame", "best_frame", "proposal", "prediction", "model_output",
}


class ProspectiveCaptureError(ValueError):
    """Raised when physical-capture provenance or extraction fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProspectiveCaptureError(message)


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty text")
    return value.strip()


def _timestamp(value: Any, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProspectiveCaptureError(f"{label} must be ISO-8601") from error
    _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{label} must include timezone")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_forbidden_keys(value: Any, path: str = "capture") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(str(key) not in FORBIDDEN_CAPTURE_KEYS, f"{path}.{key} is truth/outcome-derived")
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")


def _verify_hashed_body(value: Mapping[str, Any], hash_key: str, label: str) -> str:
    declared = _text(value.get(hash_key), hash_key)
    body = dict(value)
    body.pop(hash_key, None)
    _require(content_sha256(body) == declared, f"{label} body hash mismatch")
    return declared


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _default_probe(path: Path) -> dict[str, Any]:
    try:
        import av
    except ImportError as error:  # pragma: no cover - environment failure
        raise ProspectiveCaptureError("PyAV is required for capture materialization") from error
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams.video]
        _require(len(streams) == 1, "capture must contain exactly one video stream")
        stream = streams[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        if duration is None and container.duration is not None:
            duration = float(container.duration / av.time_base)
        _require(duration is not None, "capture duration is unavailable")
        return {"width": int(stream.width), "height": int(stream.height), "duration_seconds": duration}


def _default_extract(path: Path, timestamps: Sequence[float], outputs: Sequence[Path]) -> list[float]:
    try:
        import av
    except ImportError as error:  # pragma: no cover - environment failure
        raise ProspectiveCaptureError("PyAV is required for frame extraction") from error
    targets = list(timestamps)
    best: list[tuple[float, float, Any] | None] = [None] * len(targets)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            seconds = float(frame.pts * frame.time_base)
            for index, target in enumerate(targets):
                distance = abs(seconds - target)
                if distance <= 0.25 and (best[index] is None or distance < best[index][0]):
                    best[index] = (distance, seconds, frame.to_image())
    _require(all(row is not None for row in best), "capture did not yield every deterministic frame")
    actual = []
    for row, output, target in zip(best, outputs, targets, strict=True):
        assert row is not None
        _require(row[0] <= 0.25, f"no decoded frame within 250 ms of {target:.3f}s")
        output.parent.mkdir(parents=True, exist_ok=True)
        row[2].save(output, format="JPEG", quality=95, optimize=False, progressive=False)
        actual.append(row[1])
    return actual


def materialize_capture(
    *,
    c0: Mapping[str, Any],
    capture_plan: Mapping[str, Any],
    capture_receipt: Mapping[str, Any],
    capture_root: Path,
    output_dir: Path,
    probe_media: Callable[[Path], Mapping[str, Any]] = _default_probe,
    extract_frames: Callable[[Path, Sequence[float], Sequence[Path]], Sequence[float]] = _default_extract,
) -> Path:
    capture_root = capture_root.resolve()
    output_dir = output_dir.resolve()
    _require(capture_root.is_dir(), "capture root does not exist")
    _require(not output_dir.exists(), "capture materialization output already exists")
    _require(c0.get("schema_version") == C0_SCHEMA, "C0 receipt schema mismatch")
    c0_body_sha = _verify_hashed_body(c0, "receipt_body_sha256", "C0 receipt")
    _require(c0.get("private_truth_access") is False, "C0 receipt must be provider-public")
    _require(c0.get("pa3_inference_authorized") is False, "C0 authority drift")
    _require(capture_plan.get("schema_version") == CAPTURE_PLAN_SCHEMA, "capture plan schema mismatch")
    capture_plan_sha = _verify_hashed_body(capture_plan, "capture_plan_body_sha256", "capture plan")
    _reject_forbidden_keys(capture_plan, "capture_plan")
    _require(capture_plan.get("goal_receipt_body_sha256") == c0_body_sha, "capture plan is not bound to C0")
    _require(capture_plan.get("source_role") == SOURCE_ROLE, "capture plan source role mismatch")
    _require(capture_plan.get("capture_instruction_id") == CAPTURE_INSTRUCTION, "capture plan instruction drift")
    _require(capture_plan.get("truth_state_at_arming") == "NOT_CREATED", "truth existed at capture arming")
    _require(capture_plan.get("provider_state_at_arming") == "NOT_STARTED", "provider ran before capture arming")
    _require(capture_plan.get("capture_state_at_arming") == "NOT_STARTED", "capture started before arming")
    _require(capture_plan.get("frame_selection_rule") == "FIXED_OFFSETS_FROM_VIDEO_END_NO_PIXEL_OR_TRUTH_SELECTION", "capture plan frame-selection rule drift")
    _require(list(capture_plan.get("frame_offsets_from_end_seconds", [])) == list(FRAME_OFFSETS_FROM_END_SECONDS), "capture plan frame offsets drift")
    armed_at = _timestamp(capture_plan.get("armed_at_utc"), "armed_at_utc")
    _require(capture_receipt.get("schema_version") == CAPTURE_RECEIPT_SCHEMA, "capture receipt schema mismatch")
    capture_body_sha = _verify_hashed_body(capture_receipt, "capture_body_sha256", "capture receipt")
    _reject_forbidden_keys(capture_receipt)
    _require(capture_receipt.get("goal_receipt_body_sha256") == c0_body_sha, "capture receipt is not bound to C0")
    _require(capture_receipt.get("capture_plan_body_sha256") == capture_plan_sha, "capture receipt is not bound to capture plan")
    _require(capture_receipt.get("source_role") == SOURCE_ROLE, "capture source role mismatch")
    _require(capture_receipt.get("recorder_authority") == "DEVICE_OWNED_CONTINUOUS_VIDEO_RECORDER", "capture recorder authority mismatch")
    _require(capture_receipt.get("capture_instruction_id") == CAPTURE_INSTRUCTION, "capture instruction drift")
    _require(capture_receipt.get("truth_state_at_capture") == "NOT_CREATED", "truth existed during capture")
    _require(capture_receipt.get("provider_state_at_capture") == "NOT_STARTED", "provider ran before capture")
    _require(capture_receipt.get("continuous_capture_required") is True, "continuous capture contract drift")
    _require(capture_receipt.get("frame_selection_rule") == "FIXED_OFFSETS_FROM_VIDEO_END_NO_PIXEL_OR_TRUTH_SELECTION", "frame-selection rule drift")
    _require(list(capture_receipt.get("frame_offsets_from_end_seconds", [])) == list(FRAME_OFFSETS_FROM_END_SECONDS), "frame offsets drift")
    receipt_created_at = _timestamp(capture_receipt.get("receipt_created_at_utc"), "receipt_created_at_utc")

    c0_rows = c0.get("episodes")
    rows = capture_receipt.get("episodes")
    _require(isinstance(c0_rows, list) and len(c0_rows) >= MINIMUM_EPISODES, f"C0 must freeze at least {MINIMUM_EPISODES} episodes")
    _require(isinstance(rows, list) and bool(rows), "capture receipt episodes are required")
    _require(capture_receipt.get("episode_count") == len(rows), "capture receipt episode count drift")
    goals = {row["episode_id"]: row for row in c0_rows}
    _require(len(goals) == len(c0_rows), "duplicate C0 episode_id")
    by_episode = {row.get("episode_id"): row for row in rows if isinstance(row, Mapping)}
    _require(len(by_episode) == len(rows), "duplicate or invalid capture episode")
    _require(set(by_episode) == set(goals), "capture receipt must cover the entire frozen C0 roster")
    plan_rows = capture_plan.get("episodes")
    _require(isinstance(plan_rows, list), "capture plan episodes are required")
    _require(capture_plan.get("episode_count") == len(plan_rows), "capture plan episode count drift")
    plan_by_episode = {row.get("episode_id"): row for row in plan_rows if isinstance(row, Mapping)}
    _require(set(plan_by_episode) == set(goals) and len(plan_by_episode) == len(plan_rows), "capture plan roster drift")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    cases: list[dict[str, Any]] = []
    seen_media_sha256: set[str] = set()
    try:
        for episode_id, goal in goals.items():
            row = by_episode[episode_id]
            goal_at = _timestamp(goal["goal_provenance"]["goal_recorded_at_utc"], f"{episode_id} goal_recorded_at_utc")
            started_at = _timestamp(row.get("capture_started_at_utc"), f"{episode_id} capture_started_at_utc")
            completed_at = _timestamp(row.get("capture_completed_at_utc"), f"{episode_id} capture_completed_at_utc")
            _require(goal_at < armed_at < started_at < completed_at <= receipt_created_at, f"{episode_id} goal/arm/capture/receipt precedence failed")
            _require(row.get("camera_view") == "FIRST_PERSON_FORWARD", f"{episode_id} camera view mismatch")
            _require(row.get("continuous_capture") is True, f"{episode_id} capture is not continuous")
            _require(plan_by_episode[episode_id].get("camera_view") == row.get("camera_view"), f"{episode_id} camera view differs from armed plan")
            _require(plan_by_episode[episode_id].get("continuous_capture") == row.get("continuous_capture"), f"{episode_id} continuity differs from armed plan")
            _require(row.get("media_path") == plan_by_episode[episode_id].get("media_relative_path"), f"{episode_id} media path differs from armed plan")
            media = Path(_text(row.get("media_path"), f"{episode_id} media_path"))
            if not media.is_absolute():
                media = (capture_root / media).resolve()
            else:
                media = media.resolve()
            _require(_inside(media, capture_root), f"{episode_id} media escapes capture root")
            _require(media.is_file(), f"{episode_id} media is missing")
            media_sha256 = _text(row.get("media_sha256"), f"{episode_id} media_sha256")
            _require(_sha256(media) == media_sha256, f"{episode_id} media hash mismatch")
            _require(media_sha256 not in seen_media_sha256, f"{episode_id} reuses another episode's physical capture")
            seen_media_sha256.add(media_sha256)
            probed = probe_media(media)
            width, height = int(probed["width"]), int(probed["height"])
            duration = float(probed["duration_seconds"])
            _require(width > 0 and height > 0, f"{episode_id} media dimensions invalid")
            _require(MINIMUM_DURATION_SECONDS <= duration <= MAXIMUM_DURATION_SECONDS, f"{episode_id} duration outside frozen range")
            _require(abs((completed_at - started_at).total_seconds() - duration) <= 1.0, f"{episode_id} recorder timeline and media duration disagree")
            _require(width == int(row.get("width")) and height == int(row.get("height")), f"{episode_id} media dimensions drift")
            _require(abs(duration - float(row.get("duration_seconds"))) <= 0.10, f"{episode_id} media duration drift")
            timestamps = [duration + offset for offset in FRAME_OFFSETS_FROM_END_SECONDS]
            outputs = [temporary / "images" / f"{episode_id}-frame-{index:02d}.jpg" for index in range(1, 4)]
            actual = list(extract_frames(media, timestamps, outputs))
            _require(len(actual) == len(outputs), f"{episode_id} extractor result count mismatch")
            for index, (target, actual_seconds, image) in enumerate(zip(timestamps, actual, outputs, strict=True), start=1):
                _require(image.is_file(), f"{episode_id} extracted frame is missing")
                _require(abs(float(actual_seconds) - target) <= 0.25, f"{episode_id} extracted frame timestamp drift")
                physical_time = started_at + timedelta(seconds=float(actual_seconds))
                cases.append({
                    "case_id": f"{episode_id}-frame-{index:02d}",
                    "episode_id": episode_id,
                    "image_path": str(Path("images") / image.name),
                    "image_sha256": _sha256(image),
                    "capture_created_at_utc": physical_time.isoformat(),
                    "capture_time_semantics": "PHYSICAL_DEVICE_CAPTURE_TIMELINE_DERIVED_FROM_VIDEO_PTS",
                    "source_media_sha256": media_sha256,
                    "source_video_timestamp_seconds": float(actual_seconds),
                    "frame_selection_rule": "FIXED_OFFSETS_FROM_VIDEO_END_NO_PIXEL_OR_TRUTH_SELECTION",
                })
        manifest = {
            "schema_version": CAPTURE_MANIFEST_SCHEMA,
            "precedence_mode": "PHYSICAL_CAPTURE_AFTER_GOAL",
            "physical_capture_after_goal_claimed": True,
            "goal_receipt_body_sha256": c0_body_sha,
            "prospective_capture_receipt_body_sha256": capture_body_sha,
            "prospective_capture_plan_body_sha256": capture_plan_sha,
            "source_role": SOURCE_ROLE,
            "capture_instruction_id": CAPTURE_INSTRUCTION,
            "frame_offsets_from_end_seconds": list(FRAME_OFFSETS_FROM_END_SECONDS),
            "episode_count": len(rows),
            "case_count": len(cases),
            "minimum_visible_episode_count_for_later_pa3_authorization": MINIMUM_EPISODES,
            "minimum_visible_frame_count_for_later_pa3_authorization": MINIMUM_VISIBLE_FRAMES_FOR_LATER_AUTHORIZATION,
            "private_truth_access": False,
            "provider_model_calls": 0,
            "cases": cases,
        }
        manifest["capture_manifest_body_sha256"] = content_sha256(manifest)
        manifest_path = temporary / "capture_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
        return output_dir / "capture_manifest.json"
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-receipt", required=True, type=Path)
    parser.add_argument("--capture-receipt", required=True, type=Path)
    parser.add_argument("--capture-plan", required=True, type=Path)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = materialize_capture(
        c0=json.loads(args.c0_receipt.read_text(encoding="utf-8")),
        capture_plan=json.loads(args.capture_plan.read_text(encoding="utf-8")),
        capture_receipt=json.loads(args.capture_receipt.read_text(encoding="utf-8")),
        capture_root=args.capture_root,
        output_dir=args.output_dir,
    )
    print(json.dumps({"capture_manifest": str(manifest), "capture_manifest_sha256": _sha256(manifest), "provider_model_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

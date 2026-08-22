#!/usr/bin/env python3
"""Materialize a goal-before-pixels PA3 cohort from DeepDoors2 without manual curation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    content_sha256,
    sha256,
)


ROOT_FOLDER_ID = "1SxVKeJ9RBcoJXHSHw-LWaLGG07BZT-b5"
DOORDETECT_REPOSITORY = "MiguelARD/DoorDetect-Dataset"
DOORDETECT_REVISION = "1db107fe1b808fc5712f898b35bde0976ba0c0af"
ROSTER_SCHEMA = "blindassist_p1_pa3_automated_public_dataset_roster_v1"
CAPTURE_SCHEMA = "blindassist_p1_pa3_capture_manifest_v1"
TRUTH_SCHEMA = "blindassist_p1_pa3_truth_body_v1"
PRECEDENCE_MODE = "GOAL_BEFORE_FIRST_PROJECT_PIXEL_ACCESS_AND_TRUTH"
CAPTURE_TIME_SEMANTICS = "FIRST_PROJECT_PIXEL_ACCESS_NOT_PHYSICAL_CAMERA_CAPTURE"
DOOR_RGB = np.asarray([192, 224, 192], dtype=np.uint8)


class PublicDatasetError(ValueError):
    """Raised when automatic public-dataset materialization violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicDatasetError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sealed_payload(payload: dict[str, Any], hash_key: str) -> dict[str, Any]:
    _require(hash_key not in payload, f"duplicate {hash_key}")
    payload[hash_key] = content_sha256(payload)
    return payload


def _validate_seal(value: Mapping[str, Any], hash_key: str) -> None:
    declared = value.get(hash_key)
    _require(isinstance(declared, str) and len(declared) == 64, f"{hash_key} is missing")
    body = dict(value)
    body.pop(hash_key)
    _require(content_sha256(body) == declared, f"{hash_key} mismatch")


def _gdown_items() -> list[Any]:
    try:
        import gdown
    except ImportError as error:
        raise PublicDatasetError("gdown is required on PYTHONPATH for source metadata/download") from error
    items = gdown.download_folder(
        id=ROOT_FOLDER_ID,
        output="deepdoors2",
        quiet=True,
        remaining_ok=True,
        skip_download=True,
    )
    _require(items is not None, "DeepDoors2 folder metadata is unavailable")
    return list(items)


def freeze_roster(source_lock_path: Path, c0_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "roster is immutable and already exists")
    source_lock = _read_json(source_lock_path)
    c0 = _read_json(c0_path)
    _require(source_lock.get("created_before_project_pixel_access") is True, "source lock precedence is missing")
    _require(source_lock.get("created_before_private_annotation_access") is True, "source lock truth precedence is missing")
    _require(source_lock.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "source lock C0 binding mismatch")
    take = int(source_lock.get("roster_rule", {}).get("take", 0))
    _require(take == len(c0.get("episodes", [])) and take > 0, "roster size must equal frozen C0 episodes")

    images: dict[str, Any] = {}
    annotations: dict[str, Any] = {}
    for item in _gdown_items():
        path = str(getattr(item, "path", "")).replace("\\", "/")
        name = PurePosixPath(path).name
        stem = PurePosixPath(path).stem
        if path.startswith("Door Detection|Segmentation/Images/") and name.lower().endswith(".png"):
            images[stem] = item
        elif path.startswith("Door Detection|Segmentation/Annotations/") and name.lower().endswith(".png"):
            annotations[stem] = item
    eligible = sorted(set(images) & set(annotations))
    _require(len(eligible) >= take, "DeepDoors2 has fewer paired files than the frozen roster requires")
    ordered = sorted(
        eligible,
        key=lambda stem: hashlib.sha256(str(getattr(images[stem], "path")).replace("\\", "/").encode("utf-8")).hexdigest(),
    )
    cases = []
    for index, stem in enumerate(ordered[:take], start=1):
        image_path = str(getattr(images[stem], "path")).replace("\\", "/")
        cases.append({
            "case_id": f"s0v4-auto-case-{index:03d}",
            "episode_id": f"s0v4-auto-{index:03d}",
            "source_stem": stem,
            "image_member_path": image_path,
            "image_member_path_sha256": hashlib.sha256(image_path.encode("utf-8")).hexdigest(),
            "image_drive_id": str(getattr(images[stem], "id")),
            "annotation_member_path": str(getattr(annotations[stem], "path")).replace("\\", "/"),
            "annotation_drive_id": str(getattr(annotations[stem], "id")),
        })
    roster = _sealed_payload({
        "schema_version": ROSTER_SCHEMA,
        "created_at_utc": _utc_now(),
        "created_before_selected_pixel_download": True,
        "created_before_selected_annotation_download": True,
        "private_truth_access": False,
        "source_lock_sha256": sha256(source_lock_path),
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "eligible_pair_count": len(eligible),
        "selection_rule": source_lock["roster_rule"],
        "cases": cases,
    }, "roster_body_sha256")
    _atomic_json(output_path, roster)
    return roster


def _download(file_id: str, destination: Path) -> None:
    try:
        import gdown
    except ImportError as error:
        raise PublicDatasetError("gdown is required on PYTHONPATH for source metadata/download") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            with Image.open(destination) as image:
                image.verify()
            return
        except Exception as error:
            raise PublicDatasetError(f"existing partial download is invalid: {destination}") from error
    try:
        result = gdown.download(id=file_id, output=str(destination), quiet=True)
    except Exception:
        result = None
    if result is None:
        import requests

        response = requests.get(
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            timeout=60,
        )
        _require(response.status_code == 200, f"download HTTP {response.status_code} for {destination.name}")
        _require(response.headers.get("content-type", "").lower().startswith("image/"), f"download is not an image for {destination.name}")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(response.content)
        with Image.open(temporary) as image:
            image.verify()
        os.replace(temporary, destination)
    _require(destination.is_file(), f"download failed for {destination.name}")


def download_public(roster_path: Path, c0_path: Path, output_root: Path, capture_path: Path) -> dict[str, Any]:
    _require(not capture_path.exists(), "capture manifest is immutable and already exists")
    roster = _read_json(roster_path)
    c0 = _read_json(c0_path)
    _validate_seal(roster, "roster_body_sha256")
    _require(roster.get("private_truth_access") is False, "roster accessed private truth")
    _require(roster.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "roster C0 binding mismatch")
    goal_times = {row["episode_id"]: row["goal_provenance"]["goal_recorded_at_utc"] for row in c0["episodes"]}
    cases = []
    for raw in roster["cases"]:
        destination = (output_root / "public_images" / f"{raw['source_stem']}.png").resolve()
        _download(raw["image_drive_id"], destination)
        with Image.open(destination) as image:
            image.verify()
        accessed_at = _utc_now()
        _require(goal_times[raw["episode_id"]] < accessed_at, "goal must precede first project pixel access")
        cases.append({
            "case_id": raw["case_id"],
            "episode_id": raw["episode_id"],
            "capture_created_at_utc": accessed_at,
            "capture_time_semantics": CAPTURE_TIME_SEMANTICS,
            "source_capture_time_semantics": "UNKNOWN_PREEXISTING_PUBLIC_DATASET_CAPTURE",
            "image_path": str(destination),
            "image_sha256": sha256(destination),
        })
    capture = _sealed_payload({
        "schema_version": CAPTURE_SCHEMA,
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "source_role": "PREEXISTING_PUBLIC_DATASET_GOAL_BEFORE_PROJECT_PIXEL_ACCESS",
        "precedence_mode": PRECEDENCE_MODE,
        "physical_capture_after_goal_claimed": False,
        "private_truth_access": False,
        "provider_model_calls": 0,
        "roster_body_sha256": roster["roster_body_sha256"],
        "cases": cases,
    }, "capture_manifest_body_sha256")
    _atomic_json(capture_path, capture)
    return capture


def download_private_truth(roster_path: Path, capture_path: Path, output_root: Path, truth_path: Path) -> dict[str, Any]:
    _require(not truth_path.exists(), "private truth is immutable and already exists")
    roster = _read_json(roster_path)
    capture = _read_json(capture_path)
    _validate_seal(roster, "roster_body_sha256")
    _validate_seal(capture, "capture_manifest_body_sha256")
    _require(capture.get("private_truth_access") is False, "capture manifest must be sealed before truth access")
    capture_by_case = {row["case_id"]: row for row in capture["cases"]}
    cases = []
    for raw in roster["cases"]:
        destination = (output_root / "private_annotations" / f"{raw['source_stem']}.png").resolve()
        _download(raw["annotation_drive_id"], destination)
        rgb = np.asarray(Image.open(destination).convert("RGB"), dtype=np.uint8)
        foreground = np.all(rgb == DOOR_RGB, axis=2).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        boxes = []
        for label in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label])
            if area > 0:
                boxes.append([float(x), float(y), float(x + width), float(y + height)])
        boxes.sort(key=lambda box: (box[0], box[1], box[2], box[3]))
        _require(raw["case_id"] in capture_by_case, "truth case is absent from capture manifest")
        cases.append({
            "case_id": raw["case_id"],
            "reference_mode": "SET_VALUED",
            "target_visibility": "VISIBLE" if boxes else "NOT_VISIBLE",
            "legal_target_bboxes_xyxy": boxes,
            "private_annotation_sha256": sha256(destination),
        })
    truth = {
        "schema_version": TRUTH_SCHEMA,
        "truth_created_at_utc": _utc_now(),
        "primary_iou_threshold": 0.30,
        "diagnostic_iou_thresholds": [0.10, 0.50],
        "recall_at_k": [1, 3, 5, 10],
        "truth_source": "DEEPDOORS2_EXACT_COLOR_SEMANTIC_MASK_CONNECTED_COMPONENTS",
        "cases": cases,
    }
    _atomic_json(truth_path, truth)
    return truth


def freeze_doordetect_roster(source_lock_path: Path, c0_path: Path, output_path: Path) -> dict[str, Any]:
    import requests

    _require(not output_path.exists(), "roster is immutable and already exists")
    source_lock = _read_json(source_lock_path)
    c0 = _read_json(c0_path)
    _require(source_lock.get("created_before_repository_tree_access") is True, "source lock tree precedence is missing")
    _require(source_lock.get("created_before_project_pixel_access") is True, "source lock pixel precedence is missing")
    _require(source_lock.get("created_before_private_label_access") is True, "source lock label precedence is missing")
    _require(source_lock.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "source lock C0 binding mismatch")
    _require(source_lock.get("source", {}).get("repository_revision") == DOORDETECT_REVISION, "DoorDetect revision drift")
    take = int(source_lock.get("roster_rule", {}).get("take", 0))
    _require(take == len(c0.get("episodes", [])) and take > 0, "roster size must equal frozen C0 episodes")
    response = requests.get(
        f"https://api.github.com/repos/{DOORDETECT_REPOSITORY}/git/trees/{DOORDETECT_REVISION}?recursive=1",
        timeout=60,
        headers={"Accept": "application/vnd.github+json"},
    )
    _require(response.status_code == 200, f"GitHub tree HTTP {response.status_code}")
    tree = response.json().get("tree", [])
    _require(isinstance(tree, list), "GitHub tree response is invalid")
    images: dict[str, Mapping[str, Any]] = {}
    labels: dict[str, Mapping[str, Any]] = {}
    for item in tree:
        if not isinstance(item, Mapping) or item.get("type") != "blob":
            continue
        path = str(item.get("path", "")).replace("\\", "/")
        pure = PurePosixPath(path)
        if path.startswith("images/") and pure.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images[pure.stem] = item
        elif path.startswith("labels/") and pure.suffix.lower() == ".txt":
            labels[pure.stem] = item
    eligible = sorted(set(images) & set(labels))
    _require(len(eligible) >= take, "DoorDetect has fewer paired files than the frozen roster requires")
    ordered = sorted(eligible, key=lambda stem: hashlib.sha256(str(images[stem]["path"]).encode("utf-8")).hexdigest())
    cases = []
    for index, stem in enumerate(ordered[:take], start=1):
        image_path = str(images[stem]["path"])
        cases.append({
            "case_id": f"s0v5-auto-case-{index:03d}",
            "episode_id": f"s0v4-auto-{index:03d}",
            "source_stem": stem,
            "image_path": image_path,
            "image_path_sha256": hashlib.sha256(image_path.encode("utf-8")).hexdigest(),
            "image_blob_sha": str(images[stem]["sha"]),
            "label_path": str(labels[stem]["path"]),
            "label_blob_sha": str(labels[stem]["sha"]),
        })
    roster = _sealed_payload({
        "schema_version": ROSTER_SCHEMA,
        "source_kind": "DOORDETECT_GITHUB_TREE",
        "repository": DOORDETECT_REPOSITORY,
        "repository_revision": DOORDETECT_REVISION,
        "created_at_utc": _utc_now(),
        "created_before_selected_pixel_download": True,
        "created_before_selected_label_download": True,
        "private_truth_access": False,
        "source_lock_sha256": sha256(source_lock_path),
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "eligible_pair_count": len(eligible),
        "selection_rule": source_lock["roster_rule"],
        "cases": cases,
    }, "roster_body_sha256")
    _atomic_json(output_path, roster)
    return roster


def _download_url(url: str, destination: Path, expected_kind: str) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_kind == "image":
            with Image.open(destination) as image:
                image.verify()
        else:
            destination.read_text(encoding="utf-8")
        return
    response = requests.get(url, timeout=60)
    _require(response.status_code == 200, f"download HTTP {response.status_code} for {destination.name}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(response.content)
    if expected_kind == "image":
        with Image.open(temporary) as image:
            image.verify()
    else:
        temporary.read_text(encoding="utf-8")
    os.replace(temporary, destination)


def download_doordetect_public(roster_path: Path, c0_path: Path, output_root: Path, capture_path: Path) -> dict[str, Any]:
    _require(not capture_path.exists(), "capture manifest is immutable and already exists")
    roster = _read_json(roster_path)
    c0 = _read_json(c0_path)
    _validate_seal(roster, "roster_body_sha256")
    _require(roster.get("source_kind") == "DOORDETECT_GITHUB_TREE", "DoorDetect roster source mismatch")
    _require(roster.get("private_truth_access") is False, "roster accessed private truth")
    _require(roster.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "roster C0 binding mismatch")
    goal_times = {row["episode_id"]: row["goal_provenance"]["goal_recorded_at_utc"] for row in c0["episodes"]}
    cases = []
    for raw in roster["cases"]:
        suffix = PurePosixPath(raw["image_path"]).suffix.lower()
        destination = (output_root / "public_images" / f"{raw['source_stem']}{suffix}").resolve()
        url = f"https://raw.githubusercontent.com/{DOORDETECT_REPOSITORY}/{DOORDETECT_REVISION}/{raw['image_path']}"
        _download_url(url, destination, "image")
        accessed_at = _utc_now()
        _require(goal_times[raw["episode_id"]] < accessed_at, "goal must precede first project pixel access")
        cases.append({
            "case_id": raw["case_id"],
            "episode_id": raw["episode_id"],
            "capture_created_at_utc": accessed_at,
            "capture_time_semantics": CAPTURE_TIME_SEMANTICS,
            "source_capture_time_semantics": "UNKNOWN_PREEXISTING_PUBLIC_DATASET_CAPTURE",
            "image_path": str(destination),
            "image_sha256": sha256(destination),
        })
    capture = _sealed_payload({
        "schema_version": CAPTURE_SCHEMA,
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "source_role": "PREEXISTING_PUBLIC_DATASET_GOAL_BEFORE_PROJECT_PIXEL_ACCESS",
        "precedence_mode": PRECEDENCE_MODE,
        "physical_capture_after_goal_claimed": False,
        "private_truth_access": False,
        "provider_model_calls": 0,
        "roster_body_sha256": roster["roster_body_sha256"],
        "cases": cases,
    }, "capture_manifest_body_sha256")
    _atomic_json(capture_path, capture)
    return capture


def download_doordetect_private_truth(roster_path: Path, capture_path: Path, output_root: Path, truth_path: Path) -> dict[str, Any]:
    _require(not truth_path.exists(), "private truth is immutable and already exists")
    roster = _read_json(roster_path)
    capture = _read_json(capture_path)
    _validate_seal(roster, "roster_body_sha256")
    _validate_seal(capture, "capture_manifest_body_sha256")
    _require(capture.get("private_truth_access") is False, "capture manifest must be sealed before label access")
    capture_by_case = {row["case_id"]: row for row in capture["cases"]}
    cases = []
    for raw in roster["cases"]:
        label_destination = (output_root / "private_labels" / f"{raw['source_stem']}.txt").resolve()
        label_url = f"https://raw.githubusercontent.com/{DOORDETECT_REPOSITORY}/{DOORDETECT_REVISION}/{raw['label_path']}"
        _download_url(label_url, label_destination, "text")
        captured = capture_by_case.get(raw["case_id"])
        _require(captured is not None, "truth case is absent from capture manifest")
        with Image.open(captured["image_path"]) as image:
            width, height = image.size
        boxes = []
        for line_number, line in enumerate(label_destination.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            _require(len(parts) == 5, f"invalid YOLO label at {raw['source_stem']}:{line_number}")
            class_id = int(parts[0])
            center_x, center_y, box_width, box_height = (float(value) for value in parts[1:])
            _require(all(np.isfinite(value) for value in (center_x, center_y, box_width, box_height)), "non-finite YOLO label")
            if class_id != 0:
                continue
            x1 = (center_x - box_width / 2.0) * width
            y1 = (center_y - box_height / 2.0) * height
            x2 = (center_x + box_width / 2.0) * width
            y2 = (center_y + box_height / 2.0) * height
            _require(x2 > x1 and y2 > y1, "non-positive DoorDetect box")
            boxes.append([x1, y1, x2, y2])
        cases.append({
            "case_id": raw["case_id"],
            "reference_mode": "SET_VALUED",
            "target_visibility": "VISIBLE" if boxes else "NOT_VISIBLE",
            "legal_target_bboxes_xyxy": boxes,
            "private_label_sha256": sha256(label_destination),
        })
    truth = {
        "schema_version": TRUTH_SCHEMA,
        "truth_created_at_utc": _utc_now(),
        "primary_iou_threshold": 0.30,
        "diagnostic_iou_thresholds": [0.10, 0.50],
        "recall_at_k": [1, 3, 5, 10],
        "truth_source": "DOORDETECT_FROZEN_CLASS_ZERO_YOLO_BBOX",
        "cases": cases,
    }
    _atomic_json(truth_path, truth)
    return truth


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-roster")
    freeze.add_argument("--source-lock", required=True, type=Path)
    freeze.add_argument("--c0", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    public = subparsers.add_parser("download-public")
    public.add_argument("--roster", required=True, type=Path)
    public.add_argument("--c0", required=True, type=Path)
    public.add_argument("--output-root", required=True, type=Path)
    public.add_argument("--capture", required=True, type=Path)
    private = subparsers.add_parser("download-private-truth")
    private.add_argument("--roster", required=True, type=Path)
    private.add_argument("--capture", required=True, type=Path)
    private.add_argument("--output-root", required=True, type=Path)
    private.add_argument("--truth", required=True, type=Path)
    door_freeze = subparsers.add_parser("freeze-doordetect-roster")
    door_freeze.add_argument("--source-lock", required=True, type=Path)
    door_freeze.add_argument("--c0", required=True, type=Path)
    door_freeze.add_argument("--output", required=True, type=Path)
    door_public = subparsers.add_parser("download-doordetect-public")
    door_public.add_argument("--roster", required=True, type=Path)
    door_public.add_argument("--c0", required=True, type=Path)
    door_public.add_argument("--output-root", required=True, type=Path)
    door_public.add_argument("--capture", required=True, type=Path)
    door_private = subparsers.add_parser("download-doordetect-private-truth")
    door_private.add_argument("--roster", required=True, type=Path)
    door_private.add_argument("--capture", required=True, type=Path)
    door_private.add_argument("--output-root", required=True, type=Path)
    door_private.add_argument("--truth", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze-roster":
        freeze_roster(args.source_lock, args.c0, args.output)
    elif args.command == "download-public":
        download_public(args.roster, args.c0, args.output_root, args.capture)
    elif args.command == "download-private-truth":
        download_private_truth(args.roster, args.capture, args.output_root, args.truth)
    elif args.command == "freeze-doordetect-roster":
        freeze_doordetect_roster(args.source_lock, args.c0, args.output)
    elif args.command == "download-doordetect-public":
        download_doordetect_public(args.roster, args.c0, args.output_root, args.capture)
    else:
        download_doordetect_private_truth(args.roster, args.capture, args.output_root, args.truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

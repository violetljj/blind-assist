#!/usr/bin/env python3
"""Audit the exact F0.1 SANPO acquisition before geometry is opened."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_sanpo_synthetic_metric_replay import inspect_depth


LOCK_SCHEMA = "blindassist_hftf_stage_c_f0_1_sanpo_source_lock"
LOCK_TERMINAL = "F0_1_SANPO_CROSS_SPLIT_SOURCE_LOCK_VALIDATED"
SCHEMA = "blindassist_hftf_stage_c_f0_1_sanpo_acquisition_audit"
READY = "F0_1_SANPO_ACQUISITION_AND_TRANSPORT_READY"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at {path}:{line_number}")
        values.append(value)
    return values


def _root_name(source: dict[str, Any]) -> str:
    return (
        f"hftf-f0-1-{source['role']}-{source['official_split']}-"
        f"{str(source['session_id'])[:8]}-25frames-20260801"
    )


def _safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("non-empty relative path required")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes replay root: {relative}") from error
    if not candidate.is_file():
        raise ValueError(f"missing file: {relative}")
    return candidate


def _receipt_matches_file(receipt: Any, path: Path) -> bool:
    return (
        isinstance(receipt, dict)
        and int(receipt.get("size", -1)) == path.stat().st_size
        and receipt.get("md5_base64") == _md5_base64(path)
    )


def _validate_spec(
    spec: dict[str, Any], source: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    identity = spec.get("source", {})
    sampling = spec.get("sampling", {})
    inventory = spec.get("source_inventory", {})
    expected = {
        "official_split": source["official_split"],
        "session_id": source["session_id"],
    }
    if not isinstance(identity, dict) or any(
        identity.get(key) != value for key, value in expected.items()
    ):
        errors.append("dataset_spec_source_identity_mismatch")
    if (
        not isinstance(sampling, dict)
        or float(sampling.get("source_fps", -1)) != source["source_fps"]
        or float(sampling.get("target_fps", -1)) != source["target_fps"]
        or sampling.get("selected_source_frames")
        != source["selected_source_frames"]
    ):
        errors.append("dataset_spec_sampling_mismatch")
    if (
        not isinstance(inventory, dict)
        or inventory.get("description") != source["description_object"]
        or inventory.get("camera_poses") != source["camera_poses_object"]
        or not isinstance(inventory.get("official_split_receipt"), dict)
        or len(inventory.get("rgb", [])) != 25
        or len(inventory.get("masks", [])) != 25
        or len(inventory.get("depth", [])) != 25
    ):
        errors.append("dataset_spec_inventory_mismatch")
    return errors


def _validate_source(
    source: dict[str, Any], datasets_root: Path
) -> dict[str, Any]:
    root = (datasets_root / _root_name(source)).resolve()
    errors: list[str] = []
    if not root.is_dir():
        return {
            "role": source["role"],
            "official_split": source["official_split"],
            "session_id": source["session_id"],
            "root": str(root),
            "ok": False,
            "errors": ["replay_root_missing"],
        }
    manifest_path = root / "manifest.replay.jsonl"
    spec_path = root / "dataset_spec.json"
    qa_path = root / "qa/replay_validation.json"
    pose_path = root / "source_metadata/camera_poses.csv"
    description_path = root / "source_metadata/source_session_description.json"
    required = [
        manifest_path,
        spec_path,
        qa_path,
        pose_path,
        description_path,
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        return {
            "role": source["role"],
            "official_split": source["official_split"],
            "session_id": source["session_id"],
            "root": str(root),
            "ok": False,
            "errors": [f"required_files_missing:{','.join(missing)}"],
        }
    rows = _load_jsonl(manifest_path)
    spec = _load_json(spec_path)
    qa = _load_json(qa_path)
    errors.extend(_validate_spec(spec, source))
    split = str(source["official_split"])
    role = str(source["role"])
    if (
        qa.get("ok") is not True
        or qa.get("frame_count") != 25
        or qa.get("required_modalities_hash_bound") is not True
        or qa.get("all_rgb_mask_dimensions_match") is not True
        or qa.get("official_split") != split
        or qa.get("all_frames_official_split_match") is not True
        or qa.get("pretraining_candidate") is not (role != "heldout")
        or qa.get("synthetic_heldout_evaluation_candidate")
        is not (role == "heldout")
        or qa.get("production_authorized") is not False
    ):
        errors.append("qa_contract_mismatch")
    expected_frames = list(source["selected_source_frames"])
    if (
        len(rows) != 25
        or len({row.get("id") for row in rows}) != 25
        or [row.get("frame_index") for row in rows] != list(range(25))
        or [row.get("source_frame_index") for row in rows] != expected_frames
    ):
        errors.append("manifest_timeline_or_uniqueness_mismatch")
    pose_sha = _sha256(pose_path)
    if not _receipt_matches_file(source["description_object"], description_path):
        errors.append("description_source_receipt_mismatch")
    if not _receipt_matches_file(source["camera_poses_object"], pose_path):
        errors.append("camera_pose_source_receipt_mismatch")
    depth_fractions: list[float] = []
    local_files: set[Path] = set(required)
    for index, row in enumerate(rows):
        prefix = f"frame_{index:02d}"
        authorization = row.get("authorization", {})
        row_source = row.get("source", {})
        if (
            row.get("session_id") != source["session_id"]
            or not isinstance(row_source, dict)
            or row_source.get("session_id") != source["session_id"]
            or row_source.get("official_split") != split
            or row.get("event_truth") is not None
            or not isinstance(authorization, dict)
            or authorization.get("pretraining_candidate")
            is not (role != "heldout")
            or authorization.get("synthetic_heldout_evaluation_candidate")
            is not (role == "heldout")
            or authorization.get("production_model_replacement") is not False
        ):
            errors.append(f"{prefix}:identity_or_authorization_mismatch")
        expected_timestamp = round(
            1000 * expected_frames[index] / float(source["source_fps"])
        )
        if row.get("source_timestamp_ms") != expected_timestamp:
            errors.append(f"{prefix}:physical_time_mismatch")
        try:
            width, height = int(row["width"]), int(row["height"])
            if (width, height) != (
                int(source["camera"]["image_width"]),
                int(source["camera"]["image_height"]),
            ):
                raise ValueError("manifest dimensions mismatch locked camera")
            image_path = _safe_file(root, row.get("image_path"))
            mask_path = _safe_file(root, row.get("source_mask_path"))
            depth_path = _safe_file(root, row.get("source_depth_path"))
            local_files.update((image_path, mask_path, depth_path))
            if (
                _sha256(image_path) != row.get("image_sha256")
                or _sha256(mask_path) != row.get("source_mask_sha256")
                or _sha256(depth_path) != row.get("source_depth_sha256")
            ):
                raise ValueError("local modality sha256 mismatch")
            with Image.open(image_path) as image:
                if image.size != (width, height):
                    raise ValueError("RGB dimensions mismatch")
                image.verify()
            with Image.open(mask_path) as mask:
                if mask.size != (width, height):
                    raise ValueError("mask dimensions mismatch")
                mask.verify()
            depth = inspect_depth(depth_path, width, height)
            depth_fractions.append(float(depth["finite_positive_fraction"]))
            pose_binding = row.get("modalities", {}).get("camera_poses", {})
            if (
                pose_binding.get("path") != "source_metadata/camera_poses.csv"
                or pose_binding.get("sha256") != pose_sha
            ):
                raise ValueError("manifest pose binding mismatch")
        except (KeyError, TypeError, ValueError, OSError) as error:
            errors.append(f"{prefix}:{error}")
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    allowed_metadata = {
        (root / "source_licenses.md").resolve(),
        (root / "source_metadata/source_annotation_types.json").resolve(),
        (root / "source_metadata/source_labelmap.json").resolve(),
    }
    local_files.update(allowed_metadata)
    extras = sorted(str(path.relative_to(root)) for path in actual_files - local_files)
    if extras:
        errors.append(f"unexpected_files:{','.join(extras)}")
    total_bytes = sum(path.stat().st_size for path in actual_files)
    return {
        "role": role,
        "official_split": split,
        "session_id": source["session_id"],
        "root": str(root),
        "manifest_sha256": _sha256(manifest_path),
        "dataset_spec_sha256": _sha256(spec_path),
        "qa_sha256": _sha256(qa_path),
        "camera_poses_sha256": pose_sha,
        "description_sha256": _sha256(description_path),
        "frame_count": len(rows),
        "rgb_count": sum(1 for row in rows if row.get("image_path")),
        "mask_count": sum(1 for row in rows if row.get("source_mask_path")),
        "depth_count": len(depth_fractions),
        "minimum_finite_positive_depth_fraction": (
            min(depth_fractions) if depth_fractions else None
        ),
        "file_count": len(actual_files),
        "total_bytes": total_bytes,
        "ok": not errors,
        "errors": errors,
    }


def audit(source_lock_path: Path, datasets_root: Path) -> dict[str, Any]:
    source_lock = _load_json(source_lock_path)
    if (
        source_lock.get("schema") != LOCK_SCHEMA
        or source_lock.get("terminal") != LOCK_TERMINAL
        or source_lock.get("exact_media_acquisition_authorized") is not True
        or source_lock.get("teacher_label_or_corpus_authorized") is not False
        or source_lock.get("student_training_authorized") is not False
    ):
        raise ValueError("F0.1 source lock contract mismatch")
    sources = source_lock.get("sources", [])
    if (
        len(sources) != 12
        or [item.get("role") for item in sources]
        != ["train"] * 6 + ["dev"] * 3 + ["heldout"] * 3
    ):
        raise ValueError("F0.1 source lock role set mismatch")
    results = [_validate_source(source, datasets_root) for source in sources]
    ready = all(result["ok"] for result in results)
    return {
        "schema": SCHEMA,
        "terminal": (
            READY if ready else "F0_1_SANPO_ACQUISITION_NOT_EVALUABLE"
        ),
        "source_lock_path": str(source_lock_path.resolve()),
        "source_lock_sha256": _sha256(source_lock_path),
        "datasets_root": str(datasets_root.resolve()),
        "source_count": len(results),
        "role_counts": {
            role: sum(result["role"] == role for result in results)
            for role in ("train", "dev", "heldout")
        },
        "all_sources_ok": ready,
        "frame_count": sum(int(result.get("frame_count", 0)) for result in results),
        "modality_counts": {
            key: sum(int(result.get(f"{key}_count", 0)) for result in results)
            for key in ("rgb", "mask", "depth")
        },
        "total_bytes": sum(int(result.get("total_bytes", 0)) for result in results),
        "sources": results,
        "authorization": {
            "source_pose_geometry_authority_audit_authorized": ready,
            "teacher_geometry_outcome_authorized": False,
            "teacher_label_or_corpus_authorized": False,
            "student_training_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
        },
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
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = audit(
            args.source_lock.resolve(), args.datasets_root.resolve()
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "source_count": report["source_count"],
                    "frame_count": report["frame_count"],
                    "modality_counts": report["modality_counts"],
                    "output": str(output),
                }
            )
        )
        return 0 if report["all_sources_ok"] else 1
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

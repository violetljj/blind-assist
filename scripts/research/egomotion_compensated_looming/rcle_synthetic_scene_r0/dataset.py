from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
from typing import Any

import cv2
import numpy as np

from .io_utils import relative_posix, sha256_file, write_json, write_jsonl
from .renderer import camera_intrinsics, render_frame


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load_spec(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != "RCLE_SYNTHETIC_SCENE_MECHANISM_VALIDATION_R0":
        raise ValueError("PROTOCOL_ID")
    return value


def _ensure_new_root(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"DATASET_ROOT_NOT_EMPTY:{root}")
    root.mkdir(parents=True, exist_ok=True)


def _validate_json_schema(
    value: Any, schema: dict[str, Any], path: str = "$"
) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"ACTIVATION_SCHEMA_TYPE:{path}")
        required = set(schema.get("required", []))
        if not required.issubset(value):
            raise ValueError(f"ACTIVATION_SCHEMA_REQUIRED:{path}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and (
            set(value) - set(properties)
        ):
            raise ValueError(f"ACTIVATION_SCHEMA_ADDITIONAL:{path}")
        for key, child in properties.items():
            if key in value:
                _validate_json_schema(value[key], child, f"{path}.{key}")
    elif expected_type == "string" and not isinstance(value, str):
        raise ValueError(f"ACTIVATION_SCHEMA_TYPE:{path}")
    elif expected_type == "boolean" and not isinstance(value, bool):
        raise ValueError(f"ACTIVATION_SCHEMA_TYPE:{path}")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"ACTIVATION_SCHEMA_CONST:{path}")
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise ValueError(f"ACTIVATION_SCHEMA_MIN_LENGTH:{path}")
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(pattern, value) is None:
            raise ValueError(f"ACTIVATION_SCHEMA_PATTERN:{path}")


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sealed_authorized(
    spec_path: Path,
    spec: dict[str, Any],
    output_root: Path,
    activation: Path | None,
    *,
    consume_claim: bool = True,
) -> bool:
    if activation is None or not activation.is_file():
        return False
    value = json.loads(activation.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[4]
    schema_path = (
        repo_root / spec["sealed_activation"]["json_schema_path"]
    ).resolve()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    _validate_json_schema(value, schema)
    required = set(spec["sealed_activation"]["required_fields"])
    if set(value) != required:
        return False
    if value["schema"] != spec["sealed_activation"]["receipt_schema"]:
        return False
    if (
        value["protocol_id"] != spec["protocol_id"]
        or value["split"] != "sealed"
        or value["execution_authorized"] is not True
        or value["current_governance_reconciled"] is not True
        or value["output_root_preflight_empty"] is not True
    ):
        return False
    if Path(value["canonical_output_root"]).resolve() != output_root.resolve():
        return False
    if output_root.exists() and any(output_root.iterdir()):
        return False
    module_root = Path(__file__).resolve().parent
    identities = {
        "dataset_spec_sha256": sha256_file(spec_path),
        "renderer_sha256": sha256_file(module_root / "renderer.py"),
        "dataset_builder_sha256": sha256_file(module_root / "dataset.py"),
        "truth_builder_sha256": sha256_file(module_root / "truth.py"),
        "algorithm_adapter_sha256": sha256_file(
            module_root / "evaluator.py"
        ),
        "validator_sha256": sha256_file(module_root / "validator.py"),
    }
    if any(value[name] != digest for name, digest in identities.items()):
        return False
    dependency = value["dependency_identity"]
    if dependency != {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }:
        return False
    review_path = Path(value["independent_review_receipt_path"]).resolve()
    if (
        not review_path.is_file()
        or sha256_file(review_path)
        != value["independent_review_receipt_sha256"]
    ):
        return False
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review_required = {
        "schema",
        "protocol_id",
        "claim_id",
        "authorizer_id",
        "decision",
        "execution_authorized",
        "sealed_activation_schema_sha256",
        "governance_reconciliation_path",
        "governance_reconciliation_sha256",
        "sealed_executor_sha256",
        "cli_sha256",
    }
    if set(review) != review_required:
        return False
    if (
        review["schema"]
        != "rcle.synthetic_scene.sealed_activation_review.v1"
        or review["protocol_id"] != spec["protocol_id"]
        or review["claim_id"] != value["claim_id"]
        or review["authorizer_id"] != value["authorizer_id"]
        or review["decision"] != "PASS"
        or review["execution_authorized"] is not True
        or review["sealed_activation_schema_sha256"]
        != sha256_file(schema_path)
        or review["sealed_executor_sha256"]
        != sha256_file(module_root / "sealed_executor.py")
        or review["cli_sha256"] != sha256_file(module_root / "cli.py")
    ):
        return False
    governance_path = Path(
        review["governance_reconciliation_path"]
    ).resolve()
    if (
        not governance_path.is_file()
        or sha256_file(governance_path)
        != review["governance_reconciliation_sha256"]
    ):
        return False
    governance = json.loads(governance_path.read_text(encoding="utf-8"))
    if (
        governance.get("schema")
        != "rcle.synthetic_scene.sealed_governance_reconciliation.v1"
        or governance.get("protocol_id") != spec["protocol_id"]
        or governance.get("status") != "PASS"
        or governance.get("sealed_execution_authorized") is not True
    ):
        return False
    claim_path = Path(value["claim_path"]).resolve()
    if claim_path.is_relative_to(output_root.resolve()):
        return False
    claim_root = (
        repo_root / spec["sealed_activation"]["claim_ledger_root"]
    ).resolve()
    if not claim_path.is_relative_to(claim_root):
        return False
    if claim_path.exists():
        return False
    if not consume_claim:
        return True
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "schema": "rcle.synthetic_scene.sealed_claim.v1",
            "protocol_id": spec["protocol_id"],
            "claim_id": value["claim_id"],
            "activation_sha256": sha256_file(activation),
            "canonical_output_root": str(output_root.resolve()),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(
        claim_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def validate_sealed_activation(
    spec_path: Path, output_root: Path, activation: Path
) -> dict[str, Any]:
    """Validate every sealed precondition without consuming the claim."""
    spec_path = spec_path.resolve()
    output_root = output_root.resolve()
    spec = _load_spec(spec_path)
    authorized = _sealed_authorized(
        spec_path,
        spec,
        output_root,
        activation.resolve(),
        consume_claim=False,
    )
    return {
        "schema": "rcle.synthetic_scene.sealed_activation_preflight.v1",
        "protocol_id": spec["protocol_id"],
        "status": "PASS" if authorized else "FAIL",
        "claim_consumed": False,
        "canonical_output_root": str(output_root),
        "activation_sha256": (
            sha256_file(activation) if activation.is_file() else None
        ),
    }


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB_FORMAT")
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"RGB_WRITE_FAILED:{path}")


def _write_depth(path_npy: Path, path_png: Path, depth_m: np.ndarray) -> None:
    path_npy.parent.mkdir(parents=True, exist_ok=True)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    depth32 = np.asarray(depth_m, dtype=np.float32)
    np.save(path_npy, depth32, allow_pickle=False)
    millimeters = np.rint(depth32 * 1000.0)
    if np.any(~np.isfinite(millimeters)) or np.any(millimeters <= 0):
        raise ValueError("DEPTH_NON_POSITIVE_OR_NONFINITE")
    if float(np.max(millimeters)) > float(np.iinfo(np.uint16).max):
        raise ValueError("DEPTH_UINT16_OVERFLOW")
    ok = cv2.imwrite(str(path_png), millimeters.astype(np.uint16))
    if not ok:
        raise OSError(f"DEPTH_WRITE_FAILED:{path_png}")


def _write_surface(path: Path, surface_id: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(surface_id, dtype=np.int16), allow_pickle=False)


def _write_valid_mask(path: Path, surface_id: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.where(surface_id > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), mask):
        raise OSError(f"VALID_MASK_WRITE_FAILED:{path}")


def _stage_algorithm_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def build_dataset(
    spec_path: Path,
    output_root: Path,
    split: str = "development",
    activation: Path | None = None,
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    spec_path = spec_path.resolve()
    output_root = output_root.resolve()
    spec = _load_spec(spec_path)
    if split not in spec["splits"]:
        raise ValueError(f"UNKNOWN_SPLIT:{split}")
    if split == "sealed" and not _sealed_authorized(
        spec_path, spec, output_root, activation
    ):
        raise PermissionError("SEALED_EXECUTION_NOT_AUTHORIZED")
    _ensure_new_root(output_root)
    preflight = spec["resource_preflight"]
    if platform.python_version() != preflight["supported_python"]:
        raise RuntimeError("UNSUPPORTED_PYTHON")
    if np.__version__ != preflight["supported_numpy"]:
        raise RuntimeError("UNSUPPORTED_NUMPY")
    if not cv2.__version__.startswith(
        preflight["supported_opencv_prefix"]
    ):
        raise RuntimeError("UNSUPPORTED_OPENCV")

    canonical_spec_path = output_root / "dataset_spec.json"
    shutil.copyfile(spec_path, canonical_spec_path)
    spec_sha = sha256_file(canonical_spec_path)
    fps = int(spec["rendering"]["fps"])
    duration = int(spec["rendering"]["duration_seconds"])
    frame_count = int(spec["rendering"]["frame_count"])
    if frame_count != duration * fps + 1:
        raise ValueError("FRAME_COUNT_ENDPOINT_DRIFT")
    intrinsics = camera_intrinsics(spec)
    generated_at = _iso_now()
    started_perf = time.perf_counter()
    expected_sequences = len(
        spec["splits"][split]["scene_families"]
    ) * len(spec["motions"])
    write_json(
        output_root / "run_state.json",
        {
            "status": "RUNNING",
            "split": split,
            "planned_sequences": expected_sequences,
            "completed_sequences": 0,
            "planned_frames": expected_sequences * frame_count,
            "completed_frames": 0,
            "started_at": generated_at,
        },
    )
    scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    algorithm_input_rows: list[dict[str, Any]] = []

    sequence_ordinal = 0
    for scene in spec["splits"][split]["scene_families"]:
        for motion in spec["motions"]:
            sequence_id = f"{scene['scene_id']}__{motion['motion_id']}"
            sequence_root = output_root / "sequences" / sequence_id
            frame_hashes: list[str] = []
            previous_rotation: np.ndarray | None = None
            for frame_index in range(frame_count):
                rendered = render_frame(
                    spec, scene, motion, frame_index
                )
                stem = f"{frame_index:06d}"
                rgb_path = sequence_root / "rgb" / f"{stem}.png"
                depth_npy_path = sequence_root / "depth_npy" / f"{stem}.npy"
                depth_png_path = sequence_root / "depth_mm" / f"{stem}.png"
                surface_path = sequence_root / "surface_id" / f"{stem}.npy"
                valid_mask_path = sequence_root / "valid_mask" / f"{stem}.png"
                staged_rgb_path = (
                    output_root
                    / "algorithm_staging"
                    / sequence_id
                    / "rgb"
                    / f"{stem}.png"
                )
                staged_valid_mask_path = (
                    output_root
                    / "algorithm_staging"
                    / sequence_id
                    / "valid_mask"
                    / f"{stem}.png"
                )
                _write_rgb(rgb_path, rendered.rgb)
                _write_depth(depth_npy_path, depth_png_path, rendered.depth_m)
                _write_surface(surface_path, rendered.surface_id)
                _write_valid_mask(valid_mask_path, rendered.surface_id)
                _stage_algorithm_file(rgb_path, staged_rgb_path)
                _stage_algorithm_file(
                    valid_mask_path, staged_valid_mask_path
                )
                hashes = {
                    "rgb_sha256": sha256_file(rgb_path),
                    "depth_npy_sha256": sha256_file(depth_npy_path),
                    "depth_png_sha256": sha256_file(depth_png_path),
                    "surface_id_sha256": sha256_file(surface_path),
                    "valid_mask_sha256": sha256_file(valid_mask_path),
                }
                frame_hashes.extend(hashes.values())
                frame_rows.append(
                    {
                        "schema": "rcle.synthetic_scene.frame.v1",
                        "protocol_id": spec["protocol_id"],
                        "split": split,
                        "scene_id": scene["scene_id"],
                        "scene_seed": int(scene["seed"]),
                        "motion_id": motion["motion_id"],
                        "role": motion["role"],
                        "sequence_id": sequence_id,
                        "frame_index": frame_index,
                        "timestamp": {
                            "numerator": frame_index,
                            "denominator": fps,
                            "seconds": frame_index / fps,
                        },
                        "rgb_path": relative_posix(rgb_path, output_root),
                        "depth_npy_path": relative_posix(
                            depth_npy_path, output_root
                        ),
                        "depth_png_path": relative_posix(
                            depth_png_path, output_root
                        ),
                        "surface_id_path": relative_posix(
                            surface_path, output_root
                        ),
                        "valid_mask_path": relative_posix(
                            valid_mask_path, output_root
                        ),
                        **hashes,
                        "intrinsics": intrinsics.tolist(),
                        "t_world_camera": rendered.t_world_camera.tolist(),
                        "t_camera_world": rendered.t_camera_world.tolist(),
                        "review_status": "candidate_pending_automated_geometry_validation",
                    }
                )
                current_rotation = rendered.t_world_camera[:3, :3]
                rotation_current_from_previous = (
                    np.eye(3, dtype=np.float64)
                    if previous_rotation is None
                    else current_rotation.T @ previous_rotation
                )
                algorithm_input_rows.append(
                    {
                        "schema": "rcle.synthetic_scene.algorithm_input.v1",
                        "protocol_id": spec["protocol_id"],
                        "split": split,
                        "sequence_id": sequence_id,
                        "frame_index": frame_index,
                        "timestamp": {
                            "numerator": frame_index,
                            "denominator": fps,
                            "seconds": frame_index / fps,
                        },
                        "rgb_path": relative_posix(
                            staged_rgb_path, output_root
                        ),
                        "rgb_sha256": sha256_file(staged_rgb_path),
                        "valid_mask_path": relative_posix(
                            staged_valid_mask_path, output_root
                        ),
                        "valid_mask_sha256": sha256_file(
                            staged_valid_mask_path
                        ),
                        "intrinsics": intrinsics.tolist(),
                        "rotation_current_from_previous": (
                            rotation_current_from_previous.tolist()
                        ),
                        "algorithm_seed": (
                            int(spec["split_seed"])
                            + sequence_ordinal * 1009
                            + frame_index
                        )
                        % (2**31 - 1),
                    }
                )
                previous_rotation = current_rotation
            sequence_digest = __import__("hashlib").sha256(
                "".join(frame_hashes).encode("ascii")
            ).hexdigest()
            scene_rows.append(
                {
                    "schema": "rcle.synthetic_scene.sequence.v1",
                    "protocol_id": spec["protocol_id"],
                    "split": split,
                    "scene_id": scene["scene_id"],
                    "scene_seed": int(scene["seed"]),
                    "panel_variant": int(scene["panel_variant"]),
                    "motion": motion,
                    "sequence_id": sequence_id,
                    "frame_count": frame_count,
                    "pair_count": frame_count - 1,
                    "sequence_content_sha256": sequence_digest,
                    "generator": {
                        "backend": spec["rendering"]["backend"],
                        "authoritative_geometry_dtype": "float64",
                        "opencv_threads": 1,
                    },
                }
            )
            sequence_ordinal += 1
            elapsed = time.perf_counter() - started_perf
            completed_frames = sequence_ordinal * frame_count
            write_json(
                output_root / "run_state.json",
                {
                    "status": "RUNNING",
                    "split": split,
                    "planned_sequences": expected_sequences,
                    "completed_sequences": sequence_ordinal,
                    "planned_frames": expected_sequences * frame_count,
                    "completed_frames": completed_frames,
                    "elapsed_seconds": elapsed,
                    "frames_per_second": (
                        completed_frames / elapsed if elapsed > 0 else None
                    ),
                    "eta_seconds": (
                        (expected_sequences * frame_count - completed_frames)
                        / (completed_frames / elapsed)
                        if elapsed > 0 and completed_frames > 0
                        else None
                    ),
                },
            )
            runtime_ceiling = float(
                preflight[f"{split}_runtime_ceiling_seconds"]
            )
            if elapsed > runtime_ceiling:
                raise TimeoutError("GENERATION_RUNTIME_CEILING_EXCEEDED")

    scene_manifest_sha = write_jsonl(
        output_root / "scene_manifest.jsonl", scene_rows
    )
    frame_manifest_sha = write_jsonl(
        output_root / "frame_manifest.jsonl", frame_rows
    )
    algorithm_input_manifest_sha = write_jsonl(
        output_root / "algorithm_input_manifest.jsonl",
        algorithm_input_rows,
    )
    receipt = {
        "schema": "rcle.synthetic_scene.dataset_receipt.v1",
        "protocol_id": spec["protocol_id"],
        "split": split,
        "status": "CANDIDATE_DATASET_GENERATED_VALIDATION_REQUIRED",
        "generated_at": generated_at,
        "dataset_spec_sha256": spec_sha,
        "scene_manifest_sha256": scene_manifest_sha,
        "frame_manifest_sha256": frame_manifest_sha,
        "algorithm_input_manifest_sha256": algorithm_input_manifest_sha,
        "sequence_count": len(scene_rows),
        "frame_count": len(frame_rows),
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "authority": spec["authority"],
    }
    receipt_sha = write_json(output_root / "receipt.json", receipt)
    final_elapsed = time.perf_counter() - started_perf
    dataset_bytes = sum(
        path.stat().st_size
        for path in output_root.rglob("*")
        if path.is_file()
    )
    disk_ceiling = int(preflight[f"{split}_disk_ceiling_bytes"])
    if dataset_bytes > disk_ceiling:
        raise RuntimeError("GENERATION_DISK_CEILING_EXCEEDED")
    write_json(
        output_root / "run_state.json",
        {
            "status": "COMPLETE",
            "split": split,
            "planned_sequences": expected_sequences,
            "completed_sequences": len(scene_rows),
            "planned_frames": expected_sequences * frame_count,
            "completed_frames": len(frame_rows),
            "elapsed_seconds": final_elapsed,
            "dataset_bytes": dataset_bytes,
            "receipt_sha256": receipt_sha,
        },
    )
    return {
        "dataset_root": str(output_root),
        "split": split,
        "sequence_count": len(scene_rows),
        "frame_count": len(frame_rows),
        "receipt_sha256": receipt_sha,
    }


def build_dataset_governed(
    spec_path: Path,
    output_root: Path,
    split: str = "development",
    activation: Path | None = None,
) -> dict[str, Any]:
    """Build a dataset and preserve a one-shot sealed failure terminal."""
    claim_preexisted = False
    if split == "sealed" and activation is not None and activation.is_file():
        try:
            activation_value = json.loads(
                activation.read_text(encoding="utf-8")
            )
            claim_preexisted = Path(
                activation_value["claim_path"]
            ).resolve().exists()
        except (KeyError, json.JSONDecodeError, OSError, TypeError):
            claim_preexisted = False
    try:
        return build_dataset(spec_path, output_root, split, activation)
    except Exception as exc:
        if (
            split == "sealed"
            and not claim_preexisted
            and activation is not None
            and activation.is_file()
        ):
            try:
                activation_value = json.loads(
                    activation.read_text(encoding="utf-8")
                )
                claim_path = Path(activation_value["claim_path"]).resolve()
                if claim_path.is_file():
                    terminal_path = claim_path.with_name(
                        claim_path.stem + ".terminal.json"
                    )
                    _write_exclusive_json(
                        terminal_path,
                        {
                            "schema": "rcle.synthetic_scene.sealed_terminal.v1",
                            "protocol_id": activation_value.get("protocol_id"),
                            "claim_id": activation_value.get("claim_id"),
                            "status": "INVALID_SEALED_EXECUTION_FAIL_CLOSED",
                            "claim_consumed": True,
                            "canonical_output_root": str(
                                output_root.resolve()
                            ),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:500],
                            "created_at": _iso_now(),
                            "retry_or_replacement_forbidden": True,
                        },
                    )
            except FileExistsError:
                pass
            except Exception as terminal_error:
                raise RuntimeError(
                    "SEALED_FAILURE_TERMINAL_WRITE_FAILED"
                ) from terminal_error
        raise

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_low_reference_false_trigger_r1.temporal_confirmation import (
    REQUIRED_CONSECUTIVE_PAIRS,
    THRESHOLD,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.dataset import (
    build_dataset,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.io_utils import (
    read_json,
    read_jsonl,
    relative_posix,
    sha256_file,
    write_json,
    write_jsonl,
)


MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SPEC = MODULE_ROOT / "dataset_spec.json"


def _nearest_existing(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"NO_EXISTING_ANCESTOR:{path}")
        candidate = candidate.parent
    return candidate


def _check_budget(
    started: float,
    written_bytes: int,
    resource_budget: dict[str, Any],
) -> None:
    if time.perf_counter() - started > float(
        resource_budget["runtime_ceiling_seconds"]
    ):
        raise TimeoutError("STRESS_BUILD_RUNTIME_CEILING_EXCEEDED")
    if written_bytes > int(resource_budget["disk_ceiling_bytes"]):
        raise RuntimeError("STRESS_DISK_CEILING_EXCEEDED")


def _validate_frozen_parent_chain(spec: dict[str, Any]) -> dict[str, Any]:
    chain = spec["frozen_parent_chain"]
    loaded: dict[str, dict[str, Any]] = {}
    for name in ("activation", "claim", "terminal"):
        entry = chain[name]
        path = (REPO_ROOT / entry["path"]).resolve()
        if not path.is_relative_to(REPO_ROOT.resolve()):
            raise ValueError(f"PARENT_CHAIN_PATH_ESCAPE:{name}")
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"PARENT_CHAIN_HASH_MISMATCH:{name}")
        loaded[name] = read_json(path)
    activation = loaded["activation"]
    claim = loaded["claim"]
    terminal = loaded["terminal"]
    if terminal.get("status") != chain["required_terminal_status"]:
        raise ValueError("PARENT_TERMINAL_STATUS")
    if (
        terminal.get("scientific_outcome")
        != chain["required_scientific_outcome"]
    ):
        raise ValueError("PARENT_SCIENTIFIC_OUTCOME")
    if (
        terminal.get("claim_consumed")
        is not chain["claim_consumed_required"]
    ):
        raise ValueError("PARENT_CLAIM_CONSUMED")
    if (
        terminal.get("retry_or_replacement_forbidden")
        is not chain["retry_or_replacement_forbidden_required"]
    ):
        raise ValueError("PARENT_RETRY_BOUNDARY")
    if not (
        activation.get("claim_id")
        == claim.get("claim_id")
        == terminal.get("claim_id")
    ):
        raise ValueError("PARENT_CLAIM_IDENTITY")
    activation_sha = chain["activation"]["sha256"]
    if (
        claim.get("activation_sha256") != activation_sha
        or terminal.get("activation_sha256") != activation_sha
    ):
        raise ValueError("PARENT_ACTIVATION_BINDING")

    implementation_hashes: dict[str, str] = {}
    for entry in spec["implementation_bindings"]:
        path = (REPO_ROOT / entry["path"]).resolve()
        if not path.is_relative_to(REPO_ROOT.resolve()):
            raise ValueError("IMPLEMENTATION_PATH_ESCAPE")
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"IMPLEMENTATION_HASH_MISMATCH:{entry['path']}")
        implementation_hashes[entry["path"]] = entry["sha256"]
    activation_binding_map = {
        "renderer_sha256": (
            "scripts/research/egomotion_compensated_looming/"
            "rcle_synthetic_scene_r0/renderer.py"
        ),
        "dataset_builder_sha256": (
            "scripts/research/egomotion_compensated_looming/"
            "rcle_synthetic_scene_r0/dataset.py"
        ),
        "truth_builder_sha256": (
            "scripts/research/egomotion_compensated_looming/"
            "rcle_synthetic_scene_r0/truth.py"
        ),
        "algorithm_adapter_sha256": (
            "scripts/research/egomotion_compensated_looming/"
            "rcle_synthetic_scene_r0/evaluator.py"
        ),
        "validator_sha256": (
            "scripts/research/egomotion_compensated_looming/"
            "rcle_synthetic_scene_r0/validator.py"
        ),
    }
    for activation_field, implementation_path in activation_binding_map.items():
        if (
            activation.get(activation_field)
            != implementation_hashes.get(implementation_path)
        ):
            raise ValueError(
                f"PARENT_ACTIVATION_IMPLEMENTATION:{activation_field}"
            )
    if float(spec["parent_r0"]["algorithm_threshold_per_s"]) != THRESHOLD:
        raise ValueError("FROZEN_THRESHOLD_DRIFT")
    if (
        int(spec["parent_r0"]["confirmation_consecutive_pairs"])
        != REQUIRED_CONSECUTIVE_PAIRS
    ):
        raise ValueError("FROZEN_CONFIRMATION_DRIFT")
    return {
        "claim_id": terminal["claim_id"],
        "activation_sha256": activation_sha,
        "claim_sha256": chain["claim"]["sha256"],
        "terminal_sha256": chain["terminal"]["sha256"],
        "scientific_outcome": terminal["scientific_outcome"],
        "implementation_bindings": implementation_hashes,
    }


def _ensure_new_root(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"STRESS_ROOT_NOT_EMPTY:{root}")
    root.mkdir(parents=True, exist_ok=True)


def _seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _yaw_rotation(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _transform_rgb(
    rgb_bgr: np.ndarray,
    condition: dict[str, Any],
    frame_index: int,
) -> np.ndarray:
    condition_id = condition["condition_id"]
    if condition_id in (
        "matched_clean_control",
        "depth_noise_gaussian_2pct",
        "pose_yaw_error",
    ):
        return rgb_bgr.copy()
    if condition_id == "lighting_low_exposure":
        return np.rint(
            np.clip(
                rgb_bgr.astype(np.float64) * float(condition["gain"]),
                0.0,
                255.0,
            )
        ).astype(np.uint8)
    if condition_id == "low_texture_contrast":
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        mean = float(np.mean(gray))
        low = np.rint(
            np.clip(
                mean + float(condition["contrast"]) * (gray - mean),
                0.0,
                255.0,
            )
        ).astype(np.uint8)
        return cv2.cvtColor(low, cv2.COLOR_GRAY2BGR)
    if condition_id == "motion_blur_horizontal_9":
        size = int(condition["kernel_pixels"])
        kernel = np.ones((1, size), dtype=np.float64) / size
        return cv2.filter2D(
            rgb_bgr, -1, kernel, borderType=cv2.BORDER_REFLECT_101
        )
    if condition_id == "partial_occlusion_35x45":
        result = rgb_bgr.copy()
        height, width = result.shape[:2]
        box_width = int(round(width * float(condition["width_fraction"])))
        box_height = int(
            round(height * float(condition["height_fraction"]))
        )
        travel = max(width - box_width, 1)
        x0 = int(round((frame_index / 100.0) * travel))
        y0 = (height - box_height) // 2
        fill_rgb = condition["fill_rgb"]
        fill_bgr = tuple(int(value) for value in reversed(fill_rgb))
        result[y0 : y0 + box_height, x0 : x0 + box_width] = fill_bgr
        return result
    raise ValueError(f"UNKNOWN_STRESS_CONDITION:{condition_id}")


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"PNG_WRITE_FAILED:{path}")


def _index_base_algorithm_inputs(
    base_frames: list[dict[str, Any]],
    base_algorithm_inputs: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    frame_keys = {
        (row["sequence_id"], int(row["frame_index"])) for row in base_frames
    }
    indexed = {
        (row["sequence_id"], int(row["frame_index"])): row
        for row in base_algorithm_inputs
    }
    if len(indexed) != len(base_algorithm_inputs) or set(indexed) != frame_keys:
        raise ValueError("BASE_ALGORITHM_INPUT_IDENTITY")
    return indexed


def build_stress_dataset(
    output_root: Path, spec_path: Path = DEFAULT_SPEC
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    started = time.perf_counter()
    output_root = output_root.resolve()
    spec_path = spec_path.resolve()
    spec = read_json(spec_path)
    if spec["status"] != "PREREGISTERED_BEFORE_STRESS_OUTCOME_ACCESS":
        raise ValueError("STRESS_SPEC_NOT_PREREGISTERED")
    budget = spec["resource_budget"]
    free_before = shutil.disk_usage(_nearest_existing(output_root)).free
    if free_before < int(budget["minimum_free_bytes_before_start"]):
        raise RuntimeError("STRESS_MINIMUM_FREE_BYTES_NOT_MET")
    parent_chain = _validate_frozen_parent_chain(spec)
    parent_path = (
        REPO_ROOT / spec["parent_r0"]["spec_path"]
    ).resolve()
    if sha256_file(parent_path) != spec["parent_r0"]["spec_sha256"]:
        raise ValueError("PARENT_R0_SPEC_HASH_MISMATCH")
    parent = read_json(parent_path)
    if platform.python_version() != spec["resource_budget"]["supported_python"]:
        raise RuntimeError("UNSUPPORTED_PYTHON")
    if np.__version__ != spec["resource_budget"]["supported_numpy"]:
        raise RuntimeError("UNSUPPORTED_NUMPY")
    if not cv2.__version__.startswith(
        spec["resource_budget"]["supported_opencv_prefix"]
    ):
        raise RuntimeError("UNSUPPORTED_OPENCV")
    _ensure_new_root(output_root)

    shutil.copyfile(spec_path, output_root / "dataset_spec.json")
    motion_by_id = {
        motion["motion_id"]: motion for motion in parent["motions"]
    }
    base_spec = json.loads(json.dumps(parent))
    base_spec["splits"]["development"]["scene_families"] = spec[
        "allocation"
    ]["scene_families"]
    base_spec["motions"] = [
        motion_by_id[motion_id]
        for motion_id in spec["allocation"]["motion_ids"]
    ]
    base_spec_path = output_root / "base_r0_spec.json"
    write_json(base_spec_path, base_spec)
    base_root = output_root / "base"
    build_dataset(base_spec_path, base_root, split="development")
    written_bytes = sum(
        path.stat().st_size
        for path in output_root.rglob("*")
        if path.is_file()
    )
    _check_budget(started, written_bytes, budget)

    base_frames = read_jsonl(base_root / "frame_manifest.jsonl")
    base_algorithm_inputs = read_jsonl(
        base_root / "algorithm_input_manifest.jsonl"
    )
    base_algorithm_by_key = _index_base_algorithm_inputs(
        base_frames, base_algorithm_inputs
    )
    base_by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in base_frames:
        base_by_sequence.setdefault(row["sequence_id"], []).append(row)
    algorithm_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    observed_depth_rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(spec["conditions"]):
        condition_id = condition["condition_id"]
        for base_sequence_id in sorted(base_by_sequence):
            selected = sorted(
                base_by_sequence[base_sequence_id],
                key=lambda row: row["frame_index"],
            )
            first = selected[0]
            stress_sequence_id = f"{base_sequence_id}__{condition_id}"
            sequence_rows.append(
                {
                    "schema": "rcle.synthetic_stress.sequence.v1",
                    "protocol_id": spec["protocol_id"],
                    "sequence_id": stress_sequence_id,
                    "base_sequence_id": base_sequence_id,
                    "condition_id": condition_id,
                    "condition_family": condition["family"],
                    "condition_parameters": condition,
                    "scene_id": first["scene_id"],
                    "motion_id": first["motion_id"],
                    "role": first["role"],
                    "frame_count": len(selected),
                    "pair_count": len(selected) - 1,
                }
            )
            for row in selected:
                frame_index = int(row["frame_index"])
                base_algorithm = base_algorithm_by_key.get(
                    (base_sequence_id, frame_index)
                )
                if base_algorithm is None:
                    raise ValueError("BASE_ALGORITHM_INPUT_MISSING")
                source_rgb = base_root / row["rgb_path"]
                rgb = cv2.imread(str(source_rgb), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise ValueError(f"BASE_RGB_READ:{source_rgb}")
                transformed = _transform_rgb(rgb, condition, frame_index)
                staged_rgb = (
                    output_root
                    / "algorithm_staging"
                    / condition_id
                    / base_sequence_id
                    / "rgb"
                    / f"{frame_index:06d}.png"
                )
                staged_mask = (
                    output_root
                    / "algorithm_staging"
                    / condition_id
                    / base_sequence_id
                    / "valid_mask"
                    / f"{frame_index:06d}.png"
                )
                _write_png(staged_rgb, transformed)
                written_bytes += staged_rgb.stat().st_size
                staged_mask.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    base_root / row["valid_mask_path"], staged_mask
                )
                written_bytes += staged_mask.stat().st_size
                rotation = np.asarray(
                    base_algorithm["rotation_current_from_previous"],
                    dtype=np.float64,
                )
                if condition_id == "pose_yaw_error":
                    error_degrees = float(condition["bias_degrees"]) + float(
                        condition["sinusoid_amplitude_degrees"]
                    ) * float(
                        np.sin(
                            float(condition["sinusoid_frequency_per_frame"])
                            * frame_index
                        )
                    )
                    rotation = _yaw_rotation(error_degrees) @ rotation
                algorithm_rows.append(
                    {
                        "schema": "rcle.synthetic_scene.algorithm_input.v1",
                        "protocol_id": parent["protocol_id"],
                        "split": "development",
                        "sequence_id": stress_sequence_id,
                        "frame_index": frame_index,
                        "timestamp": base_algorithm["timestamp"],
                        "rgb_path": relative_posix(
                            staged_rgb, output_root
                        ),
                        "rgb_sha256": sha256_file(staged_rgb),
                        "valid_mask_path": relative_posix(
                            staged_mask, output_root
                        ),
                        "valid_mask_sha256": sha256_file(staged_mask),
                        "intrinsics": base_algorithm["intrinsics"],
                        "rotation_current_from_previous": rotation.tolist(),
                        "algorithm_seed": int(
                            base_algorithm["algorithm_seed"]
                        ),
                    }
                )
                if condition_id == "depth_noise_gaussian_2pct":
                    base_depth = np.load(
                        base_root / row["depth_npy_path"],
                        allow_pickle=False,
                    ).astype(np.float64)
                    rng = np.random.default_rng(
                        _seed(
                            spec["allocation"]["split_seed"],
                            base_sequence_id,
                            frame_index,
                            condition_id,
                        )
                    )
                    noisy = np.clip(
                        base_depth
                        * (
                            1.0
                            + rng.normal(
                                0.0,
                                float(condition["relative_sigma"]),
                                size=base_depth.shape,
                            )
                        ),
                        0.05,
                        50.0,
                    ).astype(np.float32)
                    noisy_path = (
                        output_root
                        / "observed_depth"
                        / condition_id
                        / base_sequence_id
                        / f"{frame_index:06d}.npy"
                    )
                    noisy_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(noisy_path, noisy, allow_pickle=False)
                    written_bytes += noisy_path.stat().st_size
                    observed_depth_rows.append(
                        {
                            "schema": (
                                "rcle.synthetic_stress.observed_depth.v1"
                            ),
                            "sequence_id": stress_sequence_id,
                            "base_sequence_id": base_sequence_id,
                            "frame_index": frame_index,
                            "depth_npy_path": relative_posix(
                                noisy_path, output_root
                            ),
                            "depth_npy_sha256": sha256_file(noisy_path),
                            "relative_sigma": float(
                                condition["relative_sigma"]
                            ),
                        }
                    )
                _check_budget(started, written_bytes, budget)

    sequence_sha = write_jsonl(
        output_root / "stress_sequence_manifest.jsonl", sequence_rows
    )
    algorithm_sha = write_jsonl(
        output_root / "algorithm_input_manifest.jsonl", algorithm_rows
    )
    observed_depth_sha = write_jsonl(
        output_root / "observed_depth_manifest.jsonl", observed_depth_rows
    )
    dataset_bytes = sum(
        path.stat().st_size
        for path in output_root.rglob("*")
        if path.is_file()
    )
    if dataset_bytes > int(spec["resource_budget"]["disk_ceiling_bytes"]):
        raise RuntimeError("STRESS_DISK_CEILING_EXCEEDED")
    receipt = {
        "schema": "rcle.synthetic_stress.dataset_receipt.v1",
        "protocol_id": spec["protocol_id"],
        "status": "CANDIDATE_STRESS_DATASET_VALIDATION_REQUIRED",
        "sequence_count": len(sequence_rows),
        "algorithm_frame_count": len(algorithm_rows),
        "observed_depth_frame_count": len(observed_depth_rows),
        "stress_sequence_manifest_sha256": sequence_sha,
        "algorithm_input_manifest_sha256": algorithm_sha,
        "observed_depth_manifest_sha256": observed_depth_sha,
        "dataset_spec_sha256": sha256_file(
            output_root / "dataset_spec.json"
        ),
        "parent_r0_spec_sha256": sha256_file(parent_path),
        "parent_chain": parent_chain,
        "base_r0_spec_sha256": sha256_file(base_spec_path),
        "base_dataset_spec_sha256": sha256_file(
            base_root / "dataset_spec.json"
        ),
        "base_dataset_receipt_sha256": sha256_file(
            base_root / "receipt.json"
        ),
        "stress_code_sha256": {
            name: sha256_file(MODULE_ROOT / name)
            for name in (
                "builder.py",
                "evaluator.py",
                "validator.py",
                "qa.py",
            )
        },
        "dataset_bytes": dataset_bytes,
        "free_bytes_before_start": free_before,
        "runtime_ceiling_seconds": budget["runtime_ceiling_seconds"],
        "disk_ceiling_bytes": budget["disk_ceiling_bytes"],
        "elapsed_seconds": time.perf_counter() - started,
        "authority": spec["authority"],
    }
    receipt_sha = write_json(output_root / "receipt.json", receipt)
    return {
        "dataset_root": str(output_root),
        "sequence_count": len(sequence_rows),
        "algorithm_frame_count": len(algorithm_rows),
        "receipt_sha256": receipt_sha,
    }

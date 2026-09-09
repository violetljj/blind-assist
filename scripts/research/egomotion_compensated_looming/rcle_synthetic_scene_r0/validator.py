"""Independent, fail-closed validator for an RCLE synthetic dataset.

This module deliberately does not import the renderer.  It validates only the
bytes and metadata present below a completed dataset root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SPEC_NAME = "dataset_spec.json"
SCENE_MANIFEST_NAME = "scene_manifest.jsonl"
FRAME_MANIFEST_NAME = "frame_manifest.jsonl"
ALGORITHM_INPUT_MANIFEST_NAME = "algorithm_input_manifest.jsonl"
RECEIPT_NAME = "receipt.json"
HASH_NAMES = (SPEC_NAME, SCENE_MANIFEST_NAME, FRAME_MANIFEST_NAME)


class ValidationFailure(ValueError):
    """A dataset invariant was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationFailure(f"{path.name} must contain a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValidationFailure(f"{path.name}:{line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValidationFailure(f"{path.name} is empty")
    return rows


def normalized_receipt_for_determinism(
    receipt: dict[str, Any], exclusions: Iterable[str]
) -> dict[str, Any]:
    """Return the frozen normalized receipt view used for repeat checks."""
    normalized = json.loads(json.dumps(receipt))
    for dotted in exclusions:
        parts = dotted.split(".")
        cursor: Any = normalized
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValidationFailure(
                    f"DETERMINISM_EXCLUSION_MISSING:{dotted}"
                )
            cursor = cursor[part]
        if not isinstance(cursor, dict) or parts[-1] not in cursor:
            raise ValidationFailure(
                f"DETERMINISM_EXCLUSION_MISSING:{dotted}"
            )
        del cursor[parts[-1]]
    return normalized


def compare_development_generations(first: Path, second: Path) -> dict[str, Any]:
    """Compare two fresh development generations under the frozen contract."""
    first = first.resolve()
    second = second.resolve()
    first_spec = _json(first / SPEC_NAME)
    second_spec = _json(second / SPEC_NAME)
    if first_spec != second_spec:
        raise ValidationFailure("DETERMINISM_SPEC_MISMATCH")
    names = (
        SCENE_MANIFEST_NAME,
        FRAME_MANIFEST_NAME,
        ALGORITHM_INPUT_MANIFEST_NAME,
    )
    mismatches = [
        name
        for name in names
        if sha256_file(first / name) != sha256_file(second / name)
    ]
    exclusions = first_spec["resource_preflight"][
        "determinism_receipt_excluded_fields_exact"
    ]
    receipts_equal = normalized_receipt_for_determinism(
        _json(first / RECEIPT_NAME), exclusions
    ) == normalized_receipt_for_determinism(
        _json(second / RECEIPT_NAME), exclusions
    )
    if not receipts_equal:
        mismatches.append("receipt.json(normalized)")
    return {
        "schema": "rcle.synthetic_scene.determinism_validation.v1",
        "protocol_id": first_spec["protocol_id"],
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "receipt_exclusions": exclusions,
    }


def _first(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _required(mapping: dict[str, Any], names: Iterable[str], label: str) -> Any:
    value = _first(mapping, names)
    if value is None:
        raise ValidationFailure(f"missing {label}")
    return value


def _contained_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValidationFailure(f"{label} must be a non-empty relative path")
    candidate_rel = Path(relative)
    if candidate_rel.is_absolute() or ".." in candidate_rel.parts:
        raise ValidationFailure(f"{label} escapes dataset root: {relative!r}")
    root_resolved = root.resolve()
    candidate = (root_resolved / candidate_rel).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValidationFailure(f"{label} escapes dataset root: {relative!r}") from exc
    if not candidate.is_file():
        raise ValidationFailure(f"{label} does not exist: {relative!r}")
    return candidate


def _receipt_hashes(receipt: dict[str, Any]) -> dict[str, str]:
    found: dict[str, str] = {}
    aliases = {
        SPEC_NAME: ("dataset_spec_sha256", "spec_sha256"),
        SCENE_MANIFEST_NAME: ("scene_manifest_sha256",),
        FRAME_MANIFEST_NAME: ("frame_manifest_sha256",),
        ALGORITHM_INPUT_MANIFEST_NAME: (
            "algorithm_input_manifest_sha256",
        ),
    }
    for filename, keys in aliases.items():
        value = _first(receipt, keys)
        if isinstance(value, str):
            found[filename] = value

    for container_name in ("files", "artifacts", "hashes", "manifest_hashes"):
        container = receipt.get(container_name)
        if isinstance(container, dict):
            for filename in HASH_NAMES:
                value = container.get(filename)
                if isinstance(value, str):
                    found[filename] = value
                elif isinstance(value, dict) and isinstance(value.get("sha256"), str):
                    found[filename] = value["sha256"]
        elif isinstance(container, list):
            for item in container:
                if not isinstance(item, dict):
                    continue
                name = _first(item, ("path", "name", "file"))
                value = _first(item, ("sha256", "hash"))
                if name in HASH_NAMES and isinstance(value, str):
                    found[str(name)] = value
    return found


def _asset(
    row: dict[str, Any], kind: str
) -> tuple[str, str, dict[str, Any]]:
    nested = row.get(kind)
    meta = nested if isinstance(nested, dict) else {}
    path_names = {
        "rgb": ("rgb_path", "image_path"),
        "depth": ("depth_npy_path", "depth_path"),
    }[kind]
    hash_names = {
        "rgb": ("rgb_sha256", "image_sha256"),
        "depth": ("depth_npy_sha256", "depth_sha256"),
    }[kind]
    path_value = _first(meta, ("path", "npy_path", "file"))
    if path_value is None:
        path_value = _first(row, path_names)
    hash_value = _first(meta, ("sha256", "npy_sha256", "hash"))
    if hash_value is None:
        hash_value = _first(row, hash_names)
    if not isinstance(path_value, str) or not isinstance(hash_value, str):
        raise ValidationFailure(f"frame lacks {kind} path/hash")
    return path_value, hash_value, meta


def _matrix(row: dict[str, Any], names: tuple[str, ...], label: str) -> np.ndarray:
    value = _required(row, names, label)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValidationFailure(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValidationFailure(f"{label} has an invalid homogeneous final row")
    return matrix


def _intrinsics(row: dict[str, Any]) -> np.ndarray:
    value = _required(row, ("K", "intrinsics"), "camera intrinsics")
    if isinstance(value, dict):
        if "K" in value:
            value = value["K"]
        else:
            value = [
                [
                    value.get("fx", value.get("fx_pixels")),
                    0.0,
                    value.get("cx", value.get("cx_pixels")),
                ],
                [
                    0.0,
                    value.get("fy", value.get("fy_pixels")),
                    value.get("cy", value.get("cy_pixels")),
                ],
                [0.0, 0.0, 1.0],
            ]
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValidationFailure("K must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] == 0:
        raise ValidationFailure("K focal magnitudes must be non-zero")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-12):
        raise ValidationFailure("K has an invalid final row")
    return matrix


def _validate_impl(root: Path, check_algorithm: bool) -> dict[str, Any]:
    root = root.resolve()
    required = {
        name: _contained_file(root, name, name)
        for name in (*HASH_NAMES, RECEIPT_NAME)
    }
    spec = _json(required[SPEC_NAME])
    scenes = _jsonl(required[SCENE_MANIFEST_NAME])
    frames = _jsonl(required[FRAME_MANIFEST_NAME])
    receipt = _json(required[RECEIPT_NAME])

    actual_hashes = {name: sha256_file(required[name]) for name in HASH_NAMES}
    expected_hashes = _receipt_hashes(receipt)
    for name in HASH_NAMES:
        if name not in expected_hashes:
            raise ValidationFailure(f"receipt lacks {name} SHA-256")
        if expected_hashes[name].lower() != actual_hashes[name]:
            raise ValidationFailure(f"{name} SHA-256 mismatch")
    algorithm_manifest_path = root / ALGORITHM_INPUT_MANIFEST_NAME
    if ALGORITHM_INPUT_MANIFEST_NAME in expected_hashes:
        algorithm_manifest_path = _contained_file(
            root,
            ALGORITHM_INPUT_MANIFEST_NAME,
            ALGORITHM_INPUT_MANIFEST_NAME,
        )
        actual_algorithm_manifest_hash = sha256_file(
            algorithm_manifest_path
        )
        if (
            expected_hashes[ALGORITHM_INPUT_MANIFEST_NAME].lower()
            != actual_algorithm_manifest_hash
        ):
            raise ValidationFailure(
                f"{ALGORITHM_INPUT_MANIFEST_NAME} SHA-256 mismatch"
            )
        actual_hashes[
            ALGORITHM_INPUT_MANIFEST_NAME
        ] = actual_algorithm_manifest_hash

    spec_hash = actual_hashes[SPEC_NAME]
    split_by_scene: dict[str, str] = {}
    sequences: dict[str, dict[str, Any]] = {}
    scene_rows: set[tuple[str, str]] = set()
    for row in scenes:
        scene_id = str(_required(row, ("scene_id",), "scene_id"))
        split = str(_required(row, ("split",), "scene split"))
        sequence_id = str(
            _required(row, ("sequence_id",), "scene sequence_id")
        )
        row_spec_hash = _first(row, ("dataset_spec_sha256", "spec_sha256"))
        if row_spec_hash is not None and row_spec_hash != spec_hash:
            raise ValidationFailure(f"scene {scene_id} has a spec hash mismatch")
        previous = split_by_scene.setdefault(scene_id, split)
        if previous != split:
            raise ValidationFailure(f"scene leakage across splits: {scene_id}")
        if sequence_id in sequences:
            raise ValidationFailure(f"duplicate sequence manifest row: {sequence_id}")
        sequences[sequence_id] = row
        scene_rows.add((scene_id, split))

    rendering = spec.get("rendering", {})
    width = int(_required(rendering, ("width",), "spec rendering width"))
    height = int(_required(rendering, ("height",), "spec rendering height"))
    signed_projection = "signed_projection_matrix" in rendering.get(
        "coordinate_convention", {}
    )
    y_focal = (
        -rendering.get("intrinsics", {}).get("fy_pixels")
        if signed_projection
        else rendering.get("intrinsics", {}).get("fy_pixels")
    )
    expected_k = np.array(
        [
            [
                rendering.get("intrinsics", {}).get("fx_pixels"),
                0.0,
                rendering.get("intrinsics", {}).get("cx_pixels"),
            ],
            [
                0.0,
                y_focal,
                rendering.get("intrinsics", {}).get("cy_pixels"),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    timestamps: dict[str, list[tuple[int, float]]] = defaultdict(list)
    hashes_by_sequence: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    asset_split: dict[tuple[str, str], str] = {}
    seen_frames: set[tuple[str, int]] = set()
    for row_number, row in enumerate(frames, 1):
        scene_id = str(_required(row, ("scene_id",), "frame scene_id"))
        split = str(_required(row, ("split",), "frame split"))
        if (scene_id, split) not in scene_rows:
            raise ValidationFailure(
                f"frame {row_number} references undeclared scene/split {scene_id}/{split}"
            )
        sequence_id = str(
            _first(row, ("sequence_id", "sequence_key"))
            or f"{scene_id}/{_required(row, ('motion_id',), 'motion_id')}"
        )
        sequence = sequences.get(sequence_id)
        if sequence is None:
            raise ValidationFailure(f"frame references undeclared sequence {sequence_id}")
        if sequence.get("scene_id") != scene_id or sequence.get("split") != split:
            raise ValidationFailure(f"{sequence_id} frame identity differs from sequence")
        frame_index = int(
            _required(row, ("frame_index", "index"), "frame_index")
        )
        timestamp_value = _required(
            row, ("timestamp", "timestamp_s", "timestamp_seconds"), "timestamp"
        )
        if isinstance(timestamp_value, dict):
            numerator = int(
                _required(timestamp_value, ("numerator",), "timestamp numerator")
            )
            denominator = int(
                _required(timestamp_value, ("denominator",), "timestamp denominator")
            )
            timestamp = float(
                _required(timestamp_value, ("seconds",), "timestamp seconds")
            )
            if denominator <= 0 or numerator != frame_index:
                raise ValidationFailure(
                    f"{sequence_id}/{frame_index} timestamp rational is invalid"
                )
            if not np.isclose(
                timestamp, numerator / denominator, rtol=0.0, atol=1e-12
            ):
                raise ValidationFailure(
                    f"{sequence_id}/{frame_index} timestamp rational mismatch"
                )
        else:
            timestamp = float(timestamp_value)
        if not np.isfinite(timestamp):
            raise ValidationFailure(f"{sequence_id} has a non-finite timestamp")
        frame_key = (sequence_id, frame_index)
        if frame_key in seen_frames:
            raise ValidationFailure(f"duplicate frame {sequence_id}/{frame_index}")
        seen_frames.add(frame_key)
        timestamps[sequence_id].append((frame_index, timestamp))

        row_spec_hash = _first(row, ("dataset_spec_sha256", "spec_sha256"))
        if row_spec_hash is not None and row_spec_hash != spec_hash:
            raise ValidationFailure(f"{sequence_id}/{frame_index} spec hash mismatch")

        rgb_rel, rgb_hash, _ = _asset(row, "rgb")
        depth_rel, depth_hash, _ = _asset(row, "depth")
        rgb_path = _contained_file(root, rgb_rel, "RGB path")
        depth_path = _contained_file(root, depth_rel, "depth path")
        if sha256_file(rgb_path) != rgb_hash:
            raise ValidationFailure(f"{sequence_id}/{frame_index} RGB hash mismatch")
        if sha256_file(depth_path) != depth_hash:
            raise ValidationFailure(f"{sequence_id}/{frame_index} depth hash mismatch")
        frame_digests = [rgb_hash, depth_hash]

        for path_key, hash_key, label in (
            ("depth_png_path", "depth_png_sha256", "depth PNG"),
            ("surface_id_path", "surface_id_sha256", "surface ID"),
            ("valid_mask_path", "valid_mask_sha256", "valid mask"),
        ):
            relative = row.get(path_key)
            expected = row.get(hash_key)
            if (relative is None) != (expected is None):
                raise ValidationFailure(
                    f"{sequence_id}/{frame_index} incomplete {label} path/hash"
                )
            if relative is not None:
                extra_path = _contained_file(root, relative, f"{label} path")
                if sha256_file(extra_path) != expected:
                    raise ValidationFailure(
                        f"{sequence_id}/{frame_index} {label} hash mismatch"
                    )
                frame_digests.append(expected)
        hashes_by_sequence[sequence_id].append((frame_index, frame_digests))

        for kind, digest in (("rgb", rgb_hash), ("depth", depth_hash)):
            asset_key = (kind, digest)
            previous_split = asset_split.setdefault(asset_key, split)
            if previous_split != split:
                raise ValidationFailure(
                    f"identical {kind} bytes leak across splits"
                )

        with Image.open(rgb_path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (width, height):
                raise ValidationFailure(
                    f"{sequence_id}/{frame_index} RGB shape/mode mismatch"
                )
        depth = np.load(depth_path, allow_pickle=False)
        if depth.shape != (height, width):
            raise ValidationFailure(
                f"{sequence_id}/{frame_index} depth shape mismatch"
            )
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValidationFailure(
                f"{sequence_id}/{frame_index} depth is not floating point"
            )
        if not np.isfinite(depth).all() or not np.all(depth > 0.0):
            raise ValidationFailure(
                f"{sequence_id}/{frame_index} depth must be finite and positive"
            )
        if row.get("depth_png_path") is not None:
            with Image.open(root / row["depth_png_path"]) as depth_png:
                depth_png.load()
                if depth_png.size != (width, height):
                    raise ValidationFailure(
                        f"{sequence_id}/{frame_index} depth PNG shape mismatch"
                    )
        if row.get("surface_id_path") is not None:
            surface = np.load(root / row["surface_id_path"], allow_pickle=False)
            if surface.shape != (height, width):
                raise ValidationFailure(
                    f"{sequence_id}/{frame_index} surface ID shape mismatch"
                )
        if row.get("valid_mask_path") is not None:
            with Image.open(root / row["valid_mask_path"]) as valid_mask:
                valid_mask.load()
                mask_values = np.asarray(valid_mask)
                if valid_mask.size != (width, height) or not set(
                    np.unique(mask_values).tolist()
                ).issubset({0, 255}):
                    raise ValidationFailure(
                        f"{sequence_id}/{frame_index} valid mask contract mismatch"
                    )

        k = _intrinsics(row)
        if np.isfinite(expected_k).all() and not np.allclose(
            k, expected_k, rtol=0.0, atol=1e-9
        ):
            raise ValidationFailure(f"{sequence_id}/{frame_index} K differs from spec")
        world_camera = _matrix(
            row,
            ("T_world_camera", "t_world_camera"),
            "T_world_camera",
        )
        camera_world = _matrix(
            row,
            ("T_camera_world", "t_camera_world"),
            "T_camera_world",
        )
        if not np.allclose(
            world_camera @ camera_world, np.eye(4), rtol=0.0, atol=1e-8
        ) or not np.allclose(
            camera_world @ world_camera, np.eye(4), rtol=0.0, atol=1e-8
        ):
            raise ValidationFailure(
                f"{sequence_id}/{frame_index} bidirectional poses are not inverses"
            )

    for sequence_id, values in timestamps.items():
        ordered = sorted(values)
        indices = [item[0] for item in ordered]
        times = [item[1] for item in ordered]
        if indices != sorted(indices) or any(
            right <= left for left, right in zip(times, times[1:])
        ):
            raise ValidationFailure(
                f"{sequence_id} timestamps are not strictly increasing"
            )
        sequence = sequences[sequence_id]
        if len(values) != int(sequence.get("frame_count", -1)):
            raise ValidationFailure(f"{sequence_id} frame_count mismatch")
        if int(sequence.get("pair_count", -1)) != max(0, len(values) - 1):
            raise ValidationFailure(f"{sequence_id} pair_count mismatch")
        ordered_hashes = sorted(hashes_by_sequence[sequence_id])
        digest = hashlib.sha256(
            "".join(
                item
                for _, frame_hashes in ordered_hashes
                for item in frame_hashes
            ).encode("ascii")
        ).hexdigest()
        if digest != sequence.get("sequence_content_sha256"):
            raise ValidationFailure(f"{sequence_id} content SHA-256 mismatch")

    if set(timestamps) != set(sequences):
        missing = sorted(set(sequences) - set(timestamps))
        raise ValidationFailure(f"sequence(s) have no frames: {missing}")

    if algorithm_manifest_path.is_file():
        algorithm_rows = _jsonl(algorithm_manifest_path)
        allowlist = set(
            spec.get("algorithm_firewall", {}).get(
                "algorithm_manifest_allowlist_exact", []
            )
        )
        staging = (root / "algorithm_staging").resolve()
        seen_algorithm_frames: set[tuple[str, int]] = set()
        for row in algorithm_rows:
            if allowlist and set(row) != allowlist:
                raise ValidationFailure("algorithm manifest allowlist drift")
            sequence_id = str(
                _required(row, ("sequence_id",), "algorithm sequence_id")
            )
            frame_index = int(
                _required(row, ("frame_index",), "algorithm frame_index")
            )
            key = (sequence_id, frame_index)
            if key in seen_algorithm_frames:
                raise ValidationFailure("duplicate algorithm input frame")
            seen_algorithm_frames.add(key)
            for path_field, hash_field in (
                ("rgb_path", "rgb_sha256"),
                ("valid_mask_path", "valid_mask_sha256"),
            ):
                path = _contained_file(
                    root, row[path_field], f"algorithm {path_field}"
                )
                if not path.resolve().is_relative_to(staging):
                    raise ValidationFailure(
                        f"algorithm {path_field} escapes staging root"
                    )
                if sha256_file(path) != row[hash_field]:
                    raise ValidationFailure(
                        f"algorithm {path_field} hash mismatch"
                    )
            rotation = np.asarray(
                row["rotation_current_from_previous"], dtype=np.float64
            )
            if (
                rotation.shape != (3, 3)
                or not np.isfinite(rotation).all()
                or not np.allclose(
                    rotation.T @ rotation,
                    np.eye(3),
                    rtol=0.0,
                    atol=1e-9,
                )
                or not np.isclose(
                    np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-9
                )
            ):
                raise ValidationFailure(
                    "algorithm relative rotation is invalid"
                )
        if seen_algorithm_frames != seen_frames:
            raise ValidationFailure(
                "algorithm input frame identity differs from truth manifest"
            )

    algorithm_ledger = root / "algorithm_pair_ledger.jsonl"
    algorithm_ledger_parse_checked = False
    if check_algorithm and algorithm_ledger.exists():
        _jsonl(_contained_file(root, algorithm_ledger.name, algorithm_ledger.name))
        algorithm_ledger_parse_checked = True

    return {
        "schema_version": 1,
        "validator": "rcle_synthetic_scene_r0_independent_validator",
        "status": "PASS",
        "dataset_root": str(root),
        "counts": {
            "scenes": len(scenes),
            "frames": len(frames),
            "sequences": len(timestamps),
        },
        "verified_sha256": actual_hashes,
        "algorithm_results_present": algorithm_ledger.is_file(),
        "algorithm_ledger_parse_checked": algorithm_ledger_parse_checked,
        "scientific_outcome_recomputed": False,
        "errors": [],
    }


def validate_dataset(
    dataset_root: Path | str, *, check_algorithm: bool = True
) -> dict[str, Any]:
    root = Path(dataset_root)
    try:
        return _validate_impl(root, check_algorithm)
    except (
        ValidationFailure,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        return {
            "schema_version": 1,
            "validator": "rcle_synthetic_scene_r0_independent_validator",
            "status": "FAIL",
            "dataset_root": str(root.resolve()),
            "errors": [str(exc)],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-algorithm-results",
        action="store_true",
        help="Do not parse the optional algorithm pair ledger.",
    )
    args = parser.parse_args()
    report = validate_dataset(
        args.dataset_root, check_algorithm=not args.skip_algorithm_results
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
